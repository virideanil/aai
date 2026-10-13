"""Collect attempts and evaluate teaching through the same pinned native core."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoai.agent import DEFAULT_IDENTITY
from nanoai.core import JsonlCore, file_digest, digest_bytes
from nanoai.experience import Experience
from nanoai.memory import Memory, canonical
from nanoai.teaching import Teaching
from teaching_tasks import assess, evaluation_rows, guidance, messages, qualify


def run(args, result, record):
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("max_tokens") is not None:
        raise ValueError("Active native teaching evaluation requires natural generation without a token quota")
    corpus = json.loads(args.corpus.read_text())
    lessons = json.loads(args.lessons.read_text())["lessons"]
    preflight = json.loads(args.preflight.read_text())
    if (preflight.get("model") != "Qwen3.5-2B-BF16" or preflight.get("thinking") is not True
            or preflight.get("reasoning_token_limit", 0) is not None
            or not preflight.get("parent_source")):
        raise ValueError("Sourced native launch preflight required")
    if file_digest(args.corpus) != protocol["corpus_digest"] or file_digest(args.lessons) != protocol["lesson_digest"]:
        raise ValueError("Frozen teaching data changed")
    if digest_bytes(canonical({k: v for k, v in corpus.items() if k != "identity"})) != corpus["identity"]:
        raise ValueError("Teaching corpus identity differs")
    for name, expected in protocol["implementation"].items():
        if file_digest(Path(__file__).resolve().parents[1] / name) != expected:
            raise ValueError("Frozen implementation changed: " + name)
    config = json.loads(args.config.read_text())
    manifest = json.loads(Path(config["manifest"]).read_text())
    base = {k: v for k, v in manifest.items() if k not in {"adapter", "adapter_tensor_catalogue"}}
    if digest_bytes(canonical(base)) != protocol["native_base_identity"]:
        raise ValueError("Matched original native source or execution configuration changed")
    trained = args.stage in {"learned", "reload"}
    if trained:
        training = json.loads((args.out / "train.json").read_text())
        adapter = manifest.get("adapter", {})
        if training["status"] != "completed" or adapter.get("candidate") != training["candidate"]:
            raise ValueError("Native adapter is not the selected completed training candidate")
        result["candidate"] = training["candidate"]
    elif "adapter" in manifest:
        raise ValueError("The original and guided controls require the unchanged source")
    result.update(protocol=file_digest(args.protocol), corpus=corpus["identity"],
                  parameter_updates=0, admitted=False, preflight=preflight)
    record("data-qualified", **qualify(corpus))
    memory = Memory(args.out / "state"); memory.initialize(DEFAULT_IDENTITY)
    owner, teaching = Experience(memory), Teaching(memory)
    for lesson in lessons: teaching.install(lesson)
    train = [r for r in corpus["rows"] if r["split"] == "train"]
    teacher = guidance(lessons, train) if args.stage == "guided" else None
    if args.stage == "collect":
        if args.review_cases is not None:
            raise ValueError("Independent cases stay outside collection and training")
        rows = [{**r, "cohort": "authored"} for r in train if r["id"] in protocol["collection_ids"]]
    else:
        rows = evaluation_rows(corpus, review_cases=args.review_cases,
                               expected_review=protocol.get("independent_review"))
    with JsonlCore(config, stderr_path=args.out / (args.stage + "-worker.log")) as core:
        result.update(core_identity=core.identity, load=core.load)
        if args.stage == "reload":
            prior = json.loads((args.out / "learned.json").read_text())
            if prior["status"] != "completed" or prior["core_identity"] != core.identity:
                raise ValueError("Fresh native consumer identity differs")
            replays = []
            for row in (rows[0], rows[-1]):
                generated = core.generate(messages(row), max_tokens=protocol["max_tokens"], trace=True)
                old = next(r for r in prior["rows"] if r["case_id"] == row["id"])["generation"]
                if generated["token_ids"] != old["token_ids"] or generated["output"] != old["text"]:
                    raise AssertionError("Fresh native generated tokens or text differ")
                replays.append(row["id"])
            for row in prior["rows"]:
                if owner.export(row["experience_id"])["eligibility"]["status"] != "checked":
                    raise ValueError("Recovered assessment is stale")
                path = memory.root / row["receipt"]["path"]
                before = (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
                if owner.execute(row["experience_id"]) != row["receipt"]:
                    raise AssertionError("Recovered receipt changed")
                if before != (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns):
                    raise AssertionError("Recovered artifact was republished")
            result.update(status="completed", identical_generation_replays=replays,
                          once_only_artifacts=len(prior["rows"]))
            record("fresh-recovery", cases=replays, artifact_count=len(prior["rows"]))
            return
        for row in rows:
            prompt = messages(row, guidance=teacher)
            started = time.monotonic()
            native = core.generate(prompt, max_tokens=protocol["max_tokens"], trace=True)
            generated = {"text": native["output"], "token_ids": native["token_ids"],
                         "finish_reason": native["finish_reason"], "policy": native.get("sampling", "greedy-token-v1")}
            task = args.stage + "-" + row["id"]
            source = memory.put_source(task + "-input", json.dumps({"observations": row["observations"]}, ensure_ascii=False))
            memory.set_task(task, row["question"], filenames=["answer.txt"])
            snapshot = memory.snapshot(task, [source], core.identity)
            experience = owner.record_generation(snapshot, question=row["question"], observations=row["observations"],
                messages=prompt, generation=generated, representation_identity=corpus["identity"], source_id=source["id"],
                metadata={"case_id": row["id"], "arm": args.stage, "split": row["split"], "family": row["family"],
                          "cohort": row["cohort"],
                          "corpus": corpus["identity"], "native_generation": native,
                          "lessons": protocol["lesson_digest"] if teacher else None})
            receipt = owner.execute(experience["id"])
            assessment = owner.assess_generation(experience["id"],
                rubric={"scope": protocol["assessment_scope"], "criteria": row["criteria"],
                        "training_target": row["target"] if row["split"] in {"train", "replay"} else None},
                evaluator_identity=protocol["evaluator"], evaluate=assess)
            answer = {"case_id": row["id"], "family": row["family"], "split": row["split"], "arm": args.stage,
                      "cohort": row["cohort"],
                      "generation": generated, "checks": assessment["checks"],
                      "correct": assessment["observed_outcome"] == "yes", "seconds": time.monotonic() - started,
                      "experience_id": experience["id"], "receipt": receipt, "prompt_tokens": native["prompt_tokens"]}
            result.setdefault("rows", []).append(answer)
            record("artifact", **answer)
    scores = {split: {"correct": sum(r["correct"] for r in result["rows"] if r["split"] == split),
                      "total": sum(r["split"] == split for r in result["rows"])}
              for split in sorted({r["split"] for r in result["rows"]})}
    result.update(status="completed", scores=scores)
    cohorts = {cohort: {"correct": sum(r["correct"] for r in result["rows"] if r["cohort"] == cohort),
                        "total": sum(r["cohort"] == cohort for r in result["rows"])}
               for cohort in sorted({r["cohort"] for r in result["rows"]})}
    result["cohort_scores"] = cohorts
    record("completed", scores=scores, cohort_scores=cohorts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "config", "protocol", "corpus", "lessons", "preflight"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--stage", choices=("collect", "original", "guided", "learned", "reload"), required=True)
    parser.add_argument("--review-cases", type=Path)
    args = parser.parse_args()
    path = args.out / (args.stage + ".json")
    if path.exists(): raise ValueError("Keep the previous native stage")
    result = {"schema": "nanoai.native-teaching-evaluation.v1", "status": "running", "stage": args.stage, "phases": []}
    started = time.monotonic()
    from nanoai.tensor_store import TensorStore
    store = TensorStore(args.out / "state/tensors.sqlite3", create=True)
    protocol_identity = file_digest(args.protocol)
    def record(phase, **values):
        entry = {"phase": phase, "elapsed_seconds": time.monotonic() - started, **values}
        store.record(args.stage, protocol=protocol_identity, phase=phase, values=entry)
        result["phases"].append(entry)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(entry, ensure_ascii=False), flush=True)
    try:
        run(args, result, record)
    except BaseException as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        record("failure", error=repr(error)); raise


if __name__ == "__main__":
    main()
