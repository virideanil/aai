"""Compare supplied teaching and actual BF16 adapter learning on fresh tasks.

Run stages in separate guarded processes. Training selects only from validation
and replay; confirmation cases are never used by its optimizer or selection.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import random
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from teaching_tasks import assess, guidance, messages, qualify, training_input


def run(args, result, record):
    import torch
    from nanoai.agent import DEFAULT_IDENTITY
    from nanoai.core import digest_bytes, file_digest
    from nanoai.experience import Experience
    from nanoai.generative_learning import (attach, basis, cache_prefix, candidate_bytes,
        chunked_backward, encode, generate, loss, restore_candidate, RECIPE)
    from nanoai.memory import Memory, canonical
    from nanoai.source_residency import SourceResidency
    from nanoai.teaching import Teaching
    from nanoai.tensor_store import TensorStore
    from nanoai.torch_source import load_full_text

    protocol = json.loads(args.protocol.read_text())
    if protocol.get("max_tokens") is not None:
        raise ValueError("Active teaching evaluation requires natural generation without a token quota")
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
    manifest = json.loads(args.source_before.read_text())
    if manifest != {p.name: {"bytes": p.stat().st_size, "digest": file_digest(p)}
                    for p in sorted(args.source.iterdir()) if p.is_file()}:
        raise ValueError("Original source manifest changed")
    record("data-qualified", **qualify(corpus))
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.manual_seed(protocol["seed"])
    model, tokenizer = load_full_text(args.source)
    SourceResidency(args.source).install(model)
    parameters = attach(model)
    source_basis = basis(manifest, model)
    memory = Memory(args.out / "state"); memory.initialize(DEFAULT_IDENTITY)
    tensor_store = TensorStore(memory.root / "tensors.sqlite3", create=True)
    owner, teaching = Experience(memory), Teaching(memory)
    for lesson in lessons: teaching.install(lesson)
    result.update(source_basis=source_basis, parameter_updates=0, admitted=False)
    record("loaded", trainable=sum(p.numel() for p in parameters.values()), recipe=RECIPE)
    train = [r for r in corpus["rows"] if r["split"] == "train"]
    replay = [r for r in corpus["rows"] if r["split"] == "replay"]
    validation = [r for r in corpus["rows"] if r["split"] == "validation"]

    def publish(row, arm, *, teaching_text=None):
        prompt = messages(row, guidance=teaching_text)
        start = time.monotonic()
        generated = generate(model, tokenizer, prompt, max_tokens=protocol["max_tokens"])
        elapsed = time.monotonic() - start
        task = arm + "-" + row["id"]
        source = memory.put_source(task + "-input", json.dumps({"observations": row["observations"]}, ensure_ascii=False))
        memory.set_task(task, row["question"], filenames=["answer.txt"])
        snapshot = memory.snapshot(task, [source], source_basis + "@" + arm)
        experience = owner.record_generation(snapshot, question=row["question"], observations=row["observations"],
            messages=prompt, generation=generated, representation_identity=corpus["identity"], source_id=source["id"],
            metadata={"case_id": row["id"], "arm": arm, "split": row["split"], "family": row["family"],
                      "corpus": corpus["identity"], "lessons": protocol["lesson_digest"] if teaching_text else None})
        receipt = owner.execute(experience["id"])
        assessment = owner.assess_generation(experience["id"],
            rubric={"scope": protocol["assessment_scope"], "criteria": row["criteria"],
                    "training_target": row["target"] if row["split"] in {"train", "replay"} else None},
            evaluator_identity=protocol["evaluator"], evaluate=assess)
        answer = {"case_id": row["id"], "family": row["family"], "split": row["split"], "arm": arm,
                  "generation": generated, "checks": assessment["checks"],
                  "correct": assessment["observed_outcome"] == "yes", "seconds": elapsed,
                  "experience_id": experience["id"], "receipt": receipt,
                  "prompt_tokens": len(tokenizer.apply_chat_template(prompt, enable_thinking=True,
                        add_generation_prompt=True, tokenize=True, return_dict=False))}
        result.setdefault("rows", []).append(answer)
        record("artifact", **answer)
        return answer

    if args.stage == "train":
        collected = {}
        if args.collection:
            collection = json.loads(args.collection.read_text())
            if (collection.get("status") != "completed" or collection.get("stage") != "collect"
                    or collection.get("protocol") != file_digest(args.protocol)
                    or collection.get("corpus") != corpus["identity"]
                    or {r["case_id"] for r in collection["rows"]} != set(protocol["collection_ids"])):
                raise ValueError("Observed native collection differs from the frozen training inputs")
            for attempt in collection["rows"]:
                row = next(r for r in train if r["id"] == attempt["case_id"])
                pair = teaching.training_pair(attempt["experience_id"])
                if pair["target"] != row["target"]:
                    raise ValueError("Observed correction target differs from the qualified data")
                collected[row["id"]] = pair
        else:
            for case_id in protocol["collection_ids"]:
                row = next(r for r in train if r["id"] == case_id)
                attempt = publish(row, "collection")
                collected[case_id] = teaching.training_pair(attempt["experience_id"])
        chunked_backward(model)
        examples = {r["id"]: encode(tokenizer, **training_input(r, collected.get(r["id"])))
                    for r in [*train, *replay, *validation]}
        prefixes = {name: cache_prefix(model, example, source_basis=source_basis) for name, example in examples.items()}
        record("prefixes", count=len(prefixes), maximum_tokens=max(len(x["input_ids"]) for x in examples.values()))
        @torch.no_grad()
        def losses(rows):
            return sum(float(loss(model, examples[r["id"]], prefix=prefixes[r["id"]], source_basis=source_basis)[0])
                       for r in rows) / len(rows)
        original_validation, original_replay = losses(validation), losses(replay)
        initial_data = candidate_bytes(parameters, source_basis=source_basis)
        (args.out / "initial.safetensors").write_bytes(initial_data)
        previous_version = tensor_store.register(args.out / "initial.safetensors", expected=digest_bytes(initial_data),
            basis=source_basis, recipe=RECIPE, source="training:initial")["version"]
        record("original-loss", validation=original_validation, replay=original_replay,
               candidate_identity=digest_bytes(initial_data))
        masters = {n: torch.nn.Parameter(p.detach().float().clone()) for n, p in parameters.items()}
        optimizer = torch.optim.AdamW(list(masters.values()), lr=protocol["learning_rate"], weight_decay=0)
        best_score, best_epoch, best_data = float("inf"), None, None
        for epoch in range(1, protocol["epochs"] + 1):
            rows = [*train, *replay]; random.Random(protocol["seed"] + epoch).shuffle(rows)
            total = 0.0
            for row in rows:
                model.zero_grad(set_to_none=True); optimizer.zero_grad(set_to_none=True)
                value, _ = loss(model, examples[row["id"]], prefix=prefixes[row["id"]], source_basis=source_basis)
                if not bool(torch.isfinite(value)): raise ValueError("Nonfinite learning loss")
                total += float(value.detach()); value.backward()
                for name, p in parameters.items():
                    if p.grad is None or not bool(torch.isfinite(p.grad).all()):
                        raise ValueError("Missing or nonfinite learning gradient")
                    masters[name].grad = p.grad.float().clone()
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(list(masters.values()), 1.0))
                optimizer.step()
                with torch.no_grad():
                    for name, p in parameters.items(): p.copy_(masters[name].to(p.dtype))
                result["parameter_updates"] += 1
                record("update", case_id=row["id"], epoch=epoch, loss=float(value.detach()), gradient_norm=gradient_norm)
            val, ret = losses(validation), losses(replay)
            score = val + protocol["replay_selection_weight"] * ret
            data = candidate_bytes(parameters, source_basis=source_basis)
            (args.out / f"epoch-{epoch}.safetensors").write_bytes(data)
            previous_version = tensor_store.register(args.out / f"epoch-{epoch}.safetensors", expected=digest_bytes(data),
                basis=source_basis, recipe=RECIPE, source=f"training:epoch-{epoch}", parent=previous_version)["version"]
            if score < best_score: best_score, best_epoch, best_data = score, epoch, data
            record("epoch", epoch=epoch, training_loss=total / len(rows), validation=val, replay=ret,
                   selection_score=score, tensor_version=previous_version)
        if digest_bytes(best_data) == digest_bytes(initial_data):
            raise ValueError("The learning run made no parameter change")
        (args.out / "selected.safetensors").write_bytes(best_data)
        candidate = owner.register_candidate(best_data, feature_basis=source_basis, tensor_format="safetensors",
            training={"recipe": RECIPE, "corpus": corpus["identity"], "epoch": best_epoch,
                "selection_score": best_score, "updates": result["parameter_updates"],
                "observed_teaching_pairs": list(collected.values()),
                "supervision": "Checked actual-attempt corrections plus authored demonstrations and separate replay"})
        restore_candidate(parameters, best_data, expected_identity=candidate["id"], expected_basis=source_basis)
        probe = validation[0]
        with torch.no_grad():
            _, logits = loss(model, examples[probe["id"]], prefix=prefixes[probe["id"]], source_basis=source_basis)
        torch.save({"example": examples[probe["id"]], "logits": logits, "basis": source_basis,
                    "candidate": candidate["id"]}, args.out / "reload-probe.pt")
        result.update(status="completed", candidate=candidate["id"], selected_epoch=best_epoch,
                      selected_score=best_score, original_validation=original_validation, original_replay=original_replay)
        record("candidate-saved", candidate=candidate["id"], admitted=False)
        return

    if args.stage in {"learned", "reload"}:
        training = json.loads((args.out / "train.json").read_text())
        if training["status"] != "completed": raise ValueError("Completed training is required")
        data = (args.out / "selected.safetensors").read_bytes()
        restore_candidate(parameters, data, expected_identity=training["candidate"], expected_basis=source_basis)
        result["candidate"] = training["candidate"]
    if args.stage == "reload":
        probe = torch.load(args.out / "reload-probe.pt", map_location="cpu", weights_only=True)
        if probe["basis"] != source_basis or probe["candidate"] != result["candidate"]:
            raise ValueError("Fresh candidate probe basis differs")
        with torch.no_grad(): _, logits = loss(model, probe["example"])
        if not torch.equal(logits, probe["logits"]): raise ValueError("Fresh full-model logits differ")
        checks = []
        for row in json.loads((args.out / "learned.json").read_text())["rows"]:
            saved = owner.export(row["experience_id"])
            if saved["eligibility"]["status"] != "checked": raise ValueError("Recovered assessment is stale")
            path = memory.root / row["receipt"]["path"]
            before = (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
            receipt = owner.execute(row["experience_id"])
            if receipt != row["receipt"] or before != (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns):
                raise ValueError("Recovery repeated or changed the artifact")
            checks.append(row["case_id"])
        result.update(status="completed", exact_logits=True, once_only_cases=checks)
        record("fresh-recovery", exact_logits=True, cases=len(checks))
        return

    teacher = guidance(lessons, train) if args.stage == "guided" else None
    for row in corpus["rows"]:
        if row["split"] in {"confirmation", "control"}: publish(row, args.stage, teaching_text=teacher)
    scores = {split: {"correct": sum(r["correct"] for r in result["rows"] if r["split"] == split),
                      "total": sum(r["split"] == split for r in result["rows"])}
              for split in ("confirmation", "control")}
    result.update(status="completed", scores=scores)
    record("completed", scores=scores, admitted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "source", "preflight", "source-before", "protocol", "corpus", "lessons"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--stage", choices=("train", "original", "guided", "learned", "reload"), required=True)
    parser.add_argument("--collection", type=Path)
    args = parser.parse_args()
    path = args.out / (args.stage + ".json")
    if path.exists(): raise ValueError("Keep the previous measurement; use a fresh run")
    result = {"schema": "nanoai.teaching-evaluation.v1", "status": "running", "stage": args.stage, "phases": []}
    started = time.monotonic()
    from nanoai.core import file_digest
    from nanoai.tensor_store import TensorStore
    store = TensorStore(args.out / "state/tensors.sqlite3", create=True)
    protocol_identity = file_digest(args.protocol)
    def record(phase, **values):
        entry = {"phase": phase, "elapsed_seconds": time.monotonic() - started, **values}
        store.record(args.stage, protocol=protocol_identity, phase=phase,
            version=values.get("tensor_version"), values={**entry, "parameter_updates": result.get("parameter_updates", 0)})
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
