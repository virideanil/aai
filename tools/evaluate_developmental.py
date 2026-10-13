"""Run durable experience collection, candidate learning and held-out artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from nanoai.agent import DEFAULT_IDENTITY
from nanoai.core import digest_bytes, file_digest, process_memory, swap_bytes
from nanoai.experience import Experience, policy_choice
from nanoai.judgments import CHANNELS
from nanoai.learning import (Features, Learner, act, checked_episode, checkpoint_bytes,
                             encoded_texts, evaluate, load_checkpoint, restore_workspace, train)
from nanoai.memory import Memory, canonical, token_set


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")


def open_task(memory, episode):
    source_id = "notice-"+episode["id"]
    memory.initialize(DEFAULT_IDENTITY)
    if source_id not in memory.state["sources"]:
        memory.put_source(source_id, json.dumps({"observations": episode["observations"]}, ensure_ascii=False))
        memory.set_task(episode["id"], episode["question"], filenames=["answer.md"])
    source = memory.state["sources"][source_id]
    if json.loads(source["text"]) != {"observations": episode["observations"]}:
        raise ValueError("Existing episode source differs from the frozen input")
    return source_id


def collect(corpus, out, basis):
    memory = Memory(out/"collection-state")
    owner = Experience(memory)
    existing = {row["metadata"].get("id"): row["id"] for row in memory.state["experiences"].values()}
    records = []
    for i, episode in enumerate(corpus["episodes"]):
        source_id = open_task(memory, episode)
        if episode["id"] not in existing:
            source = memory.state["sources"][source_id]
            snapshot = memory.snapshot(episode["id"], [source], "uniform-experience-v1")
            behavior = policy_choice([c["id"] for c in episode["candidates"]], seed=1000+i)
            row = owner.record(snapshot, question=episode["question"], observations=episode["observations"],
                candidates=episode["candidates"], behavior=behavior,
                judgments=[dict(candidate_id=c["id"], verdict="maybe", reason="No learned assessment before this attempt", uncertainty="undecided") for c in episode["candidates"]],
                representation_identity=basis, source_id=source_id,
                metadata={k:episode[k] for k in ("id", "family", "split", "language", "mode")})
            existing[episode["id"]] = row["id"]
        experience_id = existing[episode["id"]]
        owner.execute(experience_id)
        owner.assess(experience_id, rubric=episode["rubric"], evaluator_identity=corpus["identity"])
        records.append(owner.export(experience_id))
    save(out/"checked-experiences.json", records)
    return records


def control_scores(features, train_episodes, test):
    lexical, nearest = [], []
    training_pairs = []
    for episode in train_episodes:
        context = F.normalize(torch.tensor([features.rows[t]["vector"] for t in episode["observations"]]).mean(0), dim=0)
        for candidate in episode["candidates"]:
            answer = F.normalize(torch.tensor(features.rows[candidate["content"]]["vector"]), dim=0)
            training_pairs.append((torch.cat((context, answer, context*answer)),
                                   float(candidate["id"] == episode["rubric"]["target_id"])))
    keys = F.normalize(torch.stack([k for k, _ in training_pairs]), dim=-1)
    values = torch.tensor([v for _, v in training_pairs])
    for episode in test:
        observed = [token_set(t) for t in episode["observations"]]
        scores = [max(len(token_set(c["content"]) & row)/max(1, len(token_set(c["content"]) | row)) for row in observed)
                  for c in episode["candidates"]]
        context = F.normalize(torch.tensor([features.rows[t]["vector"] for t in episode["observations"]]).mean(0), dim=0)
        nscores = []
        for candidate in episode["candidates"]:
            answer = F.normalize(torch.tensor(features.rows[candidate["content"]]["vector"]), dim=0)
            query = F.normalize(torch.cat((context, answer, context*answer)), dim=0)
            similarity, indices = (keys@query).topk(min(5,len(values)))
            nscores.append(float((similarity.softmax(0)*values[indices]).sum()))
        for arm, score in ((lexical, scores), (nearest, nscores)):
            selected = episode["candidates"][max(range(len(score)), key=lambda i:score[i])]["id"]
            arm.append({"id": episode["id"], "selected_id": selected, "correct": selected == episode["rubric"]["target_id"], "scores": score})
    return {"lexical_all_observations": lexical, "nearest_checked_pairs": nearest}


def run_artifacts(out, name, corpus, features, model, data, episodes):
    memory = Memory(out/(name+"-state")); memory.initialize(DEFAULT_IDENTITY)
    owner = Experience(memory)
    candidate = owner.register_candidate(data, feature_basis=features.basis,
                                         training={"name": name, "corpus": corpus["identity"]})
    results = []
    for episode in episodes:
        source_id = open_task(memory, episode)
        result = act(memory, features, model, episode["id"], source_id, episode["candidates"], learner_id=candidate["id"])
        assessment = owner.assess(result["experience"]["id"], rubric=episode["rubric"], evaluator_identity=corpus["identity"])
        results.append({"id": episode["id"], "experience_id": result["experience"]["id"], "mode": episode["mode"],
            "selected_id": result["experience"]["behavior"]["selected_id"],
            "correct": assessment["observed_outcome"] == "yes", "seconds": result["elapsed_seconds"],
            "receipt": result["receipt"], "workspace_id": result["workspace"]["id"]})
    save(out/(name+"-artifacts.json"), results)
    return {"rows": results, "correct": sum(r["correct"] for r in results), "count":len(results),
            "candidate":candidate, "state":str(memory.root)}


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=["collect", "train", "artifacts"], required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()
    preflight = json.loads(args.preflight.read_text())
    if preflight.get("model") != "nanoai-developmental-f32" or preflight.get("effort") != "none" or not preflight.get("parent_source"):
        raise ValueError("A sourced candidate launch preflight is required")
    corpus = json.loads(args.corpus.read_text())
    if digest_bytes(canonical({k:v for k,v in corpus.items() if k != "identity"})) != corpus["identity"]:
        raise ValueError("Frozen corpus identity differs")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == "collect":
        manifest = json.loads((args.features/"manifest.json").read_text())
        records = collect(corpus, args.out, manifest["basis"])
        print(json.dumps({"checked": len(records), "actual_successes": sum(r["assessment"]["observed_outcome"] == "yes" for r in records)}))
        return
    features = Features(args.features)
    if features.manifest["corpus"] != file_digest(args.corpus):
        raise ValueError("Features belong to another frozen corpus")
    memory = Memory(args.out/"collection-state")
    owner = Experience(memory)
    records = [owner.export(row["id"]) for row in memory.state["experiences"].values()]
    episodes = [checked_episode(record) for record in records]
    if {e["id"] for e in episodes} != {e["id"] for e in corpus["episodes"]}:
        raise ValueError("Checked experience collection is incomplete")
    train_rows = [e for e in episodes if e["split"] == "train" and e["mode"] != "correction_back"]
    corrective = [e for e in episodes if e["split"] == "train" and e["mode"] == "correction_back"]
    validation = [e for e in episodes if e["split"] == "validation"]
    test = [e for e in episodes if e["split"] == "test"]
    if args.phase == "train":
        import os
        started, swap = time.monotonic(), swap_bytes()
        summary = {"schema":"nanoai.developmental-results.v1", "corpus":corpus["identity"], "feature_basis":features.basis,
            "controls":control_scores(features, train_rows, test), "runs":[], "status":"running"}
        try:
            for seed in (11, 23, 37):
                torch.manual_seed(seed)
                untrained = Learner()
                summary.setdefault("untrained",[]).append({"seed":seed, "test":evaluate(untrained, features, test)})
                for no_memory in (False, True):
                    name = f"seed{seed}"+("-last" if no_memory else "-recurrent")
                    model, training = train(features, train_rows, validation, seed=seed, no_memory=no_memory)
                    data = checkpoint_bytes(model, feature_basis=features.basis, corpus_identity=corpus["identity"],
                        experience_ids=[e["experience_id"] for e in train_rows], training=training)
                    path = args.out/(name+".pt"); path.write_bytes(data)
                    baseline = {"name":name, "checkpoint":path.name, "identity":digest_bytes(data), "training":training,
                        "test":evaluate(model, features, test), "retention":evaluate(model, features, train_rows+validation)}
                    summary["runs"].append(baseline)
                    print(json.dumps({"name":name, "selected_epoch":training["selected_epoch"], "test_correct":baseline["test"]["correct"], "test_n":len(test), "seconds":training["seconds"]}), flush=True)
                    save(args.out/"results.json", summary)
                    if not no_memory:
                        revised, retraining = train(features, train_rows+corrective, validation, seed=seed,
                                                   parent=model, epochs=100)
                        revised_data = checkpoint_bytes(revised, feature_basis=features.basis, corpus_identity=corpus["identity"],
                            experience_ids=[e["experience_id"] for e in train_rows+corrective], training=retraining)
                        revised_path = args.out/(name+"-consolidated.pt"); revised_path.write_bytes(revised_data)
                        revised_retention = evaluate(revised, features, train_rows+validation)
                        regressions = [b["id"] for b,a in zip(baseline["retention"]["rows"], revised_retention["rows"]) if b["correct"] and not a["correct"]]
                        summary["runs"].append({"name":name+"-consolidated", "checkpoint":revised_path.name,
                            "identity":digest_bytes(revised_data), "parent":baseline["identity"], "training":retraining,
                            "test":evaluate(revised, features, test), "retention":revised_retention,
                            "retention_regressions":regressions})
                        save(args.out/"results.json", summary)
            summary["status"] = "completed"
        except BaseException as error:
            summary.update(status="failed", error=str(error)); raise
        finally:
            summary.update(seconds=time.monotonic()-started, process_memory=process_memory(os.getpid()),
                           host_swap_before=swap, host_swap_after=swap_bytes())
            save(args.out/"results.json",summary)
    else:
        summary = json.loads((args.out/"results.json").read_text())
        artifacts = []
        for run in summary["runs"]:
            path = args.out/run["checkpoint"]
            model, _ = load_checkpoint(path, expected=run["identity"], feature_basis=features.basis)
            result = run_artifacts(args.out, run["name"], corpus, features, model, path.read_bytes(), test)
            predicted = {r["id"]:r["correct"] for r in run["test"]["rows"]}
            if any(predicted[r["id"]] != r["correct"] for r in result["rows"]):
                raise ValueError("Published artifact assessment differs from candidate evaluation")
            artifacts.append({"name":run["name"], **result})
            save(args.out/"artifact-results.json",artifacts)
            print(json.dumps({"name":run["name"], "published":result["count"], "correct":result["correct"]}), flush=True)


if __name__ == "__main__":
    main()
