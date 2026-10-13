"""Exercise actual candidate/workspace process recovery and observed effects."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from nanoai.core import digest_bytes, file_digest
from nanoai.experience import Experience, policy_choice
from nanoai.harmonizer import Harmonizer
from nanoai.judgments import CHANNELS
from nanoai.learning import Features, encoded_texts, load_checkpoint, restore_workspace
from nanoai.memory import Memory, StaleState
from nanoai.scratchboard import Scratchboard
from evaluate_developmental import open_task, save


def require_failure(call, message):
    try:
        call()
    except (ValueError, KeyError, OSError):
        return
    raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser()
    for name in ("corpus", "features", "checkpoint", "out", "preflight"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--phase", choices=("run", "prepare", "resume"), default="run")
    args = parser.parse_args()
    preflight = json.loads(args.preflight.read_text())
    if preflight.get("model") != "nanoai-developmental-f32" or preflight.get("effort") != "none" or not preflight.get("parent_source"):
        raise ValueError("A sourced candidate launch preflight is required")
    torch.set_num_threads(4)
    corpus = json.loads(args.corpus.read_text())
    episode = next(e for e in corpus["episodes"] if e["split"] == "test" and e["mode"] == "correction_back")
    args.out.mkdir(exist_ok=True, parents=True)
    state = args.out/"state"
    record_path = args.out/"boundary.json"
    if args.phase == "run":
        if record_path.exists() or (state/"transactions.sjql.d").exists():
            raise ValueError("Recovery evaluation requires a fresh output directory")
        commands = [sys.executable, __file__, "--corpus", str(args.corpus), "--features", str(args.features),
                    "--checkpoint", str(args.checkpoint), "--out", str(args.out), "--preflight", str(args.preflight)]
        runs = []
        for phase, expected in (("prepare", 87), ("resume", 88)):
            result = subprocess.run([*commands, "--phase", phase], capture_output=True, text=True, timeout=90)
            runs.append({"phase":phase, "returncode":result.returncode, "stdout":result.stdout, "stderr":result.stderr})
            save(args.out/"processes.json", runs)
            if result.returncode != expected:
                raise AssertionError(f"{phase} exited {result.returncode}: {result.stderr}")
        boundary = json.loads(record_path.read_text())
        memory = Memory(state)
        before = list((state/"artifacts").glob("*/*"))
        if len(before) != 1 or memory.state["receipts"]:
            raise AssertionError("Injected publication boundary did not retain one unreceipted artifact")
        artifact = before[0]; identity = file_digest(artifact)
        inode = artifact.stat().st_ino; modified = artifact.stat().st_mtime_ns
        recovered = memory.recover()
        if len(memory.state["receipts"]) != 1 or len(memory.state["operations"]) != 1:
            raise AssertionError("Recovery did not reconcile exactly one observed effect")
        if memory.recover() or file_digest(artifact) != identity or artifact.stat().st_ino != inode or artifact.stat().st_mtime_ns != modified:
            raise AssertionError("Repeated recovery changed the artifact or duplicated its receipt")
        owner = Experience(memory)
        assessment = owner.assess(boundary["experience_id"], rubric=episode["rubric"], evaluator_identity=corpus["identity"])
        view = Harmonizer(memory).learning_view(episode["id"])
        if view["action_experiences"][0]["eligibility"]["status"] != "checked":
            raise AssertionError("Recovered observed experience is ineligible")
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/"src"))
        def cli(*arguments):
            run = subprocess.run([sys.executable, "-m", "nanoai.cli", "--state", str(state), *arguments],
                                 capture_output=True, text=True, timeout=60, env=environment)
            if run.returncode: raise AssertionError(run.stderr)
            return json.loads(run.stdout)
        inspected = cli("experiences", episode["id"])
        if inspected["action_experiences"][0]["assessment"] != assessment or inspected["parameter_updates"] != 0:
            raise AssertionError("CLI did not preserve the actual assessment")
        candidates = cli("learners")
        if candidates["active_learner"] is not None or not all(row["bytes_current"] for row in candidates["candidates"]):
            raise AssertionError("Inspection changed admission or lost candidate integrity")
        # A separate task exercises the actual candidate CLI; it shares exact
        # input bytes while owning a distinct permitted artifact effect.
        trial = deepcopy(episode); trial["id"] += "-cli"
        source_id = open_task(memory, trial)
        choices = args.out/"choices.json"; save(choices, trial["candidates"])
        chosen = cli("trial-act", trial["id"], "--source", source_id, "--candidate", boundary["candidate_id"],
                     "--features", str(args.features), "--choices", str(choices))
        if chosen["candidate_admitted"] or chosen["scores"] != boundary["scores"]:
            raise AssertionError("CLI candidate state or decision differs")
        if Path(chosen["artifact_path"]).read_text() != Experience.content(chosen["experience"]):
            raise AssertionError("CLI artifact differs from its actual choice")
        # Different learned parameters are incompatible with the saved basis.
        require_failure(lambda: restore_workspace(memory, boundary["prefix_node"], learner_id="another-candidate"),
                        "Changed candidate reused the old latent basis")
        # Original observations remain available, with current-source eligibility
        # invalidated after a task-relevant correction.
        memory.put_source(boundary["source_id"], json.dumps({"observations":["The notice has been replaced."]}), expected_revision=1)
        require_failure(lambda: restore_workspace(memory, boundary["prefix_node"], learner_id=boundary["candidate_id"]),
                        "Corrected source reused stale recurrent state")
        stale = owner.export(boundary["experience_id"])
        if stale["eligibility"]["status"] != "invalidated" or stale["assessment"] != assessment or file_digest(artifact) != identity:
            raise AssertionError("Correction lost history or retained current eligibility")
        require_failure(lambda: load_checkpoint(args.checkpoint, expected=boundary["candidate_id"], feature_basis="changed-feature-basis"),
                        "Changed features reused old learned parameters")
        save(args.out/"results.json", {"schema":"nanoai.developmental-recovery.v1", "status":"completed",
            "candidate":boundary["candidate_id"], "corpus":file_digest(args.corpus), "episode":episode["id"],
            "checks":{
                "actual_process_exits":[r["returncode"] for r in runs], "fresh_f32_state_bit_exact":True,
                "fresh_decision_scores_bit_exact":True, "one_observed_publication":True,
                "repeat_recovery_keeps_bytes_inode_mtime":True, "checked_outcome_after_recovery":True,
                "cli_experience_and_candidate_inspection":True, "cli_actual_learned_artifact":True,
                "different_candidate_rejected":True, "source_correction_invalidates_state":True,
                "historical_assessment_and_artifact_retained":True, "changed_feature_basis_rejected":True},
            "observed_task_outcome":assessment["observed_outcome"], "recovered":recovered,
            "admitted":False, "scope":"This trained candidate, F32 CPU with four threads, authored correction task and real local artifacts"})
        print(json.dumps({"status":"completed", "result":str(args.out/"results.json")}), flush=True)
        return
    features = Features(args.features)
    model, _ = load_checkpoint(args.checkpoint, expected=file_digest(args.checkpoint), feature_basis=features.basis)
    memory = Memory(state); source_id = open_task(memory, episode)
    owner = Experience(memory)
    candidate = owner.register_candidate(args.checkpoint.read_bytes(), feature_basis=features.basis,
                                          training={"role":"measured recovery control"})
    snapshot = memory.snapshot(episode["id"], [memory.state["sources"][source_id]], candidate["id"])
    snapshot["learner"] = candidate["id"]
    with torch.no_grad():
        encoded = encoded_texts(model, features, [episode])
        observations = torch.stack([encoded[t] for t in episode["observations"]])
        question = encoded[episode["question"]]
        candidates = torch.stack([encoded[c["content"]] for c in episode["candidates"]])
        if args.phase == "prepare":
            complete = model.advance(observations)
            scores, verdicts = model.decide(complete, question, candidates)
            cut = max(1, len(observations)//2)
            prefix = model.advance(observations[:cut])
            saved = Scratchboard(memory).tensor(snapshot, prefix.numpy().astype("<f4").tobytes(),
                shape=list(prefix.shape), dtype="f32", origin={"operation":"learner.workspace", "prefix_observations":cut})
            save(record_path, {"candidate_id":candidate["id"], "source_id":source_id, "prefix_node":saved["id"],
                "prefix_count":cut, "complete_state_digest":digest_bytes(complete.numpy().astype("<f4").tobytes()),
                "scores":scores.tolist(), "verdicts":verdicts.tolist()})
            os._exit(87)
        boundary = json.loads(record_path.read_text())
        prefix = restore_workspace(memory, boundary["prefix_node"], learner_id=candidate["id"])
        resumed = model.advance(observations[boundary["prefix_count"]:], saved=prefix)
        scores, verdicts = model.decide(resumed, question, candidates)
        if digest_bytes(resumed.numpy().astype("<f4").tobytes()) != boundary["complete_state_digest"] or scores.tolist() != boundary["scores"] or verdicts.tolist() != boundary["verdicts"]:
            raise AssertionError("Fresh restored workspace changed the uninterrupted computation")
        saved = Scratchboard(memory).tensor(snapshot, resumed.numpy().astype("<f4").tobytes(), shape=list(resumed.shape),
            dtype="f32", origin={"operation":"learner.workspace", "observations":len(observations)})
        judgments = [{"candidate_id":c["id"], "verdict":CHANNELS[int(v.argmax())],
                      "reason":"Provisional learned classification", "uncertainty":"undecided" if int(v.argmax()) == 1 else None}
                     for c,v in zip(episode["candidates"],verdicts)]
        experience = owner.record(snapshot, question=episode["question"], observations=episode["observations"],
            candidates=episode["candidates"], behavior=policy_choice([c["id"] for c in episode["candidates"]],scores=scores.tolist()),
            judgments=judgments, representation_identity=features.basis, source_id=source_id,
            workspace_node=saved["id"], metadata={"role":"recovery control"})
        boundary["experience_id"] = experience["id"]; save(record_path, boundary)
        owner.execute(experience["id"], crash_after_publish=True)
        raise AssertionError("Publication fault injection did not exit")


if __name__ == "__main__":
    main()
