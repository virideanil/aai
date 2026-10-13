"""Attributed choices, observed artifacts and candidate versions in one journal.

The exact menu rubric assesses an authored task. It is kept outside inference
inputs and distinguishes an observed chosen effect from replayable alternatives.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import tempfile

from .core import digest_bytes, file_digest
from .judgments import CHANNELS, MAYBE_CAUSES
from .memory import StaleState, bounded_text, canonical, identifier


def apply_event(state, event):
    kind = event["kind"]
    table, key, row = None, None, None
    if kind == "experience.recorded":
        table, row = "experiences", event["experience"]
        key = row["id"]
        if digest_bytes(canonical({k: v for k, v in row.items() if k != "id"})) != key:
            raise ValueError("Experience identity differs")
    elif kind == "experience.proposed":
        table, key, row = "experience_proposals", event["experience_id"], event["proposal_id"]
        if key not in state["experiences"] or row not in state["proposals"]:
            raise ValueError("Experience proposal dependency is unavailable")
    elif kind == "experience.assessed":
        table, row = "experience_assessments", event["assessment"]
        key = row["experience_id"]
        if key not in state["experiences"] or key not in state["experience_proposals"]:
            raise ValueError("Experience assessment dependency is unavailable")
    elif kind == "learner.registered":
        table, row = "learners", event["learner"]
        key = row["id"]
        if row["parent"] is not None and row["parent"] not in state["learners"]:
            raise ValueError("Candidate parent is unavailable")
    elif kind == "learner.admitted":
        if event["learner_id"] not in state["learners"] or event["previous"] != state["active_learner"]:
            raise ValueError("Candidate admission parent differs")
        state["active_learner"] = event["learner_id"]
        return
    else:
        raise ValueError("Unknown learning event")
    if key in state[table]:
        raise ValueError("Learning records are immutable")
    state[table][key] = row


def policy_choice(candidate_ids, *, scores=None, seed=None):
    """Record the actual behavior probabilities, including greedy one-hot choice."""
    if not 2 <= len(candidate_ids) <= 16 or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Choice set extent")
    if scores is None:
        probabilities = [1/len(candidate_ids)]*len(candidate_ids)
        if type(seed) is not int:
            raise ValueError("Uniform behavior needs a reproducible seed")
        chosen = random.Random(seed).randrange(len(candidate_ids))
        policy = "uniform-v1"
    else:
        if len(scores) != len(candidate_ids) or any(type(x) not in (int, float) or not math.isfinite(x) for x in scores):
            raise ValueError("Finite score for every candidate is required")
        chosen = max(range(len(scores)), key=lambda i: scores[i])
        probabilities = [float(i == chosen) for i in range(len(scores))]
        policy = "greedy-first-tie-v1"
    return {"policy": policy, "probabilities": probabilities, "selected_id": candidate_ids[chosen],
            "seed": seed, "scores": scores}


class Experience:
    def __init__(self, memory):
        self.memory = memory
        self.implementation = file_digest(__file__)

    def record(self, snapshot, *, question, observations, candidates, behavior, judgments,
               representation_identity, source_id, metadata=None, workspace_node=None, vocabulary=None):
        """The source owns the complete observed trajectory, in its actual order."""
        bounded_text(question, "experience question", 2048)
        bounded_text(representation_identity, "representation identity", 256)
        if (not isinstance(observations, list) or not 1 <= len(observations) <= 16
                or not isinstance(candidates, list) or not 2 <= len(candidates) <= 16):
            raise ValueError("Experience trajectory or choice extent")
        for text in observations:
            bounded_text(text, "observation", 8192)
        ids = []
        for row in candidates:
            if set(row) != {"id", "kind", "content"} or row["kind"] not in {"answer", "inquire"}:
                raise ValueError("Admitted document action contract")
            ids.append(identifier(row["id"]))
            bounded_text(row["content"], "candidate content", 8192)
        if len(set(ids)) != len(ids) or behavior["selected_id"] not in ids:
            raise ValueError("Choice identity differs")
        probabilities = behavior["probabilities"]
        if (len(probabilities) != len(ids) or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
                or abs(sum(probabilities)-1) > 1e-8 or probabilities[ids.index(behavior["selected_id"])] == 0):
            raise ValueError("Actual selection propensity differs")
        expected_policy = policy_choice(ids, scores=behavior.get("scores"), seed=behavior.get("seed"))
        if behavior != expected_policy:
            raise ValueError("Recorded policy is inconsistent with its action")
        if len(judgments) != len(ids) or {j["candidate_id"] for j in judgments} != set(ids):
            raise ValueError("Each candidate needs a scoped provisional judgment")
        for judgment in judgments:
            if (set(judgment) != {"candidate_id", "verdict", "reason", "uncertainty"}
                    or judgment["verdict"] not in CHANNELS
                    or (judgment["verdict"] == "maybe" and judgment["uncertainty"] not in MAYBE_CAUSES)
                    or (judgment["verdict"] != "maybe" and judgment["uncertainty"] is not None)):
                raise ValueError("Experience judgment type")
            bounded_text(judgment["reason"], "judgment reason", 2048)
        with self.memory.locked():
            self.memory._validate_snapshot(snapshot)
            if source_id not in snapshot["sources"]:
                raise ValueError("Experience source is outside its current snapshot")
            source = self.memory.state["sources"][source_id]
            if json.loads(source["text"]) != {"observations": observations}:
                raise ValueError("Experience observations differ from their exact source")
            if self.memory.state["tasks"][snapshot["task_id"]]["goal"] != question:
                raise ValueError("Experience question differs from the current task")
            if workspace_node is not None:
                from .scratchboard import Scratchboard
                node = Scratchboard(self.memory)._validate(workspace_node)
                if node["spec"]["kind"] != "tensor" or node["spec"]["basis"] != snapshot["core_identity"]:
                    raise ValueError("Experience workspace basis differs")
                snapshot = deepcopy(snapshot)
                snapshot.setdefault("scratchboard", {})[workspace_node] = node["digest"]
                self.memory._validate_snapshot(snapshot)
            body = {"schema": "nanoai.experience.v1", "snapshot": snapshot, "question": question,
                "observations": observations, "candidates": candidates, "behavior": behavior,
                "judgments": judgments, "representation_identity": representation_identity,
                "source_id": source_id, "workspace_node": workspace_node, "vocabulary": vocabulary,
                "metadata": metadata or {}, "implementation": self.implementation}
            if len(canonical(body)) > 65536:
                raise ValueError("Experience record extent")
            row = {"id": digest_bytes(canonical(body)), **body}
            self.memory._append({"kind": "experience.recorded", "experience": row})
            return deepcopy(row)

    def _current(self, experience_id):
        row = self.memory.state["experiences"][experience_id]
        if row["implementation"] != self.implementation:
            raise StaleState("Experience implementation changed")
        self.memory._validate_snapshot(row["snapshot"])
        return row

    @staticmethod
    def content(row, candidate_id=None):
        candidate_id = candidate_id or row["behavior"]["selected_id"]
        candidate = next(c for c in row["candidates"] if c["id"] == candidate_id)
        source_id = row["source_id"]
        revision = row["snapshot"]["sources"][source_id]["revision"]
        return candidate["content"]+f" [{source_id}@{revision}]\n"

    def execute(self, experience_id, *, crash_after_publish=False):
        """Publish through the existing artifact owner and its once-only receipt."""
        with self.memory.locked():
            row = deepcopy(self._current(experience_id))
            proposal_id = self.memory.state["experience_proposals"].get(experience_id)
            filename = self.memory.state["tasks"][row["snapshot"]["task_id"]]["filenames"][0]
        if proposal_id is None:
            source_id = row["source_id"]
            citation = f"{source_id}@{row['snapshot']['sources'][source_id]['revision']}"
            proposal = self.memory.record_proposal(row["snapshot"], {"filename": filename,
                "content": self.content(row), "citations": [citation]},
                metrics={"experience_id": experience_id, "action": row["behavior"]["selected_id"]})
            with self.memory.locked():
                self._current(experience_id)
                if experience_id not in self.memory.state["experience_proposals"]:
                    self.memory._append({"kind": "experience.proposed", "experience_id": experience_id, "proposal_id": proposal["id"]})
                proposal_id = self.memory.state["experience_proposals"][experience_id]
        return self.memory.commit(proposal_id, core_identity=row["snapshot"]["core_identity"],
                                  crash_after_publish=crash_after_publish)

    def _receipt(self, experience_id):
        proposal_id = self.memory.state["experience_proposals"].get(experience_id)
        for receipt in self.memory.state["receipts"].values():
            op = self.memory.state["operations"][receipt["operation_id"]]
            if op["proposal_id"] == proposal_id:
                return self.memory._verify_receipt(receipt)
        raise ValueError("A chosen action needs an observed artifact receipt")

    def assess(self, experience_id, *, rubric, evaluator_identity):
        """Read the chosen artifact and apply a separately frozen authored rubric."""
        bounded_text(evaluator_identity, "evaluator identity", 256)
        with self.memory.locked():
            row = self._current(experience_id)
            ids = {c["id"] for c in row["candidates"]}
            if (set(rubric) != {"target_id", "verdicts", "scope"} or rubric["target_id"] not in ids
                    or set(rubric["verdicts"]) != ids or any(v not in CHANNELS for v in rubric["verdicts"].values())):
                raise ValueError("Frozen menu rubric differs")
            receipt = self._receipt(experience_id)
            actual = self.memory._artifact_path(receipt["path"]).read_text()
            if actual != self.content(row):
                raise ValueError("Observed chosen action differs from the proposal")
            expected = self.content(row, rubric["target_id"])
            assessment = {"experience_id": experience_id, "evaluator_identity": evaluator_identity,
                "scope": rubric["scope"], "observed_outcome": "yes" if actual == expected else "no",
                "reason": "Exact chosen artifact compared with the frozen authored task target",
                "receipt": receipt, "observed_content_digest": digest_bytes(actual.encode()),
                "replayable_targets": deepcopy(rubric),
                "next_observation": "The chosen artifact satisfied the task rubric." if actual == expected else "The chosen artifact missed the task rubric; a corrected choice is available.",
                "implementation": self.implementation}
            prior = self.memory.state["experience_assessments"].get(experience_id)
            if prior is not None:
                if prior != assessment:
                    raise ValueError("An assessment is immutable; revise the task or evaluator as a new experience")
                return deepcopy(prior)
            self.memory._append({"kind": "experience.assessed", "assessment": assessment})
            return deepcopy(assessment)

    def export(self, experience_id):
        with self.memory.locked():
            row = deepcopy(self.memory.state["experiences"][experience_id])
            assessment = deepcopy(self.memory.state["experience_assessments"].get(experience_id))
            status, reason = "checked" if assessment else "unassessed", None
            try:
                self._current(experience_id)
                if assessment is not None:
                    receipt = self._receipt(experience_id)
                    if receipt != assessment["receipt"] or assessment["implementation"] != self.implementation:
                        raise StaleState("Assessment receipt or evaluator changed")
            except (ValueError, KeyError, OSError) as error:
                status, reason = "invalidated", str(error)
            return {**row, "assessment": assessment, "eligibility": {"status": status, "reason": reason}}

    def register_candidate(self, data, *, feature_basis, parent=None, training=None):
        if not isinstance(data, bytes) or not 1 <= len(data) <= 32 << 20:
            raise ValueError("Candidate checkpoint extent")
        identity = digest_bytes(data)
        relative = "learners/"+identity.split(":")[1]+".pt"
        with self.memory.locked():
            if identity in self.memory.state["learners"]:
                row = self.memory.state["learners"][identity]
                if file_digest(self.memory._artifact_path(row["path"])) != identity:
                    raise ValueError("Registered candidate bytes changed")
                return deepcopy(row)
            if parent is not None and parent not in self.memory.state["learners"]:
                raise ValueError("Register the candidate's parent first")
            path = self.memory._artifact_path(relative)
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.memory._artifact_path(relative)
            temporary = None
            try:
                if not path.exists():
                    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".staging-", delete=False) as stream:
                        temporary = Path(stream.name); stream.write(data); stream.flush(); os.fsync(stream.fileno())
                    os.link(temporary, path); self.memory.fs.sync_directory(path.parent)
                if file_digest(path) != identity:
                    raise ValueError("Published candidate bytes differ")
                row = {"id": identity, "path": relative, "parent": parent,
                       "feature_basis": feature_basis, "training": training or {}, "implementation": self.implementation}
                self.memory._append({"kind": "learner.registered", "learner": row})
                return deepcopy(row)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def admit(self, candidate_id, review):
        if (set(review) != {"test_accuracy", "retention_regressions", "critical_failures", "evidence"}
                or not math.isfinite(review["test_accuracy"]) or not .75 <= review["test_accuracy"] <= 1
                or review["retention_regressions"] != 0 or review["critical_failures"] != 0):
            raise ValueError("Candidate admission comparison did not meet the frozen contract")
        bounded_text(review["evidence"], "admission evidence", 4096)
        with self.memory.locked():
            candidate = self.memory.state["learners"][candidate_id]
            if file_digest(self.memory._artifact_path(candidate["path"])) != candidate_id:
                raise ValueError("Candidate bytes changed before admission")
            self.memory._append({"kind": "learner.admitted", "learner_id": candidate_id,
                "previous": self.memory.state["active_learner"], "review": review})
