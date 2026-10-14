from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from nanoai.agent import DEFAULT_IDENTITY
from nanoai.core import digest_bytes
from nanoai.experience import Experience, policy_choice
from nanoai.harmonizer import Harmonizer
from nanoai.memory import Memory, StaleState


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = Memory(self.tmp.name)
        self.memory.initialize(DEFAULT_IDENTITY)
        self.observations = ["The reading is in the east hall."]
        self.source = self.memory.put_source("notice", json.dumps({"observations": self.observations}))
        self.memory.set_task("reading", "Where is the reading?", filenames=["answer.md"])
        self.owner = Experience(self.memory)
        self.candidates = [{"id": "east", "kind": "answer", "content": "In the east hall."},
                           {"id": "ask", "kind": "inquire", "content": "Please provide the room."}]
        self.snapshot = self.memory.snapshot("reading", [self.source], "test-core")
        self.rubric = {"target_id": "east", "verdicts": {"east": "yes", "ask": "no"}, "scope": "Authored room notice"}

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, **changes):
        values = dict(question="Where is the reading?", observations=self.observations,
            candidates=self.candidates, behavior=policy_choice(["east", "ask"], scores=[1., 0.]),
            judgments=[{"candidate_id": c["id"], "verdict": "maybe", "reason": "Unassessed", "uncertainty": "undecided"} for c in self.candidates],
            representation_identity="source-test-basis", source_id="notice")
        values.update(changes)
        return self.owner.record(self.snapshot, **values)

    def test_chosen_effect_assessment_recovery_and_correction(self):
        row = self.record()
        with self.assertRaises(ValueError):
            self.owner.assess(row["id"], rubric=self.rubric, evaluator_identity="frozen-rubric")
        receipt = self.owner.execute(row["id"])
        self.assertEqual(self.owner.execute(row["id"]), receipt)
        assessment = self.owner.assess(row["id"], rubric=self.rubric, evaluator_identity="frozen-rubric")
        self.assertEqual(assessment["observed_outcome"], "yes")
        fresh = Experience(Memory(self.tmp.name))
        self.assertEqual(fresh.export(row["id"])["eligibility"]["status"], "checked")
        viewed = Harmonizer(Memory(self.tmp.name)).learning_view("reading")
        self.assertEqual(viewed["action_experiences"][0]["assessment"]["observed_outcome"], "yes")
        self.assertEqual(viewed["parameter_updates"], 0)
        self.memory.put_source("notice", '{"observations":["The room has changed."]}', expected_revision=1)
        self.assertEqual(fresh.export(row["id"])["eligibility"]["status"], "invalidated")
        history = Harmonizer(Memory(self.tmp.name)).learning_view("reading")["action_experiences"][0]
        self.assertEqual(history["eligibility"]["status"], "invalidated")
        self.assertEqual(history["assessment"], assessment)
        self.assertTrue((Path(self.tmp.name)/receipt["path"]).is_file())

    def test_incorrect_choice_remains_checked_negative_experience(self):
        row = self.record(behavior=policy_choice(["east", "ask"], scores=[0., 1.]))
        self.owner.execute(row["id"])
        result = self.owner.assess(row["id"], rubric=self.rubric, evaluator_identity="rubric")
        self.assertEqual(result["observed_outcome"], "no")
        self.assertEqual(self.owner.export(row["id"])["eligibility"]["status"], "checked")
        fresh = Memory(self.tmp.name)
        self.assertEqual(fresh.completed("reading")["remaining"], ["answer.md"])
        status = fresh.status()
        self.assertEqual(status["obligations"]["reading"]["remaining"], ["answer.md"])
        self.assertEqual(status["artifacts"][0]["outcome"]["status"], "failed")

    def test_input_and_real_propensity_are_bound_before_mutation(self):
        before = self.memory.sequence
        with self.assertRaises(ValueError): self.record(observations=["Invented observation"])
        behavior = policy_choice(["east", "ask"], scores=[1., 0.])
        behavior["probabilities"] = [.5, .5]
        with self.assertRaises(ValueError): self.record(behavior=behavior)
        self.assertEqual(self.memory.sequence, before)
        self.assertEqual(policy_choice(["east", "ask"], seed=7), policy_choice(["east", "ask"], seed=7))

    def test_tampered_artifact_invalidates_learning(self):
        row = self.record(); receipt = self.owner.execute(row["id"])
        self.owner.assess(row["id"], rubric=self.rubric, evaluator_identity="rubric")
        (Path(self.tmp.name)/receipt["path"]).write_text("changed")
        self.assertEqual(self.owner.export(row["id"])["eligibility"]["status"], "invalidated")

    def test_candidate_lineage_admission_and_snapshot_bytes(self):
        candidate = self.owner.register_candidate(b"candidate fixture", feature_basis="test")
        self.assertEqual(candidate["id"], digest_bytes(b"candidate fixture"))
        with self.assertRaises(ValueError):
            self.owner.admit(candidate["id"], dict(test_accuracy=.5, retention_regressions=0, critical_failures=0, evidence="test"))
        self.owner.admit(candidate["id"], dict(test_accuracy=.9, retention_regressions=0, critical_failures=0, evidence="fixture admission"))
        self.assertEqual(Memory(self.tmp.name).state["active_learner"], candidate["id"])
        snapshot = {**self.snapshot, "learner": candidate["id"], "core_identity": candidate["id"]}
        self.memory._validate_snapshot(snapshot)
        (Path(self.tmp.name)/candidate["path"]).write_bytes(b"changed")
        with self.assertRaises(StaleState): self.memory._validate_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()
