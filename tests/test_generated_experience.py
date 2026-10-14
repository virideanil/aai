import json
from pathlib import Path
import tempfile
import unittest

from nanoai.agent import DEFAULT_IDENTITY
from nanoai.experience import Experience
from nanoai.harmonizer import Harmonizer
from nanoai.memory import Memory


def assess_notice(text, criteria):
    return {"current_location": criteria["location"] in text,
            "superseded_location_absent": criteria["old_location"] not in text}


class GeneratedExperienceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = Memory(self.tmp.name)
        self.memory.initialize(DEFAULT_IDENTITY)
        self.question = "Write the current visitor notice."
        self.observations = ["The tour moved from the west room to the east hall."]
        source = self.memory.put_source("notice", json.dumps({"observations": self.observations}))
        self.memory.set_task("tour", self.question, filenames=["notice.md"])
        self.snapshot = self.memory.snapshot("tour", [source], "original-model")
        self.owner = Experience(self.memory)
        self.rubric = {"scope": "Location in this visitor notice",
                       "criteria": {"location": "east hall", "old_location": "west room"},
                       "training_target": "The tour meets in the east hall."}

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, text="Please meet in the east hall.", finish="stop", **changes):
        values = dict(question=self.question, observations=self.observations,
                      messages=[{"role": "user", "content": self.question+"\n"+self.observations[0]}],
                      generation={"text": text, "token_ids": [12, 18, 4],
                                  "finish_reason": finish, "policy": "greedy-token-v1"},
                      representation_identity="qualified-tokenizer", source_id="notice")
        values.update(changes)
        return self.owner.record_generation(self.snapshot, **values)

    def assess(self, row):
        return self.owner.assess_generation(row["id"], rubric=self.rubric,
                                           evaluator_identity="notice-rubric-v1", evaluate=assess_notice)

    def test_generated_paraphrase_uses_actual_effect_and_recovers_once(self):
        row = self.record()
        self.assertNotIn("candidates", row)
        with self.assertRaises(ValueError):
            self.assess(row)
        receipt = self.owner.execute(row["id"])
        path = Path(self.tmp.name)/receipt["path"]
        before = path.stat()
        result = self.assess(row)
        self.assertEqual(result["observed_outcome"], "yes")
        self.assertNotEqual(row["generation"]["text"], self.rubric["training_target"])
        fresh = Experience(Memory(self.tmp.name))
        self.assertEqual(fresh.execute(row["id"]), receipt)
        self.assertEqual((path.stat().st_ino, path.stat().st_mtime_ns), (before.st_ino, before.st_mtime_ns))
        self.assertEqual(fresh.export(row["id"])["eligibility"]["status"], "checked")
        self.assertEqual(Harmonizer(Memory(self.tmp.name)).learning_view("tour")["action_experiences"][0]["assessment"], result)

    def test_wrong_generated_work_is_preserved_with_separate_correction(self):
        row = self.record("Meet in the west room.")
        receipt = self.owner.execute(row["id"])
        result = self.assess(row)
        self.assertEqual(result["observed_outcome"], "no")
        self.assertEqual(result["corrective_supervision"]["training_target"], self.rubric["training_target"])
        self.assertIn("west room", (Path(self.tmp.name)/receipt["path"]).read_text())

    def test_long_reasoning_trace_preserves_the_separate_final_artifact(self):
        row = self.record(generation={"text": "Meet in the east hall.", "token_ids": [8] * 900,
                          "finish_reason": "stop", "policy": "greedy-token-v1"})
        receipt = self.owner.execute(row["id"])
        self.assertEqual(len(Experience(Memory(self.tmp.name)).export(row["id"])["generation"]["token_ids"]), 900)
        self.assertEqual((Path(self.tmp.name) / receipt["path"]).read_text(), "Meet in the east hall. [notice@1]\n")

    def test_sampled_policy_survives_sqlite_recovery_with_its_actual_output(self):
        from nanoai.sampling import QWEN35_THINKING
        policy = {**QWEN35_THINKING, "seed": 73}
        row = self.record(generation={"text": "Meet in the east hall.", "token_ids": [31, 52, 4],
                                     "finish_reason": "stop", "policy": policy})
        receipt = self.owner.execute(row["id"]); self.assess(row)
        fresh = Experience(Memory(self.tmp.name))
        recovered = fresh.export(row["id"])
        self.assertEqual(recovered["generation"]["policy"], policy)
        self.assertEqual(recovered["generation"]["token_ids"], [31, 52, 4])
        self.assertEqual(fresh.execute(row["id"]), receipt)
        self.assertEqual(recovered["eligibility"]["status"], "checked")

    def test_truncated_answer_cannot_pass_by_containing_the_expected_fact(self):
        row = self.record("Meet in the east hall.", finish="length")
        self.owner.execute(row["id"])
        result = self.assess(row)
        self.assertFalse(result["checks"]["complete_generation"])
        self.assertEqual(result["observed_outcome"], "no")

    def test_empty_completed_generation_is_retained_as_a_failed_action(self):
        row = self.record("")
        self.owner.execute(row["id"])
        result = self.assess(row)
        self.assertTrue(result["checks"]["complete_generation"])
        self.assertEqual(result["observed_outcome"], "no")

    def test_source_and_prompt_must_match_before_recording(self):
        before = self.memory.sequence
        with self.assertRaises(ValueError):
            self.record(messages=[{"role": "user", "content": self.question}])
        with self.assertRaises(ValueError):
            self.record(observations=["An invented source"])
        self.assertEqual(self.memory.sequence, before)

    def test_context_checks_named_source_fields_and_rejects_hidden_or_outdated_inputs(self):
        payload = {"schema": "nanoai.task-context.v1", "task": {"goal": self.question},
                   "current_source_records": [{"text": self.observations[0]}]}
        def message(value):
            return {"role": "user", "content": json.dumps(value)}
        row = self.record(messages=[message(payload)])
        self.assertEqual(row["observations"], self.observations)
        before = self.memory.sequence
        for wrong in ({**payload, "task": {"goal": "A different request", "quote": self.question}},
                      {**payload, "current_source_records": [], "hidden": self.observations},
                      {**payload, "current_source_records": [{"text": "An unrelated observation"}], "hidden": self.observations}):
            with self.assertRaises(ValueError):
                self.record(messages=[message(payload), message(wrong)])
        self.assertEqual(self.memory.sequence, before)

    def test_incorrect_corrective_target_and_changed_artifact_are_rejected(self):
        row = self.record()
        receipt = self.owner.execute(row["id"])
        bad = {**self.rubric, "training_target": "Meet in the west room."}
        with self.assertRaises(ValueError):
            self.owner.assess_generation(row["id"], rubric=bad,
                                        evaluator_identity="notice-rubric-v1", evaluate=assess_notice)
        self.assess(row)
        (Path(self.tmp.name)/receipt["path"]).write_text("Altered work")
        self.assertEqual(self.owner.export(row["id"])["eligibility"]["status"], "invalidated")

    def test_source_correction_invalidates_current_training_eligibility(self):
        row = self.record(); self.owner.execute(row["id"]); self.assess(row)
        self.memory.put_source("notice", json.dumps({"observations": ["The tour was cancelled."]}), expected_revision=1)
        self.assertEqual(self.owner.export(row["id"])["eligibility"]["status"], "invalidated")

    def test_each_experience_keeps_one_granted_artifact_across_recovery(self):
        self.memory.set_task("tour", self.question, expected_revision=1,
                             filenames=["plan.md", "notice.md"])
        source = self.memory.state["sources"]["notice"]
        self.snapshot = self.memory.snapshot("tour", [source], "original-model")
        row = self.record()
        with self.assertRaises(ValueError):
            self.owner.execute(row["id"], filename="outside.md")
        receipt = self.owner.execute(row["id"], filename="notice.md")
        fresh = Experience(Memory(self.tmp.name))
        self.assertEqual(fresh.execute(row["id"], filename="notice.md"), receipt)
        with self.assertRaises(ValueError):
            fresh.execute(row["id"], filename="plan.md")
        self.assertEqual(self.memory.completed("tour")["remaining"], ["plan.md"])


if __name__ == "__main__":
    unittest.main()
