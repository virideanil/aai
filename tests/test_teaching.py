from copy import deepcopy
import json
from pathlib import Path
import tempfile
import sys
import unittest

from nanoai.agent import DEFAULT_IDENTITY
from nanoai.experience import Experience
from nanoai.memory import Memory, StaleState
from nanoai.teaching import Teaching, validate_lesson
from nanoai.writing import Writing

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from teaching_tasks import assess, evaluation_rows, quantity_answer
from nanoai.core import file_digest


class Core:
    identity = "teaching-boundary-fixture"

    def __init__(self):
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        self.messages = messages
        return {"output": "The brass key vanished because Ada left.", "token_ids": [1, 2], "finish_reason": "stop"}


class TeachingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.memory = Memory(self.temp.name)
        self.memory.initialize(DEFAULT_IDENTITY)
        self.memory.put_source("canon", "Ada left a brass key in the clock. Mira does not know why Ada left.")
        self.owner = Teaching(self.memory)
        self.lesson = json.loads((Path(__file__).parents[1] / "evals/teaching/lessons-v1.json").read_text())["lessons"][0]
        self.source = self.owner.install(self.lesson)
        self.core = Core()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        return Writing(self.memory).run("draft", "Write one sentence.", ["canon"], self.core,
            lesson_ids=[self.source["id"]])

    def feedback(self, result, **changes):
        values = dict(source_id="key-feedback", draft_quote="The brass key vanished",
            evidence=[{"source_id": "canon", "quote": "Ada left a brass key in the clock."}],
            diagnosis="The draft turns an unknown motive into an unsupported disappearance of the key.",
            repair="Keep the key and preserve uncertainty about Ada's motive.", reviewer="authored-case-v1",
            rule_ids=[self.source["id"]])
        values.update(changes)
        return self.owner.feedback(result["experience_id"], **values)

    def test_lessons_flow_to_actual_inputs_and_correction_invalidates_descendants(self):
        result = self.write()
        self.assertIn(self.lesson["principle"], self.core.messages[-1]["content"])
        experience = self.memory.state["experiences"][result["experience_id"]]
        self.assertEqual(experience["snapshot"]["sources"][self.source["id"]]["digest"], self.source["digest"])
        self.assertIn(experience["observations"][-1], self.core.messages[-1]["content"])
        self.assertEqual(self.owner.install(self.lesson), self.source)
        fresh = Writing(Memory(self.temp.name))
        again = fresh.run("draft", "Write one sentence.", ["canon"], self.core, lesson_ids=[self.source["id"]])
        self.assertEqual(again["receipt"], result["receipt"])
        self.assertEqual(self.core.calls, 1)
        revised = {**self.lesson, "scope": "A narrowed source-grounding lesson."}
        self.owner.install(revised, expected_revision=1)
        self.assertFalse(self.memory.source_current(result["source"]))

    def test_feedback_quotes_actual_work_and_evidence_and_recovers(self):
        result = self.write()
        feedback = self.feedback(result)
        self.assertEqual(json.loads(feedback["text"])["status"], "attributed-proposal")
        fresh = Teaching(Memory(self.temp.name))
        self.assertTrue(fresh.memory.source_current(feedback))
        self.assertEqual(self.feedback(result), feedback)
        self.memory.put_source("canon", "Ada did not leave a key.", expected_revision=1)
        self.assertFalse(self.memory.source_current(feedback))
        with self.assertRaises(StaleState):
            self.feedback(result, source_id="late-feedback")

    def test_invented_quote_and_unrelated_evidence_are_rejected_before_write(self):
        result = self.write()
        self.memory.put_source("other", "Ada left a brass key in the clock.")
        before = self.memory.sequence
        for changes in [
            {"draft_quote": "Invented draft"},
            {"evidence": [{"source_id": "canon", "quote": "Invented source"}]},
            {"evidence": [{"source_id": "other", "quote": "Ada left a brass key in the clock."}]}]:
            with self.assertRaises(ValueError):
                self.feedback(result, **changes)
        self.assertEqual(self.memory.sequence, before)

    def test_tampered_artifact_cannot_teach(self):
        result = self.write()
        Path(result["artifact_path"]).write_text("tampered")
        with self.assertRaises(ValueError):
            self.feedback(result)

    def test_practice_uses_observed_failure_and_preserves_success_without_repeating(self):
        class ImprovingCore(Core):
            def generate(self, messages, **kwargs):
                self.calls += 1
                actual_feedback = "The sentence has six words; use at most four."
                self.asserted_feedback = actual_feedback in messages[-1]["content"]
                return {"output": "Mira turned the key." if self.asserted_feedback else "Mira slowly turned the brass key.",
                        "token_ids": [1, 2], "finish_reason": "stop"}
        core = ImprovingCore()
        arguments = dict(criteria={"max": 4}, evaluate=lambda text, criteria: {"word_limit": len(text.split()) <= criteria["max"]},
            feedback=lambda text, checks, criteria: "The sentence has six words; use at most four.",
            evaluator_identity="word-count-fixture-v1", scope="Four whitespace-delimited words", max_attempts=2)
        writer = Writing(self.memory)
        result = writer.practice("practice", "Write at most four words.", ["canon"], core, **arguments)
        self.assertTrue(result["satisfied"])
        self.assertEqual(core.calls, 2)
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["selected"]["generation"]["text"], "Mira turned the key.")
        again = Writing(Memory(self.temp.name)).practice("practice", "Write at most four words.", ["canon"], core, **arguments)
        self.assertEqual(core.calls, 2)
        self.assertEqual(again["selected"]["receipt"], result["selected"]["receipt"])

    def test_unsatisfied_practice_has_no_selected_artifact(self):
        result = Writing(self.memory).practice("bounded", "Write a supported sentence.", ["canon"], self.core,
            criteria={}, evaluate=lambda text, _: {"supported": False}, feedback=lambda *args: "Use the supported source.",
            evaluator_identity="always-fail-fixture", scope="Loop termination fixture", max_attempts=2)
        self.assertFalse(result["satisfied"])
        self.assertIsNone(result["selected"])
        self.assertEqual(self.core.calls, 2)

    def test_feedback_order_survives_journal_recovery_without_new_generation(self):
        options = dict(criteria={}, evaluate=lambda *_: {"z_last": False, "a_first": False},
            feedback=lambda text, checks, criteria: "Failed: " + ", ".join(k for k, v in checks.items() if not v),
            evaluator_identity="ordered-feedback-fixture", scope="Feedback identity across recovery", max_attempts=2)
        first = Writing(self.memory).practice("ordered", "Write one sentence.", ["canon"], self.core, **options)
        self.assertEqual(self.core.calls, 2)
        repeated = Writing(Memory(self.temp.name)).practice("ordered", "Write one sentence.", ["canon"], self.core, **options)
        self.assertEqual(self.core.calls, 2)
        self.assertEqual([r["receipt"] for r in repeated["attempts"]], [r["receipt"] for r in first["attempts"]])
        self.assertTrue(all(r["model_calls"] == 0 for r in repeated["attempts"]))

    def test_only_checked_current_teaching_targets_export(self):
        owner = Experience(self.memory)
        source = self.memory.put_source("input", json.dumps({"observations": ["2 + 3"]}))
        self.memory.set_task("sum", "Return the sum.", filenames=["answer.txt"])
        snapshot = self.memory.snapshot("sum", [source], "fixture")
        for split in ("train", "confirmation"):
            row = owner.record_generation(snapshot, question="Return the sum.", observations=["2 + 3"],
                messages=[{"role": "user", "content": "Return the sum.\n2 + 3"}],
                generation={"text": "6", "token_ids": [6], "finish_reason": "stop", "policy": "greedy-token-v1"},
                representation_identity="fixture", source_id="input", metadata={"split": split})
            owner.execute(row["id"])
            owner.assess_generation(row["id"], rubric={"scope": "Exact integer sum", "criteria": {}, "training_target": "5"},
                evaluator_identity="integer-addition-v1", evaluate=lambda text, _: {"sum": text.strip() == str(2 + 3)})
            if split == "train":
                self.assertEqual(self.owner.training_pair(row["id"])["target"], "5")
                train_id = row["id"]
            else:
                with self.assertRaises(ValueError):
                    self.owner.training_pair(row["id"])
        self.memory.put_source("input", json.dumps({"observations": ["2 + 4"]}), expected_revision=1)
        with self.assertRaises(StaleState):
            self.owner.training_pair(train_id)

    def test_full_authored_bundle_has_bounded_complete_examples(self):
        bundle = json.loads((Path(__file__).parents[1] / "evals/teaching/lessons-v1.json").read_text())
        ids = set()
        for lesson in bundle["lessons"]:
            self.assertNotIn(lesson["id"], ids)
            ids.add(lesson["id"])
            self.assertEqual(validate_lesson(lesson), lesson)
        broken = deepcopy(self.lesson)
        broken["contrast"].pop("feedback")
        with self.assertRaises(ValueError):
            validate_lesson(broken)

    def test_independent_cases_require_frozen_identity_and_remain_confirmation(self):
        corpus = json.loads((Path(__file__).parents[1] / 'evals/teaching/tasks-v2.json').read_text())
        # This fixture is separate from the unopened independent review source.
        review = {'schema': corpus['schema'], 'rows': [deepcopy(corpus['rows'][-1])]}
        corpus['rows'] = corpus['rows'][:-1]
        review['rows'][0]['split'] = 'confirmation'
        path = Path(self.temp.name) / 'review.json'
        path.write_text(json.dumps(review))
        expected = {'digest': file_digest(path), 'rows': 1}
        rows = evaluation_rows(corpus, review_cases=path, expected_review=expected)
        self.assertEqual(rows[-1]['cohort'], 'independent')
        self.assertEqual(rows[-1]['split'], 'confirmation')
        self.assertTrue(all(r['split'] in {'confirmation', 'control'} for r in rows))
        with self.assertRaises(ValueError):
            evaluation_rows(corpus, review_cases=path)
        path.write_text(json.dumps({**review, 'changed': True}))
        with self.assertRaisesRegex(ValueError, 'source differs'):
            evaluation_rows(corpus, review_cases=path, expected_review=expected)
        review['rows'][0]['split'] = 'train'
        path.write_text(json.dumps(review))
        with self.assertRaisesRegex(ValueError, 'confirmation set'):
            evaluation_rows(corpus, review_cases=path, expected_review={'digest': file_digest(path), 'rows': 1})

    def test_independent_review_rejects_overlap_with_development(self):
        corpus = json.loads((Path(__file__).parents[1] / 'evals/teaching/tasks-v2.json').read_text())
        row = deepcopy(next(r for r in corpus['rows'] if r['split'] == 'confirmation'))
        path = Path(self.temp.name) / 'overlap.json'
        path.write_text(json.dumps({'schema': corpus['schema'], 'rows': [row]}))
        with self.assertRaisesRegex(ValueError, 'Duplicated task identity or complete input'):
            evaluation_rows(corpus, review_cases=path, expected_review={'digest': file_digest(path), 'rows': 1})

    def test_quantity_review_accepts_equivalent_ordinary_english(self):
        corpus = json.loads((Path(__file__).parents[1] / 'evals/teaching/tasks-v2.json').read_text())
        row = next(r for r in corpus['rows'] if r['id'] == 'train-quantity-1')
        answer = json.loads(row['target'])
        answer['text'] = 'The elapsed duration is twenty-four seconds.'
        self.assertTrue(all(assess(json.dumps(answer), row['criteria']).values()))
        answer['text'] = 'The elapsed duration is twenty-five seconds.'
        self.assertFalse(assess(json.dumps(answer), row['criteria'])['computed_quantity'])

    def test_quantity_mentions_reject_conflicts_wrong_units_and_negative_values(self):
        for text in ('51 seconds', 'fifty-one seconds', 'fifty one seconds', '51.0 seconds'):
            self.assertTrue(quantity_answer(text, 3 * 17, 'seconds'), text)
        for text in ('fifty-two seconds', '-51 seconds', '51 seconds and 17 seconds',
                     'fifty-one minutes', '151 seconds'):
            self.assertFalse(quantity_answer(text, 3 * 17, 'seconds'), text)


if __name__ == "__main__":
    unittest.main()
