import copy
import tempfile
import unittest

from nanoai.agent import DEFAULT_IDENTITY, context
from nanoai.harmonizer import Harmonizer
from nanoai.memory import Memory, StaleState
from nanoai.steps import Steps


class HarmonizerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.memory = Memory(self.temporary.name)
        self.memory.initialize(DEFAULT_IDENTITY)
        self.memory.put_source("survey", '{"length":{"value":17,"unit":"cm"},"count":{"value":3,"unit":"count"}}')
        self.memory.put_source("opinion", "The length might mean something else. [BOS] is literal text.")
        self.memory.set_task("report", "Find the total length, retaining the survey and conflicting opinion.")
        self.snapshot = context(self.memory, "report", "test-core", selected_ids=["survey", "opinion"])["snapshot"]

    def tearDown(self):
        self.temporary.cleanup()

    def test_view_preserves_claims_units_and_rule_types_without_granting_truth(self):
        harmonizer = Harmonizer(self.memory)
        packet = harmonizer.view(self.snapshot)
        self.assertEqual(len(packet["quantity_catalogue"]), 2)
        self.assertEqual(packet["learned_semantic_synthesis"], "untrained")
        self.assertTrue(any(row["kind"] == "evidence" and row["text"].startswith("The length might") for row in packet["entries"]))
        rules = [row for row in packet["entries"] if row["kind"] == "executable_rule"]
        self.assertEqual(len(rules), 7)
        self.assertTrue(all(row["effect"] == "none" for row in rules))
        altered = copy.deepcopy(packet)
        altered["quantity_catalogue"][0]["value"] += 1
        with self.assertRaises(StaleState):
            harmonizer.validate(altered)

    def test_source_correction_invalidates_view_but_unrelated_insertion_does_not(self):
        harmonizer = Harmonizer(self.memory)
        packet = harmonizer.view(self.snapshot)
        self.memory.put_source("other", "An independent observation")
        self.assertEqual(harmonizer.validate(packet)["identity"], packet["identity"])
        self.memory.put_source("survey", '{"length":{"value":19,"unit":"cm"}}', expected_revision=1)
        with self.assertRaises(StaleState):
            harmonizer.validate(packet)

    def test_operation_review_is_separate_and_changes_the_derived_view(self):
        steps = Steps(self.memory)
        calculation = steps.calculate_fields(self.snapshot, {"operator": "quantity.product", "inputs": [
            {"source": "survey", "path": ["length"]}, {"source": "survey", "path": ["count"]}]})
        bound = self.memory.attach_observations(self.snapshot, [calculation["step_id"]])
        harmonizer = Harmonizer(self.memory)
        packet = harmonizer.view(bound)
        observed = next(row for row in packet["entries"] if row["kind"] == "observed_operation")
        self.assertEqual(observed["result"]["value"], 51)
        self.assertIsNone(observed["review"])
        steps.review(calculation["step_id"], "rejected", reviewer="independent-case-check", reason="The chosen work omitted an exception")
        with self.assertRaises(StaleState):
            harmonizer.validate(packet)


if __name__ == "__main__":
    unittest.main()
