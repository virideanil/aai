import copy
import json
import tempfile
import unittest

from nanoai.agent import DEFAULT_IDENTITY, context
from nanoai.harmonizer import Harmonizer
from nanoai.memory import Memory, StaleState
from nanoai.steps import Steps
from nanoai.scratchboard import Scratchboard
from nanoai.writing import Writing


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

    def test_working_units_share_dictionary_model_context_and_revision_dependencies(self):
        board = Scratchboard(self.memory)
        unit = board.add(self.snapshot, {"kind":"literal", "text":"Inspection code: PERA-73 [survey@1]"})
        prepared = context(self.memory, "report", "test-core", selected_ids=["survey", "opinion"], scratchboard_ids=[unit["id"]])
        packet = Harmonizer(self.memory).view(prepared["snapshot"])
        entry = next(row for row in packet["entries"] if row["kind"] == "working_unit")
        payload = json.loads(prepared["messages"][1]["content"])
        self.assertEqual(payload["working_units"], [{k: v for k, v in entry.items() if k != "digest"}])
        self.assertEqual(prepared["snapshot"]["scratchboard"][unit["id"]], unit["digest"])
        self.assertTrue(entry["preview_complete"])
        self.assertIn("PERA-73", entry["preview"])
        self.assertTrue(any(r["kind"] == "graph_rule" and r["id"] == "scratchboard.import" for r in packet["entries"]))
        proposal = self.memory.record_proposal(prepared["snapshot"], {"filename":"result.md", "content":entry["preview"], "citations":["survey@1"]})
        board.add(self.snapshot, {"kind":"literal", "text":"Independent candidate"})
        self.assertEqual(Harmonizer(self.memory).validate(packet)["identity"], packet["identity"])
        board.revise(unit["id"], {"kind":"literal", "text":"Inspection code: PERA-74 [survey@1]"}, reason="Adopt corrected inspection code")
        with self.assertRaises(StaleState): Harmonizer(self.memory).validate(packet)
        with self.assertRaises(StaleState): self.memory.commit(proposal["id"], core_identity="test-core")

    def test_dictionary_keeps_unresolved_and_truncated_views_explicit(self):
        board = Scratchboard(self.memory)
        slots = board.reserve(self.snapshot, 10_000_000, constraints="Future work remains unresolved")
        literal = board.add(self.snapshot, {"kind":"literal", "text":"Ş"*1500})
        prepared = context(self.memory, "report", "test-core", selected_ids=["survey", "opinion"], scratchboard_ids=[slots["id"],literal["id"]])
        units = json.loads(prepared["messages"][1]["content"])["working_units"]
        self.assertEqual(units[0]["unresolved_slots"], 10_000_000)
        self.assertIsNone(units[0]["preview"])
        self.assertFalse(units[0]["preview_complete"])
        self.assertFalse(units[1]["representation_complete"])
        self.assertNotIn("text", units[1]["representation"])
        self.assertEqual(units[1]["representation"]["text_prefix"], "Ş"*512)
        other = context(self.memory, "report", "another-core", selected_ids=["survey", "opinion"], scratchboard_ids=[literal["id"]])
        self.assertEqual(json.loads(other["messages"][1]["content"])["working_units"][0]["representation"],units[1]["representation"])
        tensor = board.tensor(self.snapshot,b"\x00\x00",shape=[1])
        with self.assertRaises(StaleState):
            context(self.memory, "report", "another-core", selected_ids=["survey", "opinion"], scratchboard_ids=[tensor["id"]])

    def test_authoring_and_writing_use_the_same_current_evidence_envelope(self):
        class Capture:
            identity = "test-core"
            def generate(self, messages, **kwargs):
                self.messages = copy.deepcopy(messages)
                return {"output": "The total length is 51 cm.", "token_ids": [1], "finish_reason": "stop"}
        core = Capture()
        Writing(self.memory).run("report", self.memory.state["tasks"]["report"]["goal"],
                                 ["survey", "opinion"], core)
        written = json.loads(core.messages[1]["content"])
        authored = context(self.memory, "report", core.identity, selected_ids=["survey", "opinion"])
        payload = json.loads(authored["messages"][1]["content"])
        for key in ("schema", "identity", "task", "current_source_records", "allowed_citations"):
            self.assertEqual(written[key], payload[key])
        self.assertEqual(written["schema"], "nanoai.task-context.v1")
        self.assertEqual([row["text"] for row in written["current_source_records"]],
                         [self.memory.state["sources"][name]["text"] for name in ("survey", "opinion")])
        self.assertNotIn("Ground criticism", core.messages[0]["content"])
        self.assertNotIn("Selected lessons", core.messages[0]["content"])
        self.assertNotIn("fiction", core.messages[0]["content"].casefold())


if __name__ == "__main__":
    unittest.main()
