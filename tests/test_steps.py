import json
from pathlib import Path
import tempfile
import unittest

from nanoai.agent import DEFAULT_IDENTITY, calculate, context
from nanoai.memory import Memory, StaleState
from nanoai.steps import Steps
from nanoai.documents import quantity_note


class StepConsumerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.memory = Memory(self.root)
        self.memory.initialize(DEFAULT_IDENTITY)
        self.memory.put_source("spec", '{"rail":{"length_cm":23,"count":2}}')
        self.memory.set_task("display", "Calculate two rails and write a note.", filenames=["display.md"])
        self.steps = Steps(self.memory)
        self.plan = {"operator": "integer.product", "inputs": [
            {"source": "spec", "path": ["rail", "length_cm"]},
            {"source": "spec", "path": ["rail", "count"]}]}

    def tearDown(self):
        self.tmp.cleanup()

    def snapshot(self):
        return context(self.memory, "display", "test-core", selected_ids=["spec"])["snapshot"]

    def test_dependency_graph_reuse_correction_and_restart(self):
        first = self.steps.calculate_fields(self.snapshot(), self.plan)
        self.assertEqual(first["result"]["value"], 46)
        self.assertFalse(any(x["cache_hit"] for x in first["observations"]))
        self.memory.put_source("other", "Unrelated observation")
        second = Steps(Memory(self.root)).calculate_fields(self.snapshot(), self.plan)
        self.assertEqual(second["result"]["value"], 46)
        self.assertTrue(all(x["cache_hit"] for x in second["observations"]))
        self.memory.put_source("spec", '{"rail":{"length_cm":19,"count":2}}', expected_revision=1)
        with self.assertRaisesRegex(StaleState, "stale"):
            self.steps.execute(first["step_id"])
        current = self.steps.calculate_fields(self.snapshot(), self.plan)
        self.assertEqual(current["result"]["value"], 38)
        self.assertFalse(any(x["cache_hit"] for x in current["observations"]))
        self.assertEqual(current["result"]["sources"], ["spec@2"])

    def test_calculation_bound_document_and_zero_model_resume(self):
        calculation = self.steps.calculate_fields(self.snapshot(), self.plan)
        snap = self.memory.attach_observations(self.snapshot(), [calculation["step_id"]])
        proposal = self.memory.record_proposal(snap, {"filename": "display.md", "content": "Total 46 cm [spec@1]", "citations": ["spec@1"]})
        receipt = self.memory.commit(proposal["id"], core_identity="test-core")
        target = self.root / receipt["path"]
        before = target.stat().st_mtime_ns
        resumed = Memory(self.root).completed("display")
        self.assertEqual(resumed["remaining"], [])
        self.assertEqual(resumed["model_calls"], 0)
        self.assertEqual(target.stat().st_mtime_ns, before)
        self.steps.review(calculation["step_id"], "rejected", reviewer="test-assessor", reason="Task calls for addition instead")
        self.assertEqual(self.memory.completed("display")["remaining"], ["display.md"])
        self.assertEqual(target.read_text(), "Total 46 cm [spec@1]")

    def test_source_scope_duplicate_keys_types_and_operator_admission(self):
        snap = self.snapshot()
        with self.assertRaisesRegex(ValueError, "outside"):
            self.steps.propose(snap, "source.read", {"source": "foreign"})
        with self.assertRaisesRegex(ValueError, "executor"):
            self.steps.propose(snap, "shell.run", {"command": "anything"})
        self.memory.put_source("spec", '{"rail":{"length_cm":23,"length_cm":19,"count":2}}', expected_revision=1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.steps.calculate_fields(self.snapshot(), self.plan)
        self.memory.put_source("spec", '{"rail":{"length_cm":true,"count":2}}', expected_revision=2)
        with self.assertRaisesRegex(ValueError, "Integer input"):
            self.steps.calculate_fields(self.snapshot(), self.plan)

    def test_pending_and_cancelled_steps_do_not_execute_on_recovery(self):
        step = self.steps.propose(self.snapshot(), "source.read", {"source": "spec"})
        restored = Memory(self.root)
        sequence = restored.sequence
        self.assertEqual(restored.recover(), [])
        self.assertEqual(restored.sequence, sequence)
        self.assertEqual(restored.state["step_observations"], {})
        Steps(restored).review(step["id"], "cancelled", reviewer="test-controller", reason="Request cancelled")
        with self.assertRaisesRegex(StaleState, "cancelled"):
            Steps(restored).execute(step["id"])

    def test_changed_task_and_foreign_observation_cannot_supply_a_document(self):
        calculated = self.steps.calculate_fields(self.snapshot(), self.plan)
        self.memory.set_task("other", "Independent task")
        foreign = self.memory.snapshot("other", [self.memory.state["sources"]["spec"]], "test-core")
        with self.assertRaisesRegex(StaleState, "different task"):
            self.memory.attach_observations(foreign, [calculated["step_id"]])
        self.memory.set_task("display", "Change the plan", expected_revision=1, filenames=["display.md"])
        with self.assertRaisesRegex(StaleState, "stale"):
            self.steps.execute(calculated["step_id"])

    def test_dimension_error_is_observed_before_a_fact_can_be_rendered(self):
        self.memory.put_source("spec", '{"rail":{"length":{"value":23,"unit":"cm"},"count":{"value":2,"unit":"count"}}}', expected_revision=1)
        plan = {"operator": "quantity.sum", "inputs": [
            {"source": "spec", "path": ["rail", "length"]},
            {"source": "spec", "path": ["rail", "count"]}]}
        with self.assertRaisesRegex(ValueError, "Addition requires equal units"):
            self.steps.calculate_fields(self.snapshot(), plan)
        self.assertTrue(any(row["state"] == "rejected" for row in self.steps.status()["steps"]))
        plan["operator"] = "quantity.product"
        result = self.steps.calculate_fields(self.snapshot(), plan)
        self.assertEqual(result["result"]["value"], 46)
        self.assertEqual(result["result"]["dimensions"], {"cm": 1})
        with self.assertRaisesRegex(ValueError, "review"):
            quantity_note(self.memory, result["step_id"])
        self.steps.review(result["step_id"], "confirmed", reviewer="test-task-predicate", reason="Two equal rails require length times count")
        proposal = quantity_note(self.memory, result["step_id"], language="tr")
        self.assertIn("23 cm × 2 = 46 cm", proposal["output"]["content"])
        self.assertIn("[spec@2]", proposal["output"]["content"])
        receipt = self.memory.commit(proposal["id"], core_identity="test-core")
        self.assertEqual((self.root / receipt["path"]).read_text(), proposal["output"]["content"])
        self.memory.put_source("spec", '{"rail":{"length":{"value":19,"unit":"cm"},"count":{"value":2,"unit":"count"}}}', expected_revision=2)
        with self.assertRaises(StaleState):
            quantity_note(self.memory, result["step_id"])
        self.assertEqual(self.memory.completed("display")["remaining"], ["display.md"])

    def test_quantity_contract_rejects_unknown_units_and_false_counts(self):
        plan = {"operator": "quantity.product", "inputs": [
            {"source": "spec", "path": ["size"]}]}
        for revision, value in enumerate([{"value": 12, "unit": "guess"}, {"value": True, "unit": "count"}], 1):
            self.memory.put_source("spec", json.dumps({"size": value}), expected_revision=revision)
            with self.assertRaisesRegex(ValueError, "quantity field"):
                self.steps.calculate_fields(self.snapshot(), plan)

    def test_calculation_ports_keep_the_actual_task_in_the_model_input(self):
        class Capture:
            identity = "test-core"
            def generate(core, messages, **kwargs):
                core.messages = messages
                payload = json.loads(messages[1]["content"])
                output = ({"operator": "multiply", "fields": [row["ref"] for row in payload["quantities"]]}
                          if "quantities" in payload else self.plan)
                return {"output": json.dumps(output), "finish_reason": "stop"}
        for domain in ("integer", "quantity"):
            with self.subTest(domain=domain):
                if domain == "quantity":
                    self.memory.put_source("spec", '{"rail":{"length":{"value":23,"unit":"cm"},"count":{"value":2,"unit":"count"}}}', expected_revision=1)
                core = Capture()
                result = calculate(self.memory, core, "display", selected_ids=["spec"], domain=domain)
                self.assertEqual(result["result"]["value"], 46)
                self.assertEqual(json.loads(core.messages[1]["content"])["task"]["goal"], self.memory.state["tasks"]["display"]["goal"])
                self.assertNotIn('"output": "document"', core.messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
