import json
from pathlib import Path
import tempfile
import unittest

from nanoai.agent import DEFAULT_IDENTITY, calculate, context
from nanoai.memory import Memory, StaleState
from nanoai.steps import Steps, pack_rank_plane, unpack_rank_plane
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

    def test_rank_plane_2_trit_pack5_roundtrip(self):
        ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 42]
        packed = pack_rank_plane(ranks)
        self.assertEqual(len(packed), (2 * len(ranks) + 4) // 5)
        unpacked = unpack_rank_plane(packed, len(ranks))
        self.assertEqual(unpacked, [1, 2, 3, 4, 5, 6, 7, 8, 9, 9])
        with self.assertRaises(ValueError):
            pack_rank_plane([0])
        with self.assertRaises(ValueError):
            pack_rank_plane([True])

    def test_system_one_calibration_band_proposal_and_audit_correction(self):
        snap = self.snapshot()
        initial_cal = self.steps.calibration("display")
        self.assertEqual(initial_cal["operators"]["integer.product"]["band"], "propose")

        first = self.steps.calculate_fields(snap, self.plan)
        field_steps = list(first["step"]["arguments"]["steps"])
        self.steps.review(first["step_id"], "confirmed", reviewer="audit-1", reason="Verified product")
        second = self.steps.propose(snap, "integer.product", {"steps": [field_steps[0]]})
        self.steps.execute(second["id"])
        self.steps.review(second["id"], "confirmed", reviewer="audit-2", reason="Verified second product")

        # Default policy="bracket": 2 confirmations have lower_5pct ~ 0.368 < 0.8 -> band="propose"
        cal_default = self.steps.calibration("display")
        self.assertEqual(cal_default["policy"], "bracket")
        self.assertEqual(cal_default["operators"]["integer.product"]["band"], "propose")
        self.assertTrue(cal_default["operators"]["integer.product"]["configured_rule_fires"])
        self.assertFalse(cal_default["operators"]["integer.product"]["statistical_competence_established"])

        # Deliberate opt-in policy="configured_rule": fires act_now at 2 confirmations
        cal = self.steps.calibration("display", policy="configured_rule")
        self.assertEqual(cal["operators"]["integer.product"]["band"], "act_now")
        self.assertEqual(cal["operators"]["integer.product"]["empirical_precision"], 1.0)

        acted = self.steps.band_proposal(
            snap, "integer.product", {"steps": field_steps}, policy="configured_rule"
        )
        self.assertEqual(acted["band"], "act_now")
        self.assertTrue(acted["executed"])
        self.assertEqual(acted["observation"]["result"]["value"], 46)

        unreviewed = self.steps.propose(snap, "integer.product", {"steps": field_steps})
        corr = self.steps.issue_correction(
            unreviewed["id"],
            reviewer="master-reviewer",
            reason="Should use quantity.product with explicit units",
            conduct_virtue="epistemic_humility",
        )
        self.assertEqual(corr["conduct_virtue"], "epistemic_humility")
        self.assertEqual(self.memory.state["step_reviews"][unreviewed["id"]]["decision"], "rejected")
        after_corr = self.steps.calibration("display")
        self.assertEqual(after_corr["operators"]["integer.product"]["band"], "escalate")
        self.assertEqual(after_corr["operators"]["integer.product"]["conduct_flags"], 1)

    def test_hybrid_decision_integration_unfamiliar_tasks_and_core_judgment(self):
        from nanoai.judgments import Judgments
        snap = self.snapshot()
        first = self.steps.calculate_fields(snap, self.plan)
        field_steps = list(first["step"]["arguments"]["steps"])
        self.steps.review(first["step_id"], "confirmed", reviewer="audit-1", reason="Verified product")
        second = self.steps.propose(snap, "integer.product", {"steps": [field_steps[0]]})
        self.steps.execute(second["id"])
        self.steps.review(second["id"], "confirmed", reviewer="audit-2", reason="Verified second product")

        # 1. In a familiar task ("display"), 2 confirmations justify rule firing when context is aligned,
        #    EXCEPT when the learned core signals context-sensitive uncertainty ("maybe" or low usefulness).
        uncertain_act = self.steps.band_proposal(
            snap,
            "integer.product",
            {"steps": field_steps},
            core_signal={"verdict": "maybe", "usefulness": 0.45, "uncertainty": "missing_evidence"},
        )
        self.assertEqual(uncertain_act["band"], "propose")
        self.assertFalse(uncertain_act["executed"])
        self.assertEqual(uncertain_act["signals"]["calibration"]["trigger"], "learned_core_uncertainty_override")

        conflict_act = self.steps.band_proposal(
            snap,
            "integer.product",
            {"steps": field_steps},
            core_signal={"verdict": "no", "usefulness": 0.1, "uncertainty": "conflict"},
        )
        self.assertEqual(conflict_act["band"], "escalate")
        self.assertFalse(conflict_act["executed"])

        # 2. On an unfamiliar task ("unfamiliar_bridge"), two confirmations on "display" alone do NOT
        #    automatically fire act_now without context-sensitive core endorsement.
        self.memory.set_task("unfamiliar_bridge", "Compute bridge load on unfamiliar structure.", filenames=["bridge.md"])
        unfamiliar_snap = context(self.memory, "unfamiliar_bridge", "test-core", selected_ids=["spec"])["snapshot"]
        r1 = self.steps.propose(unfamiliar_snap, "source.read", {"source": "spec"})
        self.steps.execute(r1["id"])
        f1 = self.steps.propose(unfamiliar_snap, "json.field", {"step": r1["id"], "path": ["rail", "length_cm"]})
        self.steps.execute(f1["id"])
        f2 = self.steps.propose(unfamiliar_snap, "json.field", {"step": r1["id"], "path": ["rail", "count"]})
        self.steps.execute(f2["id"])

        unfamiliar_cal = self.steps.calibration("unfamiliar_bridge")
        self.assertFalse(unfamiliar_cal["operators"]["integer.product"]["task_familiar"])
        self.assertTrue(unfamiliar_cal["operators"]["integer.product"]["cross_task_transfer_eligible"])
        self.assertEqual(unfamiliar_cal["operators"]["integer.product"]["band"], "propose")

        without_core = self.steps.band_proposal(
            unfamiliar_snap, "integer.product", {"steps": [f1["id"], f2["id"]]}
        )
        self.assertEqual(without_core["band"], "propose")
        self.assertFalse(without_core["executed"])
        self.assertEqual(
            without_core["signals"]["calibration"]["trigger"],
            "unfamiliar_task_requires_core_or_review",
        )

        # When the learned core records a context-sensitive "yes" judgment on the unfamiliar task,
        # cross-task history + core endorsement combine to authorize act_now.
        j_node = Judgments(self.memory).predict(
            unfamiliar_snap,
            "Does integer.product apply to unfamiliar_bridge rail length and count?",
            "yes",
            "Same multiplicative schema applies to bridge segments",
        )
        with_core = self.steps.band_proposal(
            unfamiliar_snap,
            "integer.product",
            {"steps": [f1["id"], f2["id"]]},
            judgment_node_id=j_node["id"],
        )
        self.assertEqual(with_core["band"], "act_now")
        self.assertTrue(with_core["executed"])
        self.assertEqual(with_core["observation"]["result"]["value"], 46)
        self.assertEqual(
            with_core["signals"]["calibration"]["trigger"],
            "cross_task_history_plus_core_endorsement",
        )

        # 3. Verify decision_support_audit answers "Does this hybrid make well-supported decisions in practice?"
        #    and distinguishes caller-supplied core_signal from assessed_judgment / core_generated_plan.
        self.assertTrue(uncertain_act["signals"]["learned_core"]["caller_supplied"])
        self.assertEqual(uncertain_act["signals"]["learned_core"]["provenance"], "caller_supplied_estimate")
        self.assertFalse(with_core["signals"]["learned_core"]["caller_supplied"])
        self.assertEqual(with_core["signals"]["learned_core"]["provenance"], "provisional_judgment")

        self.steps.review(with_core["step"]["id"], "confirmed", reviewer="audit-3", reason="Valid transfer")
        support = self.steps.decision_support_audit()
        self.assertEqual(support["question"], "Does this hybrid make well-supported decisions in practice?")
        self.assertEqual(support["verdict"], "well_supported")
        self.assertEqual(support["well_supported_rate"], 1.0)
        self.assertEqual(support["unfamiliar_task_decisions"], 3)
        self.assertEqual(support["unfamiliar_task_precision"], 1.0)
        self.assertGreaterEqual(support["core_generated_or_assessed_decisions"], 1)
        self.assertGreaterEqual(support["caller_supplied_signal_decisions"], 1)
        self.assertIn("evidence_boundary", support)

    def test_calibrated_inquire_producer_path_and_empty_final_guard(self):
        from nanoai.agent import ProposalError, inquire, run
        from nanoai.relations import RELATION

        # C1: Empty core_signal={} never fabricates endorsement ('maybe', usefulness=None, supplied=False)
        snap0 = self.snapshot()
        empty_sig = self.steps._resolve_core_signal(snap0, core_signal={})
        self.assertFalse(empty_sig["supplied"])
        self.assertEqual(empty_sig["verdict"], "maybe")
        self.assertIsNone(empty_sig["usefulness"])

        partial_sig = self.steps._resolve_core_signal(snap0, core_signal={"uncertainty": "none"})
        self.assertTrue(partial_sig["supplied"])
        self.assertEqual(partial_sig["verdict"], "maybe")
        self.assertIsNone(partial_sig["usefulness"])

        # C2: Verify Beta(3, 1) 5%-95% credible bracket and policy='bracket'
        self.assertAlmostEqual(self.steps.beta_quantile(3, 1, 0.05), 0.05 ** (1.0 / 3.0), places=6)
        self.assertAlmostEqual(self.steps.beta_quantile(3, 1, 0.95), 0.95 ** (1.0 / 3.0), places=6)

        # Set up relational source and 2 confirmed relation.query steps on task "display"
        rel_model = {"type": RELATION, "variables": ["x", "y"], "worlds": [[0, 0], [1, 1]], "scope": "Rail switch"}
        self.memory.put_source("rel_src", json.dumps(rel_model))
        snap = context(self.memory, "display", "test-core", selected_ids=["spec", "rel_src"])["snapshot"]
        r_step = self.steps.execute(self.steps.propose(snap, "source.read", {"source": "rel_src"})["id"])
        d_step = self.steps.execute(self.steps.propose(snap, "relation.decode", {"step": r_step["step_id"]})["id"])
        q1 = self.steps.execute(self.steps.propose(snap, "relation.query", {"step": d_step["step_id"], "expression": {"eq": ["x", "y"]}})["id"])
        self.steps.review(q1["step_id"], "confirmed", reviewer="audit-1", reason="Confirmed eq query")
        q2 = self.steps.execute(self.steps.propose(snap, "relation.query", {"step": d_step["step_id"], "expression": "x"})["id"])
        self.steps.review(q2["step_id"], "confirmed", reviewer="audit-2", reason="Confirmed x query")

        cal_rule = self.steps.band_proposal(
            snap, "relation.query", {"step": d_step["step_id"], "expression": "y"}, execute_act_now=False
        )
        self.assertEqual(cal_rule["signals"]["review_history"]["bracket_trit"], "both")
        self.assertTrue(cal_rule["signals"]["review_history"]["configured_rule_fires"])
        self.assertFalse(cal_rule["signals"]["review_history"]["statistical_competence_established"])

        cal_bracket = self.steps.band_proposal(
            snap,
            "relation.query",
            {"step": d_step["step_id"], "expression": "y"},
            policy="bracket",
            execute_act_now=False,
        )
        self.assertEqual(cal_bracket["band"], "propose")
        self.assertEqual(cal_bracket["signals"]["calibration"]["policy"], "bracket")

        class SequenceCore:
            identity = "test-core"
            def __init__(self, responses):
                self.responses = list(responses)
            def generate(self, messages, max_tokens=None, schema=None):
                item = self.responses.pop(0)
                if isinstance(item, dict) and "output" in item:
                    return item
                return {"output": json.dumps(item), "elapsed_ms": 1, "tokens": 6, "finish_reason": "stop"}

        # 1. Familiar task ("display"), core-generated plan with uncertainty ("maybe") -> calibrated=True holds in "propose"
        uncertain_inq = inquire(
            self.memory,
            SequenceCore([{"operator": "relation.query", "expression": "y", "verdict": "maybe", "confidence": 0.4, "uncertainty": "ambiguous_branch"}]),
            "display",
            selected_ids=["spec", "rel_src"],
            state_step_id=d_step["step_id"],
            calibrated=True,
        )
        self.assertEqual(uncertain_inq["band"], "propose")
        self.assertFalse(uncertain_inq["executed"])
        self.assertIsNone(uncertain_inq["result"])
        self.assertEqual(uncertain_inq["calibration"]["signals"]["learned_core"]["provenance"], "core_generated_plan")
        self.assertFalse(uncertain_inq["calibration"]["signals"]["learned_core"]["caller_supplied"])

        # 2. Unfamiliar task ("novel_rail"), core-generated plan with high confidence ("yes", 0.93) -> cross-task transfer + core endorsement fires act_now
        self.memory.set_task("novel_rail", "Inspect novel rail switch.", filenames=["novel.md"])
        novel_snap = context(self.memory, "novel_rail", "test-core", selected_ids=["spec", "rel_src"])["snapshot"]
        nr_step = self.steps.execute(self.steps.propose(novel_snap, "source.read", {"source": "rel_src"})["id"])
        nd_step = self.steps.execute(self.steps.propose(novel_snap, "relation.decode", {"step": nr_step["step_id"]})["id"])

        endorsed_inq = inquire(
            self.memory,
            SequenceCore([{"operator": "relation.query", "expression": {"eq": ["x", "y"]}, "verdict": "yes", "confidence": 0.93, "uncertainty": "none"}]),
            "novel_rail",
            selected_ids=["spec", "rel_src"],
            state_step_id=nd_step["step_id"],
            calibrated=True,
        )
        self.assertEqual(endorsed_inq["band"], "act_now")
        self.assertTrue(endorsed_inq["executed"])
        self.assertEqual(endorsed_inq["result"]["value"]["verdict"], "yes")
        self.assertEqual(
            endorsed_inq["calibration"]["signals"]["calibration"]["trigger"],
            "cross_task_history_plus_core_endorsement",
        )
        self.assertEqual(endorsed_inq["calibration"]["signals"]["learned_core"]["provenance"], "core_generated_plan")

        # 3. Post-correction: issuing a conduct correction on relation.query demotes subsequent calibrated inquire to escalate
        self.steps.issue_correction(
            endorsed_inq["step_id"],
            reviewer="safety-auditor",
            reason="Novel rail switch requires manual sign-off before querying",
            conduct_virtue="epistemic_humility",
        )
        after_corr_inq = inquire(
            self.memory,
            SequenceCore([{"operator": "relation.query", "expression": "x", "verdict": "yes", "confidence": 0.95}]),
            "novel_rail",
            selected_ids=["spec", "rel_src"],
            state_step_id=nd_step["step_id"],
            calibrated=True,
        )
        self.assertEqual(after_corr_inq["band"], "escalate")
        self.assertFalse(after_corr_inq["executed"])
        self.assertIsNone(after_corr_inq["result"])

        # 4. Empty-final generation guard in agent.run rejects <think>-only output and retries to completion
        run_core = SequenceCore([
            {"filename": "display.md", "content": "<think>Only thinking, no final body</think>   "},
            {"filename": "display.md", "content": "<think>Verified spec@1</think>Two rails total 46 cm [spec@1]."},
        ])
        run_out = run(self.memory, run_core, "display", selected_ids=["spec"], max_attempts=2)
        self.assertEqual(run_out["proposal"]["output"]["content"], "Two rails total 46 cm [spec@1].")
        self.assertEqual(run_out["proposal"]["metrics"]["thinking_chars"], len("Verified spec@1"))
        self.assertEqual(len(run_out["proposal"]["metrics"]["attempts"]), 2)
        self.assertTrue(any("empty final document content" in r["reason"] for r in self.memory.state["rejections"]))

    def test_exact_rational_bracket_and_four_admission_conditions(self):
        from fractions import Fraction
        from nanoai.steps import band_bracket, mass_at_or_below

        t, a = Fraction(4, 5), Fraction(1, 20)
        for c in (0, 2, 12, 13, 1000):
            self.assertEqual(mass_at_or_below(c, 0, t), t ** (c + 1))
        self.assertEqual([band_bracket(c, 0, t, a) for c in (2, 12, 13)], ["both", "both", "yes"])
        self.assertEqual(band_bracket(8, 2, t, a), "both")
        self.assertEqual(band_bracket(1, 9, t, a), "no")
        self.assertEqual(band_bracket(1000, 1000, t, a), "no")
        self.assertEqual(band_bracket(1000, 150, t, a), "yes")
        for c, r in ((0, 0), (3, 5), (40, 7), (1000, 1000)):
            self.assertEqual(
                mass_at_or_below(c, r, Fraction(1, 3)) + mass_at_or_below(r, c, Fraction(2, 3)),
                Fraction(1, 1),
            )
        # Equality convention mass <= alpha: exact boundary hits 'yes' and 'no'
        self.assertEqual(band_bracket(3, 0, threshold=t, alpha=t ** 4), "yes")
        self.assertEqual(band_bracket(0, 3, threshold=Fraction(1, 5), alpha=t ** 4), "no")
        # 14/0 against 0.819 (819/1000): (819/1000)^15 > 1/20 -> stays 'both' ('propose'), not rounded to 0.819
        self.assertEqual(band_bracket(14, 0, threshold=0.819, alpha=0.05), "both")

        # Verify all 4 admission conditions (a)-(d) and persisted decision_record
        snap = self.snapshot()
        first = self.steps.calculate_fields(snap, self.plan)
        field_steps = list(first["step"]["arguments"]["steps"])
        self.steps.review(first["step_id"], "confirmed", reviewer="audit-1", reason="First distinct step")

        # Condition (b): repeating identical step signature does not inflate confirmed
        dup = self.steps.propose(snap, "integer.product", {"steps": field_steps})
        self.steps.execute(dup["id"])
        self.steps.review(dup["id"], "confirmed", reviewer="audit-2", reason="Duplicate step signature")

        # Condition (c): self-review by the same core that proposed the step is excluded
        self_rev = self.steps.propose(snap, "integer.product", {"steps": [field_steps[0]]})
        self.steps.execute(self_rev["id"])
        self.steps.review(self_rev["id"], "confirmed", reviewer="test-core", reason="Self review")

        # Condition (a): review under a different core version is excluded when active core changes
        snap_old_core = context(self.memory, "display", "old-core-v0", selected_ids=["spec"])["snapshot"]
        old_core_step = self.steps.propose(snap_old_core, "integer.product", {"steps": [field_steps[1]]})
        self.steps.execute(old_core_step["id"])
        self.steps.review(old_core_step["id"], "confirmed", reviewer="audit-3", reason="Old core version")

        banded = self.steps.band_proposal(
            snap,
            "integer.product",
            {"steps": [field_steps[0], field_steps[0]]},
            threshold=0.8,
            execute_act_now=False,
        )
        rh = banded["signals"]["review_history"]
        self.assertEqual(rh["in_task_confirmed"], 1)
        self.assertEqual(rh["point_agreement_rate"], 1.0)
        self.assertEqual(rh["threshold_rational"], "4/5")
        self.assertEqual(rh["alpha_rational"], "1/20")
        self.assertEqual(rh["equality_convention"], "mass <= alpha")
        self.assertEqual(rh["posterior_mass_at_or_below"], "16/25")
        conds = rh["admission_conditions"]
        self.assertEqual(conds["distinct_step_evidence"]["excluded_duplicate_steps"], 1)
        self.assertEqual(conds["reviewer_independence"]["excluded_self_reviews"], 1)
        self.assertEqual(conds["exchangeability_scope"]["excluded_core_version_changes"], 1)

        # Fresh process reload preserves persisted decision_record in journaled step.proposed
        fresh_steps = Steps(Memory(self.root))
        persisted = fresh_steps.memory.state["steps"][banded["step"]["id"]]["decision_record"]
        self.assertEqual(persisted["threshold_rational"], "4/5")
        self.assertEqual(persisted["alpha_rational"], "1/20")
        self.assertEqual(persisted["equality_convention"], "mass <= alpha")
        self.assertEqual(persisted["posterior_mass_at_or_below"], "16/25")
        self.assertEqual(persisted["confirmed"], 1)


if __name__ == "__main__":
    unittest.main()



