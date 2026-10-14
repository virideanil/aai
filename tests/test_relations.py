from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from nanoai.agent import DEFAULT_IDENTITY, context
from nanoai.judgments import Judgments, pack, unpack
from nanoai.memory import Memory, StaleState
from nanoai.relations import RELATION, expression_variables, validate_model
from nanoai.steps import Steps


class RelationConsumers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.memory = Memory(self.root)
        self.memory.initialize(DEFAULT_IDENTITY)
        self.memory.set_task("work", "Explore the shared relation and retain its evidence.")
        self.steps = Steps(self.memory)
        self.source = "relation"
        self.model = {"type": RELATION, "variables": ["x", "y"],
                      "worlds": [[0, 0], [1, 1]], "scope": "Declared linked switches"}
        self.memory.put_source(self.source, json.dumps(self.model))

    def tearDown(self):
        self.tmp.cleanup()

    def snapshot(self):
        return context(self.memory, "work", "test-core", selected_ids=[self.source])["snapshot"]

    def perform(self, operator, **args):
        step = self.steps.propose(self.snapshot(), operator, args)
        return self.steps.execute(step["id"])

    def decode(self):
        raw = self.perform("source.read", source=self.source)
        return self.perform("relation.decode", step=raw["step_id"])

    def test_equal_marginals_preserve_three_distinct_joint_answers(self):
        for revision, (worlds, expected) in enumerate([
                ([[0, 0], [1, 1]], "yes"),
                ([[0, 1], [1, 0]], "no"),
                ([[0, 0], [0, 1], [1, 0], [1, 1]], "maybe")]):
            if revision:
                model = {**self.model, "worlds": worlds}
                self.memory.put_source(self.source, json.dumps(model), expected_revision=revision)
            state = self.decode()
            for variable in ("x", "y"):
                answer = self.perform("relation.query", step=state["step_id"], expression=variable)
                self.assertEqual(answer["result"]["value"]["possible_values"], [0, 1])
                self.assertEqual(set(answer["result"]["value"]["witnesses"]), {"0", "1"})
            answer = self.perform("relation.query", step=state["step_id"], expression={"eq": ["x", "y"]})
            self.assertEqual(answer["result"]["value"]["verdict"], expected)
            self.assertEqual(unpack(pack([expected]), 1), [expected])

    def test_shared_identity_and_cache_parameters(self):
        state = self.decode()
        a = self.perform("relation.query", step=state["step_id"], expression={"xor": ["x", "x"]})
        b = self.perform("relation.query", step=state["step_id"], expression={"eq": ["x", "x"]})
        self.assertEqual(a["result"]["value"]["verdict"], "no")
        self.assertEqual(b["result"]["value"]["verdict"], "yes")
        self.assertNotEqual(a["cache_key"], b["cache_key"])
        repeated = self.perform("relation.query", step=state["step_id"], expression={"eq": ["x", "x"]})
        self.assertTrue(repeated["cache_hit"])
        self.assertEqual(repeated["reused_from"], b["step_id"])

    def test_refinement_extends_and_preserves_ancestry_through_restart(self):
        state = self.decode()
        chosen = self.perform("relation.refine", step=state["step_id"], expression="x", expected=1)
        contrary = self.perform("relation.refine", step=state["step_id"], expression="x", expected=0)
        self.assertEqual(chosen["result"]["value"]["worlds"], [[1, 1]])
        self.assertEqual(contrary["result"]["value"]["worlds"], [[0, 0]])
        self.assertNotEqual(chosen["cache_key"], contrary["cache_key"])
        expanded = self.perform("relation.extend", step=chosen["step_id"], variable="z", expression=None)
        self.assertEqual(expanded["result"]["value"]["worlds"], [[1, 1, 0], [1, 1, 1]])
        new_question = self.perform("relation.query", step=expanded["step_id"], expression="z")
        self.assertEqual(new_question["result"]["value"]["state"], "both")
        restored = Memory(self.root)
        sequence = restored.sequence
        restored.recover()
        again = Steps(restored).execute(new_question["step_id"])
        self.assertTrue(again["reobserved"])
        self.assertEqual(again["digest"], new_question["digest"])
        self.assertEqual(restored.sequence, sequence)
        self.assertEqual(restored.state["step_observations"][state["step_id"]]["result"]["value"], self.model)

    def test_derived_extension_and_hypothetical_action_keep_distinct_meanings(self):
        state = self.decode()
        derived = self.perform("relation.extend", step=state["step_id"], variable="z", expression={"not": "x"})
        self.assertEqual(derived["result"]["value"]["worlds"], [[0, 0, 1], [1, 1, 0]])
        mapped = self.perform("relation.map", step=state["step_id"], variable="y", expression={"not": "y"})
        self.assertEqual(mapped["result"]["value"]["worlds"], [[0, 1], [1, 0]])
        self.assertEqual(mapped["effect"], "none")
        self.assertIn("Hypothetical", mapped["result"]["scope"])

    def test_conflict_is_separate_from_an_open_or_unexecuted_question(self):
        state = self.decode()
        empty = self.perform("relation.refine", step=state["step_id"], expression={"xor": ["x", "y"]}, expected=1)
        q = self.steps.propose(self.snapshot(), "relation.query", {"step": empty["step_id"], "expression": "x"})
        self.assertNotIn(q["id"], Memory(self.root).state["step_observations"])
        answer = self.steps.execute(q["id"])["result"]["value"]
        self.assertEqual(answer["state"], "conflict")
        self.assertIsNone(answer["verdict"])
        self.assertEqual(answer["possible_values"], [])
        with self.assertRaises(ValueError):
            pack([answer["verdict"]])
        with self.assertRaisesRegex(ValueError, "unavailable variable"):
            self.perform("relation.query", step=empty["step_id"], expression="foreign")

    def test_source_correction_invalidates_descendants_and_preserves_history(self):
        state = self.decode()
        answer = self.perform("relation.query", step=state["step_id"], expression={"eq": ["x", "y"]})
        model = {**self.model, "worlds": [[0, 1], [1, 0]]}
        self.memory.put_source(self.source, json.dumps(model), expected_revision=1)
        with self.assertRaises(StaleState):
            self.steps.execute(answer["step_id"])
        current = self.decode()
        revised = self.perform("relation.query", step=current["step_id"], expression={"eq": ["x", "y"]})
        self.assertEqual(revised["result"]["value"]["verdict"], "no")
        self.assertEqual(revised["result"]["sources"], ["relation@2"])
        self.assertEqual(Memory(self.root).state["step_observations"][answer["step_id"]]["result"]["value"]["verdict"], "yes")

    def test_executor_change_invalidates_inherited_observations(self):
        state = self.decode()
        snapshot = self.memory.attach_observations(self.snapshot(), [state["step_id"]])
        with patch("nanoai.steps.file_digest", return_value="changed-module"):
            with self.assertRaises(StaleState):
                self.memory.attach_observations(snapshot, [])

    def test_checked_both_retains_two_witnesses_in_training_and_bf16_view(self):
        state = self.decode()
        answer = self.perform("relation.query", step=state["step_id"], expression="x")
        judgments = Judgments(self.memory)
        pred = judgments.predict(self.snapshot(), "Is x one?", "yes", "Provisional guess",
            predicate={"operator":"relation.image", "step_id":answer["step_id"], "expected":None})
        checked = judgments.assess(pred["id"])
        self.assertEqual(checked["spec"]["verdict"], "maybe")
        self.assertEqual(checked["spec"]["uncertainty"], "both_alternatives")
        carried = judgments.carry([checked["id"]])
        self.assertEqual(unpack(carried["packed"], 1), ["maybe"])
        self.assertEqual(judgments.board.read_tensor(carried["tensor"]["id"], core_identity="test-core"),
                         bytes.fromhex("0000803f0000"))
        restored = Judgments(Memory(self.root))
        row = restored.experiences("work")[0]
        self.assertEqual((row["predicted"],row["observed"]), ("yes","maybe"))
        self.assertTrue(row["eligible_for_predicate_training"])
        self.assertEqual(row["observation_uncertainty"], "both_alternatives")
        self.assertEqual(restored.memory.state["step_observations"][answer["step_id"]]["result"]["value"]["possible_values"], [0,1])

    def test_conflicting_relation_cannot_be_exported_as_a_checked_maybe(self):
        state = self.decode()
        empty = self.perform("relation.refine", step=state["step_id"], expression={"xor":["x","y"]}, expected=1)
        answer = self.perform("relation.query", step=empty["step_id"], expression="x")
        judgments = Judgments(self.memory)
        pred = judgments.predict(self.snapshot(), "Is x one?", "maybe", "An unassessed prediction",
            predicate={"operator":"relation.image", "step_id":answer["step_id"], "expected":None})
        before = self.memory.sequence
        with self.assertRaisesRegex(ValueError, "no ternary image"):
            judgments.assess(pred["id"])
        self.assertEqual(self.memory.sequence, before)
        self.assertEqual(judgments.experiences("work"), [])

    def test_rejected_assumption_blocks_its_descendants(self):
        state = self.decode()
        chosen = self.perform("relation.refine", step=state["step_id"], expression="x", expected=1)
        answer = self.perform("relation.query", step=chosen["step_id"], expression="y")
        self.steps.review(chosen["step_id"], "rejected", reviewer="independent-source-check", reason="The premise was unsupported")
        with self.assertRaises(StaleState):
            self.steps.execute(answer["step_id"])
        self.assertTrue(self.steps.execute(state["step_id"])["reobserved"])

    def test_bounds_and_admission_do_not_invent_state(self):
        for mutation in [lambda m:m.update(variables=["x", "x"]),
                         lambda m:m.update(worlds=[[True, 0]]),
                         lambda m:m.update(worlds=[[0, 0], [0, 0]]),
                         lambda m:m.update(worlds=[[0]])]:
            model = deepcopy(self.model)
            mutation(model)
            with self.assertRaises(ValueError):
                validate_model(model)
        expr = "x"
        for _ in range(18):
            expr = {"not": expr}
        with self.assertRaisesRegex(ValueError, "bound"):
            expression_variables(expr)
        state = self.decode()
        before = self.memory.sequence
        with self.assertRaises(ValueError):
            self.steps.propose(self.snapshot(), "relation.refine", {"step":state["step_id"], "expression":"x", "expected":True})
        self.assertEqual(self.memory.sequence, before)
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.perform("relation.extend", step=state["step_id"], variable="x", expression=None)

    def test_extension_preserves_old_worlds_and_refuses_hidden_filtering(self):
        state = self.decode()
        for expression in (None, 0, {"not": "x"}):
            extended = self.perform("relation.extend", step=state["step_id"], variable="z", expression=expression)
            old_projection = {tuple(row[:-1]) for row in extended["result"]["value"]["worlds"]}
            self.assertEqual(old_projection, {tuple(row) for row in self.model["worlds"]})
        before = self.memory.sequence
        with self.assertRaisesRegex(ValueError, "fields"):
            self.steps.propose(self.snapshot(), "relation.extend", {
                "step": state["step_id"], "variable": "z", "expression": None,
                "condition": {"eq": ["x", 1]}})
        self.assertEqual(self.memory.sequence, before)
        self.assertEqual(self.memory.state["step_observations"][state["step_id"]]["result"]["value"], self.model)

    def test_real_cli_replays_a_joint_query(self):
        state = self.decode()
        args = self.root / "args.json"
        args.write_text(json.dumps({"step":state["step_id"], "expression":{"eq":["x","y"]}}))
        run = subprocess.run([sys.executable, "-m", "nanoai.cli", "--state", str(self.root),
                              "step", "work", "relation.query", str(args), "--source", self.source,
                              "--core-identity", "test-core"], check=True, capture_output=True, text=True)
        result = json.loads(run.stdout)
        self.assertEqual(result["observation"]["result"]["value"]["verdict"], "yes")
        self.assertEqual(result["model_calls"], 0)

    def test_core_inquires_and_delivers_task_relative_completion_while_retaining_open_questions(self):
        from nanoai.agent import inquire
        from nanoai.documents import relation_note
        from nanoai.scratchboard import Scratchboard

        class SequenceCore:
            identity = "test-core"
            def __init__(self, plans):
                self.plans = list(plans)
                self.seen = []
            def generate(self, messages, max_tokens=None, schema=None):
                self.seen.append(json.loads(messages[-1]["content"]))
                output = self.plans.pop(0)
                return {"output": json.dumps(output), "elapsed_ms": 2, "tokens": 6, "finish_reason": "stop"}

        self.memory.set_task("work", "Certify whether x equals y, retain open marginals, and prepare followup.",
                             expected_revision=1, filenames=["equality.md", "followup.md"])
        state = self.decode()
        core = SequenceCore([
            {"operator": "relation.query", "expression": {"eq": ["x", "y"]}},
            {"operator": "relation.query", "expression": "x"},
        ])
        settled = inquire(self.memory, core, "work", state_step_id=state["step_id"])
        self.assertEqual(core.seen[0]["request"]["relation_state"]["worlds"], [[0, 0], [1, 1]])
        self.assertEqual(core.seen[0]["observed_calculations"][0]["operator"], "relation.decode")
        self.assertEqual(settled["result"]["value"]["state"], "decided")
        self.assertEqual(settled["result"]["value"]["verdict"], "yes")
        self.assertEqual(settled["next_state_step_id"], state["step_id"])

        open_q = inquire(self.memory, core, "work", state_step_id=settled["next_state_step_id"],
                         observation_ids=[settled["step_id"]])
        self.assertEqual(core.seen[1]["observed_calculations"][1]["operator"], "relation.query")
        self.assertEqual(core.seen[1]["observed_calculations"][1]["value"]["verdict"], "yes")
        self.assertEqual(open_q["result"]["value"]["state"], "both")
        self.assertEqual(open_q["result"]["value"]["verdict"], "maybe")
        self.assertEqual(open_q["result"]["value"]["possible_values"], [0, 1])
        self.assertEqual(open_q["result"]["value"]["witnesses"], {"0": [0, 0], "1": [1, 1]})

        with self.assertRaisesRegex(ValueError, "explicit task review"):
            relation_note(self.memory, settled["step_id"], open_step_ids=[open_q["step_id"]], filename="equality.md")
        self.steps.review(settled["step_id"], "confirmed", reviewer="task-contract-v1", reason="Equality is certified across all consistent worlds.")
        self.steps.review(open_q["step_id"], "confirmed", reviewer="task-contract-v1", reason="Marginal x remains open.")

        other_decided = self.perform("relation.query", step=state["step_id"], expression={"eq": ["x", "x"]})
        with self.assertRaisesRegex(ValueError, "decided relation query"):
            relation_note(self.memory, open_q["step_id"], filename="equality.md")
        with self.assertRaisesRegex(ValueError, "unresolved both state"):
            relation_note(self.memory, settled["step_id"], open_step_ids=[other_decided["step_id"]], filename="equality.md")
        with self.assertRaisesRegex(ValueError, "distinct step identities"):
            relation_note(self.memory, settled["step_id"], open_step_ids=[settled["step_id"]], filename="equality.md")

        node = Scratchboard(self.memory).relation(settled["step_id"], open_step_ids=[open_q["step_id"]], language="tr")
        self.assertIn("Kesinleşen soru", Scratchboard(self.memory).materialize(node["id"])["bytes"].decode("utf-8"))

        proposal = relation_note(self.memory, settled["step_id"], open_step_ids=[open_q["step_id"]], filename="equality.md")
        receipt = self.memory.commit(proposal["id"], core_identity=core.identity)
        self.assertTrue((self.root / receipt["path"]).is_file())

        fresh = Memory(self.root)
        self.assertEqual(fresh.recover(), [])
        completed = fresh.completed("work")
        self.assertEqual(completed["model_calls"], 0)
        self.assertEqual(completed["remaining"], ["followup.md"])
        self.assertEqual(len(completed["open_questions"]), 1)
        self.assertEqual(completed["open_questions"][0]["step_id"], open_q["step_id"])
        self.assertEqual(completed["open_questions"][0]["question"], "x")
        self.assertEqual(completed["open_questions"][0]["state"], "both")
        self.assertEqual(completed["open_questions"][0]["witnesses"], {"0": [0, 0], "1": [1, 1]})

        outcome = completed["artifacts"]["equality.md"]["outcome"]
        self.assertEqual(outcome["status"], "artifact-only")
        self.assertTrue(outcome["allows_completion"])
        self.assertEqual(outcome["evaluator_identity"], "task-contract-v1")
        self.assertEqual(outcome["completed_question"]["step_id"], settled["step_id"])
        self.assertEqual(outcome["completed_question"]["question"], {"eq": ["x", "y"]})
        self.assertEqual(outcome["completed_question"]["verdict"], "yes")
        self.assertEqual(outcome["completed_question"]["state"], "decided")
        self.assertEqual(outcome["completed_question"]["state_operator"], "relation.decode")
        self.assertEqual(outcome["uncertainty"][0]["step_id"], open_q["step_id"])
        self.assertEqual(outcome["uncertainty"][0]["witnesses"], {"0": [0, 0], "1": [1, 1]})
        self.assertIn("## Retained Open Questions", completed["artifacts"]["equality.md"]["content"])
        self.assertIn("[relation@1]", completed["artifacts"]["equality.md"]["content"])

        status = fresh.status()
        self.assertEqual(status["obligations"]["work"]["completed"], ["equality.md"])
        self.assertEqual(status["obligations"]["work"]["remaining"], ["followup.md"])
        self.assertEqual(status["obligations"]["work"]["open_questions"][0]["step_id"], open_q["step_id"])

    def test_continuing_inquiry_preserves_distinct_refinement_and_extension_contracts(self):
        from nanoai.agent import inquire
        from nanoai.documents import relation_note

        class SequenceCore:
            identity = "test-core"
            def __init__(self, plans):
                self.plans = list(plans)
            def generate(self, messages, max_tokens=None, schema=None):
                return {"output": json.dumps(self.plans.pop(0)), "elapsed_ms": 1, "tokens": 5, "finish_reason": "stop"}

        self.memory.set_task("work", "Deliver equality, then explore conditional refinement and conservative extension.",
                             expected_revision=1, filenames=["equality.md", "refined.md"])
        state = self.decode()
        core = SequenceCore([
            {"operator": "relation.query", "expression": {"eq": ["x", "y"]}},
            {"operator": "relation.query", "expression": "x"},
            {"operator": "relation.refine", "expression": "x", "expected": 1},
            {"operator": "relation.query", "expression": "y"},
            {"operator": "relation.extend", "variable": "z", "expression": None},
            {"operator": "relation.query", "expression": {"eq": ["x", "y"]}},
            {"operator": "relation.query", "expression": "z"},
        ])
        settled = inquire(self.memory, core, "work", state_step_id=state["step_id"])
        open_x = inquire(self.memory, core, "work", state_step_id=state["step_id"], observation_ids=[settled["step_id"]])
        self.steps.review(settled["step_id"], "confirmed", reviewer="task-contract-v1", reason="Base equality settled.")
        first_prop = relation_note(self.memory, settled["step_id"], open_step_ids=[open_x["step_id"]], filename="equality.md")
        self.memory.commit(first_prop["id"], core_identity=core.identity)

        refined = inquire(self.memory, core, "work", state_step_id=state["step_id"],
                          observation_ids=[settled["step_id"], open_x["step_id"]])
        self.assertEqual(refined["next_state_step_id"], refined["step_id"])
        self.assertEqual(refined["result"]["value"]["worlds"], [[1, 1]])
        self.assertIn("Conditional refinement", refined["result"]["scope"])

        refined_y = inquire(self.memory, core, "work", state_step_id=refined["next_state_step_id"])
        self.assertEqual(refined_y["result"]["value"]["state"], "decided")
        self.assertEqual(refined_y["result"]["value"]["verdict"], "yes")
        self.steps.review(refined_y["step_id"], "confirmed", reviewer="conditional-branch-v1", reason="Under premise x=1, y=1 is decided.")
        second_prop = relation_note(self.memory, refined_y["step_id"], filename="refined.md")
        self.memory.commit(second_prop["id"], core_identity=core.identity)

        extended = inquire(self.memory, core, "work", state_step_id=state["step_id"])
        self.assertEqual(extended["next_state_step_id"], extended["step_id"])
        self.assertEqual(extended["result"]["value"]["variables"], ["x", "y", "z"])
        self.assertEqual(extended["result"]["value"]["worlds"], [[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]])
        self.assertIn("Conservative extension", extended["result"]["scope"])

        ext_eq = inquire(self.memory, core, "work", state_step_id=extended["next_state_step_id"])
        self.assertEqual(ext_eq["result"]["value"]["verdict"], "yes")
        open_z = inquire(self.memory, core, "work", state_step_id=extended["next_state_step_id"])
        self.assertEqual(open_z["result"]["value"]["state"], "both")

        completed = Memory(self.root).completed("work")
        self.assertEqual(completed["remaining"], [])
        self.assertEqual(completed["artifacts"]["equality.md"]["outcome"]["completed_question"]["state_operator"], "relation.decode")
        self.assertEqual(completed["artifacts"]["refined.md"]["outcome"]["completed_question"]["state_operator"], "relation.refine")
        self.assertIn("Conditional refinement", completed["artifacts"]["refined.md"]["outcome"]["completed_question"]["state_scope"])
        open_By_question = {row["question"]: row for row in completed["open_questions"]}
        self.assertEqual(set(open_By_question), {"x", "z"})
        self.assertEqual(open_By_question["x"]["state_operator"], "relation.decode")
        self.assertEqual(open_By_question["z"]["state_operator"], "relation.extend")

    def test_source_correction_and_step_rejection_reopen_affected_deliveries_only(self):
        from nanoai.documents import relation_note

        self.memory.set_task("other", "Independent relational check.", filenames=["independent.md"])
        other_model = {"type": RELATION, "variables": ["a", "b"], "worlds": [[0, 1], [1, 0]], "scope": "Opposite pair"}
        self.memory.put_source("other_rel", json.dumps(other_model))
        other_snap = context(self.memory, "other", "test-core", selected_ids=["other_rel"])["snapshot"]
        other_read = self.steps.execute(self.steps.propose(other_snap, "source.read", {"source": "other_rel"})["id"])
        other_dec = self.steps.execute(self.steps.propose(other_snap, "relation.decode", {"step": other_read["step_id"]})["id"])
        other_q = self.steps.execute(self.steps.propose(other_snap, "relation.query", {"step": other_dec["step_id"], "expression": {"xor": ["a", "b"]}})["id"])
        self.steps.review(other_q["step_id"], "confirmed", reviewer="test", reason="Opposite pair xor is 1.")
        other_prop = relation_note(self.memory, other_q["step_id"], filename="independent.md")
        self.memory.commit(other_prop["id"], core_identity="test-core")

        state = self.decode()
        q_eq = self.perform("relation.query", step=state["step_id"], expression={"eq": ["x", "y"]})
        q_x = self.perform("relation.query", step=state["step_id"], expression="x")
        self.steps.review(q_eq["step_id"], "confirmed", reviewer="test", reason="Settled equality.")
        prop = relation_note(self.memory, q_eq["step_id"], open_step_ids=[q_x["step_id"]], filename="result.md")
        self.memory.commit(prop["id"], core_identity="test-core")
        self.assertEqual(self.memory.completed("work")["remaining"], [])
        self.assertEqual(self.memory.completed("other")["remaining"], [])

        self.steps.review(q_x["step_id"], "rejected", reviewer="test", reason="Open question superseded.")
        self.assertEqual(self.memory.completed("work")["remaining"], ["result.md"])
        self.assertEqual(self.memory.completed("work")["open_questions"], [])
        self.assertEqual(self.memory.completed("other")["remaining"], [])

        prop2 = relation_note(self.memory, q_eq["step_id"], filename="result.md")
        self.memory.commit(prop2["id"], core_identity="test-core")
        self.assertEqual(self.memory.completed("work")["remaining"], [])

        self.memory.put_source(self.source, json.dumps({**self.model, "worlds": [[0, 1], [1, 0]]}), expected_revision=1)
        status = Memory(self.root).status()
        self.assertEqual(status["obligations"]["work"]["remaining"], ["result.md"])
        self.assertEqual(status["obligations"]["other"]["remaining"], [])

    def test_empty_generation_and_scoped_writing_stay_separate_from_certified_enclosures(self):
        from nanoai.documents import relation_note
        from nanoai.experience import Experience
        from nanoai.harmonizer import Harmonizer, context_messages

        self.memory.set_task("work", "Deliver certified equality and a narrative summary.",
                             expected_revision=1, filenames=["equality.md", "summary.md"])
        state = self.decode()
        q_eq = self.perform("relation.query", step=state["step_id"], expression={"eq": ["x", "y"]})
        q_x = self.perform("relation.query", step=state["step_id"], expression="x")
        self.steps.review(q_eq["step_id"], "confirmed", reviewer="enclosure-check-v1", reason="Exact finite enclosure.")
        prop = relation_note(self.memory, q_eq["step_id"], open_step_ids=[q_x["step_id"]], filename="equality.md")
        self.memory.commit(prop["id"], core_identity="test-core")

        obs_text = ["Switches x and y move together in every declared world."]
        canon = self.memory.put_source("canon", obs_text[0])
        gen_src = self.memory.put_source("summary-input", json.dumps({"observations": obs_text}), role="observation")
        snap = self.memory.snapshot("work", [canon, gen_src], "test-core")
        msgs = context_messages(Harmonizer(self.memory).context(snap), output="artifact", source_ids=["canon"])
        exp = Experience(self.memory)
        empty_row = exp.record_generation(snap, question=self.memory.state["tasks"]["work"]["goal"],
            observations=obs_text, messages=msgs,
            generation={"text": "", "token_ids": [1, 2], "finish_reason": "length", "policy": "greedy-token-v1"},
            representation_identity="tok-v1", source_id="summary-input")
        empty_receipt = exp.execute(empty_row["id"], filename="summary.md")
        after_empty = Memory(self.root).completed("work")
        self.assertEqual(after_empty["remaining"], ["summary.md"])
        self.assertIn("equality.md", after_empty["artifacts"])
        self.assertEqual(self.memory.artifact_outcome(empty_receipt["operation_id"])["status"], "failed")
        self.assertFalse(self.memory.artifact_outcome(empty_receipt["operation_id"])["allows_completion"])

        good_row = exp.record_generation(snap, question=self.memory.state["tasks"]["work"]["goal"],
            observations=obs_text, messages=msgs,
            generation={"text": "Switches x and y match across the declared model while x remains open.", "token_ids": [3, 4], "finish_reason": "stop", "policy": "greedy-token-v1"},
            representation_identity="tok-v1", source_id="summary-input")
        exp.execute(good_row["id"], filename="summary.md")
        exp.assess_generation(good_row["id"],
            rubric={"scope": "Mechanical mention check; literary quality and external completeness remain separate",
                    "criteria": {"mentions_switches": True}, "training_target": None},
            evaluator_identity="scoped-rubric-v1",
            evaluate=lambda text, criteria: {"mentions_switches": "Switches x and y" in text})
        both_done = Memory(self.root).completed("work")
        self.assertEqual(both_done["remaining"], [])
        rel_outcome = both_done["artifacts"]["equality.md"]["outcome"]
        gen_outcome = both_done["artifacts"]["summary.md"]["outcome"]
        self.assertEqual(rel_outcome["status"], "artifact-only")
        self.assertIn("completed_question", rel_outcome)
        self.assertEqual(gen_outcome["status"], "passed")
        self.assertEqual(gen_outcome["evaluator_identity"], "scoped-rubric-v1")
        self.assertNotIn("completed_question", gen_outcome)
        self.assertEqual(len(both_done["open_questions"]), 1)

    def test_cli_relation_note_and_resume_preserve_open_questions_without_model(self):
        state = self.decode()
        q_eq = self.perform("relation.query", step=state["step_id"], expression={"eq": ["x", "y"]})
        q_x = self.perform("relation.query", step=state["step_id"], expression="x")
        subprocess.run([sys.executable, "-m", "nanoai.cli", "--state", str(self.root),
                        "review-step", q_eq["step_id"], "confirmed",
                        "--reviewer", "cli-reviewer", "--reason", "Certified equality"],
                       check=True, capture_output=True, text=True)
        board_run = subprocess.run([sys.executable, "-m", "nanoai.cli", "--state", str(self.root),
                                    "board-note", q_eq["step_id"], "--open-step", q_x["step_id"]],
                                   check=True, capture_output=True, text=True)
        board_out = json.loads(board_run.stdout)
        self.assertEqual(board_out["view"]["kind"], "relation")
        self.assertEqual(board_out["view"]["model_calls"], 0)
        note_run = subprocess.run([sys.executable, "-m", "nanoai.cli", "--state", str(self.root),
                                   "note", q_eq["step_id"], "--open-step", q_x["step_id"],
                                   "--filename", "result.md", "--commit"],
                                  check=True, capture_output=True, text=True)
        note_out = json.loads(note_run.stdout)
        self.assertEqual(note_out["model_calls"], 0)
        self.assertIsNotNone(note_out["receipt"])

        resume_run = subprocess.run([sys.executable, "-m", "nanoai.cli", "--state", str(self.root),
                                     "resume", "work"],
                                    check=True, capture_output=True, text=True)
        resumed = json.loads(resume_run.stdout)
        self.assertEqual(resumed["remaining"], [])
        self.assertEqual(resumed["model_calls"], 0)
        self.assertEqual(resumed["open_questions"][0]["step_id"], q_x["step_id"])
        self.assertEqual(resumed["artifacts"]["result.md"]["outcome"]["completed_question"]["verdict"], "yes")

    def test_inquire_retries_invalid_proposal_and_rejects_truncated_plan(self):
        from nanoai.agent import inquire
        from nanoai.memory import ProposalError

        class RetryCore:
            identity = "test-core"
            def __init__(self, responses):
                self.responses = list(responses)
                self.messages = []
            def generate(self, messages, max_tokens=None, schema=None):
                self.messages.append(messages)
                return self.responses.pop(0)

        state = self.decode()
        core = RetryCore([
            {"output": json.dumps({"operator": "relation.query", "expression": "missing_var"}),
             "elapsed_ms": 1, "tokens": 4, "finish_reason": "stop"},
            {"output": json.dumps({"operator": "relation.query", "expression": {"eq": ["x", "y"]}}),
             "elapsed_ms": 2, "tokens": 5, "finish_reason": "stop"},
        ])
        out = inquire(self.memory, core, "work", state_step_id=state["step_id"], max_attempts=2)
        self.assertEqual(out["result"]["value"]["verdict"], "yes")
        self.assertEqual(len(out["attempts"]), 2)
        self.assertEqual(len(self.memory.state["rejections"]), 1)
        self.assertIn("unavailable variable", self.memory.state["rejections"][0]["reason"])
        self.assertTrue(any(r["decision"] == "rejected" for r in self.memory.state["step_reviews"].values()))

        truncated = RetryCore([
            {"output": "{\"operator\":\"relation.query\"", "elapsed_ms": 1, "tokens": 3, "finish_reason": "length"},
        ])
        with self.assertRaisesRegex(ProposalError, "ended before completion"):
            inquire(self.memory, truncated, "work", state_step_id=state["step_id"], max_attempts=1)

    def test_support2_lattice_enclosure_and_bf16_relation_tensor_roundtrip(self):
        from nanoai.relations import (
            SUPPORT_BOTH, SUPPORT_CONFLICT, SUPPORT_NO, SUPPORT_YES,
            decode_relation_tensor, encode_relation_tensor, evaluate_support2,
            pack_support2, query_tensor_enclosure, unpack_support2,
        )
        from nanoai.scratchboard import Scratchboard
        from nanoai.tensor_store import TensorStore

        codes = [SUPPORT_NO, SUPPORT_BOTH, SUPPORT_YES, SUPPORT_CONFLICT, SUPPORT_YES]
        packed = pack_support2(codes)
        self.assertEqual(unpack_support2(packed, len(codes), labels=False), codes)
        self.assertEqual(evaluate_support2({"eq": ["x", "y"]}, {"x": SUPPORT_BOTH, "y": SUPPORT_BOTH}), SUPPORT_BOTH)

        state = self.decode()
        q_eq = self.perform("relation.query", step=state["step_id"], expression={"eq": ["x", "y"]})
        q_x = self.perform("relation.query", step=state["step_id"], expression="x")
        self.assertEqual(q_eq["result"]["value"]["verdict"], "yes")
        self.assertEqual(q_eq["result"]["value"]["box_verdict"], "maybe")
        self.assertTrue(q_eq["result"]["value"]["correlation_loss"])
        enclosure = query_tensor_enclosure(state["result"]["value"], {"eq": ["x", "y"]})
        self.assertTrue(enclosure["enclosure_holds"])
        self.assertEqual(enclosure["exact_code"] & enclosure["box_code"], enclosure["exact_code"])

        for mode in ("bf16_support", "bf16_signed", "pack5", "support2"):
            tensor = encode_relation_tensor(state["result"]["value"], mode=mode)
            decoded = decode_relation_tensor(tensor)
            self.assertEqual(decoded["variables"], ["x", "y"])
            self.assertEqual(decoded["worlds"], [[0, 0], [1, 1]])
            self.assertEqual(tensor[2]["box_marginals"], {"x": "maybe", "y": "maybe"})

        board = Scratchboard(self.memory)
        node_support = board.relation_tensor(q_eq["step_id"], open_step_ids=[q_x["step_id"]], mode="bf16_support")
        node_signed = board.relation_tensor(state["step_id"], mode="bf16_signed")
        roundtripped = board.decode_relation_tensor(node_support["id"], core_identity="test-core")
        self.assertEqual(roundtripped["worlds"], [[0, 0], [1, 1]])
        self.assertEqual(roundtripped["settled_questions"][0]["verdict"], "yes")
        self.assertEqual(roundtripped["open_questions"][0]["question"], "x")

        archived = board.archive_tensors([node_support["id"], node_signed["id"]], recipe="relational-tensor-test-v1")
        self.assertEqual(len(archived["tensors"]), 2)
        store = TensorStore(self.root / "tensors.sqlite3")
        described = store.resolve(archived["version_id"])
        self.assertEqual(described["recipe"], "relational-tensor-test-v1")
        self.assertEqual(len(described["tensors"]), 2)

    def test_relation_extend_relational_constraints_and_hidden_narrowing_rejection(self):
        state = self.decode()
        extended = self.perform(
            "relation.extend",
            step=state["step_id"],
            variable="z",
            expression={"or": [{"eq": ["z", "x"]}, {"eq": ["z", 1]}]},
        )
        self.assertEqual(extended["result"]["value"]["variables"], ["x", "y", "z"])
        self.assertEqual(extended["result"]["value"]["worlds"], [[0, 0, 0], [0, 0, 1], [1, 1, 1]])

        with self.assertRaisesRegex(ValueError, "narrows existing alternatives"):
            self.perform(
                "relation.extend",
                step=state["step_id"],
                variable="z",
                expression={"and": ["x", "z"]},
            )

    def test_relation_project_revise_and_curiosity_with_inquire_tensor_carry(self):
        from nanoai.agent import inquire

        state = self.decode()
        extended = self.perform("relation.extend", step=state["step_id"], variable="z", expression=None)
        projected = self.perform("relation.project", step=extended["step_id"], variables=["x", "y"])
        self.assertEqual(projected["result"]["value"]["variables"], ["x", "y"])
        self.assertEqual(projected["result"]["value"]["worlds"], [[0, 0], [1, 1]])
        self.assertEqual(projected["result"]["projection"]["eliminated_variables"], ["z"])
        self.assertFalse(projected["result"]["projection"]["exact_box"])
        self.assertEqual(projected["result"]["projection"]["fiber_counts"], {"[0,0]": 2, "[1,1]": 2})

        r1 = self.perform("relation.refine", step=state["step_id"], expression="x", expected=1)
        conflict = self.perform("relation.refine", step=r1["step_id"], expression="y", expected=0)
        self.assertEqual(conflict["result"]["value"]["worlds"], [])

        alt_model = {"type": RELATION, "variables": ["x", "y"], "worlds": [[0, 1], [1, 1]], "scope": "Updated switch table"}
        self.memory.put_source("alt_rel", json.dumps(alt_model))
        snap = context(self.memory, "work", "test-core", selected_ids=[self.source, "alt_rel"])["snapshot"]
        alt_read = self.steps.execute(self.steps.propose(snap, "source.read", {"source": "alt_rel"})["id"])
        alt_dec = self.steps.execute(self.steps.propose(snap, "relation.decode", {"step": alt_read["step_id"]})["id"])
        revised = self.steps.execute(
            self.steps.propose(
                snap,
                "relation.revise",
                {"step": conflict["step_id"], "replacement": alt_dec["step_id"], "reason": "Replace contradicted branch with updated switch table"},
            )["id"]
        )
        rev_val = revised["result"]["value"]
        rev_meta = revised["result"]["revision"]
        self.assertEqual(rev_val["worlds"], [[0, 1], [1, 1]])
        self.assertTrue(rev_meta["reopened_conflict"])
        self.assertEqual(rev_meta["added_worlds"], [[0, 1], [1, 1]])

        curiosity = self.perform(
            "relation.curiosity",
            step=extended["step_id"],
            questions=["x", "z", {"eq": ["x", "y"]}],
            observations=["x", "z", {"eq": ["x", "y"]}],
        )
        c_val = curiosity["result"]["value"]
        self.assertEqual(c_val["open_questions"], ["x", "z"])
        self.assertEqual(c_val["settled_questions"], [{"expression": {"eq": ["x", "y"]}, "verdict": "yes"}])
        self.assertEqual(c_val["recommended_observation"]["observation"], "x")
        self.assertEqual(c_val["recommended_observation"]["information_bits"], 1.0)
        self.assertEqual(c_val["candidates"][-1]["observation"], {"eq": ["x", "y"]})
        self.assertEqual(c_val["candidates"][-1]["information_bits"], 0.0)

        class PlanCore:
            identity = "test-core"
            def __init__(self, plan):
                self.plan = plan
            def generate(self, messages, max_tokens=None, schema=None):
                return {"output": json.dumps(self.plan), "elapsed_ms": 1, "tokens": 5, "finish_reason": "stop"}

        inq = inquire(
            self.memory,
            PlanCore({"operator": "relation.curiosity", "questions": ["x", "y"], "observations": ["x", {"eq": ["x", "y"]}]}),
            "work",
            state_step_id=state["step_id"],
            carry_tensor=True,
        )
        self.assertEqual(inq["next_state_step_id"], state["step_id"])
        self.assertIn("tensor_node_id", inq)
        self.assertEqual(inq["result"]["value"]["recommended_observation"]["observation"], "x")

    def test_cli_board_relation_tensor_and_archive_tensors(self):
        state = self.decode()
        q_eq = self.perform("relation.query", step=state["step_id"], expression={"eq": ["x", "y"]})
        q_x = self.perform("relation.query", step=state["step_id"], expression="x")
        tensor_run = subprocess.run(
            [sys.executable, "-m", "nanoai.cli", "--state", str(self.root),
             "board-relation-tensor", q_eq["step_id"], "--open-step", q_x["step_id"], "--mode", "bf16_support"],
            check=True, capture_output=True, text=True,
        )
        tensor_out = json.loads(tensor_run.stdout)
        node_id = tensor_out["node"]["id"]
        self.assertEqual(tensor_out["view"]["kind"], "tensor")
        self.assertEqual(tensor_out["view"]["dtype"], "BF16")
        self.assertEqual(tensor_out["view"]["shape"], [2, 4])

        archive_run = subprocess.run(
            [sys.executable, "-m", "nanoai.cli", "--state", str(self.root),
             "board-archive-tensors", node_id, "--recipe", "cli-tensor-archive-v1"],
            check=True, capture_output=True, text=True,
        )
        archive_out = json.loads(archive_run.stdout)
        self.assertEqual(archive_out["recipe"], "cli-tensor-archive-v1")
        self.assertTrue(Path(archive_out["archive_path"]).is_file())


if __name__ == "__main__":
    unittest.main()


