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


if __name__ == "__main__":
    unittest.main()
