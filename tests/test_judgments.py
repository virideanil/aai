from copy import deepcopy
import itertools
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from nanoai.agent import DEFAULT_IDENTITY, context
from nanoai.harmonizer import Harmonizer
from nanoai.judgments import CHANNELS, Judgments, pack, unpack
from nanoai.memory import Memory, StaleState
from nanoai.scratchboard import Scratchboard
from nanoai.steps import Steps


class JudgmentConsumerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.memory = Memory(self.root)
        self.memory.initialize(DEFAULT_IDENTITY)
        self.memory.put_source("dimensions", '{"length":{"value":23,"unit":"cm"},"count":{"value":2,"unit":"count"}}')
        self.memory.set_task("total", "Determine the total rail length", filenames=["total.md"])
        self.core = "original-test-core"
        self.judgments = Judgments(self.memory)

    def tearDown(self):
        self.tmp.cleanup()

    def snapshot(self):
        return context(self.memory, "total", self.core, selected_ids=["dimensions"])["snapshot"]

    def target(self, *, pending=False):
        steps = Steps(self.memory)
        calc = steps.calculate_fields(self.snapshot(), {"operator":"quantity.product", "inputs":[
            {"source":"dimensions", "path":["length"]}, {"source":"dimensions", "path":["count"]}]})
        step = self.memory.state["steps"][calc["step_id"]]
        if pending:
            step = steps.propose(self.snapshot(), step["operator"], step["arguments"])
        return step["id"]

    @staticmethod
    def predicate(step_id, value=46, dimensions=None):
        return {"operator":"result.equals", "step_id":step_id,
                "expected":{"value":value,"dimensions":{"cm":1} if dimensions is None else dimensions}}

    def test_all_243_codes_tails_and_unused_bit_patterns(self):
        for values in itertools.product(CHANNELS, repeat=5):
            self.assertEqual(unpack(pack(values), 5), list(values))
        for count in range(16):
            values = [CHANNELS[i%3] for i in range(count)]
            self.assertEqual(unpack(pack(values), count), values)
            self.assertEqual(len(pack(values)), (count+4)//5)
        for byte in range(243,256):
            with self.assertRaises(ValueError): unpack(bytes([byte]), 5)
        for raw, count in ((b"\x03",1), (b"\x09",2), (b"\x00",0), (b"",1), (b"\x00",True)):
            with self.assertRaises(ValueError): unpack(raw,count)
        for values in ([True], [0], ["unknown"]):
            with self.assertRaises(ValueError): pack(values)

    def test_wrong_prediction_and_checked_outcome_remain_separate_after_restart(self):
        pred = self.judgments.predict(self.snapshot(), "Does the declared product equal 44 cm?", "yes", "A fallible prediction",
                                      predicate=self.predicate(self.target(),44))
        assessed = self.judgments.assess(pred["id"])
        self.assertEqual(assessed["spec"]["verdict"],"no")
        restored = Judgments(Memory(self.root))
        row = restored.experiences("total")[0]
        self.assertEqual((row["predicted"],row["observed"]),("yes","no"))
        self.assertTrue(row["eligible_for_predicate_training"])
        self.assertIn("task relevance",row["scope"])
        self.assertEqual(restored.memory.state["scratchboard"][pred["id"]]["spec"]["verdict"],"yes")
        snapshot = context(restored.memory,"total",self.core,selected_ids=["dimensions"],scratchboard_ids=[assessed["id"]])["snapshot"]
        unit = next(r for r in Harmonizer(restored.memory).view(snapshot)["entries"] if r["kind"]=="working_unit")
        self.assertEqual(unit["representation"]["prediction"],pred["id"])
        self.assertEqual(unit["representation"]["reason"],"different_observed_result")

    def test_pending_observation_keeps_its_original_maybe_when_new_result_arrives(self):
        sid = self.target(pending=True)
        pred = self.judgments.predict(self.snapshot(),"Does the product equal 46 cm?","maybe","Await the check",predicate=self.predicate(sid))
        pending = self.judgments.assess(pred["id"])
        self.assertEqual(pending["spec"]["reason"],"awaiting_observation")
        Steps(self.memory).execute(sid)
        later = self.judgments.assess(pred["id"])
        self.assertEqual(later["spec"]["verdict"],"yes")
        rows = Judgments(Memory(self.root)).experiences("total")
        self.assertEqual([r["observed"] for r in rows],["maybe","yes"])
        self.assertEqual([r["eligible_for_predicate_training"] for r in rows],[False,True])
        self.assertTrue(all(r["current"] for r in rows))

    def test_unclear_meaning_and_incompatible_dimensions_are_not_negative_labels(self):
        missing = self.judgments.predict(self.snapshot(),"Are the rails blue?","maybe","No color source")
        wrong_unit = self.judgments.predict(self.snapshot(),"Does the result equal 46 seconds?","no","Provisional only",
                                            predicate=self.predicate(self.target(),46,{"s":1}))
        outcomes = [self.judgments.assess(p["id"])["spec"] for p in (missing,wrong_unit)]
        self.assertEqual([x["verdict"] for x in outcomes],["maybe","maybe"])
        self.assertEqual([x["reason"] for x in outcomes],["no_executable_predicate","incompatible_dimensions"])
        self.assertFalse(any(r["eligible_for_predicate_training"] for r in self.judgments.experiences("total")))

    def test_packed_and_bf16_carriers_keep_semantics_and_dependency_identity(self):
        nodes = [self.judgments.predict(self.snapshot(),f"Candidate {v}",v,"Unassessed example") for v in CHANNELS]
        carried = self.judgments.carry([n["id"] for n in nodes])
        self.assertEqual(unpack(carried["packed"],carried["count"]),list(CHANNELS))
        raw = Scratchboard(Memory(self.root)).read_tensor(carried["tensor"]["id"],core_identity=self.core)
        self.assertEqual(struct.unpack("<9H",raw),(0x3F80,0,0,0,0x3F80,0,0,0,0x3F80))
        self.assertEqual(carried["tensor"]["origin"]["channels"],list(CHANNELS))
        with self.assertRaises(StaleState): self.judgments.board.read_tensor(carried["tensor"]["id"],core_identity="another-basis")
        self.memory.put_source("unrelated","Independent evidence")
        self.assertEqual(self.judgments.board.read_tensor(carried["tensor"]["id"],core_identity=self.core),raw)
        self.memory.put_source("dimensions",'{"changed":true}',expected_revision=1)
        with self.assertRaises(StaleState): self.judgments.board.read_tensor(carried["tensor"]["id"],core_identity=self.core)

    def test_forged_assessment_rejected_and_source_correction_excludes_feedback(self):
        pred = self.judgments.predict(self.snapshot(),"Is the result 46 cm?","no","Potential mistake",predicate=self.predicate(self.target()))
        assessed = self.judgments.assess(pred["id"])
        forged = deepcopy(assessed["spec"]);forged["verdict"]="no"
        with self.assertRaisesRegex(ValueError,"differs from"):
            self.judgments.board.add(assessed["snapshot"],forged)
        self.memory.put_source("dimensions",'{"length":{"value":24,"unit":"cm"}}',expected_revision=1)
        row = self.judgments.experiences("total")[0]
        self.assertFalse(row["current"])
        self.assertFalse(row["eligible_for_predicate_training"])
        self.assertEqual((row["predicted"],row["observed"]),("no","yes"))
        self.assertIn("stale",row["invalidated_by"])

    def test_rejected_target_and_foreign_task_are_rejected_as_current_evidence(self):
        sid = self.target()
        pred = self.judgments.predict(self.snapshot(),"Is the result 46 cm?","yes","Prediction",predicate=self.predicate(sid))
        assessed = self.judgments.assess(pred["id"])
        Steps(self.memory).review(sid,"rejected",reviewer="task-review",reason="Arithmetic was not relevant to this task")
        self.assertFalse(self.judgments.experiences("total")[0]["eligible_for_predicate_training"])
        with self.assertRaises(StaleState): self.judgments.carry([assessed["id"]])
        self.memory.set_task("other","Another task",filenames=["other.md"])
        snap = context(self.memory,"other",self.core,selected_ids=["dimensions"])["snapshot"]
        with self.assertRaises(StaleState):
            self.judgments.predict(snap,"A different task","maybe","Unknown",predicate=self.predicate(sid))

    def test_fresh_cli_reads_question_role_and_carries_binary_without_model_start(self):
        def cli(*args):
            result = subprocess.run([sys.executable,"-m","nanoai.cli","--state",str(self.root),*args],capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)
            return json.loads(result.stdout)
        pred = cli("judge","total","Are the rails blue?","--verdict","maybe","--reason","Color is unspecified","--source","dimensions")["prediction"]
        assessed = cli("assess-judgment",pred["id"])["assessment"]
        carried = cli("carry-judgments",pred["id"],assessed["id"])
        self.assertEqual(unpack(bytes.fromhex(carried["packed_hex"]),2),["maybe","maybe"])
        view = cli("dictionary","total","--source","dimensions","--board",assessed["id"])
        unit = next(r for r in view["entries"] if r["kind"]=="working_unit")
        self.assertEqual(unit["representation"]["question"],"Are the rails blue?")
        self.assertEqual(cli("experiences","total")["parameter_updates"],0)

    def test_revision_rejects_dependency_through_a_judgment_predicate_atomically(self):
        board, steps = self.judgments.board, Steps(self.memory)
        for composed in (False, True):
            with self.subTest(composed=composed):
                original = board.add(self.snapshot(), {"kind":"literal", "text":"Current working assumption"})
                dependent = self.snapshot()
                dependent["scratchboard"] = {original["id"]:original["digest"]}
                result = steps.calculate_fields(dependent, {"operator":"quantity.product", "inputs":[
                    {"source":"dimensions", "path":["length"]}, {"source":"dimensions", "path":["count"]}]})
                prediction = self.judgments.predict(self.snapshot(), "Is the result 46 cm?", "yes", "A provisional result",
                                                    predicate=self.predicate(result["step_id"]))
                spec = {"kind":"concat", "nodes":[prediction["id"]]} if composed else prediction["spec"]
                before = self.memory.sequence
                with self.assertRaisesRegex(ValueError, "must not depend"):
                    board.revise(original["id"], spec, reason="Attempt to adopt a judgment based on the superseded unit")
                self.assertEqual(self.memory.sequence, before)
                restored = Scratchboard(Memory(self.root))
                self.assertEqual(restored.materialize(original["id"])["bytes"], b"Current working assumption")
                self.assertEqual(restored.latest(original["id"])["id"], original["id"])


if __name__ == "__main__":
    unittest.main()
