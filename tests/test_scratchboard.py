import json
from copy import deepcopy
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from nanoai.agent import DEFAULT_IDENTITY, context
from nanoai.documents import quantity_note
from nanoai.memory import Memory, StaleState
from nanoai.scratchboard import Scratchboard, UnresolvedSlot
from nanoai.steps import Steps


class ScratchboardConsumerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.memory = Memory(self.root)
        self.memory.initialize(DEFAULT_IDENTITY)
        self.memory.put_source("spec", "Şule has two rails.")
        self.memory.set_task("note", "Prepare the current source note.", filenames=["note.md"])
        self.board = Scratchboard(self.memory)

    def tearDown(self):
        self.tmp.cleanup()

    def snapshot(self, *, core="original-test-core"):
        return context(self.memory, "note", core, selected_ids=["spec"])["snapshot"]

    def test_ten_million_slots_are_a_sparse_persistent_range(self):
        before = self.memory.sequence
        reserved = self.board.reserve(self.snapshot(), 10_000_000, constraints="A source-bound text or tensor decision per address")
        view = self.board.describe(reserved["id"])
        self.assertEqual(view["logical_units"], 10_000_000)
        self.assertEqual(view["unresolved_slots"], 10_000_000)
        self.assertEqual(self.memory.sequence, before+1)
        self.assertLess(view["stored_node_bytes"], 2048)
        self.assertEqual(view["reachable_nodes"], 1)
        piece = self.board.add(self.snapshot(), {"kind":"literal", "text":"Source copy [spec@1]"})
        bound = self.board.bind(reserved["id"], 9_999_999, piece["id"])
        restored = Scratchboard(Memory(self.root))
        self.assertEqual(restored.slot(bound["id"], 9_999_999)["node"], piece["id"])
        self.assertEqual(restored.slot(reserved["id"], 9_999_999)["state"], "unresolved")
        self.assertEqual(restored.describe(bound["id"])["unresolved_slots"], 9_999_999)
        with self.assertRaises(UnresolvedSlot): restored.materialize(bound["id"])
        with self.assertRaisesRegex(ValueError,"immutable"): restored.bind(bound["id"],9_999_999,piece["id"])
        with self.assertRaises(ValueError): restored.slot(bound["id"],10_000_000)
        with self.assertRaises(ValueError): restored.reserve(self.snapshot(),True,constraints="values")

    def test_copied_content_needs_current_dependencies_even_without_novel_text(self):
        snapshot = self.snapshot()
        raw = self.memory.state["sources"]["spec"]["text"].encode()
        source = self.board.add(snapshot,{"kind":"source","source":"spec","start":0,"stop":len(raw)})
        citation = self.board.add(snapshot,{"kind":"literal","text":" [spec@1]\n"})
        composed = self.board.add(snapshot,{"kind":"concat","nodes":[source["id"],citation["id"]]})
        proposal = self.board.propose_document(composed["id"])
        self.assertEqual(proposal["output"]["content"],"Şule has two rails. [spec@1]\n")
        self.memory.put_source("unrelated","Independent current source")
        self.assertEqual(self.board.materialize(composed["id"])["bytes"],proposal["output"]["content"].encode())
        self.memory.put_source("spec","Şule has three rails.",expected_revision=1)
        with self.assertRaises(StaleState): self.board.materialize(composed["id"])
        with self.assertRaises(StaleState): self.memory.commit(proposal["id"],core_identity="original-test-core")

    def test_unicode_boundaries_and_materialization_budgets(self):
        snap = self.snapshot()
        with self.assertRaises(UnicodeDecodeError):
            self.board.add(snap,{"kind":"source","source":"spec","start":0,"stop":1})
        with self.assertRaisesRegex(ValueError,"outside"):
            self.board.add(snap,{"kind":"source","source":"other","start":0,"stop":0})
        leaf = self.board.add(snap,{"kind":"literal","text":"ABC"})
        repeated = self.board.add(snap,{"kind":"repeat","node":leaf["id"],"count":100_000})
        self.assertEqual(self.board.describe(repeated["id"])["text_bytes"],300_000)
        with self.assertRaisesRegex(ValueError,"byte budget"): self.board.materialize(repeated["id"])
        self.assertEqual(self.board.materialize(repeated["id"],max_bytes=300_000)["bytes"],b"ABC"*100_000)
        with self.assertRaises(ValueError): self.board.propose_document(repeated["id"])

    def test_tensor_words_basis_restart_and_tamper(self):
        words = struct.pack("<65536H",*range(65536))
        node = self.board.tensor(self.snapshot(),words,shape=[256,256],dtype="bf16")
        restored = Scratchboard(Memory(self.root))
        self.assertEqual(restored.read_tensor(node["id"],core_identity="original-test-core"),words)
        with self.assertRaisesRegex(StaleState,"basis"): restored.read_tensor(node["id"],core_identity="different-core")
        with self.assertRaisesRegex(ValueError,"tensor"): restored.materialize(node["id"])
        path = self.root/'scratchboard/blobs'/f"{node['spec']['blob']}.bin"
        path.write_bytes(words[:-2]+b"\x00\x00")
        with self.assertRaisesRegex(ValueError,"identity"): restored.read_tensor(node["id"],core_identity="original-test-core")

    def test_cross_task_basis_and_unresolved_publication(self):
        child = self.board.add(self.snapshot(core="other-core"),{"kind":"literal","text":"Text"})
        compatible = self.board.add(self.snapshot(),{"kind":"concat","nodes":[child["id"]]})
        self.assertEqual(self.board.materialize(compatible["id"])["bytes"],b"Text")
        tensor = self.board.tensor(self.snapshot(core="other-core"),b"\x00\x00",shape=[1])
        with self.assertRaisesRegex(StaleState,"basis"):
            self.board.add(self.snapshot(),{"kind":"concat","nodes":[tensor["id"]]})
        slots = self.board.reserve(self.snapshot(),2,constraints="Finish both sections")
        with self.assertRaises(UnresolvedSlot): self.board.propose_document(slots["id"])
        self.assertEqual(self.memory.state["operations"],{})
        self.memory.set_task("note","A revised task",expected_revision=1,filenames=["note.md"])
        with self.assertRaises(StaleState): self.board.describe(slots["id"])

    def test_explicit_cross_task_reuse_retains_origin_relevance_and_new_core(self):
        from nanoai.judgments import Judgments
        original = self.board.add(self.snapshot(),{"kind":"literal","text":"Şule has two rails. [spec@1]"})
        self.memory.set_task("other","Reuse the current source statement",filenames=["other.md"])
        snap = context(self.memory,"other","new-core",selected_ids=["spec"])["snapshot"]
        with self.assertRaisesRegex(StaleState,"another task"):
            self.board.add(snap,{"kind":"concat","nodes":[original["id"]]})
        relevant = Judgments(self.memory).predict(snap,"Is this source statement relevant to the requested quotation?","yes","The task requests this exact statement")
        imported = self.board.import_node(original["id"],snap,relevance=relevant["id"],reason="Carry the source statement into the requested quotation")
        proposal = self.board.propose_document(imported["id"])
        receipt = self.memory.commit(proposal["id"],core_identity="new-core")
        self.assertEqual(proposal["snapshot"]["task_id"],"other")
        self.assertEqual(imported["origin"]["parent"],original["id"])
        self.assertEqual((self.root/receipt["path"]).read_text(),"Şule has two rails. [spec@1]")
        self.assertEqual(Scratchboard(Memory(self.root)).materialize(imported["id"])["bytes"],proposal["output"]["content"].encode())
        new_relevance = deepcopy(relevant["spec"]);new_relevance["verdict"]="no"
        self.board.revise(relevant["id"],new_relevance,reason="The requested quotation changed")
        with self.assertRaises(StaleState): self.board.materialize(imported["id"])
        self.assertEqual((self.root/receipt["path"]).read_text(),proposal["output"]["content"])

    def test_reuse_does_not_discard_transitive_tensor_basis_or_dependency_cycles(self):
        from nanoai.judgments import Judgments
        snap = self.snapshot()
        tensor = self.board.tensor(snap,b"\x00\x00",shape=[1])
        with_tensor = deepcopy(snap);with_tensor["scratchboard"]={tensor["id"]:tensor["digest"]}
        text = self.board.add(with_tensor,{"kind":"literal","text":"Interpretation of a tensor [spec@1]"})
        other_core = self.snapshot(core="new-core")
        with self.assertRaisesRegex(StaleState,"basis"):
            self.board.add(other_core,{"kind":"concat","nodes":[text["id"]]})
        original = self.board.add(snap,{"kind":"literal","text":"Original [spec@1]"})
        relevance = Judgments(self.memory).predict(snap,"Is the exact original still requested?","yes","Current task asks for it")
        imported = self.board.import_node(original["id"],snap,relevance=relevance["id"],reason="Explicit reuse")
        with self.assertRaisesRegex(ValueError,"must not depend"):
            self.board.revise(original["id"],imported["spec"],reason="Attempt to replace a source with its import")

    def calculation(self):
        self.memory.put_source("spec",'{"length":{"value":23,"unit":"cm"},"count":{"value":2,"unit":"count"}}',expected_revision=1)
        steps = Steps(self.memory)
        result = steps.calculate_fields(self.snapshot(),{"operator":"quantity.product","inputs":[
            {"source":"spec","path":["length"]},{"source":"spec","path":["count"]}]})
        steps.review(result["step_id"],"confirmed",reviewer="authored-task-predicate",reason="The requested total is length times count")
        return result

    def test_quantity_artifact_is_exact_and_recovers_after_real_process_interruption(self):
        result = self.calculation()
        proposal = quantity_note(self.memory,result["step_id"],language="tr")
        expected = "# Hesaplama\n\nHesaplanan toplam: **46 cm**. [spec@2]\n\n23 cm × 2 = 46 cm.\n"
        self.assertEqual(proposal["output"]["content"],expected)
        self.assertIn("scratchboard",proposal["snapshot"])
        self.assertEqual(proposal["metrics"]["render_model_calls"],0)
        code = "from nanoai.memory import Memory; import sys; Memory(sys.argv[1]).commit(sys.argv[2],core_identity='original-test-core',crash_after_publish=True)"
        child = subprocess.run([sys.executable,"-c",code,str(self.root),proposal["id"]],capture_output=True,text=True,timeout=15)
        self.assertEqual(child.returncode,88,child.stderr)
        restored = Memory(self.root)
        self.assertEqual(len(restored.recover()),1)
        self.assertEqual(restored.recover(),[])
        complete = restored.completed("note")
        self.assertEqual(complete["remaining"],[])
        self.assertEqual(Path(complete["artifacts"]["note.md"]["artifact_path"]).read_text(),expected)

    def test_archive_reference_detects_later_artifact_change(self):
        leaf = self.board.add(self.snapshot(),{"kind":"literal","text":"Reviewed current text [spec@1]"})
        proposal = self.board.propose_document(leaf["id"])
        receipt = self.memory.commit(proposal["id"],core_identity="original-test-core")
        archived = self.board.add(self.snapshot(),{"kind":"artifact","operation_id":receipt["operation_id"]})
        copy = self.board.propose_document(archived["id"])
        self.assertEqual(copy["output"]["content"],proposal["output"]["content"])
        (self.root/receipt["path"]).write_text("Changed by another writer")
        with self.assertRaisesRegex(ValueError,"artifact changed"):
            self.memory.commit(copy["id"],core_identity="original-test-core")

    def test_cli_reserve_and_reload_do_not_start_a_model(self):
        def cli(*args):
            r = subprocess.run([sys.executable,"-m","nanoai.cli","--state",str(self.root),*args],capture_output=True,text=True,timeout=15)
            self.assertEqual(r.returncode,0,r.stderr)
            return json.loads(r.stdout)
        output = cli("board-reserve","note","10000000","--constraints","Unresolved future work","--source","spec")
        view = cli("board-show",output["node"]["id"],"--slot","9999999")
        self.assertEqual(view["state"],"unresolved")
        self.assertEqual(view["model_calls"],0)

    def test_revision_invalidates_descendants_and_preserves_independent_branches(self):
        snap = self.snapshot()
        original = self.board.add(snap, {"kind":"literal", "text":"Candidate A [spec@1]"})
        branch = self.board.branch(original["id"], reason="Keep the first candidate for comparison")
        dependent = self.board.add(snap, {"kind":"concat", "nodes":[original["id"]]})
        proposal = self.board.propose_document(dependent["id"])
        receipt = self.memory.commit(proposal["id"], core_identity="original-test-core")
        old_path = self.root/receipt["path"]
        old_bytes = old_path.read_bytes()
        pending = self.board.propose_document(dependent["id"])
        before = self.memory.sequence
        revised = self.board.revise(original["id"], {"kind":"literal", "text":"Candidate B [spec@1]"}, reason="Compare the selected representation")
        self.assertEqual(self.memory.sequence, before+1)
        restored = Scratchboard(Memory(self.root))
        self.assertEqual(restored.latest(original["id"])["id"], revised["id"])
        self.assertEqual(restored.materialize(branch["id"])["bytes"], old_bytes)
        for node in (original, dependent):
            with self.assertRaises(StaleState): restored.materialize(node["id"])
        with self.assertRaises(StaleState): restored.memory.commit(pending["id"], core_identity="original-test-core")
        self.assertEqual(restored.memory.completed("note")["remaining"], ["note.md"])
        self.assertEqual(old_path.read_bytes(), old_bytes)
        recomposed = restored.add(self.snapshot(), {"kind":"concat", "nodes":[revised["id"]]})
        fresh = restored.propose_document(recomposed["id"])
        new_receipt = restored.memory.commit(fresh["id"], core_identity="original-test-core")
        self.assertEqual(new_receipt["supersedes"], receipt["operation_id"])
        self.assertFalse(restored.memory.completed("note")["remaining"])

    def test_refinement_checks_exact_bytes_before_an_atomic_revision(self):
        snap = self.snapshot()
        original = self.board.add(snap, {"kind":"literal", "text":"Şule [spec@1]"})
        parts = [self.board.add(snap, {"kind":"literal", "text":text}) for text in ("Şule", " [spec@1]", "wrong")]
        before = self.memory.sequence
        with self.assertRaisesRegex(ValueError, "changes materialized"):
            self.board.refine(original["id"], [parts[2]["id"]], reason="Attempt a different meaning")
        self.assertEqual(self.memory.sequence, before)
        self.assertEqual(self.board.materialize(original["id"])["bytes"], "Şule [spec@1]".encode())
        refined = self.board.refine(original["id"], [p["id"] for p in parts[:2]], reason="Represent the name and citation separately")
        self.assertEqual(self.memory.sequence, before+1)
        self.assertEqual(self.board.materialize(refined["id"])["bytes"], "Şule [spec@1]".encode())
        self.assertEqual(self.memory.state["scratchboard_revisions"][original["id"]]["invariant"], "byte-identical materialization")
        last = self.board.revise(refined["id"], {"kind":"literal", "text":"A deliberate alternative [spec@1]"}, reason="Change the proposal")
        self.assertEqual(Scratchboard(Memory(self.root)).latest(original["id"])["id"], last["id"])

    def test_replacement_rejects_direct_observation_and_artifact_dependency_cycles(self):
        snap = self.snapshot()
        old = self.board.add(snap, {"kind":"literal", "text":"Current source [spec@1]"})
        proposal = self.board.propose_document(old["id"])
        receipt = self.memory.commit(proposal["id"], core_identity="original-test-core")
        step_snapshot = self.snapshot()
        step_snapshot["scratchboard"] = {old["id"]:old["digest"]}
        steps = Steps(self.memory)
        step = steps.propose(step_snapshot, "source.read", {"source":"spec"})
        steps.execute(step["id"])
        observed = self.memory.attach_observations(self.snapshot(), [step["id"]])
        before = self.memory.sequence
        for spec, snapshot in (({"kind":"concat", "nodes":[old["id"]]}, None),
                               ({"kind":"artifact", "operation_id":receipt["operation_id"]}, None),
                               ({"kind":"literal", "text":"Observed candidate"}, observed)):
            with self.assertRaisesRegex(ValueError, "must not depend"):
                self.board.revise(old["id"], spec, reason="Attempt a dependent replacement", snapshot=snapshot)
        self.assertEqual(self.memory.sequence, before)
        self.assertEqual(self.board.materialize(old["id"])["bytes"], b"Current source [spec@1]")

    def test_source_correction_requires_explicit_current_snapshot_and_preserves_lineage(self):
        old = self.board.add(self.snapshot(), {"kind":"source", "source":"spec", "start":0, "stop":len("Şule has two rails.".encode())})
        self.memory.put_source("spec", "Şule has three rails.", expected_revision=1)
        spec = {"kind":"source", "source":"spec", "start":0, "stop":len("Şule has three rails.".encode())}
        with self.assertRaises(StaleState): self.board.revise(old["id"], spec, reason="Use corrected evidence")
        new = self.board.revise(old["id"], spec, reason="Use corrected evidence", snapshot=self.snapshot())
        self.assertEqual(new["origin"]["parent"], old["id"])
        self.assertEqual(self.board.materialize(new["id"])["bytes"], "Şule has three rails.".encode())
        self.memory.set_task("other", "A separate obligation", filenames=["other.md"])
        other = context(self.memory, "other", "original-test-core", selected_ids=["spec"])["snapshot"]
        with self.assertRaisesRegex(StaleState, "keeps its task"):
            self.board.revise(new["id"], spec, reason="Attempt a different obligation", snapshot=other)

    def test_tensor_reshape_preserves_all_bf16_words_and_rejects_a_new_extent(self):
        words = struct.pack("<65536H", *range(65536))
        tensor = self.board.tensor(self.snapshot(), words, shape=[256,256])
        branch = self.board.branch(tensor["id"], reason="Keep the original view")
        before = self.memory.sequence
        with self.assertRaisesRegex(ValueError, "shape"):
            self.board.reshape_tensor(tensor["id"], [32,32], reason="Attempt an invalid extent")
        self.assertEqual(self.memory.sequence, before)
        reshaped = self.board.reshape_tensor(tensor["id"], [16,16,256], reason="Use a three-axis view")
        restored = Scratchboard(Memory(self.root))
        self.assertEqual(restored.read_tensor(reshaped["id"], core_identity="original-test-core"), words)
        self.assertEqual(restored.read_tensor(branch["id"], core_identity="original-test-core"), words)
        self.assertEqual(reshaped["spec"]["blob"], tensor["spec"]["blob"])
        self.assertEqual(len(list((self.root/"scratchboard/blobs").glob("*.bin"))), 1)
        self.assertEqual(restored.latest(tensor["id"])["spec"]["shape"], [16,16,256])
        with self.assertRaises(StaleState): restored.read_tensor(tensor["id"], core_identity="original-test-core")

    def test_missing_tensor_dependency_reopens_only_affected_artifact(self):
        tensor = self.board.tensor(self.snapshot(), b"\x00\x00", shape=[1])
        snapshot = self.snapshot()
        snapshot["scratchboard"] = {tensor["id"]:tensor["digest"]}
        proposal = self.memory.record_proposal(snapshot, {"filename":"note.md", "content":"Recorded view [spec@1]", "citations":["spec@1"]})
        self.memory.commit(proposal["id"], core_identity="original-test-core")
        (self.root/"scratchboard/blobs"/f"{tensor['spec']['blob']}.bin").unlink()
        status = self.memory.status()
        self.assertFalse(status["artifacts"][0]["dependency_current"])
        self.assertTrue(status["artifacts"][0]["bytes_verified"])
        self.assertEqual(status["obligations"]["note"]["remaining"], ["note.md"])

    def test_cli_revision_is_visible_to_a_fresh_consumer(self):
        def cli(*args):
            r = subprocess.run([sys.executable,"-m","nanoai.cli","--state",str(self.root),*args],capture_output=True,text=True,timeout=15)
            self.assertEqual(r.returncode,0,r.stderr)
            return json.loads(r.stdout)
        spec_path = self.root/"candidate.json"
        spec_path.write_text(json.dumps({"kind":"literal","text":"Candidate A [spec@1]"}))
        original = cli("board-add","note",str(spec_path),"--source","spec")["node"]
        branch = cli("board-branch",original["id"],"--reason","Keep the earlier candidate")["node"]
        spec_path.write_text(json.dumps({"kind":"literal","text":"Candidate B [spec@1]"}))
        revised = cli("board-revise",original["id"],str(spec_path),"--reason","Adopt another candidate")["node"]
        self.assertEqual(cli("board-latest",original["id"])["node"]["id"], revised["id"])
        self.assertEqual(cli("board-document",branch["id"])["proposal"]["output"]["content"], "Candidate A [spec@1]")
        self.assertEqual(cli("board-document",revised["id"],"--commit")["proposal"]["output"]["content"], "Candidate B [spec@1]")

    def test_generator_source_lines_relation_table_and_slot_sequence(self):
        self.memory.put_source("spec", "Line one\nLine two\nLine three", expected_revision=1)
        self.memory.put_source(
            "rel",
            json.dumps({
                "type": "finite_relation_v1",
                "variables": ["a", "b"],
                "worlds": [[1, 0], [1, 1]],
                "scope": "Declared finite relation",
            }),
        )
        snap = context(self.memory, "note", "original-test-core", selected_ids=["spec", "rel"])["snapshot"]
        lines_node = self.board.generator(snap, "source_lines", {"source": "spec", "start_line": 1, "stop_line": 2, "numbered": True})
        self.assertEqual(lines_node["metrics"]["logical_units"], 2)
        self.assertEqual(self.board.materialize(lines_node["id"])["bytes"].decode(), "1: Line one\n2: Line two\n")

        steps = Steps(self.memory)
        read_step = steps.propose(snap, "source.read", {"source": "rel"})
        steps.execute(read_step["id"])
        decode_step = steps.propose(snap, "relation.decode", {"step": read_step["id"]})
        steps.execute(decode_step["id"])
        steps.review(decode_step["id"], "confirmed", reviewer="test-reviewer", reason="Valid relation model")
        table_node = self.board.generator(
            snap,
            "relation_table",
            {"step_id": decode_step["id"], "questions": [{"xor": ["a", "b"]}]},
        )
        table_text = self.board.materialize(table_node["id"])["bytes"].decode()
        self.assertIn("| `a` | yes | `01` | `[1]` |", table_text)
        self.assertIn("| `b` | maybe | `11` | `[0, 1]` |", table_text)
        self.assertIn("### Open Questions", table_text)

        item_a = self.board.add(snap, {"kind": "literal", "text": "Alpha [spec@2]"})
        item_b = self.board.add(snap, {"kind": "literal", "text": "Beta [rel@1]"})
        seq_node = self.board.generator(
            snap,
            "slot_sequence",
            {"items": [item_a["id"], item_b["id"]], "template": "{index}. {item}\n", "start_index": 1},
        )
        self.assertEqual(self.board.materialize(seq_node["id"])["bytes"].decode(), "1. Alpha [spec@2]\n2. Beta [rel@1]\n")
        self.board.revise(item_a["id"], {"kind": "literal", "text": "Alpha-2 [spec@2]"}, reason="Update first item")
        with self.assertRaises(StaleState):
            self.board.materialize(seq_node["id"])

    def test_bind_many_atomic_bulk_slot_binding(self):
        snap = self.snapshot()
        slots = self.board.reserve(snap, 4, constraints="Four ordered lines")
        parts = [self.board.add(snap, {"kind": "literal", "text": f"Part {i} [spec@1]\n"}) for i in range(4)]
        before = self.memory.sequence
        with self.assertRaises(ValueError):
            self.board.bind_many(slots["id"], {0: parts[0]["id"], 9: parts[1]["id"]})
        self.assertEqual(self.memory.sequence, before)
        half = self.board.bind_many(slots["id"], {0: parts[0]["id"], 2: parts[2]["id"]})
        self.assertEqual(self.memory.sequence, before + 1)
        self.assertEqual(self.board.describe(half["id"])["unresolved_slots"], 2)
        full = self.board.bind_many(half["id"], {1: parts[1]["id"], 3: parts[3]["id"]})
        self.assertEqual(self.board.describe(full["id"])["unresolved_slots"], 0)
        self.assertEqual(
            self.board.materialize(full["id"])["bytes"].decode(),
            "".join(f"Part {i} [spec@1]\n" for i in range(4)),
        )

    def test_visual_report_generator_and_visual_presentation_audit(self):
        from nanoai.documents import audit_visual_presentation
        snap = self.snapshot()
        node = self.board.generator(
            snap,
            "visual_report",
            {
                "title": "Argon Structural & Visual Verification",
                "citations": ["spec@1"],
                "sections": [
                    {
                        "heading": "Quantitative Summary",
                        "body": "Verified rail geometry and unit alignment [spec@1].",
                        "table": {
                            "headers": ["Component", "Length", "Count", "Total"],
                            "rows": [["Rail", "23 cm", "2", "46 cm"]],
                        },
                        "code": {
                            "language": "python",
                            "snippet": "total_cm = 23 * 2",
                        },
                    }
                ],
                "perception_regions": [
                    {"region_id": 1, "bbox": [10, 20, 120, 64]},
                ],
            },
        )
        audit = self.board.audit_visual(
            node["id"],
            perception_regions=[{"region_id": 1, "bbox": [10, 20, 120, 64]}],
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["score"], 1.0)
        self.assertEqual(audit["metrics"]["table_count"], 2)
        self.assertEqual(audit["metrics"]["code_block_count"], 1)
        self.assertEqual(audit["metrics"]["perception_ref_count"], 1)

        proposal = self.board.propose_document(
            node["id"],
            enforce_visual_rules=True,
            perception_regions=[{"region_id": 1, "bbox": [10, 20, 120, 64]}],
        )
        self.assertIn("# Argon Structural & Visual Verification", proposal["output"]["content"])

        # Malformed visual markdown fails audit and blocks enforce_visual_rules proposal
        bad_node = self.board.add(
            snap,
            {
                "kind": "literal",
                "text": "### Skipped H1\n\n| A | B |\n| --- |\n\n```\nuntagged code\n",
            },
        )
        bad_audit = audit_visual_presentation(
            self.board.materialize(bad_node["id"])["bytes"].decode()
        )
        self.assertFalse(bad_audit["passed"])
        rule_ids = {v["rule_id"] for v in bad_audit["violations"]}
        self.assertIn("visual.hierarchy", rule_ids)
        self.assertIn("visual.table_alignment", rule_ids)
        self.assertIn("visual.code_fences", rule_ids)
        with self.assertRaisesRegex(ValueError, "visual presentation rules"):
            self.board.propose_document(bad_node["id"], enforce_visual_rules=True)


if __name__ == "__main__":
    unittest.main()


