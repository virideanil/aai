import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from nanoai.agent import DEFAULT_IDENTITY, context
from nanoai.memory import Memory, document_references


class MemoryConsumerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.memory = Memory(self.root)
        self.memory.initialize(DEFAULT_IDENTITY)
        self.memory.put_source("spec", "The panel width is 19 cm.")
        self.memory.set_task("panel", "Create a panel width note.", filenames=["panel.md"])

    def tearDown(self):
        self.temporary.cleanup()

    def proposal(self, *, content="# Panel\nWidth: 19 cm. [spec@1]\n"):
        source = self.memory.state["sources"]["spec"]
        snap = self.memory.snapshot("panel", [source], "test-core-v1")
        return self.memory.record_proposal(snap, {"filename": "panel.md", "content": content, "citations": ["spec@1"]})

    def test_current_source_history_and_stale_proposal(self):
        proposal = self.proposal()
        self.memory.put_source("spec", "The panel width is 23 cm.", expected_revision=1, note="Reviewed correction")
        restored = Memory(self.root)
        self.assertEqual([r["text"] for r in restored.state["source_history"]["spec"]], ["The panel width is 19 cm.", "The panel width is 23 cm."])
        with self.assertRaisesRegex(ValueError, "stale"):
            restored.commit(proposal["id"], core_identity="test-core-v1")
        self.assertFalse((self.root / "artifacts").exists())

    def test_core_change_and_foreign_citation_rejected(self):
        proposal = self.proposal()
        with self.assertRaisesRegex(ValueError, "Core version"):
            self.memory.commit(proposal["id"], core_identity="test-core-v2")
        snap = proposal["snapshot"]
        with self.assertRaisesRegex(ValueError, "Citations"):
            self.memory.record_proposal(snap, {"filename": "panel.md", "content": "Unsupported [other@1]", "citations": ["other@1"]})

    def test_malformed_reference_candidates_cannot_hide_behind_a_valid_citation(self):
        snap = context(self.memory, "panel", "test-core-v1", selected_ids=["spec"])["snapshot"]
        for bad in ("[source_id@revision]", "[spec@1x]", "[spec @1]", "[spec@1 extra]"):
            content = "Width 19 cm. [spec@1] " + bad
            self.assertIn(bad[1:-1], document_references(content))
            with self.assertRaisesRegex(ValueError, "source revisions|references"):
                self.memory.record_proposal(snap, {"filename":"panel.md", "content":content,
                    "citations":sorted(document_references(content))})
            with self.assertRaisesRegex(ValueError, "references"):
                self.memory.record_proposal(snap, {"filename":"panel.md", "content":content, "citations":["spec@1"]})

    def test_same_revision_writers_and_user_file_drift(self):
        other = Memory(self.root)
        self.memory.put_source("spec", "New value 23 cm.", expected_revision=1)
        with self.assertRaisesRegex(ValueError, "Stale source"):
            other.put_source("spec", "Wrong stale update.", expected_revision=1)
        self.memory.put_source("spec", "The panel width is 19 cm.", expected_revision=2)
        snap = self.memory.snapshot("panel", [self.memory.state["sources"]["spec"]], "test-core-v1")
        p = self.memory.record_proposal(snap, {"filename": "panel.md", "content": "19 cm [spec@3]", "citations": ["spec@3"]})
        receipt = self.memory.commit(p["id"], core_identity="test-core-v1")
        target = self.root / receipt["path"]
        target.write_text("User changed this file.")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.memory.commit(p["id"], core_identity="test-core-v1")
        self.assertEqual(target.read_text(), "User changed this file.")

    def test_crash_after_real_publication_and_idempotent_recovery(self):
        proposal = self.proposal()
        code = "from nanoai.memory import Memory; import sys; Memory(sys.argv[1]).commit(sys.argv[2],core_identity='test-core-v1',crash_after_publish=True)"
        child = subprocess.run([sys.executable, "-c", code, str(self.root), proposal["id"]], capture_output=True, text=True, timeout=15)
        self.assertEqual(child.returncode, 88, child.stderr)
        files = list((self.root / "artifacts").glob("*/*.md"))
        self.assertEqual(len(files), 1)
        before = files[0].stat()
        restored = Memory(self.root)
        self.assertEqual(restored.state["identity"], DEFAULT_IDENTITY)
        self.assertEqual(len(restored.status()["pending_operations"]), 1)
        receipts = restored.recover()
        self.assertEqual(len(receipts), 1)
        self.assertTrue(receipts[0]["reconciled_existing"])
        self.assertEqual(receipts[0]["core_identity_at_proposal"], "test-core-v1")
        self.assertEqual(restored.recover(), [])
        self.assertEqual(files[0].stat().st_mtime_ns, before.st_mtime_ns)
        self.assertEqual(files[0].read_text(), "# Panel\nWidth: 19 cm. [spec@1]\n")
        self.assertEqual(len(restored.state["receipts"]), 1)

    def test_unrelated_completed_work_and_changed_dependency(self):
        proposal = self.proposal()
        self.memory.commit(proposal["id"], core_identity="test-core-v1")
        self.memory.put_source("other", "An unrelated color is blue.")
        status = self.memory.status()["artifacts"][0]
        self.assertTrue(status["dependency_current"])
        self.assertTrue(status["selection_current"])
        self.memory.put_source("spec", "The panel width is 24 cm.", expected_revision=1)
        self.assertFalse(self.memory.status()["artifacts"][0]["dependency_current"])

    def test_journal_integrity_and_identity_persistence(self):
        self.memory.initialize(DEFAULT_IDENTITY)
        with self.assertRaisesRegex(ValueError, "migration"):
            self.memory.initialize({**DEFAULT_IDENTITY, "name": "Different"})
        import sqlite3
        from contextlib import closing
        with closing(sqlite3.connect(self.memory.segments / "journal.sqlite3")) as connection:
            connection.execute("UPDATE transactions SET payload=? WHERE digest=?", (b'{}', self.memory.head))
            connection.commit()
        with self.assertRaisesRegex(ValueError, "digest"):
            Memory(self.root)

    def test_changed_document_does_not_block_independent_recovery(self):
        first = self.proposal()
        receipt = self.memory.commit(first["id"], core_identity="test-core-v1")
        target = self.root / receipt["path"]
        target.write_text("User's revised document.")
        self.memory.set_task("independent", "A separate document", filenames=["independent.md"])
        snap = self.memory.snapshot("independent", [], "test-core-v1")
        second = self.memory.record_proposal(snap, {"filename": "independent.md", "content": "Independent work", "citations": []})
        self.memory.commit(second["id"], core_identity="test-core-v1")
        restored = Memory(self.root)
        self.assertEqual(restored.recover(), [])
        status = restored.status()
        self.assertEqual(status["recovery_issues"][0]["task_id"], "panel")
        self.assertEqual(status["obligations"]["panel"]["remaining"], ["panel.md"])
        self.assertEqual(status["obligations"]["independent"]["remaining"], [])
        self.assertEqual(target.read_text(), "User's revised document.")

    def test_corrected_source_reopens_only_dependent_obligations(self):
        proposal = self.proposal()
        self.memory.commit(proposal["id"], core_identity="test-core-v1")
        self.assertEqual(self.memory.status()["obligations"]["panel"]["remaining"], [])
        self.memory.put_source("other", "Different evidence")
        self.assertEqual(self.memory.status()["obligations"]["panel"]["remaining"], [])
        self.memory.put_source("spec", "The width is 27 cm.", expected_revision=1)
        self.assertEqual(self.memory.status()["obligations"]["panel"]["remaining"], ["panel.md"])

    def test_selector_output_changes_only_when_selected_evidence_changes(self):
        prepared = context(self.memory, "panel", "test-core-v1", source_count=1)
        output = {"filename": "panel.md", "content": "19 cm [spec@1]", "citations": ["spec@1"]}
        self.memory.put_source("color", "The unrelated sky is blue.")
        proposal = self.memory.record_proposal(prepared["snapshot"], output)
        self.memory.commit(proposal["id"], core_identity="test-core-v1")
        self.assertEqual(self.memory.status()["obligations"]["panel"]["remaining"], [])
        self.memory.put_source("panel", "panel width note")
        with self.assertRaisesRegex(ValueError, "selection is stale"):
            self.memory.record_proposal(prepared["snapshot"], output)
        status = self.memory.status()
        self.assertTrue(status["artifacts"][0]["dependency_current"])
        self.assertFalse(status["artifacts"][0]["selection_current"])
        self.assertEqual(status["obligations"]["panel"]["remaining"], ["panel.md"])

    def test_replacement_artifact_retains_prior_bytes_and_lineage(self):
        first = self.proposal()
        old = self.memory.commit(first["id"], core_identity="test-core-v1")
        self.memory.put_source("spec", "The panel width is 23 cm.", expected_revision=1)
        snap = self.memory.snapshot("panel", [self.memory.state["sources"]["spec"]], "test-core-v2")
        second = self.memory.record_proposal(snap, {"filename": "panel.md", "content": "23 cm [spec@2]", "citations": ["spec@2"]})
        new = self.memory.commit(second["id"], core_identity="test-core-v2")
        self.assertEqual(new["supersedes"], old["operation_id"])
        self.assertEqual(new["core_identity_at_proposal"], "test-core-v2")
        self.assertIn("19 cm", (self.root / old["path"]).read_text())
        self.assertIn("23 cm", (self.root / new["path"]).read_text())
        self.assertEqual(Memory(self.root).state["operations"][new["operation_id"]]["supersedes"], old["operation_id"])


if __name__ == "__main__":
    unittest.main()
