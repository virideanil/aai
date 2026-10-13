from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from nanoai.agent import DEFAULT_IDENTITY
from nanoai.memory import Memory, StaleState, sjql_modules
from nanoai import sqlite_journal as journal


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sqlite_requires_no_sjql_and_preserves_exact_json(self):
        memory = Memory(self.root)
        self.assertEqual(memory.backend, "sqlite")
        memory.initialize(DEFAULT_IDENTITY)
        source = memory.put_source("exact", 'İstanbul 雨 😀\x00 9007199254740993')
        self.assertEqual(Memory(self.root).state["sources"]["exact"], source)
        with self.assertRaisesRegex(ValueError, "owner"):
            Memory(self.root, backend="sjql")

    def test_stale_append_and_uncommitted_process_death(self):
        memory = Memory(self.root)
        memory.initialize(DEFAULT_IDENTITY)
        head = memory.head
        tx = {"catalogueId": memory.catalogue, "previous": 1, "sequence": 2,
              "events": [{"kind": "test", "number": 2**90, "nothing": None}]}
        next_head = journal.append(memory.segments, tx, head)
        self.assertEqual(list(journal.tail(memory.segments, head, at_head=next_head))[0][1], tx)
        with self.assertRaisesRegex(ValueError, "Stale"):
            journal.append(memory.segments, tx, head)
        result = subprocess.run([sys.executable, "-c", """
import os,sqlite3,sys
c=sqlite3.connect(sys.argv[1],isolation_level=None)
c.execute('BEGIN IMMEDIATE')
c.execute('UPDATE journal SET sequence=999')
os._exit(71)
""", str(memory.segments / "journal.sqlite3")])
        self.assertEqual(result.returncode, 71)
        self.assertEqual(journal.head(memory.segments), next_head)
        self.assertEqual(len(list(journal.tail(memory.segments, None, at_head=next_head))), 2)

    def test_unrelated_database_is_preserved(self):
        directory = self.root / "transactions.sqlite.d"
        directory.mkdir()
        path = directory / "journal.sqlite3"
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("CREATE TABLE unrelated (value TEXT)")
            connection.commit()
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "identity"):
            Memory(self.root)
        self.assertEqual(path.read_bytes(), before)

    def test_corrupt_earlier_row_and_missing_chain_are_rejected(self):
        memory = Memory(self.root)
        memory.initialize(DEFAULT_IDENTITY)
        memory.put_source("first", "one")
        path = memory.segments / "journal.sqlite3"
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("UPDATE transactions SET payload=? WHERE sequence=1", (b'{}',))
            connection.commit()
        with self.assertRaisesRegex(ValueError, "digest"):
            Memory(self.root)

    def test_two_real_writers_preserve_both_histories(self):
        memory = Memory(self.root)
        memory.initialize(DEFAULT_IDENTITY)
        code = """
import sys
from nanoai.memory import Memory
m=Memory(sys.argv[1])
for i in range(8):m.put_source(sys.argv[2]+str(i), 'observed '+str(i))
"""
        children = [subprocess.Popen([sys.executable, "-c", code, str(self.root), prefix],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE) for prefix in ("a", "b")]
        for child in children:
            _, error = child.communicate(timeout=30)
            self.assertEqual(child.returncode, 0, error.decode())
        fresh = Memory(self.root)
        self.assertEqual(fresh.sequence, 17)
        self.assertEqual(set(fresh.state["sources"]), {p+str(i) for p in ("a", "b") for i in range(8)})

    def test_legacy_journal_keeps_its_source_owner(self):
        try:
            sjql_modules()
        except RuntimeError:
            self.skipTest("Legacy SJQL package is optional")
        memory = Memory(self.root, backend="sjql")
        memory.initialize(DEFAULT_IDENTITY)
        memory.put_source("legacy", "Exact retained history.")
        fresh = Memory(self.root)
        self.assertEqual(fresh.backend, "sjql")
        self.assertEqual(fresh.state, memory.state)
        self.assertFalse((self.root / "transactions.sqlite.d").exists())


class DerivedSourceTests(unittest.TestCase):
    def test_correction_and_changed_artifact_invalidate_derived_work(self):
        with tempfile.TemporaryDirectory() as root:
            memory = Memory(root)
            memory.initialize(DEFAULT_IDENTITY)
            original = memory.put_source("canon", "Mira keeps three clocks.")
            memory.set_task("book", "Write.", filenames=["story.md"])
            snapshot = memory.snapshot("book", [original], "core")
            proposal = memory.record_proposal(snapshot, {"filename": "story.md",
                "content": "Mira wound three clocks. [canon@1]", "citations": ["canon@1"]})
            receipt = memory.commit(proposal["id"], core_identity="core")
            basis = memory.attach_artifacts(snapshot, [receipt["operation_id"]])
            derived = memory.put_source("draft", "Mira wound three clocks.", role="observation", derived_from=basis)
            self.assertTrue(memory.source_current(derived))
            path = Path(root) / receipt["path"]
            original_bytes = path.read_bytes()
            path.write_text("User edit.")
            self.assertFalse(memory.source_current(derived))
            path.write_bytes(original_bytes)
            memory.put_source("canon", "Mira keeps four clocks.", expected_revision=1)
            with self.assertRaises(StaleState):
                memory.snapshot("book", [derived], "core")
            self.assertNotIn("draft", {r["id"] for r in memory.retrieve("Mira clocks", count=8)})
            self.assertEqual(Memory(root).state["source_history"]["draft"][0], derived)


if __name__ == "__main__":
    unittest.main()
