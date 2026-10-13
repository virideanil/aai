from contextlib import closing
import json
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest

from nanoai.core import digest_bytes, file_digest
from nanoai.sqlite_journal import canonical
from nanoai.tensor_store import TensorStore, describe_reference, inventory


def tensor_file(path):
    header = json.dumps({"weight": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
                         "bias": {"dtype": "F32", "shape": [3], "data_offsets": [12, 24]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(range(24)))
    return file_digest(path)


class TensorStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "candidate.safetensors"
        self.identity = tensor_file(self.path)
        self.db = self.root / "tensors.sqlite3"
        self.store = TensorStore(self.db, create=True)

    def tearDown(self):
        self.tmp.cleanup()

    def register(self):
        return self.store.register(self.path, expected=self.identity, basis=self.identity,
                                   recipe="authored-fixture", source="test_tensor_store.py")

    def test_typed_shapes_lineage_and_fresh_read_preserve_native_bytes(self):
        before = self.path.read_bytes()
        ref = self.register()
        self.assertEqual(ref, self.register())
        row = TensorStore(self.db).resolve(ref["version"], path=self.path)
        self.assertEqual(row["tensor_count"], 2)
        self.assertEqual({r["name"]: r["shape"] for r in row["tensors"]}, {"bias": [3], "weight": [2, 3]})
        with self.store.connection() as con:
            self.assertEqual(con.execute("SELECT SUM(elements) FROM tensors WHERE dtype='BF16'").fetchone()[0], 6)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM versions").fetchone()[0], 1)
        child = self.store.register(self.path, expected=self.identity, basis=self.identity,
            recipe="reviewed", source="review", parent=ref["version"])
        self.assertEqual(self.store.resolve(child["version"])["parent"], ref["version"])
        self.assertEqual(self.path.read_bytes(), before)

    def test_foreign_key_failure_leaves_no_location_or_version(self):
        self.register()
        copy = self.root / "copy.safetensors"; copy.write_bytes(self.path.read_bytes())
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.register(copy, expected=self.identity, basis=self.identity,
                recipe="child", source="test", parent="shake256:" + "a" * 128)
        with self.store.connection() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM locations").fetchone()[0], 1)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM versions").fetchone()[0], 1)

    def test_changed_bytes_and_corrupted_inventory_are_rejected(self):
        ref = self.register()
        data = self.path.read_bytes(); self.path.write_bytes(data[:-1] + b'\xff')
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            self.store.resolve(ref["version"])
        self.path.write_bytes(data)
        with self.store.connection(write=True) as con:
            con.execute("UPDATE dimensions SET extent=4 WHERE name='weight' AND axis=0")
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.store.resolve(ref["version"])

    def test_self_consistent_sql_edit_cannot_reinterpret_unchanged_tensor_bytes(self):
        ref = self.register()
        with self.store.connection(write=True) as con:
            con.execute("UPDATE tensors SET dtype='I16' WHERE name='weight'")
            rows = self.store._tensors(con, self.identity)
            con.execute("UPDATE artifacts SET inventory_id=? WHERE id=?", (digest_bytes(canonical(rows)), self.identity))
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.store.resolve(ref["version"])

    def test_read_open_does_not_create_or_rebrand_another_database(self):
        absent = self.root / "absent.sqlite3"
        with self.assertRaises(sqlite3.OperationalError):
            TensorStore(absent)
        self.assertFalse(absent.exists())
        other = self.root / "other.sqlite3"
        with closing(sqlite3.connect(other)) as con:
            con.execute("CREATE TABLE unrelated(value)"); con.commit()
        before = other.read_bytes()
        for create in (False, True):
            with self.assertRaisesRegex(ValueError, "identity"):
                TensorStore(other, create=create)
        self.assertEqual(other.read_bytes(), before)

    def test_manifest_reference_binds_version_and_survives_unrelated_writes(self):
        ref = {**self.register(), "database": "tensors.sqlite3"}
        first = describe_reference(ref, directory=self.root, expected=self.identity)
        self.store.record("new-run", protocol=self.identity, phase="observed", values={})
        self.assertEqual(describe_reference(ref, directory=self.root, expected=self.identity), first)
        for change in ({"artifact": "shake256:" + "b" * 128}, {"database": ""},
                       {"version": "unknown"}, {"extra": True}, {"database": "missing.sqlite3"}):
            with self.subTest(change=change), self.assertRaises((ValueError, sqlite3.OperationalError)):
                describe_reference({**ref, **change}, directory=self.root, expected=self.identity)
        self.assertFalse((self.root / "missing.sqlite3").exists())
        with self.store.connection(write=True) as con:
            con.execute("UPDATE versions SET source='changed' WHERE id=?", (ref["version"],))
        with self.assertRaisesRegex(ValueError, "provenance"):
            describe_reference(ref, directory=self.root, expected=self.identity)

    def test_learning_events_survive_process_death_and_concurrent_writers(self):
        ref = self.register()
        self.store.record("train", protocol=self.identity, phase="update", version=ref["version"],
            values={"epoch": 1, "parameter_updates": 1, "loss": .25, "elapsed_seconds": 2.5, "exact": 2**90})
        code = """
import sys
from nanoai.tensor_store import TensorStore
s=TensorStore(sys.argv[1])
for i in range(4):s.record('train', protocol=sys.argv[2],phase='update',values={'worker':sys.argv[3],'epoch':i})
"""
        children = [subprocess.Popen([sys.executable, "-c", code, str(self.db), self.identity, str(i)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE) for i in range(2)]
        for child in children:
            _, error = child.communicate(timeout=30)
            self.assertEqual(child.returncode, 0, error.decode())
        killed = subprocess.run([sys.executable, "-c", """
import os,sqlite3,sys
c=sqlite3.connect(sys.argv[1],isolation_level=None)
c.execute('BEGIN IMMEDIATE');c.execute("DELETE FROM learning_events");os._exit(71)
""", str(self.db)])
        self.assertEqual(killed.returncode, 71)
        events = TensorStore(self.db).events("train")
        self.assertEqual([r["sequence"] for r in events], list(range(1, 10)))
        self.assertEqual(events[0]["values"]["exact"], 2**90)
        with self.assertRaisesRegex(ValueError, "protocol"):
            self.store.record("train", protocol="shake256:" + "b" * 128, phase="update", values={})
        self.assertEqual(len(self.store.events("train")), 9)

    def test_malformed_tensor_payload_is_rejected_before_publication(self):
        self.path.write_bytes(self.path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "offsets"):
            inventory(self.path)
        with self.store.connection() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0], 0)

    def test_real_candidate_writer_registers_queryable_metadata_and_reopens(self):
        from nanoai.agent import DEFAULT_IDENTITY
        from nanoai.experience import Experience
        from nanoai.memory import Memory
        memory = Memory(self.root / "state"); memory.initialize(DEFAULT_IDENTITY)
        candidate = Experience(memory).register_candidate(self.path.read_bytes(), feature_basis=self.identity,
            tensor_format="safetensors", training={"recipe": "test"})
        self.assertTrue(candidate["path"].endswith(".safetensors"))
        fresh = Memory(memory.root)
        self.assertEqual(fresh.state["learners"][self.identity], candidate)
        ref = candidate["tensor_version"]
        self.assertEqual(TensorStore(memory.root / ref["database"]).resolve(ref["version"])["tensor_count"], 2)
        self.assertEqual(Experience(fresh).register_candidate(self.path.read_bytes(), feature_basis=self.identity,
            tensor_format="safetensors", training={"recipe": "test"}), candidate)
        with self.assertRaisesRegex(ValueError, "provenance"):
            Experience(fresh).register_candidate(self.path.read_bytes(), feature_basis="changed",
                tensor_format="safetensors", training={"recipe": "test"})


if __name__ == "__main__":
    unittest.main()
