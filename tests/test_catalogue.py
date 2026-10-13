import json
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

from nanoai.agent import DEFAULT_IDENTITY
from nanoai.catalogue import LanguageCatalogue
from nanoai.core import digest_bytes
from nanoai.harmonizer import Harmonizer
from nanoai.memory import Memory
from nanoai.perception import attach_perception, read_see_bundle
from test_master_tokenizer import fixture


class CatalogueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.memory = Memory(self.root / "journal")
        self.memory.initialize(DEFAULT_IDENTITY)
        self.memory.put_source("words", "aba b aba", note="Authored test text")
        self.memory.set_task("language", "Inspect exact language observations")
        self.tokenizer = fixture()

    def test_journal_retry_restart_followers_and_correction(self):
        owner = LanguageCatalogue(self.memory)
        first = owner.observe("read-1", "language", "words", self.tokenizer)
        sequence = self.memory.sequence
        self.assertEqual(first, owner.observe("read-1", "language", "words", self.tokenizer))
        self.assertEqual(self.memory.sequence, sequence)
        view = LanguageCatalogue(Memory(self.root / "journal")).view("language")
        raw = b"aba b aba"
        expected = {}
        for span in self.tokenizer.encode(raw)["spans"]:
            piece = raw[span["start"]:span["end"]]
            expected[piece.hex()] = expected.get(piece.hex(), 0) + 1
        self.assertEqual({p["bytes_hex"]: p["uses"] for p in view["pieces"]}, expected)
        self.assertEqual(sum(f["count"] for p in view["pieces"] for f in p["followers"]), len(first["encoded"]["spans"]) - 1)
        self.assertTrue(all(len(p["barcode"]) == 137 for p in view["pieces"]))
        self.memory.put_source("unrelated", "A different observation.")
        self.assertTrue(owner.view("language")["observations"][0]["current"])
        self.memory.put_source("words", "baba", expected_revision=1, note="Correction")
        current = owner.view("language")
        self.assertEqual(current["pieces"], [])
        self.assertFalse(current["observations"][0]["current"])
        self.assertEqual(len(owner.view("language", include_history=True)["pieces"]), len(expected))
        with self.assertRaises(ValueError):
            owner.observe("read-1", "language", "words", self.tokenizer)
        owner.observe("read-2", "language", "words", self.tokenizer)
        restored = Memory(self.root / "journal")
        source = restored.state["sources"]["words"]
        snapshot = restored.snapshot("language", [source], "future-core")
        entries = Harmonizer(restored).view(snapshot)["entries"]
        self.assertEqual([r["id"] for r in entries if r["kind"] == "language_observation"], ["read-2"])
        code = "from nanoai.memory import Memory; from nanoai.catalogue import LanguageCatalogue; import json,sys; print(json.dumps(LanguageCatalogue(Memory(sys.argv[1])).view('language')))"
        fresh = json.loads(subprocess.check_output([sys.executable, "-c", code, str(self.root / "journal")], text=True))
        self.assertEqual(fresh, owner.view("language"))

    @unittest.skipUnless(os.environ.get("NANOAI_SEE_EXECUTABLE"), "SEE port requires an explicitly selected native engine")
    def test_real_see_bundle_mask_and_stale_reference(self):
        executable = os.environ["NANOAI_SEE_EXECUTABLE"]
        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        pixels = [(65535, 0, 0, 65535), (65535, 0, 0, 65535), (0, 12345, 65535, 0)] * 2
        scan = b"".join(b"\x00" + b"".join(struct.pack(">4H", *p) for p in pixels[y*3:y*3+3]) for y in range(2))
        raw = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 3, 2, 16, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(scan)) + chunk(b"IEND", b"")
        image = self.root / "source.png"; image.write_bytes(raw)
        out = self.root / "see"
        out.mkdir()
        subprocess.run([executable, "decode", str(image), "-o", str(out)], capture_output=True, check=True)
        bundle = out / "source.see.sqlite"
        with sqlite3.connect(bundle) as database:
            root = database.execute("SELECT id FROM region WHERE parent IS NULL").fetchone()[0]
        record = attach_perception(self.memory, "frame-1", "language", "words", bundle=bundle,
                                   executable=executable, region_ids=[root], source_image=image)
        measured = record["measurement"]
        self.assertEqual(measured["source"]["depth"], 16)
        self.assertEqual(measured["source"]["file_digest"], digest_bytes(raw))
        self.assertEqual(measured["regions"][0]["area"], 6)
        self.assertEqual(measured["regions"][0]["box"], [0, 0, 3, 2])
        native_mask = measured["regions"][0]["mask_identity"]
        # Independent installed SP800-185 implementation; packed bits cross rows
        # without PNG's row padding, so the six-pixel root has one byte 0xfc.
        code = "from Crypto.Hash import TupleHash256; h=TupleHash256.new(custom=b'see.mask/1',digest_bytes=64); h.update(b'3x2',b'\\xfc'); print('tuplehash256(see.mask/1):'+h.hexdigest())"
        reference_python = os.environ.get("NANOAI_TUPLEHASH_PYTHON", sys.executable)
        expected = subprocess.check_output([reference_python, "-c", code], text=True).strip()
        self.assertEqual(native_mask, expected)
        self.assertEqual(LanguageCatalogue(Memory(self.root / "journal")).view("language")["perceptions"][0]["measurement"], measured)
        with sqlite3.connect(bundle) as database:
            database.execute("UPDATE region SET area=area+1 WHERE id=?", (root,))
        view = LanguageCatalogue(Memory(self.root / "journal")).view("language", include_history=True)
        self.assertFalse(view["perceptions"][0]["current"])
        with self.assertRaises(ValueError):
            read_see_bundle(bundle, executable, [root])


if __name__ == "__main__":
    unittest.main()
