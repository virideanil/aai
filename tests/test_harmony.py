import importlib.util
import os
from pathlib import Path
import struct
import tempfile
import unittest

from nanoai.harmony import Library, Matrix, MODES


@unittest.skipUnless(os.environ.get("NANOAI_HARMONY_LIB"), "Native library is an explicit dependency")
class HarmonyConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np = np
        cls.library = Library(os.environ["NANOAI_HARMONY_LIB"])

    def test_every_bf16_word_and_parallel_decode(self):
        words = self.np.arange(65536, dtype=self.np.uint16).reshape(256, 256)
        for mode in MODES:
            with Matrix(self.library, words, mode) as matrix:
                for threads in [1, 4]:
                    self.np.testing.assert_array_equal(words, matrix.unpack(threads))

    def test_mixed_selection_uses_both_forms(self):
        normal = ((120 + self.np.arange(8192, dtype=self.np.uint16) % 8) << 7) | 64
        noise = self.np.arange(8192, dtype=self.np.uint16)
        with Matrix(self.library, self.np.stack([normal, noise]), "mixed") as matrix:
            s = matrix.stats()
            self.assertEqual((s["binary_blocks"], s["ternary_exponent_byte_blocks"]), (1, 1))
            self.np.testing.assert_array_equal(matrix.unpack(), self.np.stack([normal, noise]))

    def test_projection_matches_independent_oracle_and_all_routes(self):
        np = self.np
        rng = np.random.default_rng(731)
        weights = rng.normal(size=(173, 257)).astype(np.float32)
        words = (weights.view(np.uint32) >> 16).astype(np.uint16)
        exact = (words.astype(np.uint32) << 16).view(np.float32)
        for batch in [1, 7, 32]:
            x = rng.normal(size=(batch, 257)).astype(np.float32)
            x = ((x.view(np.uint32) >> 16) << 16).view(np.float32)
            oracle = x.astype(np.float64) @ exact.astype(np.float64).T
            absolute_sum = np.abs(x.astype(np.float64)) @ np.abs(exact.astype(np.float64)).T
            bound = 2 * np.finfo(np.float32).eps * 257 * absolute_sum + 1e-8
            reference = None
            for mode in MODES:
                with Matrix(self.library, words, mode) as matrix:
                    for threads in [1, 2, 4]:
                        y = matrix.project(x, threads=threads)
                        self.assertTrue(np.all(np.abs(y - oracle) <= bound))
                        if reference is None: reference = y
                        np.testing.assert_array_equal(reference, y)
                    matrix.prepare_cache(4)
                    y = matrix.project(x, cached=True)
                    self.assertTrue(np.all(np.abs(y - oracle) <= bound))
                    np.testing.assert_array_equal(reference, y)
                    full = matrix.project(x, dense_baseline=True)
                    self.assertTrue(np.all(np.abs(full - oracle) <= bound))
                    self.assertEqual(matrix.stats()["decoded_cache_bytes"], words.size * 4)

    def test_closed_and_nonfinite_compute_paths_reject(self):
        np = self.np
        with Matrix(self.library, np.array([[0x7fc1]], dtype=np.uint16)) as matrix:
            with self.assertRaisesRegex(ValueError, "finite weights"):
                matrix.project(np.zeros((1, 1), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "closed"):
            matrix.unpack()
        with Matrix(self.library, np.array([[0x3f80]], dtype=np.uint16)) as matrix:
            with self.assertRaisesRegex(ValueError, "not been prepared"):
                matrix.project(np.zeros((1, 1), dtype=np.float32), cached=True)

    def test_file_reconstruction_and_malformed_input(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for index, data in enumerate([b"", b"x", self.np.arange(65536, dtype="<u2").tobytes() + b"tailx"]):
                source, packed, restored = [root / f"{index}.{s}" for s in ["raw", "nbh", "restored"]]
                source.write_bytes(data)
                self.library.pack_file(source, packed)
                self.library.unpack_file(packed, restored)
                self.assertEqual(restored.read_bytes(), data)
                with self.assertRaisesRegex(ValueError, "new output"):
                    self.library.unpack_file(packed, restored)
                self.assertEqual(restored.read_bytes(), data)
            source = root / "normal.raw"; packed = root / "normal.nbh"
            source.write_bytes(struct.pack("<128H", *([0x3f80] * 128)))
            self.library.pack_file(source, packed, "binary_trit5")
            good = packed.read_bytes()
            changes = [("magic", b"X" + good[1:]), ("short", good[:-1]), ("trailing", good + b"x"),
                ("kind", good[:16] + b"\xff" + good[17:]),
                ("count", good[:17] + b"\x00\x00" + good[19:]),
                ("trit", good[:-1] + b"\xff")]
            for name, data in changes:
                bad = root / f"{name}.nbh"; bad.write_bytes(data)
                with self.assertRaises(ValueError, msg=name):
                    self.library.unpack_file(bad, root / f"{name}.result")

    @unittest.skipUnless(os.environ.get("NANOAI_BF5P_REFERENCE"), "Independent existing BF5P encoder")
    def test_compatibility_with_existing_bf5p_payload(self):
        spec = importlib.util.spec_from_file_location("bf5p_reference", os.environ["NANOAI_BF5P_REFERENCE"])
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        words = ([0, 0x8000, 0x7fc1, 0xffff, 0x3f80, 0x3fff, 0xbf00] * 97) + list(range(0x3e80, 0x3fff))
        with tempfile.TemporaryDirectory() as d:
            source, packed = Path(d) / "source", Path(d) / "packed"
            source.write_bytes(struct.pack("<" + "H" * len(words), *words))
            self.library.pack_file(source, packed, "binary_trit5")
            # One block: NBH1 has a 16-byte file and 7-byte block header;
            # BF5P has a 28-byte file and one 8-byte block offset.
            self.assertEqual(packed.read_bytes()[23:], module.encode(words)[36:])


if __name__ == "__main__":
    unittest.main()
