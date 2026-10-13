"""Independent enumeration checks the finite law, including ambiguous paths."""
import copy
from decimal import Decimal, localcontext
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from nanoai.core import digest_bytes
from nanoai.master_tokenizer import MasterTokenizer, MAX_INPUT


def fixture():
    pieces = sorted({bytes([i]) for i in range(256)} | {b"aa", b"ab", b"ba", b"aba"})
    weights = [17 if p in (b"a", b"b") else 11 if len(p) > 1 else 1 for p in pieces]
    return MasterTokenizer(pieces, list(range(len(pieces))), [w / sum(weights) for w in weights],
                           .2, digest_bytes(b"common"), digest_bytes(b"inventory"))


def paths(raw, tokenizer):
    """Enumerate by cutting the byte string; never call the runtime lattice."""
    if not raw:
        return [()]
    result = []
    for size in range(1, len(raw) + 1):
        if raw[:size] in tokenizer.slots:
            slot = tokenizer.slots[raw[:size]]
            result.extend((slot,) + rest for rest in paths(raw[size:], tokenizer))
    return result


def law(path, tokenizer):
    stop = Decimal.from_float(tokenizer.p_end)
    weight = stop
    for slot in path:
        weight *= (1 - stop) * Decimal.from_float(tokenizer.probabilities[slot])
    return weight


class MasterTokenizerTests(unittest.TestCase):
    def test_exhaustive_oracle_and_expected_counts(self):
        tokenizer = fixture()
        with localcontext() as ctx:
            ctx.prec = 60
            for size in range(7):
                for letters in itertools.product(b"ab", repeat=size):
                    raw = bytes(letters)
                    alternatives = paths(raw, tokenizer)
                    weights = [law(path, tokenizer) for path in alternatives]
                    total = sum(weights)
                    encoded = tokenizer.encode(raw)
                    self.assertEqual(tokenizer.decode(encoded), raw)
                    self.assertAlmostEqual(encoded["forward_logp"], float(total.ln()), places=11)
                    self.assertAlmostEqual(encoded["viterbi_logp"], float(max(weights).ln()), places=11)
                    counts, logp = tokenizer.expectation(tokenizer.lattice(raw))
                    self.assertAlmostEqual(logp, encoded["forward_logp"], places=12)
                    for slot in {s for p in alternatives for s in p}:
                        expected = sum(w * p.count(slot) for p, w in zip(alternatives, weights)) / total
                        self.assertAlmostEqual(counts[slot], float(expected), places=11)

    def test_conditional_sampling_matches_enumerated_distribution(self):
        tokenizer = fixture()
        raw = b"aba"
        alternatives = paths(raw, tokenizer)
        weights = [float(law(p, tokenizer)) for p in alternatives]
        counts = dict.fromkeys(alternatives, 0)
        rng = random.Random(192)
        for _ in range(16000):
            counts[tokenizer.sample(raw, rng)] += 1
        for path, weight in zip(alternatives, weights):
            self.assertLess(abs(counts[path] / 16000 - weight / sum(weights)), .013)

    def test_every_byte_unicode_and_binary_round_trip(self):
        tokenizer = fixture()
        rng = random.Random(71)
        for raw in [b"", bytes(range(256)), "İ é café 🧬\r\n\t　".encode(),
                    b"\xff\xc0\x80\x00\"\\\n", rng.randbytes(5000)]:
            self.assertEqual(tokenizer.decode(tokenizer.encode(raw)), raw)
        with self.assertRaises(ValueError):
            tokenizer.encode(b"a" * (MAX_INPUT + 1))

    def test_snapshot_and_serialized_artifact_are_immutable(self):
        tokenizer = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "master.nmt"
            tokenizer.save(path)
            restored = MasterTokenizer.load(path, expected_identity=tokenizer.identity)
            self.assertEqual(restored.encode(b"aba"), tokenizer.encode(b"aba"))
            with self.assertRaises(FileExistsError):
                tokenizer.save(path)
            with self.assertRaises(ValueError):
                MasterTokenizer.load(path, expected_identity=digest_bytes(b"foreign"))
            raw = path.read_bytes()
            path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
            with self.assertRaises(ValueError):
                MasterTokenizer.load(path)
            body = raw[:-64] + b"x"
            path.write_bytes(body + hashlib.shake_256(body).digest(64))
            with self.assertRaises(ValueError):
                MasterTokenizer.load(path)
        with self.assertRaises(TypeError):
            tokenizer.slots[b"new"] = 1
        with self.assertRaises(AttributeError):
            tokenizer.p_end = .3

    def test_foreign_tokens_malformed_types_spans_and_hash_reject(self):
        tokenizer = fixture()
        baseline = tokenizer.encode(b"aba")
        for field, value in [("tokenizer_identity", digest_bytes(b"other")), ("byte_identity", "x"),
                             ("schema", "other"), ("vocabulary_identity", "x")]:
            row = copy.deepcopy(baseline); row[field] = value
            with self.assertRaises(ValueError):
                tokenizer.decode(row)
        for field, value in [("active_id", -1), ("active_id", True), ("id", 9999),
                             ("start", .0), ("end", 999), ("id", float(baseline["spans"][0]["id"]))]:
            row = copy.deepcopy(baseline); row["spans"][0][field] = value
            with self.assertRaises(ValueError):
                tokenizer.decode(row)

    def test_law_and_fallback_reject_invalid_snapshots(self):
        t = fixture()
        values = dict(pieces=t.pieces, common_ids=t.common_ids, probabilities=t.probabilities,
                      p_end=t.p_end, vocabulary_identity=t.vocabulary_identity, inventory_identity=t.inventory_identity)
        for key, bad in [("p_end", 0), ("p_end", float("nan")), ("probabilities", [1] * len(t.pieces)),
                         ("pieces", t.pieces[1:]), ("common_ids", [0] * len(t.pieces)),
                         ("vocabulary_identity", "x")]:
            with self.assertRaises(ValueError):
                MasterTokenizer(**{**values, key: bad})

    def test_protected_scalar_prior_has_exact_mass_and_source_ids(self):
        source_pieces = sorted(set(fixture().pieces) | {"雨".encode(), "🧬".encode(), b"\xff\xfe", b"unused"})
        all_ids = {piece: i for i, piece in enumerate(source_pieces)}
        base = fixture()
        base = MasterTokenizer(base.pieces, [all_ids[p] for p in base.pieces], base.probabilities, base.p_end,
                               base.vocabulary_identity, base.inventory_identity)
        common = SimpleNamespace(pieces=source_pieces, identity=base.vocabulary_identity, inventory_identity=base.inventory_identity)
        protected = base.with_source_scalars(common, reserve=.02)
        self.assertAlmostEqual(sum(protected.probabilities), 1, places=13)
        for piece in ("雨".encode(), "🧬".encode()):
            self.assertAlmostEqual(protected.probabilities[protected.slots[piece]], .01)
            self.assertEqual(protected.encode(piece)["spans"][0]["id"], all_ids[piece])
            self.assertEqual(len(protected.encode(piece)["spans"]), 1)
        self.assertNotIn(b"\xff\xfe", protected.slots)
        self.assertNotIn(b"unused", protected.slots)
        self.assertIs(protected.with_source_scalars(common), protected)
        common.identity = "foreign"
        with self.assertRaises(ValueError):
            base.with_source_scalars(common)

    @unittest.skipUnless(os.environ.get("NANOAI_MASTER_NATIVE"), "Native consumer is qualified with an explicit executable")
    def test_native_oracle_parity_and_rejection(self):
        tokenizer = fixture()
        native = str(Path(os.environ["NANOAI_MASTER_NATIVE"]).resolve())
        rng = random.Random(918)
        rows = [bytes(chars) for size in range(7) for chars in itertools.product(b"ab", repeat=size)]
        rows += [bytes(range(256)), b"\\x41\x00\xff\n", "İ é 🧬".encode()]
        rows += [rng.randbytes(rng.randrange(1024)) for _ in range(60)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "master.nmt"
            tokenizer.save(path)
            response = subprocess.run([native, str(path), tokenizer.identity],
                input="".join(row.hex() + "\n" for row in rows), text=True, capture_output=True, timeout=30, check=True)
            actual = [json.loads(row) for row in response.stdout.splitlines()]
            self.assertEqual(len(actual), len(rows))
            for row, encoded in zip(rows, actual):
                expected = tokenizer.encode(row)
                self.assertEqual(encoded["spans"], expected["spans"])
                self.assertEqual(tokenizer.decode(encoded), row)
                self.assertEqual(encoded["decoded_hex"], row.hex())
                self.assertAlmostEqual(encoded["forward_logp"], expected["forward_logp"], places=9)
                self.assertAlmostEqual(encoded["viterbi_logp"], expected["viterbi_logp"], places=9)
            for value in ("0\n", "zz\n", "a" * (2 * MAX_INPUT + 1) + "\n"):
                bad = subprocess.run([native, str(path)], input=value, text=True, capture_output=True, timeout=10)
                self.assertEqual(bad.returncode, 2)
                self.assertEqual(bad.stdout, "")
            missing = subprocess.run([native, str(path) + ".missing"], capture_output=True, timeout=10)
            self.assertEqual(missing.returncode, 2)
            foreign = subprocess.run([native, str(path), digest_bytes(b"foreign")], capture_output=True, timeout=10)
            self.assertEqual(foreign.returncode, 2)
            original = path.read_bytes()
            bodies = [original[:-65], original[:-64] + b"x", b"NOPE" + original[4:-64],
                      original[:8] + struct.pack("<d", float("nan")) + original[16:-64],
                      original[:4] + struct.pack("<I", 131073) + original[8:-64],
                      original[:144] + struct.pack("<I", 2000000) + original[148:-64]]
            for body in bodies:
                path.write_bytes(body + hashlib.shake_256(body).digest(64))
                bad = subprocess.run([native, str(path)], input="61\n", text=True, capture_output=True, timeout=10)
                self.assertEqual(bad.returncode, 2)
                self.assertEqual(bad.stdout, "")
                with self.assertRaises(ValueError):
                    MasterTokenizer.load(path)


if __name__ == "__main__":
    unittest.main()
