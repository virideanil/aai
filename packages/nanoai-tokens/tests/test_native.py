"""Independent path enumeration and real native boundary consumers."""
from concurrent.futures import ThreadPoolExecutor
import copy
from decimal import Decimal, localcontext
import hashlib
import itertools
from fractions import Fraction
import random
import struct
import tempfile
from pathlib import Path
import unittest

from nanoai_tokens import Tokenizer, frozen_path, load
from nanoai_tokens import _native
from nanoai_tokens.identity import digest_bytes
from nanoai_tokens.reference import MasterTokenizer as Reference


def fixture():
    pieces = sorted({bytes([i]) for i in range(256)} | {b"aa", b"ab", b"ba", b"aba"})
    weights = [17 if p in (b"a", b"b") else 11 if len(p) > 1 else 1 for p in pieces]
    return Tokenizer(pieces, range(len(pieces)), [w/sum(weights) for w in weights], .2,
                     digest_bytes(b"dictionary"), digest_bytes(b"inventory"))


def paths(raw, codec, offset=0, boundaries=()):
    if not raw:
        return [()]
    result = []
    for size in range(1, len(raw)+1):
        if raw[:size] not in codec.slots or any(offset < b < offset+size for b in boundaries):
            continue
        result.extend((codec.slots[raw[:size]],)+rest for rest in paths(raw[size:],codec,offset+size,boundaries))
    return result


def probability(codec, path):
    stop = Decimal.from_float(codec.p_end)
    value = stop
    for slot in path:
        value *= (1-stop)*Decimal.from_float(codec.probabilities[slot])
    return value


def exact_probability(codec, path):
    stop = Fraction(codec.p_end)
    value = stop
    for slot in path:
        value *= (1-stop)*Fraction(codec.probabilities[slot])
    return value


class NativeTests(unittest.TestCase):
    def test_complete_law_minimum_and_boundaries(self):
        codec = fixture()
        with localcontext() as context:
            context.prec = 60
            for n in range(7):
                for letters in itertools.product(b"ab", repeat=n):
                    raw = bytes(letters)
                    for boundaries in ((), tuple(range(1,n,2))):
                        options = paths(raw,codec,boundaries=boundaries)
                        weights = [probability(codec,p) for p in options]
                        best = max(weights); total = sum(weights); shortest = min(map(len,options))
                        for objective in ("likelihood", "minimum"):
                            record = codec.encode(raw,objective=objective,boundaries=boundaries)
                            actual = tuple(s["active_id"] for s in record["spans"])
                            self.assertIn(actual,options)
                            self.assertEqual(codec.decode(record),raw)
                            self.assertAlmostEqual(record["forward_logp"],float(total.ln()),places=11)
                            self.assertAlmostEqual(record["viterbi_logp"],float(best.ln()),places=11)
                            if objective == "minimum":
                                self.assertEqual(len(actual),shortest)
                                self.assertEqual(exact_probability(codec,actual),max(exact_probability(codec,p) for p in options if len(p)==shortest))

    def test_native_expectation_matches_independent_decimal(self):
        codec = fixture()
        for raw in (b"",b"a",b"aba",b"ababaa"):
            options=paths(raw,codec); weights=[probability(codec,p) for p in options]; total=sum(weights)
            actual, logp=codec.expected_counts(raw)
            for i in range(len(codec.pieces)):
                expected=sum(w*p.count(i) for p,w in zip(options,weights))/total
                self.assertAlmostEqual(actual[i],float(expected),places=11)
            self.assertAlmostEqual(logp,float(total.ln()),places=11)

    def test_full_artifact_reference_and_parallel_calls(self):
        codec=load(); reference=Reference.load(frozen_path()); rng=random.Random(826)
        inputs=[b"",bytes(range(256)),"é é İ 🧬 雨".encode(),b"\x00\xff\xfe"]+[rng.randbytes(i*17) for i in range(16)]
        for raw in inputs:
            native=codec.encode(raw); expected=reference.encode(raw)
            self.assertEqual(native['spans'],expected['spans'])
            self.assertEqual(codec.decode(native),raw)
            for key in ('forward_logp','viterbi_logp'):
                self.assertAlmostEqual(native[key],expected[key],places=10)
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual=list(pool.map(codec.encode,inputs*4))
        self.assertEqual(actual,[codec.encode(raw) for raw in inputs]*4)

    def test_version_ownership_and_limits(self):
        codec=load(); raw=b'x'*131072
        self.assertEqual(codec.decode(codec.encode(raw)),raw)
        with self.assertRaises(ValueError): codec.encode(raw+b'x')
        with self.assertRaises(AttributeError): codec.p_end=.1
        with self.assertRaises(AttributeError): del codec._handle
        for boundaries in ((0,),(-1,),(2,1),(1,1),(True,),(3,)):
            with self.assertRaises(ValueError): codec.encode(b'abc',boundaries=boundaries)
        record=codec.encode(b'ab'); corrupted=copy.deepcopy(record); corrupted['spans'][0]['id']+=1
        with self.assertRaises(ValueError): codec.decode(corrupted)

    def test_actual_native_parser_rejects_valid_checksum_malformed(self):
        original=frozen_path().read_bytes()
        for offset, fmt, value in [(8,'d',0),(8,'d',float('nan')),(150,'d',.5),(144,'I',2_000_000),(159,'I',0),(4,'I',131073)]:
            body=bytearray(original[:-64]); struct.pack_into('<'+fmt,body,offset,value)
            with self.assertRaises(ValueError): _native.create(bytes(body)+hashlib.shake_256(body).digest(64))

    def test_lexical_fit_is_immutable_and_reloads(self):
        codec=fixture(); original=codec.identity; before=codec.encode(b'aba')['forward_logp']
        candidate=codec.fit([b'aba',b'abaaba',b'ababa'],epochs=3,prior=.001)
        self.assertEqual(codec.identity,original)
        self.assertEqual(codec.pieces,candidate.pieces)
        self.assertGreater(candidate.encode(b'aba')['forward_logp'],before)
        self.assertNotEqual(candidate.identity,original)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'candidate.nmt'; candidate.save(path)
            restored=Tokenizer.load(path,expected_identity=candidate.identity)
            self.assertEqual(candidate.encode(b'ababa'),restored.encode(b'ababa'))
        for rows in ([],[b''],['text']):
            with self.assertRaises(ValueError): codec.fit(rows)


if __name__ == '__main__': unittest.main()
