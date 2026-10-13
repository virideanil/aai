"""Finite byte unigram law over selected System Zero IDs.

The source inventory stays cold. This immutable active vocabulary provides exact
byte spans, alternative segmentations and a bounded embedding-row contract.
Existing pretrained models continue to consume their complete native tokenizer.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random
import struct
from types import MappingProxyType

from .identity import digest_bytes


MAX_INPUT = 131072
MAX_PIECE = 128
MAX_ACTIVE = 131072
MAX_WORK = 16_777_216
HEADER = struct.Struct("<4sId")
ENTRY = struct.Struct("<IHd")


def logadd(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    high, low = (a, b) if a >= b else (b, a)
    return high + math.log1p(math.exp(low - high))


def _identity_bytes(value):
    if not isinstance(value, str) or not value.startswith("shake256:") or len(value) != 137:
        raise ValueError("Master identity requires SHAKE256 with 64 bytes")
    try:
        raw = bytes.fromhex(value[9:])
    except ValueError as error:
        raise ValueError("Master identity encoding") from error
    if len(raw) != 64:
        raise ValueError("Master identity extent")
    return raw


@dataclass(frozen=True, init=False, eq=False)
class MasterTokenizer:
    """A snapshot; active slots and common IDs are distinct explicit fields.

    Pieces and common IDs have byte-lexicographic order. Exact Viterbi ties use
    the shorter final piece, recursively. All paths use the same finite law.
    """

    def __init__(self, pieces, common_ids, probabilities, p_end, vocabulary_identity, inventory_identity):
        pieces, common_ids, probabilities = tuple(pieces), tuple(common_ids), tuple(probabilities)
        if not 256 <= len(pieces) <= MAX_ACTIVE or len(common_ids) != len(pieces) or len(probabilities) != len(pieces):
            raise ValueError("Master vocabulary extent")
        if any(not isinstance(p, bytes) or not 1 <= len(p) <= MAX_PIECE for p in pieces):
            raise ValueError("Master piece extent")
        if pieces != tuple(sorted(set(pieces))) or not {bytes([b]) for b in range(256)}.issubset(pieces):
            raise ValueError("Master ordering or complete byte fallback")
        if any(type(i) is not int or not 0 <= i < 2_000_000 for i in common_ids) or common_ids != tuple(sorted(set(common_ids))):
            raise ValueError("Master common IDs")
        if any(not math.isfinite(q) or q <= 0 for q in probabilities) or abs(math.fsum(probabilities) - 1) > 1e-10:
            raise ValueError("Master emission probabilities must sum to one")
        if not math.isfinite(p_end) or not 0 < p_end < 1:
            raise ValueError("Master stop probability")
        _identity_bytes(vocabulary_identity); _identity_bytes(inventory_identity)
        fields = dict(pieces=pieces, common_ids=common_ids, probabilities=probabilities, p_end=float(p_end),
                      vocabulary_identity=vocabulary_identity, inventory_identity=inventory_identity,
                      slots=MappingProxyType({p: i for i, p in enumerate(pieces)}),
                      common_to_slot=MappingProxyType({c: i for i, c in enumerate(common_ids)}),
                      logq=tuple(map(math.log, probabilities)), emit=math.log1p(-p_end),
                      max_length=max(map(len, pieces)))
        for key, value in fields.items():
            object.__setattr__(self, key, value)
        object.__setattr__(self, "identity", digest_bytes(self._body()))

    def _body(self):
        body = bytearray(HEADER.pack(b"NMT1", len(self.pieces), self.p_end))
        body += _identity_bytes(self.vocabulary_identity) + _identity_bytes(self.inventory_identity)
        for common_id, piece, q in zip(self.common_ids, self.pieces, self.probabilities):
            body += ENTRY.pack(common_id, len(piece), q) + piece
        return bytes(body)

    def save(self, path):
        body = self._body()
        with Path(path).open("xb") as stream:
            stream.write(body + hashlib.shake_256(body).digest(64))
        return self.identity

    @classmethod
    def load(cls, path, *, expected_identity=None):
        path = Path(path)
        if not HEADER.size + 192 <= path.stat().st_size <= 24_000_000:
            raise ValueError("Master container extent")
        data = path.read_bytes()
        body, checksum = data[:-64], data[-64:]
        if hashlib.shake_256(body).digest(64) != checksum:
            raise ValueError("Master payload identity differs")
        identity = digest_bytes(body)
        if expected_identity is not None and identity != expected_identity:
            raise ValueError("Master artifact version differs")
        magic, count, stop = HEADER.unpack_from(body)
        if magic != b"NMT1" or not 256 <= count <= MAX_ACTIVE:
            raise ValueError("Master header")
        cursor = HEADER.size
        vocabulary = "shake256:" + body[cursor:cursor + 64].hex(); cursor += 64
        inventory = "shake256:" + body[cursor:cursor + 64].hex(); cursor += 64
        pieces, ids, probabilities = [], [], []
        for _ in range(count):
            if cursor + ENTRY.size > len(body):
                raise ValueError("Truncated master record")
            common_id, size, q = ENTRY.unpack_from(body, cursor); cursor += ENTRY.size
            if not 1 <= size <= MAX_PIECE or cursor + size > len(body):
                raise ValueError("Truncated master piece")
            pieces.append(body[cursor:cursor + size]); cursor += size
            ids.append(common_id); probabilities.append(q)
        if cursor != len(body):
            raise ValueError("Master trailing bytes")
        return cls(pieces, ids, probabilities, stop, vocabulary, inventory)

    def lattice(self, raw):
        if not isinstance(raw, bytes) or len(raw) > MAX_INPUT or len(raw) * self.max_length > MAX_WORK:
            raise ValueError("Master input or work extent")
        ends = [[] for _ in range(len(raw) + 1)]
        for end in range(1, len(raw) + 1):
            for size in range(1, min(self.max_length, end) + 1):
                slot = self.slots.get(raw[end - size:end])
                if slot is not None:
                    ends[end].append((end - size, slot))
        return ends

    def _forward(self, lattice):
        fwd = [-math.inf] * len(lattice); fwd[0] = 0.0
        for end in range(1, len(lattice)):
            for start, slot in lattice[end]:
                fwd[end] = logadd(fwd[end], fwd[start] + self.emit + self.logq[slot])
        return fwd

    def encode(self, raw):
        lattice = self.lattice(raw)
        fwd = self._forward(lattice)
        best = [-math.inf] * len(lattice); best[0] = 0.0
        back = [None] * len(lattice)
        for end in range(1, len(lattice)):
            for start, slot in lattice[end]:
                candidate = best[start] + self.emit + self.logq[slot]
                if candidate > best[end]:
                    best[end], back[end] = candidate, (start, slot)
        spans, cursor = [], len(raw)
        while cursor:
            start, slot = back[cursor]
            spans.append({"id": self.common_ids[slot], "active_id": slot, "start": start, "end": cursor})
            cursor = start
        spans.reverse()
        return {"schema": "nanoai.master-tokens.v1", "tokenizer_identity": self.identity,
                "vocabulary_identity": self.vocabulary_identity, "byte_identity": digest_bytes(raw),
                "spans": spans, "forward_logp": fwd[-1] + math.log(self.p_end),
                "viterbi_logp": best[-1] + math.log(self.p_end)}

    def decode(self, encoded):
        if (not isinstance(encoded, dict) or encoded.get("schema") != "nanoai.master-tokens.v1" or encoded.get("tokenizer_identity") != self.identity
                or encoded.get("vocabulary_identity") != self.vocabulary_identity):
            raise ValueError("Master token record identity")
        if not isinstance(encoded.get("spans"), list) or len(encoded["spans"]) > MAX_INPUT:
            raise ValueError("Master token record extent")
        parts, cursor = [], 0
        for row in encoded["spans"]:
            if not isinstance(row, dict) or any(type(row.get(k)) is not int for k in ("id", "active_id", "start", "end")):
                raise ValueError("Master token fields require integers")
            slot = row["active_id"]
            if type(slot) is not int or not 0 <= slot < len(self.pieces) or row["id"] != self.common_ids[slot]:
                raise ValueError("Master token ID")
            piece = self.pieces[slot]
            if row["start"] != cursor or row["end"] != cursor + len(piece):
                raise ValueError("Master token span")
            parts.append(piece); cursor += len(piece)
            if cursor > MAX_INPUT:
                raise ValueError("Master decoded extent")
        raw = b"".join(parts)
        if digest_bytes(raw) != encoded["byte_identity"]:
            raise ValueError("Master decoded bytes differ")
        return raw

    def sample(self, raw, rng: random.Random):
        lattice = self.lattice(raw); fwd = self._forward(lattice)
        slots, end = [], len(raw)
        while end:
            choices = lattice[end]
            weights = [math.exp(fwd[start] + self.emit + self.logq[slot] - fwd[end]) for start, slot in choices]
            draw = rng.random() * math.fsum(weights)
            chosen = choices[-1]
            for choice, weight in zip(choices, weights):
                draw -= weight
                if draw <= 0:
                    chosen = choice; break
            end, slot = chosen; slots.append(slot)
        return tuple(reversed(slots))

    def expectation(self, lattice):
        """Expected piece counts for lexical EM; no neural parameter updates."""
        fwd = self._forward(lattice)
        backward = [-math.inf] * len(lattice); backward[-1] = 0.0
        counts = [0.0] * len(self.pieces)
        for end in range(len(lattice) - 1, 0, -1):
            for start, slot in lattice[end]:
                edge = self.emit + self.logq[slot]
                backward[start] = logadd(backward[start], edge + backward[end])
                counts[slot] += math.exp(fwd[start] + edge + backward[end] - fwd[-1])
        return counts, fwd[-1] + math.log(self.p_end)

    def with_source_scalars(self, common, *, reserve=.02):
        """Preserve the source-attested Unicode alphabet outside calibration.

        The finite common dictionary owns every added ID. Reserve is an explicit
        lexical prior for otherwise absent scalars; it is not a learned semantic
        probability or an estimate copied from native token IDs.
        """
        if common.identity != self.vocabulary_identity or common.inventory_identity != self.inventory_identity:
            raise ValueError("Scalar source dictionary identity")
        if not math.isfinite(reserve) or not 0 < reserve < 1:
            raise ValueError("Scalar prior mass")
        added = []
        for common_id, piece in enumerate(common.pieces):
            if len(piece) > 4 or common_id in self.common_to_slot:
                continue
            try:
                scalar = piece.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                continue
            if len(scalar) == 1:
                added.append(common_id)
        if not added:
            return self
        ids = sorted(set(self.common_ids) | set(added))
        if len(ids) > MAX_ACTIVE:
            raise ValueError("Protected alphabet exceeds the active vocabulary budget")
        q = {i: (1 - reserve) * value for i, value in zip(self.common_ids, self.probabilities)}
        q.update({i: reserve / len(added) for i in added})
        return type(self)([common.pieces[i] for i in ids], ids, [q[i] for i in ids], self.p_end,
                               self.vocabulary_identity, self.inventory_identity)
