"""System Zero: exact byte surfaces with source-specific vocabulary witnesses.

A common surface ID identifies bytes. Native IDs, control tokens, embeddings
and probabilities keep their model namespace. This layer performs no learning.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct

from .core import digest_bytes, file_digest


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def byte_alphabet():
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    result = {chr(value): value for value in values}
    extra = 0
    for value in range(256):
        if value not in values:
            result[chr(256 + extra)] = value
            extra += 1
    return result


def read_kimi_pattern(path):
    """Read the maintained literal regex; never import downloaded Python code."""
    module = ast.parse(path.read_text())
    owner = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "TikTokenTokenizer")
    fields = {target.id: node.value for node in owner.body if isinstance(node, ast.Assign)
              for target in node.targets if isinstance(target, ast.Name)}
    expression = fields["pat_str"]
    if (not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Attribute)
            or expression.func.attr != "join" or ast.literal_eval(expression.func.value) != "|"
            or len(expression.args) != 1 or expression.keywords):
        raise ValueError("The source tokenizer pattern requires a new adapter review")
    pieces = ast.literal_eval(expression.args[0])
    if not isinstance(pieces, list) or not all(isinstance(x, str) for x in pieces):
        raise ValueError("The tokenizer pattern must be literal text")
    reserved = ast.literal_eval(fields["num_reserved_special_tokens"])
    if type(reserved) is not int or not 1 <= reserved <= 4096:
        raise ValueError("Reserved token extent")
    return "|".join(pieces), reserved


class VocabularySource:
    """Two explicitly supported data adapters, with exact source provenance."""
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        api = json.loads((self.root / "hf-api.json").read_text())
        self.config = json.loads((self.root / "tokenizer_config.json").read_text())
        self.surfaces, self.controls = {}, {}
        names = ["tokenizer_config.json", "LICENSE", "hf-api.json"]
        if (self.root / "tokenizer.json").exists():
            from tokenizers import Tokenizer
            names += ["tokenizer.json"]
            data = json.loads((self.root / "tokenizer.json").read_text())
            if data["model"]["type"] != "BPE" or data["decoder"]["type"] != "ByteLevel":
                raise ValueError("This source needs a reviewed vocabulary adapter")
            alphabet = byte_alphabet()
            for token, native_id in data["model"]["vocab"].items():
                self.surfaces[native_id] = bytes(alphabet[char] for char in token)
            self.controls = {row["id"]: dict(row) for row in data["added_tokens"]}
            # Source text is literal data. Chat/control records use their own
            # model template; importing text must not introduce role IDs.
            data["added_tokens"] = []
            self.tokenizer = Tokenizer.from_str(json.dumps(data))
            self.adapter = "hf-bytelevel-bpe-literal-v1"
            self.normalization = data["normalizer"]
            self.library = {"tokenizers": importlib.metadata.version("tokenizers")}
        else:
            import tiktoken
            names += ["tiktoken.model", "tokenization_kimi.py"]
            ranks = {}
            for line in (self.root / "tiktoken.model").read_bytes().splitlines():
                word, rank = line.split()
                value, native_id = base64.b64decode(word, validate=True), int(rank)
                if not value or value in ranks or native_id in self.surfaces:
                    raise ValueError("Ambiguous native vocabulary entry")
                ranks[value] = native_id
                self.surfaces[native_id] = value
            if set(self.surfaces) != set(range(len(ranks))):
                raise ValueError("The native merge ranks must be contiguous")
            pattern, reserved = read_kimi_pattern(self.root / "tokenization_kimi.py")
            declared = self.config["added_tokens_decoder"]
            special = {}
            for native_id in range(len(ranks), len(ranks) + reserved):
                row = declared.get(str(native_id), {"content": f"<|reserved_token_{native_id}|>", "special": True})
                self.controls[native_id] = dict(row)
                special[row["content"]] = native_id
            self.tokenizer = tiktoken.Encoding(name=self.root.name, pat_str=pattern,
                                               mergeable_ranks=ranks, special_tokens=special)
            self.adapter = "kimi-tiktoken-literal-v1"
            self.normalization = None
            self.library = {"tiktoken": importlib.metadata.version("tiktoken")}
        if set(self.surfaces) & set(self.controls) or len(set(self.surfaces.values())) != len(self.surfaces):
            raise ValueError("Native token identities are ambiguous")
        if not all(bytes([byte]) in self.surfaces.values() for byte in range(256)):
            raise ValueError("The admitted adapter requires complete byte fallback")
        self.files = {name: file_digest(self.root / name) for name in names}
        tokenizer_files = {name: identity for name, identity in self.files.items() if name not in {"LICENSE", "hf-api.json"}}
        self.tokenizer_identity = digest_bytes(canonical(tokenizer_files))
        self.identity = digest_bytes(canonical({"files": self.files, "adapter": self.adapter,
                                               "library": self.library, "implementation": file_digest(__file__)}))
        self.metadata = {"repository": api["id"], "revision": api["sha"],
            "identity": self.identity, "tokenizer_identity": self.tokenizer_identity,
            "adapter": self.adapter, "library": self.library, "files": self.files,
            "normalization": self.normalization, "literal_entries": len(self.surfaces),
            "control_entries": len(self.controls), "license": {"file": "LICENSE", "identity": self.files["LICENSE"]},
            "scope": "Literal text adapter; native chat and media templates retain separate contracts"}

    def check_source(self):
        if any(file_digest(self.root / name) != value for name, value in self.files.items()):
            raise ValueError("Vocabulary source changed after import")

    def encode(self, text):
        text.encode("utf-8", errors="strict")
        if len(text) > 25000:
            raise ValueError("Literal adapter input exceeds the assessed 25000-character extent")
        if self.adapter.startswith("hf-"):
            return self.tokenizer.encode(text, add_special_tokens=False).ids
        return self.tokenizer.encode(text, disallowed_special=())

    def decode_bytes(self, ids):
        if any(type(value) is not int or value not in self.surfaces for value in ids):
            raise ValueError("A control or foreign ID cannot be decoded as literal source text")
        return b"".join(self.surfaces[value] for value in ids)

    def observe(self, text):
        original = text.encode("utf-8", errors="strict")
        ids = self.encode(text)
        decoded = self.decode_bytes(ids)
        return {"source_identity": self.identity, "tokenizer_identity": self.tokenizer_identity,
            "native_ids": ids, "original_utf8": base64.b64encode(original).decode(),
            "native_decoded_utf8": base64.b64encode(decoded).decode(), "native_roundtrip_exact": decoded == original,
            "normalization_or_segmentation_residual_required": decoded != original,
            "role": "literal_text", "byte_identity": digest_bytes(original)}


class CommonVocabulary:
    """A reversible candidate segmentation over the union of literal pieces.

    The minimum-piece objective tests lexical coverage. It is not a language
    likelihood, a semantic decision or a token sequence for an existing model.
    """
    def __init__(self, sources):
        if len({source.identity for source in sources}) != len(sources):
            raise ValueError("Import each exact vocabulary source once")
        self.sources = list(sources)
        values = {bytes([value]) for value in range(256)}
        for source in sources:
            values.update(source.surfaces.values())
        self._set_pieces(sorted(values))

    def _set_pieces(self, values):
        self.pieces = values
        if any(not value or len(value) > 65535 for value in self.pieces):
            raise ValueError("Common piece length extent")
        self.ids = {value: index for index, value in enumerate(self.pieces)}
        self.max_length = max(map(len, self.pieces))
        h = hashlib.shake_256(b"nanoai.syszero.byte-pieces.v1\0")
        for value in self.pieces:
            h.update(struct.pack("<H", len(value))); h.update(value)
        self.identity = "shake256:" + h.hexdigest(64)

    @classmethod
    def load(cls, output):
        output = Path(output)
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["schema"] != "nanoai.syszero.v1" or manifest["pieces_file"] != "pieces.bin":
            raise ValueError("Common vocabulary container schema")
        path = output / "pieces.bin"
        if file_digest(path) != manifest["pieces_identity"]:
            raise ValueError("Common vocabulary payload changed")
        pieces = []
        with path.open("rb") as stream:
            header = stream.read(8)
            if len(header) != 8 or header[:4] != b"SZL1":
                raise ValueError("Common vocabulary header")
            count = struct.unpack("<I", header[4:])[0]
            if count != manifest["piece_count"] or not 256 <= count <= 2_000_000:
                raise ValueError("Common vocabulary count")
            for _ in range(count):
                size = stream.read(2)
                if len(size) != 2:
                    raise ValueError("Truncated common vocabulary")
                length = struct.unpack("<H", size)[0]
                value = stream.read(length)
                if not value or len(value) != length:
                    raise ValueError("Truncated common piece")
                pieces.append(value)
            if stream.read(1) or pieces != sorted(set(pieces)) or not set(bytes([x]) for x in range(256)).issubset(pieces):
                raise ValueError("Common vocabulary ordering or byte coverage")
        value = cls.__new__(cls)
        value.sources = []
        value._set_pieces(pieces)
        if value.identity != manifest["vocabulary_identity"]:
            raise ValueError("Common vocabulary identity differs")
        value.manifest = manifest
        return value

    def encode(self, raw: bytes):
        if not isinstance(raw, bytes) or len(raw) > 131072:
            raise ValueError("Common text must be at most 128 KiB of exact bytes")
        if len(raw) * self.max_length > 32_000_000:
            raise ValueError("Common segmentation exceeds the bounded reference work budget")
        # Bounded dictionary lookup avoids duplicating the complete vocabulary
        # into a Python object for every byte of every imported token.
        cost, choice = [len(raw) + 1] * (len(raw) + 1), [None] * len(raw)
        cost[-1] = 0
        for start in range(len(raw) - 1, -1, -1):
            for end in range(min(len(raw), start + self.max_length), start, -1):
                token_id = self.ids.get(raw[start:end])
                if token_id is not None and cost[end] + 1 < cost[start]:
                    cost[start], choice[start] = cost[end] + 1, (end, token_id)
        spans, cursor = [], 0
        while cursor < len(raw):
            end, token_id = choice[cursor]
            spans.append({"id": token_id, "start": cursor, "end": end})
            cursor = end
        return {"vocabulary_identity": self.identity, "byte_identity": digest_bytes(raw),
                "spans": spans, "objective": "minimum literal pieces with longest-first ties"}

    def decode(self, encoded):
        if encoded["vocabulary_identity"] != self.identity:
            raise ValueError("Common vocabulary version differs")
        parts, cursor = [], 0
        for row in encoded["spans"]:
            token_id = row["id"]
            if type(token_id) is not int or not 0 <= token_id < len(self.pieces):
                raise ValueError("Common token ID extent")
            value = self.pieces[token_id]
            if row["start"] != cursor or row["end"] != cursor + len(value):
                raise ValueError("Common byte spans differ")
            parts.append(value); cursor += len(value)
        raw = b"".join(parts)
        if digest_bytes(raw) != encoded["byte_identity"]:
            raise ValueError("Common byte identity differs")
        return raw

    def witnesses(self, token_id):
        value = self.pieces[token_id]
        return [{"source_identity": source.identity, "native_id": native_id}
                for source in self.sources for native_id, surface in source.surfaces.items() if surface == value]

    def save(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        with (output / "pieces.bin").open("xb") as stream:
            stream.write(b"SZL1" + struct.pack("<I", len(self.pieces)))
            for value in self.pieces:
                stream.write(struct.pack("<H", len(value)) + value)
        # Native relation tables preserve ID order. A control has no common ID.
        relations = []
        for index, source in enumerate(self.sources):
            rows = sorted(source.surfaces.items())
            name = f"source-{index}.map"
            with (output / name).open("xb") as stream:
                for native_id, value in rows:
                    stream.write(struct.pack("<II", native_id, self.ids[value]))
            relations.append({**source.metadata, "relation_file": name,
                "relation_identity": file_digest(output / name), "controls": source.controls})
        manifest = {"schema": "nanoai.syszero.v1", "vocabulary_identity": self.identity,
            "piece_count": len(self.pieces), "pieces_file": "pieces.bin", "pieces_identity": file_digest(output / "pieces.bin"),
            "sources": relations, "meaning": "Shared literal byte surfaces; no shared embedding or probability assertion"}
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        return manifest
