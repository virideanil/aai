"""Source-preserving vocabulary imports for the common byte dictionary.

These are candidate-piece witnesses. Native encoding still runs each source's
complete tokenizer. In particular, a WordPiece continuation and a word start
can share surface bytes while retaining separate native IDs and roles.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import importlib.metadata
import json
import math
from pathlib import Path
import re
import struct

from .core import digest_bytes, file_digest
from .syszero import CommonVocabulary, byte_alphabet, canonical


ADAPTER_VERSION = "nanoai.tokenizer-source.v1"
ROLE_CODES = {"byte": 1, "literal": 2, "word_start": 3, "word_continuation": 4,
              "metaspace": 5, "user_defined": 6}


@dataclass
class TokenizerInventory:
    name: str
    surfaces: dict[int, bytes]
    roles: dict[int, str]
    controls: dict[int, dict]
    scores: dict[int, float]
    metadata: dict

    @property
    def identity(self):
        return self.metadata["identity"]


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def read_inventory(spec: dict) -> TokenizerInventory:
    """Read one pinned data payload without importing downloaded source code."""
    name = spec["name"]
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,95}", name):
        raise ValueError("Source name extent")
    path = Path(spec["path"])
    data = path.read_bytes()
    identity = digest_bytes(data)
    if identity != spec["digest"]:
        raise ValueError("Tokenizer source bytes differ: " + name)
    surfaces, roles, controls, scores = {}, {}, {}, {}
    native_ids = set()
    rules, library = {}, {}

    def add(native_id, piece, role, *, control=None, score=None):
        if type(native_id) is not int or not 0 <= native_id <= 0xFFFFFFFF or native_id in native_ids:
            raise ValueError("Native ID is invalid or repeated")
        native_ids.add(native_id)
        if control is not None:
            controls[native_id] = control
            return
        if not isinstance(piece, bytes) or not piece or len(piece) > 65535 or role not in ROLE_CODES:
            raise ValueError("Native surface or role is invalid")
        surfaces[native_id], roles[native_id] = piece, role
        if score is not None:
            if not math.isfinite(score):
                raise ValueError("Native unigram score is nonfinite")
            scores[native_id] = score

    kind = spec["kind"]
    if kind == "tiktoken":
        seen_pieces = set()
        for line in data.splitlines():
            if not line:
                continue
            encoded, ordinal = line.split()
            piece = base64.b64decode(encoded, validate=True)
            if piece in seen_pieces:
                raise ValueError("Repeated tiktoken byte piece")
            seen_pieces.add(piece)
            add(int(ordinal), piece, "byte")
        rules = {"rank_field": "native_id is the declared merge rank",
                 "native_encoding": "Requires the source regex, added tokens and template in addition to this table"}
    elif kind == "wordpiece":
        lines = data.decode("utf-8", errors="strict").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        for native_id, token in enumerate(lines):
            if re.fullmatch(r"\[(?:PAD|UNK|CLS|SEP|MASK|unused\d+)\]", token):
                add(native_id, None, None, control={"spelling": token, "kind": "reserved"})
            else:
                continuation = token.startswith("##")
                add(native_id, (token[2:] if continuation else token).encode("utf-8"),
                    "word_continuation" if continuation else "word_start")
        rules = {"continuing_subword_prefix": "##", "rank_field": "native ID only",
                 "native_encoding": "Vocabulary witness only; native normalization and pre-tokenization require source configuration"}
    elif kind == "sentencepiece":
        from sentencepiece import sentencepiece_model_pb2
        model = sentencepiece_model_pb2.ModelProto()
        model.ParseFromString(data)
        library = {n: importlib.metadata.version(n) for n in ("sentencepiece", "protobuf")}
        for native_id, entry in enumerate(model.pieces):
            token = entry.piece
            if entry.type in (2, 3, 5):
                add(native_id, None, None, control={"spelling": token, "kind": int(entry.type)})
            elif entry.type == 6:
                if not re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", token):
                    raise ValueError("SentencePiece byte spelling")
                add(native_id, bytes([int(token[3:5], 16)]), "byte", score=float(entry.score))
            elif entry.type in (1, 4):
                add(native_id, token.replace("▁", " ").encode("utf-8"),
                    "user_defined" if entry.type == 4 else "metaspace", score=float(entry.score))
            else:
                raise ValueError("Unreviewed SentencePiece entry type")
        norm = model.normalizer_spec
        rules = {"rank_field": "native ID only; source unigram scores retained separately",
                 "normalizer": {"name": norm.name, "add_dummy_prefix": norm.add_dummy_prefix,
                                "remove_extra_whitespaces": norm.remove_extra_whitespaces,
                                "escape_whitespaces": norm.escape_whitespaces,
                                "compiled_map_digest": digest_bytes(norm.precompiled_charsmap)},
                 "model_type": int(model.trainer_spec.model_type),
                 "byte_fallback": model.trainer_spec.byte_fallback,
                 "native_encoding": "The complete model protobuf owns normalization and decoding"}
    elif kind == "hf-tokenizer":
        parsed = json.loads(data)
        model = parsed["model"]
        model_type = model.get("type") or ("BPE" if isinstance(model.get("vocab"), dict) else None)
        decoder = parsed.get("decoder") or {}
        decoder_types = {row.get("type") for row in _walk(decoder)}
        added = {}
        for row in parsed.get("added_tokens", []):
            if row["id"] in added and row != added[row["id"]]:
                raise ValueError("Conflicting added-token ID")
            added[row["id"]] = row
        alphabet = byte_alphabet()
        byte_level = "ByteLevel" in decoder_types
        fallback = "ByteFallback" in decoder_types
        metaspace = ("Metaspace" in decoder_types or any(row.get("type") == "Replace"
                     and row.get("pattern") == {"String": "▁"} and row.get("content") == " " for row in _walk(decoder)))
        if model_type == "Unigram":
            entries = [(i, token, score) for i, (token, score) in enumerate(model["vocab"])]
        elif model_type in ("BPE", "WordPiece"):
            entries = [(i, token, None) for token, i in model["vocab"].items()]
        else:
            raise ValueError("Unreviewed tokenizer model type")
        if model_type == "BPE" and not (byte_level or metaspace or fallback):
            raise ValueError("Unreviewed BPE decoder")
        if model_type == "Unigram" and not metaspace:
            raise ValueError("Unreviewed Unigram decoder")
        prefix = model.get("continuing_subword_prefix") or ("##" if model_type == "WordPiece" else "")
        for native_id, token, score in entries:
            if added.get(native_id, {}).get("special") or token == model.get("unk_token") or native_id == model.get("unk_id"):
                add(native_id, None, None, control=added.get(native_id, {"spelling": token, "kind": "unknown"}))
                continue
            if byte_level:
                try:
                    surface = bytes(alphabet[char] for char in token)
                    role = "byte"
                except KeyError as error:
                    entry = added.get(native_id)
                    if entry is None or entry.get("special") or entry["content"] != token:
                        raise ValueError("Unreviewed byte-level token spelling") from error
                    surface, role = token.encode("utf-8"), "user_defined"
            elif fallback and re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", token):
                surface, role = bytes([int(token[3:5], 16)]), "byte"
            elif model_type == "WordPiece":
                continuation = bool(prefix) and token.startswith(prefix)
                surface = (token[len(prefix):] if continuation else token).encode("utf-8")
                role = "word_continuation" if continuation else "word_start"
            else:
                surface, role = token.replace("▁", " ").encode("utf-8"), "metaspace"
            add(native_id, surface, role, score=score)
        # Added entries retain native matching rules, even when they share an ID
        # with the model vocabulary. They are never invented literal controls.
        for native_id, entry in added.items():
            if native_id not in native_ids:
                if entry.get("special"):
                    add(native_id, None, None, control=entry)
                else:
                    add(native_id, entry["content"].encode("utf-8"), "user_defined")
        rules = {key: parsed.get(key) for key in ("normalizer", "pre_tokenizer", "post_processor", "decoder")}
        rules.update(model_type=model_type, rank_field="native ID only; model merge rules remain in the source payload",
                     added_tokens=parsed.get("added_tokens", []),
                     native_encoding="Run the complete native tokenizer; membership is a source witness")
    else:
        raise ValueError("Unsupported vocabulary source kind")

    if not surfaces or len(native_ids) > 2_000_000:
        raise ValueError("Vocabulary source extent")
    partial = 0
    for piece in surfaces.values():
        try:
            piece.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            partial += 1
    unique = set(surfaces.values())
    source_identity = digest_bytes(canonical({"name": name, "kind": kind, "payload": identity,
                                             "adapter": ADAPTER_VERSION, "implementation": file_digest(__file__)}))
    metadata = {"schema": ADAPTER_VERSION, "name": name, "identity": source_identity,
                "payload_identity": identity, "kind": kind, "adapter": ADAPTER_VERSION,
                "library": library, "literal_entries": len(surfaces), "unique_surfaces": len(unique),
                "partial_utf8_entries": partial, "control_entries": len(controls),
                "aliased_native_entries": len(surfaces) - len(unique), "rules": rules,
                "provenance": spec.get("provenance", {}),
                "scope": "Candidate byte surfaces and native ID/role witnesses; no native segmentation or shared embedding assertion"}
    return TokenizerInventory(name, surfaces, roles, controls, scores, metadata)


def build_common_inventory(specs: list[dict], output: str | Path):
    """Two passes bound memory to the union plus one source at a time.

    The existing System Zero format owns common byte IDs. Additional source-role
    files retain WordPiece/SentencePiece context without changing those IDs.
    """
    if len({s["name"] for s in specs}) != len(specs):
        raise ValueError("Duplicate source name")
    values = {bytes([i]) for i in range(256)}
    metadata = []
    for spec in specs:
        source = read_inventory(spec)
        values.update(source.surfaces.values())
        metadata.append(source.metadata)
        del source
    value = CommonVocabulary.__new__(CommonVocabulary)
    value.sources = []
    value._set_pieces(sorted(values))
    del values
    output = Path(output)
    manifest = value.save(output)
    manifest["source_contract"] = ADAPTER_VERSION
    manifest["role_codes"] = ROLE_CODES
    manifest["sources"] = []
    for index, (spec, expected) in enumerate(zip(specs, metadata)):
        source = read_inventory(spec)
        if source.identity != expected["identity"]:
            raise ValueError("Vocabulary source changed between import passes")
        relation_name, role_name, score_name = f"source-{index}.map", f"source-{index}.roles", f"source-{index}.scores"
        with (output / relation_name).open("xb") as relations, (output / role_name).open("xb") as roles:
            for native_id, piece in sorted(source.surfaces.items()):
                relations.write(struct.pack("<II", native_id, value.ids[piece]))
                roles.write(bytes([ROLE_CODES[source.roles[native_id]]]))
        with (output / score_name).open("xb") as scores:
            for native_id, score in sorted(source.scores.items()):
                scores.write(struct.pack("<Id", native_id, score))
        manifest["sources"].append({**source.metadata, "relation_file": relation_name,
            "relation_identity": file_digest(output / relation_name), "role_file": role_name,
            "role_identity": file_digest(output / role_name), "score_file": score_name,
            "score_identity": file_digest(output / score_name),
            "controls": {str(native_id): control for native_id, control in source.controls.items()}})
        del source
    manifest["inventory_identity"] = digest_bytes(canonical(manifest))
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    restored = load_common_inventory(output, expected_identity=manifest["inventory_identity"])
    if restored.identity != value.identity:
        raise ValueError("Reloaded source inventory differs")
    return manifest


def load_common_inventory(output: str | Path, *, expected_identity: str | None = None):
    """Verify the byte dictionary and every typed native witness file."""
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    identity = manifest.pop("inventory_identity", None)
    if identity != digest_bytes(canonical(manifest)) or (expected_identity is not None and identity != expected_identity):
        raise ValueError("Source inventory identity differs")
    if manifest.get("source_contract") != ADAPTER_VERSION or manifest.get("role_codes") != ROLE_CODES:
        raise ValueError("Source inventory contract differs")
    common = CommonVocabulary.load(output)
    for index, source in enumerate(manifest["sources"]):
        for field, suffix in (("relation", "map"), ("role", "roles"), ("score", "scores")):
            name = source[field + "_file"]
            if name != f"source-{index}.{suffix}" or file_digest(output / name) != source[field + "_identity"]:
                raise ValueError("Source witness file differs")
        relation = (output / source["relation_file"]).read_bytes()
        roles = (output / source["role_file"]).read_bytes()
        scores = (output / source["score_file"]).read_bytes()
        if len(relation) != 8 * source["literal_entries"] or len(roles) != source["literal_entries"] or len(scores) % 12:
            raise ValueError("Source witness extent differs")
        ids = []
        for native_id, common_id in struct.iter_unpack("<II", relation):
            if common_id >= len(common.pieces):
                raise ValueError("Source witness common ID extent")
            ids.append(native_id)
        if ids != sorted(set(ids)) or any(role not in ROLE_CODES.values() for role in roles):
            raise ValueError("Source witness native ID or role differs")
        if set(ids) & {int(key) for key in source["controls"]}:
            raise ValueError("Source control and literal IDs overlap")
        score_ids = []
        id_set = set(ids)
        for native_id, score in struct.iter_unpack("<Id", scores):
            if native_id not in id_set or not math.isfinite(score):
                raise ValueError("Source score witness differs")
            score_ids.append(native_id)
        if score_ids != sorted(set(score_ids)):
            raise ValueError("Source score native IDs differ")
    common.inventory_identity = identity
    return common
