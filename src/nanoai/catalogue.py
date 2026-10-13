"""Durable literal observations and followers on the existing task journal.

The catalogue is an evidence view. Its counts describe observed token sequences;
semantic interpretations and neural parameter versions have separate owners.
"""
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import struct

from .core import digest_bytes, file_digest
from .memory import canonical, identifier, StaleState


def _observation_bytes(state, row):
    source = row["source"]
    versions = state["source_history"].get(source["id"], [])
    found = [s for s in versions if s["revision"] == source["revision"] and s["digest"] == source["digest"]]
    if len(found) != 1:
        raise ValueError("Catalogue source revision is unavailable")
    return found[0]["text"].encode("utf-8")


def apply_event(state, event):
    if event["kind"] == "catalogue.observed":
        row = event["observation"]
        identifier(row["id"])
        if row.get("schema") != "nanoai.language-observation.v1" or len(canonical(row)) > 4_000_000:
            raise ValueError("Catalogue observation schema or extent")
        if row["id"] in state["language_observations"]:
            raise ValueError("Catalogue observation IDs are immutable")
        raw = _observation_bytes(state, row)
        encoded = row["encoded"]
        if (encoded["schema"] != "nanoai.master-tokens.v1" or encoded["byte_identity"] != digest_bytes(raw)
                or row["snapshot"]["core_identity"] != encoded["tokenizer_identity"]
                or row["snapshot"]["sources"] != {row["source"]["id"]: {k: row["source"][k] for k in ("revision", "digest")}}):
            raise ValueError("Catalogue source and tokenizer basis differ")
        if not isinstance(encoded["spans"], list) or len(encoded["spans"]) > len(raw):
            raise ValueError("Catalogue spans extent")
        cursor = 0
        for span in encoded["spans"]:
            if (any(type(span.get(k)) is not int for k in ("id", "active_id", "start", "end"))
                    or not 0 <= span["id"] < 2_000_000 or not 0 <= span["active_id"] < row["active_rows"]
                    or span["start"] != cursor or not cursor < span["end"] <= len(raw)):
                raise ValueError("Catalogue byte span or ID")
            cursor = span["end"]
        if cursor != len(raw) or not all(math.isfinite(encoded[k]) for k in ("forward_logp", "viterbi_logp")):
            raise ValueError("Catalogue byte coverage or probability")
        state["language_observations"][row["id"]] = deepcopy(row)
    elif event["kind"] == "catalogue.perception":
        row = event["perception"]
        identifier(row["id"])
        if row["id"] in state["perception_references"] or row.get("schema") != "nanoai.perception-reference.v1":
            raise ValueError("Perception reference identity or schema")
        if len(canonical(row)) > 262144:
            raise ValueError("Perception reference extent")
        state["perception_references"][row["id"]] = deepcopy(row)
    else:
        raise ValueError("Unknown catalogue event")


def source_witnesses(inventory, expected_identity, common_ids):
    """Read selected native witnesses from the verified cold relation files."""
    from .tokenizer_sources import ADAPTER_VERSION, ROLE_CODES
    inventory = Path(inventory)
    manifest = json.loads((inventory / "manifest.json").read_text())
    identity = manifest.pop("inventory_identity")
    if identity != expected_identity or digest_bytes(canonical(manifest)) != identity or manifest["source_contract"] != ADAPTER_VERSION:
        raise ValueError("Catalogue inventory identity")
    wanted = set(common_ids)
    if len(wanted) > 5000:
        raise ValueError("Witness view extent")
    found = {i: [] for i in wanted}
    role_names = {v: k for k, v in ROLE_CODES.items()}
    for index, source in enumerate(manifest["sources"]):
        if source["relation_file"] != f"source-{index}.map" or source["role_file"] != f"source-{index}.roles":
            raise ValueError("Catalogue witness path")
        relations = (inventory / source["relation_file"]).read_bytes()
        roles = (inventory / source["role_file"]).read_bytes()
        if (digest_bytes(relations) != source["relation_identity"] or digest_bytes(roles) != source["role_identity"]
                or len(relations) != len(roles) * 8 or len(roles) != source["literal_entries"]):
            raise ValueError("Catalogue witness changed")
        for (native_id, common_id), role in zip(struct.iter_unpack("<II", relations), roles):
            if common_id in wanted:
                found[common_id].append({"source": source["name"], "source_identity": source["identity"],
                    "native_id": native_id, "role": role_names[role],
                    "scope": "Native membership with contextual role; native encoding remains source-specific"})
    return found


class LanguageCatalogue:
    def __init__(self, memory):
        self.memory = memory

    def observe(self, observation_id, task_id, source_id, tokenizer):
        identifier(observation_id)
        memory = self.memory
        memory.reload()
        source = deepcopy(memory.state["sources"][source_id])
        snapshot = memory.snapshot(task_id, [source], tokenizer.identity)
        raw = source["text"].encode("utf-8")
        encoded = tokenizer.encode(raw)
        if tokenizer.decode(encoded) != raw:
            raise ValueError("Catalogue tokenizer byte check")
        observation = {"schema": "nanoai.language-observation.v1", "id": observation_id,
            "snapshot": snapshot, "source": {k: source[k] for k in ("id", "revision", "digest")},
            "active_rows": len(tokenizer.pieces), "inventory_identity": tokenizer.inventory_identity,
            "encoded": encoded, "interpretation": "Observed literal segmentation; contextual meaning remains unassessed"}
        with memory.locked():
            memory._validate_snapshot(snapshot)
            prior = memory.state["language_observations"].get(observation_id)
            if prior is not None:
                if prior != observation:
                    raise ValueError("Catalogue retry identity already names different evidence")
                return deepcopy(prior)
            memory._append({"kind": "catalogue.observed", "observation": observation})
        return deepcopy(observation)

    def _current(self, row):
        try:
            self.memory._validate_snapshot(row["snapshot"], include_selection=False)
            if row["schema"] == "nanoai.perception-reference.v1":
                if file_digest(row["bundle"]["path"]) != row["bundle"]["identity"]:
                    raise StaleState("Perception bundle changed")
                if row.get("original_file") is not None and file_digest(row["original_file"]["path"]) != row["original_file"]["identity"]:
                    raise StaleState("Perception original file changed")
            return True, None
        except (ValueError, OSError) as error:
            return False, str(error)

    def context_entries(self, snapshot):
        """Called while the dictionary owner holds the shared journal lock."""
        result = []
        for category, label in (("language_observations", "language_observation"), ("perception_references", "perception_reference")):
            for row in self.memory.state[category].values():
                parent = row["snapshot"]
                if (parent["task_id"] != snapshot["task_id"] or parent["task_revision"] != snapshot["task_revision"]
                        or not all(snapshot["sources"].get(key) == value for key, value in parent["sources"].items())):
                    continue
                current, _ = self._current(row)
                if current:
                    result.append({"kind": label, "id": row["id"], "source": row["source"],
                        "tokenizer_identity": row["encoded"]["tokenizer_identity"] if label == "language_observation" else None,
                        "token_count": len(row["encoded"]["spans"]) if label == "language_observation" else None,
                        "perception": row.get("measurement"), "interpretation": row["interpretation"]})
        return {"entries": result[:64], "eligible_entries": len(result), "remaining_entries": max(0, len(result) - 64)}

    def view(self, task_id, *, include_history=False, limit=500, inventory=None):
        if type(limit) is not int or not 1 <= limit <= 5000:
            raise ValueError("Catalogue view limit")
        memory = self.memory
        pieces, observations, perceptions = {}, [], []
        with memory.locked():
            if task_id not in memory.state["tasks"]:
                raise ValueError("Catalogue task is unavailable")
            sequence = memory.sequence
            for row in memory.state["language_observations"].values():
                if row["snapshot"]["task_id"] != task_id:
                    continue
                current, reason = self._current(row)
                observations.append({"id": row["id"], "current": current, "reason": reason,
                    "source": row["source"], "tokenizer_identity": row["encoded"]["tokenizer_identity"],
                    "inventory_identity": row["inventory_identity"]})
                if not current and not include_history:
                    continue
                raw, prior = _observation_bytes(memory.state, row), None
                for span in row["encoded"]["spans"]:
                    piece = raw[span["start"]:span["end"]]
                    barcode = digest_bytes(piece)
                    try:
                        label, complete = piece.decode("utf-8", errors="strict"), True
                    except UnicodeDecodeError:
                        label, complete = piece.hex(), False
                    entry = pieces.setdefault(barcode, {"barcode": barcode, "bytes_hex": piece.hex(), "uses": 0,
                        "current_uses": 0, "common_ids": set(), "observations": set(), "followers": Counter(),
                        "label": label, "utf8_complete": complete, "kind": "literal_bytes"})
                    entry["uses"] += 1; entry["current_uses"] += int(current)
                    entry["common_ids"].add((row["encoded"]["vocabulary_identity"], span["id"]))
                    entry["observations"].add(row["id"])
                    if prior is not None:
                        pieces[prior]["followers"][barcode] += 1
                    prior = barcode
            for row in memory.state["perception_references"].values():
                if row["snapshot"]["task_id"] != task_id:
                    continue
                current, reason = self._current(row)
                if current or include_history:
                    perceptions.append({**deepcopy(row), "current": current, "reason": reason})
        ordered = sorted(pieces.values(), key=lambda p: (-p["uses"], p["bytes_hex"]))
        total = len(ordered)
        selected = ordered[:limit]
        for row in selected:
            row["common_ids"] = [{"vocabulary_identity": v, "id": i} for v, i in sorted(row["common_ids"])]
            row["observations"] = sorted(row["observations"])
            row["followers"] = [{"barcode": p, "count": n} for p, n in row["followers"].most_common(8)]
        if inventory is not None:
            bases = {o["inventory_identity"] for o in observations if o["current"] or include_history}
            if len(bases) != 1:
                raise ValueError("Choose observations with one inventory for native witnesses")
            witnesses = source_witnesses(inventory, next(iter(bases)), {c["id"] for r in selected for c in r["common_ids"]})
            for row in selected:
                row["native_witnesses"] = [w for c in row["common_ids"] for w in witnesses[c["id"]]]
        return {"schema": "nanoai.language-catalogue.v1", "task_id": task_id, "journal_sequence": sequence,
            "barcode_scope": "Full SHAKE256 with 64-byte output over exact piece bytes; literal identity only",
            "observations": observations, "pieces": selected, "total_pieces": total, "remaining_pieces": total - len(selected),
            "perceptions": perceptions, "parameter_updates": 0,
            "scope": "Journaled token occurrences and within-observation followers; meanings retain their attributed context"}
