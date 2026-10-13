"""Select a finite byte tokenizer using a declared lexical calibration split.

This estimates tokenizer probabilities only. It never loads or updates a neural
model. A separate confirmation file is read only after the selection is frozen.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoai.core import digest_bytes, file_digest, process_memory
from nanoai.master_tokenizer import MasterTokenizer, MAX_PIECE
from nanoai.tokenizer_sources import load_common_inventory


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def score(tokenizer, lines):
    nats, token_count, byte_count = 0.0, 0, 0
    rows = []
    for raw in lines:
        encoded = tokenizer.encode(raw)
        if tokenizer.decode(encoded) != raw:
            raise ValueError("Lexical consumer round trip")
        row = {"byte_identity": digest_bytes(raw), "bytes": len(raw), "tokens": len(encoded["spans"]),
               "forward_logp": encoded["forward_logp"], "viterbi_logp": encoded["viterbi_logp"]}
        rows.append(row)
        byte_count += len(raw); token_count += row["tokens"]; nats -= row["forward_logp"]
    return {"nats_per_byte": nats / max(1, byte_count), "bytes": byte_count, "tokens": token_count,
            "bytes_per_token": byte_count / max(1, token_count), "rows": rows}


def family_witnesses(inventory, candidates):
    manifest = json.loads((inventory / "manifest.json").read_text())
    groups, piece_families = {}, {i: set() for i in candidates}
    for source in manifest["sources"]:
        data = (inventory / source["relation_file"]).read_bytes()
        # Families use exact surface membership, so copied rank tables or model
        # names do not create extra votes. Native ID order has a separate owner.
        ids = sorted({common for _, common in struct.iter_unpack("<II", data)})
        h = hashlib.shake_256()
        for common in ids:
            h.update(struct.pack("<I", common))
        key = "shake256:" + h.hexdigest(64)
        groups.setdefault(key, []).append(source["name"])
        for common in candidates.intersection(ids):
            piece_families[common].add(key)
    return groups, piece_families


def fit(common, ids, initial, lines, epochs, byte_prior):
    pieces = [common.pieces[i] for i in ids]
    norm = math.fsum(initial)
    tokenizer = MasterTokenizer(pieces, ids, [v / norm for v in initial], .05,
                                common.identity, common.inventory_identity)
    lattices = [tokenizer.lattice(raw) for raw in lines]
    history = []
    for epoch in range(epochs):
        counts = [0.0] * len(ids)
        logp = 0.0
        for lattice in lattices:
            expected, value = tokenizer.expectation(lattice)
            logp += value
            for i, count in enumerate(expected):
                counts[i] += count
        tokens = math.fsum(counts)
        history.append({"epoch": epoch, "train_logp": logp, "expected_tokens": tokens})
        stop = len(lines) / (len(lines) + tokens)
        smoothed = [count + (byte_prior if len(piece) == 1 else .001) for piece, count in zip(pieces, counts)]
        norm = math.fsum(smoothed)
        tokenizer = MasterTokenizer(pieces, ids, [count / norm for count in smoothed], stop,
                                    common.identity, common.inventory_identity)
    return tokenizer, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protect-source-scalars", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    corpus_bytes = args.corpus.read_bytes()
    corpus = json.loads(corpus_bytes)["lines"]
    if not isinstance(corpus, list) or not all(isinstance(s, str) and s for s in corpus):
        raise ValueError("Calibration corpus requires nonempty text rows")
    rows = [text.encode("utf-8", errors="strict") for text in corpus]
    order = list(range(len(rows))); random.Random(0).shuffle(order)
    validation_ids = sorted(order[:int(len(rows) * .1)])
    train_ids = [i for i in range(len(rows)) if i not in validation_ids]
    train, validation = [rows[i] for i in train_ids], [rows[i] for i in validation_ids]
    policy = {"schema": "nanoai.master-selection.v1", "corpus_identity": digest_bytes(corpus_bytes),
              "confirmation_identity": file_digest(args.confirmation), "seed": 0,
              "train_ids": train_ids, "validation_ids": validation_ids,
              "budgets": [1024, 2048, 4096], "family_prior_strengths": [0.0, .25],
              "byte_pseudocount": .05, "piece_pseudocount": .001, "epochs": 8,
              "minimum_multi_byte_count": 2, "max_piece_bytes": MAX_PIECE,
              "protect_source_scalars": args.protect_source_scalars,
              "source_scalar_reserve": .02 if args.protect_source_scalars else 0,
              "selection": "Smallest active vocabulary within 1 percent of minimum validation nats/byte; then lower nats/byte, then lower family prior.",
              "scope": "Lexical calibration on an authored corpus; confirmation diagnostics do not establish semantic or general language quality."}
    write(args.out / "policy.json", policy)
    common = load_common_inventory(args.inventory)
    counts = Counter()
    for raw in train:
        for end in range(1, len(raw) + 1):
            for size in range(1, min(MAX_PIECE, end) + 1):
                common_id = common.ids.get(raw[end - size:end])
                if common_id is not None:
                    counts[common_id] += 1
    byte_ids = {common.ids[bytes([i])] for i in range(256)}
    candidates = {i for i, count in counts.items() if count >= 2 and len(common.pieces[i]) > 1}
    groups, witnesses = family_witnesses(args.inventory, candidates | byte_ids)
    write(args.out / "families.json", {"vocabulary_identity": common.identity,
          "definition": "Exact set of common byte IDs, within this frozen dictionary", "families": groups})
    print(json.dumps({"phase": "candidates", "attested_multi_byte": len(candidates), "families": len(groups),
                      "loaded_seconds": time.monotonic() - start}), flush=True)
    summaries = []
    for budget in policy["budgets"]:
        for strength in policy["family_prior_strengths"]:
            key = f"rows-{budget}-family-{strength:g}"
            def rank(i):
                return counts[i] * (len(common.pieces[i]) - 1) * (1 + strength * math.log1p(len(witnesses[i])))
            selected = sorted(candidates, key=lambda i: (-rank(i), i))[:budget - 256]
            ids = sorted(byte_ids | set(selected))
            initial = [max(counts[i], .05) * len(common.pieces[i]) for i in ids]
            tokenizer, history = fit(common, ids, initial, train, policy["epochs"], policy["byte_pseudocount"])
            if args.protect_source_scalars:
                tokenizer = tokenizer.with_source_scalars(common, reserve=policy["source_scalar_reserve"])
            tokenizer.save(args.out / (key + ".nmt"))
            result = {"candidate": key, "budget": budget, "family_prior_strength": strength,
                      "active_rows": len(tokenizer.pieces), "calibrated_rows": len(ids), "artifact_identity": tokenizer.identity,
                      "artifact_bytes": (args.out / (key + ".nmt")).stat().st_size,
                      "train": score(tokenizer, train), "validation": score(tokenizer, validation), "history": history}
            write(args.out / (key + ".json"), result)
            summaries.append(result)
            print(json.dumps({"candidate": key, "active": len(tokenizer.pieces), "validation_nats_per_byte": result["validation"]["nats_per_byte"]}), flush=True)
    best = min(s["validation"]["nats_per_byte"] for s in summaries)
    eligible = [s for s in summaries if s["validation"]["nats_per_byte"] <= best * 1.01]
    chosen = min(eligible, key=lambda s: (s["active_rows"], s["validation"]["nats_per_byte"], s["family_prior_strength"]))
    selection = {"schema": "nanoai.master-selection-result.v1", "candidate": chosen["candidate"],
                 "identity": chosen["artifact_identity"], "active_rows": chosen["active_rows"],
                 "inventory_identity": common.inventory_identity, "vocabulary_identity": common.identity,
                 "minimum_validation_nats_per_byte": best, "validation_nats_per_byte": chosen["validation"]["nats_per_byte"],
                 "families": len(groups), "sources": sum(map(len, groups.values())),
                 "candidate_file": chosen["candidate"] + ".nmt", "policy_identity": file_digest(args.out / "policy.json")}
    write(args.out / "selection.json", selection)
    tokenizer = MasterTokenizer.load(args.out / selection["candidate_file"], expected_identity=selection["identity"])
    tokenizer.save(args.out / "master.nmt")
    # Selection is now durable. Confirmation cannot change candidate choice.
    confirmation_data = args.confirmation.read_bytes()
    if digest_bytes(confirmation_data) != policy["confirmation_identity"]:
        raise ValueError("Confirmation changed during calibration")
    confirmation = json.loads(confirmation_data)["rows"]
    result = score(tokenizer, [row["text"].encode("utf-8") for row in confirmation])
    for measured, source in zip(result["rows"], confirmation):
        measured["domain"] = source["domain"]
    write(args.out / "confirmation.json", {"selection_identity": file_digest(args.out / "selection.json"),
          "confirmation_identity": policy["confirmation_identity"], "metrics": result,
          "seconds": time.monotonic() - start, "process_memory": process_memory(__import__("os").getpid()),
          "neural_parameter_updates": 0})
    print(json.dumps({"phase": "frozen", **selection, "confirmation_nats_per_byte": result["nats_per_byte"]}), flush=True)


if __name__ == "__main__":
    main()
