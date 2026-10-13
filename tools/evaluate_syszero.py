"""Real source dictionaries and fresh persisted common-vocabulary consumer."""
import argparse
import base64
import json
from pathlib import Path
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoai.core import file_digest
from nanoai.syszero import CommonVocabulary, VocabularySource


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, action="append", required=True)
    p.add_argument("--dictionary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--panel", type=Path, default=Path(__file__).resolve().parents[1] / "evals/syszero-panel.json")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sources = [VocabularySource(path) for path in args.source]
    common = CommonVocabulary.load(args.dictionary)
    manifest = common.manifest
    result = {"panel_identity": file_digest(args.panel), "implementation": file_digest(Path(__file__).resolve().parents[1] / "src/nanoai/syszero.py"),
        "sources": [source.metadata for source in sources], "common_identity": common.identity,
        "cases": [], "checks": [], "scope": "Exact representation and adapter consumers; no model learning or generation speed claim"}
    def check(name, passed, **data):
        result["checks"].append({"name": name, "passed": bool(passed), **data})
    check("source_identities_pinned", [x.identity for x in sources] == [x["identity"] for x in manifest["sources"]])
    for source, stored in zip(sources, manifest["sources"]):
        path = args.dictionary / stored["relation_file"]
        check(source.metadata["repository"] + "_relation_identity", file_digest(path) == stored["relation_identity"])
        rows = list(struct.iter_unpack("<II", path.read_bytes()))
        check(source.metadata["repository"] + "_every_literal_mapping", len(rows) == len(source.surfaces) and
              all(common.pieces[global_id] == source.surfaces[native_id] for native_id, global_id in rows), checked_entries=len(rows))
    for case in json.loads(args.panel.read_text())["cases"]:
        text, raw = case["text"], case["text"].encode()
        row = {"id": case["id"], "bytes": len(raw), "native": []}
        start = time.perf_counter_ns()
        encoded = common.encode(raw)
        row.update(common=encoded, common_seconds=(time.perf_counter_ns() - start) / 1e9)
        check(case["id"] + "_common_roundtrip", common.decode(encoded) == raw)
        for source in sources:
            start = time.perf_counter_ns()
            observed = source.observe(text)
            elapsed = (time.perf_counter_ns() - start) / 1e9
            if source.adapter.startswith("hf-"):
                reference = source.tokenizer.decode(observed["native_ids"], skip_special_tokens=False).encode()
            else:
                reference = source.tokenizer.decode_bytes(observed["native_ids"])
            check(case["id"] + "_" + source.adapter + "_native_bytes", reference == base64.b64decode(observed["native_decoded_utf8"]))
            check(case["id"] + "_" + source.adapter + "_original_bytes", base64.b64decode(observed["original_utf8"]) == raw)
            check(case["id"] + "_" + source.adapter + "_literal_role", not set(observed["native_ids"]) & source.controls.keys())
            row["native"].append({"repository": source.metadata["repository"], "seconds": elapsed, **observed})
        result["cases"].append(row)
    decomposed = next(row for row in result["cases"] if row["id"] == "decomposed")
    check("normalization_information_loss_exposed", any(not row["native_roundtrip_exact"] for row in decomposed["native"]))
    raw_sets = [set(source.surfaces.values()) for source in sources]
    result["counts"] = {"common_pieces": len(common.pieces), "total_native_entries": sum(len(s) for s in raw_sets),
        "shared_between_all_sources": len(set.intersection(*raw_sets)),
        "native_token_totals": [sum(len(row["native"][i]["native_ids"]) for row in result["cases"]) for i in range(len(sources))],
        "common_piece_total": sum(len(row["common"]["spans"]) for row in result["cases"])}
    for source in sources:
        source.check_source()
    result["all_checks_pass"] = all(row["passed"] for row in result["checks"])
    (args.out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"checks": len(result["checks"]), "passed": result["all_checks_pass"], **result["counts"]}, indent=2))
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
