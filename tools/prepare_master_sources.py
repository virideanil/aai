"""Build the common byte/source inventory from a pinned local source manifest."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoai.tokenizer_sources import build_common_inventory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.sources.read_text())
    if document["schema"] != "nanoai.master-sources.v1":
        raise ValueError("Source manifest schema")
    specs = document["sources"]
    for spec in specs:
        spec["path"] = str((args.sources.parent / spec["path"]).resolve())
    start = time.monotonic()
    manifest = build_common_inventory(specs, args.out)
    print(json.dumps({"sources": len(manifest["sources"]), "pieces": manifest["piece_count"],
                      "inventory_identity": manifest["inventory_identity"],
                      "seconds": time.monotonic() - start,
                      "partial_utf8_entries": sum(s["partial_utf8_entries"] for s in manifest["sources"]),
                      "aliased_native_entries": sum(s["aliased_native_entries"] for s in manifest["sources"])}))


if __name__ == "__main__":
    main()
