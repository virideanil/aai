"""Build a local common dictionary from explicit, already acquired sources."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoai.syszero import CommonVocabulary, VocabularySource


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    start = time.monotonic()
    sources = [VocabularySource(path) for path in args.source]
    vocabulary = CommonVocabulary(sources)
    manifest = vocabulary.save(args.out)
    for source in sources:
        source.check_source()
    restored = CommonVocabulary.load(args.out)
    if restored.identity != vocabulary.identity:
        raise ValueError("Persisted dictionary identity differs")
    print(json.dumps({"vocabulary_identity": restored.identity, "piece_count": len(restored.pieces),
        "source_entries": sum(len(source.surfaces) for source in sources),
        "source_count": len(sources), "seconds": time.monotonic() - start,
        "bytes": sum(path.stat().st_size for path in args.out.iterdir()),
        "manifest": str(args.out / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
