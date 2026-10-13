"""Bounded original-BF16 feature collection with exact, resumable cache keys."""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoai.core import JsonlCore, digest_bytes, file_digest
from nanoai.memory import canonical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()
    preflight = json.loads(args.preflight.read_text())
    if (preflight.get("model") != "Qwen3.5-2B-BF16" or preflight.get("effort") != "none"
            or not preflight.get("parent_source")):
        raise ValueError("A successful sourced launch preflight is required")
    corpus = json.loads(args.corpus.read_text())
    texts = sorted({t for e in corpus["episodes"] for t in
                    [e["question"], *e["observations"], *[c["content"] for c in e["candidates"]]]})
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    start = time.monotonic()
    manifest = {"schema": "nanoai.features.v1", "corpus": file_digest(args.corpus),
                "config": file_digest(args.config), "rows": {}, "status": "running"}
    try:
        with JsonlCore(config, stderr_path=args.out/"worker.log") as core:
            manifest["basis"] = core.identity
            manifest["load"] = core.load
            for i, text in enumerate(texts):
                if time.monotonic()-start > 1800:
                    raise TimeoutError("Feature collection whole-run bound")
                key = digest_bytes(canonical({"basis": core.identity, "text": text})).split(":")[1]
                path = args.out/(key+".json")
                if path.exists():
                    row = json.loads(path.read_text())
                    if row.get("basis") != core.identity or row.get("text") != text:
                        raise ValueError("Feature cache identity differs")
                else:
                    row = {"basis": core.identity, "text": text, **core.encode(text)}
                    path.write_text(json.dumps(row, ensure_ascii=False)+"\n")
                manifest["rows"][digest_bytes(text.encode())] = {"file": path.name, "digest": file_digest(path)}
                print(json.dumps({"completed": i+1, "total": len(texts), "seconds": round(time.monotonic()-start, 2)}), flush=True)
                (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        manifest["seconds"] = time.monotonic()-start
        (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
