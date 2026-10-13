"""Acquire a pinned public model, preserving published byte identities."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request


def acquire(repo, revision, root):
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true", timeout=30) as stream:
        metadata = json.load(stream)
    if metadata["sha"] != revision:
        raise ValueError("Published revision differs")
    root.mkdir(parents=True, exist_ok=False)
    (root / "hf-api.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    for item in metadata["siblings"]:
        name = item["rfilename"]
        if Path(name).name != name or not (name.endswith((".json", ".safetensors", ".model", ".txt", ".jinja", ".gguf")) or name in ("README.md", "LICENSE")):
            continue
        target = root / name
        temporary = root / (name + ".partial")
        h = hashlib.sha256()
        count = 0
        with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/{name}?download=true", timeout=60) as stream, temporary.open("xb") as out:
            for block in iter(lambda: stream.read(4 << 20), b""):
                out.write(block)
                h.update(block)
                count += len(block)
            out.flush()
            os.fsync(out.fileno())
        expected = item.get("lfs", {}).get("sha256")
        if (expected and h.hexdigest() != expected) or (item.get("size") is not None and count != item["size"]):
            raise ValueError("Published payload identity differs: " + name)
        os.rename(temporary, target)
        rows.append({"name": name, "bytes": count, "sha256": h.hexdigest(), "published_sha256": expected})
        print(name, count, flush=True)
    (root / "acquisition.json").write_text(json.dumps({"repo": repo, "revision": revision, "files": rows}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("revision")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    acquire(args.repo, args.revision, args.output)
