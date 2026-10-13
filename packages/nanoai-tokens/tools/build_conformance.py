"""Write small independent NMT1 acceptance fixtures shared by all consumers."""
import hashlib
import json
from pathlib import Path
import struct


def container(rows, *, stop=.25, count=None, magic=b"NMT1", trailing=b""):
    body = (struct.pack("<4sId", magic, len(rows) if count is None else count, stop)
            + hashlib.shake_256(b"conformance-vocabulary").digest(64)
            + hashlib.shake_256(b"conformance-inventory").digest(64))
    for common_id, piece, probability in rows:
        body += struct.pack("<IHd", common_id, len(piece), probability) + piece
    body += trailing
    return body + hashlib.shake_256(body).digest(64)


def main():
    root = Path(__file__).resolve().parents[1] / "tests/conformance"
    root.mkdir(parents=True, exist_ok=True)
    rows = [(i * 2, bytes([i]), 1 / 256) for i in range(256)]
    fixtures = [("valid-byte-law", container(rows), True, "complete byte alphabet with positive normalized probabilities")]
    for name, options, reason in [
        ("stop-zero", {"stop": 0.0}, "terminal probability in the open unit interval"),
        ("stop-one", {"stop": 1.0}, "terminal probability in the open unit interval"),
        ("stop-nan", {"stop": float("nan")}, "finite terminal probability"),
        ("count-low", {"count": 255}, "active row lower bound"),
        ("count-high", {"count": 131073}, "active row upper bound"),
        ("magic", {"magic": b"NMT2"}, "format magic"),
        ("trailing", {"trailing": b"x"}, "exact record extent")]:
        fixtures.append((name, container(rows, **options), False, reason))
    for name, index, replacement, reason in [
        ("probability-zero", 0, (0, b"\0", 0.0), "positive emission probability"),
        ("probability-nan", 0, (0, b"\0", float("nan")), "finite emission probability"),
        ("probability-sum", 0, (0, b"\0", .5), "normalized emission mass"),
        ("id-duplicate", 1, (0, b"\1", 1/256), "strict common ID ordering"),
        ("id-range", 255, (2_000_000, b"\xff", 1/256), "common ID upper bound"),
        ("piece-empty", 0, (0, b"", 1/256), "nonempty literal piece"),
        ("piece-long", 0, (0, b"\0"*129, 1/256), "literal piece upper bound"),
        ("piece-order", 1, (2, b"\0", 1/256), "strict byte ordering"),
        ("fallback-missing", 255, (510, b"\xff\xff", 1/256), "complete single-byte fallback")]:
        changed = list(rows); changed[index] = replacement
        fixtures.append((name, container(changed), False, reason))
    raw = container(rows)
    truncated = raw[:-65]
    fixtures.extend([
        ("truncated-record", truncated + hashlib.shake_256(truncated).digest(64), False, "complete final piece"),
        ("corrupt-footer", raw[:-1] + bytes([raw[-1] ^ 1]), False, "payload digest"),
        ("container-short", b"NMT1", False, "minimum container extent")])
    manifest = {"schema": "nanoai.nmt-conformance.v1", "provenance": "Independent authored NMT1 byte fixtures; no imported model data.",
                "contract": "accept describes parser admission; violation identifies the required invariant, not a language-specific exception string.", "cases": []}
    for name, data, accept, reason in fixtures:
        filename = name + ".nmt"
        (root / filename).write_bytes(data)
        manifest["cases"].append({"id": name, "file": filename, "bytes": len(data), "accept": accept,
            "violation": None if accept else reason, "description": reason,
            "file_identity": "shake256:" + hashlib.shake_256(data).hexdigest(64)})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"root": str(root), "cases": len(fixtures), "bytes": sum(len(row[1]) for row in fixtures)}))


if __name__ == "__main__": main()
