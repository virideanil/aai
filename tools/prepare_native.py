"""Pin local native model/runtime inputs and create the JSONL core config."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoai.core import file_digest
from nanoai.precision import PRECISION_PROFILE, inspect_bf16
from nanoai.tensor_store import TensorStore
from nanoai.sampling import GREEDY, QWEN35_THINKING


def entry(path):
    path = path.resolve(strict=True)
    return {"path": str(path), "digest": file_digest(path), "bytes": path.stat().st_size}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--server", type=Path, required=True)
    p.add_argument("--library", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--catalogue", type=Path)
    p.add_argument("--rss-limit-gib", type=int, default=7, choices=range(1, 11))
    p.add_argument("--weight-access", choices=("mapped", "stream"), default="mapped")
    p.add_argument("--backend", choices=("cpu", "metal"), default="cpu")
    p.add_argument("--ring-mb", type=int, choices=(256, 512, 1024), default=512)
    p.add_argument("--storage-codec", choices=("direct", "mixed"), default="direct")
    p.add_argument("--decoder-library", type=Path)
    p.add_argument("--context-tokens", type=int, default=4096)
    p.add_argument("--sampling", choices=("greedy", "qwen35-thinking"), default="greedy")
    args = p.parse_args()
    if not 512 <= args.context_tokens <= 262144:
        p.error("Context capacity must fit the qualified model extent")
    if args.backend == "metal" and args.weight_access != "mapped":
        p.error("Metal requires mapped original weights")
    if (args.storage_codec != "direct") != bool(args.decoder_library):
        p.error("Mixed storage requires its exact decoder library; direct storage uses the native model file")
    precision = inspect_bf16(args.model)
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    manifest = {"schema": "nanoai.native-model.v1", "model": entry(args.model), "server": entry(args.server),
        "libraries": [entry(path) for path in args.library], "precision_profile": PRECISION_PROFILE,
        "precision_inspection": precision}
    manifest["tensor_catalogue"] = TensorStore(args.catalogue or args.output / "tensors.sqlite3", create=True).register(
        args.model, expected=manifest["model"]["digest"], basis=manifest["model"]["digest"],
        recipe="original-bf16-gguf-v1", source="native-profile:original-model")
    manifest["tensor_catalogue"]["database"] = os.path.relpath(manifest["tensor_catalogue"]["database"], args.output.resolve())
    manifest["context_tokens"] = args.context_tokens
    manifest["sampling"] = dict(GREEDY if args.sampling == "greedy" else QWEN35_THINKING)
    manifest["runtime"] = {"weight_access": args.weight_access, "ring_mb": args.ring_mb, "readers": 4,
                           "backend": args.backend}
    if args.storage_codec == "mixed":
        from nanoai.harmony import Library
        source = manifest["model"]
        before = args.model.stat()
        packed = args.output / "weights.nbh"
        start = time.monotonic()
        decoder = entry(args.decoder_library)
        Library(args.decoder_library).pack_file(args.model, packed, mode="mixed")
        after = args.model.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("Source changed during packing")
        manifest["model"] = {**entry(packed), "format": "NBH1", "restored_digest": source["digest"],
            "restored_bytes": source["bytes"]}
        manifest["decoder"] = decoder
        manifest["storage_preparation"] = {"codec": "mixed", "seconds": time.monotonic() - start,
            "original_bytes": source["bytes"], "packed_bytes": packed.stat().st_size,
            "scope": "Lossless file storage; the native consumer restores ordinary BF16 weights"}
    manifest_path = args.output / "model.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    config = {"command": [sys.executable, str(root / "tools/native_entry.py")], "precision_profile": PRECISION_PROFILE,
        "manifest": str(manifest_path.resolve()), "source_files": [str(root / path) for path in
            ("src/nanoai/native_worker.py", "src/nanoai/core.py", "src/nanoai/precision.py", "src/nanoai/harmony.py",
             "src/nanoai/tensor_store.py", "src/nanoai/sqlite_journal.py", "src/nanoai/gguf.py",
             "src/nanoai/sampling.py", "tools/native_entry.py")],
        "request_seconds": None, "startup_seconds": 120, "rss_limit_bytes": args.rss_limit_gib * 1024**3,
        "physical_footprint_limit_bytes": 3 * 1024**3,
        "swap_growth_limit_bytes": 512 * 1024**2, "include_process_group": True}
    (args.output / "core.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
