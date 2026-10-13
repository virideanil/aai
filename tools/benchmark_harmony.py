"""Measure exact codecs and reference projections on complete source matrices.

This is a CPU reference experiment. Its F32 accumulation does not replace the
native model's arithmetic or establish whole-model inference performance.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import resource
import struct
import sys
import time

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from nanoai.core import file_digest
from nanoai.harmony import Library, Matrix, MODES


def samples(call, count=5):
    times = []
    result = None
    for _ in range(count):
        start = time.perf_counter_ns()
        result = call()
        times.append((time.perf_counter_ns() - start) / 1e6)
    return result, {"samples_ms": times, "median_ms": float(np.median(times)),
                    "p95_ms": float(np.percentile(times, 95))}


def exact_float(words):
    return (words.astype(np.uint32) << 16).view(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--layer", type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with args.source.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
    library = Library(args.library)
    rng = np.random.default_rng(20261013)
    record = {"schema": "nanoai.harmony-measurement.v1", "source": str(args.source.resolve()),
        "source_digest": file_digest(args.source), "library_digest": file_digest(args.library),
        "benchmark_digest": file_digest(__file__), "matrix_results": [], "mlp_results": [],
        "scope": "Complete original BF16 matrices and one MLP; CPU F32 reference arithmetic",
        "timing_policy": "3 calibration repetitions followed by 5 validation repetitions; separate arrays",
        "precision": "All stored 16-bit words reconstructed exactly; BF16-exact projection inputs",
        "veclib_threads": os.environ["VECLIB_MAXIMUM_THREADS"]}
    start = time.monotonic()

    def save():
        record["wall_seconds"] = time.monotonic() - start
        record["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.out / "result.json").write_text(json.dumps(record, indent=2) + "\n")

    tensors = {}
    for kind in ("gate", "up", "down"):
        name = f"model.language_model.layers.{args.layer}.mlp.{kind}_proj.weight"
        info = header[name]
        if info["dtype"] != "BF16" or len(info["shape"]) != 2:
            raise ValueError("Expected an original BF16 source matrix")
        words = np.memmap(args.source, mode="r", dtype="<u2", shape=tuple(info["shape"]),
                          offset=8 + size + info["data_offsets"][0])
        tensors[kind] = words
        reference = {}
        for mode in MODES:
            setup = time.monotonic()
            with Matrix(library, words, mode) as matrix:
                result = {"tensor": name, "shape": list(words.shape), "mode": mode,
                    "encode_seconds": time.monotonic() - setup, "stats": matrix.stats(), "routes": []}
                for threads in (1, 4):
                    restored, timing = samples(lambda: matrix.unpack(threads))
                    np.testing.assert_array_equal(restored, words)
                    result[f"decode_{threads}"] = timing
                del restored
                setup = time.monotonic(); matrix.prepare_cache(4)
                result["cache_build_seconds"] = time.monotonic() - setup
                result["stats_with_cache"] = matrix.stats()
                for batch in (1, 8, 32):
                    if batch not in reference:
                        inputs = []
                        for _ in range(2):
                            f = rng.normal(0, 0.1, (batch, words.shape[1])).astype(np.float32)
                            inputs.append(exact_float((f.view(np.uint32) >> 16).astype(np.uint16)))
                        reference[batch] = inputs
                    calibration, validation = reference[batch]
                    rows = np.linspace(0, words.shape[0] - 1, 32, dtype=int)
                    selected = exact_float(words[rows]).astype(np.float64)
                    oracle = validation.astype(np.float64) @ selected.T
                    absolute = np.abs(validation.astype(np.float64)) @ np.abs(selected).T
                    bound = 2 * np.finfo(np.float32).eps * words.shape[1] * absolute + 1e-8
                    for cache, threads in ((False, 1), (False, 4), (True, 1), (True, 4), ("full", 1)):
                        options = {"threads": threads, "cached": cache is True, "dense_baseline": cache == "full"}
                        _, training = samples(lambda: matrix.project(calibration, **options), 3)
                        output, testing = samples(lambda: matrix.project(validation, **options), 5)
                        if not np.all(np.abs(output[:, rows] - oracle) <= bound):
                            raise AssertionError("Projection exceeds the declared F32 reference error bound")
                        result["routes"].append({"batch": batch, "cache": cache, "threads": threads,
                            "calibration": training, "validation": testing,
                            "max_abs_error_checked_rows": float(np.max(np.abs(output[:, rows] - oracle)))})
                record["matrix_results"].append(result); save()
                print(kind, mode, round(result["stats"]["serialized_block_bytes"] / words.nbytes, 4), flush=True)

    # Compose all three complete matrices. Parallel branches share four workers.
    mlp_reference = {}
    for mode in ("binary", "binary_trit5", "mixed"):
        matrices = {kind: Matrix(library, words, mode) for kind, words in tensors.items()}
        try:
            for matrix in matrices.values(): matrix.prepare_cache(4)
            for batch in (1, 8, 32):
                if batch not in mlp_reference:
                    values = rng.normal(0, 0.1, (batch, tensors["gate"].shape[1])).astype(np.float32)
                    mlp_reference[batch] = exact_float((values.view(np.uint32) >> 16).astype(np.uint16))
                x = mlp_reference[batch]
                for parallel in (False, True):
                    for cached in (False, True):
                        with ThreadPoolExecutor(max_workers=2) as pool:
                            def forward():
                                options = {"threads": 2 if parallel else 4, "cached": cached}
                                if parallel:
                                    gate = pool.submit(matrices["gate"].project, x, **options)
                                    up = matrices["up"].project(x, **options)
                                    gate = gate.result()
                                else:
                                    gate = matrices["gate"].project(x, **options)
                                    up = matrices["up"].project(x, **options)
                                activated = (gate / (1 + np.exp(-gate))) * up
                                return matrices["down"].project(activated, threads=4, cached=cached)
                            output, timing = samples(forward)
                        key = (batch, "output")
                        if key not in mlp_reference: mlp_reference[key] = output
                        np.testing.assert_array_equal(output, mlp_reference[key])
                        record["mlp_results"].append({"mode": mode, "batch": batch, "parallel_branches": parallel,
                            "cached": cached, "timing": timing, "bit_identical_reference_route": True})
                        save()
        finally:
            for matrix in matrices.values(): matrix.close()
    record["selection"] = []
    for kind in ("gate", "up", "down"):
        rows = [r for r in record["matrix_results"] if f".{kind}_proj." in r["tensor"]]
        for batch in (1, 8, 32):
            candidates = [(r["mode"], route) for r in rows for route in r["routes"] if route["batch"] == batch]
            mode, route = min(candidates, key=lambda pair: pair[1]["calibration"]["median_ms"])
            best_validation = min(r["validation"]["median_ms"] for _, r in candidates)
            record["selection"].append({"matrix": kind, "batch": batch, "mode": mode,
                "cache": route["cache"], "threads": route["threads"],
                "validation_ms": route["validation"]["median_ms"],
                "regret_vs_best_observed": route["validation"]["median_ms"] / best_validation})
    record["complete"] = True; save()


if __name__ == "__main__":
    main()
