"""Check a frozen artifact in native encoding and a fixed BF16 data consumer."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoai.core import digest_bytes, file_digest, process_memory, swap_bytes
from nanoai.master_tokenizer import MasterTokenizer
from nanoai.master_data import causal_batch, model_contract, validate_batch


def percentile(values, p):
    return sorted(values)[max(0, math.ceil(p * len(values)) - 1)]


def native_run(executable, artifact, identity, rows):
    samples = []
    start = time.perf_counter()
    process = subprocess.Popen([str(executable), str(artifact), identity], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    def monitor():
        while process.poll() is None:
            try:
                samples.append(process_memory(process.pid))
            except (ProcessLookupError, subprocess.SubprocessError):
                pass
            time.sleep(.005)
    sampler = threading.Thread(target=monitor)
    sampler.start()
    output, error = process.communicate("".join(raw.hex() + "\n" for raw in rows), timeout=60)
    sampler.join()
    elapsed = time.perf_counter() - start
    if process.returncode != 0 or error:
        raise ValueError("Native consumer failure: " + error)
    records = [json.loads(line) for line in output.splitlines()]
    if len(records) != len(rows):
        raise ValueError("Native record count")
    return records, elapsed, samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.native = args.native.resolve(); args.artifact = args.artifact.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    before_swap = swap_bytes()
    start = time.perf_counter()
    tokenizer = MasterTokenizer.load(args.artifact, expected_identity=args.identity)
    load_seconds = time.perf_counter() - start
    resident = process_memory(os.getpid())
    rows = [s.encode() for s in json.loads(args.corpus.read_text())["lines"]]
    confirmation = [s["text"].encode() for s in json.loads(args.confirmation.read_text())["rows"]]
    rng = random.Random(10213)
    probes = [b"", bytes(range(256)), b"\xc0\x80\xff\x00\\u0041\n"] + [rng.randbytes(n) for n in (1, 127, 1024, 8192)]
    panel = rows + confirmation + probes
    reference, python_seconds = [], []
    for raw in panel:
        start = time.perf_counter()
        record = tokenizer.encode(raw)
        python_seconds.append(time.perf_counter() - start)
        if tokenizer.decode(record) != raw:
            raise ValueError("Python byte restoration")
        reference.append(record)
    records, native_seconds, memory = native_run(args.native, args.artifact, tokenizer.identity, panel)
    error = 0.0
    for raw, expected, actual in zip(panel, reference, records):
        if actual["spans"] != expected["spans"] or tokenizer.decode(actual) != raw or actual["decoded_hex"] != raw.hex():
            raise ValueError("Native byte/ID/span parity")
        for field in ("forward_logp", "viterbi_logp"):
            difference = abs(actual[field] - expected[field]); error = max(error, difference)
            if difference > 1e-8 * max(1, len(raw)):
                raise ValueError("Native likelihood parity")
    warm, wall, warm_memory = native_run(args.native, args.artifact, tokenizer.identity, rows * 20)
    warm = warm[len(rows):]  # first pass warms the dictionary and allocator
    timings = [r["encode_ns"] / 1e6 for r in warm]
    total_bytes = sum(map(len, rows)) * 19
    warm_sum_seconds = sum(timings) / 1000
    startup = []
    for _ in range(12):
        _, elapsed, _ = native_run(args.native, args.artifact, tokenizer.identity, [b"a"])
        startup.append(elapsed * 1000)
    batch = causal_batch(tokenizer, [b"", b"a\x00\xff", rows[0], confirmation[0], confirmation[4]])
    batch_path = args.out / "causal-batch.json"
    batch_path.write_text(json.dumps(batch, ensure_ascii=False, separators=(",", ":")) + "\n")
    recovered = json.loads(batch_path.read_text())
    if validate_batch(tokenizer, recovered) != [b"", b"a\x00\xff", rows[0], confirmation[0], confirmation[4]]:
        raise ValueError("Reloaded model batch differs")
    import torch
    torch.manual_seed(10213)
    torch.set_num_threads(2)
    # Fixed random weights only exercise index, dtype, output and loss contracts.
    # This fixture is not a language model quality baseline and makes no update.
    embedding = torch.nn.Embedding(batch["contract"]["model_rows"], 64, dtype=torch.bfloat16)
    embedding.requires_grad_(False)
    weight_before = digest_bytes(embedding.weight.detach().view(torch.uint8).numpy().tobytes())
    with torch.no_grad():
        inputs = torch.tensor(batch["input_ids"], dtype=torch.long)
        targets = torch.tensor(batch["labels"], dtype=torch.long)
        embedded = embedding(inputs)
        logits = embedded @ embedding.weight.T
        loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
    weight_after = digest_bytes(embedding.weight.detach().view(torch.uint8).numpy().tobytes())
    if weight_before != weight_after or not torch.isfinite(loss) or embedded.dtype != torch.bfloat16 or logits.dtype != torch.bfloat16:
        raise ValueError("Fixed BF16 input/output consumer")
    memories = memory + warm_memory
    report = {"schema": "nanoai.master-qualification.v1", "artifact_identity": tokenizer.identity,
        "artifact_file_identity": file_digest(args.artifact), "artifact_bytes": args.artifact.stat().st_size,
        "native_executable_identity": file_digest(args.native), "runtime": platform.platform(),
        "compiler": subprocess.check_output(["clang++", "--version"], text=True).splitlines()[0],
        "checks": {"python_native_rows": len(panel), "byte_id_span_failures": 0, "maximum_logp_difference": error},
        "warm_native": {"rows": len(warm), "bytes": total_bytes, "median_ms": statistics.median(timings),
            "p95_ms": percentile(timings, .95), "p99_ms": percentile(timings, .99),
            "bytes_per_second": total_bytes / warm_sum_seconds, "whole_batch_wall_seconds": wall,
            "scope": "Encoding, including token record byte digest; excludes serialization, decode and process startup"},
        "startup_one_byte": {"runs": len(startup), "median_ms": statistics.median(startup), "p95_ms": percentile(startup, .95),
            "scope": "Process startup, artifact read/hash, dictionary build, one encode/decode, JSON, shutdown and monitor overhead"},
        "python": {"load_seconds": load_seconds, "panel_encode_seconds": sum(python_seconds), "after_load": resident},
        "native_sampled_peak": {key: max((m[key] or 0 for m in memories), default=0) for key in ("rss_bytes", "physical_footprint_bytes")},
        "data_consumer": {"contract": model_contract(tokenizer), "batch_identity": file_digest(batch_path),
            "batch_shape": list(inputs.shape), "output_shape": list(logits.shape), "embedding_dtype": str(embedded.dtype),
            "output_dtype": str(logits.dtype), "torch": torch.__version__, "fixed_fixture_loss": loss.item(),
            "weight_identity_before": weight_before, "weight_identity_after": weight_after, "neural_parameter_updates": 0,
            "embedding_bytes_at_width_64": embedding.weight.numel() * embedding.weight.element_size(),
            "scope": "Fixed BF16 index/output/loss consumer only; no trained language or generation claim"},
        "swap_before": before_swap, "swap_after": swap_bytes()}
    (args.out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
