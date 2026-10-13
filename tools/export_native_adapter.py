"""Export the qualified 2B BF16 update through the pinned native converter.

The exported tensors preserve the candidate coefficients. Native and PyTorch
accumulation differ; this conversion does not assert identical model logits.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoai.core import file_digest


MAPPING = {"model.layers.22.linear_attn.in_proj_qkv": "blk.22.attn_qkv.weight",
           **{f"model.layers.{layer}.mlp.{name}_proj": f"blk.{layer}.ffn_{name}.weight"
              for layer in (22, 23) for name in ("gate", "up")}}
RECIPE = "original-bf16-forward-chunked-fp32-input-gradient-v1"


def read_candidate(path, expected_identity, expected_basis, source):
    import torch
    from safetensors import safe_open
    if Path(path).stat().st_size > 32 << 20 or file_digest(path) != expected_identity:
        raise ValueError("Candidate identity or byte extent differs")
    config = json.loads((Path(source) / "config.json").read_text())["text_config"]
    expected = {"hidden_size": 2048, "intermediate_size": 6144, "num_hidden_layers": 24,
                "linear_num_key_heads": 16, "linear_num_value_heads": 16,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Adapter export requires the qualified 2B layout with equal K and V heads")
    tensors = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if handle.metadata() != {"basis": expected_basis, "recipe": RECIPE}:
            raise ValueError("Candidate learning provenance differs")
        if set(handle.keys()) != {n + suffix for n in MAPPING for suffix in (".a", ".b")}:
            raise ValueError("Candidate projection inventory differs")
        for name in handle.keys():
            value = handle.get_tensor(name)
            shape = (2048, 4) if name.endswith(".a") else (4, 6144)
            if tuple(value.shape) != shape or value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
                raise ValueError("Candidate tensor contract differs: " + name)
            tensors[name] = value
    return tensors


def export(args):
    import numpy as np
    import torch
    from safetensors.torch import save_file
    tensors = read_candidate(args.candidate, args.identity, args.basis, args.source)
    if args.out.exists():
        raise ValueError("Keep the previous export; use a fresh directory")
    args.out.mkdir(parents=True)
    peft = {"base_model.model." + n + ".lora_" + suffix.upper() + ".weight":
            tensors[n + "." + suffix].T.contiguous() for n in MAPPING for suffix in ("a", "b")}
    save_file(peft, str(args.out / "adapter_model.safetensors"))
    (args.out / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 4, "lora_alpha": 32,
        "lora_dropout": 0, "bias": "none", "inference_mode": True,
        "target_modules": ["gate_proj", "up_proj", "in_proj_qkv"]}, indent=2) + "\n")
    output = args.out / "adapter-bf16.gguf"
    command = [sys.executable, str(args.converter.resolve()), "--base", str(args.source.resolve()),
               "--outtype", "bf16", "--no-lazy", "--outfile", str(output.resolve()), str(args.out.resolve())]
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "HF_HUB_DISABLE_TELEMETRY": "1"}
    with (args.out / "conversion.log").open("w") as log:
        subprocess.run(command, check=True, env=environment, stdout=log, stderr=subprocess.STDOUT)
    sys.path.insert(0, str(args.converter.resolve().parent / "gguf-py"))
    import gguf
    reader = gguf.GGUFReader(str(output))
    actual = {t.name: t for t in reader.tensors}
    expected_names = {target + ".lora_" + suffix for target in MAPPING.values() for suffix in ("a", "b")}
    if set(actual) != expected_names:
        raise ValueError("Native adapter tensor inventory differs")
    for name, target in MAPPING.items():
        for suffix in ("a", "b"):
            tensor = actual[target + ".lora_" + suffix]
            reference = tensors[name + "." + suffix].T.contiguous()
            if tensor.tensor_type != gguf.GGMLQuantizationType.BF16:
                raise ValueError("Native adapter reduced coefficient precision")
            bits = np.asarray(tensor.data).view(np.uint16).reshape(tuple(reference.shape))
            if not np.array_equal(bits, reference.view(torch.uint16).numpy()):
                raise ValueError("Native adapter coefficient bits differ")
    metadata = {name: reader.fields[name].contents() for name in
                ("general.architecture", "general.type", "adapter.type", "adapter.lora.alpha")}
    if metadata != {"general.architecture": "qwen35", "general.type": "adapter",
                    "adapter.type": "lora", "adapter.lora.alpha": 32.0}:
        raise ValueError("Native adapter architecture or scale differs")
    result = {"schema": "nanoai.native-adapter-export.v1", "candidate": args.identity,
              "basis": args.basis, "recipe": RECIPE, "source_config": file_digest(args.source / "config.json"),
              "converter": file_digest(args.converter), "implementation": file_digest(__file__),
              "file": {"path": str(output.resolve()), "bytes": output.stat().st_size, "digest": file_digest(output)},
              "metadata": metadata, "rank": 4, "scale": 8.0, "tensor_count": len(actual),
              "coefficient_bits_equal": True, "v_head_permutation": "identity for qualified 16 K and 16 V heads",
              "limits": "Coefficient conversion only. Actual native execution and arithmetic comparison remain separate."}
    from nanoai.tensor_store import TensorStore
    store = TensorStore(args.catalogue or args.out / "tensors.sqlite3", create=True)
    candidate = store.register(args.candidate, expected=args.identity, basis=args.basis,
        recipe=RECIPE, source="native-export:candidate")
    result["tensor_catalogue"] = store.register(output, expected=result["file"]["digest"], basis=args.basis,
        recipe="bf16-lora-gguf-coefficient-preserving-v1", source="native-export:" + args.identity,
        parent=candidate["version"])
    (args.out / "conversion.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "source", "converter", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("identity", "basis"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--catalogue", type=Path)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
