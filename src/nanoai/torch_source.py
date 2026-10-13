"""Experimental full-source Qwen3.5 CPU computation and hybrid-state boundaries.

Original BF16/F32 tensors are assigned from safetensors views. The caller owns
source manifests, process/resource guards, task dependencies and candidate
admission. This module performs no download, publication or default selection.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, DynamicCache, Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from .core import digest_bytes, file_digest
from .memory import canonical


def load_full_text(source):
    """Load the complete qualified 24-layer text computation from local sources."""
    if importlib.metadata.version("transformers") != "5.19.0":
        raise ValueError("This experimental source adapter requires Transformers 5.19.0")
    source = Path(source)
    cfg = Qwen3_5TextConfig(**json.loads((source/"config.json").read_text())["text_config"])
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(cfg)
    state = {}
    for path in sorted(source.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("model.language_model."):
                    mapped = "model."+key.removeprefix("model.language_model.")
                    if mapped in state:
                        raise ValueError("Duplicate source tensor")
                    state[mapped] = handle.get_tensor(key)
    if len(state) != 320:
        raise ValueError("Complete source text tensor inventory differs")
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model.load_state_dict(state, strict=True, assign=True)
    model.lm_head.weight = model.model.embed_tokens.weight
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(cfg)
    actual = model.state_dict()
    if any(actual[k].data_ptr() != v.data_ptr() or actual[k].dtype != v.dtype for k, v in state.items()):
        raise ValueError("Original source storage or dtype changed during assignment")
    if any(v.is_meta for v in (*model.parameters(), *model.buffers())):
        raise ValueError("Unmaterialized source computation")
    if len(model.model.layers) != 24 or tuple(model.model.embed_tokens.weight.shape) != (248320, 2048):
        raise ValueError("Qualified complete model shape differs")
    if any(v.dtype not in (torch.bfloat16, torch.float32) for v in model.parameters()):
        raise ValueError("Original parameter precision differs")
    model.requires_grad_(False); model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=False)
    return model, tokenizer


class LowRankLinear(torch.nn.Module):
    """An independently saved BF16 update, with explicit causal-credit control."""
    def __init__(self, linear, *, rank=4, scale=8.0):
        super().__init__()
        if type(rank) is not int or not 1 <= rank <= 16 or not math.isfinite(scale):
            raise ValueError("Low-rank update extent")
        if linear.weight.dtype != torch.bfloat16:
            raise ValueError("The qualified projection retains original BF16")
        self.linear = linear
        self.scale = float(scale)
        self.a = torch.nn.Parameter(torch.empty(linear.in_features, rank, dtype=torch.bfloat16))
        self.b = torch.nn.Parameter(torch.zeros(rank, linear.out_features, dtype=torch.bfloat16))
        with torch.no_grad():
            self.a.uniform_(-1/math.sqrt(linear.in_features), 1/math.sqrt(linear.in_features))
        self.detach_parameters = False

    def forward(self, x):
        a = self.a.detach() if self.detach_parameters else self.a
        b = self.b.detach() if self.detach_parameters else self.b
        return self.linear(x)+(self.scale*((x@a)@b)).to(x.dtype)


def _cache_contract(config):
    import transformers.cache_utils as owner
    import transformers.models.qwen3_5.modeling_qwen3_5 as computation
    return {"implementation":file_digest(__file__), "upstream_cache":file_digest(owner.__file__),
            "upstream_computation":file_digest(computation.__file__),
            "transformers":importlib.metadata.version("transformers"),
            "torch":importlib.metadata.version("torch"), "config":digest_bytes(canonical(config.to_dict()))}


def export_hybrid_state(cache, config, *, basis):
    """Copy a complete prefilled CPU state at a boundary before known next input."""
    if not isinstance(basis, str) or not basis or len(cache.layers) != config.num_hidden_layers:
        raise ValueError("Hybrid state basis or layer count differs")
    rows = []
    for kind, layer in zip(config.layer_types, cache.layers):
        if kind == "full_attention":
            if not layer.is_initialized:
                raise ValueError("State boundary precedes complete prefill")
            row = {"kind":kind, "keys":layer.keys.detach().clone(), "values":layer.values.detach().clone()}
        elif kind == "linear_attention":
            if (layer.number_of_states != 1 or layer.record_past or not layer.has_previous_state[0]
                    or not layer.is_conv_states_initialized[0] or not layer.is_recurrent_states_initialized[0]):
                raise ValueError("Linear state is incomplete or has a different contract")
            row = {"kind":kind, "conv":layer.conv_states[0].detach().clone(),
                   "recurrent":layer.recurrent_states[0].detach().clone(),
                   "kernel":layer.conv_kernel_size[0]}
        else:
            raise ValueError("Unqualified hybrid layer")
        rows.append(row)
    return {"schema":"nanoai.hybrid-state.v1", "boundary":"before-known-next-input",
            "basis":basis, "contract":_cache_contract(config), "sequence_length":cache.get_seq_length(),
            "layers":rows}


def restore_hybrid_state(record, config, *, basis):
    """Restore every hybrid variable only under its exact current source basis."""
    if (record.get("schema") != "nanoai.hybrid-state.v1" or record.get("basis") != basis
            or record.get("boundary") != "before-known-next-input" or record.get("contract") != _cache_contract(config)
            or len(record["layers"]) != config.num_hidden_layers or type(record["sequence_length"]) is not int
            or not 1 <= record["sequence_length"] <= 4096):
        raise ValueError("Hybrid state source, runtime or boundary changed")
    cache = DynamicCache(config=config)
    for index, (kind, row) in enumerate(zip(config.layer_types, record["layers"])):
        if row.get("kind") != kind:
            raise ValueError("Hybrid layer type changed")
        if kind == "full_attention":
            shape = (1, config.num_key_value_heads, record["sequence_length"], config.head_dim)
            expected = {"keys":(shape, torch.bfloat16), "values":(shape, torch.bfloat16)}
        elif kind == "linear_attention":
            dim = 2*config.linear_num_key_heads*config.linear_key_head_dim+config.linear_num_value_heads*config.linear_value_head_dim
            expected = {"conv":((1, dim, config.linear_conv_kernel_dim), torch.bfloat16),
                        "recurrent":((1, config.linear_num_value_heads, config.linear_key_head_dim,
                                       config.linear_value_head_dim), torch.float32)}
            if row.get("kernel") != config.linear_conv_kernel_dim:
                raise ValueError("Convolution context changed")
        else:
            raise ValueError("Unqualified hybrid layer")
        for name, (shape, dtype) in expected.items():
            value = row[name]
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype
                    or value.device.type != "cpu" or not bool(torch.isfinite(value).all())):
                raise ValueError("Hybrid state tensor shape, precision or value differs")
        layer = cache.layers[index]
        if kind == "full_attention":
            layer.update(row["keys"].clone(), row["values"].clone())
        else:
            cache.update_conv_state(row["conv"].clone(), index, conv_kernel_size=row["kernel"])
            cache.update_recurrent_state(row["recurrent"].clone(), index)
    return cache
