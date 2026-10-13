"""Qualified full-source adaptation with explicit training arithmetic.

The original BF16 forward remains the inference computation. Chunked FP32
input gradients are a separately named training recipe, not native-gradient
equivalence. A frozen-prefix record retains every input token and its basis.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.metadata

import torch
from safetensors.torch import load, save

from .core import digest_bytes, file_digest
from .memory import canonical
from .torch_source import LowRankLinear


RECIPE = "original-bf16-forward-chunked-fp32-input-gradient-v1"
FIRST_LAYER = 22


class _FrozenInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.weight, ctx.dtype = weight, x.dtype
        return torch.nn.functional.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, gradient):
        weight = ctx.weight
        result = torch.zeros((*gradient.shape[:-1], weight.shape[1]), dtype=torch.float32)
        for start in range(0, weight.shape[0], 4096):
            stop = min(start + 4096, weight.shape[0])
            result.add_(gradient[..., start:stop].float() @ weight[start:stop].float())
        return result.to(ctx.dtype), None, None


class FrozenLinear(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        if base.weight.requires_grad or base.weight.dtype != torch.bfloat16:
            raise ValueError("The frozen recipe requires original BF16 source weights")
        self.base = base

    def forward(self, x):
        if x.requires_grad and torch.is_grad_enabled():
            return _FrozenInput.apply(x, self.base.weight, self.base.bias)
        return self.base(x)


def attach(model):
    if len(model.model.layers) != 24 or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Attach to the complete frozen source model once")
    for layer in model.model.layers[-2:]:
        for name in ("gate_proj", "up_proj"):
            setattr(layer.mlp, name, LowRankLinear(getattr(layer.mlp, name)))
    layer = model.model.layers[FIRST_LAYER]
    layer.linear_attn.in_proj_qkv = LowRankLinear(layer.linear_attn.in_proj_qkv)
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if sum(p.numel() for p in parameters.values()) != 163840:
        raise ValueError("Qualified trainable parameter inventory differs")
    return parameters


def chunked_backward(model):
    """Install only after retaining the native reference measurement."""
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear) and (
                name == "lm_head" or name.startswith("model.layers.22.")
                or name.startswith("model.layers.23.")):
            parent, _, key = name.rpartition(".")
            owner = model.get_submodule(parent) if parent else model
            setattr(owner, key, FrozenLinear(module))
            count += 1
    if count == 0:
        raise ValueError("The input-gradient recipe was already installed")
    return count


def basis(source_manifest, model):
    import transformers.models.qwen3_5.modeling_qwen3_5 as upstream
    from . import torch_source
    residency = getattr(model, "_source_residency", None)
    return digest_bytes(canonical({
        "source": source_manifest, "config": model.config.to_dict(),
        "implementation": file_digest(__file__), "loader": file_digest(torch_source.__file__),
        "upstream": file_digest(upstream.__file__),
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "prefix_layers": FIRST_LAYER, "recipe": RECIPE,
        "source_residency": (file_digest(__import__(residency.__module__, fromlist=["__file__"]).__file__)
                             if residency is not None else None),
    }))


def encode(tokenizer, messages, target, *, reasoning):
    if not isinstance(reasoning, str) or not reasoning.strip() or len(reasoning.encode()) > 4096:
        raise ValueError("Reasoning-enabled teaching needs an attributed, checked rationale")
    if "<think>" in reasoning or "</think>" in reasoning or "</think>" in target:
        raise ValueError("Teaching text contains a reserved reasoning boundary")
    prompt = tokenizer.apply_chat_template(messages, enable_thinking=True,
        add_generation_prompt=True, tokenize=True, return_dict=False)
    answer = tokenizer.encode(reasoning.strip() + "\n</think>\n\n" + target, add_special_tokens=False) + [tokenizer.eos_token_id]
    if not 1 <= len(answer) <= 256 or len(prompt) + len(answer) > 512:
        raise ValueError("The complete training example exceeds its declared extent")
    return {"input_ids": prompt + answer[:-1], "target_ids": answer, "answer_start": len(prompt)-1}


def _copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy(v) for v in value)
    if value is None or isinstance(value, (int, bool, str, float)):
        return value
    raise ValueError("Unsupported frozen-prefix input " + type(value).__name__)


def cache_prefix(model, example, *, source_basis):
    captured = {}
    def hook(index):
        def capture(module, args, kwargs):
            captured[index] = {"kwargs": _copy(kwargs)}
            if index == FIRST_LAYER:
                captured[index]["hidden"] = args[0].detach().clone()
        return capture
    handles = [model.model.layers[i].register_forward_pre_hook(hook(i), with_kwargs=True)
               for i in (22, 23)]
    try:
        with torch.no_grad():
            model.model(torch.tensor([example["input_ids"]]), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return {"schema": "nanoai.frozen-prefix.v1", "basis": source_basis,
            "example": deepcopy(example), "layers": captured}


def loss(model, example, *, prefix=None, source_basis=None):
    if prefix is None:
        hidden = model.model(torch.tensor([example["input_ids"]]), use_cache=False).last_hidden_state
    else:
        if (prefix["schema"] != "nanoai.frozen-prefix.v1" or prefix["basis"] != source_basis
                or prefix["example"] != example):
            raise ValueError("Frozen-prefix source or complete token sequence changed")
        hidden = prefix["layers"][22]["hidden"]
        for index in (22, 23):
            hidden = model.model.layers[index](hidden, **prefix["layers"][index]["kwargs"])
        hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden[:, example["answer_start"]:, :]).float()[0]
    return torch.nn.functional.cross_entropy(logits, torch.tensor(example["target_ids"])), logits


def candidate_bytes(parameters, *, source_basis):
    return save({n: p.detach().contiguous() for n, p in parameters.items()},
                metadata={"basis": source_basis, "recipe": RECIPE})


def restore_candidate(parameters, data, *, expected_identity, expected_basis):
    import json
    import struct
    if not isinstance(data, bytes) or not 8 <= len(data) <= 32 << 20:
        raise ValueError("Candidate byte extent differs")
    if digest_bytes(data) != expected_identity:
        raise ValueError("Candidate bytes changed")
    length = struct.unpack("<Q", data[:8])[0]
    if not 2 <= length <= min(len(data)-8, 65536):
        raise ValueError("Candidate header extent differs")
    header = json.loads(data[8:8+length])
    if header.get("__metadata__") != {"basis": expected_basis, "recipe": RECIPE}:
        raise ValueError("Candidate source basis or arithmetic recipe changed")
    tensors = load(data)
    if set(tensors) != set(parameters):
        raise ValueError("Candidate parameter inventory differs")
    for name, parameter in parameters.items():
        value = tensors[name]
        if (value.shape != parameter.shape or value.dtype != parameter.dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError("Candidate parameter contract differs")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(tensors[name])


@torch.no_grad()
def generate(model, tokenizer, messages, *, max_tokens=None):
    prompt = tokenizer.apply_chat_template(messages, enable_thinking=True,
        add_generation_prompt=True, tokenize=True, return_dict=False)
    context = model.config.max_position_embeddings
    if len(prompt) >= context or (max_tokens is not None and (type(max_tokens) is not int or not 1 <= max_tokens <= context)):
        raise ValueError("Generation extent differs")
    ids = torch.tensor([prompt])
    output = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_tokens, max_length=context,
        do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    tokens = output[0, len(prompt):].tolist()
    end = tokenizer.convert_tokens_to_ids("</think>")
    final_tokens = tokens[tokens.index(end) + 1:] if end in tokens else []
    return {"text": tokenizer.decode(final_tokens, skip_special_tokens=True).strip(), "token_ids": tokens,
            "finish_reason": "stop" if tokens[-1] == tokenizer.eos_token_id else "length",
            "policy": "greedy-token-v1"}
