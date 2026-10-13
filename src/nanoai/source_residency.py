"""Experimental synchronous CPU residency hints for immutable source tensors.

Each tensor remains a read-only view of the original safetensors file. The
kernel's DONTNEED hint may release file pages after their forward consumer.
It establishes neither a physical-memory cap nor a speed improvement.
"""
from __future__ import annotations

import json
import math
import mmap
from pathlib import Path
import struct
import warnings

import torch


class SourceResidency:
    def __init__(self, source):
        if not hasattr(mmap.mmap, "madvise"):
            raise RuntimeError("This platform has no mmap residency advice")
        self.maps = []
        self.ranges = {}
        self.handles = []
        self.calls = 0
        self.advised_bytes = 0
        self.state = {}
        for path in sorted(Path(source).glob("*.safetensors")):
            with path.open("rb") as stream:
                length_bytes = stream.read(8)
                if len(length_bytes) != 8:
                    raise ValueError("Truncated source header")
                length = struct.unpack("<Q", length_bytes)[0]
                if not 2 <= length <= min(path.stat().st_size - 8, 16 << 20):
                    raise ValueError("Source header extent differs")
                header = json.loads(stream.read(length))
                mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            self.maps.append(mapped)
            for name, info in header.items():
                if not name.startswith("model.language_model."):
                    continue
                key = "model." + name.removeprefix("model.language_model.")
                if key in self.state:
                    raise ValueError("Duplicate source tensor")
                dtype = {"BF16": torch.bfloat16, "F32": torch.float32}.get(info["dtype"])
                shape = info["shape"]
                if dtype is None or not shape or any(type(x) is not int or x < 1 for x in shape):
                    raise ValueError("Source tensor type or shape differs")
                count = math.prod(shape)
                start, stop = info["data_offsets"]
                size = count * torch.empty((), dtype=dtype).element_size()
                if (type(start) is not int or type(stop) is not int or start < 0
                        or stop - start != size or 8 + length + stop > len(mapped)):
                    raise ValueError("Source tensor extent differs")
                offset = 8 + length + start
                # The consumer freezes these parameters; the OS enforces the
                # source mapping's read-only contract as well.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="The given buffer is not writable")
                    tensor = torch.frombuffer(mapped, dtype=dtype, count=count, offset=offset).reshape(shape)
                self.state[key] = tensor
                first = ((offset + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE
                end = ((offset + size) // mmap.PAGESIZE) * mmap.PAGESIZE
                self.ranges[key] = (mapped, first, max(0, end-first))
        if len(self.state) != 320:
            raise ValueError("Complete source text tensor inventory differs")

    def release(self, names):
        for name in names:
            mapped, start, size = self.ranges[name]
            if size:
                mapped.madvise(mmap.MADV_DONTNEED, start, size)
                self.calls += 1
                self.advised_bytes += size

    def install(self, model):
        if self.handles or hasattr(model, "_source_residency"):
            raise ValueError("Source residency is already installed")
        if any(p.requires_grad or p.device.type != "cpu" for p in model.parameters()):
            raise ValueError("Remap only the complete frozen CPU source before adaptation")
        state = {**self.state, "lm_head.weight": self.state["model.embed_tokens.weight"]}
        current = model.state_dict()
        if (set(current) != set(state) or any(current[n].shape != p.shape or current[n].dtype != p.dtype
                                            for n, p in state.items())):
            raise ValueError("Mapped source inventory, shape or precision differs")
        model.load_state_dict(state, strict=True, assign=True)
        model.lm_head.weight = model.model.embed_tokens.weight
        model.requires_grad_(False)
        groups = [(model.model.embed_tokens, ["model.embed_tokens.weight"]),
                  (model.lm_head, ["model.embed_tokens.weight"])]
        for index, layer in enumerate(model.model.layers):
            groups.append((layer, [n for n in self.ranges if n.startswith(f"model.layers.{index}.")]))
        for module, names in groups:
            def release_after_forward(module, inputs, output, names=names):
                self.release(names)
            self.handles.append(module.register_forward_hook(release_after_forward))
        model._source_residency = self
        return self

    def stats(self):
        return {"kind": "read-only-source-madvise-dontneed-v1", "calls": self.calls,
                "advised_bytes": self.advised_bytes, "files": len(self.maps),
                "scope": "OS advice after synchronous CPU forward; no hard residency cap"}
