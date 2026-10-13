"""Inspect actual GGUF tensor storage before admitting an original-BF16 core.

GGUF type identifiers follow the maintained llama.cpp gguf/constants.py.
Original FP32 vectors remain permitted; F16 and quantized tensors are rejected.
"""
from collections import Counter
from pathlib import Path
import struct


PRECISION_PROFILE = "original-bf16"


def inspect_bf16(path):
    with Path(path).open("rb") as stream:
        extent = Path(path).stat().st_size
        def take(count):
            if count < 0 or stream.tell() + count > extent:
                raise ValueError("Truncated GGUF metadata")
            data = stream.read(count)
            if len(data) != count:
                raise ValueError("Truncated GGUF metadata")
            return data
        def number(fmt):
            return struct.unpack("<" + fmt, take(struct.calcsize(fmt)))[0]
        def string(*, keep=True):
            count = number("Q")
            if count > 16 * 1024**2:
                raise ValueError("GGUF string extent")
            value = take(count)
            return value.decode("utf-8") if keep else None
        formats = {0:"B", 1:"b", 2:"H", 3:"h", 4:"I", 5:"i", 6:"f", 7:"?", 10:"Q", 11:"q", 12:"d"}
        def value(kind, keep=False, depth=0):
            if kind in formats:
                return number(formats[kind])
            if kind == 8:
                return string(keep=keep)
            if kind == 9 and depth < 4:
                subtype, count = number("I"), number("Q")
                if count > 2**20:
                    raise ValueError("GGUF array extent")
                for _ in range(count):
                    value(subtype, False, depth + 1)
                return None
            raise ValueError("Unsupported GGUF metadata type")
        if take(4) != b"GGUF" or number("I") not in (2, 3):
            raise ValueError("Expected a supported little-endian GGUF")
        tensors, fields = number("Q"), number("Q")
        if not 1 <= tensors <= 2**20 or fields > 2**20:
            raise ValueError("GGUF table extent")
        metadata = {}
        for _ in range(fields):
            key, kind = string(), number("I")
            item = value(kind, key in ("general.architecture", "general.file_type"))
            if key in ("general.architecture", "general.file_type"):
                metadata[key] = item
        counts, elements = Counter(), Counter()
        names = set()
        for _ in range(tensors):
            name, ndim = string(), number("I")
            if name in names or not 1 <= ndim <= 4:
                raise ValueError("Invalid GGUF tensor table")
            names.add(name)
            size = 1
            for _ in range(ndim):
                dimension = number("Q")
                if dimension == 0:
                    raise ValueError("Empty GGUF tensor")
                size *= dimension
            kind, _offset = number("I"), number("Q")
            if kind not in (0, 30):
                raise ValueError(f"Original BF16 required: {name} has GGUF tensor type {kind}")
            label = "BF16" if kind == 30 else "F32"
            counts[label] += 1
            elements[label] += size
        if not counts["BF16"]:
            raise ValueError("Original BF16 model requires BF16 tensors")
        return {"profile": PRECISION_PROFILE, "tensor_types": dict(counts),
            "elements": dict(elements), "tensor_count": tensors, "metadata": metadata}
