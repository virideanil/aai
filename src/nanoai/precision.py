"""Admit original BF16 GGUF values through the qualified native inventory."""
from collections import Counter

from .gguf import inspect

PRECISION_PROFILE = "original-bf16"


def inspect_bf16(path):
    inspected = inspect(path)
    counts, elements = Counter(), Counter()
    for row in inspected["tensors"]:
        if row["dtype"] not in ("BF16", "F32"):
            raise ValueError("Original BF16 required: " + row["name"] + " has dtype " + row["dtype"])
        counts[row["dtype"]] += 1
        elements[row["dtype"]] += row["elements"]
    if not counts["BF16"]:
        raise ValueError("Original BF16 model requires BF16 tensors")
    return {"profile": PRECISION_PROFILE, "tensor_types": dict(counts),
            "elements": dict(elements), "tensor_count": len(inspected["tensors"]),
            "metadata": {k: v for k, v in inspected["metadata"].items()
                         if k in ("general.architecture", "general.file_type")}}
