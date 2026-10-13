"""Header and byte-range inspection for the qualified floating GGUF formats.

The retained dimensions follow GGML order, with dimension zero contiguous.
Quantized storage needs a separately qualified block-layout reader.
"""
import math
from pathlib import Path
import struct


def inspect(path):
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
            if count > 16 << 20:
                raise ValueError("GGUF string extent")
            value = take(count)
            return value.decode("utf-8") if keep else None
        formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        def value(kind, *, keep=False, depth=0):
            if kind in formats:
                return number(formats[kind])
            if kind == 8:
                return string(keep=keep)
            if kind == 9 and depth < 4:
                subtype, count = number("I"), number("Q")
                if count > 2**20:
                    raise ValueError("GGUF array extent")
                for _ in range(count):
                    value(subtype, depth=depth + 1)
                return None
            raise ValueError("Unsupported GGUF metadata type")
        if take(4) != b"GGUF" or number("I") not in (2, 3):
            raise ValueError("Expected a supported little-endian GGUF")
        count, fields = number("Q"), number("Q")
        if not 1 <= count <= 2**20 or fields > 2**20:
            raise ValueError("GGUF table extent")
        metadata, keys = {}, set()
        selected = {"general.architecture": 8, "general.file_type": 4, "general.alignment": 4}
        for _ in range(fields):
            key, kind = string(), number("I")
            if key in keys:
                raise ValueError("Duplicate GGUF metadata key")
            keys.add(key)
            if key in selected and selected[key] != kind:
                raise ValueError("GGUF selected metadata type differs")
            item = value(kind, keep=key in selected)
            if key in selected:
                metadata[key] = item
        alignment = metadata.get("general.alignment", 32)
        if not 1 <= alignment <= 1 << 20 or alignment & (alignment - 1):
            raise ValueError("GGUF alignment must be a bounded nonzero power of two")
        rows, names = [], set()
        for _ in range(count):
            name, ndim = string(), number("I")
            if not name or name in names or not 1 <= ndim <= 4:
                raise ValueError("Invalid GGUF tensor table")
            names.add(name)
            shape = [number("Q") for _ in range(ndim)]
            if any(d == 0 for d in shape):
                raise ValueError("Empty GGUF tensor")
            kind, offset = number("I"), number("Q")
            if kind not in (0, 1, 30):
                raise ValueError("The BF16/F32/F16 inventory reader requires a qualified floating GGUF tensor type")
            dtype, width = {0: ("F32", 4), 1: ("F16", 2), 30: ("BF16", 2)}[kind]
            if offset % alignment:
                raise ValueError("GGUF tensor offset is unaligned")
            elements = math.prod(shape)
            rows.append({"name": name, "dtype": dtype, "shape": shape, "elements": elements,
                         "offset": offset, "bytes": elements * width, "layout": "ggml-ne0-first"})
        data_start = ((stream.tell() + alignment - 1) // alignment) * alignment
        cursor = data_start
        for row in sorted(rows, key=lambda r: r["offset"]):
            row["offset"] += data_start
            if row["offset"] < cursor:
                raise ValueError("GGUF tensor extents overlap")
            cursor = row["offset"] + row["bytes"]
            if cursor > extent:
                raise ValueError("Truncated GGUF tensor data")
        return {"metadata": metadata, "tensors": sorted(rows, key=lambda r: r["name"])}
