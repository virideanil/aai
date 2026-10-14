"""Bounded joint alternatives for the source-bound Steps graph.

Queries retain shared variable identity. Their verdict describes exactly the
declared finite model; completeness and relevance need their own assessment.
Each operation is pure, including the explicitly hypothetical state map.
"""
from copy import deepcopy
import math
import struct

from .memory import bounded_text, identifier


MAX_VARIABLES = 12
MAX_WORLDS = 4096
MAX_EXPRESSION_NODES = 128
MAX_EXPRESSION_DEPTH = 16
RELATION = "finite_relation_v1"
OPERATORS = {"relation.decode", "relation.query", "relation.refine",
             "relation.extend", "relation.map", "relation.project",
             "relation.revise", "relation.curiosity"}

SUPPORT_CONFLICT = 0
SUPPORT_NO = 1
SUPPORT_YES = 2
SUPPORT_BOTH = 3
SUPPORT_TO_LABEL = {0: "conflict", 1: "no", 2: "yes", 3: "maybe"}
LABEL_TO_SUPPORT = {"conflict": 0, "no": 1, "yes": 2, "maybe": 3}
BF16_ZERO = 0x0000
BF16_ONE = 0x3F80
BF16_NEG_ONE = 0xBF80


def pack_support2(labels):
    """Pack 4 four-valued support codes (conflict=0, no=1, yes=2, maybe=3) per byte."""
    codes = [LABEL_TO_SUPPORT[v] if isinstance(v, str) and v in LABEL_TO_SUPPORT else v for v in labels]
    if len(codes) > 65536 or any(type(c) is not int or c not in (0, 1, 2, 3) for c in codes):
        raise ValueError("Support lattice value or extent")
    return bytes(sum(codes[i + j] << (2 * j) for j in range(min(4, len(codes) - i)))
                 for i in range(0, len(codes), 4))


def unpack_support2(data, count, *, labels=True):
    if (not isinstance(data, bytes) or type(count) is not int or not 0 <= count <= 65536
            or len(data) != (count + 3) // 4):
        raise ValueError("Packed support extent")
    result = []
    for i, byte in enumerate(data):
        width = min(4, count - 4 * i)
        if byte >= (1 << (2 * width)):
            raise ValueError("Packed support contains noncanonical tail bits")
        for j in range(width):
            code = (byte >> (2 * j)) & 3
            result.append(SUPPORT_TO_LABEL[code] if labels else code)
    return result


def evaluate_support2(expression, support_assignment):
    """Evaluate an admitted expression over 2-bit power-set codes (00, 01, 10, 11)."""
    if type(expression) is int:
        return 2 if expression else 1
    if isinstance(expression, str):
        code = support_assignment[expression]
        if type(code) is not int or code not in (0, 1, 2, 3):
            raise ValueError("Support assignment requires 2-bit codes")
        return code
    operator, operands = next(iter(expression.items()))
    if operator == "not":
        c = evaluate_support2(operands, support_assignment)
        return ((c & 1) << 1) | ((c >> 1) & 1)
    a, b = (evaluate_support2(x, support_assignment) for x in operands)
    if a == 0 or b == 0:
        return 0
    a0, a1 = a & 1, (a >> 1) & 1
    b0, b1 = b & 1, (b >> 1) & 1
    if operator == "and":
        c0, c1 = a0 | b0, a1 & b1
    elif operator == "or":
        c0, c1 = a0 & b0, a1 | b1
    elif operator == "xor":
        c0, c1 = (a0 & b0) | (a1 & b1), (a0 & b1) | (a1 & b0)
    elif operator == "implies":
        c0, c1 = a1 & b0, a0 | b1
    else:
        c0, c1 = (a0 & b1) | (a1 & b0), (a0 & b0) | (a1 & b1)
    return c0 | (c1 << 1)


def query_tensor_enclosure(model, expression):
    """Compare exact joint evaluation against its 1D marginal 2-bit box enclosure."""
    validate_model(model)
    if expression_variables(expression) - set(model["variables"]):
        raise ValueError("Relation question references an unavailable variable")
    if not model["worlds"]:
        return {"exact_code": 0, "exact_verdict": None, "box_code": 0, "box_verdict": None,
                "box_marginals": {v: "conflict" for v in model["variables"]},
                "sound_enclosure": True, "enclosure_holds": True, "correlation_loss": False}
    exact_code = 0
    box = {v: 0 for v in model["variables"]}
    for row in model["worlds"]:
        support_row = {v: (2 if bit else 1) for v, bit in zip(model["variables"], row)}
        exact_code |= evaluate_support2(expression, support_row)
        for v, code in support_row.items():
            box[v] |= code
    box_code = evaluate_support2(expression, box)
    holds = (exact_code & box_code) == exact_code
    return {
        "exact_code": exact_code,
        "exact_verdict": SUPPORT_TO_LABEL[exact_code],
        "box_code": box_code,
        "box_verdict": SUPPORT_TO_LABEL[box_code],
        "box_marginals": {v: SUPPORT_TO_LABEL[box[v]] for v in model["variables"]},
        "sound_enclosure": holds,
        "enclosure_holds": holds,
        "correlation_loss": exact_code != box_code,
    }


def pack_bit_plane(bits):
    """Pack binary world bits (0 or 1) at 1 bit per cell into canonical U8 bytes."""
    out = bytearray()
    row = list(bits)
    for i in range(0, len(row), 8):
        chunk = row[i:i + 8]
        byte = 0
        for j, bit in enumerate(chunk):
            if type(bit) is not int or bit not in (0, 1):
                raise ValueError("Bit-plane packing requires exact 0/1 bits")
            byte |= (bit << j)
        out.append(byte)
    return bytes(out)


def unpack_bit_plane(raw, count):
    """Unpack 1-bit-per-cell binary world bits from canonical U8 bytes."""
    if not isinstance(raw, (bytes, bytearray)) or type(count) is not int or count < 0:
        raise ValueError("Bit-plane extent")
    if len(raw) != (count + 7) // 8:
        raise ValueError("Bit-plane byte length mismatch")
    result = []
    for i, byte in enumerate(raw):
        width = min(8, count - 8 * i)
        if width < 8 and byte >= (1 << width):
            raise ValueError("Bit-plane contains noncanonical tail bits")
        for j in range(width):
            result.append((byte >> j) & 1)
    return result


def pack_indicator_2n(variables, worlds):
    """Encode a finite relation as its canonical order-free 2^n characteristic bitset."""
    n = len(variables)
    if not 1 <= n <= MAX_VARIABLES:
        raise ValueError("Characteristic indicator variable extent")
    total_assignments = 1 << n
    byte_len = (total_assignments + 7) // 8
    out = bytearray(byte_len)
    for row in worlds:
        if len(row) != n:
            raise ValueError("World row width mismatch")
        idx = 0
        for bit in row:
            if type(bit) is not int or bit not in (0, 1):
                raise ValueError("Characteristic indicator requires binary worlds")
            idx = (idx << 1) | bit
        out[idx >> 3] |= 1 << (idx & 7)
    return bytes(out)


def unpack_indicator_2n(raw, variables):
    """Decode the canonical lexicographical world set from a 2^n characteristic bitset."""
    n = len(variables)
    if not isinstance(raw, (bytes, bytearray)) or not 1 <= n <= MAX_VARIABLES:
        raise ValueError("Characteristic indicator extent")
    total_assignments = 1 << n
    expected_bytes = (total_assignments + 7) // 8
    if len(raw) != expected_bytes:
        raise ValueError("Characteristic indicator byte length mismatch")
    if total_assignments < 8 and raw and raw[0] >= (1 << total_assignments):
        raise ValueError("Characteristic indicator contains noncanonical tail bits")
    worlds = []
    for idx in range(total_assignments):
        if (raw[idx >> 3] >> (idx & 7)) & 1:
            worlds.append([(idx >> (n - 1 - j)) & 1 for j in range(n)])
    return worlds


def indicator_expression_mask(variables, expression) -> int:
    """Compile an admitted boolean expression into a 2^n bitmask over assignment indices."""
    n = len(variables)
    mask = 0
    for idx in range(1 << n):
        assignment = {variables[j]: (idx >> (n - 1 - j)) & 1 for j in range(n)}
        if evaluate(expression, assignment) == 1:
            mask |= 1 << idx
    return mask


def indicator_query_verdict(raw_indicator: bytes, variables: list[str], expression) -> str:
    """Bitwise query evaluation directly over a 2^n characteristic vector."""
    state_int = int.from_bytes(raw_indicator, "little")
    if state_int == 0:
        raise ValueError("Empty characteristic indicator has no consistent worlds")
    expr_mask = indicator_expression_mask(variables, expression)
    has_true = (state_int & expr_mask) != 0
    has_false = (state_int & ~expr_mask) != 0
    if has_true and not has_false:
        return "yes"
    if has_false and not has_true:
        return "no"
    return "maybe"


def indicator_dalal_revise(prior_indicator: bytes, incoming_indicator: bytes, variables: list[str]) -> tuple[bytes, int | None]:
    """Bitwise Dalal minimal-Hamming belief revision over 2^n characteristic vectors."""
    n = len(variables)
    total = 1 << n
    byte_len = (total + 7) // 8
    p_int = int.from_bytes(prior_indicator, "little")
    m_int = int.from_bytes(incoming_indicator, "little")
    if m_int == 0:
        return bytes(byte_len), None
    if p_int == 0:
        return incoming_indicator, None
    overlap = p_int & m_int
    if overlap != 0:
        return overlap.to_bytes(byte_len, "little"), 0
    p_indices = [i for i in range(total) if (p_int >> i) & 1]
    m_indices = [i for i in range(total) if (m_int >> i) & 1]
    best_d = n + 1
    selected_mask = 0
    for w in m_indices:
        d = min((w ^ v).bit_count() for v in p_indices)
        if d < best_d:
            best_d = d
            selected_mask = 1 << w
        elif d == best_d:
            selected_mask |= 1 << w
    return selected_mask.to_bytes(byte_len, "little"), best_d


def encode_relation_tensor(model, *, mode="bf16_support", open_questions=()):
    """Encode a non-empty finite relation into exact BF16 words or compact packed bit/trit carriers."""
    from .judgments import pack
    validate_model(model)
    if not model["worlds"]:
        raise ValueError("An inconsistent relation has no non-empty world tensor")
    variables = list(model["variables"])
    worlds = [list(row) for row in model["worlds"]]
    if mode == "auto_compact":
        n_vars = len(variables)
        bit_plane_len = (len(worlds) * n_vars + 7) // 8
        ind_len = ((1 << n_vars) + 7) // 8
        mode = "indicator_2n" if ind_len <= bit_plane_len else "bit_plane"
    if mode not in {"bf16_support", "bf16_signed", "pack5", "support2", "bit_plane", "indicator_2n"}:
        raise ValueError("Unsupported relation tensor mode")
    words = []
    flat_bits = []
    flat_trits = []
    flat_support = []
    for row in worlds:
        for bit in row:
            flat_bits.append(bit)
            flat_trits.append("yes" if bit else "no")
            flat_support.append(2 if bit else 1)
            if mode == "bf16_support":
                words.extend((BF16_ONE, BF16_ZERO) if bit == 0 else (BF16_ZERO, BF16_ONE))
            elif mode == "bf16_signed":
                words.append(BF16_ONE if bit else BF16_NEG_ONE)
    packed_worlds_bits = pack_bit_plane(flat_bits)
    packed_worlds_indicator = pack_indicator_2n(variables, worlds)
    packed_worlds_trits = pack(flat_trits)
    packed_worlds_support2 = pack_support2(flat_support)
    if mode in {"bf16_support", "bf16_signed"}:
        cols = len(variables) * (2 if mode == "bf16_support" else 1)
        shape = [len(worlds), cols]
        raw = struct.pack("<" + "H" * len(words), *words)
        dtype = "bf16"
    elif mode == "pack5":
        raw = packed_worlds_trits
        shape = [len(raw)]
        dtype = "u8"
    elif mode == "support2":
        raw = packed_worlds_support2
        shape = [len(raw)]
        dtype = "u8"
    elif mode == "bit_plane":
        raw = packed_worlds_bits
        shape = [len(raw)]
        dtype = "u8"
    else:
        raw = packed_worlds_indicator
        shape = [len(raw)]
        dtype = "u8"
    box_codes = []
    for j in range(len(variables)):
        col = {row[j] for row in worlds}
        box_codes.append(1 if col == {0} else 2 if col == {1} else 3)
    box_labels = [SUPPORT_TO_LABEL[c] for c in box_codes]
    questions = []
    for q in open_questions:
        enc = query_tensor_enclosure(model, q)
        questions.append({"expression": deepcopy(q), "verdict": enc["exact_verdict"],
                          "box_verdict": enc["box_verdict"], "correlation_loss": enc["correlation_loss"]})
    metadata = {
        "operation": "relation.tensor",
        "mode": mode,
        "dtype": dtype,
        "variables": variables,
        "world_count": len(worlds),
        "model_scope": model["scope"],
        "box_marginals": dict(zip(variables, box_labels)),
        "packed_box_trits_hex": pack(box_labels).hex(),
        "packed_box_support2_hex": pack_support2(box_codes).hex(),
        "packed_worlds_bits_hex": packed_worlds_bits.hex(),
        "packed_worlds_indicator_2n_hex": packed_worlds_indicator.hex(),
        "packed_worlds_trits_hex": packed_worlds_trits.hex(),
        "packed_worlds_support2_hex": packed_worlds_support2.hex(),
        "questions": questions,
        "meaning": "Exact joint relation worlds carried in BF16 or binary/ternary/support planes with pack5 box marginals",
    }
    return raw, shape, metadata


def decode_relation_tensor(raw, shape=None, metadata=None):
    """Reconstruct and verify the exact finite_relation_v1 model and all metadata from its tensor carrier."""
    from .judgments import pack, unpack
    if shape is None and metadata is None and isinstance(raw, tuple) and len(raw) == 3:
        raw, shape, metadata = raw
    if not isinstance(metadata, dict) or metadata.get("operation") != "relation.tensor":
        raise ValueError("Tensor metadata is not a relational carrier")
    mode = metadata.get("mode")
    variables = metadata.get("variables")
    world_count = metadata.get("world_count")
    if (mode not in {"bf16_support", "bf16_signed", "pack5", "support2", "bit_plane", "indicator_2n"}
            or not isinstance(variables, list)
            or not 1 <= len(variables) <= MAX_VARIABLES or type(world_count) is not int
            or not 1 <= world_count <= MAX_WORLDS or not isinstance(raw, bytes)):
        raise ValueError("Relational tensor metadata extent")
    expected_dtype = "u8" if mode in {"pack5", "support2", "bit_plane", "indicator_2n"} else "bf16"
    if metadata.get("dtype") != expected_dtype:
        raise ValueError("Relational tensor metadata dtype is missing or differs from mode")
    count = world_count * len(variables)
    worlds = []
    flat_bits = []
    flat_trits = []
    flat_support = []
    if mode in {"bf16_support", "bf16_signed"}:
        cols = len(variables) * (2 if mode == "bf16_support" else 1)
        if list(shape) != [world_count, cols] or len(raw) != world_count * cols * 2:
            raise ValueError("Relational tensor shape or byte count differs")
        words = struct.unpack("<" + "H" * (world_count * cols), raw)
        for r in range(world_count):
            row = []
            offset = r * cols
            for j in range(len(variables)):
                if mode == "bf16_support":
                    w0, w1 = words[offset + 2 * j], words[offset + 2 * j + 1]
                    if (w0, w1) == (BF16_ONE, BF16_ZERO):
                        bit = 0
                    elif (w0, w1) == (BF16_ZERO, BF16_ONE):
                        bit = 1
                    else:
                        raise ValueError("Noncanonical BF16 support pair in relational tensor")
                else:
                    w = words[offset + j]
                    if w == BF16_NEG_ONE:
                        bit = 0
                    elif w == BF16_ONE:
                        bit = 1
                    else:
                        raise ValueError("Noncanonical BF16 signed trit in relational tensor")
                row.append(bit)
                flat_bits.append(bit)
                flat_trits.append("yes" if bit else "no")
                flat_support.append(2 if bit else 1)
            worlds.append(row)
    elif mode == "pack5":
        expected_bytes = (count + 4) // 5
        if list(shape) != [expected_bytes] or len(raw) != expected_bytes:
            raise ValueError("Relational pack5 tensor shape or byte count differs")
        flat_trits = unpack(raw, count)
        if pack(flat_trits) != raw:
            raise ValueError("Noncanonical trailing trit padding in pack5 relational tensor")
        for r in range(world_count):
            row = []
            offset = r * len(variables)
            for j in range(len(variables)):
                trit = flat_trits[offset + j]
                if trit == "no":
                    bit = 0
                elif trit == "yes":
                    bit = 1
                else:
                    raise ValueError("World tensor cannot contain unresolved marginal trit in a concrete world row")
                row.append(bit)
                flat_bits.append(bit)
                flat_support.append(2 if bit else 1)
            worlds.append(row)
    elif mode == "support2":
        expected_bytes = (count + 3) // 4
        if list(shape) != [expected_bytes] or len(raw) != expected_bytes:
            raise ValueError("Relational support2 tensor shape or byte count differs")
        flat_support = unpack_support2(raw, count, labels=False)
        if pack_support2(flat_support) != raw:
            raise ValueError("Noncanonical trailing support bits in support2 relational tensor")
        for r in range(world_count):
            row = []
            offset = r * len(variables)
            for j in range(len(variables)):
                code = flat_support[offset + j]
                if code == 1:
                    bit = 0
                elif code == 2:
                    bit = 1
                else:
                    raise ValueError("World tensor cannot contain non-atomic support code in a concrete world row")
                row.append(bit)
                flat_bits.append(bit)
                flat_trits.append("yes" if bit else "no")
            worlds.append(row)
    elif mode == "bit_plane":
        expected_bytes = (count + 7) // 8
        if list(shape) != [expected_bytes] or len(raw) != expected_bytes:
            raise ValueError("Relational bit_plane tensor shape or byte count differs")
        flat_bits = unpack_bit_plane(raw, count)
        for r in range(world_count):
            row = flat_bits[r * len(variables):(r + 1) * len(variables)]
            for bit in row:
                flat_trits.append("yes" if bit else "no")
                flat_support.append(2 if bit else 1)
            worlds.append(row)
    else:
        expected_bytes = ((1 << len(variables)) + 7) // 8
        if list(shape) != [expected_bytes] or len(raw) != expected_bytes:
            raise ValueError("Relational indicator_2n tensor shape or byte count differs")
        ind_worlds = unpack_indicator_2n(raw, variables)
        if len(ind_worlds) != world_count:
            raise ValueError("Decoded indicator_2n world count differs from metadata")
        if isinstance(metadata.get("packed_worlds_bits_hex"), str):
            candidate_bits = unpack_bit_plane(bytes.fromhex(metadata["packed_worlds_bits_hex"]), count)
            candidate_worlds = [
                candidate_bits[r * len(variables):(r + 1) * len(variables)]
                for r in range(world_count)
            ]
            if (len({tuple(r) for r in candidate_worlds}) != world_count
                    or sorted(candidate_worlds) != ind_worlds
                    or pack_indicator_2n(variables, candidate_worlds) != raw):
                raise ValueError("Packed binary world bit-plane differs from the decoded indicator_2n world set")
            worlds = candidate_worlds
        else:
            worlds = ind_worlds
        for row in worlds:
            for bit in row:
                flat_bits.append(bit)
                flat_trits.append("yes" if bit else "no")
                flat_support.append(2 if bit else 1)
    if (not isinstance(metadata.get("packed_worlds_trits_hex"), str)
            or not isinstance(metadata.get("packed_worlds_support2_hex"), str)
            or unpack(bytes.fromhex(metadata["packed_worlds_trits_hex"]), count) != flat_trits
            or unpack_support2(bytes.fromhex(metadata["packed_worlds_support2_hex"]), count, labels=False) != flat_support):
        raise ValueError("Packed ternary/support planes differ from the decoded relational tensor")
    if "packed_worlds_bits_hex" in metadata:
        if (not isinstance(metadata["packed_worlds_bits_hex"], str)
                or unpack_bit_plane(bytes.fromhex(metadata["packed_worlds_bits_hex"]), count) != flat_bits):
            raise ValueError("Packed binary world bit-plane differs from the decoded relational tensor")
    if "packed_worlds_indicator_2n_hex" in metadata:
        if (not isinstance(metadata["packed_worlds_indicator_2n_hex"], str)
                or bytes.fromhex(metadata["packed_worlds_indicator_2n_hex"]) != pack_indicator_2n(variables, worlds)):
            raise ValueError("Packed 2^n characteristic indicator differs from the decoded relational tensor")
    model = validate_model({"type": RELATION, "variables": list(variables),
                            "worlds": worlds, "scope": metadata.get("model_scope")})
    expected_box_codes = []
    for j in range(len(variables)):
        col = {row[j] for row in worlds}
        expected_box_codes.append(1 if col == {0} else 2 if col == {1} else 3)
    expected_box_labels = [SUPPORT_TO_LABEL[c] for c in expected_box_codes]
    expected_box_marginals = dict(zip(variables, expected_box_labels))
    if (metadata.get("box_marginals") != expected_box_marginals
            or metadata.get("packed_box_trits_hex") != pack(expected_box_labels).hex()
            or metadata.get("packed_box_support2_hex") != pack_support2(expected_box_codes).hex()):
        raise ValueError("Relational tensor box marginals disagree with the decoded worlds")
    raw_questions = metadata.get("questions", [])
    if not isinstance(raw_questions, list):
        raise ValueError("Relational tensor questions metadata must be a list")
    for q in raw_questions:
        if not isinstance(q, dict) or set(q) != {"expression", "verdict", "box_verdict", "correlation_loss"}:
            raise ValueError("Relational tensor question entry is malformed")
        enc = query_tensor_enclosure(model, q["expression"])
        if (q["verdict"] != enc["exact_verdict"]
                or q["box_verdict"] != enc["box_verdict"]
                or q["correlation_loss"] != enc["correlation_loss"]):
            raise ValueError("Relational tensor question verdict disagrees with the decoded worlds")
    return model


def benchmark_relation_carriers(model, *, expression=None, incoming_model=None) -> dict:
    """Measure payload bytes, metadata/container bytes, round-trip fidelity, bitwise query and Dalal revision across all relational tensor carriers."""
    import json
    from .tensor_store import pack_safetensors
    validate_model(model)
    modes = ("bf16_support", "bf16_signed", "pack5", "support2", "bit_plane", "indicator_2n")
    byte_sizes = {}
    metadata_bytes = {}
    container_bytes = {}
    for m in modes:
        raw, shape, meta = encode_relation_tensor(model, mode=m)
        decoded = decode_relation_tensor(raw, shape, meta)
        if decoded["worlds"] != model["worlds"] or decoded["variables"] != model["variables"]:
            raise ValueError(f"Carrier {m} round-trip mismatch")
        serial = json.dumps(meta, sort_keys=True, separators=(",", ":"))
        archive = pack_safetensors(
            {"state": {"dtype": meta["dtype"].upper(), "shape": shape, "data": raw}},
            metadata={"origin": serial},
        )
        byte_sizes[m] = len(raw)
        metadata_bytes[m] = len(serial.encode("utf-8"))
        container_bytes[m] = len(archive)
    auto_raw, _, auto_meta = encode_relation_tensor(model, mode="auto_compact")
    out = {
        "variables": len(model["variables"]),
        "worlds": len(model["worlds"]),
        "cells": len(model["variables"]) * len(model["worlds"]),
        "bytes": byte_sizes,
        "payload_bytes": byte_sizes,
        "metadata_utf8_bytes": metadata_bytes,
        "complete_safetensors_bytes": container_bytes,
        "auto_compact_mode": auto_meta["mode"],
        "auto_compact_bytes": len(auto_raw),
        "storage_note": "payload_bytes measures raw tensor data only; complete_safetensors_bytes includes JSON header and sidecar metadata",
    }
    ind_raw = pack_indicator_2n(model["variables"], model["worlds"])
    if expression is not None:
        enc = query_tensor_enclosure(model, expression)
        ind_verdict = indicator_query_verdict(ind_raw, model["variables"], expression)
        if ind_verdict != enc["exact_verdict"]:
            raise ValueError("Characteristic indicator bitwise query disagrees with world-table query")
        out["query_verdict"] = ind_verdict
        out["query_parity"] = True
    if incoming_model is not None:
        validate_model(incoming_model)
        inc_ind = pack_indicator_2n(incoming_model["variables"], incoming_model["worlds"])
        rev_ind, rev_d = indicator_dalal_revise(ind_raw, inc_ind, model["variables"])
        rev_worlds_ind = unpack_indicator_2n(rev_ind, model["variables"])
        rev_worlds_table, table_d = km_revise_worlds(model["worlds"], incoming_model["worlds"])
        if rev_worlds_ind != rev_worlds_table or rev_d != table_d:
            raise ValueError("Characteristic indicator bitwise Dalal revision disagrees with km_revise_worlds")
        out["revision_min_hamming_distance"] = rev_d
        out["revision_parity"] = True
    return out


def expression_variables(expression):
    remaining = MAX_EXPRESSION_NODES
    variables = set()

    def visit(expr, depth):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > MAX_EXPRESSION_DEPTH:
            raise ValueError("Relation expression exceeds the declared bound")
        if type(expr) is int and expr in (0, 1):
            return
        if isinstance(expr, str):
            identifier(expr)
            variables.add(expr)
            return
        if not isinstance(expr, dict) or len(expr) != 1:
            raise ValueError("Relation expression needs a bit, variable or named operation")
        operator, value = next(iter(expr.items()))
        if operator == "not":
            visit(value, depth + 1)
        elif operator in {"and", "or", "xor", "eq", "implies"}:
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError("A relation junction needs two operands")
            for child in value:
                visit(child, depth + 1)
        else:
            raise ValueError("Relation expression operator is unavailable")

    visit(expression, 0)
    return variables


def evaluate(expression, assignment):
    """Evaluate an already admitted expression in one shared assignment."""
    if type(expression) is int:
        return expression
    if isinstance(expression, str):
        return assignment[expression]
    operator, operands = next(iter(expression.items()))
    if operator == "not":
        return 1 - evaluate(operands, assignment)
    a, b = (evaluate(x, assignment) for x in operands)
    return {"and": a & b, "or": a | b, "xor": a ^ b, "eq": int(a == b), "implies": (1 - a) | b}[operator]


def validate_arguments(operator, arguments):
    fields = {"relation.decode": {"step"}, "relation.query": {"step", "expression"},
              "relation.refine": {"step", "expression", "expected"},
              "relation.extend": {"step", "variable", "expression"},
              "relation.map": {"step", "variable", "expression"},
              "relation.project": {"step", "variables"},
              "relation.revise": {"step", "replacement", "reason"}}
    if operator == "relation.curiosity":
        if (not isinstance(arguments, dict) or "step" not in arguments or "questions" not in arguments
                or not isinstance(arguments["step"], str)
                or set(arguments) - {"step", "questions", "candidates", "observations"}
                or ("candidates" in arguments and "observations" in arguments)):
            raise ValueError("Relation operation fields differ from the contract")
        questions = arguments["questions"]
        if not isinstance(questions, list) or not 1 <= len(questions) <= 16:
            raise ValueError("Relation curiosity needs one to 16 target questions")
        for q in questions:
            expression_variables(q)
        candidates = arguments.get("candidates", arguments.get("observations"))
        if candidates is not None:
            if not isinstance(candidates, list) or not 1 <= len(candidates) <= 16:
                raise ValueError("Relation curiosity candidates must be one to 16 expressions")
            for c in candidates:
                expression_variables(c)
        return [arguments["step"]]
    if operator == "relation.revise":
        if (not isinstance(arguments, dict) or not {"step", "replacement", "reason"} <= set(arguments)
                or set(arguments) - {"step", "replacement", "reason", "policy"}
                or not isinstance(arguments["step"], str)):
            raise ValueError("Relation operation fields differ from the contract")
        if not isinstance(arguments["replacement"], str) or arguments["replacement"] == arguments["step"]:
            raise ValueError("Relation revision needs a distinct replacement step")
        bounded_text(arguments["reason"], "relation revision reason", 2048)
        if arguments.get("policy", "replace") not in {"replace", "km"}:
            raise ValueError("Relation revision policy must be 'replace' or 'km'")
        return [arguments["step"], arguments["replacement"]]
    if (operator not in fields or not isinstance(arguments, dict)
            or set(arguments) != fields[operator] or not isinstance(arguments["step"], str)):
        raise ValueError("Relation operation fields differ from the contract")
    if "variable" in arguments:
        identifier(arguments["variable"])
    if "expression" in arguments:
        if not (operator == "relation.extend" and arguments["expression"] is None):
            expression_variables(arguments["expression"])
    if operator == "relation.refine" and (type(arguments["expected"]) is not int
                                          or arguments["expected"] not in (0, 1)):
        raise ValueError("Relation refinement needs an observed or assumed bit")
    if operator == "relation.project":
        variables = arguments["variables"]
        if (not isinstance(variables, list) or not 1 <= len(variables) <= MAX_VARIABLES
                or len(set(variables)) != len(variables)):
            raise ValueError("Relation projection needs distinct variable identities")
        for name in variables:
            identifier(name)
    return [arguments["step"]]


KM_ASSUMPTIONS = {
    "reference": "Katsuno & Mendelzon (1991), Propositional Knowledge Base Revision and Minimal Change, AIJ 52(3):263-294, Theorem 3.3; Dalal (1988) Hamming preorder",
    "theorem_reference": "Katsuno & Mendelzon (1991) Theorem 3.3 + Dalal (1988) uniform Hamming distance",
    "policy_guidance": "Use policy='replace' for authoritative source corrections that supersede prior state; use policy='km' for partial new evidence under minimal Hamming change",
    "domain": "Finite boolean worlds W = {0,1}^n over fixed shared variable identities",
    "preconditions": "Prior psi != empty and incoming evidence mu != empty for KM1-KM8; when psi is empty (conflict), replacement adopts mu with reopened_conflict=True",
    "metric": "Uniform unit L1 Hamming distance d_H(psi, w) = min_{v in psi} sum_i |w_i - v_i|",
    "non_claims": "Does not assign domain-specific variable weights or causal intervention semantics (use relation.map for state transitions)",
}



def km_revise_worlds(prior_worlds, incoming_worlds):
    """Katsuno-Mendelzon minimal-Hamming-distance belief revision over finite worlds (KM1-KM8)."""
    psi = {tuple(row) for row in prior_worlds}
    mu = {tuple(row) for row in incoming_worlds}
    if not mu:
        return [], None
    if not psi:
        return [list(r) for r in sorted(mu)], None
    overlap = psi & mu
    if overlap:
        return [list(r) for r in sorted(overlap)], 0
    distances = {
        w: min(sum(int(a != b) for a, b in zip(w, v)) for v in psi)
        for w in mu
    }
    min_d = min(distances.values())
    selected = [list(w) for w in sorted(mu) if distances[w] == min_d]
    return selected, min_d


def validate_model(model):
    if (not isinstance(model, dict) or set(model) != {"type", "variables", "worlds", "scope"}
            or model["type"] != RELATION):
        raise ValueError("A finite relation needs its typed model and scope")
    variables, worlds = model["variables"], model["worlds"]
    if not isinstance(variables, list) or not 1 <= len(variables) <= MAX_VARIABLES:
        raise ValueError("Relation variable extent")
    for name in variables:
        identifier(name)
    if len(set(variables)) != len(variables):
        raise ValueError("Relation variable identities must be distinct")
    bounded_text(model["scope"], "relation scope", 2048)
    if not isinstance(worlds, list) or len(worlds) > MAX_WORLDS:
        raise ValueError("Relation world extent")
    for row in worlds:
        if (not isinstance(row, list) or len(row) != len(variables)
                or any(type(bit) is not int or bit not in (0, 1) for bit in row)):
            raise ValueError("Relation row needs one exact bit per variable")
    if len({tuple(row) for row in worlds}) != len(worlds):
        raise ValueError("Relation world identities must be distinct")
    return model


def _entropy_bits(counts, total):
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def curiosity_table(model, questions, candidates=None):
    model = validate_model(model)
    candidates = list(model["variables"]) if candidates is None else list(candidates)
    available = set(model["variables"])
    for expr in [*questions, *candidates]:
        if expression_variables(expr) - available:
            raise ValueError("Relation curiosity references an unavailable variable")
    worlds = model["worlds"]
    if not worlds:
        raise ValueError("An inconsistent relation admits no curiosity ranking; revise the conflicting premises")
    assignments = [dict(zip(model["variables"], row)) for row in worlds]
    total = len(assignments)
    q_vectors = [tuple(evaluate(q, row) for q in questions) for row in assignments]
    prior_counts = {}
    for vec in q_vectors:
        prior_counts[vec] = prior_counts.get(vec, 0) + 1
    prior_entropy = _entropy_bits(prior_counts, total)
    open_questions = []
    settled_questions = []
    for j, q in enumerate(questions):
        vals = sorted({vec[j] for vec in q_vectors})
        if len(vals) == 2:
            open_questions.append(deepcopy(q))
        else:
            settled_questions.append({"expression": deepcopy(q), "verdict": "yes" if vals == [1] else "no"})
    open_count = len(open_questions)
    evaluations = []
    for idx, candidate in enumerate(candidates):
        branches = {0: [], 1: []}
        for row, vec in zip(assignments, q_vectors):
            branches[evaluate(candidate, row)].append(vec)
        n0, n1 = len(branches[0]), len(branches[1])
        counts0, counts1 = {}, {}
        for vec in branches[0]:
            counts0[vec] = counts0.get(vec, 0) + 1
        for vec in branches[1]:
            counts1[vec] = counts1.get(vec, 0) + 1
        h0 = _entropy_bits(counts0, n0)
        h1 = _entropy_bits(counts1, n1)
        expected_posterior = (n0 * h0 + n1 * h1) / total
        info_bits = max(0.0, prior_entropy - expected_posterior)
        decided0 = sum(1 for j in range(len(questions)) if len({v[j] for v in branches[0]}) <= 1) if n0 else len(questions)
        decided1 = sum(1 for j in range(len(questions)) if len({v[j] for v in branches[1]}) <= 1) if n1 else len(questions)
        guaranteed = sum(
            1 for j in range(len(questions))
            if (not n0 or len({v[j] for v in branches[0]}) == 1)
            and (not n1 or len({v[j] for v in branches[1]}) == 1)
        )
        expected_decided = (n0 * decided0 + n1 * decided1) / total
        evaluations.append({
            "index": idx,
            "expression": deepcopy(candidate),
            "observation": deepcopy(candidate),
            "split_counts": {"0": n0, "1": n1},
            "branch_probabilities": {"0": f"{n0}/{total}", "1": f"{n1}/{total}"},
            "prior_entropy_bits": prior_entropy,
            "expected_posterior_entropy_bits": expected_posterior,
            "information_bits": info_bits,
            "expected_gain_bits": info_bits,
            "guaranteed_decided_questions": guaranteed,
            "expected_decided_questions": expected_decided,
            "worst_case_worlds": max(n0, n1),
            "informative": n0 > 0 and n1 > 0,
        })
    ranked = sorted(
        evaluations,
        key=lambda r: (
            -r["guaranteed_decided_questions"],
            -round(r["expected_decided_questions"], 12),
            -round(r["information_bits"], 12),
            r["worst_case_worlds"],
            r["index"],
        ),
    )
    return {
        "type": "relation_curiosity_v1",
        "world_count": total,
        "question_count": len(questions),
        "open_question_count": open_count,
        "open_questions": open_questions,
        "settled_questions": settled_questions,
        "prior_entropy_bits": prior_entropy,
        "candidates": ranked,
        "recommended_index": ranked[0]["index"] if ranked else None,
        "recommended_expression": deepcopy(ranked[0]["expression"]) if ranked else None,
        "recommended_observation": deepcopy(ranked[0]) if ranked else None,
        "model_scope": model["scope"],
    }


def calculate(operator, arguments, parent, extra_inputs=()):
    validate_arguments(operator, arguments)
    if operator == "relation.decode":
        from .steps import strict_json
        if not isinstance(parent.get("value"), str):
            raise ValueError("Relation decoding needs exact recorded JSON text")
        model = validate_model(strict_json(parent["value"]))
        model = deepcopy(model)
        model["worlds"].sort()
        return {"value": model, "sources": parent["sources"],
                "scope": "Exact declared finite model; external completeness remains assessed separately"}

    model = validate_model(parent.get("value"))
    if operator == "relation.revise":
        from .steps import strict_json
        if len(extra_inputs) != 1:
            raise ValueError("Relation revision needs its replacement input")
        replacement_parent = extra_inputs[0]
        raw_value = replacement_parent.get("value")
        if isinstance(raw_value, str):
            raw_value = strict_json(raw_value)
        replacement = deepcopy(validate_model(raw_value))
        replacement["worlds"].sort()
        if replacement["variables"] != model["variables"]:
            raise ValueError("Relation revision requires matching variable identities; extend or project explicitly")
        policy = arguments.get("policy", "replace")
        km_worlds, km_min_distance = km_revise_worlds(model["worlds"], replacement["worlds"])
        chosen_worlds = km_worlds if policy == "km" else replacement["worlds"]
        revised_model = validate_model({
            "type": RELATION,
            "variables": list(replacement["variables"]),
            "worlds": chosen_worlds,
            "scope": replacement["scope"],
        })
        old_set = {tuple(row) for row in model["worlds"]}
        new_set = {tuple(row) for row in revised_model["worlds"]}
        sources = sorted(set(parent["sources"]) | set(replacement_parent["sources"]))
        return {
            "value": revised_model,
            "sources": sources,
            "revision": {
                "previous_step": arguments["step"],
                "replacement_step": arguments["replacement"],
                "reason": arguments["reason"],
                "policy": policy,
                "retained_worlds": [list(r) for r in sorted(old_set & new_set)],
                "removed_worlds": [list(r) for r in sorted(old_set - new_set)],
                "added_worlds": [list(r) for r in sorted(new_set - old_set)],
                "retained_world_count": len(old_set & new_set),
                "removed_world_count": len(old_set - new_set),
                "added_world_count": len(new_set - old_set),
                "km_minimal_worlds": km_worlds,
                "km_min_hamming_distance": km_min_distance,
                "km_assumptions": deepcopy(KM_ASSUMPTIONS),
                "monotone": new_set <= old_set,
                "reopened_conflict": len(old_set) == 0 and len(new_set) > 0,
            },
            "scope": "Explicit relational revision; prior and replacement models remain bound in the provenance graph",
        }

    if operator == "relation.project":
        import json
        retained = arguments["variables"]
        if set(retained) - set(model["variables"]):
            raise ValueError("Relation projection references an unavailable variable")
        indices = [model["variables"].index(v) for v in retained]
        eliminated = [v for v in model["variables"] if v not in set(retained)]
        fibers = {}
        for row in model["worlds"]:
            key = tuple(row[i] for i in indices)
            fibers[key] = fibers.get(key, 0) + 1
        projected_worlds = [list(key) for key in sorted(fibers)]
        box_marginals = {}
        for j, name in enumerate(retained):
            col = {row[j] for row in projected_worlds}
            box_marginals[name] = {(0,): "no", (1,): "yes", (0, 1): "maybe", (): None}[tuple(sorted(col))]
        box_world_count = (2 ** sum(1 for v in box_marginals.values() if v == "maybe")) if projected_worlds else 0
        updated = {"type": RELATION, "variables": list(retained),
                   "worlds": projected_worlds, "scope": model["scope"]}
        return {
            "value": validate_model(updated),
            "sources": parent["sources"],
            "projection": {
                "retained_variables": list(retained),
                "eliminated_variables": eliminated,
                "input_world_count": len(model["worlds"]),
                "projected_world_count": len(projected_worlds),
                "fiber_counts": {json.dumps(list(k), separators=(",", ":")): v for k, v in sorted(fibers.items())},
                "min_fiber": min(fibers.values()) if fibers else 0,
                "max_fiber": max(fibers.values()) if fibers else 0,
                "box_marginals": box_marginals,
                "box_world_count": box_world_count,
                "exact_box": len(projected_worlds) == box_world_count,
            },
            "scope": "Exact coordinate projection over retained variables; marginal box enclosure over-approximates when correlations exist",
        }

    if operator == "relation.curiosity":
        questions = arguments["questions"]
        raw_candidates = arguments.get("candidates", arguments.get("observations"))
        answer = curiosity_table(model, questions, raw_candidates)
        return {"value": answer, "sources": parent["sources"],
                "scope": "Exact counting-measure information gain and minimax world split over the declared finite relation"}

    expr = arguments["expression"]
    allowed_vars = set(model["variables"]) | ({arguments["variable"]} if operator == "relation.extend" else set())
    if expr is not None and expression_variables(expr) - allowed_vars:
        raise ValueError("Relation question references an unavailable variable")
    assignments = [dict(zip(model["variables"], row)) for row in model["worlds"]]
    if operator == "relation.query":
        values = [evaluate(expr, row) for row in assignments]
        witnesses = {}
        for row, answer in zip(model["worlds"], values):
            witnesses.setdefault(str(answer), row)
        possible = sorted(set(values))
        verdict = {(0,): "no", (1,): "yes", (0, 1): "maybe", (): None}[tuple(possible)]
        enclosure = query_tensor_enclosure(model, expr)
        answer = {"type": "relation_answer_v1", "possible_values": possible,
                  "verdict": verdict, "state": "both" if len(possible) == 2 else "decided" if possible else "conflict",
                  "variables": model["variables"], "witnesses": witnesses,
                  "question": expr, "model_scope": model["scope"],
                  "box_verdict": enclosure["box_verdict"], "correlation_loss": enclosure["correlation_loss"]}
        return {"value": deepcopy(answer), "sources": parent["sources"],
                "scope": "Exact image of the question over the declared joint alternatives"}

    updated = deepcopy(model)
    if operator == "relation.refine":
        values = [evaluate(expr, row) for row in assignments]
        updated["worlds"] = [row for row, answer in zip(model["worlds"], values)
                             if answer == arguments["expected"]]
        scope = "Conditional refinement; the supplied bit is an explicit premise"
    elif operator == "relation.extend":
        variable = arguments["variable"]
        if variable in model["variables"]:
            raise ValueError("Relation extension needs a fresh variable identity")
        if len(model["variables"]) >= MAX_VARIABLES:
            raise ValueError("Relation extension exceeds the declared bound")
        if expr is None:
            new_worlds = [row + [bit] for row in model["worlds"] for bit in (0, 1)]
        elif variable in expression_variables(expr):
            new_worlds = []
            for row, assignment in zip(model["worlds"], assignments):
                fiber = [bit for bit in (0, 1) if evaluate(expr, {**assignment, variable: bit}) == 1]
                if not fiber:
                    raise ValueError("Relation extension narrows existing alternatives; refine before extending")
                for bit in fiber:
                    new_worlds.append(row + [bit])
        else:
            values = [evaluate(expr, row) for row in assignments]
            new_worlds = [row + [bit] for row, bit in zip(model["worlds"], values)]
        if len(new_worlds) > MAX_WORLDS:
            raise ValueError("Relation extension exceeds the declared bound")
        updated["variables"].append(variable)
        updated["worlds"] = new_worlds
        scope = "Conservative extension; projection preserves every previous joint alternative"
    else:
        values = [evaluate(expr, row) for row in assignments]
        variable = arguments["variable"]
        if variable not in model["variables"]:
            raise ValueError("Relation map needs an existing variable identity")
        index = model["variables"].index(variable)
        for row, value in zip(updated["worlds"], values):
            row[index] = value
        updated["worlds"] = [list(row) for row in sorted({tuple(row) for row in updated["worlds"]})]
        scope = "Hypothetical state transformation; no external action occurred"
    return {"value": validate_model(updated), "sources": parent["sources"], "scope": scope}

