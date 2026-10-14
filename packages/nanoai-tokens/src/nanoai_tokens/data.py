"""Version-bound causal data for a future consumer of master active slots.

Existing pretrained model adapters retain their native tokenizer. This module
prepares exact data and explicit model control rows without training parameters.
"""
from .tokenizer import Tokenizer as MasterTokenizer


def model_contract(tokenizer: MasterTokenizer):
    size = len(tokenizer.pieces)
    return {"schema": "nanoai.master-model-input.v1", "tokenizer_identity": tokenizer.identity,
            "vocabulary_identity": tokenizer.vocabulary_identity, "literal_rows": size,
            "model_rows": size + 3, "bos_id": size, "eos_id": size + 1, "pad_id": size + 2,
            "label_ignore": -100, "byte_controls": "BOS/EOS/PAD are model controls with no literal byte surface"}


def causal_batch(tokenizer: MasterTokenizer, byte_rows, *, max_tokens=4096):
    if not isinstance(byte_rows, (list, tuple)) or not 1 <= len(byte_rows) <= 256:
        raise ValueError("Master batch extent")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 131073:
        raise ValueError("Master sequence extent")
    records = [tokenizer.encode(raw) for raw in byte_rows]
    contract = model_contract(tokenizer)
    paths = [[span["active_id"] for span in row["spans"]] for row in records]
    width = max(len(path) + 1 for path in paths)
    if width > max_tokens or width * len(paths) > 1048576:
        raise ValueError("Master batch token budget; input remains untruncated")
    inputs, labels, masks, lengths = [], [], [], []
    for path, record, raw in zip(paths, records, byte_rows):
        if tokenizer.decode(record) != raw:
            raise ValueError("Master data reconstruction")
        length = len(path) + 1; padding = width - length
        inputs.append([contract["bos_id"]] + path + [contract["pad_id"]] * padding)
        labels.append(path + [contract["eos_id"]] + [contract["label_ignore"]] * padding)
        masks.append([1] * length + [0] * padding)
        lengths.append(length)
    return {"contract": contract, "input_ids": inputs, "labels": labels, "attention_mask": masks,
            "lengths": lengths, "records": records}


def validate_batch(tokenizer: MasterTokenizer, batch):
    """Read the serialized consuming representation, including shifted targets."""
    if not isinstance(batch, dict) or batch.get("contract") != model_contract(tokenizer):
        raise ValueError("Master model input version")
    records = batch.get("records")
    if not isinstance(records, list) or not 1 <= len(records) <= 256:
        raise ValueError("Master model record extent")
    raw = [tokenizer.decode(record) for record in records]
    expected = causal_batch(tokenizer, raw, max_tokens=131073)
    if batch != expected:
        raise ValueError("Master model input shifted IDs, labels or mask")
    return raw


AGENTIC_CONTROLS = (
    "think_start", "think_end",
    "talk_start", "talk_end",
    "code_start", "code_end",
    "tool_call_start", "tool_call_end",
    "tool_obs_start", "tool_obs_end",
    "visual_start", "visual_end",
)

OPERATOR_VOCABULARY_VERSION = "nanoai.admitted-operators.v1"

# Append-only operator slot vocabulary synchronized with nanoai.steps.ADMITTED_OPERATORS_V1.
# Never insert or reorder existing entries; new operators may only be appended under a new version.
ADMITTED_OPERATORS = (
    "source.read",
    "json.field",
    "integer.product",
    "integer.sum",
    "quantity.field",
    "quantity.product",
    "quantity.sum",
    "relation.decode",
    "relation.query",
    "relation.refine",
    "relation.extend",
    "relation.map",
    "relation.project",
    "relation.revise",
    "relation.curiosity",
    "scratchboard.add",
    "scratchboard.reserve",
    "scratchboard.bind",
    "scratchboard.bind_many",
    "scratchboard.generator",
    "scratchboard.revise",
    "scratchboard.refine",
    "judgment.predict",
    "judgment.assess",
    "judgment.combine",
)

MODE_CODES = {
    "pad": 0,
    "think": 1,
    "talk": 2,
    "code": 3,
    "tool_call": 4,
    "tool_obs": 5,
    "visual": 6,
}

_ASCII_DELIMITERS = frozenset(b"{}[]():,;.\"'`+-*/=<>!&|^~%#\\@\n\r\t ")


def agentic_contract(tokenizer: MasterTokenizer, *, operators=None):
    """Return the deterministic control-row and append-only operator-row contract with SHAKE256 identity."""
    import hashlib
    import json
    base = model_contract(tokenizer)
    size = base["literal_rows"]
    cursor = size + 3
    control_ids = {}
    for name in AGENTIC_CONTROLS:
        control_ids[name] = cursor
        cursor += 1
    op_tuple = tuple(ADMITTED_OPERATORS if operators is None else operators)
    if len(op_tuple) < len(ADMITTED_OPERATORS) or op_tuple[:len(ADMITTED_OPERATORS)] != ADMITTED_OPERATORS:
        raise ValueError("Agentic operator vocabulary is append-only; prefix must match ADMITTED_OPERATORS v1")
    if len(set(op_tuple)) != len(op_tuple):
        raise ValueError("Agentic operator vocabulary contains duplicate operators")
    operator_ids = {}
    for op in op_tuple:
        operator_ids[op] = cursor
        cursor += 1
    identity_payload = json.dumps({
        "schema": "nanoai.master-agentic-input.v1",
        "operator_vocabulary_version": OPERATOR_VOCABULARY_VERSION,
        "tokenizer_identity": tokenizer.identity,
        "vocabulary_identity": tokenizer.vocabulary_identity,
        "literal_rows": size,
        "control_ids": control_ids,
        "operator_ids": operator_ids,
        "mode_codes": MODE_CODES,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    contract_identity = "shake256:" + hashlib.shake_256(identity_payload).hexdigest(64)
    return {
        "schema": "nanoai.master-agentic-input.v1",
        "operator_vocabulary_version": OPERATOR_VOCABULARY_VERSION,
        "contract_identity": contract_identity,
        "tokenizer_identity": tokenizer.identity,
        "vocabulary_identity": tokenizer.vocabulary_identity,
        "literal_rows": size,
        "bos_id": base["bos_id"],
        "eos_id": base["eos_id"],
        "pad_id": base["pad_id"],
        "control_ids": control_ids,
        "operator_ids": operator_ids,
        "mode_codes": dict(MODE_CODES),
        "model_rows": cursor,
        "label_ignore": -100,
        "byte_controls": (
            "BOS/EOS/PAD, mode delimiters and admitted operator slots occupy control rows >= literal_rows "
            "with no literal byte surface; literal bytes cannot spoof control or tool tokens"
        ),
    }



def structural_boundaries(raw: bytes, mode: str = "code") -> set[int]:
    """Compute UTF-8-safe byte boundaries so identifiers, delimiters and numbers never merge across syntax."""
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise ValueError("Structural boundary input must be bytes")
    data = bytes(raw)
    if mode not in {"plain", "json", "code", "markdown"}:
        raise ValueError("Unsupported structural boundary mode")
    if mode == "plain" or not data:
        return set()
    bounds = set()
    n = len(data)
    for i, b in enumerate(data):
        if b < 0x80 and b in _ASCII_DELIMITERS:
            if b in b"\n\r":
                if i > 0:
                    bounds.add(i)
                if i + 1 < n:
                    bounds.add(i + 1)
            elif b in b" \t":
                if i > 0 and data[i - 1] not in b" \t":
                    bounds.add(i)
                if i + 1 < n and data[i + 1] not in b" \t":
                    bounds.add(i + 1)
            else:
                if i > 0:
                    bounds.add(i)
                if i + 1 < n:
                    bounds.add(i + 1)
        elif mode in {"json", "code"} and i > 0:
            prev = data[i - 1]
            if prev < 0x80 and b < 0x80:
                prev_digit = 0x30 <= prev <= 0x39
                curr_digit = 0x30 <= b <= 0x39
                if prev_digit != curr_digit and prev != ord("_") and b != ord("_"):
                    bounds.add(i)
    return bounds


def _canonical_json_bytes(value) -> bytes:
    import json
    if isinstance(value, (bytes, bytearray)):
        parsed = json.loads(bytes(value).decode("utf-8"))
    elif isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def encode_agentic_turn(tokenizer: MasterTokenizer, segments, *, loss_mask_observations: bool = True):
    """Encode a structured sequence of reasoning, tool_call, tool_obs, code, visual and talk segments."""
    if not isinstance(segments, (list, tuple)) or not 1 <= len(segments) <= 128:
        raise ValueError("Agentic turn requires between 1 and 128 segments")
    contract = agentic_contract(tokenizer)
    cids = contract["control_ids"]
    oids = contract["operator_ids"]
    token_ids = []
    mode_ids = []
    loss_mask = []
    encoded_segments = []

    for seg in segments:
        if not isinstance(seg, dict) or "kind" not in seg:
            raise ValueError("Each agentic segment must be an object with a kind")
        kind = seg["kind"]
        if kind not in {"think", "talk", "code", "visual", "tool_call", "tool_obs"}:
            raise ValueError(f"Unsupported agentic segment kind: {kind}")
        start_id = cids[f"{kind}_start"]
        end_id = cids[f"{kind}_end"]
        mode_code = MODE_CODES[kind]
        seg_train = 0 if (kind == "tool_obs" and loss_mask_observations) else 1

        if kind in {"tool_call", "tool_obs"}:
            operator = seg.get("operator")
            if operator not in oids:
                raise ValueError(f"Unadmitted agentic operator: {operator}")
            op_id = oids[operator]
            payload_key = "arguments" if kind == "tool_call" else "result"
            if payload_key not in seg:
                raise ValueError(f"Segment {kind} requires {payload_key}")
            payload_val = seg[payload_key]
            if kind == "tool_call" and not isinstance(payload_val, dict):
                raise ValueError("Tool call arguments must be a dictionary")
            raw = _canonical_json_bytes(payload_val)
            bounds = sorted(structural_boundaries(raw, mode="json"))
            record = tokenizer.encode(raw, boundaries=bounds)
            piece_ids = [span["active_id"] for span in record["spans"]]
            seg_tokens = [start_id, op_id] + piece_ids + [end_id]
            # For tool_obs when loss_mask_observations is True, mask the entire observation span
            # so the model is never trained to hallucinate environment/tool observations.
            seg_loss = [seg_train] * len(seg_tokens)
            token_ids.extend(seg_tokens)
            mode_ids.extend([mode_code] * len(seg_tokens))
            loss_mask.extend(seg_loss)
            encoded_segments.append({
                "kind": kind,
                "operator": operator,
                "operator_token_id": op_id,
                payload_key: payload_val,
                "byte_identity": record["byte_identity"],
                "byte_length": len(raw),
                "token_count": len(seg_tokens),
                "piece_ids": piece_ids,
                "boundaries": bounds,
            })
        else:
            content = seg.get("content")
            if isinstance(content, str):
                raw = content.encode("utf-8")
            elif isinstance(content, (bytes, bytearray)):
                raw = bytes(content)
            else:
                raise ValueError(f"Segment {kind} requires str or bytes content")
            boundary_mode = "code" if kind == "code" else ("markdown" if kind == "visual" else "plain")
            bounds = sorted(structural_boundaries(raw, mode=boundary_mode))
            record = tokenizer.encode(raw, boundaries=bounds)
            piece_ids = [span["active_id"] for span in record["spans"]]
            seg_tokens = [start_id] + piece_ids + [end_id]
            token_ids.extend(seg_tokens)
            mode_ids.extend([mode_code] * len(seg_tokens))
            loss_mask.extend([seg_train] * len(seg_tokens))
            encoded_segments.append({
                "kind": kind,
                "byte_identity": record["byte_identity"],
                "byte_length": len(raw),
                "token_count": len(seg_tokens),
                "piece_ids": piece_ids,
                "boundaries": bounds,
            })

    return {
        "schema": "nanoai.agentic-turn.v1",
        "tokenizer_identity": tokenizer.identity,
        "vocabulary_identity": tokenizer.vocabulary_identity,
        "token_ids": token_ids,
        "mode_ids": mode_ids,
        "loss_mask": loss_mask,
        "segments": encoded_segments,
    }


def decode_agentic_turn(tokenizer: MasterTokenizer, turn_or_tokens):
    """Losslessly decode an agentic turn or raw token ID stream into structured segments."""
    import json
    contract = agentic_contract(tokenizer)
    cids = contract["control_ids"]
    oids = contract["operator_ids"]
    inv_ops = {v: k for k, v in oids.items()}
    start_to_kind = {cids[f"{k}_start"]: k for k in ("think", "talk", "code", "visual", "tool_call", "tool_obs")}
    end_to_kind = {cids[f"{k}_end"]: k for k in ("think", "talk", "code", "visual", "tool_call", "tool_obs")}
    literal_limit = contract["literal_rows"]

    if isinstance(turn_or_tokens, dict):
        if turn_or_tokens.get("tokenizer_identity") != tokenizer.identity:
            raise ValueError("Agentic turn tokenizer identity mismatch")
        tokens = list(turn_or_tokens.get("token_ids", ()))
    elif isinstance(turn_or_tokens, (list, tuple)):
        tokens = list(turn_or_tokens)
    else:
        raise ValueError("Agentic turn input must be a turn dict or token sequence")

    # Strip optional framing BOS / EOS / PAD if caller passes a framed sequence
    while tokens and tokens[-1] == contract["pad_id"]:
        tokens.pop()
    if tokens and tokens[0] == contract["bos_id"]:
        tokens = tokens[1:]
    if tokens and tokens[-1] == contract["eos_id"]:
        tokens = tokens[:-1]
    if not tokens:
        raise ValueError("Agentic turn contains no segments")

    segments = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok not in start_to_kind:
            raise ValueError(f"Expected segment start control token at position {i}, got {tok}")
        kind = start_to_kind[tok]
        expected_end = cids[f"{kind}_end"]
        i += 1
        operator = None
        if kind in {"tool_call", "tool_obs"}:
            if i >= n or tokens[i] not in inv_ops:
                raise ValueError(f"Segment {kind} requires an admitted operator control token at position {i}")
            operator = inv_ops[tokens[i]]
            i += 1
        piece_ids = []
        while i < n and tokens[i] != expected_end:
            ptok = tokens[i]
            if type(ptok) is not int or not 0 <= ptok < literal_limit:
                raise ValueError(f"Unexpected control or out-of-range token {ptok} inside {kind} segment")
            piece_ids.append(ptok)
            i += 1
        if i >= n or tokens[i] != expected_end:
            raise ValueError(f"Unclosed {kind} segment")
        i += 1
        raw = b"".join(tokenizer.pieces[pid] for pid in piece_ids)
        text = raw.decode("utf-8")
        if kind == "tool_call":
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError("Decoded tool_call arguments must be a JSON object")
            segments.append({"kind": kind, "operator": operator, "arguments": parsed, "raw": raw})
        elif kind == "tool_obs":
            parsed = json.loads(text)
            segments.append({"kind": kind, "operator": operator, "result": parsed, "raw": raw})
        else:
            segments.append({"kind": kind, "content": text, "raw": raw})
    return segments


def agentic_causal_batch(tokenizer: MasterTokenizer, turns, *, max_tokens: int = 4096, loss_mask_observations: bool = True):
    """Prepare a causal training/inference batch from structured multi-segment agentic turns."""
    if not isinstance(turns, (list, tuple)) or not 1 <= len(turns) <= 256:
        raise ValueError("Agentic batch extent")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 131073:
        raise ValueError("Agentic sequence extent")
    contract = agentic_contract(tokenizer)
    encoded_turns = [
        turn if (isinstance(turn, dict) and turn.get("schema") == "nanoai.agentic-turn.v1")
        else encode_agentic_turn(tokenizer, turn, loss_mask_observations=loss_mask_observations)
        for turn in turns
    ]
    width = max(len(t["token_ids"]) + 1 for t in encoded_turns)
    if width > max_tokens or width * len(encoded_turns) > 1048576:
        raise ValueError("Agentic batch token budget; input remains untruncated")

    inputs, labels, attention_masks, mode_Matrix, loss_masks, lengths = [], [], [], [], [], []
    for t in encoded_turns:
        # Verify lossless round-trip of every turn before batching
        decode_agentic_turn(tokenizer, t)
        path = t["token_ids"]
        modes = t["mode_ids"]
        lmask = t["loss_mask"]
        length = len(path) + 1
        padding = width - length
        inp = [contract["bos_id"]] + path + [contract["pad_id"]] * padding
        raw_targets = path + [contract["eos_id"]]
        target_mask = lmask + [1]
        lbl = [
            tok if mask_bit == 1 else contract["label_ignore"]
            for tok, mask_bit in zip(raw_targets, target_mask)
        ] + [contract["label_ignore"]] * padding
        inputs.append(inp)
        labels.append(lbl)
        attention_masks.append([1] * length + [0] * padding)
        mode_Matrix.append([MODE_CODES["pad"]] + modes + [MODE_CODES["pad"]] * padding)
        loss_masks.append(target_mask + [0] * padding)
        lengths.append(length)

    return {
        "contract": contract,
        "loss_mask_observations": bool(loss_mask_observations),
        "input_ids": inputs,
        "labels": labels,
        "attention_mask": attention_masks,
        "mode_ids": mode_Matrix,
        "loss_mask": loss_masks,
        "lengths": lengths,
        "turns": encoded_turns,
    }


def audit_agentic_causal_masking(tokenizer: MasterTokenizer, batch) -> dict:
    """Directly verify that tool_obs targets are excluded from loss while remaining active in causal history for downstream supervised tokens."""
    if not isinstance(batch, dict) or batch.get("contract") != agentic_contract(tokenizer):
        raise ValueError("Agentic model input contract mismatch")
    contract = batch["contract"]
    ignore_id = contract["label_ignore"]
    obs_mode = MODE_CODES["tool_obs"]
    mask_obs = bool(batch.get("loss_mask_observations", True))
    turns = batch.get("turns")
    input_ids = batch.get("input_ids")
    labels = batch.get("labels")
    attention_mask = batch.get("attention_mask")
    mode_ids = batch.get("mode_ids")
    loss_mask = batch.get("loss_mask")
    if not (
        isinstance(turns, list)
        and isinstance(input_ids, list)
        and isinstance(labels, list)
        and isinstance(attention_mask, list)
        and isinstance(mode_ids, list)
        and isinstance(loss_mask, list)
        and len(turns) == len(input_ids) == len(labels) == len(attention_mask) == len(mode_ids) == len(loss_mask)
    ):
        raise ValueError("Agentic batch arrays must have matching row counts")

    masked_obs_targets = 0
    active_obs_history = 0
    conditioned_supervised_targets = 0
    row_reports = []

    for row_idx, (turn, inp, lbl, attn, mrow, lrow) in enumerate(
        zip(turns, input_ids, labels, attention_mask, mode_ids, loss_mask)
    ):
        path = list(turn.get("token_ids", ()))
        t_modes = list(turn.get("mode_ids", ()))
        if len(path) != len(t_modes):
            raise ValueError(f"Row {row_idx} turn token_ids and mode_ids length mismatch")
        if not (len(inp) == len(lbl) == len(attn) == len(mrow) == len(lrow)):
            raise ValueError(f"Row {row_idx} batch width mismatch")
        if inp[0] != contract["bos_id"] or attn[0] != 1:
            raise ValueError(f"Row {row_idx} missing active BOS prefix")

        obs_input_positions = []
        row_masked_obs = 0
        row_conditioned_supervised = 0

        for t, (tok, mode_code) in enumerate(zip(path, t_modes)):
            input_pos = t + 1
            if inp[input_pos] != tok:
                raise ValueError(f"Row {row_idx} input_ids[{input_pos}] does not match turn token at {t}")
            if mrow[input_pos] != mode_code:
                raise ValueError(f"Row {row_idx} mode_ids[{input_pos}] does not match turn mode at {t}")
            if attn[input_pos] != 1:
                raise ValueError(f"Row {row_idx} active token at input_ids[{input_pos}] has attention_mask != 1")

            if mode_code == obs_mode:
                obs_input_positions.append(input_pos)
                active_obs_history += 1
                if mask_obs:
                    if lbl[t] != ignore_id or lrow[t] != 0:
                        raise ValueError(
                            f"Row {row_idx} target position {t} is tool_obs but leaked into supervision "
                            f"(label={lbl[t]}, loss_mask={lrow[t]})"
                        )
                    masked_obs_targets += 1
                    row_masked_obs += 1
            else:
                if lbl[t] != tok or lrow[t] != 1:
                    raise ValueError(
                        f"Row {row_idx} non-observation target position {t} is not supervised "
                        f"(label={lbl[t]}, expected={tok}, loss_mask={lrow[t]})"
                    )
                if obs_input_positions:
                    # Target t is predicted from causal prefix inp[:t+1], which includes every prior tool_obs input_pos <= t
                    prefix_obs = [pos for pos in obs_input_positions if pos <= t]
                    if prefix_obs:
                        if any(attn[pos] != 1 or inp[pos] != path[pos - 1] for pos in prefix_obs):
                            raise ValueError(
                                f"Row {row_idx} supervised target at {t} lost tool_obs history in causal prefix"
                            )
                        conditioned_supervised_targets += 1
                        row_conditioned_supervised += 1

        # EOS target at index len(path) is predicted from the full sequence inp[:len(path)+1]
        eos_t = len(path)
        if lbl[eos_t] != contract["eos_id"] or lrow[eos_t] != 1:
            raise ValueError(f"Row {row_idx} EOS target at {eos_t} must remain supervised")
        if obs_input_positions:
            conditioned_supervised_targets += 1
            row_conditioned_supervised += 1

        row_reports.append({
            "row": row_idx,
            "masked_observation_targets": row_masked_obs,
            "active_observation_history_tokens": len(obs_input_positions),
            "conditioned_supervised_targets": row_conditioned_supervised,
        })

    return {
        "loss_mask_observations": mask_obs,
        "row_count": len(turns),
        "masked_observation_target_count": masked_obs_targets,
        "active_observation_history_count": active_obs_history,
        "conditioned_supervised_target_count": conditioned_supervised_targets,
        "rows": row_reports,
    }


def validate_agentic_batch(tokenizer: MasterTokenizer, batch):
    """Validate an agentic causal batch and return the decoded structured turns."""
    if not isinstance(batch, dict) or batch.get("contract") != agentic_contract(tokenizer):
        raise ValueError("Agentic model input contract mismatch")
    turns = batch.get("turns")
    if not isinstance(turns, list) or not 1 <= len(turns) <= 256:
        raise ValueError("Agentic turn record extent")
    audit_agentic_causal_masking(tokenizer, batch)
    decoded = [decode_agentic_turn(tokenizer, t) for t in turns]
    rebuilt = agentic_causal_batch(
        tokenizer,
        decoded,
        max_tokens=131073,
        loss_mask_observations=batch.get("loss_mask_observations", True),
    )
    if batch != rebuilt:
        raise ValueError("Agentic model input shifted IDs, labels, mode_ids or loss_mask mismatch")
    return decoded

