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
