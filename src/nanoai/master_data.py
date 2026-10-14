"""Shared token-core causal and agentic input contracts, retained at the existing import path."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import nanoai_tokens

_LOCAL_DATA = Path(__file__).resolve().parents[2] / "packages" / "nanoai-tokens" / "src" / "nanoai_tokens" / "data.py"
if _LOCAL_DATA.is_file():
    _spec = spec_from_file_location("nanoai_tokens.data", _LOCAL_DATA)
    if _spec is not None and _spec.loader is not None:
        _mod = module_from_spec(_spec)
        _mod.__package__ = "nanoai_tokens"
        sys.modules["nanoai_tokens.data"] = _mod
        setattr(nanoai_tokens, "data", _mod)
        _spec.loader.exec_module(_mod)

from nanoai_tokens.data import (
    ADMITTED_OPERATORS,
    AGENTIC_CONTROLS,
    MODE_CODES,
    OPERATOR_VOCABULARY_VERSION,
    agentic_causal_batch,
    agentic_contract,
    audit_agentic_causal_masking,
    causal_batch,
    decode_agentic_turn,
    encode_agentic_turn,
    model_contract,
    structural_boundaries,
    validate_agentic_batch,
    validate_batch,
)
