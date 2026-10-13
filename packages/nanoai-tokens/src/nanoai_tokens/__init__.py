"""Exact-byte tokenization through the native C++ core."""
from pathlib import Path
from .tokenizer import Tokenizer

__version__ = "0.1.0"


def frozen_path():
    """Installed immutable NMT1 artifact, with its source/license records beside it."""
    return Path(__file__).parent / "data/master-tokenizer-v1/master.nmt"


def native_source():
    return Path(__file__).parent / "native/master_tokenizer.cpp"


def load(path=None, *, expected_identity=None):
    return Tokenizer.load(frozen_path() if path is None else path, expected_identity=expected_identity)
