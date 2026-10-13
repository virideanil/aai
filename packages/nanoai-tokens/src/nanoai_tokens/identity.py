"""Byte identities shared by the standalone format and its consumers."""
import hashlib


def digest_bytes(data: bytes) -> str:
    return "shake256:" + hashlib.shake_256(data).hexdigest(64)
