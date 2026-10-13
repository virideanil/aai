"""Explicit native dependency for lossless BF16 blocks and projection consumers."""
import ctypes as C
import os
from pathlib import Path


MODES = {"binary": 0, "binary_trit5": 1, "mixed": 2, "binary12": 3, "ternary_exponent": 4,
         "binary11": 5, "ternary_exponent_bytes": 6}


class Library:
    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        self.lib = C.CDLL(str(self.path))
        if not hasattr(self.lib, "harmony_abi_version") or self.lib.harmony_abi_version() != 2:
            raise ValueError("Harmony library ABI differs; rebuild the pinned source")
        self.lib.harmony_error.restype = C.c_char_p
        signatures = {
            "harmony_matrix": ([C.c_void_p, C.c_size_t, C.c_size_t, C.c_int], C.c_void_p),
            "harmony_free": ([C.c_void_p], None),
            "harmony_stats": ([C.c_void_p, C.POINTER(C.c_uint64)], C.c_int),
            "harmony_unpack": ([C.c_void_p, C.c_void_p, C.c_uint], C.c_int),
            "harmony_cache": ([C.c_void_p, C.c_uint], C.c_int),
            "harmony_project": ([C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t, C.c_uint, C.c_int], C.c_int),
            "harmony_pack_file": ([C.c_char_p, C.c_char_p, C.c_int], C.c_int),
            "harmony_unpack_file": ([C.c_char_p, C.c_char_p], C.c_int),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, name); fn.argtypes = args; fn.restype = result

    def check(self, code):
        if code:
            raise ValueError(self.lib.harmony_error().decode())

    def pack_file(self, source, target, mode="mixed"):
        self.check(self.lib.harmony_pack_file(os.fsencode(source), os.fsencode(target), MODES[mode]))

    def unpack_file(self, source, target):
        self.check(self.lib.harmony_unpack_file(os.fsencode(source), os.fsencode(target)))


class Matrix:
    def __init__(self, library, words, mode="mixed"):
        import numpy as np
        words = np.ascontiguousarray(words, dtype=np.uint16)
        if words.ndim != 2:
            raise ValueError("A BF16 word matrix requires two dimensions")
        self.library, self.shape = library, words.shape
        self.handle = library.lib.harmony_matrix(words.ctypes.data, *words.shape, MODES[mode])
        if not self.handle:
            raise ValueError(library.lib.harmony_error().decode())

    def stats(self):
        self._live()
        values = (C.c_uint64 * 10)()
        self.library.check(self.library.lib.harmony_stats(self.handle, values))
        result = dict(zip(("serialized_block_bytes", "raw_binary_blocks", "binary_trit5_blocks", "binary12_blocks",
            "ternary_exponent_blocks", "binary11_blocks", "ternary_exponent_byte_blocks",
            "block_object_bytes", "decoded_cache_bytes", "original_bf16_bytes"), values))
        result["binary_blocks"] = result["raw_binary_blocks"] + result["binary12_blocks"] + result["binary11_blocks"]
        return result

    def _live(self):
        if not self.handle:
            raise ValueError("Matrix is closed")

    def unpack(self, threads=1):
        import numpy as np
        self._live()
        result = np.empty(self.shape, dtype=np.uint16)
        self.library.check(self.library.lib.harmony_unpack(self.handle, result.ctypes.data, threads))
        return result

    def prepare_cache(self, threads=1):
        self._live()
        self.library.check(self.library.lib.harmony_cache(self.handle, threads))

    def project(self, x, *, threads=1, cached=False, dense_baseline=False):
        import numpy as np
        self._live()
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.shape[1]:
            raise ValueError("Projection input shape")
        output = np.empty((x.shape[0], self.shape[0]), dtype=np.float32)
        cache_mode = 2 if dense_baseline else int(bool(cached))
        self.library.check(self.library.lib.harmony_project(self.handle, x.ctypes.data, output.ctypes.data, x.shape[0], threads, cache_mode))
        return output

    def close(self):
        if getattr(self, "handle", None):
            self.library.lib.harmony_free(self.handle); self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
