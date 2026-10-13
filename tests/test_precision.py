from pathlib import Path
import struct
import tempfile
import unittest

from nanoai.core import JsonlCore
from nanoai.precision import inspect_bf16


def fixture(kind):
    header = (b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + struct.pack("<Q", 6) + b"weight"
        + struct.pack("<IQQIQ", 2, 2, 2, kind, 0))
    return header + bytes((-len(header)) % 32) + bytes(16 if kind == 0 else 8)


class PrecisionPolicyTests(unittest.TestCase):
    def test_real_metadata_types_govern_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "model.gguf"
            p.write_bytes(fixture(30))
            self.assertEqual(inspect_bf16(p)["tensor_types"], {"BF16": 1})
            for kind in [0, 1, 2, 12, 34, 35]:
                p.write_bytes(fixture(kind))
                with self.assertRaisesRegex(ValueError, "BF16"):
                    inspect_bf16(p)
            p.write_bytes(fixture(30)[:-1])
            with self.assertRaisesRegex(ValueError, "Truncated"):
                inspect_bf16(p)

    def test_old_quantized_configuration_is_rejected_before_process_start(self):
        with self.assertRaisesRegex(ValueError, "original BF16"):
            JsonlCore({"command": ["must-not-run"]}, stderr_path="must-not-create.log")


if __name__ == "__main__":
    unittest.main()
