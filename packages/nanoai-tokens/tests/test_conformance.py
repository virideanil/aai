import hashlib
import json
from pathlib import Path
import unittest

from nanoai_tokens import _native
from nanoai_tokens.reference import MasterTokenizer


class ConformanceTests(unittest.TestCase):
    def test_shared_artifact_admission(self):
        root = Path(__file__).with_name("conformance")
        manifest = json.loads((root / "manifest.json").read_text())
        for case in manifest["cases"]:
            path = root / case["file"]
            data = path.read_bytes()
            self.assertEqual("shake256:" + hashlib.shake_256(data).hexdigest(64), case["file_identity"])
            self.assertEqual(len(data), case["bytes"])
            for name, parse in (("reference", lambda: MasterTokenizer.load(path)),
                                ("native", lambda: _native.create(data))):
                with self.subTest(case=case["id"], consumer=name):
                    if case["accept"]:
                        self.assertIsNotNone(parse())
                    else:
                        with self.assertRaises(ValueError, msg=case["violation"]): parse()


if __name__ == "__main__": unittest.main()
