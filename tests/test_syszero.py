import copy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from nanoai.syszero import CommonVocabulary


class SystemZeroTests(unittest.TestCase):
    def vocabulary(self):
        return CommonVocabulary([
            SimpleNamespace(identity="a", surfaces={0: b"cat", 1: b"cat ", 2: b"sat"}),
            SimpleNamespace(identity="b", surfaces={0: b"dog", 1: b"cat", 2: b" sat"})])

    def test_shared_surface_keeps_native_witnesses_separate(self):
        common = self.vocabulary()
        self.assertEqual(common.witnesses(common.ids[b"cat"]), [
            {"source_identity": "a", "native_id": 0}, {"source_identity": "b", "native_id": 1}])
        self.assertNotEqual(common.ids[b"cat"], common.ids[b"dog"])

    def test_all_bytes_and_unicode_fragments_remain_exact(self):
        common = self.vocabulary()
        for raw in [bytes(range(256)), "İstanbul e\u0301 🧪".encode(), b"cat sat", b""]:
            self.assertEqual(common.decode(common.encode(raw)), raw)

    def test_minimum_piece_segmentation_is_not_greedy(self):
        common = self.vocabulary()
        encoded = common.encode(b"cat sat")
        self.assertEqual(len(encoded["spans"]), 2)
        self.assertEqual(common.decode(encoded), b"cat sat")

    def test_changed_vocabulary_or_spans_cannot_silently_decode(self):
        common = self.vocabulary()
        encoded = common.encode(b"cat")
        damaged = copy.deepcopy(encoded)
        damaged["spans"][0]["end"] += 1
        with self.assertRaisesRegex(ValueError, "spans"):
            common.decode(damaged)
        damaged = copy.deepcopy(encoded)
        damaged["vocabulary_identity"] = "different"
        with self.assertRaisesRegex(ValueError, "version"):
            common.decode(damaged)

    def test_persisted_dictionary_is_verified_in_a_fresh_reader(self):
        common = CommonVocabulary([])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "common"
            common.save(path)
            restored = CommonVocabulary.load(path)
            self.assertEqual(restored.decode(common.encode(bytes(range(256)))), bytes(range(256)))
            with (path / "pieces.bin").open("ab") as stream:
                stream.write(b"extra")
            with self.assertRaisesRegex(ValueError, "changed"):
                CommonVocabulary.load(path)


if __name__ == "__main__":
    unittest.main()
