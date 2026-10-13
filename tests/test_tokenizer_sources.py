import base64
import json
from pathlib import Path
import struct
import tempfile
import unittest

from nanoai.core import digest_bytes
from nanoai.tokenizer_sources import build_common_inventory, load_common_inventory, read_inventory


class TokenizerSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def spec(self, name, kind, data):
        path = self.root / name
        path.write_bytes(data)
        return {"name": name, "kind": kind, "path": str(path), "digest": digest_bytes(data)}

    def test_partial_bytes_and_native_aliases_survive_real_dictionary_reload(self):
        tiktoken = self.spec("bytes", "tiktoken", b"\n".join(base64.b64encode(p) + b" " + str(i).encode()
                             for i, p in enumerate([b"a", b"\xc3", b"\xa9", b"\xff"])) + b"\n")
        wordpiece = self.spec("turkish", "wordpiece", "[UNK]\na\n##a\nş\n##ş\n\u2028\n".encode())
        first, second = read_inventory(tiktoken), read_inventory(wordpiece)
        self.assertEqual(first.metadata["partial_utf8_entries"], 3)
        self.assertEqual(first.surfaces[3], b"\xff")
        self.assertEqual(second.surfaces[1], second.surfaces[2])
        self.assertEqual((second.roles[1], second.roles[2]), ("word_start", "word_continuation"))
        self.assertEqual(second.surfaces[5], "\u2028".encode())
        output = self.root / "inventory"
        manifest = build_common_inventory([tiktoken, wordpiece], output)
        restored = load_common_inventory(output, expected_identity=manifest["inventory_identity"])
        raw = b"\xff\x00\n\t" + "İstanbul é 🧬".encode()
        self.assertEqual(restored.decode(restored.encode(raw)), raw)
        mapped = dict(struct.iter_unpack("<II", (output / "source-1.map").read_bytes()))
        self.assertEqual(mapped[1], mapped[2])
        self.assertEqual(set(mapped), {1, 2, 3, 4, 5})
        self.assertIn("0", manifest["sources"][1]["controls"])

    def test_multi_digit_control_ids_keep_manifest_identity_after_json_reload(self):
        data = {"model": {"type": "BPE", "vocab": {"a": 0}},
                "decoder": {"type": "ByteLevel"}, "added_tokens": [
                    {"id": i, "content": f"<control-{i}>", "special": True}
                    for i in (2, 10, 31, 100)]}
        spec = self.spec("controls", "hf-tokenizer", json.dumps(data).encode())
        output = self.root / "inventory"
        manifest = build_common_inventory([spec], output)
        serialized = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest, serialized)
        self.assertEqual(set(serialized["sources"][0]["controls"]), {"2", "10", "31", "100"})
        restored = load_common_inventory(output, expected_identity=manifest["inventory_identity"])
        self.assertEqual(restored.identity, manifest["vocabulary_identity"])

    def test_hf_byte_fallback_and_controls_are_typed(self):
        data = {"model": {"type": "BPE", "byte_fallback": True,
                          "vocab": {"<unk>": 0, "<0xC3>": 1, "▁café": 2}, "unk_token": "<unk>"},
                "normalizer": {"type": "NFC"}, "pre_tokenizer": None,
                "decoder": {"type": "Sequence", "decoders": [
                    {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                    {"type": "ByteFallback"}, {"type": "Fuse"}]},
                "added_tokens": [{"id": 3, "content": "<stop>", "special": True}]}
        inventory = read_inventory(self.spec("fallback", "hf-tokenizer", json.dumps(data).encode()))
        self.assertEqual(inventory.surfaces, {1: b"\xc3", 2: " café".encode()})
        self.assertEqual(set(inventory.controls), {0, 3})
        self.assertEqual(inventory.metadata["rules"]["normalizer"], {"type": "NFC"})
        data["added_tokens"].append(dict(data["added_tokens"][0]))
        duplicate = read_inventory(self.spec("identical", "hf-tokenizer", json.dumps(data).encode()))
        self.assertEqual(duplicate.controls, inventory.controls)
        data["added_tokens"][-1]["content"] = "<other>"
        with self.assertRaisesRegex(ValueError, "Conflicting added-token ID"):
            read_inventory(self.spec("conflicting", "hf-tokenizer", json.dumps(data).encode()))

    def test_sentencepiece_keeps_type_score_and_normalizer(self):
        try:
            from sentencepiece import sentencepiece_model_pb2 as pb
        except ImportError:
            self.skipTest("SentencePiece source importer dependency is unavailable")
        model = pb.ModelProto()
        model.normalizer_spec.name = "identity"
        for text, kind, score in [("<unk>", 2, 0), ("▁a", 1, -2.0), ("<0xC3>", 6, -3.0), ("<s>", 3, 0)]:
            piece = model.pieces.add(); piece.piece = text; piece.type = kind; piece.score = score
        inventory = read_inventory(self.spec("spm", "sentencepiece", model.SerializeToString()))
        self.assertEqual(inventory.surfaces, {1: b" a", 2: b"\xc3"})
        self.assertEqual(inventory.scores, {1: -2.0, 2: -3.0})
        self.assertEqual(set(inventory.controls), {0, 3})
        self.assertEqual(inventory.metadata["rules"]["normalizer"]["name"], "identity")

    def test_literal_added_whitespace_survives_bytelevel_import(self):
        data = {"model": {"type": "BPE", "vocab": {"a": 0, "   ": 1}},
                "decoder": {"type": "ByteLevel"}, "added_tokens": [
                    {"id": 1, "content": "   ", "special": False, "normalized": False},
                    {"id": 2, "content": "  \t", "special": False, "normalized": False}]}
        inventory = read_inventory(self.spec("spaces", "hf-tokenizer", json.dumps(data).encode()))
        self.assertEqual(inventory.surfaces, {0: b"a", 1: b"   ", 2: b"  \t"})
        self.assertEqual(inventory.roles[1], "user_defined")
        self.assertEqual(inventory.controls, {})

    def test_changed_payload_repeated_id_and_foreign_inventory_reject(self):
        spec = self.spec("bytes", "tiktoken", b"YQ== 0\n")
        Path(spec["path"]).write_bytes(b"Yg== 0\n")
        with self.assertRaisesRegex(ValueError, "source bytes differ"):
            read_inventory(spec)
        spec = self.spec("duplicate", "tiktoken", b"YQ== 0\nYg== 0\n")
        with self.assertRaisesRegex(ValueError, "Native ID"):
            read_inventory(spec)
        spec = self.spec("valid", "tiktoken", b"YQ== 0\n")
        output = self.root / "inventory"
        build_common_inventory([spec], output)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            load_common_inventory(output, expected_identity="foreign")
        path = output / "source-0.roles"
        path.write_bytes(b"\xff")
        with self.assertRaisesRegex(ValueError, "witness file differs"):
            load_common_inventory(output)


if __name__ == "__main__":
    unittest.main()
