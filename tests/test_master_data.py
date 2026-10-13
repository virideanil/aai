import copy
import json
import unittest

from nanoai.master_data import causal_batch, validate_batch
from test_master_tokenizer import fixture


class MasterDataTests(unittest.TestCase):
    def test_next_token_alignment_and_empty_byte_record(self):
        tokenizer = fixture()
        rows = [b"", b"aba", bytes(range(256)), "İstanbul 🧬".encode()]
        batch = causal_batch(tokenizer, rows)
        self.assertEqual(validate_batch(tokenizer, json.loads(json.dumps(batch))), rows)
        self.assertEqual(batch["input_ids"][0][0], batch["contract"]["bos_id"])
        self.assertEqual(batch["labels"][0][0], batch["contract"]["eos_id"])
        self.assertEqual(batch["labels"][0][1], -100)
        self.assertEqual(batch["attention_mask"][0][1], 0)
        self.assertEqual(batch["contract"]["model_rows"], len(tokenizer.pieces) + 3)

    def test_mutated_or_foreign_learning_inputs_reject(self):
        tokenizer = fixture()
        batch = causal_batch(tokenizer, [b"aba", b"b"])
        for key in ("input_ids", "labels", "attention_mask"):
            changed = copy.deepcopy(batch); changed[key][0][0] = 99999
            with self.assertRaises(ValueError):
                validate_batch(tokenizer, changed)
        changed = copy.deepcopy(batch); changed["contract"]["tokenizer_identity"] = "foreign"
        with self.assertRaises(ValueError):
            validate_batch(tokenizer, changed)
        with self.assertRaises(ValueError):
            causal_batch(tokenizer, [b"aba"], max_tokens=1)


if __name__ == "__main__":
    unittest.main()
