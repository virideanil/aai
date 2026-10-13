from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from artifact_facts import assess


class ArtifactFactTests(unittest.TestCase):
    contract = {"must_contain": [{"value": 19, "unit": "cm"}, {"value": 38, "unit": "cm"}],
        "must_not_contain": [{"value": 11, "unit": "m"}, {"value": 23, "unit": "cm"}]}

    def test_bilingual_quantity_forms(self):
        for text in ["19cm each; **38 cm** total.", "Nineteen centimetres each; thirty-eight centimeters total.",
                     "Her çubuk on dokuz santimetre; toplam otuz sekiz santimetre.", "１９ cm; 38,0 cm."]:
            self.assertTrue(assess(text, self.contract)["passed"], text)

    def test_citation_does_not_mask_an_obsolete_fact(self):
        for old in ["11 m", "eleven metres", "on bir metre"]:
            r = assess("19 cm, 38 cm. Eski ölçü: " + old + " [spec@2]", self.contract)
            self.assertFalse(r["passed"])
            self.assertEqual(r["obsolete"], [{"value": 11, "unit": "m"}])

    def test_substrings_and_bare_numbers_do_not_supply_facts(self):
        for text in ["119 cm; 138 cm", "19 and 38", "19.5 cm and 38 cm"]:
            self.assertFalse(assess(text, self.contract)["passed"])


if __name__ == "__main__":
    unittest.main()
