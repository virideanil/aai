import itertools
import unittest

from nanoai.meaning import assess, denotation, parse_checked_english, render, separate, separate_meanings


class MeaningTests(unittest.TestCase):
    symbols = {"x": "the door is open", "y": "the lamp is lit", "z": "the pump is running"}

    def test_roundtrip_composition_and_independent_truth_tables(self):
        laws = {"and": lambda a,b: bool(a) and bool(b), "or": lambda a,b: bool(a) or bool(b),
                "xor": lambda a,b: bool(a) != bool(b), "eq": lambda a,b: bool(a) == bool(b)}
        for op, law in laws.items():
            formula = {"not": {op: ["x", {"not": "y"}]}}
            english = render(formula, self.symbols)
            self.assertEqual(parse_checked_english(english, self.symbols), formula)
            independent = tuple(int(not law(x, not y)) for x,y,z in itertools.product((0,1),repeat=3))
            self.assertEqual(denotation(formula,self.symbols), independent)

    def test_equivalence_accepts_different_syntax_and_preserves_variable_identity(self):
        first={"not":{"and":["x","y"]}}
        second={"or":[{"not":"x"},{"not":"y"}]}
        self.assertIsNone(separate(first,second,self.symbols))
        self.assertIsNone(separate({"xor":["x","x"]},0,self.symbols))
        self.assertIsNotNone(separate({"xor":["x","y"]},0,self.symbols))

    def test_one_correct_output_does_not_certify_meaning(self):
        criteria={"direction":"english_to_math","symbols":self.symbols,
                  "expression":{"and":["x","y"]},"assignment":{"x":1,"y":1,"z":0}}
        result=assess('{"or":["x","y"]}',criteria)
        self.assertTrue(result["current_execution"])
        self.assertFalse(result["same_denotation"])
        witness=separate({"and":["x","y"]},{"or":["x","y"]},self.symbols)
        self.assertNotEqual(witness["first"],witness["second"])

    def test_unsupported_and_ambiguous_language_is_explicit(self):
        for text in ['the door is open or the lamp is lit and the pump is running',
                     'both (the door is open) and (the lamp is lit). Additional promise.']:
            with self.assertRaises(ValueError):parse_checked_english(text,self.symbols)
        with self.assertRaises(ValueError):render('outside',self.symbols)
        with self.assertRaises(ValueError):render('x',{'x':'true'})

    def test_ambiguous_readings_survive_agreement_in_one_situation(self):
        readings={'alternatives':[{'and':['x','y']},{'or':['x','y']}]}
        self.assertEqual(parse_checked_english(render(readings,self.symbols),self.symbols),readings)
        criteria={'direction':'english_to_math','symbols':self.symbols,'expression':readings,
                  'assignment':{'x':1,'y':1,'z':0}}
        result=assess('{"and":["x","y"]}',criteria)
        self.assertTrue(result['current_execution']);self.assertFalse(result['same_denotation'])

    def test_ambiguity_preserves_joint_identity_across_tests(self):
        first={'alternatives':['x',{'not':'x'}]}
        second={'alternatives':['y',{'not':'y'}]}
        # Every individual assignment has both possible values in both sets.
        # Two linked observations distinguish the functions themselves.
        witness=separate_meanings(first,second,self.symbols)
        self.assertIsNotNone(witness)
        self.assertEqual(len(witness['assignments']),2)
        self.assertNotEqual(witness['first_traces'],witness['second_traces'])


if __name__=='__main__':unittest.main()
