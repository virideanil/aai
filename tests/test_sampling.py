import unittest

from nanoai.sampling import GREEDY, QWEN35_THINKING, observed_policy, profile


class SamplingTests(unittest.TestCase):
    def test_complete_numeric_policy_preserves_each_value_and_returns_a_copy(self):
        policy = profile(QWEN35_THINKING)
        self.assertEqual(policy, QWEN35_THINKING)
        policy["seed"] = 123
        self.assertEqual(QWEN35_THINKING["seed"], 0)
        self.assertEqual(profile(), GREEDY)
        self.assertEqual(observed_policy("greedy-token-v1"), "greedy-token-v1")
        self.assertEqual(observed_policy(policy), policy)

    def test_nonfinite_ambiguous_and_unsupported_values_reject(self):
        for change in ({"temperature": True}, {"temperature": float("nan")},
                       {"top_p": 0}, {"min_p": -1}, {"top_k": 3.5},
                       {"seed": -1}, {"seed": 2**32}, {"repeat_last_n": True},
                       {"frequency_penalty": float("inf")}, {"extra": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                profile({**GREEDY, **change})
        with self.assertRaises(ValueError):
            profile({k: v for k, v in GREEDY.items() if k != "seed"})
        with self.assertRaises(ValueError):
            observed_policy("sampled")


if __name__ == "__main__":
    unittest.main()
