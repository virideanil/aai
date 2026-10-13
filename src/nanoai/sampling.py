"""Explicit decoding settings, retained with each generated experience."""
import math

SCHEMA = "nanoai.token-sampling.v1"
GREEDY = {"schema": SCHEMA, "temperature": 0.0, "top_p": 1.0, "top_k": 0,
          "min_p": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
          "repeat_penalty": 1.0, "repeat_last_n": 64, "seed": 0}
# Qwen3.5's text-thinking recommendation. The native penalty window is explicit;
# this candidate still needs task-specific qualification, including retention.
QWEN35_THINKING = {**GREEDY, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                   "presence_penalty": 1.5}


def profile(value=None):
    if value is None:
        return dict(GREEDY)
    if not isinstance(value, dict) or set(value) != set(GREEDY) or value["schema"] != SCHEMA:
        raise ValueError("Token sampling profile fields")
    for key, low, high in (("temperature", 0, 2), ("top_p", 0, 1), ("min_p", 0, 1),
                           ("presence_penalty", 0, 2), ("frequency_penalty", 0, 2), ("repeat_penalty", 1, 2)):
        item = value[key]
        if type(item) not in (int, float) or not math.isfinite(item) or not low <= item <= high:
            raise ValueError("Token sampling parameter: " + key)
    if value["top_p"] == 0:
        raise ValueError("Token sampling top_p must be positive")
    for key, low, high in (("top_k", 0, 248320), ("repeat_last_n", 0, 262144), ("seed", 0, 2**32-1)):
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise ValueError("Token sampling parameter: " + key)
    return dict(value)


def observed_policy(value):
    # Existing experiments retain their exact historical contract.
    if value == "greedy-token-v1":
        return value
    return profile(value)
