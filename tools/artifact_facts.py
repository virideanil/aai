"""Bounded, bilingual quantity checks for the authored artifact panel.

These checks detect specified required and obsolete measurements. They are an
evaluation oracle for these cases, rather than a general semantic verifier.
"""
import re
import unicodedata


ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "sıfır": 0, "bir": 1, "iki": 2, "üç": 3, "dört": 4, "beş": 5,
    "altı": 6, "yedi": 7, "sekiz": 8, "dokuz": 9,
}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "on": 10, "yirmi": 20, "otuz": 30, "kırk": 40, "elli": 50,
    "altmış": 60, "yetmiş": 70, "seksen": 80, "doksan": 90}


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("i\u0307", "i")
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"\b(?:centimet(?:er|re)s?|santimetre(?:dir)?)\b", "cm", text)
    text = re.sub(r"\b(?:met(?:er|re)s?|metre(?:dir)?)\b", "m", text)
    for tens, value in TENS.items():
        for one, digit in ONES.items():
            if 0 < digit < 10:
                text = re.sub(r"\b" + re.escape(tens) + r"[\s-]+" + re.escape(one) + r"\b", str(value + digit), text)
    for word, value in {**ONES, **TENS}.items():
        text = re.sub(r"\b" + re.escape(word) + r"\b", str(value), text)
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    return text


def assess(text, contract):
    normalized = normalize(text)
    def contains(fact):
        value = re.escape(str(fact["value"]))
        unit = re.escape(fact["unit"])
        return re.search(r"(?<![\w.])" + value + r"(?:\.0+)?\s*" + unit + r"\b", normalized) is not None
    missing = [f for f in contract["must_contain"] if not contains(f)]
    obsolete = [f for f in contract["must_not_contain"] if contains(f)]
    return {"passed": not missing and not obsolete, "missing": missing, "obsolete": obsolete,
        "scope": "Declared quantity facts in committed document bytes"}
