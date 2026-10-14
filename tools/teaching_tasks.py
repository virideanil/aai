"""Authored bounded teaching tasks and their independent mechanical rubric.

This rubric measures the declared short repair contract. It does not infer
literary merit or certify arbitrary reasoning from a quoted source.
"""
from copy import deepcopy
import json
from decimal import Decimal
from pathlib import Path
import re

from nanoai.core import file_digest
from nanoai.steps import strict_json

QUESTION = ("Review the draft against the current source and any proposed criticism. "
            "Return only JSON with exactly decision (keep or revise), evidence (one short exact source quotation), "
            "and text (the final sentence, at most 24 words). Keep a sound draft exactly; repair a real conflict.")

SMALL = ('zero one two three four five six seven eight nine ten eleven twelve '
         'thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split()
TENS = 'zero ten twenty thirty forty fifty sixty seventy eighty ninety'.split()


def english_integer(value):
    """Ordinary English forms for the bounded authored quantity panel."""
    if not 0 <= value < 1000:
        return None
    if value < 20:
        return SMALL[value]
    if value < 100:
        return TENS[value // 10] + (' ' + SMALL[value % 10] if value % 10 else '')
    return SMALL[value // 100] + ' hundred' + (' ' + english_integer(value % 100) if value % 100 else '')


def quantity_answer(text, expected, unit):
    """Require one stated amount, accepting exact digits or scoped English words."""
    words = '|'.join(sorted(set(SMALL + TENS + ['hundred', 'and']), key=len, reverse=True))
    word_number = rf'(?:{words})(?:[ -]+(?:{words}))*'
    pattern = rf'(?<![\w.+-])([+-]?\d+(?:\.\d+)?|{word_number})\s+{re.escape(unit)}\b'
    amounts = re.findall(pattern, text.casefold())
    if len(amounts) != 1:
        return False
    value = amounts[0]
    if re.fullmatch(r'[+-]?\d+(?:\.\d+)?', value):
        return Decimal(value) == expected
    normalized = ' '.join(word for word in value.replace('-', ' ').split() if word != 'and')
    return normalized == english_integer(expected)


def messages(row, *, guidance=None):
    from nanoai.agent import DEFAULT_IDENTITY
    from nanoai.harmonizer import context_messages
    sources = [{"id": f"observation-{index + 1}", "role": "supplied-observation", "text": text}
               for index, text in enumerate(row["observations"])]
    if guidance and row["kind"] == "review":
        # This is already rendered teaching text, with explicitly separate
        # worked examples. It has no journal revision or acquired authority.
        sources.append({"id": "teaching-guidance", "role": "supplied-procedure", "text": guidance})
    prepared = {"identity": DEFAULT_IDENTITY, "task": {"goal": row["question"]},
                "current_source_records": sources}
    return context_messages(prepared, output="answer")


def training_input(row, pair=None):
    """Keep an observed correction attached to its actual decision prompt."""
    if row["split"] not in {"train", "replay", "validation"}:
        raise ValueError("Confirmation and control cases stay outside training inputs")
    if pair is not None:
        if row["split"] not in {"train", "replay"}:
            raise ValueError("Observed corrections require a teaching or replay split")
        if pair["target"] != row["target"]:
            raise ValueError("Observed correction target differs from the qualified data")
        prompt = deepcopy(pair["messages"])
    else:
        prompt = messages(row)
    return {"messages": prompt, "target": row["target"], "reasoning": row["teaching_reason"]}


def guidance(lessons, train):
    selected = {"evidence", "correction", "knowledge", "causality", "quantity"}
    rules = [r["principle"] for r in lessons if r["id"] in selected]
    # Both examples are fixed before test evaluation. No family label, outcome
    # or test answer selects an in-context teaching example.
    examples = [next(r for r in train if r["id"] == name)
                for name in ("train-correction-1", "train-evidence-1")]
    return ("Apply these teaching principles: " + " ".join(rules) + "\nSeparate worked examples:\n" +
            "\n".join("\n".join(r["observations"]) + "\nResponse: " + r["target"] for r in examples))


def assess(text, criteria):
    if criteria["kind"] == "plain":
        value = text.strip().casefold().removesuffix(".")
        return {"exact_task_answer": value in criteria["accepted"]}
    try:
        answer = strict_json(text)
        valid = (isinstance(answer, dict) and set(answer) == {"decision", "evidence", "text"}
                 and all(isinstance(v, str) for v in answer.values())
                 and answer["decision"] in {"keep", "revise"})
    except (ValueError, TypeError):
        valid = False
    if not valid:
        return {"review_contract": False}
    result = {"review_contract": True, "decision": answer["decision"] == criteria["decision"],
              "source_quotation": bool(answer["evidence"].strip()) and answer["evidence"] in criteria["source"],
              "word_limit": 1 <= len(re.findall(r"\b[\w'-]+\b", answer["text"])) <= 24}
    value = answer["text"].casefold()
    if criteria["decision"] == "keep":
        result["supported_draft_preserved"] = answer["text"] == criteria["draft"]
    else:
        result["required_details"] = all(any(v.casefold() in value for v in group)
                                          for group in criteria["required"])
        result["unsupported_details_absent"] = not any(v.casefold() in value for v in criteria["forbidden"])
        result["effective_change"] = answer["text"] != criteria["draft"]
        if "quantity" in criteria:
            quantity = criteria["quantity"]
            expected = quantity["period"] * (quantity["count"] if quantity["mode"] == "consecutive" else 1)
            result["computed_quantity"] = quantity_answer(value, expected, quantity["unit"])
    return result


def qualify(corpus):
    ids, inputs = set(), set()
    rows = corpus["rows"]
    for row in rows:
        if row["id"] in ids or tuple(row["observations"]) in inputs:
            raise ValueError("Duplicated task identity or complete input")
        ids.add(row["id"]); inputs.add(tuple(row["observations"]))
        if row["split"] not in {"train", "validation", "confirmation", "replay", "control"}:
            raise ValueError("Task split differs")
        if corpus["schema"] == "nanoai.teaching-tasks.v2" and row["split"] in {"train", "validation", "replay"}:
            reason = row.get("teaching_reason")
            if (not isinstance(reason, str) or not reason.strip() or len(reason.encode()) > 4096
                    or "<think>" in reason or "</think>" in reason or not row.get("reason_provenance")):
                raise ValueError("Attributed teaching rationale is unavailable")
        checks = assess(row["target"], row["criteria"])
        if not all(checks.values()):
            raise ValueError("Teaching target fails its rubric: " + row["id"] + " " + str(checks))
        if row["kind"] == "review":
            if row["observations"][:2] != ["SOURCE: " + row["criteria"]["source"], "DRAFT: " + row["criteria"]["draft"]]:
                raise ValueError("Rubric differs from the observed source and draft")
            bad = json.loads(row["target"])
            bad["decision"] = "keep" if bad["decision"] == "revise" else "revise"
            if all(assess(json.dumps(bad), row["criteria"]).values()):
                raise ValueError("Wrong decision was accepted")
    return {"rows": len(rows), "split_counts": {s: sum(r["split"] == s for r in rows)
            for s in ("train", "validation", "confirmation", "replay", "control")},
            "targets_pass": True, "wrong_decisions_rejected": True, "distinct_inputs": len(inputs),
            "scope": "Authored short repair contract; overlapping rule families, new inputs and referents, no broad generalization claim"}


def evaluation_rows(corpus, *, review_cases=None, expected_review=None):
    """Attach a precommitted independent test set only at evaluation time."""
    rows = [{**row, "cohort": "authored"} for row in corpus["rows"]
            if row["split"] in {"confirmation", "control"}]
    if expected_review is None:
        if review_cases is not None:
            raise ValueError("Independent review cases need a frozen source identity")
        return rows
    if review_cases is None or file_digest(review_cases) != expected_review["digest"]:
        raise ValueError("Independent review case source differs from the frozen protocol")
    review = json.loads(Path(review_cases).read_text())
    if (review.get("schema") != "nanoai.teaching-tasks.v2"
            or len(review["rows"]) != expected_review["rows"]
            or any(row["split"] != "confirmation" for row in review["rows"])):
        raise ValueError("Independent review cases must be the declared confirmation set")
    qualify(review)
    combined = {"schema": corpus["schema"], "rows": corpus["rows"] + review["rows"]}
    qualify(combined)  # Reject reused identities or inputs across development and review.
    return rows + [{**row, "cohort": "independent"} for row in review["rows"]]
