"""Finite compositional semantic checks over the existing relation owner.

Canonical English is an explicit checking grammar. Natural-language meaning is
proposed by the learned core; this module does not certify arbitrary prose.
"""
from __future__ import annotations

from itertools import product

from .memory import bounded_text, identifier
from .relations import MAX_VARIABLES, evaluate, expression_variables


PHRASES = {"and": "both", "or": "at least one of", "xor": "exactly one of", "eq": "equivalent"}


def context(symbols):
    if not isinstance(symbols, dict) or not 1 <= len(symbols) <= MAX_VARIABLES:
        raise ValueError("Finite meaning context extent")
    for name, phrase in symbols.items():
        identifier(name)
        bounded_text(phrase, "English condition", 256)
        if any(c in phrase for c in "()\n\r") or phrase.strip() != phrase:
            raise ValueError("A condition needs an unambiguous leaf phrase")
        if phrase in {"true", "false"} or phrase.startswith(tuple(PHRASES.values())+("not ",)):
            raise ValueError("A condition collides with the checking grammar")
    if len(set(symbols.values())) != len(symbols):
        raise ValueError("Condition phrases need distinct referents")
    return symbols


def validate(expression, symbols):
    context(symbols)
    if expression_variables(expression) - set(symbols):
        raise ValueError("Meaning refers to an unavailable condition")
    return expression


def render(expression, symbols):
    if isinstance(expression, dict) and set(expression) == {"alternatives"}:
        readings = interpretations(expression, symbols)
        return "readings (" + ") or (".join(render(e, symbols) for e in readings) + ")"
    validate(expression, symbols)
    def visit(node):
        if type(node) is int:
            return "true" if node else "false"
        if isinstance(node, str):
            return symbols[node]
        op, values = next(iter(node.items()))
        if op == "not":
            return "not (" + visit(values) + ")"
        return PHRASES[op]+" ("+visit(values[0])+") and ("+visit(values[1])+")"
    return visit(expression)


def parse_checked_english(text, symbols):
    """Parse only the published complete grammar; reject unconsumed prose."""
    context(symbols)
    bounded_text(text, "checked English", 8192)
    phrases = sorted(((v, k) for k, v in symbols.items()), key=lambda p: -len(p[0]))
    def visit(offset, depth):
        if depth > 16:
            raise ValueError("Checked English depth")
        for op, lead in [("not", "not"), *PHRASES.items()]:
            prefix = lead+" ("
            if text.startswith(prefix, offset):
                left, end = visit(offset+len(prefix), depth+1)
                if not text.startswith(")", end):
                    raise ValueError("Unclosed English operand")
                end += 1
                if op == "not":
                    return {"not": left}, end
                if not text.startswith(" and (", end):
                    raise ValueError("Missing second English operand")
                right, end = visit(end+6, depth+1)
                if not text.startswith(")", end):
                    raise ValueError("Unclosed English operand")
                return {op: [left, right]}, end+1
        for phrase, name in [("true", 1), ("false", 0), *phrases]:
            if text.startswith(phrase, offset):
                end = offset+len(phrase)
                if end == len(text) or text[end] == ")":
                    return name, end
        raise ValueError("English is outside the declared checking grammar")
    if text.startswith("readings ("):
        values=[];offset=10
        while True:
            value,end=visit(offset,0)
            if not text.startswith(")",end):raise ValueError("Unclosed interpretation")
            values.append(value);end+=1
            if end==len(text):break
            if not text.startswith(" or (",end):raise ValueError("Interpretation separator")
            offset=end+5
        expression={"alternatives":values}
        interpretations(expression,symbols)
        return expression
    value, end = visit(0, 0)
    if end != len(text):
        raise ValueError("Unconsumed English changes the meaning contract")
    return validate(value, symbols)


def assignments(symbols):
    context(symbols)
    names = sorted(symbols)
    return [dict(zip(names, row)) for row in product((0, 1), repeat=len(names))]


def denotation(expression, symbols):
    validate(expression, symbols)
    return tuple(evaluate(expression, row) for row in assignments(symbols))


def interpretations(expression, symbols):
    readings=(expression["alternatives"] if isinstance(expression,dict)
              and set(expression)=={"alternatives"} else [expression])
    if not isinstance(readings,list) or not 1<=len(readings)<=4:
        raise ValueError("Explicit interpretation extent")
    for reading in readings:validate(reading,symbols)
    return readings


def meaning_functions(expression, symbols):
    return {denotation(reading,symbols) for reading in interpretations(expression,symbols)}


def separate_meanings(first, second, symbols):
    """Keep interpretation identity across a sequence of discriminating tests.

    Per-test possible values can agree while the joint meaning differs. Choose
    a trace admitted by one set of functions and excluded by the other.
    """
    left,right=meaning_functions(first,symbols),meaning_functions(second,symbols)
    if left==right:return None
    side='first' if left-right else 'second'
    chosen=sorted(left-right if side=='first' else right-left)[0]
    others=right if side=='first' else left
    indices=sorted({next(i for i,(a,b) in enumerate(zip(chosen,other)) if a!=b) for other in others})
    worlds=assignments(symbols)
    return {'assignments':[worlds[i] for i in indices],
            'first_traces':sorted({tuple(row[i] for i in indices) for row in left}),
            'second_traces':sorted({tuple(row[i] for i in indices) for row in right}),
            'exclusive_side':side,'exclusive_trace':[chosen[i] for i in indices]}


def separate(first, second, symbols):
    """Return an actual finite counterexample, or exact scoped equivalence."""
    validate(first, symbols); validate(second, symbols)
    for assignment in assignments(symbols):
        left, right = evaluate(first, assignment), evaluate(second, assignment)
        if left != right:
            return {"assignment": assignment, "first": left, "second": right}
    return None


def assess(text, criteria):
    """Keep full-domain denotation separate from coincidental current output."""
    from .steps import strict_json
    symbols = criteria["symbols"]
    expected = criteria["expression"]
    try:
        predicted = (strict_json(text) if criteria["direction"] == "english_to_math"
                     else parse_checked_english(text.strip().removesuffix("."), symbols))
        readings=interpretations(predicted, symbols)
    except (ValueError, KeyError, TypeError):
        return {"supported_form": False, "same_denotation": False, "current_execution": False}
    return {"supported_form": True,
            "same_denotation": meaning_functions(predicted,symbols)==meaning_functions(expected,symbols),
            "current_execution": {evaluate(e,criteria["assignment"]) for e in readings}
               == {evaluate(e,criteria["assignment"]) for e in interpretations(expected,symbols)}}
