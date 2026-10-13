"""Bounded joint alternatives for the source-bound Steps graph.

Queries retain shared variable identity. Their verdict describes exactly the
declared finite model; completeness and relevance need their own assessment.
Each operation is pure, including the explicitly hypothetical state map.
"""
from copy import deepcopy

from .memory import bounded_text, identifier


MAX_VARIABLES = 12
MAX_WORLDS = 4096
MAX_EXPRESSION_NODES = 128
MAX_EXPRESSION_DEPTH = 16
RELATION = "finite_relation_v1"
OPERATORS = {"relation.decode", "relation.query", "relation.refine",
             "relation.extend", "relation.map"}


def expression_variables(expression):
    remaining = MAX_EXPRESSION_NODES
    variables = set()

    def visit(expr, depth):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > MAX_EXPRESSION_DEPTH:
            raise ValueError("Relation expression exceeds the declared bound")
        if type(expr) is int and expr in (0, 1):
            return
        if isinstance(expr, str):
            identifier(expr)
            variables.add(expr)
            return
        if not isinstance(expr, dict) or len(expr) != 1:
            raise ValueError("Relation expression needs a bit, variable or named operation")
        operator, value = next(iter(expr.items()))
        if operator == "not":
            visit(value, depth + 1)
        elif operator in {"and", "or", "xor", "eq"}:
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError("A relation junction needs two operands")
            for child in value:
                visit(child, depth + 1)
        else:
            raise ValueError("Relation expression operator is unavailable")

    visit(expression, 0)
    return variables


def evaluate(expression, assignment):
    """Evaluate an already admitted expression in one shared assignment."""
    if type(expression) is int:
        return expression
    if isinstance(expression, str):
        return assignment[expression]
    operator, operands = next(iter(expression.items()))
    if operator == "not":
        return 1 - evaluate(operands, assignment)
    a, b = (evaluate(x, assignment) for x in operands)
    return {"and": a & b, "or": a | b, "xor": a ^ b, "eq": int(a == b)}[operator]


def validate_arguments(operator, arguments):
    fields = {"relation.decode": {"step"}, "relation.query": {"step", "expression"},
              "relation.refine": {"step", "expression", "expected"},
              "relation.extend": {"step", "variable", "expression"},
              "relation.map": {"step", "variable", "expression"}}
    if (operator not in fields or not isinstance(arguments, dict)
            or set(arguments) != fields[operator] or not isinstance(arguments["step"], str)):
        raise ValueError("Relation operation fields differ from the contract")
    if "variable" in arguments:
        identifier(arguments["variable"])
    if "expression" in arguments:
        if not (operator == "relation.extend" and arguments["expression"] is None):
            expression_variables(arguments["expression"])
    if operator == "relation.refine" and (type(arguments["expected"]) is not int
                                          or arguments["expected"] not in (0, 1)):
        raise ValueError("Relation refinement needs an observed or assumed bit")
    return [arguments["step"]]


def validate_model(model):
    if (not isinstance(model, dict) or set(model) != {"type", "variables", "worlds", "scope"}
            or model["type"] != RELATION):
        raise ValueError("A finite relation needs its typed model and scope")
    variables, worlds = model["variables"], model["worlds"]
    if not isinstance(variables, list) or not 1 <= len(variables) <= MAX_VARIABLES:
        raise ValueError("Relation variable extent")
    for name in variables:
        identifier(name)
    if len(set(variables)) != len(variables):
        raise ValueError("Relation variable identities must be distinct")
    bounded_text(model["scope"], "relation scope", 2048)
    if not isinstance(worlds, list) or len(worlds) > MAX_WORLDS:
        raise ValueError("Relation world extent")
    for row in worlds:
        if (not isinstance(row, list) or len(row) != len(variables)
                or any(type(bit) is not int or bit not in (0, 1) for bit in row)):
            raise ValueError("Relation row needs one exact bit per variable")
    if len({tuple(row) for row in worlds}) != len(worlds):
        raise ValueError("Relation world identities must be distinct")
    return model


def calculate(operator, arguments, parent):
    validate_arguments(operator, arguments)
    if operator == "relation.decode":
        from .steps import strict_json
        if not isinstance(parent.get("value"), str):
            raise ValueError("Relation decoding needs exact recorded JSON text")
        model = validate_model(strict_json(parent["value"]))
        model = deepcopy(model)
        model["worlds"].sort()
        return {"value": model, "sources": parent["sources"],
                "scope": "Exact declared finite model; external completeness remains assessed separately"}

    model = validate_model(parent.get("value"))
    expr = arguments["expression"]
    if expr is not None and expression_variables(expr) - set(model["variables"]):
        raise ValueError("Relation question references an unavailable variable")
    assignments = [dict(zip(model["variables"], row)) for row in model["worlds"]]
    values = [evaluate(expr, row) for row in assignments] if expr is not None else []
    if operator == "relation.query":
        witnesses = {}
        for row, answer in zip(model["worlds"], values):
            witnesses.setdefault(str(answer), row)
        possible = sorted(set(values))
        verdict = {(0,): "no", (1,): "yes", (0, 1): "maybe", (): None}[tuple(possible)]
        answer = {"type": "relation_answer_v1", "possible_values": possible,
                  "verdict": verdict, "state": "both" if len(possible) == 2 else "decided" if possible else "conflict",
                  "variables": model["variables"], "witnesses": witnesses,
                  "question": expr, "model_scope": model["scope"]}
        return {"value": deepcopy(answer), "sources": parent["sources"],
                "scope": "Exact image of the question over the declared joint alternatives"}

    updated = deepcopy(model)
    if operator == "relation.refine":
        updated["worlds"] = [row for row, answer in zip(model["worlds"], values)
                             if answer == arguments["expected"]]
        scope = "Conditional refinement; the supplied bit is an explicit premise"
    elif operator == "relation.extend":
        variable = arguments["variable"]
        if variable in model["variables"]:
            raise ValueError("Relation extension needs a fresh variable identity")
        if len(model["variables"]) >= MAX_VARIABLES or len(model["worlds"]) * (2 if expr is None else 1) > MAX_WORLDS:
            raise ValueError("Relation extension exceeds the declared bound")
        updated["variables"].append(variable)
        updated["worlds"] = ([row + [bit] for row in model["worlds"] for bit in (0, 1)] if expr is None
                             else [row + [bit] for row, bit in zip(model["worlds"], values)])
        scope = "Conservative extension; projection preserves every previous joint alternative"
    else:
        variable = arguments["variable"]
        if variable not in model["variables"]:
            raise ValueError("Relation map needs an existing variable identity")
        index = model["variables"].index(variable)
        for row, value in zip(updated["worlds"], values):
            row[index] = value
        updated["worlds"] = [list(row) for row in sorted({tuple(row) for row in updated["worlds"]})]
        scope = "Hypothetical state transformation; no external action occurred"
    return {"value": validate_model(updated), "sources": parent["sources"], "scope": scope}
