"""Typed, source-bound pure operations on nano'ai's existing SJQL journal.

The learned core chooses work. These operators observe sources and calculate
declared integer results. A result's exactness does not certify the relevance
of the chosen work or grant permission for external effects.
"""
from __future__ import annotations

from copy import deepcopy
import json
import time
import uuid

from .core import digest_bytes, file_digest
from .memory import Memory, StaleState, bounded_text, canonical, identifier


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Ambiguous duplicate source field")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Non-finite source value")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def quantity_value(value, sources):
    if (not isinstance(value, dict) or set(value) != {"value", "unit"}
            or type(value["value"]) is not int or abs(value["value"]) > 10**12
            or not isinstance(value["unit"], str) or value["unit"] not in {"cm", "s", "count", "scalar"}):
        raise ValueError("A quantity field needs a bounded integer value and a declared cm, s, count or scalar unit")
    unit = value["unit"]
    return {"value": value["value"], "dimensions": {} if unit in {"count", "scalar"} else {unit: 1},
            "declared_unit": unit, "sources": sources, "scope": "Exact declared source quantity"}


def quantity_catalogue(sources):
    """Bind compact decision references to exact, eligible source fields."""
    fields = []
    def walk(value, path, source):
        if not isinstance(value, dict) or len(path) > 8:
            return
        if set(value) == {"value", "unit"} and path:
            parsed = quantity_value(value, [f"{source['id']}@{source['revision']}"])
            if len(fields) >= 64:
                raise ValueError("Quantity catalogue exceeds the decision bound")
            fields.append({"ref": "q" + str(len(fields)), "source": source["id"],
                "revision": source["revision"], "source_digest": source["digest"], "path": path,
                "value": parsed["value"], "unit": parsed["declared_unit"]})
            return
        for key in sorted(value):
            bounded_text(key, "JSON field", 256)
            walk(value[key], path + [key], source)
    for source in sources:
        try:
            data = strict_json(source["text"])
        except ValueError:
            continue  # Unstructured evidence supplies no executable quantity.
        walk(data, [], source)
    return fields


class Steps:
    def __init__(self, memory: Memory):
        self.memory = memory
        from . import relations
        self.implementation = digest_bytes(canonical({"steps": file_digest(__file__),
                                                       "relations": file_digest(relations.__file__)}))

    @staticmethod
    def parents(operator, arguments):
        if not isinstance(arguments, dict):
            raise ValueError("Step arguments must be an object")
        from .relations import OPERATORS, validate_arguments
        if operator in OPERATORS:
            return validate_arguments(operator, arguments)
        if operator == "source.read":
            if set(arguments) != {"source"}:
                raise ValueError("source.read arguments")
            identifier(arguments["source"])
            return []
        if operator in {"json.field", "quantity.field"}:
            if set(arguments) != {"step", "path"} or not isinstance(arguments["step"], str):
                raise ValueError("json.field arguments")
            path = arguments["path"]
            if not isinstance(path, list) or not 1 <= len(path) <= 8:
                raise ValueError("JSON path extent")
            for part in path:
                bounded_text(part, "JSON path key", 256)
            return [arguments["step"]]
        if operator in {"integer.product", "integer.sum", "quantity.product", "quantity.sum"}:
            ids = arguments.get("steps")
            if set(arguments) != {"steps"} or not isinstance(ids, list) or not 1 <= len(ids) <= 16 or any(not isinstance(x, str) for x in ids):
                raise ValueError("Integer operation arguments")
            return ids
        raise ValueError("The operator has no admitted pure executor")

    def propose(self, snapshot, operator, arguments):
        parents = self.parents(operator, arguments)
        memory = self.memory
        with memory.locked():
            memory._validate_snapshot(snapshot)
            if operator == "source.read" and arguments["source"] not in snapshot["sources"]:
                raise ValueError("Source is outside the task evidence selection")
            prepared = deepcopy(snapshot)
            prepared.setdefault("observations", {})
            for parent in parents:
                observation = memory.state["step_observations"].get(parent)
                if observation is None:
                    raise ValueError("A parent operation must have an observed result")
                prepared["observations"][parent] = observation["digest"]
            memory._validate_snapshot(prepared)
            step = {"id": str(uuid.uuid4()), "operator": operator,
                    "arguments": deepcopy(arguments), "snapshot": prepared,
                    "implementation": self.implementation, "execution": "pure observation",
                    "proposer_identity": snapshot["core_identity"]}
            memory._append({"kind": "step.proposed", "step": step})
            return deepcopy(step)

    def _resolved(self, step):
        memory = self.memory
        if step["operator"] == "source.read":
            name = step["arguments"]["source"]
            row = memory.state["sources"][name]
            return {"source": {"id": name, "revision": row["revision"], "digest": row["digest"]}}, []
        ids = self.parents(step["operator"], step["arguments"])
        observations = [memory.state["step_observations"][sid] for sid in ids]
        normalized = {"inputs": [row["digest"] for row in observations]}
        if step["operator"] in {"json.field", "quantity.field"}:
            normalized["path"] = step["arguments"]["path"]
        if step["operator"].startswith("relation."):
            normalized["parameters"] = {k: deepcopy(v) for k, v in step["arguments"].items() if k != "step"}
        return normalized, observations

    def _calculate(self, step, inputs):
        operator, args = step["operator"], step["arguments"]
        if operator == "source.read":
            row = self.memory.state["sources"][args["source"]]
            return {"value": row["text"], "sources": [f"{row['id']}@{row['revision']}"],
                    "scope": "Exact recorded source bytes"}
        refs = sorted({ref for row in inputs for ref in row["result"]["sources"]})
        if operator.startswith("relation."):
            from .relations import calculate
            return calculate(operator, args, inputs[0]["result"])
        if operator in {"json.field", "quantity.field"}:
            text = inputs[0]["result"]["value"]
            if not isinstance(text, str):
                raise ValueError("JSON field input must be recorded text")
            value = strict_json(text)
            for key in args["path"]:
                if not isinstance(value, dict) or key not in value:
                    raise ValueError("Source JSON field is unavailable")
                value = value[key]
            if operator == "quantity.field":
                return quantity_value(value, refs)
            if isinstance(value, (dict, list)) or isinstance(value, float):
                raise ValueError("The current field contract admits exact scalar values")
            return {"value": value, "sources": refs, "scope": "Exact selected JSON field"}
        values = [row["result"]["value"] for row in inputs]
        if any(type(value) is not int or abs(value) > 10**12 for value in values):
            raise ValueError("Integer input extent or type")
        dimensions = None
        if operator.startswith("quantity."):
            units = [row["result"].get("dimensions") for row in inputs]
            if any(not isinstance(unit, dict) or any(k not in {"cm", "s"} or type(v) is not int or abs(v) > 16 for k, v in unit.items()) for unit in units):
                raise ValueError("Quantity operations require dimensioned inputs")
            if operator == "quantity.sum":
                if any(unit != units[0] for unit in units[1:]):
                    raise ValueError("Addition requires equal units; a count cannot be added to a length")
                dimensions = units[0].copy()
            else:
                dimensions = {}
                for unit in units:
                    for key, exponent in unit.items():
                        dimensions[key] = dimensions.get(key, 0) + exponent
                dimensions = {key: exponent for key, exponent in dimensions.items() if exponent}
                if any(abs(exponent) > 16 for exponent in dimensions.values()):
                    raise ValueError("Result unit exponent exceeds the declared extent")
        product = operator.endswith(".product")
        value = 1 if product else 0
        for item in values:
            value = value * item if product else value + item
            if abs(value) >= 2**127:
                raise ValueError("Integer result exceeds the declared extent")
        result = {"value": value, "sources": refs, "scope": "Exact declared integer operation"}
        if dimensions is not None:
            result.update(dimensions=dimensions, scope="Exact declared operation with dimensional consistency")
        return result

    def execute(self, step_id):
        start = time.perf_counter_ns()
        memory = self.memory
        with memory.locked():
            step = memory.state["steps"][step_id]
            if step["implementation"] != self.implementation:
                raise StaleState("Pure executor version changed")
            memory._validate_snapshot(step["snapshot"])
            review = memory.state["step_reviews"].get(step_id)
            if review and review["decision"] in {"rejected", "cancelled"}:
                raise StaleState("The proposed operation was rejected or cancelled")
            existing = memory.state["step_observations"].get(step_id)
            if existing:
                if digest_bytes(canonical(existing["result"])) != existing["digest"]:
                    raise ValueError("Recorded result identity differs")
                return {**deepcopy(existing), "reobserved": True}
            normalized, inputs = self._resolved(step)
            key = digest_bytes(canonical({"operator": step["operator"],
                "implementation": self.implementation, "arguments": normalized}))
            cached_id = memory.state["step_cache"].get(key)
            cached = memory.state["step_observations"].get(cached_id)
            if cached:
                # Reuse the pure value only. The new proposal has already checked
                # its own task, source selection, parent chain and permissions.
                if digest_bytes(canonical(cached["result"])) != cached["digest"]:
                    raise ValueError("Cached pure result identity differs")
                value = deepcopy(cached["result"])
            else:
                value = self._calculate(step, inputs)
            result = {"step_id": step_id, "cache_key": key, "result": value,
                      "digest": digest_bytes(canonical(value)), "cache_hit": cached is not None,
                      "reused_from": cached_id, "elapsed_ns": time.perf_counter_ns() - start,
                      "semantic_review": "unassessed", "effect": "none"}
            memory._append({"kind": "step.observed", "observation": result})
            return deepcopy(result)

    def review(self, step_id, decision, *, reviewer, reason, replacement=None):
        if decision not in {"confirmed", "rejected", "cancelled"}:
            raise ValueError("Step review decision")
        bounded_text(reviewer, "reviewer", 256)
        bounded_text(reason, "review reason", 2048)
        memory = self.memory
        with memory.locked():
            step = memory.state["steps"][step_id]
            if step_id in memory.state["step_reviews"]:
                raise ValueError("A reviewed step requires a new proposal")
            if decision == "confirmed":
                memory._validate_snapshot(step["snapshot"])
                if step_id not in memory.state["step_observations"]:
                    raise ValueError("Confirmation requires an observed result")
            if replacement is not None:
                other = memory.state["steps"].get(replacement)
                if (decision != "rejected" or replacement == step_id or not other
                        or other["snapshot"]["task_id"] != step["snapshot"]["task_id"]):
                    raise ValueError("Replacement step contract")
            review = {"step_id": step_id, "decision": decision, "reviewer": reviewer,
                      "reason": reason, "replacement": replacement,
                      "scope": "Recorded reviewer decision; grants no external effects"}
            memory._append({"kind": "step.reviewed", "review": review})
            return deepcopy(review)

    def status(self):
        self.memory.reload()
        rows = []
        for sid, step in self.memory.state["steps"].items():
            try:
                self.memory._validate_snapshot(step["snapshot"])
                current = step["implementation"] == self.implementation
            except ValueError:
                current = False
            observation = self.memory.state["step_observations"].get(sid)
            review = self.memory.state["step_reviews"].get(sid)
            rows.append({"id": sid, "operator": step["operator"], "current": current,
                "state": review["decision"] if review else "observed provisional" if observation else "proposed",
                "cache_hit": observation["cache_hit"] if observation else None})
        return {"steps": rows, "recovery": "Observe journaled results; pending proposals await an explicit decision"}

    def calculate_fields(self, snapshot, plan):
        """Compile a learned typed choice into source reads and a small DAG."""
        if not isinstance(plan, dict) or set(plan) != {"operator", "inputs"}:
            raise ValueError("Calculation plan contract")
        if not isinstance(plan["operator"], str) or plan["operator"] not in {"integer.product", "integer.sum", "quantity.product", "quantity.sum"}:
            raise ValueError("Calculation operator")
        items = plan["inputs"]
        if not isinstance(items, list) or not 1 <= len(items) <= 8:
            raise ValueError("Calculation input extent")
        field_operator = "quantity.field" if plan["operator"].startswith("quantity.") else "json.field"
        # Admit every field before any proposed operation is journaled.
        for item in items:
            if not isinstance(item, dict) or set(item) != {"source", "path"}:
                raise ValueError("Calculation source field")
            self.parents("source.read", {"source": item["source"]})
            self.parents(field_operator, {"step": "placeholder", "path": item["path"]})
            if item["source"] not in snapshot["sources"]:
                raise ValueError("Calculation source is outside the task evidence selection")
        def perform(operator, arguments):
            step = self.propose(snapshot, operator, arguments)
            try:
                observation = self.execute(step["id"])
            except ValueError as error:
                self.review(step["id"], "rejected", reviewer="pure-operator-contract-v1", reason=str(error))
                raise
            return step, observation

        reads, fields, observations = {}, [], []
        for item in items:
            if item["source"] not in reads:
                step, observed = perform("source.read", {"source": item["source"]})
                observations.append(observed)
                reads[item["source"]] = step["id"]
            step, observed = perform(field_operator, {"step": reads[item["source"]], "path": item["path"]})
            observations.append(observed); fields.append(step["id"])
        step, result = perform(plan["operator"], {"steps": fields})
        observations.append(result)
        return {"plan": deepcopy(plan), "observations": observations,
                "step_id": step["id"], "result": result["result"]}
