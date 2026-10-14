"""Typed, source-bound pure operations on nano'ai's existing SJQL journal.

The learned core chooses work. These operators observe sources and calculate
declared integer results. A result's exactness does not certify the relevance
of the chosen work or grant permission for external effects.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
import math
import time
import uuid

from .core import digest_bytes, file_digest
from .memory import Memory, StaleState, bounded_text, canonical, identifier

OPERATOR_VOCABULARY_VERSION = "nanoai.admitted-operators.v1"
ADMITTED_OPERATORS_V1 = (
    "source.read",
    "json.field",
    "integer.product",
    "integer.sum",
    "quantity.field",
    "quantity.product",
    "quantity.sum",
    "relation.decode",
    "relation.query",
    "relation.refine",
    "relation.extend",
    "relation.map",
    "relation.project",
    "relation.revise",
    "relation.curiosity",
    "scratchboard.add",
    "scratchboard.reserve",
    "scratchboard.bind",
    "scratchboard.bind_many",
    "scratchboard.generator",
    "scratchboard.revise",
    "scratchboard.refine",
    "judgment.predict",
    "judgment.assess",
    "judgment.combine",
)


def _to_rational(value) -> Fraction:
    """Convert a Fraction, int, str or finite decimal float (via Fraction(str(value))) into an exact Fraction."""
    if isinstance(value, Fraction):
        return value
    if type(value) is int or isinstance(value, str):
        return Fraction(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Rational parameter must be finite")
        return Fraction(str(value))
    raise ValueError("Rational parameter must be a Fraction, int, str or finite float")


def mass_at_or_below(c: int, r: int, t=Fraction(4, 5), prior_yes: int = 0, prior_no: int = 0) -> Fraction:
    """Exact Fraction P(p <= t) for the Beta(1 + c + prior_yes, 1 + r + prior_no) posterior."""
    if any(type(v) is not int or v < 0 for v in (c, r, prior_yes, prior_no)):
        raise ValueError("Counts and prior pseudo-counts are non-negative integers")
    t_frac = _to_rational(t)
    if not 0 < t_frac < 1:
        raise ValueError("Threshold lies strictly between 0 and 1")
    a, b = 1 + c + prior_yes, 1 + r + prior_no
    n, num, den = a + b - 1, t_frac.numerator, t_frac.denominator
    tail = sum(math.comb(n, j) * (num ** j) * ((den - num) ** (n - j)) for j in range(a, n + 1))
    return Fraction(tail, den ** n)


def band_bracket(c: int, r: int, threshold=Fraction(4, 5), alpha=Fraction(1, 20), prior_yes: int = 0, prior_no: int = 0) -> str:
    """yes (act), no (escalate) or both (propose), exact under the declared model with equality convention mass <= alpha."""
    alpha_frac = _to_rational(alpha)
    if not 0 < alpha_frac < Fraction(1, 2):
        raise ValueError("alpha lies strictly between 0 and 1/2")
    below = mass_at_or_below(c, r, threshold, prior_yes=prior_yes, prior_no=prior_no)
    if below <= alpha_frac:
        return "yes"
    if 1 - below <= alpha_frac:
        return "no"
    return "both"


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
    OPERATOR_VOCABULARY_VERSION = OPERATOR_VOCABULARY_VERSION
    ADMITTED_OPERATORS_V1 = ADMITTED_OPERATORS_V1

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

    def propose(self, snapshot, operator, arguments, *, core_signal_provenance=None, caller_supplied_signal=None, decision_record=None):
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
            if core_signal_provenance is not None:
                step["core_signal_provenance"] = str(core_signal_provenance)
            if caller_supplied_signal is not None:
                step["caller_supplied_signal"] = bool(caller_supplied_signal)
            if decision_record is not None:
                step["decision_record"] = deepcopy(decision_record)
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
            normalized["parameters"] = {k: deepcopy(v) for k, v in step["arguments"].items()
                                        if k not in {"step", "replacement"}}
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
            return calculate(operator, args, inputs[0]["result"],
                             [row["result"] for row in inputs[1:]])
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

    def issue_correction(self, step_id, *, reviewer, reason, replacement=None, conduct_virtue=None):
        """Record an audit or master correction event and reject the flawed step if unreviewed."""
        bounded_text(reviewer, "reviewer", 256)
        bounded_text(reason, "correction reason", 2048)
        allowed_virtues = {
            "Gi", "Yu", "Jin", "Rei", "Makoto", "Meiyo", "Chugi",
            "epistemic_humility", "accuracy", "scope_honesty", "provenance_integrity", "restraint",
        }
        if conduct_virtue is not None and conduct_virtue not in allowed_virtues:
            raise ValueError("Conduct virtue must be an admitted conduct or epistemic virtue")
        memory = self.memory
        with memory.locked():
            step = memory.state["steps"].get(step_id)
            if step is None:
                raise ValueError("Correction target step is unavailable")
            if replacement is not None:
                other = memory.state["steps"].get(replacement)
                if replacement == step_id or not other or other["snapshot"]["task_id"] != step["snapshot"]["task_id"]:
                    raise ValueError("Correction replacement step contract")
            if step_id not in memory.state["step_reviews"]:
                review = {"step_id": step_id, "decision": "rejected", "reviewer": reviewer,
                          "reason": reason, "replacement": replacement,
                          "scope": "Recorded reviewer decision; grants no external effects"}
                memory._append({"kind": "step.reviewed", "review": review})
            body = {"step_id": step_id, "task_id": step["snapshot"]["task_id"],
                    "operator": step["operator"], "reviewer": reviewer, "reason": reason,
                    "replacement": replacement, "conduct_virtue": conduct_virtue,
                    "scope": "Journaled audit correction; informs System One calibration and replay"}
            correction = {"id": digest_bytes(canonical({**body, "sequence": memory.sequence})), **body}
            memory._append({"kind": "correction.issued", "correction": correction})
            return deepcopy(correction)

    def _resolve_core_signal(self, snapshot, *, core_signal=None, judgment_node_id=None):
        """Extract the learned core's context-sensitive judgment, usefulness and uncertainty."""
        if core_signal is not None and judgment_node_id is not None:
            raise ValueError("Provide at most one of core_signal or judgment_node_id")
        if judgment_node_id is not None:
            node = self.memory.state["scratchboard"].get(judgment_node_id)
            if not node or node["spec"].get("kind") != "judgment":
                raise ValueError("Calibration judgment_node_id must reference a scratchboard judgment node")
            if any(node["snapshot"].get(k) != snapshot.get(k) for k in ("task_id", "task_revision", "identity")):
                raise StaleState("Calibration judgment node belongs to a different or stale task snapshot")
            spec = node["spec"]
            verdict = spec["verdict"]
            uncertainty = spec.get("uncertainty")
            assessed = spec.get("role") == "assessment"
            default_usefulness = 1.0 if (verdict == "yes" and assessed) else None
            return {
                "supplied": True,
                "source": "judgment_node",
                "provenance": "assessed_judgment" if assessed else "provisional_judgment",
                "caller_supplied": False,
                "core_identity": node["snapshot"].get("core_identity"),
                "judgment_node_id": judgment_node_id,
                "verdict": verdict,
                "usefulness": default_usefulness,
                "uncertainty": uncertainty,
                "assessed": assessed,
            }
        if core_signal is not None:
            if not isinstance(core_signal, dict):
                raise ValueError("Learned core signal must be a dictionary")
            has_signal = any(
                core_signal.get(k) is not None
                for k in ("verdict", "usefulness", "confidence", "uncertainty")
            )
            raw_verdict = core_signal.get("verdict")
            if raw_verdict is None:
                verdict = "maybe"
            elif raw_verdict in {"yes", "maybe", "no"}:
                verdict = raw_verdict
            else:
                raise ValueError("Learned core signal verdict must be yes, maybe or no")
            usefulness = core_signal.get("usefulness", core_signal.get("confidence"))
            if usefulness is not None:
                if type(usefulness) not in (int, float) or not 0.0 <= float(usefulness) <= 1.0:
                    raise ValueError("Learned core usefulness must lie in [0.0, 1.0]")
                usefulness = round(float(usefulness), 6)
            uncertainty = core_signal.get("uncertainty")
            if uncertainty is not None and not isinstance(uncertainty, str):
                raise ValueError("Learned core uncertainty must be a string or None")
            if uncertainty == "none":
                uncertainty = None
            from_plan = core_signal.get("source") == "core_generation_plan"
            return {
                "supplied": bool(has_signal),
                "source": ("core_generation_plan" if from_plan else "core_estimate") if has_signal else "none",
                "provenance": ("core_generated_plan" if from_plan else "caller_supplied_estimate") if has_signal else "none",
                "caller_supplied": (not from_plan) if has_signal else False,
                "core_identity": core_signal.get("core_identity"),
                "judgment_node_id": None,
                "verdict": verdict,
                "usefulness": usefulness,
                "uncertainty": uncertainty,
                "assessed": bool(core_signal.get("assessed", False)),
            }
        # Inspect whether the active task has an open conflict judgment on the scratchboard
        task_id = snapshot.get("task_id")
        for node in self.memory.state["scratchboard"].values():
            if node["id"] in self.memory.state["scratchboard_revisions"]:
                continue
            if node["snapshot"].get("task_id") == task_id and node["spec"].get("kind") == "judgment":
                if node["spec"].get("verdict") == "maybe" and node["spec"].get("uncertainty") == "conflict":
                    return {
                        "supplied": True,
                        "source": "task_open_conflict",
                        "provenance": "task_open_conflict",
                        "caller_supplied": False,
                        "core_identity": node["snapshot"].get("core_identity"),
                        "judgment_node_id": node["id"],
                        "verdict": "maybe",
                        "usefulness": None,
                        "uncertainty": "conflict",
                        "assessed": False,
                    }
        return {
            "supplied": False,
            "source": "none",
            "provenance": "none",
            "caller_supplied": False,
            "core_identity": snapshot.get("core_identity"),
            "judgment_node_id": None,
            "verdict": None,
            "usefulness": None,
            "uncertainty": None,
            "assessed": False,
        }

    @staticmethod
    def _reg_incomplete_beta(a: int, b: int, x: float) -> float:
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        if x > (a + 1.0) / (a + b + 2.0):
            return 1.0 - Steps._reg_incomplete_beta(b, a, 1.0 - x)
        log_bt = (
            math.lgamma(a + b)
            - math.lgamma(a)
            - math.lgamma(b)
            + a * math.log(x)
            + b * math.log1p(-x)
        )
        bt = math.exp(log_bt) if log_bt > -745.0 else 0.0
        if bt == 0.0:
            return 0.0
        tiny = 1e-30
        qab = a + b
        qap = a + 1.0
        qam = a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        h = d
        for m in range(1, 500):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-15:
                break
        return min(1.0, max(0.0, bt * h / a))

    _to_rational = staticmethod(_to_rational)
    posterior_mass_at_or_below = staticmethod(mass_at_or_below)
    band_bracket = staticmethod(band_bracket)

    @staticmethod
    def beta_quantile(alpha: int, beta: int, q: float) -> float:
        """High-precision quantile of Beta(alpha, beta) for positive integers alpha, beta via closed-form edge cases and log-gamma Lentz continued-fraction bisection."""
        if type(alpha) is not int or type(beta) is not int or alpha < 1 or beta < 1:
            raise ValueError("Beta parameters must be positive integers")
        if not 0.0 < q < 1.0:
            raise ValueError("Quantile q must lie in (0, 1)")
        if alpha == 1 and beta == 1:
            return float(q)
        if beta == 1:
            return math.exp(math.log(q) / alpha)
        if alpha == 1:
            return -math.expm1(math.log1p(-q) / beta)
        if alpha == beta and q == 0.5:
            return 0.5
        lo, hi = 0.0, 1.0
        for _ in range(64):
            mid = (lo + hi) * 0.5
            if Steps._reg_incomplete_beta(alpha, beta, mid) < q:
                lo = mid
            else:
                hi = mid
        return (lo + hi) * 0.5

    def decision_support_audit(self, task_id=None, *, min_confirmations=2, threshold=0.8):
        """Evaluate whether the hybrid system makes well-supported decisions in practice across familiar and unfamiliar tasks."""
        self.memory.reload()
        threshold_float = float(_to_rational(threshold))
        prior_task_conf = {}
        corrected_steps = {c["step_id"] for c in self.memory.state["corrections"].values()
                           if task_id is None or c["task_id"] == task_id}
        decided = 0
        well_supported = 0
        familiar_decided = 0
        familiar_supported = 0
        unfamiliar_decided = 0
        unfamiliar_supported = 0
        replacements = 0
        core_generated_or_assessed = 0
        caller_supplied_signals = 0

        for sid, step in self.memory.state["steps"].items():
            st_task = step["snapshot"]["task_id"]
            op = step["operator"]
            key = (st_task, op)
            was_familiar = prior_task_conf.get(key, 0) >= min_confirmations
            provenance = step.get("core_signal_provenance", "none")
            if task_id is None or st_task == task_id:
                if provenance in {"assessed_judgment", "provisional_judgment", "core_generated_plan", "task_open_conflict"}:
                    core_generated_or_assessed += 1
                elif provenance == "caller_supplied_estimate" or step.get("caller_supplied_signal"):
                    caller_supplied_signals += 1
            rev = self.memory.state["step_reviews"].get(sid)
            if rev and rev["decision"] in {"confirmed", "rejected"}:
                if task_id is None or st_task == task_id:
                    decided += 1
                    is_supported = (
                        rev["decision"] == "confirmed"
                        and rev.get("replacement") is None
                        and sid not in corrected_steps
                    )
                    if rev.get("replacement") is not None:
                        replacements += 1
                    if is_supported:
                        well_supported += 1
                    if was_familiar:
                        familiar_decided += 1
                        if is_supported:
                            familiar_supported += 1
                    else:
                        unfamiliar_decided += 1
                        if is_supported:
                            unfamiliar_supported += 1
                if rev["decision"] == "confirmed" and rev.get("replacement") is None and sid not in corrected_steps:
                    prior_task_conf[key] = prior_task_conf.get(key, 0) + 1

        well_supported_rate = round(well_supported / decided, 6) if decided > 0 else 0.0
        familiar_precision = round(familiar_supported / familiar_decided, 6) if familiar_decided > 0 else None
        unfamiliar_precision = round(unfamiliar_supported / unfamiliar_decided, 6) if unfamiliar_decided > 0 else None
        corr_count = len(corrected_steps)
        if decided >= min_confirmations and well_supported_rate >= threshold_float and corr_count == 0:
            verdict = "well_supported"
        elif decided > 0:
            verdict = "needs_calibration_or_review"
        else:
            verdict = "insufficient_evidence"
        return {
            "question": "Does this hybrid make well-supported decisions in practice?",
            "verdict": verdict,
            "decided_steps": decided,
            "well_supported_decisions": well_supported,
            "well_supported_rate": well_supported_rate,
            "familiar_task_decisions": familiar_decided,
            "familiar_task_precision": familiar_precision,
            "unfamiliar_task_decisions": unfamiliar_decided,
            "unfamiliar_task_precision": unfamiliar_precision,
            "core_generated_or_assessed_decisions": core_generated_or_assessed,
            "caller_supplied_signal_decisions": caller_supplied_signals,
            "correction_count": corr_count,
            "replacement_count": replacements,
            "evidence_boundary": {
                "review_backed_decisions": decided,
                "familiar_task_decisions": familiar_decided,
                "unfamiliar_task_decisions": unfamiliar_decided,
                "core_generated_or_assessed_decisions": core_generated_or_assessed,
                "caller_supplied_signal_decisions": caller_supplied_signals,
                "scope": "Distinguishes caller-supplied confidence/novelty fields and scripted held-out cases from journaled judgment nodes, core.generate plans and measured cross-task transfer",
            },
        }

    def calibration(
        self,
        task_id=None,
        *,
        min_confirmations=2,
        threshold=0.8,
        alpha=0.05,
        policy="bracket",
        core_identity=None,
        implementation=None,
    ):
        """Derive per-operator System One action bands integrating review history, task familiarity, exact Beta-binomial posterior-mass brackets and explicit rules."""
        from .relations import OPERATORS as REL_OPS
        try:
            threshold_frac = _to_rational(threshold)
            alpha_frac = _to_rational(alpha)
        except ValueError as error:
            raise ValueError("Calibration bounds") from error
        if (type(min_confirmations) is not int or min_confirmations < 1
                or not 0 < threshold_frac < 1 or not 0 < alpha_frac < Fraction(1, 2)
                or policy not in {"configured_rule", "bracket"}):
            raise ValueError("Calibration bounds")
        threshold_float = float(threshold_frac)
        alpha_float = float(alpha_frac)
        self.memory.reload()
        active_impl = implementation if implementation is not None else self.implementation
        active_core = core_identity
        if active_core is None:
            for step in reversed(list(self.memory.state["steps"].values())):
                if task_id is None or step["snapshot"].get("task_id") == task_id:
                    active_core = step["snapshot"].get("core_identity")
                    break
        admitted = {"source.read", "json.field", "integer.product", "integer.sum",
                    "quantity.field", "quantity.product", "quantity.sum"} | set(REL_OPS)

        def _empty_row(op):
            return {
                "operator": op, "proposed": 0, "observed": 0, "confirmed": 0,
                "rejected": 0, "cancelled": 0, "replaced": 0, "corrections": 0,
                "conduct_flags": 0, "global_confirmed": 0, "global_rejected": 0,
                "global_corrections": 0, "global_conduct_flags": 0,
                "excluded_implementation_changes": 0,
                "excluded_core_version_changes": 0,
                "excluded_duplicate_steps": 0,
                "excluded_self_reviews": 0,
                "_tasks_confirmed": set(),
                "_seen_task_sigs": set(),
                "_seen_global_sigs": set(),
            }

        stats = {op: _empty_row(op) for op in sorted(admitted)}
        for sid, step in self.memory.state["steps"].items():
            op = step["operator"]
            st_task = step["snapshot"]["task_id"]
            row = stats.setdefault(op, _empty_row(op))
            obs = self.memory.state["step_observations"].get(sid)
            rev = self.memory.state["step_reviews"].get(sid)
            in_scope_task = task_id is None or st_task == task_id
            if in_scope_task:
                row["proposed"] += 1
                if obs is not None:
                    row["observed"] += 1
                if rev and rev["decision"] == "cancelled":
                    row["cancelled"] += 1
            if not rev or rev["decision"] not in {"confirmed", "rejected"}:
                continue
            # Condition (a): counts restart when operator implementation or core version changes
            if step.get("implementation") != active_impl:
                if in_scope_task:
                    row["excluded_implementation_changes"] += 1
                continue
            step_core = step["snapshot"].get("core_identity")
            if active_core is not None and step_core != active_core:
                if in_scope_task:
                    row["excluded_core_version_changes"] += 1
                continue
            # Condition (c): a review by the same core that proposed the step is not evidence about that core
            if step_core is not None and rev.get("reviewer") == step_core:
                if in_scope_task:
                    row["excluded_self_reviews"] += 1
                continue
            # Condition (b): one count per distinct eligible step (confirmations cannot inflate by repeating identical step signature)
            sig_body = obs["cache_key"] if obs is not None else digest_bytes(
                canonical({
                    "operator": op,
                    "implementation": step.get("implementation"),
                    "arguments": step.get("arguments"),
                    "sources": step["snapshot"].get("sources"),
                })
            )
            step_sig = (st_task, sig_body)
            if rev["decision"] == "confirmed":
                if obs is None:
                    continue
                if step_sig not in row["_seen_global_sigs"]:
                    row["_seen_global_sigs"].add(step_sig)
                    row["global_confirmed"] += 1
                    row["_tasks_confirmed"].add(st_task)
                if in_scope_task:
                    if step_sig in row["_seen_task_sigs"]:
                        row["excluded_duplicate_steps"] += 1
                    else:
                        row["_seen_task_sigs"].add(step_sig)
                        row["confirmed"] += 1
            elif rev["decision"] == "rejected":
                row["global_rejected"] += 1
                if in_scope_task:
                    row["rejected"] += 1
                    if rev.get("replacement") is not None:
                        row["replaced"] += 1
        for corr in self.memory.state["corrections"].values():
            op = corr["operator"]
            row = stats.setdefault(op, _empty_row(op))
            row["global_corrections"] += 1
            if corr.get("conduct_virtue") is not None:
                row["global_conduct_flags"] += 1
            if task_id is not None and corr["task_id"] != task_id:
                continue
            row["corrections"] += 1
            if corr.get("conduct_virtue") is not None:
                row["conduct_flags"] += 1

        threshold_str = f"{threshold_frac.numerator}/{threshold_frac.denominator}"
        alpha_str = f"{alpha_frac.numerator}/{alpha_frac.denominator}"
        operators_out = {}
        for op, row in sorted(stats.items()):
            tasks_confirmed = sorted(row.pop("_tasks_confirmed"))
            row.pop("_seen_task_sigs", None)
            row.pop("_seen_global_sigs", None)
            decided = row["confirmed"] + row["rejected"]
            agreement = (row["confirmed"] / decided) if decided > 0 else 0.0
            replacement_rate = (row["replaced"] / decided) if decided > 0 else 0.0
            g_decided = row["global_confirmed"] + row["global_rejected"]
            g_agreement = (row["global_confirmed"] / g_decided) if g_decided > 0 else 0.0
            lower_5pct_raw = self.beta_quantile(1 + row["confirmed"], 1 + row["rejected"], alpha_float)
            upper_95pct_raw = self.beta_quantile(1 + row["confirmed"], 1 + row["rejected"], 1.0 - alpha_float)
            lower_5pct = round(lower_5pct_raw, 6)
            upper_95pct = round(upper_95pct_raw, 6)
            mass_below = mass_at_or_below(row["confirmed"], row["rejected"], threshold_frac)
            mass_above = Fraction(1, 1) - mass_below
            raw_bracket = band_bracket(row["confirmed"], row["rejected"], threshold=threshold_frac, alpha=alpha_frac)
            if row["conduct_flags"] > 0 or row["global_conduct_flags"] > 0:
                bracket_trit = "no"
            elif raw_bracket == "yes" and row["corrections"] == 0:
                bracket_trit = "yes"
            elif raw_bracket == "no":
                bracket_trit = "no"
            else:
                bracket_trit = "both"
            task_familiar = row["confirmed"] >= min_confirmations
            configured_rule_fires = task_familiar and agreement >= threshold_float and row["corrections"] == 0
            statistical_competence = bracket_trit == "yes"
            cross_task_eligible = (
                row["global_confirmed"] >= min_confirmations
                and g_agreement >= threshold_float
                and row["global_corrections"] == 0
                and row["corrections"] == 0
            )
            if row["conduct_flags"] > 0 or row["global_conduct_flags"] > 0:
                band = "escalate"
            elif policy == "bracket":
                band = "act_now" if bracket_trit == "yes" else ("escalate" if bracket_trit == "no" else "propose")
            elif configured_rule_fires:
                band = "act_now"
            elif bracket_trit == "no":
                band = "escalate"
            else:
                band = "propose"
            admission_conditions = {
                "exchangeability_scope": {
                    "implementation": active_impl,
                    "core_identity": active_core,
                    "excluded_implementation_changes": row["excluded_implementation_changes"],
                    "excluded_core_version_changes": row["excluded_core_version_changes"],
                    "rule": "Counts restart when operator implementation or core version changes, since reviews before and after are not exchangeable",
                },
                "distinct_step_evidence": {
                    "distinct_eligible_steps": decided,
                    "excluded_duplicate_steps": row["excluded_duplicate_steps"],
                    "rule": "One count per distinct eligible step, not repeated confirmations of the same step",
                },
                "reviewer_independence": {
                    "proposing_core_identity": active_core,
                    "excluded_self_reviews": row["excluded_self_reviews"],
                    "rule": "A review by the same core that proposed the step is not evidence about that core",
                },
                "sequential_interpretation": {
                    "prior": "Beta(1,1)",
                    "threshold_rational": threshold_str,
                    "alpha_rational": alpha_str,
                    "rational_conversion": "Fraction(str(value)) for decimal float inputs; exact Fraction preserved",
                    "equality_convention": "mass <= alpha",
                    "posterior_mass_at_or_below": f"{mass_below.numerator}/{mass_below.denominator}",
                    "posterior_mass_at_or_above": f"{mass_above.numerator}/{mass_above.denominator}",
                    "rule": "Posterior statement under declared Beta(1,1) prior, not a frequentist 5% guarantee when re-evaluated after every review; an anytime-valid sequence is the alternative if sequential frequentist control is required",
                },
            }
            operators_out[op] = {
                **row,
                "distinct_eligible_steps": decided,
                "distinct_tasks_confirmed": len(tasks_confirmed),
                "task_familiar": task_familiar,
                "point_agreement_rate": round(agreement, 6),
                "configured_rule_fires": configured_rule_fires,
                "statistical_competence_established": statistical_competence,
                "cross_task_transfer_eligible": cross_task_eligible,
                "global_agreement_rate": round(g_agreement, 6),
                "agreement_rate": round(agreement, 6),
                "empirical_precision": round(agreement, 6),
                "agreement_lower_5pct": lower_5pct,
                "agreement_upper_95pct": upper_95pct,
                "threshold_rational": threshold_str,
                "alpha_rational": alpha_str,
                "posterior_mass_at_or_below": f"{mass_below.numerator}/{mass_below.denominator}",
                "posterior_mass_at_or_above": f"{mass_above.numerator}/{mass_above.denominator}",
                "agreement_bracket": {
                    "lower_5pct": lower_5pct,
                    "upper_95pct": upper_95pct,
                    "prior": "Beta(1,1)",
                    "threshold_rational": threshold_str,
                    "alpha_rational": alpha_str,
                    "rational_conversion": "Fraction(str(value)) for decimal float inputs; exact Fraction preserved",
                    "equality_convention": "mass <= alpha",
                    "posterior_mass_at_or_below": f"{mass_below.numerator}/{mass_below.denominator}",
                    "posterior_mass_at_or_above": f"{mass_above.numerator}/{mass_above.denominator}",
                    "conditional_assumption": "Exchangeable Bernoulli review outcomes; conditional statistical bound, not an exact deductive pi certificate",
                },
                "admission_conditions": admission_conditions,
                "bracket_trit": bracket_trit,
                "replacement_rate": round(replacement_rate, 6),
                "band": band,
            }
        return {
            "task_id": task_id,
            "min_confirmations": min_confirmations,
            "threshold": threshold_float,
            "threshold_rational": threshold_str,
            "alpha": alpha_float,
            "alpha_rational": alpha_str,
            "rational_conversion": "Fraction(str(value)) for decimal float inputs; exact Fraction preserved",
            "equality_convention": "mass <= alpha",
            "policy": policy,
            "calibration_assumptions": {
                "configured_rule": "Diagnostic or opt-in rule (policy='configured_rule') firing at confirmed >= min_confirmations and point_agreement_rate >= threshold in a familiar task with zero corrections; two confirmations can justify a configured rule firing with core endorsement, but do not alone establish statistical competence",
                "posterior_bracket": "Default policy='bracket' evaluates exact rational Beta-binomial tail mass P(p <= threshold) <= alpha for 'yes' (act) and 1 - P(p <= threshold) <= alpha for 'no' (escalate) under Beta(1 + confirmed, 1 + rejected)",
                "exchangeability_and_independence": "Counts restart on operator implementation or core version changes, count at most once per distinct eligible step, and exclude self-reviews by the proposing core",
                "sequential_scope": "Posterior statement under declared Beta(1,1) prior, not a frequentist alpha guarantee under repeated re-evaluation after every review (which would require an anytime-valid sequence)",
                "certificate_distinction": "Statistical posterior brackets are conditional evidence under an exchangeability model, unlike exact deductive pi completion certificates over finite relational worlds",
            },
            "hybrid_components": {
                "learned_core": "Context-sensitive action proposal, usefulness and uncertainty estimation",
                "review_history": "In-task and cross-task outcome evidence, replacements and audit corrections",
                "explicit_rules": "Operator admission, argument validity, source scope and conduct boundaries",
                "calibration": "Signal integration into act_now, propose or escalate with unfamiliar-task safeguards",
            },
            "decision_support": self.decision_support_audit(
                task_id, min_confirmations=min_confirmations, threshold=threshold_float
            ),
            "operators": operators_out,
            "scope": "Derived by replay from journaled step reviews, corrections and task familiarity",
        }

    def band_proposal(
        self,
        snapshot,
        operator,
        arguments,
        *,
        core_signal=None,
        judgment_node_id=None,
        conduct_flag=False,
        min_confirmations=2,
        threshold=0.8,
        alpha=0.05,
        policy="bracket",
        execute_act_now=True,
    ):
        """Classify a candidate step into act_now, propose or escalate by integrating the 4 hybrid signals."""
        learned_core = self._resolve_core_signal(
            snapshot, core_signal=core_signal, judgment_node_id=judgment_node_id
        )
        if conduct_flag:
            return {
                "operator": operator, "band": "escalate", "executed": False, "step": None, "observation": None,
                "signals": {
                    "learned_core": learned_core,
                    "explicit_rules": {"valid": False, "conduct_boundary_clear": False, "reason": "conduct_flag"},
                },
                "reason": "Conduct-sensitive step requires explicit escalation",
            }
        norm_args = deepcopy(arguments)
        if operator in {"integer.product", "integer.sum", "quantity.product", "quantity.sum"} and "inputs" in norm_args and "steps" not in norm_args:
            norm_args = {"steps": list(norm_args["inputs"])}
        try:
            self.parents(operator, norm_args)
            if operator == "source.read" and norm_args["source"] not in snapshot.get("sources", {}):
                return {
                    "operator": operator, "band": "escalate", "executed": False, "step": None, "observation": None,
                    "signals": {
                        "learned_core": learned_core,
                        "explicit_rules": {"valid": False, "sources_in_scope": False, "reason": "source_outside_selection"},
                    },
                    "reason": "Source lies outside the task's current evidence selection",
                }
        except ValueError as error:
            return {
                "operator": operator, "band": "escalate", "executed": False, "step": None, "observation": None,
                "signals": {
                    "learned_core": learned_core,
                    "explicit_rules": {"valid": False, "reason": str(error)},
                },
                "reason": str(error),
            }

        explicit_rules = {
            "valid": True,
            "operator_admitted": True,
            "arguments_valid": True,
            "sources_in_scope": True,
            "conduct_boundary_clear": True,
        }
        cal = self.calibration(
            snapshot.get("task_id"),
            min_confirmations=min_confirmations,
            threshold=threshold,
            alpha=alpha,
            policy=policy,
            core_identity=snapshot.get("core_identity"),
        )
        threshold_float = cal["threshold"]
        op_stats = cal["operators"].get(operator)
        review_history = {
            "in_task_confirmed": op_stats["confirmed"] if op_stats else 0,
            "in_task_rejected": op_stats["rejected"] if op_stats else 0,
            "distinct_eligible_steps": op_stats["distinct_eligible_steps"] if op_stats else 0,
            "in_task_corrections": op_stats["corrections"] if op_stats else 0,
            "in_task_agreement_rate": op_stats["agreement_rate"] if op_stats else 0.0,
            "point_agreement_rate": op_stats["point_agreement_rate"] if op_stats else 0.0,
            "agreement_lower_5pct": op_stats["agreement_lower_5pct"] if op_stats else 0.05,
            "agreement_upper_95pct": op_stats["agreement_upper_95pct"] if op_stats else 0.95,
            "threshold_rational": op_stats["threshold_rational"] if op_stats else cal["threshold_rational"],
            "alpha_rational": op_stats["alpha_rational"] if op_stats else cal["alpha_rational"],
            "rational_conversion": cal["rational_conversion"],
            "equality_convention": cal["equality_convention"],
            "posterior_mass_at_or_below": op_stats["posterior_mass_at_or_below"] if op_stats else None,
            "posterior_mass_at_or_above": op_stats["posterior_mass_at_or_above"] if op_stats else None,
            "admission_conditions": deepcopy(op_stats["admission_conditions"]) if op_stats else None,
            "bracket_trit": op_stats["bracket_trit"] if op_stats else "both",
            "configured_rule_fires": op_stats["configured_rule_fires"] if op_stats else False,
            "statistical_competence_established": op_stats["statistical_competence_established"] if op_stats else False,
            "global_confirmed": op_stats["global_confirmed"] if op_stats else 0,
            "global_agreement_rate": op_stats["global_agreement_rate"] if op_stats else 0.0,
            "global_corrections": op_stats["global_corrections"] if op_stats else 0,
            "distinct_tasks_confirmed": op_stats["distinct_tasks_confirmed"] if op_stats else 0,
            "task_familiar": op_stats["task_familiar"] if op_stats else False,
            "cross_task_transfer_eligible": op_stats["cross_task_transfer_eligible"] if op_stats else False,
        }

        if op_stats and (op_stats["conduct_flags"] > 0 or op_stats["global_conduct_flags"] > 0):
            return {
                "operator": operator, "band": "escalate", "executed": False, "step": None, "observation": None,
                "agreement_rate": op_stats["agreement_rate"], "confirmed": op_stats["confirmed"],
                "corrections": op_stats["corrections"],
                "signals": {
                    "learned_core": learned_core,
                    "review_history": review_history,
                    "explicit_rules": explicit_rules,
                    "calibration": {
                        "decision": "escalate",
                        "trigger": "conduct_history",
                        "policy": policy,
                        "threshold_rational": cal["threshold_rational"],
                        "alpha_rational": cal["alpha_rational"],
                        "equality_convention": cal["equality_convention"],
                    },
                },
                "reason": "Conduct flag present in history",
            }

        if learned_core["supplied"] and (learned_core["verdict"] == "no" or learned_core["uncertainty"] == "conflict"):
            return {
                "operator": operator, "band": "escalate", "executed": False, "step": None, "observation": None,
                "agreement_rate": op_stats["agreement_rate"] if op_stats else 0.0,
                "confirmed": op_stats["confirmed"] if op_stats else 0,
                "corrections": op_stats["corrections"] if op_stats else 0,
                "signals": {
                    "learned_core": learned_core,
                    "review_history": review_history,
                    "explicit_rules": explicit_rules,
                    "calibration": {
                        "decision": "escalate",
                        "trigger": "learned_core_conflict_or_rejection",
                        "policy": policy,
                        "threshold_rational": cal["threshold_rational"],
                        "alpha_rational": cal["alpha_rational"],
                        "equality_convention": cal["equality_convention"],
                    },
                },
                "reason": "Learned core context-sensitive judgment opposes action or reports unresolved conflict",
            }

        core_uncertain = (
            learned_core["supplied"]
            and (
                learned_core["verdict"] == "maybe"
                or learned_core["uncertainty"] is not None
                or (learned_core["usefulness"] is not None and learned_core["usefulness"] < threshold_float)
            )
        )
        core_endorses = (
            learned_core["supplied"]
            and learned_core["verdict"] == "yes"
            and learned_core["uncertainty"] is None
            and (
                (learned_core["usefulness"] is not None and learned_core["usefulness"] >= threshold_float)
                or (
                    learned_core["usefulness"] is None
                    and learned_core["provenance"] in {"assessed_judgment", "provisional_judgment"}
                )
            )
        )

        if core_uncertain:
            band = "propose"
            trigger = "learned_core_uncertainty_override"
            reason = "Learned core estimates context-sensitive uncertainty or usefulness below threshold; review required"
        elif op_stats and op_stats["band"] == "act_now":
            band = "act_now"
            trigger = "familiar_task_posterior_bracket" if policy == "bracket" else "familiar_task_calibrated_rule"
            reason = "Empirical master agreement meets threshold in familiar task with zero corrections and aligned context"
        elif op_stats and op_stats["band"] == "escalate":
            band = "escalate"
            trigger = "posterior_bracket_below_threshold"
            reason = "Empirical agreement bracket lies below threshold"
        elif op_stats and op_stats["cross_task_transfer_eligible"]:
            if core_endorses:
                band = "act_now"
                trigger = "cross_task_history_plus_core_endorsement"
                reason = "Unfamiliar task supported by cross-task review history and context-sensitive learned core confirmation"
            else:
                band = "propose"
                trigger = "unfamiliar_task_requires_core_or_review"
                reason = "Unfamiliar task: prior confirmations on other tasks alone do not establish competence without context-sensitive core judgment"
        else:
            band = "propose"
            trigger = "insufficient_confirmations_or_prior_corrections"
            reason = "Below confirmation threshold or has prior corrections"

        decision_record = {
            "band": band,
            "policy": policy,
            "trigger": trigger,
            "threshold_rational": cal["threshold_rational"],
            "alpha_rational": cal["alpha_rational"],
            "rational_conversion": cal["rational_conversion"],
            "equality_convention": cal["equality_convention"],
            "posterior_mass_at_or_below": review_history["posterior_mass_at_or_below"],
            "posterior_mass_at_or_above": review_history["posterior_mass_at_or_above"],
            "bracket_trit": review_history["bracket_trit"],
            "confirmed": review_history["in_task_confirmed"],
            "rejected": review_history["in_task_rejected"],
            "distinct_eligible_steps": review_history["distinct_eligible_steps"],
            "point_agreement_rate": review_history["point_agreement_rate"],
            "configured_rule_fires": review_history["configured_rule_fires"],
            "admission_conditions": review_history["admission_conditions"],
        }
        step = self.propose(
            snapshot,
            operator,
            norm_args,
            core_signal_provenance=learned_core["provenance"],
            caller_supplied_signal=bool(learned_core["caller_supplied"]),
            decision_record=decision_record,
        )
        obs = self.execute(step["id"]) if (band == "act_now" and execute_act_now) else None
        return {
            "operator": operator,
            "band": band,
            "executed": obs is not None,
            "step": step,
            "observation": obs,
            "agreement_rate": op_stats["agreement_rate"] if op_stats else 0.0,
            "point_agreement_rate": op_stats["point_agreement_rate"] if op_stats else 0.0,
            "confirmed": op_stats["confirmed"] if op_stats else 0,
            "corrections": op_stats["corrections"] if op_stats else 0,
            "signals": {
                "learned_core": learned_core,
                "review_history": review_history,
                "explicit_rules": explicit_rules,
                "calibration": {
                    "decision": band,
                    "trigger": trigger,
                    "policy": policy,
                    "threshold_rational": cal["threshold_rational"],
                    "alpha_rational": cal["alpha_rational"],
                    "rational_conversion": cal["rational_conversion"],
                    "equality_convention": cal["equality_convention"],
                },
            },
            "reason": reason,
        }


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
        if not isinstance(plan["operator"], str) or plan["operator"] not in {"integer.product", "integer.sum", "quantity.field", "quantity.product", "quantity.sum"}:
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
                "step": step, "step_id": step["id"], "result": result["result"]}


def pack_rank_plane(ranks):
    """Encode 9-symbol master rank buckets (1..8, and 9 for >8) as 2 trits per position in pack5 bytes."""
    ranks = list(ranks)
    if len(ranks) > 32768 or any(type(r) is not int or r < 1 for r in ranks):
        raise ValueError("Rank plane requires integer ranks >= 1 (where >=9 is bucketed to 9)")
    trits = []
    for r in ranks:
        code = min(r, 9) - 1
        trits.append(code % 3)
        trits.append(code // 3)
    return bytes(
        sum(t * (3 ** j) for j, t in enumerate(trits[i:i + 5]))
        for i in range(0, len(trits), 5)
    )


def unpack_rank_plane(data, count):
    """Decode pack5 two-trit rank plane bytes back into 9-symbol ranks in 1..9."""
    trit_count = 2 * count if type(count) is int else -1
    if (not isinstance(data, bytes) or type(count) is not int or not 0 <= count <= 32768
            or len(data) != (trit_count + 4) // 5):
        raise ValueError("Packed rank plane extent")
    trits = []
    for i, byte in enumerate(data):
        width = min(5, trit_count - 5 * i)
        if byte >= 3 ** width:
            raise ValueError("Packed rank plane contains an unused code or noncanonical tail")
        for _ in range(width):
            trits.append(byte % 3)
            byte //= 3
    return [1 + trits[2 * k] + 3 * trits[2 * k + 1] for k in range(count)]
