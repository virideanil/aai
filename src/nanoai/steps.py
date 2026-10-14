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
            default_usefulness = (
                1.0 if (verdict == "yes" and assessed)
                else (0.85 if verdict == "yes" else (0.5 if verdict == "maybe" else 0.0))
            )
            return {
                "supplied": True,
                "source": "judgment_node",
                "judgment_node_id": judgment_node_id,
                "verdict": verdict,
                "usefulness": default_usefulness,
                "uncertainty": uncertainty,
                "assessed": assessed,
            }
        if core_signal is not None:
            if not isinstance(core_signal, dict):
                raise ValueError("Learned core signal must be a dictionary")
            verdict = core_signal.get("verdict", "yes")
            if verdict not in {"yes", "maybe", "no"}:
                raise ValueError("Learned core signal verdict must be yes, maybe or no")
            usefulness = core_signal.get("usefulness")
            if usefulness is not None:
                if type(usefulness) not in (int, float) or not 0.0 <= float(usefulness) <= 1.0:
                    raise ValueError("Learned core usefulness must lie in [0.0, 1.0]")
                usefulness = float(usefulness)
            else:
                usefulness = 0.85 if verdict == "yes" else (0.5 if verdict == "maybe" else 0.0)
            uncertainty = core_signal.get("uncertainty")
            if uncertainty is not None and not isinstance(uncertainty, str):
                raise ValueError("Learned core uncertainty must be a string or None")
            return {
                "supplied": True,
                "source": "core_estimate",
                "judgment_node_id": None,
                "verdict": verdict,
                "usefulness": round(usefulness, 6),
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
                        "judgment_node_id": node["id"],
                        "verdict": "maybe",
                        "usefulness": 0.25,
                        "uncertainty": "conflict",
                        "assessed": False,
                    }
        return {
            "supplied": False,
            "source": "none",
            "judgment_node_id": None,
            "verdict": None,
            "usefulness": None,
            "uncertainty": None,
            "assessed": False,
        }

    def decision_support_audit(self, task_id=None, *, min_confirmations=2, threshold=0.8):
        """Evaluate whether the hybrid system makes well-supported decisions in practice across familiar and unfamiliar tasks."""
        self.memory.reload()
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

        for sid, step in self.memory.state["steps"].items():
            st_task = step["snapshot"]["task_id"]
            op = step["operator"]
            key = (st_task, op)
            was_familiar = prior_task_conf.get(key, 0) >= min_confirmations
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
        if decided >= min_confirmations and well_supported_rate >= threshold and corr_count == 0:
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
            "correction_count": corr_count,
            "replacement_count": replacements,
        }

    def calibration(self, task_id=None, *, min_confirmations=2, threshold=0.8):
        """Derive per-operator System One action bands integrating review history, task familiarity and explicit rules."""
        from .relations import OPERATORS as REL_OPS
        if type(min_confirmations) is not int or min_confirmations < 1 or not 0.0 < threshold <= 1.0:
            raise ValueError("Calibration bounds")
        self.memory.reload()
        admitted = {"source.read", "json.field", "integer.product", "integer.sum",
                    "quantity.field", "quantity.product", "quantity.sum"} | set(REL_OPS)

        def _empty_row(op):
            return {
                "operator": op, "proposed": 0, "observed": 0, "confirmed": 0,
                "rejected": 0, "cancelled": 0, "replaced": 0, "corrections": 0,
                "conduct_flags": 0, "global_confirmed": 0, "global_rejected": 0,
                "global_corrections": 0, "global_conduct_flags": 0,
                "_tasks_confirmed": set(),
            }

        stats = {op: _empty_row(op) for op in sorted(admitted)}
        for sid, step in self.memory.state["steps"].items():
            op = step["operator"]
            st_task = step["snapshot"]["task_id"]
            row = stats.setdefault(op, _empty_row(op))
            rev = self.memory.state["step_reviews"].get(sid)
            if rev:
                if rev["decision"] == "confirmed":
                    row["global_confirmed"] += 1
                    row["_tasks_confirmed"].add(st_task)
                elif rev["decision"] == "rejected":
                    row["global_rejected"] += 1
            if task_id is not None and st_task != task_id:
                continue
            row["proposed"] += 1
            if sid in self.memory.state["step_observations"]:
                row["observed"] += 1
            if rev:
                row[rev["decision"]] += 1
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

        operators_out = {}
        for op, row in sorted(stats.items()):
            tasks_confirmed = sorted(row.pop("_tasks_confirmed"))
            decided = row["confirmed"] + row["rejected"]
            agreement = (row["confirmed"] / decided) if decided > 0 else 0.0
            replacement_rate = (row["replaced"] / decided) if decided > 0 else 0.0
            g_decided = row["global_confirmed"] + row["global_rejected"]
            g_agreement = (row["global_confirmed"] / g_decided) if g_decided > 0 else 0.0
            task_familiar = row["confirmed"] >= min_confirmations
            cross_task_eligible = (
                row["global_confirmed"] >= min_confirmations
                and g_agreement >= threshold
                and row["global_corrections"] == 0
                and row["corrections"] == 0
            )
            if row["conduct_flags"] > 0 or row["global_conduct_flags"] > 0:
                band = "escalate"
            elif task_familiar and agreement >= threshold and row["corrections"] == 0:
                band = "act_now"
            else:
                band = "propose"
            operators_out[op] = {
                **row,
                "distinct_tasks_confirmed": len(tasks_confirmed),
                "task_familiar": task_familiar,
                "cross_task_transfer_eligible": cross_task_eligible,
                "global_agreement_rate": round(g_agreement, 6),
                "agreement_rate": round(agreement, 6),
                "empirical_precision": round(agreement, 6),
                "replacement_rate": round(replacement_rate, 6),
                "band": band,
            }
        return {
            "task_id": task_id,
            "min_confirmations": min_confirmations,
            "threshold": threshold,
            "hybrid_components": {
                "learned_core": "Context-sensitive action proposal, usefulness and uncertainty estimation",
                "review_history": "In-task and cross-task outcome evidence, replacements and audit corrections",
                "explicit_rules": "Operator admission, argument validity, source scope and conduct boundaries",
                "calibration": "Signal integration into act_now, propose or escalate with unfamiliar-task safeguards",
            },
            "decision_support": self.decision_support_audit(
                task_id, min_confirmations=min_confirmations, threshold=threshold
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
        cal = self.calibration(snapshot.get("task_id"), min_confirmations=min_confirmations, threshold=threshold)
        op_stats = cal["operators"].get(operator)
        review_history = {
            "in_task_confirmed": op_stats["confirmed"] if op_stats else 0,
            "in_task_rejected": op_stats["rejected"] if op_stats else 0,
            "in_task_corrections": op_stats["corrections"] if op_stats else 0,
            "in_task_agreement_rate": op_stats["agreement_rate"] if op_stats else 0.0,
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
                    "calibration": {"decision": "escalate", "trigger": "conduct_history"},
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
                    "calibration": {"decision": "escalate", "trigger": "learned_core_conflict_or_rejection"},
                },
                "reason": "Learned core context-sensitive judgment opposes action or reports unresolved conflict",
            }

        core_uncertain = (
            learned_core["supplied"]
            and (
                learned_core["verdict"] == "maybe"
                or learned_core["uncertainty"] is not None
                or (learned_core["usefulness"] is not None and learned_core["usefulness"] < threshold)
            )
        )
        core_endorses = (
            learned_core["supplied"]
            and learned_core["verdict"] == "yes"
            and learned_core["uncertainty"] is None
            and (learned_core["usefulness"] is None or learned_core["usefulness"] >= threshold)
        )

        if core_uncertain:
            band = "propose"
            trigger = "learned_core_uncertainty_override"
            reason = "Learned core estimates context-sensitive uncertainty or usefulness below threshold; review required"
        elif op_stats and op_stats["band"] == "act_now":
            band = "act_now"
            trigger = "familiar_task_calibrated_rule"
            reason = "Empirical master agreement meets threshold in familiar task with zero corrections and aligned context"
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

        step = self.propose(snapshot, operator, norm_args)
        obs = self.execute(step["id"]) if (band == "act_now" and execute_act_now) else None
        return {
            "operator": operator,
            "band": band,
            "executed": obs is not None,
            "step": step,
            "observation": obs,
            "agreement_rate": op_stats["agreement_rate"] if op_stats else 0.0,
            "confirmed": op_stats["confirmed"] if op_stats else 0,
            "corrections": op_stats["corrections"] if op_stats else 0,
            "signals": {
                "learned_core": learned_core,
                "review_history": review_history,
                "explicit_rules": explicit_rules,
                "calibration": {"decision": band, "trigger": trigger},
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
