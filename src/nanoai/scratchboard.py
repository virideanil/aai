"""Immutable figurative units, exact tensor data and bounded materialization.

The same SJQL task journal owns these nodes. Large unresolved ranges reserve
addresses, not learned content. Every use retains source, task and model-basis
checks; rendering or copying confers no new factual verdict.
"""
from __future__ import annotations

from copy import deepcopy
import math
import os
from pathlib import Path
import tempfile
import time
import uuid

from .core import digest_bytes, file_digest
from .memory import StaleState, bounded_text, canonical, document_references

MAX_SLOTS = 10_000_000
MAX_BINDINGS = 256
MAX_TENSOR_BYTES = 1 << 20
MAX_MATERIALIZED_BYTES = 64 << 20


class UnresolvedSlot(ValueError):
    pass


class Scratchboard:
    def __init__(self, memory):
        self.memory = memory
        self.implementation = file_digest(__file__)

    @staticmethod
    def _count(value, maximum=MAX_SLOTS):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError("Scratchboard count exceeds its declared bound")
        return value

    @staticmethod
    def _children(spec):
        if spec["kind"] == "concat":
            return spec["nodes"]
        if spec["kind"] == "repeat":
            return [spec["node"]]
        if spec["kind"] == "slots":
            return list(spec["bindings"].values())
        if spec["kind"] == "generator" and spec.get("operator") == "slot_sequence":
            return list(spec["arguments"]["items"])
        if spec["kind"] == "judgment" and spec["prediction"] is not None:
            return [spec["prediction"]]
        if spec["kind"] == "import":
            return [spec["relevance"]]
        return []

    def _bases(self, snapshot, *, node=None, seen=None):
        """Find numerical bases transitively; exact provenance is not a basis."""
        seen = set() if seen is None else seen
        bases = set()
        if node is not None:
            key = ("node", node["id"])
            if key in seen:
                return bases
            seen.add(key)
            spec = node["spec"]
            if spec["kind"] == "tensor":
                bases.add(spec["basis"])
            if spec["kind"] == "artifact":
                op = self.memory.state["operations"][spec["operation_id"]]
                bases.update(self._bases(self.memory.state["proposals"][op["proposal_id"]]["snapshot"], seen=seen))
            if spec["kind"] == "import":
                parent = self.memory.state["scratchboard"][spec["node"]]
                bases.update(self._bases(parent["snapshot"], node=parent, seen=seen))
            if spec["kind"] == "judgment" and spec["predicate"] is not None:
                bases.update(self._bases(self.memory.state["steps"][spec["predicate"]["step_id"]]["snapshot"], seen=seen))
        for node_id in snapshot.get("scratchboard", {}):
            parent = self.memory.state["scratchboard"][node_id]
            bases.update(self._bases(parent["snapshot"], node=parent, seen=seen))
        for step_id in snapshot.get("observations", {}):
            key = ("step", step_id)
            if key not in seen:
                seen.add(key)
                bases.update(self._bases(self.memory.state["steps"][step_id]["snapshot"], seen=seen))
        return bases

    def _compatible(self, node, snapshot, *, cross_task=False):
        keys = ("identity",) if cross_task else ("task_id", "task_revision", "identity")
        if any(node["snapshot"][k] != snapshot[k] for k in keys):
            raise StaleState("Scratchboard dependency belongs to another task or continuing identity")
        if self._bases(node["snapshot"], node=node) - {snapshot["core_identity"]}:
            raise StaleState("Scratchboard numerical dependency has an incompatible model basis")

    def _generator_components(self, snapshot, spec, *, cache=None):
        if spec["renderer"] != self.implementation:
            raise StaleState("Generator renderer changed")
        op, args = spec["operator"], spec["arguments"]
        if not isinstance(args, dict):
            raise ValueError("Generator arguments must be an object")
        if op == "source_lines":
            if (set(args) != {"source", "start_line", "stop_line", "numbered"}
                    or args["source"] not in snapshot["sources"]
                    or type(args["start_line"]) is not int or type(args["stop_line"]) is not int
                    or type(args["numbered"]) is not bool):
                raise ValueError("Generator source_lines contract")
            raw_text = self.memory.state["sources"][args["source"]]["text"]
            lines = raw_text.splitlines()
            start_line, stop_line = args["start_line"], args["stop_line"]
            if not lines or not 1 <= start_line <= stop_line <= len(lines):
                raise ValueError("Generator source_lines range extent")
            rendered = ("\n".join(
                f"{line_no}: {lines[line_no - 1]}" if args["numbered"] else lines[line_no - 1]
                for line_no in range(start_line, stop_line + 1)
            ) + "\n").encode("utf-8")
            return {}, stop_line - start_line + 1, rendered
        if op == "relation_table":
            import json
            from .relations import evaluate, expression_variables, query_tensor_enclosure, validate_model
            allowed_keys = {"step_id", "open_step_ids", "questions"}
            if (not set(args) <= allowed_keys or "step_id" not in args or not isinstance(args["step_id"], str)
                    or not isinstance(args.get("open_step_ids", []), list) or len(args.get("open_step_ids", [])) > 16
                    or len(set(args.get("open_step_ids", []))) != len(args.get("open_step_ids", []))
                    or not isinstance(args.get("questions", []), list) or len(args.get("questions", [])) > 16):
                raise ValueError("Generator relation_table contract")
            step_id = args["step_id"]
            step = self.memory.state["steps"].get(step_id)
            obs = self.memory.state["step_observations"].get(step_id)
            review = self.memory.state["step_reviews"].get(step_id)
            if not step or not obs or not review or review["decision"] != "confirmed":
                raise StaleState("Generator relation_table requires a confirmed observed step")
            if any(step["snapshot"][k] != snapshot[k] for k in ("task_id", "task_revision", "identity")):
                raise StaleState("Generator relation_table belongs to another task")
            if self._bases(step["snapshot"]) - {snapshot["core_identity"]}:
                raise StaleState("Generator relation_table has an incompatible model basis")
            bound = {step_id: obs["digest"]}
            if step["operator"] == "relation.query":
                state_id = step["arguments"]["step"]
                state_obs = self.memory.state["step_observations"].get(state_id)
                if not state_obs:
                    raise StaleState("Generator relation_table parent state is unobserved")
                bound[state_id] = state_obs["digest"]
                model = validate_model(deepcopy(state_obs["result"].get("value")))
            else:
                model = validate_model(deepcopy(obs["result"].get("value")))
            sources = sorted(set(obs["result"].get("sources", [])))
            assignments = [dict(zip(model["variables"], row)) for row in model["worlds"]]
            header = "| Variable | Verdict | Support2 | Possible |\n|---|---|---|---|\n"
            rows = []
            code_map = {(): "00", (1,): "01", (0,): "10", (0, 1): "11"}
            for var in model["variables"]:
                poss = sorted({row[var] for row in assignments})
                verdict = {(0,): "no", (1,): "yes", (0, 1): "maybe", (): "conflict"}[tuple(poss)]
                bits = code_map[tuple(poss)]
                rows.append(f"| `{var}` | {verdict} | `{bits}` | `{poss}` |")
            open_lines = []
            for oid in args.get("open_step_ids", []):
                ostep = self.memory.state["steps"].get(oid)
                oobs = self.memory.state["step_observations"].get(oid)
                if not ostep or not oobs or ostep["operator"] != "relation.query":
                    raise ValueError("Generator relation_table open_step_ids must name observed relation.query steps")
                if any(ostep["snapshot"][k] != snapshot[k] for k in ("task_id", "task_revision", "identity")):
                    raise StaleState("Generator relation_table open question belongs to another task")
                bound[oid] = oobs["digest"]
                ans = oobs["result"].get("value", {})
                open_lines.append(f"- `{json.dumps(ans.get('question'), sort_keys=True)}`: state={ans.get('state')} verdict={ans.get('verdict')}")
            for q_expr in args.get("questions", []):
                if expression_variables(q_expr) - set(model["variables"]):
                    raise ValueError("Generator relation_table question references an unavailable variable")
                enc = query_tensor_enclosure(model, q_expr)
                poss = sorted({evaluate(q_expr, row) for row in assignments})
                open_lines.append(f"- `{json.dumps(q_expr, sort_keys=True)}`: verdict={enc['exact_verdict']} box={enc['box_verdict']} possible={poss}")
            cite = " ".join(f"[{s}]" for s in sources)
            text = header + "\n".join(rows)
            if open_lines:
                text += "\n\n### Open Questions\n" + "\n".join(open_lines)
            if cite:
                text += f"\n\nSources: {cite}\n"
            units = max(1, len(model["variables"]) + len(model["worlds"]) + len(open_lines))
            return bound, units, text.encode("utf-8")
        if op == "slot_sequence":
            allowed_keys = {"template", "items", "start_index"}
            start_index = args.get("start_index", 0)
            if (not set(args) <= allowed_keys or "template" not in args or "items" not in args
                    or not isinstance(args["template"], str) or type(start_index) is not int
                    or not 1 <= len(args["template"].encode()) <= 2048 or "{item}" not in args["template"]
                    or not isinstance(args["items"], list) or not 1 <= len(args["items"]) <= 128
                    or any(not isinstance(x, str) for x in args["items"])):
                raise ValueError("Generator slot_sequence contract")
            if cache is None:
                return {}, len(args["items"]), None
            lines = []
            for idx, item_id in enumerate(args["items"], start=start_index):
                item_text = self._render(item_id, cache).decode("utf-8")
                lines.append(args["template"].replace("{index}", str(idx)).replace("{item}", item_text))
            sep = "" if args["template"].endswith("\n") else "\n"
            return {}, len(args["items"]), sep.join(lines).encode("utf-8")
        if op == "visual_report":
            from .documents import render_visual_report
            allowed_keys = {"title", "sections", "citations", "perception_regions"}
            if (not set(args) <= allowed_keys or "title" not in args or "sections" not in args
                    or not isinstance(args["title"], str) or not isinstance(args["sections"], list)):
                raise ValueError("Generator visual_report contract")
            allowed_refs = {
                f"{src_id}@{self.memory.state['sources'][src_id]['revision']}"
                for src_id in snapshot.get("sources", {})
                if src_id in self.memory.state["sources"]
            }
            citations = list(args.get("citations", sorted(allowed_refs)))
            if any(c not in allowed_refs for c in citations):
                raise ValueError("Generator visual_report citation lies outside selected sources")
            rendered_str = render_visual_report(
                args["title"],
                args["sections"],
                citations=citations,
                perception_regions=args.get("perception_regions", ()),
            )
            return {}, max(1, len(args["sections"])), rendered_str.encode("utf-8")
        raise ValueError("Generator operator is outside the deterministic contract")

    def _spec(self, snapshot, spec):
        if not isinstance(spec, dict):
            raise ValueError("Scratchboard specification must be an object")
        kind = spec.get("kind")
        fields = {"literal": {"kind", "text"}, "source": {"kind", "source", "start", "stop"},
                  "quantity": {"kind", "step_id", "language", "renderer"},
                  "relation": {"kind", "step_id", "open_step_ids", "language", "renderer"},
                  "generator": {"kind", "operator", "arguments", "renderer"},
                  "artifact": {"kind", "operation_id"}, "concat": {"kind", "nodes"},
                  "repeat": {"kind", "node", "count"}, "slots": {"kind", "count", "constraints", "bindings"},
                  "tensor": {"kind", "blob", "dtype", "shape", "basis"},
                  "judgment": {"kind", "question", "verdict", "reason", "uncertainty", "role", "predicate", "prediction", "evaluator"},
                  "import": {"kind", "node", "relevance", "reason"}}
        if kind not in fields or set(spec) != fields[kind]:
            raise ValueError("Scratchboard kind or fields are outside the contract")
        if kind == "literal":
            if not isinstance(spec["text"], str) or len(spec["text"].encode()) > 32768:
                raise ValueError("Literal text extent")
        elif kind == "source":
            if spec["source"] not in snapshot["sources"]:
                raise ValueError("Source span lies outside selected evidence")
            raw = self.memory.state["sources"][spec["source"]]["text"].encode()
            if (type(spec["start"]) is not int or type(spec["stop"]) is not int
                    or not 0 <= spec["start"] <= spec["stop"] <= len(raw)):
                raise ValueError("Source byte span extent")
            raw[spec["start"]:spec["stop"]].decode("utf-8")
        elif kind == "quantity":
            from . import documents
            if spec["renderer"] != file_digest(documents.__file__):
                raise StaleState("Calculation renderer changed")
            parent, _, _ = documents._quantity_components(self.memory, spec["step_id"], spec["language"])
            if any(parent[k] != snapshot[k] for k in ("task_id", "task_revision", "identity")):
                raise StaleState("Calculation belongs to another task")
            if self._bases(parent) - {snapshot["core_identity"]}:
                raise StaleState("Calculation numerical dependency has an incompatible model basis")
            snapshot.setdefault("observations", {}).update(parent["observations"])
        elif kind == "relation":
            from . import documents
            if spec["renderer"] != file_digest(documents.__file__):
                raise StaleState("Relation renderer changed")
            parent, _, _ = documents._relation_components(
                self.memory, spec["step_id"], spec["open_step_ids"], spec["language"])
            if any(parent[k] != snapshot[k] for k in ("task_id", "task_revision", "identity")):
                raise StaleState("Relation decision belongs to another task")
            if self._bases(parent) - {snapshot["core_identity"]}:
                raise StaleState("Relation numerical dependency has an incompatible model basis")
            snapshot.setdefault("observations", {}).update(parent["observations"])
        elif kind == "generator":
            bound_obs, _, _ = self._generator_components(snapshot, spec)
            if bound_obs:
                snapshot.setdefault("observations", {}).update(bound_obs)
        elif kind == "artifact":
            receipt = self.memory.state["receipts"][spec["operation_id"]]
            self.memory._verify_receipt(receipt)
            op = self.memory.state["operations"][spec["operation_id"]]
            parent = self.memory.state["proposals"][op["proposal_id"]]["snapshot"]
            self.memory._validate_snapshot(parent)
            if any(parent[k] != snapshot[k] for k in ("task_id", "task_revision", "identity")):
                raise StaleState("Archived artifact belongs to another task")
            if self._bases(parent) - {snapshot["core_identity"]}:
                raise StaleState("Archived artifact numerical dependency has an incompatible model basis")
        elif kind == "concat":
            if not isinstance(spec["nodes"], list) or not 1 <= len(spec["nodes"]) <= 128:
                raise ValueError("Concatenation extent")
        elif kind == "repeat":
            self._count(spec["count"])
        elif kind == "slots":
            self._count(spec["count"])
            if not isinstance(spec["constraints"], str) or not spec["constraints"].strip() or len(spec["constraints"].encode()) > 2048:
                raise ValueError("Slot constraints extent")
            bindings = spec["bindings"]
            if not isinstance(bindings, dict) or len(bindings) > MAX_BINDINGS:
                raise ValueError("Sparse binding extent")
            for key in bindings:
                if not isinstance(key, str) or not key.isascii() or not key.isdecimal() or key != str(int(key)) or not 0 <= int(key) < spec["count"]:
                    raise ValueError("Slot binding index")
        elif kind == "tensor":
            self._tensor_bytes(spec, snapshot["core_identity"])
        elif kind == "judgment":
            from .judgments import validate_spec
            validate_spec(self.memory, snapshot, spec, recording=True)
        elif kind == "import":
            bounded_text(spec["reason"], "reuse reason", 2048)
            parent = self._validate(spec["node"])
            self._compatible(parent, snapshot, cross_task=True)
            if any(snapshot["sources"].get(key) != value for key, value in parent["snapshot"]["sources"].items()):
                raise StaleState("Reuse needs the original sources in the current explicit selection")
            relevance = self._validate(spec["relevance"])
            if relevance["spec"]["kind"] != "judgment" or relevance["spec"]["verdict"] != "yes":
                raise ValueError("Reuse needs an explicit affirmative relevance judgment; its scope remains recorded")
        for child in self._children(spec):
            if not isinstance(child, str):
                raise ValueError("A symbolic reference needs a node identity")
            parent = self._validate(child)
            self._compatible(parent, snapshot)
            snapshot.setdefault("scratchboard", {})[child] = parent["digest"]

    def _prepare(self, snapshot, spec, *, origin=None):
        snapshot, spec = deepcopy(snapshot), deepcopy(spec)
        self.memory._validate_snapshot(snapshot)
        self._spec(snapshot, spec)
        self.memory._validate_snapshot(snapshot)
        body = {"spec": spec, "snapshot": snapshot, "implementation": self.implementation, "origin": origin}
        node = {"id": str(uuid.uuid4()), **body, "digest": digest_bytes(canonical(body))}
        return node

    def _record(self, snapshot, spec, *, origin=None):
        node = self._prepare(snapshot, spec, origin=origin)
        self.memory._append({"kind": "scratchboard.recorded", "node": node})
        return deepcopy(node)

    def add(self, snapshot, spec):
        """Record one bounded symbolic decision, without a model invocation."""
        with self.memory.locked():
            return self._record(snapshot, spec)

    def import_node(self, node_id, snapshot, *, relevance, reason):
        """Explicitly reuse a current node with both origin and current judgment."""
        with self.memory.locked():
            return self._record(snapshot, {"kind": "import", "node": node_id, "relevance": relevance, "reason": reason},
                                origin={"operation": "import", "parent": node_id})

    def quantity(self, step_id, *, language="en"):
        from . import documents
        with self.memory.locked():
            snapshot, _, _ = documents._quantity_components(self.memory, step_id, language)
            return self._record(snapshot, {"kind": "quantity", "step_id": step_id, "language": language,
                                          "renderer": file_digest(documents.__file__)})

    def relation(self, step_id, *, open_step_ids=(), language="en"):
        from . import documents
        with self.memory.locked():
            open_ids = list(open_step_ids)
            snapshot, _, _ = documents._relation_components(self.memory, step_id, open_ids, language)
            return self._record(snapshot, {"kind": "relation", "step_id": step_id,
                                          "open_step_ids": open_ids, "language": language,
                                          "renderer": file_digest(documents.__file__)})

    def generator(self, snapshot, operator, arguments):
        """Record a deterministic multi-unit expander without a model invocation."""
        args = deepcopy(arguments)
        if operator == "source_lines" and isinstance(args, dict) and "numbered" not in args:
            args["numbered"] = False
        if operator == "relation_table" and isinstance(args, dict) and "open_step_ids" not in args:
            args["open_step_ids"] = []
        with self.memory.locked():
            node = self._record(snapshot, {"kind": "generator", "operator": operator,
                                           "arguments": args, "renderer": self.implementation},
                                origin={"operation": "generator", "operator": operator})
            _, units, _ = self._generator_components(node["snapshot"], node["spec"])
            return {**node, "metrics": {"logical_units": units, "model_calls": 0}}

    def reserve(self, snapshot, count, *, constraints):
        return self.add(snapshot, {"kind": "slots", "count": count, "constraints": constraints, "bindings": {}})

    def bind(self, node_id, index, target_id):
        with self.memory.locked():
            node = self._validate(node_id)
            spec = deepcopy(node["spec"])
            if spec["kind"] != "slots" or type(index) is not int or not 0 <= index < spec["count"]:
                raise ValueError("Binding needs an existing slot index")
            if str(index) in spec["bindings"]:
                raise ValueError("A binding is immutable; create a separate candidate range")
            spec["bindings"][str(index)] = target_id
            return self._record(node["snapshot"], spec, origin={"operation": "bind", "parent": node_id, "index": index})

    def bind_many(self, node_id, bindings):
        """Bind multiple slot indices atomically in a single symbolic decision."""
        if isinstance(bindings, dict):
            pairs = list(bindings.items())
        elif isinstance(bindings, (list, tuple)):
            pairs = list(bindings)
        else:
            raise ValueError("Bulk bindings must be a mapping or sequence of (index, target_id)")
        if not 1 <= len(pairs) <= MAX_BINDINGS:
            raise ValueError("Bulk binding extent")
        with self.memory.locked():
            node = self._validate(node_id)
            spec = deepcopy(node["spec"])
            if spec["kind"] != "slots":
                raise ValueError("Bulk binding needs a slots node")
            indices = []
            for raw_idx, target_id in pairs:
                idx = int(raw_idx) if isinstance(raw_idx, str) and raw_idx.isascii() and raw_idx.isdecimal() else raw_idx
                if type(idx) is not int or not 0 <= idx < spec["count"]:
                    raise ValueError("Binding needs an existing slot index")
                key = str(idx)
                if key in spec["bindings"]:
                    raise ValueError("A binding is immutable; create a separate candidate range")
                spec["bindings"][key] = target_id
                indices.append(idx)
            return self._record(node["snapshot"], spec,
                                origin={"operation": "bind_many", "parent": node_id, "indices": indices})

    def branch(self, node_id, *, reason):
        """Keep an independent candidate with the same current representation."""
        bounded_text(reason, "branch reason", 2048)
        with self.memory.locked():
            node = self._validate(node_id)
            return self._record(node["snapshot"], node["spec"], origin={"operation": "branch", "parent": node_id, "reason": reason})

    def _depends_on(self, snapshot, target, visited, *, spec=None):
        if spec is not None:
            dependency = None
            if spec["kind"] == "import":
                dependency = {"scratchboard": {spec["node"]: None}}
            elif spec["kind"] == "artifact":
                op = self.memory.state["operations"][spec["operation_id"]]
                dependency = self.memory.state["proposals"][op["proposal_id"]]["snapshot"]
            elif spec["kind"] == "judgment" and spec["predicate"] is not None:
                dependency = {"observations": {spec["predicate"]["step_id"]: None}}
            if dependency is not None and self._depends_on(dependency, target, visited):
                return True
        for node_id in snapshot.get("scratchboard", {}):
            if node_id == target:
                return True
            if ("node", node_id) in visited:
                continue
            visited.add(("node", node_id))
            node = self.memory.state["scratchboard"][node_id]
            if self._depends_on(node["snapshot"], target, visited, spec=node["spec"]):
                return True
        for step_id in snapshot.get("observations", {}):
            if ("step", step_id) not in visited:
                visited.add(("step", step_id))
                if self._depends_on(self.memory.state["steps"][step_id]["snapshot"], target, visited):
                    return True
        return False

    def _revise(self, node_id, spec, reason, *, snapshot=None, exact_bytes=False, reshape=False):
        bounded_text(reason, "revision reason", 2048)
        old = self._validate(node_id) if snapshot is None else self.memory.state["scratchboard"][node_id]
        if node_id in self.memory.state["scratchboard_revisions"]:
            raise StaleState("Revise the current scratchboard version")
        snapshot = old["snapshot"] if snapshot is None else snapshot
        if any(old["snapshot"][k] != snapshot[k] for k in ("task_id", "identity")):
            raise StaleState("Revision keeps its task and continuing identity")
        operation = "refine" if exact_bytes else "reshape" if reshape else "revise"
        node = self._prepare(snapshot, spec, origin={"operation": operation, "parent": node_id, "reason": reason})
        depends_on_old = self._depends_on(node["snapshot"], node_id, set(), spec=node["spec"])
        if depends_on_old:
            raise ValueError("A replacement must not depend on the version it supersedes")
        if exact_bytes:
            before = self._stats(node_id, {})
            if before["unresolved_slots"] or before["text_bytes"] is None or before["text_bytes"] > 32768:
                raise ValueError("Byte-preserving refinement needs resolved text within the comparison budget")
            # A temporary projection permits the same executor to check both
            # representations before the single atomic revision event exists.
            self.memory.state["scratchboard"][node["id"]] = node
            try:
                after = self._stats(node["id"], {})
                if (after["unresolved_slots"] or after["text_bytes"] != before["text_bytes"]
                        or self._render(node_id, {}) != self._render(node["id"], {})):
                    raise ValueError("The refinement changes materialized bytes")
            finally:
                del self.memory.state["scratchboard"][node["id"]]
        revision = {"previous": node_id, "replacement": node["id"], "reason": reason,
                    "invariant": ("byte-identical materialization" if exact_bytes else
                                  "same tensor bytes, element order, dtype and basis" if reshape else
                                  "intentional representation replacement; meaning requires its own assessment")}
        self.memory._append({"kind": "scratchboard.revised", "node": node, "revision": revision})
        return deepcopy(node)

    def revise(self, node_id, spec, *, reason, snapshot=None):
        """Adopt a replacement; current dependents of the old version go stale."""
        with self.memory.locked():
            return self._revise(node_id, spec, reason, snapshot=snapshot)

    def refine(self, node_id, parts, *, reason):
        """Split or regroup resolved text with exact byte equivalence checked."""
        with self.memory.locked():
            return self._revise(node_id, {"kind": "concat", "nodes": list(parts)}, reason, exact_bytes=True)

    def latest(self, node_id):
        with self.memory.locked():
            visited = set()
            while node_id in self.memory.state["scratchboard_revisions"]:
                if node_id in visited:
                    raise ValueError("Cyclic revision history")
                visited.add(node_id)
                node_id = self.memory.state["scratchboard_revisions"][node_id]["replacement"]
            return deepcopy(self._validate(node_id))

    def reshape_tensor(self, node_id, shape, *, reason):
        with self.memory.locked():
            node = self._validate(node_id)
            if node["spec"]["kind"] != "tensor":
                raise ValueError("Reshape needs a tensor node")
            spec = deepcopy(node["spec"]); spec["shape"] = list(shape)
            return self._revise(node_id, spec, reason, reshape=True)

    def slot(self, node_id, index):
        with self.memory.locked():
            node = self._validate(node_id)
            spec = node["spec"]
            if spec["kind"] != "slots" or type(index) is not int or not 0 <= index < spec["count"]:
                raise ValueError("Slot lookup extent")
            target = spec["bindings"].get(str(index))
            return {"range": node_id, "index": index, "state": "bound" if target else "unresolved",
                    "node": target, "constraints": spec["constraints"], "model_calls": 0}

    def tensor(self, snapshot, data, *, shape, dtype="bf16", origin=None):
        if not isinstance(data, bytes) or not 1 <= len(data) <= MAX_TENSOR_BYTES:
            raise ValueError("Tensor payload extent")
        blob = digest_bytes(data).split(":")[1]
        spec = {"kind": "tensor", "blob": blob, "dtype": dtype, "shape": list(shape), "basis": snapshot["core_identity"]}
        self._tensor_shape(spec)
        if math.prod(spec["shape"]) * {"u8": 1, "bf16": 2, "f32": 4}[dtype] != len(data):
            raise ValueError("Tensor shape and byte count differ")
        with self.memory.locked():
            self.memory._validate_snapshot(snapshot)
            path = self.memory._artifact_path(f"scratchboard/blobs/{blob}.bin")
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.memory._artifact_path(f"scratchboard/blobs/{blob}.bin")
            temporary = None
            try:
                if not path.exists():
                    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".staging-", delete=False) as stream:
                        temporary = Path(stream.name)
                        stream.write(data); stream.flush(); os.fsync(stream.fileno())
                    os.link(temporary, path)
                    self.memory.fs.sync_directory(path.parent)
                return self._record(snapshot, spec, origin=origin)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    @staticmethod
    def _tensor_shape(spec):
        shape = spec["shape"]
        if (spec["dtype"] not in {"u8", "bf16", "f32"} or not isinstance(shape, list) or not 1 <= len(shape) <= 8
                or any(type(x) is not int or not 1 <= x <= MAX_TENSOR_BYTES for x in shape)
                or math.prod(shape) * {"u8": 1, "bf16": 2, "f32": 4}[spec["dtype"]] > MAX_TENSOR_BYTES):
            raise ValueError("Tensor dtype or shape extent")

    def _tensor_bytes(self, spec, core_identity):
        self._tensor_shape(spec)
        if spec["basis"] != core_identity:
            raise StaleState("Tensor model basis differs")
        blob = spec["blob"]
        if not isinstance(blob, str) or len(blob) != 128 or any(c not in "0123456789abcdef" for c in blob):
            raise ValueError("Tensor blob identity")
        path = self.memory._artifact_path(f"scratchboard/blobs/{blob}.bin")
        expected = math.prod(spec["shape"]) * {"u8": 1, "bf16": 2, "f32": 4}[spec["dtype"]]
        if path.stat().st_size != expected:
            raise ValueError("Tensor bytes differ from the declared shape")
        raw = path.read_bytes()
        if digest_bytes(raw) != "shake256:" + blob:
            raise ValueError("Tensor payload identity differs")
        return raw

    def read_tensor(self, node_id, *, core_identity):
        with self.memory.locked():
            node = self._validate(node_id)
            if node["spec"]["kind"] != "tensor":
                raise ValueError("Node does not hold tensor data")
            return self._tensor_bytes(node["spec"], core_identity)

    def _verify_relation_tensor_lineage(self, snapshot, origin, model):
        """Verify that origin step_id and open_step_ids exist, are active/unrejected, match current sources/snapshot, and bind every question expression to the carried relation state."""
        from .relations import validate_model
        if not isinstance(origin, dict) or origin.get("operation") != "relation.tensor":
            raise ValueError("Relation tensor requires relational origin metadata")
        step_id = origin.get("step_id")
        if not isinstance(step_id, str) or not step_id:
            raise ValueError("Relation tensor requires an observed relational step_id")
        open_ids = origin.get("open_step_ids", [])
        if (not isinstance(open_ids, list) or len(open_ids) > 16
                or len(set(open_ids)) != len(open_ids) or step_id in open_ids
                or any(not isinstance(oid, str) or not oid for oid in open_ids)):
            raise ValueError("Open relational question step selection")

        def _check_active_step(sid):
            s = self.memory.state["steps"].get(sid)
            o = self.memory.state["step_observations"].get(sid)
            if not s or not o:
                raise ValueError("Relation tensor step_id does not exist or is unobserved")
            rev = self.memory.state["step_reviews"].get(sid)
            if rev and rev["decision"] in {"rejected", "cancelled"}:
                raise StaleState("Relation tensor lineage references a rejected or cancelled step")
            self.memory._validate_snapshot(s["snapshot"])
            if any(s["snapshot"].get(k) != snapshot.get(k)
                   for k in ("task_id", "task_revision", "identity", "core_identity", "sources")):
                raise StaleState("Relation tensor step belongs to another task, basis or source revision")
            return s, o

        step, obs = _check_active_step(step_id)
        if step["operator"] == "relation.query":
            state_id = step["arguments"]["step"]
            _, state_obs = _check_active_step(state_id)
            expected_model = validate_model(deepcopy(state_obs["result"].get("value")))
            expected_questions = [deepcopy(step["arguments"]["expression"])]
        else:
            state_id = step_id
            expected_model = validate_model(deepcopy(obs["result"].get("value")))
            expected_questions = []
        for oid in open_ids:
            open_step, _ = _check_active_step(oid)
            if open_step["operator"] != "relation.query":
                raise ValueError("Open question carrier requires observed relation.query steps")
            if open_step["arguments"].get("step") != state_id:
                raise ValueError("Open question step belongs to a different relational state")
            expected_questions.append(deepcopy(open_step["arguments"]["expression"]))
        if model != expected_model:
            raise ValueError("Decoded relation tensor model disagrees with observed step state")
        actual_questions = [q["expression"] for q in origin.get("questions", [])]
        if actual_questions != expected_questions:
            raise ValueError("Relation tensor questions are not bound to observed relation.query steps")

    def relation_tensor(self, step_id, *, open_step_ids=(), mode="bf16_support"):
        """Carry a relational state and its open questions as an exact BF16 or compact U8 bit/trit tensor."""
        from .relations import encode_relation_tensor, validate_model
        open_ids = list(open_step_ids)
        if len(open_ids) > 16 or len(set(open_ids)) != len(open_ids) or step_id in open_ids:
            raise ValueError("Open relational question step selection")
        with self.memory.locked():
            step = self.memory.state["steps"].get(step_id)
            obs = self.memory.state["step_observations"].get(step_id)
            if not step or not obs:
                raise ValueError("Relation tensor requires an observed relational step")
            rev = self.memory.state["step_reviews"].get(step_id)
            if rev and rev["decision"] in {"rejected", "cancelled"}:
                raise StaleState("Relation tensor step was rejected or cancelled")
            if step["operator"] == "relation.query":
                state_id = step["arguments"]["step"]
                state_obs = self.memory.state["step_observations"].get(state_id)
                if not state_obs:
                    raise ValueError("Relation query parent state is unobserved")
                state_rev = self.memory.state["step_reviews"].get(state_id)
                if state_rev and state_rev["decision"] in {"rejected", "cancelled"}:
                    raise StaleState("Relation query parent state was rejected or cancelled")
                model = validate_model(deepcopy(state_obs["result"].get("value")))
                questions = [deepcopy(step["arguments"]["expression"])]
                bound_ids = [state_id, step_id]
            else:
                state_id = step_id
                model = validate_model(deepcopy(obs["result"].get("value")))
                questions = []
                bound_ids = [step_id]
            for oid in open_ids:
                open_step = self.memory.state["steps"].get(oid)
                open_obs = self.memory.state["step_observations"].get(oid)
                if not open_step or not open_obs or open_step["operator"] != "relation.query":
                    raise ValueError("Open question carrier requires observed relation.query steps")
                open_rev = self.memory.state["step_reviews"].get(oid)
                if open_rev and open_rev["decision"] in {"rejected", "cancelled"}:
                    raise StaleState("Open question step was rejected or cancelled")
                if open_step["arguments"].get("step") != state_id:
                    raise ValueError("Open question step belongs to a different relational state")
                if any(open_step["snapshot"][k] != step["snapshot"][k]
                       for k in ("task_id", "task_revision", "identity", "core_identity", "sources")):
                    raise StaleState("Open question step belongs to another task or basis")
                questions.append(deepcopy(open_step["arguments"]["expression"]))
                bound_ids.append(oid)
            snapshot = deepcopy(step["snapshot"])
            snapshot.setdefault("observations", {}).update({
                sid: self.memory.state["step_observations"][sid]["digest"] for sid in bound_ids})
            self.memory._validate_snapshot(snapshot)
            raw, shape, meta = encode_relation_tensor(model, mode=mode, open_questions=questions)
        return self.tensor(snapshot, raw, shape=shape, dtype=meta["dtype"],
                           origin={**meta, "step_id": step_id, "open_step_ids": open_ids})

    def decode_relation_tensor(self, node_id, *, core_identity=None):
        """Reconstruct and verify the finite relation model and step lineage from a scratchboard tensor node."""
        from .relations import decode_relation_tensor
        with self.memory.locked():
            node = self._validate(node_id)
            if node["spec"]["kind"] != "tensor":
                raise ValueError("Node does not hold tensor data")
            basis = node["spec"]["basis"] if core_identity is None else core_identity
            raw = self._tensor_bytes(node["spec"], basis)
            origin = deepcopy(node["origin"])
            spec_dtype = node["spec"]["dtype"]
            spec_shape = list(node["spec"]["shape"])
            snapshot = deepcopy(node["snapshot"])
            expected_dtype = (
                "u8"
                if isinstance(origin, dict) and origin.get("mode") in {"pack5", "support2", "bit_plane", "indicator_2n"}
                else "bf16"
            )
            if spec_dtype != expected_dtype:
                raise ValueError("Scratchboard tensor dtype disagrees with relational tensor mode")
            model = decode_relation_tensor(raw, spec_shape, origin)
            self._verify_relation_tensor_lineage(snapshot, origin, model)
        questions = deepcopy(origin["questions"])
        settled = [{"question": q["expression"], "verdict": q["verdict"], "box_verdict": q["box_verdict"],
                    "correlation_loss": q["correlation_loss"]}
                   for q in questions if q["verdict"] in ("yes", "no")]
        open_q = [{"question": q["expression"], "verdict": q["verdict"], "box_verdict": q["box_verdict"],
                   "correlation_loss": q["correlation_loss"]}
                  for q in questions if q["verdict"] == "maybe"]
        return {
            "node_id": node_id,
            "model": model,
            "variables": list(model["variables"]),
            "worlds": [list(r) for r in model["worlds"]],
            "mode": origin["mode"],
            "dtype": spec_dtype,
            "byte_count": len(raw),
            "box_marginals": origin["box_marginals"],
            "questions": questions,
            "settled_questions": settled,
            "open_questions": open_q,
            "packed_box_trits_hex": origin["packed_box_trits_hex"],
            "packed_box_support2_hex": origin["packed_box_support2_hex"],
            "step_id": origin["step_id"],
            "open_step_ids": list(origin.get("open_step_ids", [])),
        }

    def archive_tensors(self, node_ids, *, recipe="scratchboard-tensor-archive-v1", parent=None):
        """Package validated scratchboard tensors into self-describing safetensors and register in SQLite TensorStore."""
        import json
        from .relations import decode_relation_tensor
        from .tensor_store import TensorStore, pack_safetensors
        node_ids = list(node_ids)
        if not 1 <= len(node_ids) <= 64 or len(set(node_ids)) != len(node_ids):
            raise ValueError("Select one to 64 distinct tensor nodes")
        bounded_text(recipe, "tensor archive recipe", 256)
        with self.memory.locked():
            nodes = [self._validate(nid) for nid in node_ids]
            if any(n["spec"]["kind"] != "tensor" for n in nodes):
                raise ValueError("Tensor archive requires tensor scratchboard nodes")
            snapshot = deepcopy(nodes[0]["snapshot"])
            for n in nodes:
                self._compatible(n, snapshot)
                snapshot.setdefault("scratchboard", {})[n["id"]] = n["digest"]
            self.memory._validate_snapshot(snapshot)
            tensors = {}
            node_origins = {}
            for i, n in enumerate(nodes):
                raw = self._tensor_bytes(n["spec"], snapshot["core_identity"])
                origin = deepcopy(n.get("origin"))
                if isinstance(origin, dict) and origin.get("operation") == "relation.tensor":
                    expected_dtype = "u8" if origin.get("mode") in {"pack5", "support2", "bit_plane", "indicator_2n"} else "bf16"
                    if n["spec"]["dtype"] != expected_dtype:
                        raise ValueError("Scratchboard tensor dtype disagrees with relational tensor mode")
                    model = decode_relation_tensor(raw, n["spec"]["shape"], origin)
                    self._verify_relation_tensor_lineage(snapshot, origin, model)
                name = f"t{i:02d}_{n['id'].replace('-', '')[:12]}"
                tensors[name] = {"dtype": n["spec"]["dtype"].upper(),
                                 "shape": list(n["spec"]["shape"]), "data": raw}
                if isinstance(origin, dict):
                    canonical_origin_bytes = json.dumps(origin, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    node_origins[name] = {
                        **origin,
                        "node_id": n["id"],
                        "origin_digest": digest_bytes(canonical_origin_bytes),
                    }
                else:
                    node_origins[name] = {"node_id": n["id"], "origin": origin}
            origins_json = json.dumps(node_origins, sort_keys=True, separators=(",", ":"))
            container = pack_safetensors(tensors, metadata={
                "schema": "nanoai.scratchboard-tensor-archive.v1",
                "task_id": str(snapshot["task_id"]),
                "core_identity": str(snapshot["core_identity"]),
                "node_ids": json.dumps(node_ids),
                "node_origins": origins_json,
                "node_origins_digest": digest_bytes(origins_json.encode("utf-8")),
            })
            artifact_id = digest_bytes(container)
            hex_id = artifact_id.split(":")[1]
            path = self.memory._artifact_path(f"scratchboard/archives/{hex_id}.safetensors")
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.memory._artifact_path(f"scratchboard/archives/{hex_id}.safetensors")
            temporary = None
            try:
                if not path.exists():
                    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".staging-", delete=False) as stream:
                        temporary = Path(stream.name)
                        stream.write(container); stream.flush(); os.fsync(stream.fileno())
                    os.link(temporary, path)
                    self.memory.fs.sync_directory(path.parent)
                if file_digest(path) != artifact_id:
                    raise ValueError("Published safetensors archive bytes differ")
                core_id = str(snapshot["core_identity"])
                basis_id = (core_id if core_id.startswith("shake256:") and len(core_id) == 136
                            else digest_bytes(core_id.encode()))
                store = TensorStore(self.memory.root / "tensors.sqlite3", create=True)
                ref = store.register(path, expected=artifact_id, basis=basis_id,
                                     recipe=recipe, source="scratchboard:" + ",".join(node_ids),
                                     parent=parent)
                ref["database"] = "tensors.sqlite3"
                resolved = store.resolve(ref["version"], path=path)
                return {"reference": ref, "version_id": ref["version"], "recipe": recipe,
                        "artifact": artifact_id, "basis": basis_id,
                        "path": str(path), "archive_path": str(path), "node_ids": node_ids,
                        "tensor_count": resolved["tensor_count"], "tensors": resolved["tensors"]}
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    @staticmethod
    def decode_archive_file(archive_path, *, query=None):
        """Self-describing archive decoder: reconstruct and optionally query relational tensors from a .safetensors file alone without the journal."""
        import json
        from .relations import decode_relation_tensor, query_tensor_enclosure
        from .tensor_store import unpack_safetensors
        unpacked = unpack_safetensors(archive_path)
        meta = unpacked["metadata"]
        origins_raw = meta.get("node_origins", "{}")
        if "node_origins_digest" in meta:
            if digest_bytes(origins_raw.encode("utf-8")) != meta["node_origins_digest"]:
                raise ValueError("Safetensors node_origins_digest mismatch")
        origins_map = json.loads(origins_raw)
        if not isinstance(origins_map, dict):
            raise ValueError("Safetensors node_origins metadata must be a JSON object")
        tensors_by_key = {}
        decoded_list = []
        for name, t_spec in sorted(unpacked["tensors"].items()):
            entry = origins_map.get(name, {})
            if not isinstance(entry, dict):
                raise ValueError("Safetensors tensor origin entry must be an object")
            node_id = entry.get("node_id")
            origin = entry if entry.get("operation") == "relation.tensor" else entry.get("origin")
            dtype_lower = t_spec["dtype"].lower()
            item = {
                "tensor_name": name,
                "node_id": node_id,
                "dtype": t_spec["dtype"],
                "shape": list(t_spec["shape"]),
                "byte_count": len(t_spec["data"]),
            }
            if isinstance(origin, dict) and origin.get("operation") == "relation.tensor":
                if "origin_digest" in origin:
                    clean_origin = {k: v for k, v in origin.items() if k not in {"node_id", "origin_digest"}}
                    canonical = json.dumps(clean_origin, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    if digest_bytes(canonical) != origin["origin_digest"]:
                        raise ValueError("Safetensors relational origin_digest mismatch")
                expected_dtype = "u8" if origin.get("mode") in {"pack5", "support2", "bit_plane", "indicator_2n"} else "bf16"
                if dtype_lower != expected_dtype:
                    raise ValueError("Archived tensor dtype disagrees with relational tensor mode")
                model = decode_relation_tensor(t_spec["data"], t_spec["shape"], origin)
                questions = deepcopy(origin["questions"])
                rel_info = {
                    "model": model,
                    "variables": list(model["variables"]),
                    "worlds": [list(r) for r in model["worlds"]],
                    "mode": origin["mode"],
                    "box_marginals": origin["box_marginals"],
                    "questions": questions,
                    "settled_questions": [
                        {"question": q["expression"], "verdict": q["verdict"], "box_verdict": q["box_verdict"],
                         "correlation_loss": q["correlation_loss"]}
                        for q in questions if q["verdict"] in ("yes", "no")
                    ],
                    "open_questions": [
                        {"question": q["expression"], "verdict": q["verdict"], "box_verdict": q["box_verdict"],
                         "correlation_loss": q["correlation_loss"]}
                        for q in questions if q["verdict"] == "maybe"
                    ],
                }
                if query is not None:
                    q_res = {**query_tensor_enclosure(model, query)}
                    q_res.setdefault("verdict", q_res.get("exact_verdict"))
                    rel_info["query_result"] = q_res
                    item["query"] = q_res
                item["relation"] = rel_info
            decoded_list.append(item)
            tensors_by_key[node_id or name] = item
        return {
            "metadata": meta,
            "tensor_count": len(decoded_list),
            "tensors": tensors_by_key,
            "tensor_list": decoded_list,
        }

    def decode_archived_tensors(self, version_id, *, core_identity=None, query=None):
        """Verify and decode relational tensors (including compact U8 and BF16) from a TensorStore archive."""
        from .tensor_store import TensorStore
        store = TensorStore(self.memory.root / "tensors.sqlite3")
        resolved = store.resolve(version_id)
        decoded = self.decode_archive_file(resolved["path"], query=query)
        meta = decoded["metadata"]
        core_id = str(meta.get("core_identity") or "") if core_identity is None else str(core_identity)
        if core_id:
            expected_basis = (core_id if core_id.startswith("shake256:") and len(core_id) == 136
                              else digest_bytes(core_id.encode()))
            if resolved["basis"] != expected_basis:
                raise StaleState("Archived tensor basis differs from requested core_identity")
        if core_identity is not None and meta.get("core_identity") and meta["core_identity"] != str(core_identity):
            raise StaleState("Safetensors container core_identity differs")
        return {
            "version_id": resolved.get("id", version_id),
            "artifact": resolved["artifact"],
            "recipe": resolved["recipe"],
            "basis": resolved["basis"],
            "path": resolved["path"],
            "tensor_count": decoded["tensor_count"],
            "tensors": decoded["tensors"],
            "tensor_list": decoded["tensor_list"],
        }


    def _validate(self, node_id, *, _depth=0, _checked=None):
        if _depth > 16:
            raise StaleState("Scratchboard dependency depth exceeds the contract")
        node = self.memory.state["scratchboard"].get(node_id)
        if not node:
            raise StaleState("Scratchboard dependency is unavailable")
        if node_id in self.memory.state["scratchboard_revisions"]:
            raise StaleState("Scratchboard dependency was revised; reconsider its dependent work")
        checked = set() if _checked is None else _checked
        if node_id in checked:
            return node
        body = {k: node[k] for k in ("spec", "snapshot", "implementation", "origin")}
        if digest_bytes(canonical(body)) != node["digest"] or node["implementation"] != self.implementation:
            raise StaleState("Scratchboard identity or implementation changed")
        self.memory._validate_snapshot(node["snapshot"], _depth=_depth, _board_checked=checked)
        spec = node["spec"]
        if spec["kind"] == "quantity":
            from . import documents
            if spec["renderer"] != file_digest(documents.__file__):
                raise StaleState("Calculation renderer changed")
            review = self.memory.state["step_reviews"].get(spec["step_id"])
            if not review or review["decision"] != "confirmed":
                raise StaleState("Calculation review is unavailable")
        elif spec["kind"] == "relation":
            from . import documents
            if spec["renderer"] != file_digest(documents.__file__):
                raise StaleState("Relation renderer changed")
            review = self.memory.state["step_reviews"].get(spec["step_id"])
            if not review or review["decision"] != "confirmed":
                raise StaleState("Relation review is unavailable")
            documents._relation_components(self.memory, spec["step_id"], spec["open_step_ids"], spec["language"])
        elif spec["kind"] == "generator":
            self._generator_components(node["snapshot"], spec)
        elif spec["kind"] == "tensor":
            self._tensor_bytes(spec, node["snapshot"]["core_identity"])
        elif spec["kind"] == "judgment":
            from .judgments import validate_spec
            validate_spec(self.memory, node["snapshot"], spec)
        elif spec["kind"] == "import":
            parent = self._validate(spec["node"], _depth=_depth+1, _checked=checked)
            self._compatible(parent, node["snapshot"], cross_task=True)
        elif spec["kind"] == "artifact":
            receipt = self.memory.state["receipts"][spec["operation_id"]]
            self.memory._verify_receipt(receipt)
            op = self.memory.state["operations"][spec["operation_id"]]
            self.memory._validate_snapshot(self.memory.state["proposals"][op["proposal_id"]]["snapshot"], _depth=_depth+1, _board_checked=checked)
        checked.add(node_id)
        return node

    def _stats(self, node_id, cache):
        if node_id in cache:
            return cache[node_id]
        node = self.memory.state["scratchboard"][node_id]
        spec = node["spec"]; kind = spec["kind"]
        children = [self._stats(k, cache) for k in self._children(spec)]
        unresolved = sum(c["unresolved_slots"] for c in children)
        units = sum(c["logical_units"] for c in children) if children else 1
        size = None
        if kind == "literal": size = len(spec["text"].encode())
        elif kind == "source": size = spec["stop"] - spec["start"]
        elif kind == "quantity":
            from .documents import _quantity_components
            size = len(_quantity_components(self.memory, spec["step_id"], spec["language"])[1].encode())
        elif kind == "relation":
            from .documents import _relation_components
            size = len(_relation_components(self.memory, spec["step_id"], spec["open_step_ids"], spec["language"])[1].encode())
        elif kind == "generator":
            if spec["operator"] == "slot_sequence":
                if not unresolved and all(c["text_bytes"] is not None for c in children):
                    _, units, raw = self._generator_components(node["snapshot"], spec, cache={})
                    size = len(raw)
            else:
                _, units, raw = self._generator_components(node["snapshot"], spec)
                size = len(raw)
        elif kind == "artifact":
            op = self.memory.state["operations"][spec["operation_id"]]
            size = len(self.memory.state["proposals"][op["proposal_id"]]["output"]["content"].encode())
        elif kind == "import":
            source = self._stats(spec["node"], cache)
            size, units, unresolved = source["text_bytes"], source["logical_units"], source["unresolved_slots"]
        elif kind == "repeat":
            unresolved *= spec["count"]; units *= spec["count"]
            if children[0]["text_bytes"] is not None: size = children[0]["text_bytes"] * spec["count"]
        elif kind in {"concat", "slots"}:
            if kind == "slots":
                units = spec["count"]
                unresolved += spec["count"] - len(spec["bindings"])
            if not unresolved and all(c["text_bytes"] is not None for c in children): size = sum(c["text_bytes"] for c in children)
        result = {"logical_units": units, "unresolved_slots": unresolved, "text_bytes": size}
        cache[node_id] = result
        return result

    def describe(self, node_id):
        with self.memory.locked():
            node = self._validate(node_id)
            cache = {}
            extra = ({"dtype": node["spec"]["dtype"].upper(), "shape": list(node["spec"]["shape"])}
                     if node["spec"]["kind"] == "tensor" else {})
            return {"id": node_id, "kind": node["spec"]["kind"], **extra, **self._stats(node_id, cache),
                    "stored_node_bytes": len(canonical(node)), "reachable_nodes": len(cache),
                    "model_calls": 0, "content_status": "Symbolic addresses and bindings; unresolved slots have no invented content"}

    def _entry(self, node_id):
        """A bounded dictionary view while the caller holds the journal lock."""
        node = self._validate(node_id)
        stats = self._stats(node_id, {})
        spec = deepcopy(node["spec"])
        representation_complete = True
        if spec["kind"] == "literal":
            raw = spec["text"].encode()
            if len(raw) > 1024:
                del spec["text"]
                spec["text_prefix"] = raw[:1024].decode("utf-8", errors="ignore")
                representation_complete = False
        preview = None
        if not stats["unresolved_slots"] and stats["text_bytes"] is not None and stats["text_bytes"] <= 2048:
            preview = self._render(node_id, {}).decode("utf-8")
        return {"kind": "working_unit", "id": node_id, "digest": node["digest"],
                "representation": spec, "representation_complete": representation_complete,
                "origin": deepcopy(node["origin"]), **stats,
                "preview": preview, "preview_complete": preview is not None,
                "meaning": "Current exact representation; task relevance and any semantic change need assessment"}

    def _render(self, node_id, cache):
        if node_id in cache:
            return cache[node_id]
        node = self.memory.state["scratchboard"][node_id]
        spec = node["spec"]; kind = spec["kind"]
        if kind == "literal": raw = spec["text"].encode()
        elif kind == "source": raw = self.memory.state["sources"][spec["source"]]["text"].encode()[spec["start"]:spec["stop"]]
        elif kind == "quantity":
            from .documents import _quantity_components
            raw = _quantity_components(self.memory, spec["step_id"], spec["language"])[1].encode()
        elif kind == "relation":
            from .documents import _relation_components
            raw = _relation_components(self.memory, spec["step_id"], spec["open_step_ids"], spec["language"])[1].encode()
        elif kind == "generator":
            _, _, raw = self._generator_components(node["snapshot"], spec, cache=cache)
        elif kind == "artifact":
            receipt = self.memory.state["receipts"][spec["operation_id"]]
            raw = self.memory._artifact_path(receipt["path"]).read_bytes()
            if digest_bytes(raw) != receipt["digest"]: raise StaleState("Archived bytes changed during expansion")
        elif kind == "import": raw = self._render(spec["node"], cache)
        elif kind == "concat": raw = b"".join(self._render(k, cache) for k in spec["nodes"])
        elif kind == "repeat": raw = self._render(spec["node"], cache) * spec["count"]
        elif kind == "slots": raw = b"".join(self._render(spec["bindings"][str(i)], cache) for i in range(spec["count"]))
        else: raise ValueError("Typed numerical or judgment data needs an explicit text-realization decision")
        if kind in {"literal", "source", "quantity", "relation", "generator", "artifact"}:
            cache[node_id] = raw
        return raw

    def materialize(self, node_id, *, max_bytes=32768):
        self._count(max_bytes, MAX_MATERIALIZED_BYTES)
        start = time.perf_counter_ns()
        with self.memory.locked():
            node = self._validate(node_id)
            stats = self._stats(node_id, {})
            if stats["unresolved_slots"]:
                raise UnresolvedSlot("The requested complete output contains unresolved slots")
            if stats["text_bytes"] is None:
                raise ValueError("A tensor needs a compatible text-realization decision")
            if stats["text_bytes"] > max_bytes:
                raise ValueError("Materialization exceeds the explicit byte budget")
            raw = self._render(node_id, {})
            if len(raw) != stats["text_bytes"]:
                raise ValueError("Materialized byte count differs")
            return {"node_id": node_id, "bytes": raw, "digest": digest_bytes(raw), "snapshot": deepcopy(node["snapshot"]),
                    "elapsed_ns": time.perf_counter_ns()-start, "model_calls": 0, **stats}

    def audit_visual(self, node_id, *, perception_regions=None, require_title=True):
        """Materialize a scratchboard node and audit its output against VISUAL_RULES."""
        from .documents import audit_visual_presentation
        rendered = self.materialize(node_id)
        snapshot = rendered["snapshot"]
        with self.memory.locked():
            allowed_refs = [
                f"{src_id}@{self.memory.state['sources'][src_id]['revision']}"
                for src_id in snapshot.get("sources", {})
                if src_id in self.memory.state["sources"]
            ]
        content = rendered["bytes"].decode("utf-8")
        audit = audit_visual_presentation(
            content,
            allowed_sources=allowed_refs,
            perception_regions=perception_regions,
            require_title=require_title,
        )
        return {**audit, "node_id": node_id, "digest": rendered["digest"], "model_calls": 0}

    def propose_document(self, node_id, *, filename=None, enforce_visual_rules=False, perception_regions=None):
        rendered = self.materialize(node_id)
        snapshot = rendered["snapshot"]
        with self.memory.locked():
            node = self._validate(node_id)
            snapshot.setdefault("scratchboard", {})[node_id] = node["digest"]
            self.memory._validate_snapshot(snapshot)
            task = self.memory.state["tasks"][snapshot["task_id"]]
            if filename is None:
                if len(task["filenames"]) != 1:
                    raise ValueError("Choose one of the task's document filenames")
                filename = task["filenames"][0]
            allowed_refs = [
                f"{src_id}@{self.memory.state['sources'][src_id]['revision']}"
                for src_id in snapshot.get("sources", {})
                if src_id in self.memory.state["sources"]
            ]
        content = rendered["bytes"].decode("utf-8")
        if enforce_visual_rules:
            from .documents import audit_visual_presentation
            audit = audit_visual_presentation(
                content,
                allowed_sources=allowed_refs,
                perception_regions=perception_regions,
            )
            if not audit["passed"]:
                raise ValueError(f"Document failed visual presentation rules: {audit['violations']}")
        return self.memory.record_proposal(snapshot, {"filename": filename, "content": content,
                "citations": sorted(document_references(content))}, metrics={"renderer": "scratchboard-v1",
                "scratchboard_node": node_id, "renderer_identity": self.implementation,
                "render_model_calls": 0, "materialization_ns": rendered["elapsed_ns"]})

