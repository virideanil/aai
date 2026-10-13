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

    def _spec(self, snapshot, spec):
        if not isinstance(spec, dict):
            raise ValueError("Scratchboard specification must be an object")
        kind = spec.get("kind")
        fields = {"literal": {"kind", "text"}, "source": {"kind", "source", "start", "stop"},
                  "quantity": {"kind", "step_id", "language", "renderer"},
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
        if math.prod(spec["shape"]) * {"bf16": 2, "f32": 4}[dtype] != len(data):
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
        if (spec["dtype"] not in {"bf16", "f32"} or not isinstance(shape, list) or not 1 <= len(shape) <= 8
                or any(type(x) is not int or not 1 <= x <= MAX_TENSOR_BYTES for x in shape)
                or math.prod(shape) * {"bf16": 2, "f32": 4}[spec["dtype"]] > MAX_TENSOR_BYTES):
            raise ValueError("Tensor dtype or shape extent")

    def _tensor_bytes(self, spec, core_identity):
        self._tensor_shape(spec)
        if spec["basis"] != core_identity:
            raise StaleState("Tensor model basis differs")
        blob = spec["blob"]
        if not isinstance(blob, str) or len(blob) != 128 or any(c not in "0123456789abcdef" for c in blob):
            raise ValueError("Tensor blob identity")
        path = self.memory._artifact_path(f"scratchboard/blobs/{blob}.bin")
        expected = math.prod(spec["shape"]) * {"bf16": 2, "f32": 4}[spec["dtype"]]
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
            return {"id": node_id, "kind": node["spec"]["kind"], **self._stats(node_id, cache),
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
        spec = self.memory.state["scratchboard"][node_id]["spec"]; kind = spec["kind"]
        if kind == "literal": raw = spec["text"].encode()
        elif kind == "source": raw = self.memory.state["sources"][spec["source"]]["text"].encode()[spec["start"]:spec["stop"]]
        elif kind == "quantity":
            from .documents import _quantity_components
            raw = _quantity_components(self.memory, spec["step_id"], spec["language"])[1].encode()
        elif kind == "artifact":
            receipt = self.memory.state["receipts"][spec["operation_id"]]
            raw = self.memory._artifact_path(receipt["path"]).read_bytes()
            if digest_bytes(raw) != receipt["digest"]: raise StaleState("Archived bytes changed during expansion")
        elif kind == "import": raw = self._render(spec["node"], cache)
        elif kind == "concat": raw = b"".join(self._render(k, cache) for k in spec["nodes"])
        elif kind == "repeat": raw = self._render(spec["node"], cache) * spec["count"]
        elif kind == "slots": raw = b"".join(self._render(spec["bindings"][str(i)], cache) for i in range(spec["count"]))
        else: raise ValueError("Typed numerical or judgment data needs an explicit text-realization decision")
        if kind in {"literal", "source", "quantity", "artifact"}:
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

    def propose_document(self, node_id, *, filename=None):
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
        content = rendered["bytes"].decode("utf-8")
        return self.memory.record_proposal(snapshot, {"filename": filename, "content": content,
                "citations": sorted(document_references(content))}, metrics={"renderer": "scratchboard-v1",
                "scratchboard_node": node_id, "renderer_identity": self.implementation,
                "render_model_calls": 0, "materialization_ns": rendered["elapsed_ns"]})
