"""Versioned source/task memory on the maintained SJQL transaction journal.

SJQL owns durable segments, locks, integrity and head publication. This module
owns nano'ai's domain events and reconstructible current-state projection.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import uuid

from .core import digest_bytes, file_digest


class StaleState(ValueError):
    """The proposal's dependencies changed during its computation."""


class ProposalError(ValueError):
    """A model proposal failed the owned artifact contract."""


def sjql_modules():
    source = os.environ.get("NANOAI_SJQL_SRC")
    if source:
        path = Path(source).resolve(strict=True)
        if not (path / "sjql/core/transaction_journal.py").is_file():
            raise ValueError("NANOAI_SJQL_SRC must name the maintained package's src directory")
        sys.path.insert(0, str(path))
    try:
        return (importlib.import_module("sjql.core.transaction_journal"),
                importlib.import_module("sjql.core.filesystem"))
    except ModuleNotFoundError as error:
        raise RuntimeError("Install the maintained SJQL source package or set NANOAI_SJQL_SRC to its src directory") from error


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", text.casefold(), re.UNICODE))


def document_references(text: str) -> set[str]:
    """Collect every reference candidate in the bounded document grammar.

    Bracketed name@value forms are reserved for source references. Collect
    malformed revisions too, so the proposal boundary can reject them.
    """
    return set(re.findall(r"\[([A-Za-z0-9][A-Za-z0-9_.-]*\s*@[^\]]+)\]", text))


def bounded_text(value, label, limit=32768):
    if not isinstance(value, str) or not value.strip() or len(value.encode()) > limit:
        raise ValueError(f"Invalid {label} text or extent")
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError("Identifier must be a short ASCII name")
    return value


class Memory:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.journal, self.fs = sjql_modules()
        self.segments = self.root / "transactions.sjql.d"
        self.segments.mkdir(exist_ok=True, mode=0o700)
        self.recovery_issues = []
        self.reload()

    def reload(self):
        self.head = self.journal.head(self.segments)
        self.sequence = 0
        self.catalogue = None
        self.state = {"identity": None, "sources": {}, "source_history": {}, "tasks": {},
                      "proposals": {}, "rejections": [], "operations": {}, "receipts": {}, "source_epoch": 0,
                      "steps": {}, "step_observations": {}, "step_reviews": {}, "step_cache": {},
                      "scratchboard": {}, "scratchboard_revisions": {}}
        for _, transaction in self.journal.tail(self.segments, None, at_head=self.head):
            if transaction["previous"] != self.sequence or transaction["sequence"] != self.sequence + 1:
                raise ValueError("Memory transaction sequence mismatch")
            if self.catalogue is not None and self.catalogue != transaction["catalogueId"]:
                raise ValueError("Memory identity mismatch")
            self.catalogue = transaction["catalogueId"]
            self.sequence = transaction["sequence"]
            for event in transaction["events"]:
                self._apply(event)
        return deepcopy(self.state)

    def _apply(self, e):
        kind = e["kind"]
        if kind == "identity.created":
            if self.state["identity"] is not None:
                raise ValueError("Duplicate continuing identity")
            self.state["identity"] = e["identity"]
        elif kind == "source.recorded":
            record = e["record"]
            self.state["sources"][record["id"]] = record
            self.state["source_history"].setdefault(record["id"], []).append(record)
            self.state["source_epoch"] += 1
        elif kind == "task.revised":
            self.state["tasks"][e["task"]["id"]] = e["task"]
        elif kind == "proposal.recorded":
            self.state["proposals"][e["proposal"]["id"]] = e["proposal"]
        elif kind == "proposal.rejected":
            self.state["rejections"].append(e["rejection"])
        elif kind == "operation.prepared":
            self.state["operations"][e["operation"]["id"]] = e["operation"]
        elif kind == "operation.observed":
            self.state["receipts"][e["receipt"]["operation_id"]] = e["receipt"]
        elif kind == "step.proposed":
            self.state["steps"][e["step"]["id"]] = e["step"]
        elif kind == "step.observed":
            observation = e["observation"]
            self.state["step_observations"][observation["step_id"]] = observation
            self.state["step_cache"][observation["cache_key"]] = observation["step_id"]
        elif kind == "step.reviewed":
            self.state["step_reviews"][e["review"]["step_id"]] = e["review"]
        elif kind == "scratchboard.recorded":
            node = e["node"]
            if node["id"] in self.state["scratchboard"]:
                raise ValueError("Scratchboard nodes are immutable")
            self.state["scratchboard"][node["id"]] = node
        elif kind == "scratchboard.revised":
            revision, node = e["revision"], e["node"]
            if (revision["previous"] not in self.state["scratchboard"]
                    or revision["previous"] in self.state["scratchboard_revisions"]
                    or node["id"] in self.state["scratchboard"]
                    or revision["replacement"] != node["id"]):
                raise ValueError("Scratchboard revision lineage differs")
            self.state["scratchboard"][node["id"]] = node
            self.state["scratchboard_revisions"][revision["previous"]] = revision
        else:
            raise ValueError("Unknown memory event")

    @contextmanager
    def locked(self):
        with self.fs.locked(self.root / ".operation-lock"):
            self.reload()
            yield

    def _append(self, event):
        if self.catalogue is None:
            self.catalogue = str(uuid.uuid4())
        # Validate projection before publication; recovery always replays SJQL.
        old = deepcopy(self.state)
        try:
            self._apply(event)
            transaction = {"catalogueId": self.catalogue, "previous": self.sequence,
                           "sequence": self.sequence + 1, "events": [event]}
            self.head = self.journal.append(self.segments, transaction, self.head)
            self.sequence += 1
        except BaseException:
            self.state = old
            self.reload()  # Reconcile an append whose publication may have succeeded.
            raise

    def initialize(self, identity: dict):
        if set(identity) != {"name", "commitments", "lineage"}:
            raise ValueError("Identity contract")
        bounded_text(identity["name"], "identity name", 128)
        if not isinstance(identity["commitments"], list) or not 1 <= len(identity["commitments"]) <= 32:
            raise ValueError("Identity commitments")
        for item in identity["commitments"]:
            bounded_text(item, "commitment", 1024)
        bounded_text(identity["lineage"], "lineage", 1024)
        with self.locked():
            if self.state["identity"] is not None:
                if self.state["identity"] != identity:
                    raise ValueError("A different identity requires an explicit migration")
                return
            self._append({"kind": "identity.created", "identity": identity})

    def put_source(self, source_id, text, *, expected_revision=0, role="evidence", note=""):
        identifier(source_id)
        bounded_text(text, "source")
        if role not in {"evidence", "observation"}:
            raise ValueError("Sources have data roles")
        if not isinstance(note, str) or len(note.encode()) > 4096:
            raise ValueError("Source note extent")
        with self.locked():
            prior = self.state["sources"].get(source_id)
            if (prior["revision"] if prior else 0) != expected_revision:
                raise ValueError("Stale source revision")
            record = {"id": source_id, "revision": expected_revision + 1, "text": text,
                      "digest": digest_bytes(text.encode()), "role": role, "note": note,
                      "supersedes": prior["digest"] if prior else None}
            self._append({"kind": "source.recorded", "record": record})
            return deepcopy(record)

    def set_task(self, task_id, goal, *, expected_revision=0, filenames=("result.md",)):
        identifier(task_id)
        bounded_text(goal, "task goal", 8192)
        if not 1 <= len(filenames) <= 8 or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\.(md|json|txt)", f) for f in filenames):
            raise ValueError("Artifact filenames must be granted local document names")
        with self.locked():
            if self.state["identity"] is None:
                raise ValueError("Initialize the continuing identity first")
            old = self.state["tasks"].get(task_id)
            if (old["revision"] if old else 0) != expected_revision:
                raise ValueError("Stale task revision")
            task = {"id": task_id, "revision": expected_revision + 1, "goal": goal,
                    "filenames": list(filenames)}
            self._append({"kind": "task.revised", "task": task})
            return deepcopy(task)

    def retrieve(self, query, *, count=4, max_bytes=6000):
        """Deterministic lexical control. Learned ranking is a separate candidate."""
        self.reload()
        return self._select({"kind": "lexical-v1", "query": query, "count": count, "max_bytes": max_bytes})

    def _select(self, selection):
        if selection["kind"] == "explicit-v1":
            ids = selection["ids"]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate explicit evidence selection")
            return [deepcopy(self.state["sources"][sid]) for sid in ids]
        if selection["kind"] != "lexical-v1":
            raise ValueError("Unknown evidence selector")
        query, count, max_bytes = (selection[k] for k in ("query", "count", "max_bytes"))
        if not isinstance(query, str) or type(count) is not int or not 1 <= count <= 64 or type(max_bytes) is not int or not 1 <= max_bytes <= 65536:
            raise ValueError("Evidence selection bounds")
        wanted = token_set(query)
        ranked = sorted(self.state["sources"].values(), key=lambda r: (
            -len(wanted & token_set(r["text"] + " " + r["id"])) / max(1, len(token_set(r["text"]))**0.5), r["id"]))
        chosen, total = [], 0
        for row in ranked:
            size = len(canonical(row))
            if total + size <= max_bytes:
                chosen.append(deepcopy(row)); total += size
            if len(chosen) == count:
                break
        return chosen

    @staticmethod
    def _selection_identity(sources):
        return digest_bytes(canonical([{k: row[k] for k in ("id", "revision", "digest")} for row in sources]))

    def snapshot(self, task_id, sources, core_identity, *, selection=None):
        self.reload()
        task = self.state["tasks"][task_id]
        selection = deepcopy(selection or {"kind": "explicit-v1", "ids": [r["id"] for r in sources]})
        refs = {r["id"]: {"revision": r["revision"], "digest": r["digest"]} for r in sources}
        snapshot = {"task_id": task_id, "task_revision": task["revision"],
                    "identity": digest_bytes(canonical(self.state["identity"])),
                    "sources": refs, "core_identity": core_identity, "selector_identity": selection["kind"],
                    "selection": selection, "selection_output": self._selection_identity(sources)}
        self._validate_snapshot(snapshot)
        return snapshot

    def _selection_current(self, snapshot):
        if "selection" not in snapshot:
            # Historical journals retain their original conservative contract.
            return snapshot["source_epoch"] == self.state["source_epoch"]
        try:
            return self._selection_identity(self._select(snapshot["selection"])) == snapshot["selection_output"]
        except KeyError:
            return False

    def _validate_snapshot(self, snapshot, *, include_selection=True, _depth=0, _board_checked=None):
        if _depth > 16:
            raise StaleState("Observation dependency depth exceeds the contract")
        task = self.state["tasks"].get(snapshot["task_id"])
        if not task or task["revision"] != snapshot["task_revision"]:
            raise StaleState("Proposal task revision is stale")
        if digest_bytes(canonical(self.state["identity"])) != snapshot["identity"]:
            raise StaleState("Proposal identity is stale")
        if include_selection and not self._selection_current(snapshot):
            raise StaleState("Proposal evidence selection is stale")
        for name, expected in snapshot["sources"].items():
            actual = self.state["sources"].get(name)
            if not actual or any(actual[k] != v for k,v in expected.items()):
                raise StaleState("Proposal source is stale")
        for step_id, expected in snapshot.get("observations", {}).items():
            from .steps import Steps
            observation = self.state["step_observations"].get(step_id)
            step = self.state["steps"].get(step_id)
            review = self.state["step_reviews"].get(step_id)
            if (not observation or not step or step["implementation"] != Steps(self).implementation
                    or observation["digest"] != expected
                    or digest_bytes(canonical(observation["result"])) != expected
                    or review and review["decision"] in {"rejected", "cancelled"}):
                raise StaleState("Proposal calculation observation is stale")
            parent = step["snapshot"]
            if parent["task_id"] != snapshot["task_id"] or parent["task_revision"] != snapshot["task_revision"]:
                raise StaleState("Observation belongs to a different task revision")
            self._validate_snapshot(parent, _depth=_depth + 1, _board_checked=_board_checked)
        if snapshot.get("scratchboard"):
            from .scratchboard import Scratchboard
            board = Scratchboard(self)
            checked = set() if _board_checked is None else _board_checked
            for node_id, expected in snapshot["scratchboard"].items():
                node = board._validate(node_id, _depth=_depth + 1, _checked=checked)
                if node["digest"] != expected:
                    raise StaleState("Scratchboard dependency identity differs")
                board._compatible(node, snapshot)

    def attach_observations(self, snapshot, step_ids):
        """Bind observed pure operations to a document or a later operation."""
        snapshot = deepcopy(snapshot)
        with self.locked():
            snapshot.setdefault("observations", {}).update({
                step_id: self.state["step_observations"][step_id]["digest"] for step_id in step_ids})
            self._validate_snapshot(snapshot)
            return snapshot

    def completed(self, task_id):
        """Observe completed current task artifacts without loading a model.

        This resumes an existing obligation. It does not predict what a new
        model invocation would write or authorize repeating an external effect.
        """
        with self.locked():
            task = self.state["tasks"][task_id]
            outputs = {}
            for receipt in self.state["receipts"].values():
                operation = self.state["operations"][receipt["operation_id"]]
                proposal = self.state["proposals"][operation["proposal_id"]]
                if proposal["snapshot"]["task_id"] != task_id:
                    continue
                try:
                    self._validate_snapshot(proposal["snapshot"])
                    verified = self._verify_receipt(receipt)
                except (ValueError, OSError):
                    continue
                outputs[proposal["output"]["filename"]] = {
                    "receipt": verified, "content": proposal["output"]["content"],
                    "artifact_path": str(self.root / verified["path"])}
            return {"task_id": task_id, "task_revision": task["revision"],
                    "identity": deepcopy(self.state["identity"]), "artifacts": outputs,
                    "remaining": [name for name in task["filenames"] if name not in outputs],
                    "mode": "observed completed work", "model_calls": 0}

    def record_rejection(self, snapshot, output, reason, *, metrics=None):
        if not isinstance(output, str) or len(output.encode()) > 32768:
            raise ValueError("Rejected response extent")
        bounded_text(reason, "rejection reason", 2048)
        with self.locked():
            # The historical attempt keeps its original snapshot even when a
            # concurrent correction has made that snapshot ineligible to act.
            rejection = {"snapshot": snapshot, "output": output, "reason": reason, "metrics": metrics or {}}
            self._append({"kind": "proposal.rejected", "rejection": rejection})
            return deepcopy(rejection)

    def record_proposal(self, snapshot, output, *, metrics=None):
        if not isinstance(output, dict) or set(output) != {"filename", "content", "citations"}:
            raise ProposalError("Artifact proposal needs exactly filename, content and citations")
        try:
            bounded_text(output["content"], "artifact", 32768)
        except ValueError as error:
            raise ProposalError(str(error)) from error
        if not isinstance(output["citations"], list) or any(not isinstance(v,str) for v in output["citations"]):
            raise ProposalError("citations must be a list of source references")
        with self.locked():
            self._validate_snapshot(snapshot)
            task = self.state["tasks"][snapshot["task_id"]]
            if output["filename"] not in task["filenames"]:
                raise ProposalError("Artifact filename lacks a task grant")
            eligible = {f"{sid}@{v['revision']}" for sid,v in snapshot["sources"].items()}
            citations = output["citations"]
            if len(set(citations)) != len(citations) or not set(citations) <= eligible or (eligible and not citations):
                raise ProposalError("Citations must identify supplied source revisions")
            missing = [f"[{c}]" for c in citations if f"[{c}]" not in output["content"]]
            if missing:
                raise ProposalError("Document content is missing these declared citations: " + ", ".join(missing))
            references = document_references(output["content"])
            if references != set(citations):
                raise ProposalError("Document references must match the citations list and supplied source revisions")
            proposal = {"id": str(uuid.uuid4()), "snapshot": snapshot, "output": output, "metrics": metrics or {}}
            self._append({"kind": "proposal.recorded", "proposal": proposal})
            return deepcopy(proposal)

    def commit(self, proposal_id, *, core_identity, crash_after_publish=False):
        with self.locked():
            proposal = self.state["proposals"][proposal_id]
            if proposal["snapshot"]["core_identity"] != core_identity:
                raise ValueError("Core version differs from the proposal")
            operation_id = digest_bytes(canonical({"proposal": proposal_id})).split(":")[1][:32]
            if operation_id in self.state["receipts"]:
                return self._verify_receipt(self.state["receipts"][operation_id])
            self._validate_snapshot(proposal["snapshot"])
            operation = self.state["operations"].get(operation_id)
            if operation is None:
                previous = [op["id"] for op in self.state["operations"].values()
                    if op["id"] in self.state["receipts"]
                    and self.state["proposals"][op["proposal_id"]]["snapshot"]["task_id"] == proposal["snapshot"]["task_id"]
                    and self.state["proposals"][op["proposal_id"]]["output"]["filename"] == proposal["output"]["filename"]]
                operation = {"id": operation_id, "proposal_id": proposal_id,
                             "relative_path": f"artifacts/{operation_id}/{proposal['output']['filename']}",
                             "digest": digest_bytes(proposal["output"]["content"].encode()),
                             "supersedes": previous[-1] if previous else None}
                self._append({"kind": "operation.prepared", "operation": operation})
            return self._publish(operation, crash_after_publish=crash_after_publish)

    def _artifact_path(self, relative):
        path = self.root
        for part in Path(relative).parts:
            if part in ("", ".", "..") or Path(part).is_absolute():
                raise ValueError("Artifact path escape")
            path = path / part
            if path.is_symlink():
                raise ValueError("Artifact path contains a symlink")
        return path

    def _publish(self, operation, *, crash_after_publish=False):
        proposal = self.state["proposals"][operation["proposal_id"]]
        target = self._artifact_path(operation["relative_path"])
        content = proposal["output"]["content"].encode()
        if digest_bytes(content) != operation["digest"]:
            raise ValueError("Prepared artifact bytes differ")
        already_exists = target.exists()
        if already_exists:
            if not target.is_file() or file_digest(target) != operation["digest"]:
                raise ValueError("Artifact conflicts with prepared bytes; existing file preserved")
        else:
            self._validate_snapshot(proposal["snapshot"])
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._artifact_path(operation["relative_path"])
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".staging-", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(content); stream.flush(); os.fsync(stream.fileno())
                # Create-only publication preserves existing user bytes even
                # when another writer wins the race to this pathname.
                os.link(temporary, target)
                self.fs.sync_directory(target.parent)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        if file_digest(target) != operation["digest"]:
            raise ValueError("Published artifact readback differs")
        if crash_after_publish:
            os._exit(88)  # Explicit fault injection used by the subprocess test.
        receipt = {"operation_id": operation["id"], "path": operation["relative_path"],
                   "digest": operation["digest"], "reconciled_existing": already_exists,
                   "supersedes": operation.get("supersedes"),
                   "core_identity_at_proposal": proposal["snapshot"]["core_identity"]}
        self._append({"kind": "operation.observed", "receipt": receipt})
        return self._verify_receipt(receipt)

    def _verify_receipt(self, receipt):
        path = self._artifact_path(receipt["path"])
        if not path.is_file() or file_digest(path) != receipt["digest"]:
            raise ValueError("Committed artifact changed or unavailable")
        return deepcopy(receipt)

    def recover(self):
        with self.locked():
            result = []
            self.recovery_issues = []
            for operation_id, operation in list(self.state["operations"].items()):
                try:
                    if operation_id in self.state["receipts"]:
                        self._verify_receipt(self.state["receipts"][operation_id])
                        continue
                    # Observe an existing historical effect even if the task has
                    # since changed. An absent effect needs a still-valid proposal.
                    result.append(self._publish(operation))
                except (ValueError, OSError) as error:
                    # A changed document or stale pending effect limits its own
                    # task. Independent work remains available after recovery.
                    self.recovery_issues.append({"operation_id": operation_id,
                        "task_id": self.state["proposals"][operation["proposal_id"]]["snapshot"]["task_id"],
                        "error": str(error)})
            return result

    def status(self):
        self.reload()
        artifacts = []
        completed = {task_id: set() for task_id in self.state["tasks"]}
        for receipt in self.state["receipts"].values():
            operation = self.state["operations"][receipt["operation_id"]]
            proposal = self.state["proposals"][operation["proposal_id"]]
            try:
                self._validate_snapshot(proposal["snapshot"], include_selection=False)
                current = True
            except (ValueError, OSError):
                current = False
            try:
                self._verify_receipt(receipt)
                observed, observation_error = True, None
            except (ValueError, OSError) as error:
                observed, observation_error = False, str(error)
            task_id = proposal["snapshot"]["task_id"]
            filename = proposal["output"]["filename"]
            selection_current = self._selection_current(proposal["snapshot"])
            if current and observed and selection_current:
                completed[task_id].add(filename)
            artifacts.append({**receipt, "dependency_current": current,
                              "bytes_verified": observed, "observation_error": observation_error,
                              "filename": filename,
                              "selection_current": selection_current,
                              "task_id": task_id})
        obligations = {task_id: {"completed": sorted(completed[task_id]),
            "remaining": [name for name in task["filenames"] if name not in completed[task_id]]}
            for task_id, task in self.state["tasks"].items()}
        return {"head": self.head, "sequence": self.sequence, "identity": deepcopy(self.state["identity"]),
                "tasks": deepcopy(self.state["tasks"]), "source_epoch": self.state["source_epoch"],
                "artifacts": artifacts, "obligations": obligations,
                "rejected_proposals": len(self.state["rejections"]),
                "recovery_issues": deepcopy(self.recovery_issues),
                "pending_operations": [x for x in self.state["operations"] if x not in self.state["receipts"]]}
