"""Bounded client for an existing local JSONL model worker.

The worker owns tokenization and tensor execution. This client owns request
identity, cancellation, process lifetime and the caller's resource envelope.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import time

from .precision import PRECISION_PROFILE


def digest_bytes(data: bytes) -> str:
    return "shake256:" + hashlib.shake_256(data).hexdigest(64)


def file_digest(path: str | Path) -> str:
    h = hashlib.shake_256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return "shake256:" + h.hexdigest(64)


def swap_bytes() -> int | None:
    if sys.platform != "darwin":
        return None
    text = subprocess.check_output(["sysctl", "vm.swapusage"], text=True, timeout=3)
    match = re.search(r"used = ([0-9.]+)M", text)
    if not match:
        raise RuntimeError("Unrecognized system swap counter")
    return int(float(match[1]) * 1024**2)


class RUsage(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups",
            "pageins", "wired_size", "resident_size", "phys_footprint",
            "proc_start_abstime", "proc_exit_abstime",
        )
    ]


def process_memory(pid: int) -> dict:
    if sys.platform == "darwin":
        lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        usage = RUsage()
        if lib.proc_pid_rusage(pid, 0, ctypes.byref(usage)) == 0:
            return {"rss_bytes": usage.resident_size, "physical_footprint_bytes": usage.phys_footprint}
    raw = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True, timeout=3)
    return {"rss_bytes": int(raw.strip() or 0) * 1024, "physical_footprint_bytes": None}


def process_group_memory(group: int) -> dict:
    """Include the native server, which shares its worker's process group."""
    rows = subprocess.check_output(["ps", "-axo", "pid=,pgid="], text=True, timeout=3)
    members = []
    for row in rows.splitlines():
        pid, pgid = map(int, row.split())
        if pgid == group:
            try:
                members.append({"pid": pid, **process_memory(pid)})
            except (ProcessLookupError, subprocess.CalledProcessError):
                continue  # A child can exit between inventory and observation.
    return {"rss_bytes": sum(m["rss_bytes"] for m in members),
        "physical_footprint_bytes": sum(m["physical_footprint_bytes"] for m in members) if members and all(m["physical_footprint_bytes"] is not None for m in members) else None,
        "processes": members}


def live_group_members(group: int) -> list[int]:
    """A reaped worker may still have live descendants in its session."""
    rows = subprocess.check_output(["ps", "-axo", "pid=,pgid=,stat="], text=True, timeout=3)
    return [int(pid) for pid, pgid, state in (row.split() for row in rows.splitlines())
            if int(pgid) == group and not state.startswith("Z")]


class JsonlCore:
    def __init__(self, config: dict, *, stderr_path: str | Path):
        if config.get("precision_profile") != PRECISION_PROFILE:
            raise ValueError("The resident core requires the original BF16 precision profile")
        self.config = dict(config)
        self.sequence = 0
        self.pending = b""
        self.child = None
        self.closed = False
        self.cleanup_errors = []
        self.primary_error = None
        self.selector = selectors.DefaultSelector()
        self.stderr_path = Path(stderr_path)
        self.log = Path(stderr_path).open("ab")
        self.trace = self.stderr_path.with_suffix(".events.jsonl").open("a", encoding="utf-8")
        self.samples = []
        self.next_sample = 0.0
        self.swap_before = swap_bytes()
        self.manifest_identity = file_digest(config["manifest"])
        self.worker_identities = {str(Path(p).resolve()): file_digest(p) for p in config.get("source_files", [])}
        self.identity = digest_bytes(json.dumps({"manifest": self.manifest_identity, "worker": self.worker_identities}, sort_keys=True).encode())
        try:
            self.child = subprocess.Popen(config["command"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=self.log, start_new_session=True, bufsize=0)
            self.selector.register(self.child.stdout, selectors.EVENT_READ)
            self.call("init")
            self.load = self.call("load", path=config["manifest"], expected=self.manifest_identity)
        except BaseException as error:
            self.close(primary_error=error)
            raise

    def check_sources(self):
        if file_digest(self.config["manifest"]) != self.manifest_identity:
            raise ValueError("Model manifest changed")
        if any(file_digest(path) != expected for path, expected in self.worker_identities.items()):
            raise ValueError("Model worker source changed")

    def _sample(self):
        now = time.monotonic()
        if now < self.next_sample:
            return
        self.next_sample = now + 1
        measurement = process_group_memory(self.child.pid) if self.config.get("include_process_group") else process_memory(self.child.pid)
        sample = {"at_monotonic": now, **measurement, "host_swap_bytes": swap_bytes()}
        self.samples.append(sample)
        if sample["rss_bytes"] > self.config.get("rss_limit_bytes", 7 * 1024**3):
            raise RuntimeError("Model process RSS budget reached")
        if (sample["physical_footprint_bytes"] is not None
                and sample["physical_footprint_bytes"] > self.config.get("physical_footprint_limit_bytes", 3 * 1024**3)):
            raise RuntimeError("Model process physical-footprint budget reached")
        if self.swap_before is not None and sample["host_swap_bytes"] - self.swap_before > self.config.get("swap_growth_limit_bytes", 512 * 1024**2):
            raise RuntimeError("System swap-growth budget reached")

    def _line(self, deadline: float | None) -> bytes:
        while b"\n" not in self.pending:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Model request deadline reached")
            self._sample()
            if not self.selector.select(0.05):
                if self.child.poll() is not None:
                    raise RuntimeError("Model worker exited before response")
                continue
            chunk = os.read(self.child.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("Model worker closed its response stream")
            self.pending += chunk
            if len(self.pending) > 2 * 1024**2:
                raise ValueError("Model response framing bound")
        line, self.pending = self.pending.split(b"\n", 1)
        return line

    def call(self, command: str, **fields):
        if self.child is None or self.child.poll() is not None:
            raise RuntimeError("Model worker is closed")
        self.check_sources()
        self.sequence += 1
        request = {"id": self.sequence, "command": command, **fields}
        raw = json.dumps(request, ensure_ascii=False).encode() + b"\n"
        if len(raw) > 131072:
            raise ValueError("Model request framing bound")
        try:
            self.trace.write(json.dumps({"event": "request", "request": request}, ensure_ascii=False) + "\n")
            self.trace.flush()
            self.child.stdin.write(raw)
            self.child.stdin.flush()
            duration = (self.config.get("request_seconds") if command in {"generate", "propose"}
                        else self.config.get("startup_seconds", 120))
            if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (int, float))
                                         or not 0 < duration < float("inf")):
                raise ValueError("Model request duration must be positive or null")
            result = json.loads(self._line(None if duration is None else time.monotonic() + duration))
            self.trace.write(json.dumps({"event": "response", "response": result}, ensure_ascii=False) + "\n")
            self.trace.flush()
            if result.get("id") != self.sequence:
                raise ValueError("Model response identity mismatch")
            if result.get("ok") is not True:
                raise RuntimeError(result.get("error", "Model worker failed"))
            self.check_sources()
            return result["result"]
        except BaseException as error:
            self.close(primary_error=error)
            raise

    def generate(self, messages: list, *, max_tokens=None, schema=None, trace=False):
        fields = {"messages": messages, "max_tokens": max_tokens}
        if type(trace) is not bool:
            raise ValueError("Generation trace choice must be Boolean")
        if trace:
            fields["return_trace"] = True
        if schema is not None:
            fields["response_schema"] = schema
        return self.call("propose" if schema is not None else "generate", **fields)

    def encode(self, text: str):
        """Read a full-backbone representation under an explicit feature profile."""
        return self.call("encode", text=text)

    def close(self, *, primary_error: BaseException | None = None):
        if self.closed:
            return
        if primary_error is not None and self.primary_error is None:
            self.primary_error = {"type": type(primary_error).__name__, "message": str(primary_error)}
        errors = []
        failed = object()

        def attempt(stage, operation):
            try:
                return operation()
            except Exception as error:
                errors.append(error)
                self.cleanup_errors.append({"stage": stage, "type": type(error).__name__, "message": str(error)})
                return failed

        def observe_group():
            result = attempt("observe_process_group", lambda: live_group_members(self.child.pid))
            return None if result is failed else result

        def signal_group(sig):
            def send():
                try:
                    os.killpg(self.child.pid, sig)
                except ProcessLookupError:
                    pass
            if attempt(f"signal_process_group_{sig.name}", send) is failed:
                # The directly owned process can remain signalable when a group
                # operation fails. Its descendants still require observation.
                attempt(f"signal_worker_{sig.name}", lambda: self.child.send_signal(sig))

        remaining = []
        child_reaped = None
        if self.child is not None:
            # Observe descendants even after their worker exits. An unavailable
            # process inventory is unknown and still requires cleanup attempts.
            remaining = observe_group()
            if remaining != []:
                signal_group(signal.SIGTERM)
            deadline = time.monotonic() + 3
            while remaining and time.monotonic() < deadline:
                attempt("poll_worker", self.child.poll)
                remaining = observe_group()
                time.sleep(.025)
            if remaining != []:
                signal_group(signal.SIGKILL)
            attempt("wait_worker", lambda: self.child.wait(timeout=3))
            observed_exit = attempt("observe_worker_exit", self.child.poll)
            child_reaped = None if observed_exit is failed else observed_exit is not None
            deadline = time.monotonic() + 1
            remaining = observe_group()
            while remaining and time.monotonic() < deadline:
                time.sleep(.025)
                remaining = observe_group()
            for name, stream in (("stdin", self.child.stdin), ("stdout", self.child.stdout)):
                if stream:
                    attempt(f"close_{name}", stream.close)
        for name, stream in (("selector", self.selector), ("log", self.log), ("trace", self.trace)):
            attempt(f"close_{name}", stream.close)
        receipt = {
            "swap_before_bytes": self.swap_before, "samples": self.samples,
            "child_reaped": child_reaped,
            "exit_code": self.child.returncode if self.child else None,
            "process_group_stopped": None if remaining is None else not remaining,
            "remaining_processes": remaining,
            "primary_error": self.primary_error, "cleanup_errors": self.cleanup_errors,
        }
        receipt_path = self.stderr_path.with_suffix(".resources.json")
        attempt("write_resource_receipt", lambda: receipt_path.write_text(json.dumps(receipt, indent=2) + "\n"))
        complete = remaining == [] and (self.child is None or child_reaped is True)
        self.closed = complete and not errors
        if errors or not complete:
            detail = f"Model cleanup incomplete or reported errors; inspect {receipt_path}"
            if primary_error is not None:
                primary_error.add_note(detail)
            else:
                raise RuntimeError(detail) from (errors[0] if errors else None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, error, traceback):
        self.close(primary_error=error)
