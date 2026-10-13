"""Local JSONL adapter for a pinned llama.cpp CPU server over a private socket.

The caller owns task state, authority and file effects. The server performs only
model inference. Its process shares the caller-created worker process group.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

from .core import file_digest
from .precision import PRECISION_PROFILE, inspect_bf16


def signature(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def runtime_environment(profile, inherited=None):
    """Pin streaming choices; ambient experiment flags cannot change inference."""
    profile = profile or {"weight_access": "mapped"}
    if not isinstance(profile, dict) or set(profile) - {"weight_access", "ring_mb", "readers"}:
        raise ValueError("Native runtime profile fields")
    access = profile.get("weight_access")
    if access not in ("mapped", "stream"):
        raise ValueError("Native weight access")
    ring, readers = profile.get("ring_mb", 512), profile.get("readers", 4)
    if type(ring) is not int or ring not in (256, 512, 1024) or type(readers) is not int or not 1 <= readers <= 8:
        raise ValueError("Native streaming resource extent")
    environment = dict(os.environ if inherited is None else inherited)
    for key in list(environment):
        if key.startswith("GGML_STREAM_"):
            del environment[key]
    flags = {"GGML_STREAM_WEIGHTS": "2" if access == "stream" else "0",
        "GGML_STREAM_MIN_MB": "1", "GGML_STREAM_RING_MB": str(ring),
        "GGML_STREAM_READERS": str(readers), "GGML_STREAM_PACKED": "0",
        "GGML_STREAM_GPU": "0", "GGML_STREAM_DGEMM": "0"}
    environment.update(flags)
    return environment, flags


class SocketHTTP(http.client.HTTPConnection):
    def __init__(self, path, timeout):
        super().__init__("localhost", timeout=timeout)
        self.path = str(path)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


class NativeSession:
    def __init__(self):
        self.initialized = False
        self.server = None
        self.directory = None
        self.pinned = []
        self.parameters = None
        self.group_interrupted = False

    def http(self, path, payload=None, *, timeout=110):
        client = SocketHTTP(self.socket_path, timeout)
        try:
            body = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
            client.request("POST" if payload is not None else "GET", path, body, {"Content-Type": "application/json"})
            response = client.getresponse()
            raw = response.read(2 * 1024**2 + 1)
            if len(raw) > 2 * 1024**2:
                raise ValueError("Native response bound")
            if response.status != 200:
                raise RuntimeError(f"Native HTTP {response.status}: {raw[:200].decode(errors='replace')}")
            return json.loads(raw)
        finally:
            client.close()

    def check(self):
        if self.server is None or self.server.poll() is not None:
            raise RuntimeError("Native model server is unavailable")
        if any(signature(path) != expected for path, expected in self.pinned):
            raise ValueError("Pinned native input changed")

    def load(self, request):
        if not self.initialized or self.server is not None:
            raise ValueError("Native load order")
        path = Path(request["path"]).resolve(strict=True)
        if file_digest(path) != request["expected"]:
            raise ValueError("Native manifest identity differs")
        manifest = json.loads(path.read_text())
        if manifest.get("schema") != "nanoai.native-model.v1":
            raise ValueError("Native manifest schema")
        if manifest.get("precision_profile") != PRECISION_PROFILE:
            raise ValueError("Original BF16 model manifest required")
        entries = [manifest["model"], manifest["server"], *manifest.get("libraries", [])]
        if manifest.get("decoder"):
            entries.append(manifest["decoder"])
        for entry in entries:
            target = Path(entry["path"]).resolve(strict=True)
            before = signature(target)
            if file_digest(target) != entry["digest"] or signature(target) != before:
                raise ValueError("Native source bytes differ")
            self.pinned.append((target, before))
        self.directory = tempfile.TemporaryDirectory(prefix="nai-")
        model = manifest["model"]
        model_path = Path(model["path"])
        storage = {"format": model.get("format", "GGUF"), "stored_bytes": model_path.stat().st_size,
            "restored_bytes": model_path.stat().st_size, "restore_seconds": 0}
        if model.get("format") == "NBH1":
            from .harmony import Library
            start = time.monotonic()
            model_path = Path(self.directory.name) / "restored.gguf"
            Library(manifest["decoder"]["path"]).unpack_file(model["path"], model_path)
            if model_path.stat().st_size != model["restored_bytes"] or file_digest(model_path) != model["restored_digest"]:
                raise ValueError("Restored model differs from the complete original BF16 source")
            self.pinned.append((model_path, signature(model_path)))
            storage.update(restored_bytes=model_path.stat().st_size, restore_seconds=time.monotonic() - start,
                complete_original_digest_verified=True,
                runtime_weight_storage="Ordinary BF16 GGUF in an owned temporary file")
        elif model.get("format") not in (None, "GGUF"):
            raise ValueError("Native source storage format")
        precision = inspect_bf16(model_path)
        if precision != manifest.get("precision_inspection"):
            raise ValueError("BF16 tensor inventory differs from the pinned inspection")
        self.pinned.append((path, signature(path)))
        self.socket_path = Path(self.directory.name) / "core.sock"
        if len(os.fsencode(self.socket_path)) >= 100:
            raise ValueError("Temporary socket path exceeds the native limit")
        arguments = [manifest["server"]["path"], "--model", str(model_path),
            "--host", str(self.socket_path), "--parallel", "1", "--ctx-size", "4096",
            "--batch-size", "128", "--ubatch-size", "128", "--threads", "4", "--threads-batch", "4",
            "--gpu-layers", "0", "--no-op-offload", "--no-kv-offload", "--fit", "off",
            "--cache-type-k", "bf16", "--cache-type-v", "bf16",
            "--no-webui", "--jinja", "--reasoning-budget", "0",
            "--chat-template-kwargs", '{"enable_thinking":false}']
        environment, runtime_flags = runtime_environment(manifest.get("runtime"))
        self.server = subprocess.Popen(arguments, env=environment, stdin=subprocess.DEVNULL, stdout=sys.stderr, stderr=sys.stderr)
        deadline = time.monotonic() + 100
        while True:
            self.check()
            try:
                if self.http("/health", timeout=1).get("status") == "ok":
                    break
            except (OSError, RuntimeError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Native model startup deadline")
            time.sleep(0.1)
        self.parameters = request["expected"]
        return {"parameters": self.parameters, "backend": "CPU", "runtime_pid": self.server.pid,
            "precision": precision,
            "runtime_flags": runtime_flags,
            "source_storage": storage,
            "owned_temporary_directory": self.directory.name,
            "transport": "private Unix socket", "context_tokens": 4096,
            "cache_policy": "rebuild each request", "model_identity_read_warms_file_cache": True}

    def generate(self, request):
        self.check()
        messages = request.get("messages")
        limit = request.get("max_tokens", 384)
        if not isinstance(messages, list) or not 1 <= len(messages) <= 32 or type(limit) is not int or not 1 <= limit <= 512:
            raise ValueError("Native request extent")
        for message in messages:
            if not isinstance(message, dict) or set(message) != {"role", "content"} or message["role"] not in ("system", "user", "assistant", "tool") or not isinstance(message["content"], str):
                raise ValueError("Native message contract")
        if len(json.dumps(messages).encode()) > 65536:
            raise ValueError("Native message byte extent")
        payload = {"messages": messages, "max_tokens": limit, "temperature": 0,
            "seed": 0, "stream": False, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
        if request["command"] == "propose":
            schema = request.get("response_schema")
            if not isinstance(schema, dict) or len(json.dumps(schema).encode()) > 32768:
                raise ValueError("Native schema bound")
            payload["response_format"] = {"type": "json_object", "schema": schema}
        start = time.monotonic()
        response = self.http("/v1/chat/completions", payload)
        choice = response["choices"][0]
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Native completion content contract")
        self.check()
        return {"parameters": self.parameters, "output": content, "raw": content,
            "tokens": response["usage"]["completion_tokens"], "prompt_tokens": response["usage"]["prompt_tokens"],
            "finish_reason": choice["finish_reason"], "elapsed_ms": (time.monotonic() - start) * 1000,
            "backend": "CPU", "cache_prompt": False, "timings": response.get("timings")}

    def call(self, request):
        command = request.get("command")
        if command == "init" and not self.initialized:
            self.initialized = True
            return {"protocol": "nanoai.jsonl-core.v1", "backend": "native CPU"}
        if command == "load":
            return self.load(request)
        if command in ("generate", "propose"):
            return self.generate(request)
        raise ValueError("Native command or state")

    def close(self):
        if self.server is not None:
            if self.server.poll() is None and not self.group_interrupted:
                self.server.terminate()
            try:
                self.server.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait()
        if self.directory is not None:
            self.directory.cleanup()


def main():
    session = NativeSession()
    def stop(*_):
        session.group_interrupted = True
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, stop)
    try:
        for line in sys.stdin:
            request = None
            try:
                if len(line.encode()) > 131072:
                    raise ValueError("Native request framing bound")
                request = json.loads(line)
                result = {"id": request["id"], "ok": True, "result": session.call(request)}
            except Exception as error:
                result = {"id": request.get("id") if isinstance(request, dict) else None, "ok": False, "error": str(error)}
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    main()
