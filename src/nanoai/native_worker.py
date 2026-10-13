"""Local JSONL adapter for a pinned native server over a private socket.

The caller owns task state, authority and file effects. The server performs only
model inference. Its process shares the caller-created worker process group.
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time

from .core import file_digest, digest_bytes
from .precision import PRECISION_PROFILE, inspect_bf16
from .sampling import profile as sampling_profile
from .tensor_store import describe_reference, inventory


def signature(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def backend_arguments(profile):
    """Explicit scheduling; native logs retain actual placement and fallbacks."""
    profile = profile or {"weight_access": "mapped"}
    backend = profile.get("backend", "cpu")
    if backend not in ("cpu", "metal"):
        raise ValueError("Native backend profile")
    if backend == "metal":
        if profile.get("weight_access") != "mapped":
            raise ValueError("Metal requires mapped original weights")
        return ["--device", "MTL0", "--gpu-layers", "999"]
    return ["--gpu-layers", "0", "--no-op-offload", "--no-kv-offload"]


def runtime_environment(profile, inherited=None):
    """Pin streaming choices; ambient experiment flags cannot change inference."""
    profile = profile or {"weight_access": "mapped"}
    if not isinstance(profile, dict) or set(profile) - {"weight_access", "ring_mb", "readers", "backend"}:
        raise ValueError("Native runtime profile fields")
    backend_arguments(profile)
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


def representation_profile(value):
    """A feature basis is explicit and distinct from the generation profile."""
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {"kind", "dimensions", "max_tokens"}
            or value["kind"] != "causal-last-f32-v1"
            or type(value["dimensions"]) is not int or not 1 <= value["dimensions"] <= 32768
            or type(value["max_tokens"]) is not int or not 1 <= value["max_tokens"] <= 128):
        raise ValueError("Native representation profile")
    return dict(value)


def adapter_profile(value, model_digest):
    """Bind a separate evaluated candidate to one unchanged base model."""
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {
            "path", "digest", "bytes", "candidate", "basis", "base_model_digest", "precision_inspection"}
            or not isinstance(value["path"], str) or not Path(value["path"]).is_absolute()
            or type(value["bytes"]) is not int or not 1 <= value["bytes"] <= 32 << 20
            or value["base_model_digest"] != model_digest
            or not isinstance(value["precision_inspection"], dict)
            or any(not isinstance(value[k], str) or re.fullmatch(r"shake256:[0-9a-f]{128}", value[k]) is None
                   for k in ("digest", "candidate", "basis", "base_model_digest"))):
        raise ValueError("Native adapter source and candidate profile")
    return dict(value)


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
        self.representation = None
        self.context_tokens = 4096
        self.reasoning_end_id = None
        self.sampling = sampling_profile()
        self.catalogues = []
        self.backend = "CPU"

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
        for reference, directory, expected, inventory_id in self.catalogues:
            row = describe_reference(reference, directory=directory, expected=expected)
            if row["inventory_id"] != inventory_id:
                raise ValueError("Pinned native tensor inventory changed")

    def catalogue(self, reference, directory, payload, expected):
        if reference is None:
            return None
        row = describe_reference(reference, directory=directory, expected=expected)
        fmt, tensors = inventory(payload)
        if (fmt != "GGUF" or row["format"] != fmt or row["tensors"] != tensors
                or row["byte_count"] != Path(payload).stat().st_size):
            raise ValueError("Native tensor inventory differs from its catalogue version")
        self.catalogues.append((dict(reference), Path(directory), expected, row["inventory_id"]))
        return {**reference, "format": fmt, "tensor_count": row["tensor_count"],
                "inventory_id": row["inventory_id"], "payload_verified": True}

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
        self.representation = representation_profile(manifest.get("representation"))
        self.sampling = sampling_profile(manifest.get("sampling"))
        self.context_tokens = manifest.get("context_tokens", 4096)
        if type(self.context_tokens) is not int or not 512 <= self.context_tokens <= 262144:
            raise ValueError("Native context capacity is outside the qualified model extent")
        original_digest = manifest["model"].get("restored_digest", manifest["model"]["digest"])
        adapter = adapter_profile(manifest.get("adapter"), original_digest)
        if manifest.get("adapter_tensor_catalogue") is not None and adapter is None:
            raise ValueError("An adapter catalogue requires its native adapter")
        entries = [manifest["model"], manifest["server"], *manifest.get("libraries", [])]
        if manifest.get("decoder"):
            entries.append(manifest["decoder"])
        if adapter is not None:
            entries.append(adapter)
        for entry in entries:
            target = Path(entry["path"]).resolve(strict=True)
            before = signature(target)
            if file_digest(target) != entry["digest"] or signature(target) != before:
                raise ValueError("Native source bytes differ")
            self.pinned.append((target, before))
        if adapter is not None and (Path(adapter["path"]).stat().st_size != adapter["bytes"]
                or inspect_bf16(adapter["path"]) != adapter["precision_inspection"]):
            raise ValueError("Native adapter byte extent or coefficient precision differs")
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
        catalogue = self.catalogue(manifest.get("tensor_catalogue"), path.parent, model_path, original_digest)
        adapter_catalogue = (self.catalogue(manifest.get("adapter_tensor_catalogue"), path.parent,
                             adapter["path"], adapter["digest"]) if adapter is not None else None)
        self.pinned.append((path, signature(path)))
        self.socket_path = Path(self.directory.name) / "core.sock"
        if len(os.fsencode(self.socket_path)) >= 100:
            raise ValueError("Temporary socket path exceeds the native limit")
        environment, runtime_flags = runtime_environment(manifest.get("runtime"))
        self.backend = ("Metal with CPU fallback" if (manifest.get("runtime") or {}).get("backend") == "metal" else "CPU")
        arguments = [manifest["server"]["path"], "--model", str(model_path),
            "--host", str(self.socket_path), "--parallel", "1", "--ctx-size", str(self.context_tokens),
            "--batch-size", "128", "--ubatch-size", "128", "--threads", "4", "--threads-batch", "4",
            *backend_arguments(manifest.get("runtime")), "--fit", "off",
            "--cache-type-k", "bf16", "--cache-type-v", "bf16",
            "--no-webui", "--jinja", "--reasoning", "on", "--reasoning-budget", "-1",
            "--reasoning-format", "deepseek", "--n-predict", "-1", "--no-context-shift"]
        if self.representation is not None:
            arguments += ["--embeddings", "--pooling", "last", "--embd-normalize", "-1"]
        if adapter is not None:
            arguments += ["--lora", adapter["path"]]
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
        if self.representation is None:
            ending = self.http("/tokenize", {"content": "</think>", "add_special": False, "parse_special": True})["tokens"]
            if isinstance(ending, list) and len(ending) == 1 and type(ending[0]) is int:
                self.reasoning_end_id = ending[0]
        return {"parameters": self.parameters, "backend": self.backend, "runtime_pid": self.server.pid,
            "backend_scope": "Configured scheduling; native log records actual placement and fallbacks",
            "precision": precision,
            "runtime_flags": runtime_flags,
            "source_storage": storage,
            "tensor_catalogue": catalogue, "adapter_tensor_catalogue": adapter_catalogue,
            "owned_temporary_directory": self.directory.name,
            "transport": "private Unix socket", "context_tokens": self.context_tokens,
            "reasoning": {"enabled": True, "token_budget": None, "effort_control": "natural completion and observed task checks"},
            "default_generation_token_limit": None,
            "sampling": self.sampling,
            "representation": self.representation,
            "adapter": adapter,
            "cache_policy": "rebuild each request", "model_identity_read_warms_file_cache": True}

    def generate(self, request):
        self.check()
        if self.representation is not None:
            raise ValueError("The selected representation profile serves encode requests")
        messages = request.get("messages")
        limit = request.get("max_tokens")
        trace = request.get("return_trace", False)
        if type(trace) is not bool:
            raise ValueError("Native trace choice must be Boolean")
        if (not isinstance(messages, list) or not 1 <= len(messages) <= 32
                or (limit is not None and (type(limit) is not int or not 1 <= limit <= self.context_tokens))):
            raise ValueError("Native request extent")
        for message in messages:
            if not isinstance(message, dict) or set(message) != {"role", "content"} or message["role"] not in ("system", "user", "assistant", "tool") or not isinstance(message["content"], str):
                raise ValueError("Native message contract")
        if len(json.dumps(messages).encode()) > 65536:
            raise ValueError("Native message byte extent")
        payload = {"messages": messages, "max_tokens": -1 if limit is None else limit,
            **{k: v for k, v in self.sampling.items() if k != "schema"},
            "stream": False, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget_tokens": -1}
        if trace:
            payload.update(return_tokens=True, verbose=True)
        if request["command"] == "propose":
            schema = request.get("response_schema")
            if not isinstance(schema, dict) or len(json.dumps(schema).encode()) > 32768:
                raise ValueError("Native schema bound")
            payload["response_format"] = {"type": "json_object", "schema": schema}
        start = time.monotonic()
        response = self.http("/v1/chat/completions", payload, timeout=None)
        choice = response["choices"][0]
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Native completion content contract")
        thought = choice["message"].get("reasoning_content", "")
        if not isinstance(thought, str):
            raise ValueError("Native reasoning content contract")
        self.check()
        result = {"parameters": self.parameters, "output": content, "raw": content,
            "tokens": response["usage"]["completion_tokens"], "prompt_tokens": response["usage"]["prompt_tokens"],
            "finish_reason": choice["finish_reason"], "elapsed_ms": (time.monotonic() - start) * 1000,
            "backend": self.backend, "cache_prompt": False, "timings": response.get("timings"),
            "context_tokens": self.context_tokens, "generation_token_limit": limit,
            "sampling": dict(self.sampling),
            "reasoning": {"enabled": True, "token_budget": None, "characters": len(thought), "tokens": None}}
        if trace:
            native = response.get("__verbose", {})
            settings = native.get("generation_settings")
            if settings is not None:
                if (not isinstance(settings, dict) or any(
                        type(settings.get(k)) not in (int, float)
                        or not math.isclose(settings[k], v, rel_tol=1e-6, abs_tol=1e-7)
                        for k, v in self.sampling.items() if k != "schema")):
                    raise ValueError("Native applied sampling differs from the pinned profile")
                result["sampling_observation"] = {k: settings[k] for k in self.sampling if k != "schema"}
                result["native_sampler_order"] = settings.get("samplers")
            ids = native.get("tokens")
            if (not isinstance(ids, list) or not 1 <= len(ids) <= (limit or self.context_tokens)
                    or any(type(t) is not int or not 0 <= t < 1 << 24 for t in ids)
                    or not isinstance(native.get("content"), str)):
                raise ValueError("Native sampled token trace is unavailable or malformed")
            result["token_ids"] = ids
            result["native_content"] = native["content"]
            if self.reasoning_end_id in ids:
                result["reasoning"]["tokens"] = ids.index(self.reasoning_end_id)
                result["reasoning"]["count_scope"] = "Sampled IDs before the actual closing thinking token"
            result["trace_scope"] = "Actual sampled native IDs and pre-parser text; output is the native chat parser's content"
        return result

    def encode(self, request):
        """Read the pinned backbone and optional adapter without updating parameters."""
        self.check()
        if self.representation is None:
            raise ValueError("Encoding requires a pinned representation profile")
        text = request.get("text")
        if not isinstance(text, str) or not text or len(text.encode("utf-8")) > 8192:
            raise ValueError("Native representation text extent")
        start = time.monotonic()
        tokens = self.http("/tokenize", {"content": text, "add_special": True,
                                         "parse_special": False})["tokens"]
        if (not isinstance(tokens, list) or not 1 <= len(tokens) <= self.representation["max_tokens"]
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError("Native representation token extent")
        # Supplying the literal IDs avoids reparsing source control-looking text.
        response = self.http("/embeddings", {"input": tokens, "embd_normalize": -1,
                                              "cache_prompt": False})
        if (not isinstance(response, list) or len(response) != 1
                or response[0].get("index") != 0
                or not isinstance(response[0].get("embedding"), list)
                or len(response[0]["embedding"]) != 1):
            raise ValueError("Native representation response shape")
        vector = response[0]["embedding"][0]
        if (not isinstance(vector, list) or len(vector) != self.representation["dimensions"]
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)
                or not any(vector)):
            raise ValueError("Native representation is nonfinite, empty or incompatible")
        self.check()
        return {"parameters": self.parameters, "representation": self.representation,
                "text_identity": digest_bytes(text.encode("utf-8")), "native_ids": tokens,
                "vector": vector, "dtype": "float32", "elapsed_ms": (time.monotonic()-start)*1000,
                "backend": self.backend, "cache_prompt": False}

    def call(self, request):
        command = request.get("command")
        if command == "init" and not self.initialized:
            self.initialized = True
            return {"protocol": "nanoai.jsonl-core.v1", "backend": "native; selected on load"}
        if command == "load":
            return self.load(request)
        if command in ("generate", "propose"):
            return self.generate(request)
        if command == "encode":
            return self.encode(request)
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
