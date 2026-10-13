from pathlib import Path
import tempfile
import unittest

from nanoai.native_worker import runtime_environment, backend_arguments, representation_profile, adapter_profile, NativeSession
from nanoai.core import digest_bytes, file_digest
from nanoai.sampling import QWEN35_THINKING
from nanoai.sqlite_journal import canonical
from nanoai.tensor_store import TensorStore
from test_precision import fixture


class NativeProfileTests(unittest.TestCase):
    def test_mapped_metal_can_reach_native_scheduler_with_original_precision(self):
        profile = {"weight_access": "mapped", "backend": "metal"}
        _, flags = runtime_environment(profile, {})
        self.assertEqual(flags["GGML_STREAM_WEIGHTS"], "0")
        self.assertEqual(backend_arguments(profile), ["--device", "MTL0", "--gpu-layers", "999"])
        self.assertEqual(backend_arguments(None), ["--gpu-layers", "0", "--no-op-offload", "--no-kv-offload"])
        for invalid in ({"weight_access": "stream", "backend": "metal"},
                        {"weight_access": "mapped", "backend": "cuda"}):
            with self.assertRaises(ValueError):
                runtime_environment(invalid, {})

    def test_adapter_binds_base_candidate_and_bounded_absolute_file(self):
        digest = "shake256:" + "a" * 128
        value = {"path": "/tmp/adapter.gguf", "digest": digest, "candidate": digest,
                 "basis": digest, "base_model_digest": digest, "bytes": 328640,
                 "precision_inspection": {"profile": "original-bf16"}}
        self.assertEqual(adapter_profile(value, digest), value)
        self.assertIsNone(adapter_profile(None, digest))
        for change in ({"path": "relative.gguf"}, {"bytes": True}, {"bytes": 33 << 20},
                       {"base_model_digest": "shake256:" + "b" * 128}, {"basis": "unknown"}, {"scale": 2}):
            with self.assertRaises(ValueError):
                adapter_profile({**value, **change}, digest)

    def test_streaming_flags_are_pinned_without_ambient_experiments(self):
        env, flags = runtime_environment({"weight_access": "stream", "ring_mb": 256},
            {"PATH": "/bin", "GGML_STREAM_GPU": "1", "GGML_STREAM_UNKNOWN": "1"})
        self.assertEqual(env["PATH"], "/bin")
        self.assertNotIn("GGML_STREAM_UNKNOWN", env)
        self.assertEqual(flags["GGML_STREAM_WEIGHTS"], "2")
        self.assertEqual(flags["GGML_STREAM_RING_MB"], "256")
        self.assertEqual(flags["GGML_STREAM_GPU"], "0")
        self.assertEqual(flags["GGML_STREAM_DGEMM"], "0")
        self.assertEqual(runtime_environment(None, {})[1]["GGML_STREAM_WEIGHTS"], "0")

    def test_unknown_or_unbounded_profiles_reject(self):
        for profile in ({"weight_access": "gpu"}, {"weight_access": "stream", "ring_mb": 8192},
                        {"weight_access": "stream", "readers": True},
                        {"weight_access": "mapped", "arithmetic": "half"}):
            with self.assertRaises(ValueError):
                runtime_environment(profile, {})

    def test_representation_profile_is_explicit_and_bounded(self):
        profile = {"kind": "causal-last-f32-v1", "dimensions": 2048, "max_tokens": 128}
        self.assertEqual(representation_profile(profile), profile)
        self.assertIsNone(representation_profile(None))
        for change in ({"kind": "mean"}, {"max_tokens": 129}, {"dimensions": True}, {"extra": 1}):
            with self.assertRaises(ValueError):
                representation_profile({**profile, **change})

    def test_encoding_preserves_literal_ids_and_rejects_invalid_features(self):
        session = NativeSession()
        session.representation = {"kind": "causal-last-f32-v1", "dimensions": 2, "max_tokens": 3}
        session.check = lambda: None
        calls = []
        response = [{"index": 0, "embedding": [[1.0, -2.0]]}]
        def http(path, payload, **kwargs):
            calls.append((path, payload))
            return {"tokens": [11, 12]} if path == "/tokenize" else response
        session.http = http
        encoded = session.encode({"text": "<|im_start|> veri"})
        self.assertEqual(encoded["vector"], [1.0, -2.0])
        self.assertFalse(calls[0][1]["parse_special"])
        self.assertEqual(calls[1][1]["input"], [11, 12])
        response[0]["embedding"] = [[float("nan"), 2.0]]
        with self.assertRaises(ValueError):
            session.encode({"text": "value"})

    def test_generation_keeps_actual_ids_without_retokenizing_text(self):
        session = NativeSession()
        session.check = lambda: None
        calls = []
        response = {"choices": [{"message": {"content": "parsed text"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 3, "prompt_tokens": 12},
                    "__verbose": {"tokens": [91, 72, 3], "content": "native text including controls"}}
        session.http = lambda path, payload, **kwargs: calls.append((path, payload)) or response
        request = {"command": "generate", "messages": [{"role": "user", "content": "Write."}],
                   "max_tokens": 8, "return_trace": True}
        result = session.generate(request)
        self.assertEqual(result["token_ids"], [91, 72, 3])
        self.assertEqual(result["output"], "parsed text")
        self.assertEqual(result["native_content"], "native text including controls")
        self.assertTrue(calls[0][1]["return_tokens"])
        self.assertTrue(calls[0][1]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(calls[0][1]["reasoning_budget_tokens"], -1)
        for ids in (None, [], [True], [-1], [1 << 24], list(range(9))):
            response["__verbose"]["tokens"] = ids
            with self.assertRaisesRegex(ValueError, "trace"):
                session.generate(request)
        del response["__verbose"]
        self.assertNotIn("token_ids", session.generate({**request, "return_trace": False}))

    def test_sampling_numbers_reach_native_request_and_returned_experience(self):
        session = NativeSession(); session.check = lambda: None
        session.sampling = dict(QWEN35_THINKING)
        calls = []
        response = {"choices": [{"message": {"content": "51"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 1, "prompt_tokens": 12}}
        session.http = lambda path, payload, **kwargs: calls.append(payload) or response
        result = session.generate({"command": "generate", "messages": [{"role": "user", "content": "17 * 3"}]})
        for key, value in QWEN35_THINKING.items():
            if key != "schema": self.assertEqual(calls[0][key], value)
        self.assertEqual(result["sampling"], QWEN35_THINKING)
        self.assertEqual(calls[0]["max_tokens"], -1)
        self.assertEqual(calls[0]["reasoning_budget_tokens"], -1)
        response["__verbose"] = {"tokens": [51], "content": "51", "generation_settings":
            {k: v for k, v in QWEN35_THINKING.items() if k != "schema"}}
        request = {"command": "generate", "messages": [{"role": "user", "content": "17 * 3"}], "return_trace": True}
        self.assertEqual(session.generate(request)["sampling_observation"]["top_k"], 20)
        response["__verbose"]["generation_settings"]["temperature"] = 0
        with self.assertRaisesRegex(ValueError, "applied sampling"):
            session.generate(request)

    def test_native_catalogue_matches_real_header_and_rechecks_immutable_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); payload = root / "weights.gguf"
            payload.write_bytes(fixture(30)); identity = file_digest(payload)
            store = TensorStore(root / "tensors.sqlite3", create=True)
            ref = store.register(payload, expected=identity, basis=identity, recipe="fixture", source="test")
            ref["database"] = "tensors.sqlite3"
            session = NativeSession()
            result = session.catalogue(ref, root, payload, identity)
            self.assertEqual(result["tensor_count"], 1)
            self.assertEqual(result["artifact"], identity)
            session.server = type("Alive", (), {"poll": lambda self: None})()
            session.check()
            store.record("other", protocol=identity, phase="observed", values={})
            session.check()
            with store.connection(write=True) as con:
                con.execute("UPDATE versions SET recipe='changed'")
            with self.assertRaisesRegex(ValueError, "provenance"):
                session.check()

    def test_default_thinking_can_exceed_the_old_cap_and_keeps_final_text_separate(self):
        session = NativeSession(); session.check = lambda: None; session.reasoning_end_id = 99
        captured = {}
        thought_ids = [20] * 700
        response = {"choices": [{"message": {"content": "42", "reasoning_content": "Check the sum."}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 703, "prompt_tokens": 30},
                    "__verbose": {"tokens": thought_ids + [99, 42, 3], "content": "Check the sum.</think>42"}}
        def http(path, payload, **kwargs):
            captured.update(payload=payload, options=kwargs)
            return response
        session.http = http
        row = session.generate({"command": "generate", "messages": [{"role": "user", "content": "19 + 23"}], "return_trace": True})
        self.assertEqual(captured["payload"]["max_tokens"], -1)
        self.assertIsNone(captured["options"]["timeout"])
        self.assertEqual(row["output"], "42")
        self.assertEqual(row["reasoning"]["tokens"], 700)
        self.assertIsNone(row["reasoning"]["token_budget"])

    def test_loaded_native_session_rejects_self_consistent_sql_reinterpretation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); payload = root / 'weights.gguf'
            payload.write_bytes(fixture(30)); identity = file_digest(payload)
            store = TensorStore(root / 'tensors.sqlite3', create=True)
            ref = store.register(payload, expected=identity, basis=identity, recipe='fixture', source='test')
            ref['database'] = 'tensors.sqlite3'
            session = NativeSession()
            session.catalogue(ref, root, payload, identity)
            session.server = type('Alive', (), {'poll': lambda self: None})()
            with store.connection(write=True) as connection:
                connection.execute("UPDATE tensors SET dtype='I16'")
                inventory_id = digest_bytes(canonical(store._tensors(connection, identity)))
                connection.execute('UPDATE artifacts SET inventory_id=?', (inventory_id,))
            self.assertEqual(file_digest(payload), identity)
            with self.assertRaisesRegex(ValueError, 'inventory changed'):
                session.check()
