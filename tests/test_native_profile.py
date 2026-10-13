import unittest

from nanoai.native_worker import runtime_environment, representation_profile, NativeSession


class NativeProfileTests(unittest.TestCase):
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
        def http(path, payload):
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
