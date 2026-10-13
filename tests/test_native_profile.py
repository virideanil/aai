import unittest

from nanoai.native_worker import runtime_environment


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
