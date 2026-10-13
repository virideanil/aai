from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from nanoai.agent import DEFAULT_IDENTITY
from nanoai.catalogue import LanguageCatalogue
from nanoai.memory import Memory, StaleState
from nanoai.writing import Writing


class FixedCore:
    identity = "writing-test-core"

    def __init__(self, text="Mira wound the blue clock."):
        self.text, self.calls = text, 0

    def generate(self, messages, *, max_tokens, trace):
        self.calls += 1
        return {"output": self.text, "token_ids": [1, 2, 3], "finish_reason": "stop",
                "tokens": 3, "prompt_tokens": 20}


class WritingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.memory = Memory(self.root)
        self.memory.initialize(DEFAULT_IDENTITY)
        self.memory.put_source("canon", "Mira repairs a blue clock.", note="Fictional canon.")
        self.owner = Writing(self.memory)

    def tearDown(self):
        self.tmp.cleanup()

    def test_publication_catalogue_and_zero_call_restart(self):
        core = FixedCore()
        result = self.owner.run("opening", "Write one sentence.", ["canon"], core)
        path = Path(result["artifact_path"])
        before = path.stat()
        fresh = Writing(Memory(self.root))
        again = fresh.run("opening", "Write one sentence.", ["canon"], core)
        self.assertEqual(core.calls, 1)
        self.assertEqual(again["model_calls"], 0)
        self.assertEqual(result["receipt"], again["receipt"])
        self.assertEqual((path.stat().st_ino, path.stat().st_mtime_ns), (before.st_ino, before.st_mtime_ns))
        view = LanguageCatalogue(fresh.memory).view("opening")
        self.assertTrue(view["observations"][0]["current"])
        self.assertEqual(result["source"]["text"], core.text)

    def test_late_correction_invalidates_descendants_and_history_remains_quotable(self):
        core = FixedCore()
        first = self.owner.run("opening", "Write one sentence.", ["canon"], core)
        continued = self.owner.run("next", "Continue.", ["canon", first["source"]["id"]], core, kind="continuation")
        self.memory.put_source("canon", "Mira repairs a green clock.", expected_revision=1, note="Fictional canon.")
        with self.assertRaises(StaleState):
            self.owner.assemble("book", [continued["source"]["id"]], "Clocks")
        self.assertFalse(LanguageCatalogue(self.memory).view("next")["observations"][0]["current"])
        history = self.owner.quote_history(first["experience_id"], "old-opening")
        self.assertTrue(self.memory.source_current(history))
        self.assertIn("historical-writing", history["text"])
        core.text = "Mira wound the green clock."
        revised = self.owner.run("revision", "Revise against current canon.", ["canon", history["id"]], core, kind="revision")
        manuscript = self.owner.assemble("book", [revised["source"]["id"]], "Clocks")
        self.assertIn("green clock", Path(manuscript["artifact_path"]).read_text())
        self.assertEqual(self.owner.assemble("book", [revised["source"]["id"]], "Clocks")["receipt"], manuscript["receipt"])

    def test_recovery_after_publication_does_not_repeat_generation_or_file_effect(self):
        code = """
import sys
from nanoai.memory import Memory
from nanoai.writing import Writing
class Core:
 identity='writing-test-core'
 def generate(self,*a,**k):return {'output':'Mira wound the blue clock.','token_ids':[1,2,3],'finish_reason':'stop'}
Writing(Memory(sys.argv[1])).run('opening','Write one sentence.',['canon'],Core(),crash_after_publish=True)
"""
        result = subprocess.run([sys.executable, "-c", code, str(self.root)], capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 88, result.stderr.decode())
        path = next((self.root / "artifacts").glob("*/text.md"))
        before = path.stat()
        fresh = Memory(self.root)
        recovered = fresh.recover()
        self.assertEqual(len(recovered), 1)
        core = FixedCore("This must not replace the retained generation.")
        resumed = Writing(fresh).run("opening", "Write one sentence.", ["canon"], core)
        self.assertEqual(core.calls, 0)
        self.assertEqual(resumed["model_calls"], 0)
        self.assertEqual((path.stat().st_ino, path.stat().st_mtime_ns), (before.st_ino, before.st_mtime_ns))
        self.assertEqual(fresh.recover(), [])


if __name__ == "__main__":
    unittest.main()
