import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from nanoai.agent import context
from nanoai.memory import Memory
from nanoai.steps import Steps


class CliTests(unittest.TestCase):
    def test_review_render_publish_and_correction_through_fresh_processes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "state"
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

            def cli(*arguments, succeeds=True):
                result = subprocess.run([sys.executable, "-m", "nanoai.cli", "--state", str(state), *arguments],
                    env=environment, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode == 0, succeeds, result.stderr)
                return json.loads(result.stdout) if succeeds else result.stderr

            cli("init")
            source = root / "survey.json"
            source.write_text('{"length":{"value":13,"unit":"cm"},"count":{"value":4,"unit":"count"}}')
            cli("source", "survey", str(source))
            cli("task", "report", "Compute the total length", "--filename", "report.md")
            memory = Memory(state)
            snapshot = context(memory, "report", "fixture-core", selected_ids=["survey"])["snapshot"]
            calculation = Steps(memory).calculate_fields(snapshot, {"operator": "quantity.product", "inputs": [
                {"source": "survey", "path": ["length"]}, {"source": "survey", "path": ["count"]}]})
            step_id = calculation["step_id"]
            self.assertIn("explicit task review", cli("note", step_id, succeeds=False))
            cli("review-step", step_id, "confirmed", "--reviewer", "test-predicate", "--reason", "Frozen fixture expects 52 cm")
            note = cli("note", step_id, "--language", "tr", "--commit")
            self.assertEqual(note["model_calls"], 0)
            self.assertIn("52 cm", Path(note["artifact_path"]).read_text())
            self.assertIn("Hesaplanan toplam", Path(note["artifact_path"]).read_text())
            self.assertEqual(cli("resume", "report")["remaining"], [])
            packet = cli("dictionary", "report", "--source", "survey")
            self.assertEqual(len(packet["quantity_catalogue"]), 2)
            source.write_text('{"length":{"value":17,"unit":"cm"},"count":{"value":4,"unit":"count"}}')
            cli("source", "survey", str(source), "--expected-revision", "1")
            self.assertTrue(cli("resume", "report")["remaining"])
            self.assertIn("stale", cli("note", step_id, "--commit", succeeds=False).lower())
            self.assertIn("52 cm", Path(note["artifact_path"]).read_text())


if __name__ == "__main__":
    unittest.main()
