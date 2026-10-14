"""The public writing command reuses checked work before starting its worker."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from nanoai.agent import DEFAULT_IDENTITY
from nanoai.memory import Memory
from nanoai.scratchboard import Scratchboard
from nanoai.steps import Steps


WORKER = r'''
import json, pathlib, sys
marker = pathlib.Path(sys.argv[1])
with marker.open('a') as stream:
    stream.write('start\n')
for line in sys.stdin:
    request = json.loads(line)
    result = {}
    if request['command'] in ('generate', 'propose'):
        text = '\n'.join(m['content'] for m in request['messages'])
        result = {'output': 'Mira repairs the green clock.' if 'green' in text else 'Mira repairs the blue clock.',
                  'token_ids': [1, 2, 3], 'finish_reason': 'stop'}
    print(json.dumps({'id': request['id'], 'ok': True, 'result': result}), flush=True)
'''


class WritingCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.state = self.root / "state"
        self.memory = Memory(self.state)
        self.memory.initialize(DEFAULT_IDENTITY)
        self.memory.put_source("canon", "Mira repairs the blue clock.")
        self.worker = self.root / "worker.py"
        self.worker.write_text(WORKER)
        self.marker = self.root / "starts.txt"
        manifest = self.root / "manifest.json"
        manifest.write_text("{}")
        self.config = self.root / "core.json"
        self.config.write_text(json.dumps({"precision_profile": "original-bf16",
            "manifest": str(manifest), "source_files": [str(self.worker)],
            "command": [sys.executable, str(self.worker), str(self.marker)]}))
        self.instruction = self.root / "instruction.txt"
        self.instruction.write_text("Write one sentence from the current canon.")

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, *extra, succeeds=True):
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
        result = subprocess.run([sys.executable, "-m", "nanoai.cli", "--state", str(self.state),
            "write-phase", "opening", str(self.instruction), "--source", "canon",
            "--core-config", str(self.config), *extra], env=environment, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode == 0, succeeds, result.stderr)
        return json.loads(result.stdout) if succeeds else result.stderr

    def starts(self):
        return len(self.marker.read_text().splitlines()) if self.marker.exists() else 0

    def test_completed_repetition_avoids_startup_and_source_correction_generates_again(self):
        first = self.write()
        path = Path(first["artifact_path"])
        before = (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
        self.assertEqual(self.starts(), 1)
        repeated = self.write()
        self.assertEqual(repeated["model_calls"], 0)
        self.assertEqual(self.starts(), 1, "Retained work still started a model worker")
        self.assertEqual(repeated["receipt"], first["receipt"])
        self.assertEqual((path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns), before)
        self.memory.put_source("canon", "Mira repairs the green clock.", expected_revision=1)
        revised = self.write()
        self.assertEqual(revised["model_calls"], 1)
        self.assertEqual(self.starts(), 2)
        self.assertIn("green", Path(revised["artifact_path"]).read_text())
        self.assertEqual((path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns), before)

    def test_corrupt_retained_artifact_rejects_before_startup(self):
        first = self.write()
        path = Path(first["artifact_path"])
        path.write_text("Changed outside the recorded operation.")
        self.write(succeeds=False)
        self.assertEqual(self.starts(), 1)
        self.assertEqual(path.read_text(), "Changed outside the recorded operation.")

    def test_public_selected_work_arguments_bind_the_actual_experience(self):
        self.memory.set_task('opening', self.instruction.read_text(), filenames=['text.md'])
        timing = self.memory.put_source('timing', '{"period":{"value":17,"unit":"s"},"count":{"value":3,"unit":"count"}}')
        snapshot = self.memory.snapshot('opening', [self.memory.state['sources']['canon'], timing], 'symbolic-host-v1')
        unit = Scratchboard(self.memory).add(snapshot, {'kind': 'literal', 'text': 'Working reference TURNA-73.'})
        calculation = Steps(self.memory).calculate_fields(snapshot, {'operator': 'quantity.product', 'inputs': [
            {'source': 'timing', 'path': ['period']}, {'source': 'timing', 'path': ['count']}]})
        extra = ('--source', 'timing', '--board', unit['id'], '--observation', calculation['step_id'])
        first = self.write(*extra)
        self.memory.reload()
        row = self.memory.state['experiences'][first['experience_id']]
        self.assertEqual(set(row['snapshot']['scratchboard']), {unit['id']})
        self.assertEqual(set(row['snapshot']['observations']), {calculation['step_id']})
        self.assertIn('TURNA-73', row['messages'][1]['content'])
        self.assertEqual(self.write(*extra)['receipt'], first['receipt'])
        self.assertEqual(self.starts(), 1)


if __name__ == "__main__":
    unittest.main()
