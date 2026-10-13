"""Real worker processes exercise request failure and descendant cleanup."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from nanoai.core import JsonlCore
from nanoai.precision import PRECISION_PROFILE


WORKER = r'''
import json, os, pathlib, subprocess, sys, time
mode, marker = sys.argv[1:]
for line in sys.stdin:
    request = json.loads(line)
    if request['command'] in ('init', 'load'):
        print(json.dumps({'id': request['id'], 'ok': True, 'result': {}}), flush=True)
        continue
    if mode == 'wrong-id':
        print(json.dumps({'id': request['id'] + 1, 'ok': True, 'result': {}}), flush=True)
    elif mode == 'timeout':
        pathlib.Path(marker).write_text(str(os.getpid()))
        time.sleep(60)
    elif mode == 'delayed':
        time.sleep(.6)
        print(json.dumps({'id': request['id'], 'ok': True, 'result': {'output': 'observed'}}), flush=True)
    elif mode == 'orphan':
        code = "import os,pathlib,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
        subprocess.Popen([sys.executable, '-c', code, marker], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        until = time.monotonic() + 3
        while not pathlib.Path(marker).exists() and time.monotonic() < until:
            time.sleep(.01)
        os._exit(7)
'''


def running(pid):
    result = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True)
    return bool(result.stdout.strip()) and not result.stdout.strip().startswith('Z')


class CoreProcessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.worker = self.root / 'worker.py'
        self.worker.write_text(WORKER)
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text('{}')
        self.core = None

    def tearDown(self):
        if self.core:
            self.core.close()
        self.temporary.cleanup()

    def start(self, mode, duration=.4):
        self.core = JsonlCore({
            'precision_profile': PRECISION_PROFILE,
            'manifest': str(self.manifest),
            'source_files': [str(self.worker)],
            'command': [sys.executable, str(self.worker), mode, str(self.root / 'descendant.pid')],
            'request_seconds': duration,
        }, stderr_path=self.root / 'worker.log')
        return self.core

    def test_parent_exit_still_cleans_its_live_descendant(self):
        core = self.start('orphan')
        with self.assertRaises(RuntimeError):
            core.generate([])
        descendant = int((self.root / 'descendant.pid').read_text())
        self.assertFalse(running(descendant), 'Worker exit left its model descendant running')
        receipt = json.loads((self.root / 'worker.resources.json').read_text())
        self.assertTrue(receipt['child_reaped'])
        self.assertEqual(receipt['exit_code'], 7)

    def test_deadline_cancels_and_reaps_worker(self):
        core = self.start('timeout')
        with self.assertRaises(TimeoutError):
            core.generate([])
        self.assertIsNotNone(core.child.poll())
        core.close()  # Close remains safe after failure and context-manager exit.

    def test_wrong_response_identity_closes_worker(self):
        core = self.start('wrong-id')
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            core.generate([])
        self.assertIsNotNone(core.child.poll())

    def test_natural_completion_has_no_implicit_request_deadline(self):
        core = self.start('delayed', None)
        self.assertEqual(core.generate([])['output'], 'observed')
        self.assertIsNone(core.child.poll())

    def test_cleanup_signal_error_preserves_request_error_and_receipt(self):
        core = self.start('wrong-id')
        signal_group = os.killpg

        def denied_term(group, sig):
            if sig == signal.SIGTERM:
                raise PermissionError(1, 'injected group signal denial')
            signal_group(group, sig)

        with patch('nanoai.core.os.killpg', side_effect=denied_term):
            with self.assertRaisesRegex(ValueError, 'identity mismatch') as failure:
                core.generate([])
        receipt = json.loads((self.root / 'worker.resources.json').read_text())
        self.assertEqual(receipt['primary_error']['type'], 'ValueError')
        self.assertTrue(receipt['child_reaped'])
        self.assertTrue(receipt['process_group_stopped'])
        self.assertEqual(receipt['remaining_processes'], [])
        self.assertTrue(any(row['type'] == 'PermissionError' for row in receipt['cleanup_errors']))
        self.assertTrue(any('cleanup' in note.casefold() for note in failure.exception.__notes__))

    def test_failed_cleanup_observation_is_unknown_and_can_be_retried(self):
        core = self.start('wrong-id')
        with patch('nanoai.core.live_group_members', side_effect=PermissionError(1, 'injected inventory denial')):
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                core.generate([])
        receipt = json.loads((self.root / 'worker.resources.json').read_text())
        self.assertTrue(receipt['child_reaped'])
        self.assertIsNone(receipt['process_group_stopped'])
        self.assertIsNone(receipt['remaining_processes'])
        self.assertFalse(core.closed)
        core.close()
        receipt = json.loads((self.root / 'worker.resources.json').read_text())
        self.assertTrue(receipt['process_group_stopped'])
        self.assertTrue(receipt['cleanup_errors'], 'Retry must retain the earlier cleanup failure')

    def test_real_sigint_preserves_cancellation_and_reaps_worker(self):
        config = {
            'precision_profile': PRECISION_PROFILE, 'manifest': str(self.manifest),
            'source_files': [str(self.worker)], 'request_seconds': None,
            'command': [sys.executable, str(self.worker), 'timeout', str(self.root / 'interrupt.pid')],
        }
        config_path = self.root / 'core.json'
        config_path.write_text(json.dumps(config))
        driver = """
import json, sys
from pathlib import Path
from nanoai.core import JsonlCore
with JsonlCore(json.loads(Path(sys.argv[1]).read_text()), stderr_path=Path(sys.argv[1]).with_name('interrupt.log')) as core:
    core.generate([])
"""
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
        process = subprocess.Popen([sys.executable, '-c', driver, str(config_path)],
                                   env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            marker = self.root / 'interrupt.pid'
            deadline = time.monotonic() + 10
            while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.025)
            self.assertTrue(marker.exists(), 'Worker did not reach the request')
            os.kill(process.pid, signal.SIGINT)
            _, stderr = process.communicate(timeout=10)
            self.assertIn(b'KeyboardInterrupt', stderr)
            self.assertNotEqual(process.returncode, 0)
            receipt = json.loads((self.root / 'interrupt.resources.json').read_text())
            self.assertEqual(receipt['primary_error']['type'], 'KeyboardInterrupt')
            self.assertTrue(receipt['child_reaped'])
            self.assertTrue(receipt['process_group_stopped'])
            self.assertFalse(running(int(marker.read_text())))
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=3)
            marker = self.root / 'interrupt.pid'
            if marker.exists() and running(int(marker.read_text())):
                try:
                    os.killpg(int(marker.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == '__main__':
    unittest.main()
