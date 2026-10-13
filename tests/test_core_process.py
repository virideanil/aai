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
        time.sleep(60)
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
            try:
                os.killpg(self.core.child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.core.close()
        self.temporary.cleanup()

    def start(self, mode):
        self.core = JsonlCore({
            'precision_profile': PRECISION_PROFILE,
            'manifest': str(self.manifest),
            'source_files': [str(self.worker)],
            'command': [sys.executable, str(self.worker), mode, str(self.root / 'descendant.pid')],
            'request_seconds': .4,
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


if __name__ == '__main__':
    unittest.main()
