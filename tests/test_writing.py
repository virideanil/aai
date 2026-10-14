from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from nanoai.agent import DEFAULT_IDENTITY
from nanoai.catalogue import LanguageCatalogue
from nanoai.experience import Experience
from nanoai.memory import Memory, StaleState
from nanoai.scratchboard import Scratchboard
from nanoai.steps import Steps
from nanoai.writing import Writing


class FixedCore:
    identity = "writing-test-core"

    def __init__(self, text="Mira wound the blue clock."):
        self.text, self.calls = text, 0

    def generate(self, messages, *, max_tokens, trace):
        self.calls += 1
        self.messages = deepcopy(messages)
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

    def test_recovered_failure_agrees_with_status_and_never_restarts_the_core(self):
        core = FixedCore("")
        first = self.owner.run("opening", "Write one sentence.", ["canon"], core)
        self.assertTrue(first["generation_finished"])
        self.assertFalse(first["complete"])
        self.assertEqual(first["outcome"]["status"], "failed")
        fresh = Memory(self.root)
        again = Writing(fresh).run("opening", "Write one sentence.", ["canon"], core)
        self.assertFalse(again["complete"])
        self.assertEqual(core.calls, 1)
        self.assertEqual(again["model_calls"], 0)
        self.assertEqual(again["outcome"], fresh.status()["artifacts"][0]["outcome"])
        self.assertEqual(fresh.completed("opening")["remaining"], ["text.md"])

    def test_recovered_assessed_failure_preserves_the_draft_and_its_verdict(self):
        core = FixedCore()
        first = self.owner.run("opening", "Write one sentence.", ["canon"], core)
        self.assertEqual(first["outcome"]["status"], "unassessed")
        Experience(self.memory).assess_generation(first["experience_id"], rubric={
            "scope": "Draft length", "criteria": {}, "training_target": None},
            evaluator_identity="length-test", evaluate=lambda text, _: {"short": len(text) < 5})
        again = Writing(Memory(self.root)).run("opening", "Write one sentence.", ["canon"], core)
        self.assertFalse(again["complete"])
        self.assertTrue(again["generation_finished"])
        self.assertEqual(again["receipt"], first["receipt"])
        self.assertEqual(again["outcome"]["status"], "failed")
        self.assertEqual(core.calls, 1)

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

    def working_basis(self, core):
        timing = self.memory.put_source('timing', '{"period":{"value":17,"unit":"s"},"count":{"value":3,"unit":"count"}}')
        self.memory.set_task('opening', 'Write one sentence.', filenames=['text.md'])
        return self.memory.snapshot('opening', [self.memory.state['sources']['canon'], timing], core.identity)

    def test_selected_work_is_used_recovered_and_replaced_with_its_dependencies(self):
        core = FixedCore()
        snapshot = self.working_basis(core)
        board = Scratchboard(self.memory)
        unit = board.add(snapshot, {'kind': 'literal', 'text': 'Working reference TURNA-73.'})
        calculation = Steps(self.memory).calculate_fields(snapshot, {'operator': 'quantity.product', 'inputs': [
            {'source': 'timing', 'path': ['period']}, {'source': 'timing', 'path': ['count']}]})
        Steps(self.memory).review(calculation['step_id'], 'confirmed', reviewer='test', reason='Three complete periods.')
        options = {'observation_ids': [calculation['step_id']], 'scratchboard_ids': [unit['id']]}
        first = self.owner.run('opening', 'Write one sentence.', ['canon', 'timing'], core, **options)
        work = json.loads(core.messages[1]['content'])
        self.assertEqual(work['identity'], DEFAULT_IDENTITY)
        self.assertEqual(work['working_units'][0]['preview'], 'Working reference TURNA-73.')
        self.assertEqual(work['observed_calculations'][0]['value'], 51)
        self.assertEqual(work['observed_calculations'][0]['review']['decision'], 'confirmed')
        board.add(snapshot, {'kind': 'literal', 'text': 'Unselected independent work.'})
        again = Writing(Memory(self.root)).run('opening', 'Write one sentence.', ['canon', 'timing'], core, **options)
        self.assertEqual(core.calls, 1)
        self.assertEqual(again['receipt'], first['receipt'])
        replacement = board.revise(unit['id'], {'kind': 'literal', 'text': 'Working reference TURNA-74.'}, reason='Corrected code.')
        with self.assertRaises(StaleState):
            self.owner.run('opening', 'Write one sentence.', ['canon', 'timing'], core, **options)
        self.assertEqual(core.calls, 1)
        options['scratchboard_ids'] = [replacement['id']]
        revised = self.owner.run('opening', 'Write one sentence.', ['canon', 'timing'], core, **options)
        self.assertEqual(core.calls, 2)
        self.assertIn('TURNA-74', core.messages[1]['content'])
        self.assertNotEqual(revised['experience_id'], first['experience_id'])
        self.assertTrue(Path(first['artifact_path']).is_file())

    def test_rejected_operation_and_foreign_task_work_reject_before_generation(self):
        core = FixedCore()
        snapshot = self.working_basis(core)
        calculation = Steps(self.memory).calculate_fields(snapshot, {'operator': 'quantity.product', 'inputs': [
            {'source': 'timing', 'path': ['period']}, {'source': 'timing', 'path': ['count']}]})
        Steps(self.memory).review(calculation['step_id'], 'rejected', reviewer='test', reason='Task requires an exception.')
        with self.assertRaises(StaleState):
            self.owner.run('opening', 'Write one sentence.', ['canon', 'timing'], core, observation_ids=[calculation['step_id']])
        self.memory.set_task('other', 'A different task.')
        foreign = self.memory.snapshot('other', [self.memory.state['sources']['canon']], core.identity)
        unit = Scratchboard(self.memory).add(foreign, {'kind': 'literal', 'text': 'Belongs elsewhere.'})
        with self.assertRaises(StaleState):
            self.owner.run('opening', 'Write one sentence.', ['canon'], core, scratchboard_ids=[unit['id']])
        self.assertEqual(core.calls, 0)

    def test_writing_preserves_partial_work_and_rejects_correction_during_generation(self):
        core = FixedCore()
        snapshot = self.working_basis(core)
        board = Scratchboard(self.memory)
        slots = board.reserve(snapshot, 10000000, constraints='Unresolved future content.')
        literal = board.add(snapshot, {'kind': 'literal', 'text': 'Ş' * 1500})
        original_generate = core.generate
        def change_during_generation(*args, **kwargs):
            result = original_generate(*args, **kwargs)
            board.revise(literal['id'], {'kind': 'literal', 'text': 'Changed during generation.'}, reason='Concurrent correction.')
            return result
        core.generate = change_during_generation
        with self.assertRaises(StaleState):
            self.owner.run('opening', 'Write one sentence.', ['canon', 'timing'], core, scratchboard_ids=[slots['id'], literal['id']])
        work = json.loads(core.messages[1]['content'])
        self.assertEqual(work['working_units'][0]['unresolved_slots'], 10000000)
        self.assertFalse(work['working_units'][0]['preview_complete'])
        self.assertFalse(work['working_units'][1]['representation_complete'])
        self.assertFalse(work['working_units'][1]['preview_complete'])
        self.assertEqual(core.calls, 1)
        self.assertEqual(self.memory.state['receipts'], {})


if __name__ == "__main__":
    unittest.main()
