import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from evaluate_meaning_learning import assessed_artifact,messages,observe_counterexample
from nanoai.agent import DEFAULT_IDENTITY
from nanoai.experience import Experience
from nanoai.memory import Memory


class MeaningArtifactTests(unittest.TestCase):
    def test_joint_counterexample_runs_through_actual_step_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory=Memory(tmp);memory.initialize(DEFAULT_IDENTITY)
            result=observe_counterexample(memory,{'alternatives':['x',{'not':'x'}]},
                {'alternatives':['y',{'not':'y'}]},{'x':'the door is open','y':'the lamp is lit'},
                task='linked-tests',source_basis='original-model')
            self.assertEqual(result['actual_first_traces'],[(0,0),(1,1)])
            self.assertEqual(result['actual_second_traces'],[(0,1),(1,0)])
            self.assertEqual(len(result['observations']),8)
            recovered=Memory(tmp)
            self.assertEqual(len(recovered.state['step_observations']),12)

    def test_real_source_record_generation_assessment_and_reload(self):
        row={'id':'negation','direction':'english_to_math','family':'negation','split':'train',
             'question':'Express the statement as a formula.','observations':['The door is closed.'],
             'target':'{"not":"x"}',
             'criteria':{'direction':'english_to_math','symbols':{'x':'the door is open'},
                         'expression':{'not':'x'},'assignment':{'x':1}}}
        generation={'text':row['target'],'token_ids':[1,2,3],'finish_reason':'stop','policy':'greedy-token-v1'}
        with tempfile.TemporaryDirectory() as tmp:
            memory=Memory(tmp);memory.initialize(DEFAULT_IDENTITY)
            result=assessed_artifact(memory,row,'collection',generation,messages(row),
                source_basis='full-original-model',corpus_identity='corpus-v1',evaluator='truth-table-v1')
            self.assertTrue(result['correct'])
            self.assertIn(row['target'],(Path(tmp)/result['receipt']['path']).read_text())
            recovered=Experience(Memory(tmp)).export(result['experience_id'])
            self.assertEqual(recovered['eligibility']['status'],'checked')
            self.assertEqual(recovered['snapshot']['sources'].keys(),{'source-collection-negation'})


if __name__=='__main__':unittest.main()
