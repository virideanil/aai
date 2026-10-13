"""Check the learning signal against an independently differentiated expectation."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from evaluate_outcome_learning import outcome_loss, outcome_row


class OutcomeLearningTests(unittest.TestCase):
    def test_behavior_weighted_gradient_equals_exact_expected_reward_gradient(self):
        logits=torch.tensor([.2,-.5,.9,.4,-.1],dtype=torch.float64,requires_grad=True)
        rewards=torch.tensor([0.,0.,1.,0.,0.],dtype=torch.float64)
        loss=sum(.2*outcome_loss(logits,{'chosen_index':i,'propensity':.2,'reward':float(r)})
                 for i,r in enumerate(rewards))
        actual=torch.autograd.grad(loss,logits)[0]
        exact=torch.autograd.grad(-(logits.softmax(-1)*rewards).sum(),logits)[0]
        torch.testing.assert_close(actual,exact,rtol=1e-12,atol=1e-12)

    def test_oracle_labels_do_not_enter_chosen_outcome_training_row(self):
        row={'schema':'nanoai.experience.v1','eligibility':{'status':'checked'},
            'metadata':dict(id='i',family='f',split='train',mode='first',language='en'),
            'id':'experience','question':'where','observations':['here'],
            'candidates':[{'id':str(i),'content':str(i)} for i in range(5)],
            'behavior':{'policy':'uniform-v1','probabilities':[.2]*5,'selected_id':'2'},
            'assessment':{'receipt':{'path':'checked'},'observed_outcome':'yes','replayable_targets':{'target_id':'2'}}}
        changed=deepcopy(row);changed['assessment']['replayable_targets']={'target_id':'4','verdicts':'different'}
        self.assertEqual(outcome_row(row),outcome_row(changed))
        self.assertNotIn('rubric',outcome_row(row))
        changed['behavior']['probabilities']=[0.,0.,1.,0.,0.]
        with self.assertRaises(ValueError):outcome_row(changed)


if __name__=='__main__':unittest.main()
