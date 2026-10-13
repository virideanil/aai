import io
from pathlib import Path
import tempfile
import unittest

import torch

from nanoai.core import digest_bytes
from nanoai.learning import Learner, checkpoint_bytes, checked_episode, load_checkpoint


class LearningStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        self.model = Learner().eval()

    def test_complete_recurrent_state_resumes_same_scores(self):
        observations = torch.randn(4, 64)
        question, candidates = torch.randn(64), torch.randn(5, 64)
        with torch.no_grad():
            full = self.model.advance(observations)
            first = self.model.advance(observations[:2])
            resumed = self.model.advance(observations[2:], saved=first.clone())
            self.assertTrue(torch.equal(full, resumed))
            for a, b in zip(self.model.decide(full, question, candidates), self.model.decide(resumed, question, candidates)):
                self.assertTrue(torch.equal(a, b))
        with self.assertRaises(ValueError): self.model.advance(torch.full((1, 64), float("nan")))

    def test_all_shared_components_receive_actual_gradients(self):
        native, tokens = torch.randn(8, 2048), torch.randint(0, 256, (8, 32))
        encoded = self.model.encode(native, tokens)
        states = self.model.advance(encoded[:2])
        action, evidence = self.model.decide(states, encoded[2], encoded[3:])
        loss = torch.nn.functional.cross_entropy(action[None], torch.tensor([1]))+torch.nn.functional.cross_entropy(evidence, torch.tensor([0, 1, 2, 1, 0]))
        loss.backward()
        for name in ("bytes", "local", "native", "combine", "workspace", "choice", "evidence"):
            parameters = getattr(self.model, name).parameters()
            self.assertTrue(any(p.grad is not None and bool((p.grad != 0).any()) for p in parameters), name)

    def test_checkpoint_preserves_parameters_and_rejects_wrong_basis_or_dtype(self):
        data = checkpoint_bytes(self.model, feature_basis="fixture", corpus_identity="corpus", experience_ids=["e"], training={})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.pt"; path.write_bytes(data)
            restored, _ = load_checkpoint(path, expected=digest_bytes(data), feature_basis="fixture")
            self.assertTrue(all(torch.equal(v, restored.state_dict()[k]) for k,v in self.model.state_dict().items()))
            with self.assertRaises(ValueError): load_checkpoint(path, expected=digest_bytes(data), feature_basis="other")
            record = torch.load(io.BytesIO(data), weights_only=True)
            key = next(iter(record["state"]))
            record["state"][key] = record["state"][key].half()
            torch.save(record, path)
            with self.assertRaises(ValueError): load_checkpoint(path, expected=digest_bytes(path.read_bytes()), feature_basis="fixture")

    def test_unassessed_or_invalidated_use_does_not_supply_training_targets(self):
        for status in ("unassessed", "invalidated"):
            with self.assertRaises(ValueError): checked_episode({"schema":"nanoai.experience.v1", "eligibility":{"status":status}})


if __name__ == "__main__":
    unittest.main()
