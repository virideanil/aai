"""A versioned F32 learner over original-BF16 features and exact UTF-8 bytes.

The shared encoder, recurrent workspace and two decision heads train together.
The backbone is a separately identified frozen instrument. Evidence classes
have the scope of the supplying evaluator, rather than universal truth.
"""
from __future__ import annotations

from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from .core import digest_bytes, file_digest, process_memory, swap_bytes
from .judgments import CHANNELS
from .memory import canonical


class Features:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory/"manifest.json").read_text())
        if self.manifest.get("schema") != "nanoai.features.v1" or self.manifest["status"] != "completed":
            raise ValueError("A completed native feature manifest is required")
        self.basis = self.manifest["basis"]
        self.rows = {}
        for key, entry in self.manifest["rows"].items():
            path = self.directory/entry["file"]
            if path.parent != self.directory or file_digest(path) != entry["digest"]:
                raise ValueError("Native feature bytes changed")
            row = json.loads(path.read_text())
            vector = row["vector"]
            if (row["basis"] != self.basis or row["text_identity"] != key
                    or digest_bytes(row["text"].encode()) != key or row["dtype"] != "float32"
                    or len(vector) != 2048 or any(not math.isfinite(x) for x in vector)):
                raise ValueError("Native feature provenance or extent differs")
            self.rows[row["text"]] = row
        self.identity = file_digest(self.directory/"manifest.json")

    def tensors(self, texts):
        if any(not isinstance(t, str) or not t or len(t.encode()) > 384 for t in texts):
            raise ValueError("Learner text must contain one to 384 UTF-8 bytes")
        raw = [list(t.encode()) for t in texts]
        width = max(map(len, raw))
        # 256 is padding, outside the complete byte alphabet.
        tokens = torch.full((len(raw), width), 256, dtype=torch.long)
        for i, value in enumerate(raw):
            tokens[i, :len(value)] = torch.tensor(value)
        native = torch.tensor([self.rows[t]["vector"] for t in texts], dtype=torch.float32)
        return native, tokens


class Learner(nn.Module):
    def __init__(self, *, hidden=64, no_memory=False, native_dimensions=2048):
        super().__init__()
        if hidden != 64 or native_dimensions != 2048 or type(no_memory) is not bool:
            raise ValueError("Unqualified learner shape")
        self.config = dict(hidden=hidden, no_memory=no_memory, native_dimensions=native_dimensions)
        self.bytes = nn.Embedding(257, 32, padding_idx=256)
        self.local = nn.Conv1d(32, hidden, 5, padding=2)
        self.native = nn.Linear(native_dimensions, hidden)
        self.combine = nn.Linear(2*hidden, hidden)
        self.normal = nn.LayerNorm(hidden)
        self.workspace = nn.GRU(hidden, hidden, batch_first=True)
        self.choice = nn.Sequential(nn.Linear(6*hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.evidence = nn.Sequential(nn.Linear(6*hidden, hidden), nn.Tanh(), nn.Linear(hidden, 3))

    def encode(self, native, tokens):
        mask = tokens.ne(256)
        local = torch.tanh(self.local(self.bytes(tokens).transpose(1, 2))).transpose(1, 2)
        local = (local*mask.unsqueeze(-1)).sum(1)/mask.sum(1, keepdim=True)
        native = self.native(F.normalize(native, dim=-1))
        return self.normal(torch.tanh(self.combine(torch.cat((native, local), dim=-1))))

    def advance(self, encoded, *, saved=None):
        """Retain every recurrent output required by the attention read.

        The caller validates the candidate, feature basis and exact dependencies
        of a saved tensor before passing it here.
        """
        if encoded.ndim != 2 or encoded.shape[1] != 64 or encoded.dtype != torch.float32:
            raise ValueError("Workspace input basis or shape")
        if self.config["no_memory"]:
            encoded, saved = encoded[-1:], None
        if saved is not None and (saved.ndim != 2 or saved.shape[1] != 64 or saved.dtype != torch.float32):
            raise ValueError("Saved workspace shape or dtype")
        if not bool(torch.isfinite(encoded).all()) or saved is not None and not bool(torch.isfinite(saved).all()):
            raise ValueError("Recurrent workspace must be finite")
        initial = None if saved is None else saved[-1:].unsqueeze(0)
        # A fixed one-observation call shape preserves arithmetic across a
        # saved/restarted prefix; a batched GRU can change its matrix reduction.
        outputs = []
        for observation in encoded:
            output, initial = self.workspace(observation[None, None, :], initial)
            outputs.append(output[0, 0])
        states = torch.stack(outputs)
        return states if saved is None else torch.cat((saved, states), dim=0)

    def decide(self, states, question, candidates):
        count = candidates.shape[0]
        goal = question.expand(count, -1)
        last = states[-1:].expand(count, -1)
        attention = torch.softmax((candidates+goal)@states.T/math.sqrt(64), dim=-1)
        read = attention@states
        joined = torch.cat((last, read, candidates, goal, read*candidates, last*goal), dim=-1)
        return self.choice(joined).squeeze(-1), self.evidence(joined)

    def forward(self, episode, encoded):
        states = self.advance(torch.stack([encoded[t] for t in episode["observations"]]))
        logits, verdicts = self.decide(states, encoded[episode["question"]],
            torch.stack([encoded[c["content"]] for c in episode["candidates"]]))
        return logits, verdicts, states


def encoded_texts(model, features, episodes):
    texts = sorted({text for e in episodes for text in
        [e["question"], *e["observations"], *[c["content"] for c in e["candidates"]]]})
    native, tokens = features.tensors(texts)
    encoded = model.encode(native, tokens)
    return dict(zip(texts, encoded))


def targets(episode):
    candidates = episode["candidates"]
    target = next(i for i, c in enumerate(candidates) if c["id"] == episode["rubric"]["target_id"])
    classes = [CHANNELS.index(episode["rubric"]["verdicts"][c["id"]]) for c in candidates]
    return torch.tensor([target]), torch.tensor(classes)


@torch.no_grad()
def evaluate(model, features, episodes):
    model.eval()
    encoded = encoded_texts(model, features, episodes)
    rows, loss = [], 0.0
    for episode in episodes:
        logits, verdicts, _ = model(episode, encoded)
        target, classes = targets(episode)
        loss += float(F.cross_entropy(logits.unsqueeze(0), target)+.4*F.cross_entropy(verdicts, classes))
        chosen = int(logits.argmax())
        rows.append({"id": episode["id"], "selected_id": episode["candidates"][chosen]["id"],
            "correct": chosen == int(target[0]), "scores": logits.tolist(),
            "evidence_probabilities": verdicts.softmax(-1).tolist(),
            "evidence_correct": int((verdicts.argmax(-1) == classes).sum()),
            "mode": episode["mode"], "family": episode["family"]})
    return {"count": len(rows), "correct": sum(r["correct"] for r in rows),
        "accuracy": sum(r["correct"] for r in rows)/len(rows), "loss": loss/len(rows),
        "evidence_correct": sum(r["evidence_correct"] for r in rows),
        "evidence_count": sum(len(e["candidates"]) for e in episodes), "rows": rows}


def train(features, train_episodes, validation, *, seed, no_memory=False, parent=None, epochs=300):
    if not train_episodes or not validation or any(e["split"] != "train" for e in train_episodes):
        raise ValueError("Only checked training episodes can update the candidate")
    if any(e["split"] != "validation" for e in validation):
        raise ValueError("Select the candidate with its frozen validation families")
    if {e["family"] for e in train_episodes} & {e["family"] for e in validation}:
        raise ValueError("Training and validation families overlap")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    model = Learner(no_memory=no_memory) if parent is None else deepcopy(parent)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    rng = random.Random(seed)
    start = time.monotonic()
    best, history, resources = None, [], []
    swap_before = swap_bytes()
    for epoch in range(epochs+1):
        if time.monotonic()-start > 300:
            raise TimeoutError("Candidate training exceeded its whole-run bound")
        if epoch % 5 == 0 or epoch == epochs:
            sample = {"seconds": time.monotonic()-start, **process_memory(os.getpid()), "host_swap_bytes": swap_bytes()}
            resources.append(sample)
            if (sample["rss_bytes"] > 2*1024**3
                    or sample["physical_footprint_bytes"] is not None and sample["physical_footprint_bytes"] > 2*1024**3
                    or swap_before is not None and sample["host_swap_bytes"]-swap_before > 512*1024**2):
                raise RuntimeError("Candidate training resource budget reached")
            value = evaluate(model, features, validation)
            score = (value["correct"], -value["loss"])
            history.append({"epoch": epoch, "validation_correct": value["correct"], "validation_loss": value["loss"]})
            if best is None or score > best[0]:
                best = (score, deepcopy(model.state_dict()), epoch)
        if epoch == epochs:
            break
        model.train()
        order = list(train_episodes); rng.shuffle(order)
        for offset in range(0, len(order), 24):
            batch = order[offset:offset+24]
            optimizer.zero_grad(set_to_none=True)
            encoded = encoded_texts(model, features, batch)
            losses = []
            for episode in batch:
                logits, verdicts, _ = model(episode, encoded)
                target, classes = targets(episode)
                losses.append(F.cross_entropy(logits.unsqueeze(0), target)+.4*F.cross_entropy(verdicts, classes))
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise ValueError("Candidate objective became nonfinite")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
    model.load_state_dict(best[1]); model.eval()
    return model, {"seed": seed, "epochs": epochs, "selected_epoch": best[2], "history": history,
                   "seconds": time.monotonic()-start, "parameters": sum(p.numel() for p in model.parameters()),
                   "train_ids": [e["id"] for e in train_episodes], "validation_ids": [e["id"] for e in validation],
                   "resources": resources, "swap_before": swap_before}


def checkpoint_bytes(model, *, feature_basis, corpus_identity, experience_ids, training):
    if any(p.dtype != torch.float32 or not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("Candidate parameters must be finite F32")
    record = {"schema": "nanoai.learner.v1", "config": model.config, "state": model.state_dict(),
        "feature_basis": feature_basis, "corpus_identity": corpus_identity,
        "experience_ids": list(experience_ids), "training": training,
        "implementation": file_digest(__file__)}
    stream = io.BytesIO(); torch.save(record, stream)
    return stream.getvalue()


def load_checkpoint(path, *, expected, feature_basis):
    if file_digest(path) != expected:
        raise ValueError("Candidate checkpoint identity differs")
    record = torch.load(path, weights_only=True, map_location="cpu")
    if (record["schema"] != "nanoai.learner.v1" or record["implementation"] != file_digest(__file__)
            or record["feature_basis"] != feature_basis):
        raise ValueError("Candidate implementation or feature basis changed")
    if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in record["state"].values()):
        raise ValueError("Stored candidate parameters must preserve finite F32 values")
    model = Learner(**record["config"])
    model.load_state_dict(record["state"], strict=True); model.eval()
    if any(p.dtype != torch.float32 or not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("Candidate dtype or values differ")
    return model, record


def restore_workspace(memory, node_id, *, learner_id):
    """Read exact F32 state under its original candidate and source dependencies."""
    import numpy as np
    from .scratchboard import Scratchboard
    board = Scratchboard(memory)
    with memory.locked():
        node = deepcopy(board._validate(node_id))
    raw = board.read_tensor(node_id, core_identity=learner_id)
    if node["spec"]["dtype"] != "f32" or len(node["spec"]["shape"]) != 2 or node["spec"]["shape"][1] != 64:
        raise ValueError("Saved recurrent workspace format differs")
    return torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy()).reshape(node["spec"]["shape"])


def checked_episode(record):
    """Training reads a current assessed journal export, including failed acts."""
    if record["schema"] != "nanoai.experience.v1" or record["eligibility"]["status"] != "checked":
        raise ValueError("Training requires a current checked experience")
    assessment = record["assessment"]
    if assessment is None or not assessment.get("receipt"):
        raise ValueError("Training needs the observed chosen effect")
    metadata = record["metadata"]
    return {k: metadata[k] for k in ("id", "family", "split", "language", "mode")} | {
        "question": record["question"], "observations": record["observations"],
        "candidates": record["candidates"], "rubric": assessment["replayable_targets"],
        "experience_id": record["id"]}


@torch.no_grad()
def act(memory, features, model, task_id, source_id, candidates, *, learner_id):
    """Use the current exact source to choose and publish a local document."""
    from .experience import Experience, policy_choice
    from .scratchboard import Scratchboard
    with memory.locked():
        source = memory.state["sources"][source_id]
        task = memory.state["tasks"][task_id]
        learner = memory.state["learners"][learner_id]
        if learner["feature_basis"] != features.basis:
            raise ValueError("Action feature basis differs from the candidate")
        snapshot = memory.snapshot(task_id, [source], learner_id)
        snapshot["learner"] = learner_id
        episode = {"question": task["goal"], "observations": json.loads(source["text"])["observations"],
                   "candidates": deepcopy(candidates)}
        checkpoint_path = memory._artifact_path(learner["path"])
    loaded, _ = load_checkpoint(checkpoint_path, expected=learner_id, feature_basis=features.basis)
    if any(not torch.equal(value, model.state_dict()[key]) for key, value in loaded.state_dict().items()):
        raise ValueError("The executing learned parameters differ from their registered candidate")
    model.eval()
    start = time.monotonic()
    encoded = encoded_texts(model, features, [episode])
    logits, verdicts, states = model(episode, encoded)
    behavior = policy_choice([c["id"] for c in candidates], scores=logits.tolist())
    workspace = Scratchboard(memory).tensor(snapshot, states.detach().numpy().astype("<f4").tobytes(),
        shape=list(states.shape), dtype="f32", origin={"operation": "learner.workspace",
            "learner": learner_id, "features": features.identity, "observations": len(episode["observations"])})
    judgments = [{"candidate_id": c["id"], "verdict": CHANNELS[int(v.argmax())],
        "reason": "Provisional learned evidence classification in the declared task domain",
        "uncertainty": "undecided" if int(v.argmax()) == CHANNELS.index("maybe") else None}
        for c, v in zip(candidates, verdicts)]
    owner = Experience(memory)
    experience = owner.record(snapshot, question=episode["question"], observations=episode["observations"],
        candidates=candidates, behavior=behavior, judgments=judgments,
        representation_identity=features.basis, source_id=source_id, workspace_node=workspace["id"],
        metadata={"action_seconds_before_publication": time.monotonic()-start})
    receipt = owner.execute(experience["id"])
    return {"experience": experience, "receipt": receipt, "workspace": workspace,
            "elapsed_seconds": time.monotonic()-start, "scores": logits.tolist()}
