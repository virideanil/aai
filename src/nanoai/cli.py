"""Explicit local commands for proposals, reviewed notes and observed recovery."""
import argparse
import json
from pathlib import Path

from .agent import DEFAULT_IDENTITY, calculate, run
from .core import JsonlCore
from .memory import Memory


def main():
    p = argparse.ArgumentParser(prog="nanoai")
    p.add_argument("--state", required=True, type=Path)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    source = sub.add_parser("source")
    source.add_argument("id")
    source.add_argument("file", type=Path)
    source.add_argument("--expected-revision", type=int, default=0)
    source.add_argument("--note", default="")
    task = sub.add_parser("task")
    task.add_argument("id")
    task.add_argument("goal")
    task.add_argument("--expected-revision", type=int, default=0)
    task.add_argument("--filename", default="result.md")
    sub.add_parser("status")
    sub.add_parser("recover")
    resume = sub.add_parser("resume")
    resume.add_argument("task_id")
    sub.add_parser("steps")
    review = sub.add_parser("review-step")
    review.add_argument("step_id")
    review.add_argument("decision", choices=("confirmed", "rejected", "cancelled"))
    review.add_argument("--reviewer", required=True)
    review.add_argument("--reason", required=True)
    note = sub.add_parser("note")
    note.add_argument("step_id")
    note.add_argument("--language", choices=("en", "tr"), default="en")
    note.add_argument("--filename")
    note.add_argument("--commit", action="store_true")
    dictionary = sub.add_parser("dictionary")
    dictionary.add_argument("task_id")
    dictionary.add_argument("--source", action="append", required=True)
    dictionary.add_argument("--vocabulary", type=Path)
    calculation = sub.add_parser("calculate")
    calculation.add_argument("task_id")
    calculation.add_argument("--core-config", type=Path, required=True)
    calculation.add_argument("--domain", choices=("integer", "quantity"), default="quantity")
    calculation.add_argument("--source", action="append")
    worker = sub.add_parser("run")
    worker.add_argument("task_id")
    worker.add_argument("--core-config", type=Path, required=True)
    worker.add_argument("--commit", action="store_true")
    worker.add_argument("--observation", action="append", default=[])
    worker.add_argument("--source", action="append")
    args = p.parse_args()
    memory = Memory(args.state)
    if args.command == "init":
        memory.initialize(DEFAULT_IDENTITY)
        output = memory.status()
    elif args.command == "source":
        output = memory.put_source(args.id, args.file.read_text(), expected_revision=args.expected_revision, note=args.note)
    elif args.command == "task":
        output = memory.set_task(args.id, args.goal, expected_revision=args.expected_revision, filenames=[args.filename])
    elif args.command == "status":
        output = memory.status()
    elif args.command == "recover":
        output = {"reconciled": memory.recover(), "status": memory.status()}
    elif args.command == "resume":
        memory.recover()
        output = memory.completed(args.task_id)
    elif args.command == "steps":
        from .steps import Steps
        output = Steps(memory).status()
    elif args.command == "review-step":
        from .steps import Steps
        output = Steps(memory).review(args.step_id, args.decision, reviewer=args.reviewer, reason=args.reason)
    elif args.command == "note":
        from .documents import quantity_note
        proposal = quantity_note(memory, args.step_id, language=args.language, filename=args.filename)
        receipt = memory.commit(proposal["id"], core_identity=proposal["snapshot"]["core_identity"]) if args.commit else None
        output = {"proposal": proposal, "receipt": receipt,
                  "artifact_path": str(memory.root / receipt["path"]) if receipt else None,
                  "model_calls": 0}
    elif args.command == "dictionary":
        from .harmonizer import Harmonizer
        from .syszero import CommonVocabulary
        from .agent import context
        vocabulary = CommonVocabulary.load(args.vocabulary) if args.vocabulary else None
        snapshot = context(memory, args.task_id, "dictionary-inspection", selected_ids=args.source)["snapshot"]
        output = Harmonizer(memory, vocabulary=vocabulary).view(snapshot)
    else:
        config = json.loads(args.core_config.read_text())
        with JsonlCore(config, stderr_path=args.state / "worker.log") as core:
            output = (calculate(memory, core, args.task_id, domain=args.domain, selected_ids=args.source) if args.command == "calculate"
                      else run(memory, core, args.task_id, commit=args.commit, observation_ids=args.observation, selected_ids=args.source))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
