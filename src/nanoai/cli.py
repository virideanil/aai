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
    write = sub.add_parser("write-phase", help="Generate and journal one editable writing phase")
    write.add_argument("phase_id")
    write.add_argument("instruction", type=Path)
    write.add_argument("--workflow", default="book")
    write.add_argument("--kind", choices=("plan", "draft", "critique", "revision", "continuation", "control"), default="draft")
    write.add_argument("--source", action="append", required=True)
    write.add_argument("--lesson", action="append", default=[], help="A current lesson source installed in this journal")
    write.add_argument("--observation", action="append", default=[], help="A current observed operation in this writing task")
    write.add_argument("--board", action="append", default=[], help="A current working unit in this writing task")
    write.add_argument("--core-config", type=Path, required=True)
    write.add_argument("--max-tokens", type=int, default=None,
                       help="Optional total generation limit; default uses natural completion with thinking enabled")
    write.add_argument("--filename", default="text.md")
    assemble = sub.add_parser("assemble")
    assemble.add_argument("workflow")
    assemble.add_argument("title")
    assemble.add_argument("--source", action="append", required=True)
    lesson = sub.add_parser("lesson-install", help="Install one versioned teaching lesson")
    lesson.add_argument("file", type=Path)
    lesson.add_argument("--expected-revision", type=int, default=0)
    feedback = sub.add_parser("teaching-feedback", help="Bind proposed feedback to exact observed work and evidence")
    feedback.add_argument("experience_id")
    feedback.add_argument("file", type=Path)
    pair = sub.add_parser("teaching-pair", help="Export a current checked training target and its exact prompt")
    pair.add_argument("experience_id")
    resume = sub.add_parser("resume")
    resume.add_argument("task_id")
    sub.add_parser("steps")
    pure = sub.add_parser("step")
    pure.add_argument("task_id")
    pure.add_argument("operator")
    pure.add_argument("arguments", type=Path)
    pure.add_argument("--source", action="append", required=True)
    pure.add_argument("--core-identity", default="symbolic-host-v1")
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
    board_add = sub.add_parser("board-add")
    board_add.add_argument("task_id")
    board_add.add_argument("spec", type=Path)
    board_add.add_argument("--core-identity", default="symbolic-host-v1")
    board_add.add_argument("--source", action="append", default=[])
    reserve = sub.add_parser("board-reserve")
    reserve.add_argument("task_id")
    reserve.add_argument("count", type=int)
    reserve.add_argument("--constraints", required=True)
    reserve.add_argument("--core-identity", default="symbolic-host-v1")
    reserve.add_argument("--source", action="append", default=[])
    board_show = sub.add_parser("board-show")
    board_show.add_argument("node_id")
    board_show.add_argument("--slot", type=int)
    board_bind = sub.add_parser("board-bind")
    board_bind.add_argument("node_id")
    board_bind.add_argument("index", type=int)
    board_bind.add_argument("target_id")
    board_branch = sub.add_parser("board-branch")
    board_branch.add_argument("node_id")
    board_branch.add_argument("--reason", required=True)
    board_revise = sub.add_parser("board-revise")
    board_revise.add_argument("node_id")
    board_revise.add_argument("spec", type=Path)
    board_revise.add_argument("--reason", required=True)
    board_revise.add_argument("--refresh", action="store_true", help="Use the current task and explicitly selected sources")
    board_revise.add_argument("--source", action="append")
    board_refine = sub.add_parser("board-refine")
    board_refine.add_argument("node_id")
    board_refine.add_argument("parts", nargs="+")
    board_refine.add_argument("--reason", required=True)
    board_latest = sub.add_parser("board-latest")
    board_latest.add_argument("node_id")
    board_import = sub.add_parser("board-import")
    board_import.add_argument("task_id")
    board_import.add_argument("node_id")
    board_import.add_argument("--relevance", required=True)
    board_import.add_argument("--reason", required=True)
    board_import.add_argument("--core-identity", default="symbolic-host-v1")
    board_import.add_argument("--source", action="append", required=True)
    board_reshape = sub.add_parser("board-reshape")
    board_reshape.add_argument("node_id")
    board_reshape.add_argument("shape", nargs="+", type=int)
    board_reshape.add_argument("--reason", required=True)
    board_note = sub.add_parser("board-note")
    board_note.add_argument("step_id")
    board_note.add_argument("--language", choices=("en", "tr"), default="en")
    board_document = sub.add_parser("board-document")
    board_document.add_argument("node_id")
    board_document.add_argument("--filename")
    board_document.add_argument("--commit", action="store_true")
    dictionary = sub.add_parser("dictionary")
    dictionary.add_argument("task_id")
    dictionary.add_argument("--source", action="append", required=True)
    dictionary.add_argument("--vocabulary", type=Path)
    dictionary.add_argument("--board", action="append", default=[])
    catalogue = sub.add_parser("catalogue")
    catalogue.add_argument("task_id")
    catalogue.add_argument("--history", action="store_true")
    catalogue.add_argument("--limit", type=int, default=500)
    catalogue.add_argument("--inventory", type=Path)
    observe = sub.add_parser("catalogue-observe")
    observe.add_argument("task_id")
    observe.add_argument("source_id")
    observe.add_argument("--request-id", required=True)
    observe.add_argument("--tokenizer", type=Path, required=True)
    observe.add_argument("--tokenizer-identity", required=True)
    perceive = sub.add_parser("catalogue-perceive")
    perceive.add_argument("task_id")
    perceive.add_argument("source_id")
    perceive.add_argument("--request-id", required=True)
    perceive.add_argument("--bundle", type=Path, required=True)
    perceive.add_argument("--see-executable", type=Path, required=True)
    perceive.add_argument("--source-image", type=Path)
    perceive.add_argument("--region", action="append", type=int, required=True)
    judgment = sub.add_parser("judge")
    judgment.add_argument("task_id")
    judgment.add_argument("question")
    judgment.add_argument("--predicate", type=Path)
    judgment.add_argument("--source", action="append", default=[])
    judgment.add_argument("--core-config", type=Path)
    judgment.add_argument("--core-identity", default="symbolic-host-v1")
    judgment.add_argument("--verdict", choices=("yes", "maybe", "no"))
    judgment.add_argument("--reason")
    from .judgments import MAYBE_CAUSES
    judgment.add_argument("--uncertainty", choices=MAYBE_CAUSES)
    assessment = sub.add_parser("assess-judgment")
    assessment.add_argument("node_id")
    experience = sub.add_parser("experiences")
    experience.add_argument("task_id")
    sub.add_parser("learners")
    trial = sub.add_parser("trial-act", help="Run a registered experimental candidate on the task's local document action")
    trial.add_argument("task_id")
    trial.add_argument("--source", required=True)
    trial.add_argument("--candidate", required=True)
    trial.add_argument("--features", type=Path, required=True)
    trial.add_argument("--choices", type=Path, required=True)
    carrier = sub.add_parser("carry-judgments")
    carrier.add_argument("node_ids", nargs="+")
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
    worker.add_argument("--board", action="append", default=[])
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
    elif args.command == "write-phase":
        from .writing import Writing
        memory.recover()
        config = json.loads(args.core_config.read_text())
        with JsonlCore(config, stderr_path=args.state / "writing-worker.log", lazy=True) as core:
            output = Writing(memory).run(args.phase_id, args.instruction.read_text(), args.source, core,
                workflow=args.workflow, kind=args.kind, max_tokens=args.max_tokens, filename=args.filename,
                lesson_ids=args.lesson, observation_ids=args.observation, scratchboard_ids=args.board)
    elif args.command == "assemble":
        from .writing import Writing
        output = Writing(memory).assemble(args.workflow, args.source, args.title)
    elif args.command == "lesson-install":
        from .teaching import Teaching
        output = Teaching(memory).install(json.loads(args.file.read_text()), expected_revision=args.expected_revision)
    elif args.command == "teaching-feedback":
        from .teaching import Teaching
        output = Teaching(memory).feedback(args.experience_id, **json.loads(args.file.read_text()))
    elif args.command == "teaching-pair":
        from .teaching import Teaching
        output = Teaching(memory).training_pair(args.experience_id)
    elif args.command == "resume":
        memory.recover()
        output = memory.completed(args.task_id)
    elif args.command == "learners":
        from .core import file_digest
        with memory.locked():
            output = {"active_learner": memory.state["active_learner"], "candidates": [
                {**row, "bytes_current": file_digest(memory._artifact_path(row["path"])) == row["id"]}
                for row in memory.state["learners"].values()], "parameter_updates": 0}
    elif args.command == "trial-act":
        import torch
        from .learning import Features, act, load_checkpoint
        from .steps import strict_json
        torch.set_num_threads(4)
        features = Features(args.features)
        with memory.locked():
            candidate = memory.state["learners"][args.candidate]
            checkpoint = memory._artifact_path(candidate["path"])
        model, _ = load_checkpoint(checkpoint, expected=args.candidate, feature_basis=features.basis)
        result = act(memory, features, model, args.task_id, args.source,
                     strict_json(args.choices.read_text()), learner_id=args.candidate)
        output = {**result, "artifact_path": str(memory.root/result["receipt"]["path"]),
                  "mode": "experimental_candidate", "candidate_admitted": memory.state["active_learner"] == args.candidate,
                  "parameter_updates": 0}
    elif args.command == "steps":
        from .steps import Steps
        output = Steps(memory).status()
    elif args.command == "step":
        from .agent import context
        from .steps import Steps, strict_json
        steps = Steps(memory)
        snapshot = context(memory, args.task_id, args.core_identity, selected_ids=args.source)["snapshot"]
        proposed = steps.propose(snapshot, args.operator, strict_json(args.arguments.read_text()))
        try:
            observed = steps.execute(proposed["id"])
        except ValueError as error:
            steps.review(proposed["id"], "rejected", reviewer="pure-operator-contract-v1", reason=str(error))
            raise
        output = {"step": proposed, "observation": observed, "model_calls": 0}
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
    elif args.command == "catalogue-perceive":
        from .perception import attach_perception
        output = attach_perception(memory, args.request_id, args.task_id, args.source_id,
            bundle=args.bundle, executable=args.see_executable, region_ids=args.region, source_image=args.source_image)
    elif args.command in {"catalogue", "catalogue-observe"}:
        from .catalogue import LanguageCatalogue
        from .master_tokenizer import MasterTokenizer
        catalogue = LanguageCatalogue(memory)
        if args.command == "catalogue-observe":
            tokenizer = MasterTokenizer.load(args.tokenizer, expected_identity=args.tokenizer_identity)
            output = catalogue.observe(args.request_id, args.task_id, args.source_id, tokenizer)
        else:
            output = catalogue.view(args.task_id, include_history=args.history, limit=args.limit, inventory=args.inventory)
    elif args.command == "dictionary":
        from .harmonizer import Harmonizer
        from .syszero import CommonVocabulary
        from .agent import context
        vocabulary = CommonVocabulary.load(args.vocabulary) if args.vocabulary else None
        basis = memory.state["scratchboard"][args.board[0]]["snapshot"]["core_identity"] if args.board else "dictionary-inspection"
        snapshot = context(memory, args.task_id, basis, selected_ids=args.source, scratchboard_ids=args.board)["snapshot"]
        output = Harmonizer(memory, vocabulary=vocabulary).view(snapshot)
    elif args.command in {"judge", "assess-judgment", "experiences", "carry-judgments"}:
        from .agent import context
        from .judgments import Judgments
        from .steps import strict_json
        judgments = Judgments(memory)
        if args.command == "judge":
            predicate = strict_json(args.predicate.read_text()) if args.predicate else None
            if args.core_config:
                if args.verdict is not None or args.reason is not None or args.uncertainty is not None:
                    raise ValueError("A generated prediction supplies its own verdict and reason")
                with JsonlCore(json.loads(args.core_config.read_text()), stderr_path=args.state/"worker.log") as core:
                    output = judgments.generate(core, args.task_id, args.question, predicate=predicate, selected_ids=args.source)
            else:
                if args.verdict is None or args.reason is None:
                    raise ValueError("A recorded prediction requires --verdict and --reason")
                snapshot = context(memory, args.task_id, args.core_identity, selected_ids=args.source)["snapshot"]
                output = {"prediction": judgments.predict(snapshot, args.question, args.verdict, args.reason,
                                                           predicate=predicate, uncertainty=args.uncertainty)}
        elif args.command == "assess-judgment":
            output = {"assessment": judgments.assess(args.node_id)}
        elif args.command == "experiences":
            from .harmonizer import Harmonizer
            output = {**Harmonizer(memory).learning_view(args.task_id),
                      "experiences": judgments.experiences(args.task_id)}
        else:
            output = judgments.carry(args.node_ids)
            output["packed_hex"] = output.pop("packed").hex()
    elif args.command.startswith("board-"):
        from .agent import context
        from .scratchboard import Scratchboard
        from .steps import strict_json
        board = Scratchboard(memory)
        if args.command in {"board-add", "board-reserve"}:
            snapshot = context(memory, args.task_id, args.core_identity, selected_ids=args.source)["snapshot"]
            node = (board.add(snapshot, strict_json(args.spec.read_text())) if args.command == "board-add"
                    else board.reserve(snapshot, args.count, constraints=args.constraints))
            output = {"node": node, "view": board.describe(node["id"])}
        elif args.command == "board-show":
            output = board.slot(args.node_id, args.slot) if args.slot is not None else board.describe(args.node_id)
        elif args.command == "board-bind":
            node = board.bind(args.node_id, args.index, args.target_id)
            output = {"node": node, "view": board.describe(node["id"])}
        elif args.command == "board-import":
            snapshot = context(memory, args.task_id, args.core_identity, selected_ids=args.source)["snapshot"]
            node = board.import_node(args.node_id, snapshot, relevance=args.relevance, reason=args.reason)
            output = {"node": node, "view": board.describe(node["id"])}
        elif args.command in {"board-branch", "board-revise", "board-refine", "board-latest", "board-reshape"}:
            if args.command == "board-branch":
                node = board.branch(args.node_id, reason=args.reason)
            elif args.command == "board-revise":
                snapshot = None
                if args.refresh:
                    if args.source is None:
                        raise ValueError("Refresh requires explicit --source selection")
                    prior = memory.state["scratchboard"][args.node_id]["snapshot"]
                    snapshot = context(memory, prior["task_id"], prior["core_identity"], selected_ids=args.source)["snapshot"]
                elif args.source is not None:
                    raise ValueError("Source selection requires --refresh")
                node = board.revise(args.node_id, strict_json(args.spec.read_text()), reason=args.reason, snapshot=snapshot)
            elif args.command == "board-refine":
                node = board.refine(args.node_id, args.parts, reason=args.reason)
            elif args.command == "board-reshape":
                node = board.reshape_tensor(args.node_id, args.shape, reason=args.reason)
            else:
                node = board.latest(args.node_id)
            output = {"node": node, "view": board.describe(node["id"])}
        elif args.command == "board-note":
            node = board.quantity(args.step_id, language=args.language)
            output = {"node": node, "view": board.describe(node["id"])}
        else:
            proposal = board.propose_document(args.node_id, filename=args.filename)
            receipt = memory.commit(proposal["id"], core_identity=proposal["snapshot"]["core_identity"]) if args.commit else None
            output = {"proposal": proposal, "receipt": receipt,
                      "artifact_path": str(memory.root / receipt["path"]) if receipt else None, "model_calls": 0}
    else:
        config = json.loads(args.core_config.read_text())
        with JsonlCore(config, stderr_path=args.state / "worker.log") as core:
            output = (calculate(memory, core, args.task_id, domain=args.domain, selected_ids=args.source) if args.command == "calculate"
                      else run(memory, core, args.task_id, commit=args.commit, observation_ids=args.observation,
                               scratchboard_ids=args.board, selected_ids=args.source))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
