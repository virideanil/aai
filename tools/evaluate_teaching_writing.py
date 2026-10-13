"""Apply source-grounded teaching to the continuing, observed writing episode."""
import argparse
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoai.core import JsonlCore, file_digest
from nanoai.experience import Experience
from nanoai.memory import Memory
from nanoai.teaching import Teaching
from nanoai.writing import Writing
from evaluate_writing import check_text, OPENING


def practice_story(args, preflight):
    memory = Memory(args.baseline / "state")
    writer = Writing(memory)
    result = {"status": "running", "before_head": memory.head, "preflight": preflight,
              "implementation": file_digest(__file__), "parameter_updates": 0,
              "scope": "Observed brief checks; causal and literary judgment reported separately"}
    started = time.monotonic()
    def save():
        result["elapsed_seconds"] = time.monotonic() - started
        (args.out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    def feedback(text, checks, criteria):
        count = len(re.findall(r"\b[\w'-]+\b", text))
        failed = [name for name, passed in checks.items() if not passed]
        return (f"The observed draft has {count} words. Failed checks: {', '.join(failed)}. "
                f"Revise to {criteria['min_words']} to {criteria['max_words']} words, retaining required facts. "
                "Remove redundant description before removing an action. "
                "Keep the key moving directly from Mira's palm to the lock; include no pocket action. "
                "Write only the scene.")
    try:
        for lesson in json.loads(args.lessons.read_text())["lessons"]:
            Teaching(memory).install(lesson)
        instruction = (OPENING + " The key starts in Mira's palm and moves directly to the lock. "
                       "Include no pocket action. Preserve her uncertainty about Ada.")
        criteria = {"min_words": 50, "max_words": 65, "required": ["Mira", "green", "key"],
                    "forbidden": ["blue", "pocket"], "duration": True}
        sources = ["world", "person", "duration", "key-continuity-feedback"]
        options = dict(criteria=criteria, evaluate=check_text, feedback=feedback,
            evaluator_identity=result["implementation"], scope=result["scope"],
            lesson_ids=["lesson-criticism", "lesson-purpose"], max_attempts=3,
            max_tokens=None, workflow="merewick-practice")
        with JsonlCore(json.loads(args.config.read_text()), stderr_path=args.out / "worker.log") as core:
            result.update(core_identity=core.identity, load=core.load)
            phase = args.out.name + "-opening"
            result["opening"] = writer.practice(phase, instruction, sources, core, **options)
            save()
            for row in result["opening"]["attempts"]:
                print(json.dumps({"phase": row["phase"], "text": row["generation"]["text"],
                    "checks": row["assessment"]["checks"]}), flush=True)
            calls = core.sequence
            original = {r["artifact_path"]: (Path(r["artifact_path"]).stat().st_ino,
                        Path(r["artifact_path"]).stat().st_mtime_ns) for r in result["opening"]["attempts"]}
            repeated = writer.practice(phase, instruction, sources, core, **options)
            result["repeat_preserved_effects"] = (core.sequence == calls
                and all(r["model_calls"] == 0 for r in repeated["attempts"])
                and all((Path(p).stat().st_ino, Path(p).stat().st_mtime_ns) == before
                        for p, before in original.items()))
            if not result["repeat_preserved_effects"]:
                raise AssertionError("Practice repeat changed an effect or generated again")
            selected = result["opening"]["selected"]
            if selected:
                result["continuation"] = writer.run(args.out.name + "-next",
                    "Continue directly after the opening in 50 to 65 English words. Show a specific escape route "
                    "and the neighbour's movement to safety. Preserve the key and gate state and Mira's limited "
                    "knowledge of Ada. Write only the scene.",
                    ["world", "person", selected["source"]["id"], "rescue-feedback"], core,
                    workflow="merewick-practice", kind="continuation",
                    lesson_ids=["lesson-causality"])
                row = result["continuation"]
                row["assessment"] = Experience(memory).assess_generation(row["experience_id"],
                    rubric={"scope": result["scope"], "criteria": {"min_words": 50, "max_words": 65,
                            "forbidden": ["blue"]}, "training_target": None},
                    evaluator_identity=result["implementation"], evaluate=check_text)
                result["manuscript"] = writer.assemble("merewick-practice",
                    [selected["source"]["id"], row["source"]["id"]], "The Clock at Merewick")
        fresh = Memory(args.baseline / "state")
        result["recovery"] = fresh.recover()
        rows = result["opening"]["attempts"] + ([result["continuation"]] if "continuation" in result else [])
        result["fresh_recovery_complete"] = all(fresh.completed(r["phase"])["remaining"] == [] for r in rows)
        if not result["fresh_recovery_complete"]:
            raise AssertionError("Practice artifact recovery failed")
        result.update(status="completed", after_head=fresh.head)
    except BaseException as error:
        result.update(status="failed", error=repr(error)); raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "out", "config", "lessons", "preflight"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--practice", action="store_true")
    args = parser.parse_args()
    preflight = json.loads(args.preflight.read_text())
    if (preflight.get("model") != "Qwen3.5-2B-BF16" or preflight.get("thinking") is not True
            or preflight.get("reasoning_token_limit", 0) is not None
            or not preflight.get("parent_source")):
        raise ValueError("Sourced native launch preflight required")
    if args.out.exists():
        raise ValueError("Retain the earlier teaching run")
    args.out.mkdir(parents=True)
    if args.practice:
        return practice_story(args, preflight)
    first = json.loads((args.baseline / "initial.json").read_text())
    continued = json.loads((args.baseline / "continuation.json").read_text())
    memory = Memory(args.baseline / "state")
    result = {"status": "running", "before_head": memory.head, "preflight": preflight,
              "implementation": file_digest(__file__), "phases": [], "parameter_updates": 0}
    started = time.monotonic()
    def save():
        result["elapsed_seconds"] = time.monotonic() - started
        (args.out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    teaching, writer = Teaching(memory), Writing(memory)
    for lesson in json.loads(args.lessons.read_text())["lessons"]:
        teaching.install(lesson)
    old_critique = next(r for r in first["phases"] if r["phase"] == "critique")
    writer.quote_history(old_critique["experience_id"], "historical-critic")
    opening = next(r for r in continued["phases"] if r["phase"] == "corrected-opening")
    continuation = next(r for r in continued["phases"] if r["phase"] == "continuation")
    teaching.feedback(opening["experience_id"], source_id="key-continuity-feedback",
        draft_quote="pulled the key from her pocket",
        evidence=[{"source_id": "world", "quote": "Show her choosing between keeping the key as a memory of Ada and using it to save a neighbour."}],
        diagnosis="The key was already in her palm earlier in this draft. Retrieving it again obscures the continuous action and the choice.",
        repair="Keep the key's movement continuous from her palm to the lock. Make her choice visible, preserve the current scarf colour and 51-second duration, and use 50 to 65 words.",
        reviewer="authored-writing-teacher-v1", rule_ids=["lesson-criticism", "lesson-purpose"])
    teaching.feedback(continuation["experience_id"], source_id="rescue-feedback",
        draft_quote="flooding the street with rushing water",
        evidence=[{"source_id": "world", "quote": "using it to save a neighbour."}],
        diagnosis="The described gate action increases the flood; the scene never shows a mechanism by which the neighbour reaches safety.",
        repair="Connect the use of the key to a concrete escape route and the neighbour's actual movement to safety. Preserve the prior scene's key and gate state.",
        reviewer="authored-writing-teacher-v1", rule_ids=["lesson-causality"])
    try:
        with JsonlCore(json.loads(args.config.read_text()), stderr_path=args.out / "worker.log") as core:
            result.update(core_identity=core.identity, load=core.load)
            def run(name, instruction, sources, lessons, kind, criteria, max_tokens=None):
                row = writer.run(name, instruction, sources, core, workflow="merewick-taught",
                                 kind=kind, max_tokens=max_tokens, lesson_ids=lessons)
                row["assessment"] = Experience(memory).assess_generation(row["experience_id"],
                    rubric={"scope": "Exact brief diagnostics; semantic and literary reading remain separately attributed",
                            "criteria": criteria, "training_target": None},
                    evaluator_identity=file_digest(__file__), evaluate=check_text)
                result["phases"].append(row); save()
                print(json.dumps({"phase": name, "text": row["generation"]["text"], "checks": row["assessment"]["checks"]}), flush=True)
                return row
            run("taught-critic", "Evaluate the historical criticism about the key's existence against the current canon. In at most 35 words, say whether that criticism is supported and explain the specific reason. Write only this assessment.",
                ["world", "person", "historical-critic"], ["lesson-evidence"], "critique",
                {"min_words": 1, "max_words": 35})
            revised = run("taught-opening", "Use the evidence-backed feedback to revise the draft. " + OPENING,
                ["world", "person", "duration", "corrected-opening-output", "key-continuity-feedback"],
                ["lesson-criticism", "lesson-purpose"], "revision",
                {"min_words": 50, "max_words": 65, "required": ["Mira", "green", "key"], "forbidden": ["blue"], "duration": True})
            following = run("taught-next", "Continue directly after the revised opening in 50 to 65 English words. Apply the rescue feedback and show how the neighbour reaches safety. Preserve the current gate state, scarf and Mira's limited knowledge of Ada. Write only the scene.",
                ["world", "person", "taught-opening-output", "rescue-feedback"], ["lesson-causality"], "continuation",
                {"min_words": 50, "max_words": 65, "required": ["Mira"], "forbidden": ["blue"]})
            result["manuscript"] = writer.assemble("merewick-taught", [revised["source"]["id"], following["source"]["id"]], "The Clock at Merewick")
        fresh = Memory(args.baseline / "state")
        result["fresh_recovery"] = fresh.recover()
        result["completed_without_model"] = [fresh.completed(row["phase"])["remaining"] == [] for row in result["phases"]]
        result.update(status="completed", after_head=fresh.head)
    except BaseException as error:
        result.update(status="failed", error=repr(error)); raise
    finally:
        save()


if __name__ == "__main__":
    main()
