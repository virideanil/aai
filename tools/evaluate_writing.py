"""Run an original-BF16 writing cycle and a real fresh-process continuation.

The protocol prescribes phases. Mechanical checks and attributed critique are
separate from reader judgment and learned capability admission.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoai.agent import DEFAULT_IDENTITY
from nanoai.catalogue import LanguageCatalogue
from nanoai.core import JsonlCore, file_digest
from nanoai.experience import Experience
from nanoai.memory import Memory
from nanoai.steps import Steps
from nanoai.writing import Writing

WORLD = ("Fictional canon: The coastal town of Merewick is flooding. Mira repairs its harbour clock. "
         "Ada, the ferryman, left a brass gate key inside that clock before disappearing. "
         "The gate will open after three complete clock cycles. The story is told in third-person limited "
         "through Mira. Show her choosing between keeping the key as a memory of Ada and using it to save a neighbour.")
PERSON = "Fictional canon: Mira wears a blue scarf. She cannot know why Ada disappeared."
CORRECTION = "Düzeltme: Mira'nın atkısı yeşildir, mavi değildir. Ada'nın neden kaybolduğunu hâlâ bilmiyor."
OPENING = ("Write the opening scene in 50 to 65 English words. Use concrete action and one specific sensory detail. "
           "Name Mira, the scarf colour and the exact total duration of the three cycles. End on a decision about the key.")


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def check_text(text, criteria):
    words = re.findall(r"\b[\w'-]+\b", text)
    folded = text.casefold()
    result = {"word_extent": criteria["min_words"] <= len(words) <= criteria["max_words"],
              "nonempty_text": bool(text.strip())}
    for word in criteria.get("required", []):
        result["contains_" + word] = word.casefold() in folded
    for word in criteria.get("forbidden", []):
        result["excludes_" + word] = word.casefold() not in folded
    if criteria.get("duration"):
        result["contains_exact_duration"] = bool(re.search(r"(?:51|fifty[ -]one)\s+seconds", folded))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--stage", choices=("initial", "continuation"), default="initial")
    args = parser.parse_args()
    preflight = json.loads(args.preflight.read_text())
    if (preflight.get("model") != "Qwen3.5-2B-BF16" or preflight.get("thinking") is not True
            or preflight.get("reasoning_token_limit", 0) is not None
            or not preflight.get("parent_source")):
        raise ValueError("A sourced native launch preflight is required")
    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / (args.stage + ".json")
    if result_path.exists():
        raise ValueError("Keep the previous trial; use a fresh output directory")
    result = {"stage": args.stage, "status": "running", "preflight": preflight,
              "protocol_identity": file_digest(__file__), "phases": [], "parameter_updates": 0,
              "assessment_scope": "Mechanical brief checks only; literary quality and learning remain separate"}
    start = time.monotonic()
    memory = Memory(args.out / "state")
    memory.initialize(DEFAULT_IDENTITY)
    owner = Writing(memory)
    try:
        if args.stage == "initial":
            memory.put_source("world", WORLD, note="Fictional world and plot constraints.")
            memory.put_source("person", PERSON, note="Current fictional character canon.")
            timing = memory.put_source("timing", json.dumps({"period": {"value": 17, "unit": "s"},
                "cycles": {"value": 3, "unit": "count"}}), note="Exact fictional clock specification.")
            memory.set_task("clock-duration", "Calculate the duration of three complete clock cycles.")
            snapshot = memory.snapshot("clock-duration", [timing], "exact-quantity-v1")
            steps = Steps(memory)
            calculation = steps.calculate_fields(snapshot, {"operator": "quantity.product", "inputs": [
                {"source": "timing", "path": ["period"]}, {"source": "timing", "path": ["cycles"]}]})
            steps.review(calculation["step_id"], "confirmed", reviewer="authored-writing-protocol-v1",
                         reason="The task explicitly requires three equal periods, so duration multiplies period by count.")
            basis = memory.attach_observations(snapshot, [calculation["step_id"]])
            memory.put_source("duration", "Checked quantity: three cycles of 17 seconds last exactly 51 seconds.",
                role="observation", note="Exact executor result for the declared clock; operation selected by the protocol.",
                derived_from=basis)
            if calculation["result"]["value"] != 51 or calculation["result"]["dimensions"] != {"s": 1}:
                raise ValueError("Exact clock computation differed")
            result["calculation"] = calculation
        else:
            initial = json.loads((args.out / "initial.json").read_text())
            result["recovered_effects"] = memory.recover()
            result["recovered_without_generation"] = memory.completed("opening")
            result["artifact_identity_preserved"] = all(
                Path(p).stat().st_ino == v["inode"] and Path(p).stat().st_mtime_ns == v["mtime_ns"]
                and file_digest(p) == v["identity"] for p, v in initial["artifact_before_restart"].items())
            history = owner.quote_history(initial["phases"][-1]["experience_id"], "historical-opening")
            memory.put_source("person", CORRECTION, expected_revision=1, note="Current fictional character canon, corrected in Turkish.")
            result["correction_invalidates_opening"] = memory.completed("opening")["remaining"] == ["text.md"]
            result["independent_plan_retained"] = memory.completed("plan")["remaining"] == []
            result["old_catalogue_invalidated"] = all(not o["current"] for o in LanguageCatalogue(memory).view("opening")["observations"])
            for invariant in ("artifact_identity_preserved", "correction_invalidates_opening",
                              "independent_plan_retained", "old_catalogue_invalidated"):
                if result[invariant] is not True:
                    raise AssertionError(f"Writing recovery invariant failed: {invariant}")
            if result["recovered_without_generation"]["remaining"]:
                raise AssertionError("The original opening did not recover before correction")
        with JsonlCore(json.loads(args.config.read_text()), stderr_path=args.out / (args.stage + "-worker.log")) as core:
            result.update(core_identity=core.identity, load=core.load)
            def run(phase, instruction, sources, kind, criteria, max_tokens=None):
                row = owner.run(phase, instruction, sources, core, workflow="merewick", kind=kind, max_tokens=max_tokens)
                row["assessment"] = Experience(memory).assess_generation(row["experience_id"],
                    rubric={"scope": result["assessment_scope"], "criteria": criteria, "training_target": None},
                    evaluator_identity=result["protocol_identity"], evaluate=check_text)
                result["phases"].append(row)
                save(result_path, result)
                print(json.dumps({"stage": args.stage, "phase": phase, "words": len(row["generation"]["text"].split()),
                    "finish": row["generation"]["finish_reason"], "checks": row["assessment"]["checks"]}), flush=True)
                return row
            if args.stage == "initial":
                run("plan", "Plan this original story in three numbered beats, at most 35 English words. End with a concrete choice. Write only the plan.",
                    ["world", "duration"], "plan", {"min_words": 1, "max_words": 35})
                run("opening", OPENING, ["world", "person", "duration", "plan-output"], "draft",
                    {"min_words": 50, "max_words": 65, "required": ["Mira", "blue", "key"], "duration": True})
                run("critique", "Give one specific criticism of this opening and one actionable revision, at most 30 English words. Check causality and the character's limited knowledge. Write only the critique.",
                    ["world", "person", "opening-output"], "critique", {"min_words": 1, "max_words": 30})
                run("revised-opening", "Revise the opening using the critique. " + OPENING,
                    ["world", "person", "duration", "opening-output", "critique-output"], "revision",
                    {"min_words": 50, "max_words": 65, "required": ["Mira", "blue", "key"], "duration": True})
            else:
                revised = run("corrected-opening", "Rewrite the historical opening with the current canon, including the Turkish correction. " + OPENING,
                    ["world", "person", "duration", "historical-opening"], "revision",
                    {"min_words": 50, "max_words": 65, "required": ["Mira", "green", "key"], "forbidden": ["blue"], "duration": True})
                run("continuation", "Continue directly after the opening in 50 to 65 English words. Let Mira's choice have a visible consequence for one neighbour. Preserve the current scarf colour and her uncertainty about Ada. Write only this next scene.",
                    ["world", "person", "corrected-opening-output", "plan-output"], "continuation",
                    {"min_words": 50, "max_words": 65, "required": ["Mira"], "forbidden": ["blue"]})
                run("direct-control", OPENING, ["world", "person", "duration"], "control",
                    {"min_words": 50, "max_words": 65, "required": ["Mira", "green", "key"], "forbidden": ["blue"], "duration": True})
                before = Path(revised["artifact_path"]).stat()
                calls = core.sequence
                repeat = owner.run("corrected-opening", "Rewrite the historical opening with the current canon, including the Turkish correction. " + OPENING,
                    ["world", "person", "duration", "historical-opening"], core, workflow="merewick", kind="revision")
                after = Path(revised["artifact_path"]).stat()
                result["repeat_preserved_effect"] = (repeat["model_calls"] == 0 and core.sequence == calls
                    and before.st_ino == after.st_ino and before.st_mtime_ns == after.st_mtime_ns)
                if not result["repeat_preserved_effect"]:
                    raise AssertionError("Repeated writing changed the observed effect or called the core")
                result["manuscript"] = owner.assemble("merewick", ["corrected-opening-output", "continuation-output"], "The Clock at Merewick")
            result["status"] = "completed"
        if args.stage == "initial":
            result["artifact_before_restart"] = {row["artifact_path"]: {"inode": Path(row["artifact_path"]).stat().st_ino,
                "mtime_ns": Path(row["artifact_path"]).stat().st_mtime_ns, "identity": file_digest(row["artifact_path"])} for row in result["phases"]}
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - start
        result["journal_sequence"] = memory.sequence
        result["backend"] = memory.backend
        save(result_path, result)
    if args.stage == "initial":
        child = subprocess.run([sys.executable, __file__, "--config", str(args.config.resolve()),
            "--out", str(args.out.resolve()), "--preflight", str(args.preflight.resolve()), "--stage", "continuation"])
        if child.returncode:
            raise RuntimeError(f"Fresh-process continuation returned {child.returncode}")


if __name__ == "__main__":
    main()
