"""Measure symbolic capacity, then exercise BF16 choice and durable revision.

The authored predicate evaluates operation choice. Host code composes and
adopts scratchboard revisions. This does not evaluate learned graph editing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import tracemalloc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoai.agent import DEFAULT_IDENTITY, calculate, context, run
from nanoai.core import JsonlCore, file_digest
from nanoai.memory import Memory, StaleState
from nanoai.harmonizer import Harmonizer
from nanoai.scratchboard import Scratchboard, UnresolvedSlot
from nanoai.steps import Steps
from evidence import snapshot_sources


def measure(function):
    tracemalloc.start()
    start = time.perf_counter_ns()
    value = function()
    elapsed = time.perf_counter_ns() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value, {"elapsed_ns": elapsed, "peak_traced_python_bytes": peak,
                   "process_lifetime_peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024}


def journal_bytes(memory):
    return sum(p.stat().st_size for p in memory.segments.iterdir() if p.is_file())


def expected_notes(panel, revision):
    """Independent authored expectations include both orders of multiplication."""
    source = panel["source"] if revision == 1 else panel["correction"]
    length, count = source["length"]["value"], source["count"]["value"]
    answer, sid = panel["answers"][revision-1], panel["task"]
    review = "\nİnceleme kodu: " + panel["review_code"] + "\n" if panel.get("review_code") else ""
    return {(f"# Hesaplama\n\nHesaplanan toplam: **{answer} cm**. [{sid}@{revision}]\n\n"
             f"{expression} = {answer} cm.\n\nKaynak sürümü: [{sid}@{revision}]\n{review}")
            for expression in (f"{length} cm × {count}", f"{count} × {length} cm")}


def capacity(root, check):
    memory = Memory(root)
    memory.initialize(DEFAULT_IDENTITY)
    memory.set_task("capacity", "Measure symbolic addresses and exact repeated bytes", filenames=["capacity.md"])
    snapshot = context(memory, "capacity", "symbolic-host-v1", selected_ids=[])["snapshot"]
    board = Scratchboard(memory)
    rows = []
    for count in (1, 1000, 1_000_000, 10_000_000):
        before = journal_bytes(memory)
        node, metrics = measure(lambda: board.reserve(snapshot, count, constraints="Unresolved text or tensor work"))
        view = board.describe(node["id"])
        rows.append({"count": count, "view": view, "journal_added_bytes": journal_bytes(memory)-before, **metrics})
        check(f"capacity_{count}_unresolved_range", view["logical_units"] == count and view["unresolved_slots"] == count
              and view["reachable_nodes"] == 1 and view["stored_node_bytes"] < 2048)
    leaf = board.add(snapshot, {"kind": "literal", "text": "abc"})
    bound, binding = measure(lambda: board.bind(node["id"], 9_999_999, leaf["id"]))
    check("sparse_binding_preserves_other_unresolved_work", board.describe(bound["id"])["unresolved_slots"] == 9_999_999)
    try:
        board.materialize(bound["id"])
        blocked = False
    except UnresolvedSlot:
        blocked = True
    check("unresolved_content_cannot_materialize", blocked)
    repeated = board.add(snapshot, {"kind": "repeat", "node": leaf["id"], "count": 10_000_000})
    view = board.describe(repeated["id"])
    try:
        board.materialize(repeated["id"])
        budgeted = False
    except ValueError:
        budgeted = True
    check("expansion_requires_explicit_larger_budget", budgeted)
    raw, baseline = measure(lambda: b"abc" * 10_000_000)
    expected = "shake256:" + hashlib.shake_256(raw).hexdigest(64)
    del raw
    rendered, expansion = measure(lambda: board.materialize(repeated["id"], max_bytes=30_000_000))
    check("repeated_30MB_materialization_exact", len(rendered["bytes"]) == 30_000_000 and rendered["digest"] == expected)
    del rendered
    return {"reservations": rows, "sparse_binding": binding, "repeated_view": view,
            "plain_repeat_control": baseline, "materialization": expansion,
            "materialized_bytes": 30_000_000, "expected_digest": expected,
            "scope": "Symbolic capacity and repeated known bytes; no novel content or model generation. RSS is a cumulative process peak."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--panel", type=Path, default=Path(__file__).resolve().parents[1]/"evals/scratchboard-panel.json")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    panel = json.loads(args.panel.read_text())
    snapshot_sources(args.out, config)
    result = {"panel_identity": file_digest(args.panel), "checks": [], "episodes": [], "processes": [], "completed": False, "error": None}

    def save():
        (args.out/"result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")

    def check(name, passed, **evidence):
        result["checks"].append({"name": name, "passed": bool(passed), **evidence})
        save()

    core = None
    try:
        result["capacity"] = capacity(args.out/"capacity", check)
        save()
        memory = Memory(args.out/"state")
        memory.initialize(DEFAULT_IDENTITY)
        sid = panel["task"]
        memory.put_source(sid, json.dumps(panel["source"]))
        memory.set_task(sid, panel["goal"], filenames=[sid+".md"])
        memory.put_source("independent", "An unrelated note.")
        memory.set_task("independent", "Keep the unrelated note", filenames=["other.md"])
        independent_snapshot = context(memory, "independent", "symbolic-host-v1", selected_ids=["independent"])["snapshot"]
        independent = Scratchboard(memory).add(independent_snapshot, {"kind":"literal", "text":"An unrelated note. [independent@1]"})
        other = Scratchboard(memory).propose_document(independent["id"])
        memory.commit(other["id"], core_identity="symbolic-host-v1")
        start = time.monotonic()
        core = JsonlCore(config, stderr_path=args.out/"worker.log")
        result.update(core_identity=core.identity, load=core.load, startup_seconds=time.monotonic()-start)
        retained, old_bytes, old_path, generated_proposal = None, None, None, None
        for revision, answer in enumerate(panel["answers"], 1):
            if revision == 2:
                memory.put_source(sid, json.dumps(panel["correction"]), expected_revision=1, note="Corrected rail count")
                check("source_correction_reopens_affected_work", bool(memory.completed(sid)["remaining"]))
                check("source_correction_preserves_independent_work", not memory.completed("independent")["remaining"])
                try:
                    Scratchboard(memory).materialize(retained["id"])
                    stale = False
                except StaleState:
                    stale = True
                check("old_composition_rejects_changed_source", stale)
                if generated_proposal:
                    try:
                        memory.commit(generated_proposal["id"], core_identity=core.identity)
                        rejected = False
                    except StaleState:
                        rejected = True
                    check("generated_working_unit_proposal_rejects_after_correction", rejected)
            start = time.monotonic()
            calc = calculate(memory, core, sid, domain="quantity", selected_ids=[sid])
            selected = sorted((row["source"], tuple(row["path"])) for row in calc["plan"]["inputs"])
            expected = sorted((sid, tuple(path)) for path in panel["paths"])
            passed = calc["plan"]["operator"] == panel["operator"] and selected == expected and calc["result"]["value"] == answer
            row = {"revision": revision, "calculation": calc, "model_choice_seconds": time.monotonic()-start}
            result["episodes"].append(row)
            check(f"bf16_choice_{revision}_matches_frozen_task_predicate", passed, expected=answer)
            Steps(memory).review(calc["step_id"], "confirmed" if passed else "rejected", reviewer="scratchboard-panel-v1",
                reason="Authored task predicate checks the chosen operation, source fields and exact result")
            if not passed:
                raise ValueError("The BF16 model choice failed the task predicate")
            board = Scratchboard(memory)
            quantity = board.quantity(calc["step_id"], language="tr")
            review = "\nİnceleme kodu: " + panel["review_code"] + "\n" if panel.get("review_code") else ""
            suffix = board.add(calc["snapshot"], {"kind":"literal", "text":f"\nKaynak sürümü: [{sid}@{revision}]\n{review}"})
            if revision == 1:
                slots = board.reserve(calc["snapshot"], 2, constraints="Reviewed quantity note, then its source revision")
                one = board.bind(slots["id"], 0, quantity["id"])
                composed = board.bind(one["id"], 1, suffix["id"])
                check("partial_composition_remains_unresolved", board.describe(one["id"])["unresolved_slots"] == 1)
                retained = composed
                prepared = context(memory, sid, core.identity, selected_ids=[sid], scratchboard_ids=[composed["id"]])
                dictionary = Harmonizer(memory).view(prepared["snapshot"])
                unit = next(row for row in dictionary["entries"] if row["kind"] == "working_unit")
                check("same_working_unit_reaches_dictionary_and_model_context",
                      json.loads(prepared["messages"][1]["content"])["working_units"] == [{key: value for key, value in unit.items() if key != "digest"}]
                      and unit["preview_complete"] and panel["review_code"] in unit["preview"])
                generated = run(memory, core, sid, selected_ids=[sid], scratchboard_ids=[composed["id"]], max_tokens=192)
                generated_proposal = generated["proposal"]
                generated_content = generated_proposal["output"]["content"]
                check("real_bf16_uses_code_available_only_in_working_unit",
                      panel["review_code"] in generated_content and f"{answer}" in generated_content
                      and f"[{sid}@{revision}]" in generated_content
                      and "[source_id@revision]" not in generated_content,
                      content=generated_content)
                row["working_unit_generation"] = generated
            else:
                composed = board.revise(retained["id"], {"kind":"concat", "nodes":[quantity["id"], suffix["id"]]},
                    reason="Adopt the reviewed calculation using the corrected source", snapshot=calc["snapshot"])
                check("revision_preserves_explicit_lineage", board.latest(retained["id"])["id"] == composed["id"])
            proposal = board.propose_document(composed["id"])
            permitted_contents = expected_notes(panel, revision)
            check(f"composition_{revision}_exact_turkish_bytes", proposal["output"]["content"] in permitted_contents,
                  content=proposal["output"]["content"], render_model_calls=proposal["metrics"]["render_model_calls"])
            row.update(scratchboard_node=composed, proposal=proposal)
            if revision == 1:
                receipt = memory.commit(proposal["id"], core_identity=core.identity)
                old_path = memory.root/receipt["path"]
                old_bytes = old_path.read_bytes()
                row["receipt"] = receipt
            else:
                identity = core.identity
                core.close()
                result["processes"].append(json.loads(core.stderr_path.with_suffix(".resources.json").read_text()))
                core = None
                code = "from nanoai.memory import Memory; import sys; Memory(sys.argv[1]).commit(sys.argv[2],core_identity=sys.argv[3],crash_after_publish=True)"
                environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/"src"))
                crash = subprocess.run([sys.executable,"-c",code,str(memory.root),proposal["id"],identity],
                    env=environment,capture_output=True,text=True,timeout=15)
                check("actual_publication_process_interrupted", crash.returncode == 88, stderr=crash.stderr)
                restored = Memory(memory.root)
                receipts = restored.recover()
                check("fresh_recovery_observes_once", len(receipts) == 1 and receipts[0]["reconciled_existing"] and restored.recover() == [])
                completed = restored.completed(sid)
                check("corrected_artifact_resumes_without_model", not completed["remaining"] and completed["model_calls"] == 0)
                check("original_artifact_bytes_preserved", old_path.read_bytes() == old_bytes)
                check("independent_artifact_still_complete", not restored.completed("independent")["remaining"])
                check("current_recovered_bytes_match", completed["artifacts"][sid+".md"]["content"] in permitted_contents)
                row["recovery"] = receipts
            print("scratchboard BF16 case", revision, "passed", passed, flush=True)
        result["completed"] = True
    except Exception as error:
        result["error"] = repr(error)
        raise
    finally:
        if core is not None:
            core.close()
            result["processes"].append(json.loads(core.stderr_path.with_suffix(".resources.json").read_text()))
        save()
    if not all(row["passed"] for row in result["checks"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
