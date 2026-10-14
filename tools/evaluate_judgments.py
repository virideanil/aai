"""Real BF16 judgments, checked feedback and optional exact ternary carriers.

The authored questions and predicates are frozen inputs. This measures the
new shared path, not learned question formation or a newly trained model.
"""
import argparse
import json
from pathlib import Path
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from nanoai.agent import DEFAULT_IDENTITY, context, run
from nanoai.core import JsonlCore, file_digest
from nanoai.harmonizer import Harmonizer
from nanoai.judgments import CHANNELS, Judgments, unpack
from nanoai.memory import Memory, StaleState
from nanoai.scratchboard import Scratchboard
from nanoai.steps import Steps
from evidence import snapshot_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=Path(__file__).resolve().parents[1]/"evals/judgments-panel.json")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config, panel = json.loads(args.config.read_text()), json.loads(args.panel.read_text())
    snapshot_sources(args.out, config)
    result = {"schema":"nanoai.judgment-consumer.v1", "panel_identity":file_digest(args.panel),
              "checks":[], "cases":[], "processes":[], "completed":False, "parameter_updates":0}

    def save():
        (args.out/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")

    def check(name, condition, **evidence):
        result["checks"].append({"name":name,"passed":bool(condition),**evidence})
        save()

    core = None
    try:
        memory = Memory(args.out/"state")
        memory.initialize(DEFAULT_IDENTITY)
        memory.put_source("measures",json.dumps(panel["source"]))
        memory.set_task("assessment","Assess the rail length, missing color evidence and summed durations. Keep predictions and checked properties distinct.",filenames=["assessment.md"])
        start = time.monotonic()
        core = JsonlCore(config,stderr_path=args.out/"worker.log")
        result.update(core_identity=core.identity,load=core.load,startup_seconds=time.monotonic()-start)
        judgments, steps, board = Judgments(memory), Steps(memory), Scratchboard(memory)
        predictions, assessments = [], []
        for case in panel["cases"]:
            snap = context(memory,"assessment",core.identity,selected_ids=["measures"])["snapshot"]
            predicate = None
            if case["operator"]:
                calculated = steps.calculate_fields(snap,{"operator":case["operator"],"inputs":[
                    {"source":"measures","path":[name]} for name in case["fields"]]})
                predicate = {"operator":"result.equals","step_id":calculated["step_id"],"expected":case["expected"]}
            generated = judgments.generate(core,"assessment",case["question"],predicate=predicate,selected_ids=["measures"])
            pred = generated["prediction"]
            assessed = judgments.assess(pred["id"])
            predictions.append(pred); assessments.append(assessed)
            row = {"id":case["id"],"expected":case["verdict"],"generated":generated,"assessment":assessed,
                   "prediction_correct":pred["spec"]["verdict"]==case["verdict"]}
            result["cases"].append(row)
            check(case["id"]+"_prediction_retained_separately",memory.state["scratchboard"][pred["id"]]["spec"]==pred["spec"]
                  and assessed["spec"]["prediction"]==pred["id"])
            check(case["id"]+"_checked_outcome",assessed["spec"]["verdict"]==case["verdict"])
            print(case["id"],pred["spec"]["verdict"],generated["generation"]["elapsed_ms"],flush=True)
        known = [r for r in result["cases"] if r["expected"]!="maybe"]
        answered = [r for r in known if r["generated"]["prediction"]["spec"]["verdict"]!="maybe"]
        wrong = sum(r["generated"]["prediction"]["spec"]["verdict"]!=r["expected"] for r in answered)
        result["selective_prediction"] = {"known_cases":len(known),"answered":len(answered),"wrong":wrong,
            "coverage":len(answered)/len(known),"risk_among_answered":wrong/len(answered) if answered else None,
            "unknown_cases":len(result["cases"])-len(known),
            "unsupported_binary_answers":sum(r["expected"]=="maybe" and r["generated"]["prediction"]["spec"]["verdict"]!="maybe" for r in result["cases"]),
            "scope":"One uncalibrated operating point on four authored questions; no risk-coverage curve or broad accuracy claim"}
        carried = judgments.carry([n["id"] for n in assessments])
        raw = board.read_tensor(carried["tensor"]["id"],core_identity=core.identity)
        values = [n["spec"]["verdict"] for n in assessments]
        words = [0x3F80 if v==c else 0 for v in values for c in CHANNELS]
        check("ternary_binary_and_bf16_views_exact",unpack(carried["packed"],len(values))==values and struct.unpack("<"+"H"*len(words),raw)==tuple(words))
        result["carriers"] = {**carried,"packed_hex":carried["packed"].hex()};del result["carriers"]["packed"]
        prepared = context(memory,"assessment",core.identity,selected_ids=["measures"],scratchboard_ids=[n["id"] for n in assessments])
        units = [r for r in Harmonizer(memory).view(prepared["snapshot"])["entries"] if r["kind"]=="working_unit"]
        check("same_typed_outcomes_reach_dictionary_and_context",json.loads(prepared["messages"][1]["content"])["working_units"]==[
            {key: value for key, value in unit.items() if key != "digest"} for unit in units])
        # A generated document exercises the consumer of the judgment records.
        receipt = None
        try:
            generated = run(memory,core,"assessment",selected_ids=["measures"],scratchboard_ids=[n["id"] for n in assessments])
            result["document"] = generated
            content = generated["proposal"]["output"]["content"]
            check("real_document_contains_checked_quantities_and_current_reference",all(word in content for word in ("46","42","[measures@1]")),content=content)
            receipt = memory.commit(generated["proposal"]["id"],core_identity=core.identity)
            result["document"]["receipt"] = receipt
        except Exception as error:
            # Keep a document failure visible while exercising independent
            # feedback, source correction and model compatibility consumers.
            check("real_document_completed",False,error=repr(error))
        current_id = core.identity
        core.close()
        result["processes"].append(json.loads(core.stderr_path.with_suffix(".resources.json").read_text()))
        core = None
        restored = Memory(memory.root)
        rows = Judgments(restored).experiences("assessment")
        check("fresh_feedback_has_complete_generation_and_separate_outcome",len(rows)==len(panel["cases"]) and all(r["generation"]["core_identity"]==current_id for r in rows))
        check("unknown_outcome_not_used_as_numerical_training_target",sum(r["eligible_for_predicate_training"] for r in rows)==len(known))
        # This is a compatibility check, not a second model inference run.
        context(restored,"assessment","version-change-probe",selected_ids=["measures"],scratchboard_ids=[assessments[0]["id"]])
        check("exact_judgment_survives_consumer_model_version_change",True)
        try:
            context(restored,"assessment","version-change-probe",selected_ids=["measures"],scratchboard_ids=[carried["tensor"]["id"]])
            refused = False
        except StaleState:
            refused = True
        check("numerical_view_requires_compatible_basis",refused)
        preserved = (memory.root/receipt["path"]).read_bytes() if receipt else None
        restored.put_source("measures",json.dumps({**panel["source"],"count":{"value":3,"unit":"count"}}),expected_revision=1)
        rows = Judgments(restored).experiences("assessment")
        check("source_correction_excludes_current_training_use_but_keeps_history",not any(r["current"] or r["eligible_for_predicate_training"] for r in rows)
              and len(rows)==len(panel["cases"]) and (receipt is None or (memory.root/receipt["path"]).read_bytes()==preserved))
        result["completed"] = True
    except Exception as error:
        result["error"] = repr(error)
        raise
    finally:
        if core is not None:
            core.close()
            result["processes"].append(json.loads(core.stderr_path.with_suffix(".resources.json").read_text()))
        save()
    if not all(r["passed"] for r in result["checks"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
