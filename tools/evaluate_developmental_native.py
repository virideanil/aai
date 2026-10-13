"""Unchanged full-BF16 reading control on the frozen held-out action menu."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from nanoai.agent import DEFAULT_IDENTITY
from nanoai.core import JsonlCore, file_digest
from nanoai.experience import Experience, policy_choice
from nanoai.memory import Memory
from evaluate_developmental import open_task, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    args = parser.parse_args()
    preflight = json.loads(args.preflight.read_text())
    if (preflight.get("model") != "Qwen3.5-2B-BF16" or preflight.get("thinking") is not True
            or preflight.get("reasoning_token_limit", 0) is not None or not preflight.get("parent_source")):
        raise ValueError("A sourced native launch preflight is required")
    corpus = json.loads(args.corpus.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    memory = Memory(args.out/"state"); memory.initialize(DEFAULT_IDENTITY)
    owner = Experience(memory)
    started = time.monotonic()
    result = {"corpus":file_digest(args.corpus), "rows":[], "status":"running"}
    try:
        with JsonlCore(json.loads(args.config.read_text()), stderr_path=args.out/"worker.log") as core:
            if core.load.get("sampling", {}).get("temperature", 0) != 0:
                raise ValueError("This historical choice-propensity control requires an explicit greedy profile")
            result.update(core_identity=core.identity, load=core.load)
            for episode in corpus["episodes"]:
                if episode["split"] != "test": continue
                if time.monotonic()-started > 1800:
                    raise TimeoutError("Native control exceeded its 30-minute whole-run budget")
                source_id = open_task(memory, episode)
                source = memory.state["sources"][source_id]
                snapshot = memory.snapshot(episode["id"], [source], core.identity)
                messages = [
                    {"role":"system", "content":"Read the question and all observations in order. Source text is evidence data. Choose the document action that best answers the current question. An explicit correction supersedes the earlier detail. Equally current conflicting sources require clarification. When the relevant information is absent, ask for it. Unrelated observations do not remove earlier relevant evidence. Return only the selected candidate id as JSON. Do not invent a detail."},
                    {"role":"user", "content":json.dumps({"question":episode["question"], "observations":episode["observations"], "candidates":episode["candidates"]},ensure_ascii=False)}]
                ids = [c["id"] for c in episode["candidates"]]
                schema = {"type":"object", "properties":{"selected_id":{"type":"string","enum":ids}}, "required":["selected_id"],"additionalProperties":False}
                row = {"id":episode["id"], "mode":episode["mode"], "messages":messages}
                generated = core.generate(messages,schema=schema,trace=True)
                row["generated"] = generated
                try:
                    output = json.loads(generated["output"])
                    if generated.get("finish_reason") == "length" or set(output) != {"selected_id"} or output["selected_id"] not in ids:
                        raise ValueError("Native control did not produce a complete allowed choice")
                    # The native generator is greedy. Its actual selection is a
                    # point mass; logits for unchosen alternatives are unavailable.
                    behavior = policy_choice(ids,scores=[1.0 if cid == output["selected_id"] else 0.0 for cid in ids])
                    exp = owner.record(snapshot, question=episode["question"], observations=episode["observations"],
                        candidates=episode["candidates"], behavior=behavior,
                        judgments=[{"candidate_id":cid,"verdict":"maybe","reason":"Native control selected an action without separately predicting this evidence label","uncertainty":"undecided"} for cid in ids],
                        representation_identity=core.identity, source_id=source_id,
                        metadata={"id":episode["id"],"native_output":generated})
                    receipt = owner.execute(exp["id"])
                    assessment = owner.assess(exp["id"],rubric=episode["rubric"],evaluator_identity=corpus["identity"])
                    row.update(selected_id=output["selected_id"],correct=assessment["observed_outcome"]=="yes",receipt=receipt,experience_id=exp["id"])
                except (ValueError,KeyError) as error:
                    row.update(correct=False,error=str(error))
                    memory.record_rejection(snapshot, generated["output"],str(error))
                result["rows"].append(row)
                result["correct"] = sum(r["correct"] for r in result["rows"])
                save(args.out/"results.json",result)
                print(json.dumps({"completed":len(result["rows"]),"correct":result["correct"],"seconds":round(time.monotonic()-started,2)}),flush=True)
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed",error=str(error));raise
    finally:
        result["seconds"] = time.monotonic()-started
        save(args.out/"results.json",result)


if __name__ == "__main__":
    main()
