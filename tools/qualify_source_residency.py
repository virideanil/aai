"""Compare the original mapped CPU path and source-page residency advice."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "source", "preflight", "source-before"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    import torch
    from nanoai.core import file_digest
    from nanoai.torch_source import load_full_text
    from nanoai.generative_learning import attach, encode, loss, chunked_backward, generate, cache_prefix, basis
    from nanoai.source_residency import SourceResidency
    started = time.monotonic()
    result = {"status": "running", "phases": []}
    def record(phase, **data):
        row = {"phase": phase, "seconds": time.monotonic()-started, **data}
        result["phases"].append(row)
        (args.out/"model-result.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(row), flush=True)
    try:
        preflight = json.loads(args.preflight.read_text())
        if (preflight.get("model") != "Qwen3.5-2B-BF16" or preflight.get("thinking") is not True
                or preflight.get("reasoning_token_limit", 0) is not None or not preflight.get("parent_source")):
            raise ValueError("Sourced model preflight is required")
        manifest = json.loads(args.source_before.read_text())
        if manifest != {p.name: {"bytes": p.stat().st_size, "digest": file_digest(p)} for p in sorted(args.source.iterdir()) if p.is_file()}:
            raise ValueError("Original model source manifest differs")
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        prompt = [{"role": "user", "content": "Let x mean the door is open. Express the door is closed as JSON using the operator not and the variable x. Return only JSON."}]
        references = None
        original_generation = None
        for mode in ("original", "advised"):
            torch.manual_seed(62113)
            model, tok = load_full_text(args.source)
            residency = SourceResidency(args.source).install(model) if mode == "advised" else None
            parameters = attach(model); chunked_backward(model)
            source_basis = basis(manifest, model)
            example = encode(tok, prompt, '{"not":"x"}',
                reasoning='The source defines x as open. Closed negates that proposition, so apply not to x.')
            def measure(prefix=None):
                model.zero_grad(set_to_none=True)
                value, logits = loss(model, example, prefix=prefix, source_basis=source_basis)
                value.backward()
                return {"logits": logits.detach().clone(), "gradients": {n: p.grad.detach().clone() for n, p in parameters.items()}}
            t = time.monotonic(); observed = measure()
            if references is None:
                references = observed
            elif (not torch.equal(references["logits"], observed["logits"])
                  or any(not torch.equal(g, observed["gradients"][n]) for n, g in references["gradients"].items())):
                raise ValueError("Source residency advice changed logits or own-recipe gradients")
            record(mode, full_seconds=time.monotonic()-t, logits_equal=mode == "advised",
                   own_recipe_gradients_equal=mode == "advised", stats=residency.stats() if residency else None)
            if residency:
                prefix = cache_prefix(model, example, source_basis=source_basis)
                cached = measure(prefix)
                if (not torch.equal(observed["logits"], cached["logits"])
                    or any(not torch.equal(g, cached["gradients"][n]) for n, g in observed["gradients"].items())):
                    raise ValueError("Frozen-prefix continuation differs under residency advice")
                record("prefix", exact_logits=True, exact_own_recipe_gradients=True)
            corpus = json.loads((Path(__file__).resolve().parents[1]/"evals/meaning_corpus_v3.json").read_text())
            row = next(r for r in corpus["rows"] if r["id"] == "train-leaf-en-math")
            text = row["question"]+"\n"+"\n".join(row["observations"])
            t = time.monotonic()
            generated = generate(model, tok, [{"role": "user", "content": text}])
            if original_generation is None:
                original_generation = generated
            elif generated != original_generation:
                raise ValueError("Natural generation differs between matched source residency modes")
            record("generation", mode=mode, generated=generated, generation_seconds=time.monotonic()-t,
                   exact_matched_tokens=mode == "advised", stats=residency.stats() if residency else None)
            del model, parameters, observed, tok
            gc.collect()
        result["status"] = "completed"
        record("qualified", scope="One full-source loss and gradient case, exact frozen-prefix replay and matched natural thinking-on generation; no quality or resource reduction claim")
    except BaseException as error:
        result.update(status="failed", error=str(error), traceback=traceback.format_exc())
        record("failure", error=str(error))
        raise


if __name__ == "__main__":
    main()
