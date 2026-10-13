"""Run the same native BF16 consumer after complete raw/lossless restoration."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoai.core import JsonlCore, file_digest
from nanoai.native_worker import runtime_environment
from evidence import snapshot_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=Path(__file__).resolve().parents[1] / "evals/storage-panel.json")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    panel = json.loads(args.panel.read_text())
    configs = {name: json.loads(path.read_text()) for name, path in (("raw", args.raw), ("packed", args.packed))}
    manifests = {name: json.loads(Path(config["manifest"]).read_text()) for name, config in configs.items()}
    raw, packed = manifests["raw"], manifests["packed"]
    if packed["model"]["restored_digest"] != raw["model"]["digest"]:
        raise ValueError("Raw and packed sources refer to different BF16 models")
    if (any(raw[k] != packed[k] for k in ("server", "libraries", "precision_inspection"))
            or runtime_environment(raw.get("runtime"), {})[1] != runtime_environment(packed.get("runtime"), {})[1]):
        raise ValueError("Native consumer inputs differ beyond model storage")
    result = {"panel_identity": file_digest(args.panel), "arms": {}, "checks": [], "error": None, "completed": False}

    def save():
        (args.out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    core = None
    try:
        for name, config in configs.items():
            destination = args.out / name
            destination.mkdir()
            snapshot_sources(destination, config)
            start = time.monotonic()
            core = JsonlCore(config, stderr_path=destination / "worker.log")
            arm = {"load": core.load, "identity": core.identity,
                   "startup_seconds": time.monotonic() - start, "responses": []}
            result["arms"][name] = arm
            save()
            for case in panel["cases"]:
                start = time.monotonic()
                generated = core.generate(case["messages"], max_tokens=panel["max_tokens"])
                arm["responses"].append({"case": case["id"], "generated": generated,
                                         "client_seconds": time.monotonic() - start})
                save()
                print(name, case["id"], generated["tokens"], flush=True)
            owned_directory = Path(core.load["owned_temporary_directory"])
            core.close()
            arm["resources"] = json.loads(core.stderr_path.with_suffix(".resources.json").read_text())
            arm["owned_temporary_directory_removed"] = not owned_directory.exists()
            core = None
            save()
        for original, restored in zip(result["arms"]["raw"]["responses"], result["arms"]["packed"]["responses"]):
            keys = ("output", "tokens", "prompt_tokens", "finish_reason")
            result["checks"].append({"name": original["case"] + "_matched_generation",
                "passed": all(original["generated"][key] == restored["generated"][key] for key in keys),
                "compared_fields": keys, "scope": "Text, token counts and termination; full logit vectors are not returned by this adapter"})
        result["checks"].append({"name": "complete_original_bytes_restored",
            "passed": result["arms"]["packed"]["load"]["source_storage"]["complete_original_digest_verified"]})
        result["checks"].append({"name": "all_owned_processes_and_temporary_files_closed",
            "passed": all(arm["resources"]["child_reaped"] and arm["resources"]["process_group_stopped"] and arm["owned_temporary_directory_removed"] for arm in result["arms"].values())})
        result["completed"] = True
    except Exception as error:
        result["error"] = repr(error)
        raise
    finally:
        if core:
            core.close()
            result["failed_process_cleanup"] = json.loads(core.stderr_path.with_suffix(".resources.json").read_text())
        save()


if __name__ == "__main__":
    main()
