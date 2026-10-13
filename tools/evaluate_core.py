"""Run the frozen diagnostic panel through the maintained JSONL model protocol."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoai.core import JsonlCore, file_digest, swap_bytes
from evidence import snapshot_sources


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    panel = json.loads(args.panel.read_text())
    config = json.loads(args.config.read_text())
    snapshot_sources(args.out, config)
    result = {"panel_identity": file_digest(args.panel), "runs": [], "error": None}
    start = time.monotonic()
    core = None
    try:
        core = JsonlCore(config, stderr_path=args.out / "stderr.log")
        result.update(load_seconds=time.monotonic() - start, core_identity=core.identity,
            model_manifest=core.manifest_identity, load=core.load)
        for case in panel["cases"]:
            beginning = time.monotonic()
            output = core.generate([{"role": "user", "content": case["prompt"]}], max_tokens=case["limit"])
            row = {"case_id": case["id"], "repeat": case.get("repeat", 0), "client_seconds": time.monotonic()-beginning, **output}
            result["runs"].append(row)
            (args.out / "partial.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
            print(case["id"], row["finish_reason"], round(row["client_seconds"], 3), flush=True)
    except BaseException as exc:
        result["error"] = repr(exc)
    finally:
        if core:
            core.close()
            result.update(samples=core.samples, child_reaped=core.child.poll() is not None, exit_code=core.child.returncode,
                          swap_before_bytes=core.swap_before)
        result.update(wall_seconds=time.monotonic()-start, swap_after_bytes=swap_bytes())
        (args.out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k in ("error","wall_seconds","child_reaped","load_seconds")}), flush=True)
    if result["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
