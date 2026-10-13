"""Exercise tokenizer, native SEE, journal restart and source correction together."""
import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoai.agent import DEFAULT_IDENTITY
from nanoai.catalogue import LanguageCatalogue
from nanoai.core import file_digest
from nanoai.harmonizer import Harmonizer
from nanoai.master_tokenizer import MasterTokenizer
from nanoai.memory import Memory
from nanoai.perception import attach_perception


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--source-image", type=Path, required=True)
    p.add_argument("--see-executable", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    tokenizer = MasterTokenizer.load(args.tokenizer)
    memory = Memory(args.out / "journal")
    memory.initialize(DEFAULT_IDENTITY)
    memory.put_source("description", "This is a source-bound image observation. Gözlemin kaynağı ve yorumu ayrı tutulur.",
                      note="Authored integration caption; no object recognition claim")
    memory.set_task("catalogue", "Retain exact language and image evidence with its context")
    owner = LanguageCatalogue(memory)
    observation = owner.observe("words-1", "catalogue", "description", tokenizer)
    before_retry = memory.sequence
    assert owner.observe("words-1", "catalogue", "description", tokenizer) == observation
    assert memory.sequence == before_retry
    with sqlite3.connect(args.bundle.resolve().as_uri() + "?mode=ro", uri=True) as database:
        regions = [r[0] for r in database.execute("SELECT id FROM region ORDER BY area DESC LIMIT 2")]
    perception = attach_perception(memory, "image-1", "catalogue", "description", bundle=args.bundle,
        executable=args.see_executable, region_ids=regions, source_image=args.source_image)
    current = owner.view("catalogue", inventory=args.inventory)
    assert current["perceptions"][0]["current"]
    assert any(row.get("native_witnesses") for row in current["pieces"])
    command = [sys.executable, "-m", "nanoai.cli", "--state", str(memory.root), "catalogue", "catalogue", "--inventory", str(args.inventory)]
    fresh = json.loads(subprocess.check_output(command, text=True, timeout=60))
    assert fresh == current
    source = memory.state["sources"]["description"]
    packet = Harmonizer(memory).view(memory.snapshot("catalogue", [source], "qualified-view"))
    assert {r["kind"] for r in packet["entries"]} >= {"language_observation", "perception_reference"}
    (args.out / "catalogue.json").write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n")
    (args.out / "dictionary.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n")
    memory.put_source("description", "The attribution changed; retain the original observation as history.", expected_revision=1,
                      note="Source-revision consumer check")
    stale = owner.view("catalogue", include_history=True)
    assert not stale["observations"][0]["current"] and not stale["perceptions"][0]["current"]
    assert owner.view("catalogue")["pieces"] == []
    owner.observe("words-2", "catalogue", "description", tokenizer)
    report = {"schema": "nanoai.catalogue-qualification.v1", "tokenizer_identity": tokenizer.identity,
        "bundle_identity": perception["bundle"]["identity"], "engine_identity": perception["engine"]["identity"],
        "sample_identity": perception["measurement"]["source"]["samples_digest"],
        "image_size": [perception["measurement"]["source"][k] for k in ("width", "height")],
        "regions": perception["measurement"]["regions"], "pieces": len(current["pieces"]),
        "native_witnesses": sum(len(r["native_witnesses"]) for r in current["pieces"]),
        "checks": {"retry_once": True, "fresh_cli_recovery": True, "dictionary_integration": True,
                   "source_correction_invalidates_language_and_perception": True, "new_observation_after_correction": True},
        "catalogue_identity": file_digest(args.out / "catalogue.json"), "parameter_updates": 0,
        "scope": "Real saved painting, native sample/mask checks and journal consumer; no semantic-recognition or GUI claim"}
    (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
