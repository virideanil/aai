"""Keep exact small source inputs beside local experimental receipts."""
import json
from pathlib import Path
import shutil
from nanoai.core import file_digest


def snapshot_sources(destination, config):
    project = Path(__file__).resolve().parents[1]
    rows = []
    selected = set(project.glob("src/nanoai/*.py")) | set(project.glob("tools/*.py")) | set(project.glob("evals/*"))
    for source in sorted(selected):
        if not source.is_file():
            continue
        relative = source.relative_to(project)
        target = destination / "source-snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        identity = file_digest(source)
        shutil.copyfile(source, target)
        if file_digest(target) != identity or file_digest(source) != identity:
            raise ValueError("Source changed during evidence capture")
        rows.append({"source": str(source), "snapshot": str(target.relative_to(destination)), "digest": identity})
    for index, path in enumerate(config.get("source_files", [])):
        source = Path(path)
        if source in selected:
            continue
        target = destination / "source-snapshot" / "native-dependency" / f"{index}-{source.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        identity = file_digest(source)
        shutil.copyfile(source, target)
        if file_digest(target) != identity or file_digest(source) != identity:
            raise ValueError("Dependency source changed during evidence capture")
        rows.append({"source": str(source), "snapshot": str(target.relative_to(destination)), "digest": identity})
    (destination / "source-snapshot.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows
