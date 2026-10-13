"""SEE bundle/1 references with measured masks and attributed source text."""
from copy import deepcopy
from pathlib import Path
import re
import sqlite3
import subprocess
import tempfile

from .core import file_digest
from .memory import identifier


def read_see_bundle(bundle, executable, region_ids, *, source_image=None):
    from PIL import Image
    bundle, executable = Path(bundle).resolve(strict=True), Path(executable).resolve(strict=True)
    if (not isinstance(region_ids, (list, tuple)) or not 1 <= len(region_ids) <= 16
            or any(type(i) is not int or not 0 <= i < 4_000_000 for i in region_ids)
            or len(set(region_ids)) != len(region_ids)):
        raise ValueError("Perception region IDs")
    if not bundle.is_file() or not 1 <= bundle.stat().st_size <= 268435456:
        raise ValueError("Perception bundle extent")
    if any(Path(str(bundle) + suffix).exists() for suffix in ("-wal", "-journal")):
        raise ValueError("Perception requires a sealed standalone bundle")
    before, engine = file_digest(bundle), file_digest(executable)
    with sqlite3.connect(bundle.as_uri() + "?mode=ro", uri=True) as database:
        database.execute("PRAGMA query_only=ON")
        if database.execute("PRAGMA application_id").fetchone()[0] != 1397050673:
            raise ValueError("Perception bundle application")
        metadata = dict(database.execute("SELECT key,value FROM meta"))
        if metadata.get("schema") != "see.bundle/1":
            raise ValueError("Perception bundle schema")
        source = database.execute("SELECT name,format,decoder,file_bytes,file_digest,width,height,colour,depth,samples_digest FROM source").fetchmany(2)
        if len(source) != 1:
            raise ValueError("Perception bundle source count")
        values = source[0]
        measurement = dict(zip(("name", "format", "decoder", "file_bytes", "file_digest", "width", "height", "colour", "depth", "samples_digest"), values))
        w, h = measurement["width"], measurement["height"]
        if type(w) is not int or type(h) is not int or not 1 <= w * h <= 16_777_216 or w <= 0 or h <= 0:
            raise ValueError("Perception image extent")
        if not re.fullmatch(r"tuplehash256\(see.samples/1\):[0-9a-f]{128}", measurement["samples_digest"]):
            raise ValueError("Perception sample identity")
        regions = []
        readings = []
        for region_id in region_ids:
            records = database.execute("SELECT id,lo,hi,area,x0,y0,x1,y1 FROM region WHERE id=?", (region_id,)).fetchmany(2)
            if len(records) != 1 or any(type(v) is not int for v in records[0]):
                raise ValueError("Perception region extent")
            rid, lo, hi, area, x0, y0, x1, y1 = records[0]
            if not (0 <= lo < hi and 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and 1 <= area <= w * h):
                raise ValueError("Perception region geometry")
            regions.append({"id": rid, "label_range": [lo, hi], "area": area, "box": [x0, y0, x1, y1]})
            readings += [{"region_id": rid, "role": role, "basis": basis, "kind": "engine_interpretation"}
                         for role, basis in database.execute("SELECT r.role,b.basis FROM reading r JOIN reading_basis b ON r.role=b.role WHERE r.region=?", (rid,))]
    original = None
    if source_image is not None:
        source_image = Path(source_image).resolve(strict=True)
        if source_image.stat().st_size != measurement["file_bytes"] or file_digest(source_image) != measurement["file_digest"]:
            raise ValueError("Perception original file differs from the bundle")
        original = {"path": str(source_image), "identity": measurement["file_digest"]}
    with tempfile.TemporaryDirectory(prefix="nanoai-see-") as directory:
        destination = Path(directory)
        # The native reader verifies decoded sample identity and exact re-encode.
        subprocess.run([str(executable), "encode", str(bundle), "-o", str(destination / "samples.png")],
                       check=True, capture_output=True, text=True, timeout=60)
        for region in regions:
            mask_path = destination / f"mask-{region['id']}.png"
            response = subprocess.run([str(executable), "mask", str(bundle), str(region["id"]), "-o", str(mask_path)],
                                      check=True, capture_output=True, text=True, timeout=60)
            matches = re.findall(r"^digest\s+(tuplehash256\(see.mask/1\):[0-9a-f]{128})$", response.stdout, re.M)
            if len(matches) != 1:
                raise ValueError("Perception native mask identity")
            with Image.open(mask_path) as mask:
                if mask.size != (w, h) or mask.mode != "1":
                    raise ValueError("Perception native mask shape")
                pixels = mask.convert("L")
                if sum(pixels.tobytes()) != 255 * region["area"] or list(pixels.getbbox() or ()) != region["box"]:
                    raise ValueError("Perception mask area or box differs from region")
            region["mask_identity"] = matches[0]
            region["mask_png_identity"] = file_digest(mask_path)
    if file_digest(bundle) != before or file_digest(executable) != engine:
        raise ValueError("Perception bundle or native engine changed during observation")
    if original is not None and file_digest(source_image) != original["identity"]:
        raise ValueError("Perception original file changed during observation")
    return {"bundle": {"path": str(bundle), "identity": before, "schema": "see.bundle/1"},
            "original_file": original,
            "engine": {"identity": engine, "declared": metadata.get("engine")},
            "measurement": {"source": measurement, "regions": regions,
                "coordinate_frame": {"units": "source pixels", "x": "right", "y": "down", "boxes": "half-open"},
                "transform": {"kind": "identity", "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}},
            "engine_interpretations": readings}


def attach_perception(memory, request_id, task_id, source_id, *, bundle, executable, region_ids, source_image=None):
    """Bind a source's attributed interpretation to checked measurements."""
    identifier(request_id)
    memory.reload()
    source = deepcopy(memory.state["sources"][source_id])
    snapshot = memory.snapshot(task_id, [source], "see-bundle-port-v1")
    reference = read_see_bundle(bundle, executable, region_ids, source_image=source_image)
    record = {"schema": "nanoai.perception-reference.v1", "id": request_id, "snapshot": snapshot,
              "source": {k: source[k] for k in ("id", "revision", "digest")}, **reference,
              "interpretation": {"text": source["text"], "attribution": source["note"],
                                 "status": "attributed source interpretation; sample checks establish its evidence only"}}
    with memory.locked():
        memory._validate_snapshot(snapshot)
        prior = memory.state["perception_references"].get(request_id)
        if prior is not None:
            if prior != record:
                raise ValueError("Perception retry identity names different evidence")
            return deepcopy(prior)
        memory._append({"kind": "catalogue.perception", "perception": record})
    return deepcopy(record)
