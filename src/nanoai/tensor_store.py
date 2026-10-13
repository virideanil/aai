"""SQLite tensor inventory and learning lineage over unchanged native files.

Tables own queryable metadata. GGUF/safetensors own tensor values. Readers verify
the referenced bytes; a catalogue entry grants neither execution nor admission.
"""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import struct
import time

from .core import digest_bytes, file_digest
from .sqlite_journal import canonical, locked, sync_directory

APPLICATION_ID = 0x4E415432
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
               "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "U32": 4,
               "F64": 8, "I64": 8, "U64": 8}


def identity(value):
    if not isinstance(value, str) or re.fullmatch(r"shake256:[0-9a-f]{128}", value) is None:
        raise ValueError("Expected a complete SHAKE256 identity")
    return value


def signature(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def describe_reference(reference, *, directory, expected):
    """Validate a manifest's immutable version; the caller verifies payload bytes."""
    if (not isinstance(reference, dict) or set(reference) != {"database", "version", "artifact"}
            or not isinstance(reference["database"], str) or not reference["database"]
            or "\x00" in reference["database"]):
        raise ValueError("Tensor catalogue reference fields")
    identity(expected); identity(reference["artifact"]); identity(reference["version"])
    if reference["artifact"] != expected:
        raise ValueError("Tensor catalogue refers to a different artifact")
    database = Path(directory) / reference["database"]
    row = TensorStore(database).describe(reference["version"])
    if row["artifact"] != expected:
        raise ValueError("Tensor catalogue version refers to a different artifact")
    return row


def inventory(path):
    """Read declared native tensor extents without loading tensor values."""
    path = Path(path)
    extent = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if prefix[:4] == b"GGUF":
            from .gguf import inspect
            inspected = inspect(path)
            return "GGUF", inspected["tensors"]
        if len(prefix) != 8:
            raise ValueError("Tensor file header is incomplete")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= min(16 << 20, extent - 8):
            raise ValueError("Unsupported tensor format or safetensors header extent")
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate safetensors header key")
                result[key] = value
            return result
        header = json.loads(stream.read(length), object_pairs_hook=unique)
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    rows = []
    for name, row in header.items():
        if name == "__metadata__":
            if not isinstance(row, dict) or any(not isinstance(v, str) for v in row.values()):
                raise ValueError("Safetensors source metadata differs")
            continue
        if (not name or not isinstance(row, dict) or set(row) != {"dtype", "shape", "data_offsets"}
                or row["dtype"] not in DTYPE_BYTES or not isinstance(row["shape"], list)
                or len(row["shape"]) > 16
                or any(type(d) is not int or not 0 <= d <= 2**40 for d in row["shape"])):
            raise ValueError("Safetensors tensor contract differs")
        offsets = row["data_offsets"]
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(d) is not int for d in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= extent - length - 8):
            raise ValueError("Safetensors tensor offsets differ")
        elements = math.prod(row["shape"])
        if offsets[1] - offsets[0] != elements * DTYPE_BYTES[row["dtype"]]:
            raise ValueError("Safetensors tensor byte count differs")
        rows.append({"name": name, "dtype": row["dtype"], "shape": row["shape"],
                     "elements": elements, "offset": 8 + length + offsets[0],
                     "bytes": offsets[1] - offsets[0], "layout": "row-major"})
    cursor = 8 + length
    for row in sorted(rows, key=lambda r: (r["offset"], r["bytes"])):
        if row["offset"] != cursor:
            raise ValueError("Safetensors tensor extents overlap or leave a gap")
        cursor += row["bytes"]
    if not rows or cursor != extent:
        raise ValueError("Safetensors payload is incomplete")
    return "safetensors", sorted(rows, key=lambda r: r["name"])


SCHEMA = (
    "CREATE TABLE artifacts (id TEXT PRIMARY KEY, format TEXT NOT NULL, byte_count INTEGER NOT NULL CHECK(byte_count>0), "
    "inventory_id TEXT NOT NULL, tensor_count INTEGER NOT NULL CHECK(tensor_count>0)) STRICT",
    "CREATE TABLE locations (artifact TEXT NOT NULL REFERENCES artifacts(id), path TEXT NOT NULL, "
    "PRIMARY KEY(artifact,path)) STRICT",
    "CREATE TABLE tensors (artifact TEXT NOT NULL REFERENCES artifacts(id), name TEXT NOT NULL, dtype TEXT NOT NULL, "
    "elements INTEGER NOT NULL CHECK(elements>=0), byte_offset INTEGER NOT NULL CHECK(byte_offset>=0), "
    "byte_count INTEGER NOT NULL CHECK(byte_count>=0), layout TEXT NOT NULL, rank INTEGER NOT NULL CHECK(rank BETWEEN 0 AND 16), "
    "PRIMARY KEY(artifact,name)) STRICT",
    "CREATE TABLE dimensions (artifact TEXT NOT NULL, name TEXT NOT NULL, axis INTEGER NOT NULL CHECK(axis>=0), "
    "extent INTEGER NOT NULL CHECK(extent>=0), PRIMARY KEY(artifact,name,axis), "
    "FOREIGN KEY(artifact,name) REFERENCES tensors(artifact,name)) STRICT",
    "CREATE TABLE versions (id TEXT PRIMARY KEY, artifact TEXT NOT NULL REFERENCES artifacts(id), basis TEXT NOT NULL, "
    "parent TEXT REFERENCES versions(id), recipe TEXT NOT NULL, source TEXT NOT NULL, created_ns INTEGER NOT NULL) STRICT",
    "CREATE INDEX versions_basis ON versions(basis,created_ns)",
    "CREATE TABLE runs (id TEXT PRIMARY KEY, protocol TEXT NOT NULL, created_ns INTEGER NOT NULL) STRICT",
    "CREATE TABLE learning_events (run TEXT NOT NULL REFERENCES runs(id), sequence INTEGER NOT NULL CHECK(sequence>0), "
    "phase TEXT NOT NULL, version TEXT REFERENCES versions(id), epoch INTEGER, updates INTEGER, seconds REAL, loss REAL, "
    "payload BLOB NOT NULL, payload_id TEXT NOT NULL, PRIMARY KEY(run,sequence)) STRICT",
)


class TensorStore:
    def __init__(self, path, *, create=False):
        original = Path(path).absolute()
        if original.is_symlink():
            raise ValueError("Tensor database must be an owned regular file")
        self.path = original.resolve()
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with locked(self.path.with_suffix(self.path.suffix + ".lock")):
                if not self.path.exists():
                    connection = sqlite3.connect(self.path, isolation_level=None)
                    try:
                        os.chmod(self.path, 0o600)
                        connection.execute("PRAGMA foreign_keys=ON")
                        connection.execute("PRAGMA synchronous=FULL")
                        connection.execute("PRAGMA fullfsync=ON")
                        connection.execute("BEGIN IMMEDIATE")
                        for statement in SCHEMA:
                            connection.execute(statement)
                        connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
                        connection.execute("PRAGMA user_version=1")
                        connection.execute("COMMIT")
                    finally:
                        connection.close()
                    sync_directory(self.path.parent)
        with self.connection():
            pass

    @contextmanager
    def connection(self, *, write=False):
        connection = sqlite3.connect(self.path.as_uri() + ("?mode=rw" if write else "?mode=ro"),
                                     uri=True, isolation_level=None, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            if (connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID
                    or connection.execute("PRAGMA user_version").fetchone()[0] != 1):
                raise ValueError("Unsupported tensor database identity or version")
            connection.execute("PRAGMA foreign_keys=ON")
            if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
                raise RuntimeError("Tensor database foreign keys are unavailable")
            if write:
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("PRAGMA fullfsync=ON")
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield connection
            if write:
                connection.execute("COMMIT")
        finally:
            connection.close()

    def register(self, path, *, expected, basis, recipe, source, parent=None):
        identity(expected); identity(basis)
        if not isinstance(recipe, str) or not recipe or not isinstance(source, str) or not source:
            raise ValueError("Tensor version needs a recipe and source locator")
        if parent is not None:
            identity(parent)
        path = Path(path).resolve(strict=True)
        before = signature(path)
        if file_digest(path) != expected:
            raise ValueError("Tensor source identity differs")
        fmt, rows = inventory(path)
        if signature(path) != before:
            raise ValueError("Tensor source changed during registration")
        inventory_id = digest_bytes(canonical(rows))
        version_data = dict(artifact=expected, basis=basis, parent=parent, recipe=recipe, source=source)
        version = digest_bytes(canonical(version_data))
        with self.connection(write=True) as connection:
            found = connection.execute("SELECT * FROM artifacts WHERE id=?", (expected,)).fetchone()
            values = (expected, fmt, before[2], inventory_id, len(rows))
            if found is not None and tuple(found) != values:
                raise ValueError("Registered tensor inventory differs")
            if found is None:
                connection.execute("INSERT INTO artifacts VALUES (?,?,?,?,?)", values)
                for row in rows:
                    connection.execute("INSERT INTO tensors VALUES (?,?,?,?,?,?,?,?)", (expected, row["name"],
                        row["dtype"], row["elements"], row["offset"], row["bytes"], row["layout"], len(row["shape"])))
                    connection.executemany("INSERT INTO dimensions VALUES (?,?,?,?)",
                        [(expected, row["name"], i, d) for i, d in enumerate(row["shape"])])
            elif self._tensors(connection, expected) != rows:
                raise ValueError("Stored tensor rows differ from the source")
            connection.execute("INSERT OR IGNORE INTO locations VALUES (?,?)",
                               (expected, os.path.relpath(path, self.path.parent)))
            connection.execute("INSERT OR IGNORE INTO versions VALUES (?,?,?,?,?,?,?)",
                (version, expected, basis, parent, recipe, source, time.time_ns()))
        return {"database": str(self.path), "version": version, "artifact": expected}

    @staticmethod
    def _tensors(connection, artifact):
        rows = []
        for row in connection.execute("SELECT * FROM tensors WHERE artifact=? ORDER BY name", (artifact,)):
            dimensions = list(connection.execute("SELECT axis,extent FROM dimensions WHERE artifact=? AND name=? ORDER BY axis",
                                                 (artifact, row["name"])))
            if [d[0] for d in dimensions] != list(range(row["rank"])):
                raise ValueError("Stored tensor dimensions are incomplete")
            rows.append({"name": row["name"], "dtype": row["dtype"], "shape": [d[1] for d in dimensions],
                         "elements": row["elements"], "offset": row["byte_offset"], "bytes": row["byte_count"],
                         "layout": row["layout"]})
        return rows

    def describe(self, version):
        """Read verified metadata; this is separate from checking live payload bytes."""
        identity(version)
        with self.connection() as connection:
            row = connection.execute("SELECT v.*,a.format,a.byte_count,a.inventory_id,a.tensor_count FROM versions v "
                                     "JOIN artifacts a ON a.id=v.artifact WHERE v.id=?", (version,)).fetchone()
            if row is None:
                raise ValueError("Tensor version is unavailable")
            result = dict(row)
            if digest_bytes(canonical({k: result[k] for k in ("artifact", "basis", "parent", "recipe", "source")})) != version:
                raise ValueError("Tensor version provenance changed")
            result["tensors"] = self._tensors(connection, row["artifact"])
            if (len(result["tensors"]) != row["tensor_count"]
                    or digest_bytes(canonical(result["tensors"])) != row["inventory_id"]):
                raise ValueError("Stored tensor inventory identity differs")
            result["locations"] = [str((self.path.parent / r[0]).resolve()) for r in connection.execute(
                "SELECT path FROM locations WHERE artifact=? ORDER BY path", (row["artifact"],))]
        return result

    def resolve(self, version, *, path=None):
        result = self.describe(version)
        paths = [Path(p) for p in result["locations"]]
        path = Path(path).resolve(strict=True) if path is not None else next((p for p in paths if p.is_file()), None)
        if path is None or path not in paths:
            raise ValueError("Registered tensor location is unavailable")
        before = signature(path)
        if before[2] != result["byte_count"] or file_digest(path) != result["artifact"]:
            raise ValueError("Registered tensor bytes changed")
        fmt, rows = inventory(path)
        if fmt != result["format"] or rows != result["tensors"]:
            raise ValueError("Registered tensor inventory differs from the payload")
        if signature(path) != before:
            raise ValueError("Tensor source changed during read")
        result["path"] = str(path)
        return result

    def record(self, run, *, protocol, phase, values, version=None):
        """Append one event atomically; detailed observations retain exact JSON bytes."""
        identity(protocol)
        if not run or not phase:
            raise ValueError("Learning event needs a run and phase")
        payload = canonical(values)
        if len(payload) > 4 << 20:
            raise ValueError("Learning event extent")
        fields = [values.get(k) for k in ("epoch", "parameter_updates", "elapsed_seconds", "loss")]
        if any(v is not None and (type(v) is not int or v < 0) for v in fields[:2]):
            raise ValueError("Learning counters must be nonnegative integers")
        if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v) or v < 0) for v in fields[2:]):
            raise ValueError("Learning measurements must be finite and nonnegative")
        with self.connection(write=True) as connection:
            connection.execute("INSERT OR IGNORE INTO runs VALUES (?,?,?)", (run, protocol, time.time_ns()))
            if connection.execute("SELECT protocol FROM runs WHERE id=?", (run,)).fetchone()[0] != protocol:
                raise ValueError("Learning run protocol changed")
            sequence = connection.execute("SELECT COALESCE(MAX(sequence),0)+1 FROM learning_events WHERE run=?", (run,)).fetchone()[0]
            connection.execute("INSERT INTO learning_events VALUES (?,?,?,?,?,?,?,?,?,?)",
                (run, sequence, phase, version, *fields, payload, digest_bytes(payload)))
        return sequence

    def events(self, run):
        with self.connection() as connection:
            result = []
            for row in connection.execute("SELECT * FROM learning_events WHERE run=? ORDER BY sequence", (run,)):
                if digest_bytes(row["payload"]) != row["payload_id"]:
                    raise ValueError("Learning event bytes changed")
                result.append({"sequence": row["sequence"], "phase": row["phase"], "version": row["version"],
                               "values": json.loads(row["payload"])})
            return result
