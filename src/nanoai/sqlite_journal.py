"""SQLite-owned atomic event commits with exact JSON and a verified hash chain.

The domain lock spans validation and filesystem effects. Each event append has
its own durable SQLite commit, so a prepared effect survives process death.
"""
from contextlib import contextmanager, closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3

APPLICATION_ID = 0x4E414931


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def digest(parent, payload):
    prefix = canonical({"schema": "nanoai.sqlite-transaction.v1", "parent": parent})
    return "shake256:" + hashlib.shake_256(prefix + b"\n" + payload).hexdigest(64)


@contextmanager
def locked(path):
    with open(path, "a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield stream
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def sync_directory(path):
    if os.name == "nt":
        return False  # Directory flush needs separate native Windows qualification.
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def _connect(root):
    db = (Path(root) / "journal.sqlite3").resolve()
    connection = sqlite3.connect(db.as_uri() + "?mode=rw", uri=True, isolation_level=None, timeout=10)
    try:
        if (connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID
                or connection.execute("PRAGMA user_version").fetchone()[0] != 1):
            raise ValueError("Unsupported nanoai SQLite journal identity or version")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        # macOS asks its VFS for a complete device flush; other VFSes may ignore it.
        connection.execute("PRAGMA fullfsync=ON")
        return connection
    except BaseException:
        connection.close()
        raise


def initialize(root):
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Journal directory must be owned")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    db = root / "journal.sqlite3"
    if db.is_symlink():
        raise ValueError("Journal database must be an owned regular file")
    if db.exists():
        with closing(_connect(root)):
            return
    # Caller holds the domain lock. Preserve an unrelated existing database.
    with closing(sqlite3.connect(db, isolation_level=None)) as connection:
        os.chmod(db, 0o600)
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA fullfsync=ON")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("CREATE TABLE transactions (sequence INTEGER PRIMARY KEY CHECK(sequence>0), "
                           "parent TEXT REFERENCES transactions(digest), digest TEXT NOT NULL UNIQUE, "
                           "payload BLOB NOT NULL)")
        connection.execute("CREATE TABLE journal (singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
                           "sequence INTEGER NOT NULL CHECK(sequence>=0), catalogue TEXT, "
                           "head TEXT REFERENCES transactions(digest))")
        connection.execute("INSERT INTO journal VALUES (1,0,NULL,NULL)")
        connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
        connection.execute("PRAGMA user_version=1")
        connection.execute("COMMIT")
    sync_directory(root)


def _state(connection):
    row = connection.execute("SELECT sequence,catalogue,head FROM journal WHERE singleton=1").fetchone()
    last = connection.execute("SELECT sequence,parent,digest,payload FROM transactions ORDER BY sequence DESC LIMIT 1").fetchone()
    if last is None:
        if row != (0, None, None):
            raise ValueError("SQLite journal head and committed rows differ")
    elif row is None or row != (last[0], _read(last)["catalogueId"], last[2]):
        raise ValueError("SQLite journal head and committed rows differ")
    return row


def head(root):
    with closing(_connect(root)) as connection:
        connection.execute("BEGIN")
        return _state(connection)[2]


def _read(row):
    sequence, parent, identity, payload = row
    if not isinstance(payload, bytes) or digest(parent, payload) != identity:
        raise ValueError("SQLite journal transaction digest mismatch")
    transaction = json.loads(payload)
    if (canonical(transaction) != payload or transaction["sequence"] != sequence
            or transaction["previous"] != sequence - 1
            or not isinstance(transaction["events"], list) or not transaction["events"]):
        raise ValueError("SQLite journal transaction contract differs")
    return transaction


def tail(root, after, *, at_head):
    with closing(_connect(root)) as connection:
        connection.execute("BEGIN")
        _state(connection)
        def sequence(identity):
            if identity is None:
                return 0
            row = connection.execute("SELECT sequence FROM transactions WHERE digest=?", (identity,)).fetchone()
            if row is None:
                raise ValueError("SQLite journal cursor is unavailable")
            return row[0]
        start, end = sequence(after), sequence(at_head)
        if start > end:
            raise ValueError("SQLite journal cursor precedes retained projection")
        parent, expected, catalogue = after, start + 1, None
        for row in connection.execute("SELECT sequence,parent,digest,payload FROM transactions "
                                      "WHERE sequence>? AND sequence<=? ORDER BY sequence", (start, end)):
            if row[0] != expected or row[1] != parent:
                raise ValueError("SQLite journal transaction chain differs")
            transaction = _read(row)
            if catalogue is not None and transaction["catalogueId"] != catalogue:
                raise ValueError("SQLite journal catalogue identity differs")
            catalogue, parent = transaction["catalogueId"], row[2]
            expected += 1
            yield row[2], transaction
        if parent != at_head or expected != end + 1:
            raise ValueError("SQLite journal transaction chain is incomplete")


def append(root, transaction, expected_head):
    payload = canonical(transaction)
    if (set(transaction) != {"catalogueId", "previous", "sequence", "events"}
            or not isinstance(transaction["catalogueId"], str) or not transaction["catalogueId"]
            or type(transaction["sequence"]) is not int or type(transaction["previous"]) is not int
            or not isinstance(transaction["events"], list) or not transaction["events"]):
        raise ValueError("SQLite journal append contract")
    with closing(_connect(root)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        sequence, catalogue, current = _state(connection)
        if current != expected_head:
            raise ValueError("Stale SQLite journal head")
        if (transaction["previous"] != sequence or transaction["sequence"] != sequence + 1
                or catalogue is not None and transaction["catalogueId"] != catalogue):
            raise ValueError("SQLite journal sequence or catalogue identity differs")
        identity = digest(current, payload)
        connection.execute("INSERT INTO transactions VALUES (?,?,?,?)", (sequence + 1, current, identity, payload))
        connection.execute("UPDATE journal SET sequence=?,catalogue=?,head=? WHERE singleton=1",
                           (sequence + 1, transaction["catalogueId"], identity))
        connection.execute("COMMIT")
        return identity
