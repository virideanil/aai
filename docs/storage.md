# Durable work and exact history

New nano'ai states use SQLite through Python's standard library. Transactions
store canonical UTF-8 JSON as bytes, with a sequence, parent identity and full
SHAKE256 digest. Source revisions, tasks, experiences, token observations and
artifact receipts share this event history. SQL owns the atomic event and head
commit; the in-memory domain state is rebuilt from verified records.

The first storage comparison used the same 80 source records and four native
token observations, ending at 86 committed transactions, repeated three times.
The median complete workload took 1.094 seconds with the existing SJQL path and
1.188 seconds with SQLite. Median reopen was 20.73 ms versus 5.76 ms. SQLite stored
299,008 bytes versus about 76,036 compressed SJQL bytes. SQLite therefore earns
its place through simpler installation and faster recovery at this scale, with
an explicit space cost and no write-speed claim. These are small shared-host
measurements, not a universal engine comparison.

SQLite uses explicit `BEGIN IMMEDIATE`, `synchronous=FULL` and a requested
`fullfsync` on macOS. Reads pin a transaction snapshot while traversing the
sequence. Connections close deterministically. The implementation checks its
application ID and schema version before using an existing database. An unrelated
database is preserved and rejected. The [SQLite transaction contract](https://www.sqlite.org/lang_transaction.html)
and [atomic-commit assumptions](https://www.sqlite.org/atomiccommit.html) describe
the engine guarantees and their dependency on the filesystem and hardware.

The domain lock covers validation and local effects. A prepared artifact event
commits before create-only file publication. Exact bytes are read back before the
observed receipt commits. Recovery can therefore observe a file published before
process death without writing it again. A changed artifact remains a conflict for
that operation; independent work can continue.

Derived sources carry the exact parent snapshot and any observed artifact
receipts. A change to a parent source, task, calculation or artifact makes derived
work historical. A historical draft can be quoted explicitly as a record of what
was written; the quote retains its artifact lineage and grants its old claims no
current factual status.

Existing `transactions.sjql.d/HEAD` states continue through their maintained SJQL
owner. Supply that package or `NANOAI_SJQL_SRC` only for those legacy states.
Opening them neither migrates their bytes nor creates a competing SQLite history.
Two detected journal owners are rejected for explicit reconciliation. An explicit
`Memory(path, backend="sjql")` remains available for compatibility qualification.
New SQLite states have no SJQL dependency.

The real consumer checks cover exact text and large JSON integers, stale append,
two writer processes, fresh read, corrupt earlier history, unrelated database
preservation, death before a SQL commit and death after artifact publication.
The executed native platform is macOS arm64. Windows locking and power-loss
durability require their own native-host evidence. Current projection recovery
replays the complete event history; a larger continuing workload should measure
that cost before introducing a new index or snapshot owner.

## Tensor holdings and learning history

`TensorStore` uses a separate, explicitly named SQLite database at each native
profile or experiment owner. Its normalized tables record actual tensor names,
dtypes, shapes, byte ranges, source bases, immutable versions and parent links.
Run and learning-event tables record the protocol identity, epoch, update count,
elapsed time and loss. Detailed observations remain exact JSON event bytes in
the database; SQL columns support direct queries without parsing every payload.

Native preparation and candidate registration populate this catalogue from the
actual GGUF or safetensors headers. Registration verifies file identity and
extent before committing. Readers check the metadata identity, the referenced
payload and its actual tensor header. A loaded native session retains the
verified inventory identity, so rewriting SQL metadata and its checksum cannot
reinterpret unchanged tensor bytes. Unrelated learning-event appends remain
compatible. A version records provenance; execution and candidate admission retain
their existing checks. The original tensor files keep their binary values and
their existing streaming or mapping behavior.

The native profile carries a small database/version reference. Portable runtime
configuration, upstream model headers, JSONL/HTTP messages, authored datasets and
exported measurements retain their established interchange formats. SQLite owns
the evolving catalogue and run history. JSON inside an event preserves its exact
content; it does not stand in for the catalogue's typed tables.

Filesystem publication and database commits have separate recovery boundaries.
The candidate writer publishes and verifies bytes, registers tensor metadata,
then commits the learner event. An interruption can leave an inert file or an
unreferenced catalogue row. Repeating registration checks and reuses the same
immutable version. Existing event journals and earlier measurements are retained.
