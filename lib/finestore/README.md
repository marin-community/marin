finestore
---
Raison d'être: an append-only columnar archive of Parquet shards on object
storage, for eval output that is written as many small batches and read back
with column projection and predicate filtering. It targets stores that only
create whole objects (no seek, no rewrite): a write is one immutable Parquet
object, and there is no manifest to keep consistent.

## Data model

- `DataStore` — a writable archive under one URL prefix, written by one
  `writer_id`. It holds typed `DataTable`s and a reserved `blobs` table.
- `DataTable` — an append-only stream of rows. Appends buffer in memory; a
  background thread flushes each buffer to an immutable Parquet shard.
- `CompositeReader` — composes every writer's shards for a table, unifies their
  schemas, deduplicates by primary key, and projects/filters on read.

Rows carry a caller-defined schema plus two stamped columns: `_writer` (the
producing writer) and `_seq` (a per-writer monotonic id). A table declares a
`primary_key`; the reader keeps one row per key.

`DataStore.write(name, metadata, data)` stores an opaque payload in the `blobs`
table and returns a `finestore://blobs/<name>` reference. A sample row holds
that reference in place of the payload (e.g. an agent trajectory), and
`CompositeReader.resolve(uri)` reads it back. Everything the archive owns lives
inside the archive, addressable by a `finestore://` URI.

Import from the submodules directly; the package re-exports nothing.

```python
from finestore.store import DataStore
from finestore.reader import CompositeReader

with DataStore.open("gs://bucket/run/results", writer_id="evalchemy") as store:
    samples = store.table("samples", primary_key=("task", "doc_id"))
    samples.append({"task": "arc", "doc_id": "1", "correct": True})
    store.seal()  # mark the run complete

reader = CompositeReader("gs://bucket/run/results")
table = reader.scan("samples", columns=["task", "doc_id", "correct"],
                    where=[("task", "==", "arc")])
```

## On-disk layout

An archive lives entirely under `{root}/_finestore/`, so it never collides with
sibling data under the same run prefix. Each table is a directory of immutable
shards, partitioned by the writer that produced them and the compaction
generation they belong to:

```
{root}/_finestore/
    SEALED                                   # optional: the run is complete
    {table}/_schema.json                     # primary key + schema version
    {table}/w={writer}/g={gen}/{seq:016d}-{uid}.parquet
```

Shard membership is discovered by listing the table directory. A shard's schema
and row-group statistics come from its Parquet footer. `_schema.json` records
only what a footer cannot — the dedup primary key and a logical schema version —
and is the archive's whole "manifest". The writer and generation are encoded in
the object key and recovered by listing; nothing else references a shard.

## Read semantics

`scan` lists a table's shards, unifies their footers' schemas (a column a later
writer added is promoted to null for older shards), reads only the projected
columns plus whatever the dedup key needs, and deduplicates by primary key. For
each key it keeps the row from the highest compaction generation, breaking ties
by the highest `_seq`. That single rule makes a duplicate delivery, a retried
flush, and a crash mid-compaction all converge to one row without coordination.

`point` returns the single row matching `key=value` for every key. Predicate
filters support `==`, `!=`, and `in`, pushed into the Parquet scan.

Because schema unification is null-permissive, a reader that opened the archive
without opening the writer — for example the dashboard — reads a run written by
a newer or older schema without changes: absent columns read back as null and
the row model's defaults fill them in.

## Writing

Appends are non-blocking and buffer in memory. The background thread flushes
each table on a time ceiling (`DEFAULT_FLUSH_INTERVAL`, 5s) so shards stay
fresh, and immediately if a buffer crosses a row cap (`DEFAULT_MAX_BUFFER_ROWS`)
so a burst cannot grow it without bound. `table.flush()` persists one table;
`store.flush()` persists all of them; `flush` and `close` block until every
buffered row is a durable object — the "writes block until persisted" guarantee.
Each flush writes to a temporary key and atomically renames it into place, so a
partial object is never visible.

Concurrent writers of one run (for example RL rollout workers) each open a
`DataStore` with a distinct `writer_id`. They write under their own key prefix
and never coordinate; the reader composes and deduplicates them. Call `seal`
once, after every writer has finished, to drop the `SEALED` marker.

## Compaction

`compact(root, table)` merges a table's level-0 shards into one next-generation
shard, sorted by primary key. It writes the merged shard first, then deletes the
sources. Compaction is optional and safe to run at any time — end of run, or in
the background — because the dedup rule already yields correct reads across mixed
generations. It reduces object count and improves read locality; it does not
change query results.

## Recovery

There is no manifest and no rewrite, so recovery is a listing, not a repair.

- Crash mid-flush: the atomic rename means an interrupted flush leaves at most a
  temporary object, never a torn shard. Re-running the writer re-appends the
  unflushed rows under fresh `_seq` values; dedup collapses any overlap.
- Crash mid-compaction: if the merged shard was written but its sources not yet
  deleted, both generations are present. Dedup keeps the higher-generation row,
  so reads are correct; a later `compact` finishes the cleanup.
- Duplicate or retried delivery: the same primary key re-appears in a later
  shard with a higher `_seq` and wins, so at-least-once producers are safe.

## Migration

`experiments.evaluation.migrate_archive` backfills a run written before the
archive existed. Legacy runs stored one `samples_<task>_<ts>.parquet` per (sub)task and
referenced Harbor trajectories by a `gs://` URI. The tool reads those files,
writes the rows into the run's `samples` table, and for agentic samples pulls
each referenced trajectory into the `blobs` table (rewriting the sample's
`trajectory_uri` to a `finestore://` reference) and flattens its steps into a
`steps` table. It is idempotent — samples dedupe on their primary key — and
never deletes the source files, so a migration can be validated before the
legacy layout is retired.

```shell
uv run python -m experiments.evaluation.migrate_archive gs://bucket/run/results
```

## Packaging

finestore ships as `marin-finestore` (import `finestore`), a pure-Python root
workspace member built with hatchling. It depends only on `marin-rigging` (for
`rigging.filesystem`, the object-store abstraction) and `pyarrow`. It pulls in
no query engine and no `marin` code, so an import-light consumer such as the
evaldash image can depend on it directly.
