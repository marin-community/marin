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
`primary_key`; the reader keeps one row per key (the latest write wins).

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

An archive owns its root directory. Each table is a subdirectory of immutable
shards, partitioned by the writer that produced them and the compaction
generation they belong to:

```
{root}/
    _archive.json                            # archive-wide: the on-disk format version
    SEALED                                   # optional: the run is complete
    {table}/_schema.json                     # per-table: primary key + logical schema version
    {table}/w={writer}/g={gen}/{seq:016d}-{uid}.parquet
```

Shard membership is discovered by listing the table directory. A shard's schema
and row-group statistics come from its Parquet footer. The small JSON objects
record only what a footer cannot: `_archive.json` holds finestore's on-disk
format version (the writer stamps it at open, and a reader refuses a newer
format than it understands), and each `{table}/_schema.json` holds the dedup
primary key and the caller's logical schema version. Together they are the whole
"manifest". The writer and generation are encoded in the object key and
recovered by listing; nothing else references a shard. A caller that shares the
root with sibling data (an eval run's results directory also holds JSON and
legacy parquet) passes a dedicated subdirectory as the root; finestore does not
impose one.

## Read semantics

`scan` lists a table's shards, unifies their footers' schemas (a column a later
writer added is promoted to null for older shards), reads only the projected
columns plus whatever the primary key needs, and deduplicates by primary key. For
each key it keeps the row with the highest `_seq`, breaking ties by the highest
compaction generation. Because a writer resumes its `_seq` above every persisted
row, a later write always outranks an earlier one — nothing can shadow it — and
the generation only breaks the exact-`_seq` tie a compaction leaves when it
re-emits a row unchanged. That single rule makes a duplicate delivery, a retried
flush, and a crash mid-compaction all converge to one row without coordination.

`point` returns the single row matching `key=value` for every key. Predicate
filters support `==`, `!=`, and `in`, pushed into the Parquet scan, where the
row-group footer statistics prune to the groups that can match — a `point`
lookup on a compacted (primary-key-sorted) shard reads one row group, not the
whole table.

Because schema unification is null-permissive, a reader that opened the archive
without opening the writer — for example the dashboard — reads a run written by
a newer or older schema without changes: absent columns read back as null and
the row model's defaults fill them in.

## Writing

Appends are non-blocking and buffer in memory. The background thread flushes
each table on a time ceiling (`DEFAULT_FLUSH_INTERVAL`, 5s) so shards stay
fresh, and immediately once a buffer's estimated payload crosses a byte cap
(`DEFAULT_MAX_BUFFER_BYTES`, 100 MiB) so memory stays bounded whether a table
holds many small samples or a few large blobs. `table.flush()` persists one
table; `store.flush()` persists all of them; `flush` and `close` block until
every buffered row is a durable object — the "writes block until persisted"
guarantee. A flush splits its rows into row groups of at most `ROW_GROUP_ROWS`
(16384) so a later filtered read prunes by row group, writes to a temporary key,
and atomically renames it into place, so a partial object is never visible.

Concurrent writers of one run (for example RL rollout workers) each open a
`DataStore` with a distinct `writer_id`. They write under their own key prefix
and never coordinate; the reader composes and deduplicates them. Call `seal`
once, after every writer has finished, to drop the `SEALED` marker.

## Compaction

`compact(root, table)` streams a k-way merge over a table's shards in primary-key
order and writes the surviving rows to one next-generation shard, one row group
at a time, then deletes the shards it superseded. The merge is bounded in memory:
a compacted shard (already in primary-key order) streams a row group at a time,
while a level-0 shard — one flush, bounded by the writer's row cap — is sorted in
memory, so an archive far larger than memory still compacts. Both the store flush
and this merge write through the same `ShardWriter`. Compaction is optional and
safe to run at any time — end of run, or in the background — because the dedup
rule already yields correct reads across mixed generations. It reduces object
count and improves read locality; it does not change query results.

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

## Primary key handling

Every table declares a primary key: the columns that identify a row. It is
required — a table whose rows have no natural identity declares a nonce column
rather than no key — because dedup by key is what stands in for a manifest, and a
table without one silently doubles its rows on any at-least-once delivery.

The reader keeps one row per key, so two different rows under one key would
discard one with nothing recording that it existed.
`store.table(..., on_conflict=...)` picks what happens instead:

- `OnConflict.ERROR` (the default) raises `PrimaryKeyConflict`, naming the table
  and the key. A repeat with an identical payload is an at-least-once redelivery
  and collapses. Detection spans the whole writer session, so an earlier row
  already flushed to a shard does not hide the conflict.
- `OnConflict.SUPERSEDE` is the upsert contract, for a caller that means to
  replace a row: blob rewrites and migration backfills. Compaction counts the
  replaced rows into the `SEALED` marker, per table.

Widening a primary key is a schema change, and an accumulative store cannot
absorb one: the added column reads as null on the old shards, so those rows never
collapse against the new ones and survive beside them. A writer therefore refuses
an archive whose `schema_version` predates its own. Bringing it forward means
deleting, which no code in this library does — see
`experiments.evaluation.migrations`.

## Preserved sources

`export_lm_eval_samples` writes every `samples_*.jsonl` and `results_*.json` it
reads into the archive's `blobs` table under a `sources/` prefix, alongside the
rows normalized from them. Shards are zstd-compressed, so a text artifact costs
roughly a third of its raw size. The archive is then self-describing:
`rebuild_lm_eval_samples(root)` re-derives the `samples` table from those blobs
alone, repairing a run whose results tree was pruned or whose export died half
written. It reproduces existing rows exactly, so they collapse against themselves
and nothing is deleted.

evalchemy runs the harness in a `tempfile` working directory and copies the tree
into the results path, so a retried evaluation leaves a second complete tree
under a `tmp<random>/` segment. Both are real evaluations and their
loglikelihoods differ, because inference is not deterministic, but only the
canonical tree produced the metrics on the run's record. `is_scratch_artifact`
identifies the retry: it is preserved as a source blob like any other, and
contributes no rows, so the samples a reader sees come from the same evaluation
as the headline score. Indexing both would instead put two different rows on one
primary key, which is a `PrimaryKeyConflict`, not a silent fold.

Writing to an archive clears its `SEALED` marker. "Sealed" therefore means
"these are the finished contents" rather than "some session once finished", so
an export that dies partway leaves the archive visibly incomplete instead of
carrying a marker that vouches for a table it no longer describes.

## Migration

`experiments.evaluation.migrate_archive` backfills a run written before the
archive existed. Legacy runs stored one `samples_<task>_<ts>.parquet` per (sub)task and
referenced Harbor trajectories by a `gs://` URI. The tool reads those files,
writes the rows into the run's `samples` table, and for agentic samples pulls
each referenced trajectory into the `blobs` table (rewriting the sample's
`trajectory_uri` to a `finestore://` reference) and flattens its steps into a
`steps` table. It is idempotent because samples dedupe on their primary key.
`--archive-legacy` moves validated source Parquets to deterministic region-local
30-day storage. `--from-legacy-archive` reads those preserved Parquets to rebuild
the archive at the original results path without moving the backup again.

`experiments.evaluation.migrations` holds the migrations that remove rows. They
are the only code that deletes, and each snapshots the whole table directory to
region-local 30-day storage and verifies every object's size before dropping
anything, so a failure leaves either the original or a complete copy. An object
already at the destination is left alone: the earliest snapshot is the pristine
one, and a resumed migration must not write a degraded table over it.

## Packaging

finestore ships as `marin-finestore` (import `finestore`), a pure-Python root
workspace member built with hatchling. It depends only on `marin-rigging` (for
`rigging.filesystem`, the object-store abstraction), `pyarrow`, and `pydantic`
(the typed table metadata). It pulls in no query engine and no `marin` code, so
an import-light consumer such as the evaldash image can depend on it directly.
