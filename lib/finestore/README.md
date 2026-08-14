# FineStore

FineStore is a small transactional store for immutable Parquet data on local disk,
GCS, and S3-compatible object storage. It covers workloads that need to batch many
small records or named byte objects without running a catalog service.

One conditional mutable object, `HEAD`, is the visibility boundary. It points to an
immutable manifest containing the complete active file set for every table. A
successful write returns the version of `HEAD` as a `CommitToken`. Readers resolve
`HEAD` once and keep that manifest for the lifetime of their `ReadView`.

## Transactions

Register table contracts on the store, then use a transaction when several tables
and objects must become visible together:

```python
from finestore.store import DataStore

with DataStore.open("gs://bucket/purchases", writer_id="worker-0") as store:
    store.table("purchases", primary_key=("order_id",))

    with store.transaction() as transaction:
        receipt_uri = transaction.write_object("receipts/o-1", b"paid")
        transaction.table("purchases").add(
            {"order_id": "o-1", "amount": 12, "receipt_uri": receipt_uri}
        )
        assert transaction.lookup("receipts/o-1") == b"paid"

    token = transaction.token
```

The transaction buffers its rows and object bytes in memory. Each participating
table produces one level-zero Parquet shard. FineStore writes every shard and an
immutable full manifest before it conditionally advances `HEAD`. Until that final
write succeeds, none of the transaction is visible. A caller should split a
transaction when its payload would exceed a safe process-memory bound.

Ordinary table appends use the same commit path. `store.flush()` atomically publishes
the currently claimed buffers from all registered tables. `table.flush()` publishes
only that table. If writing or publication fails, claimed store buffers are restored
for retry. Immutable files written before a failed publication are harmless orphans
and can be collected later.

## Reads

```python
from finestore.reader import ReadView

view = ReadView("gs://bucket/purchases")
purchase = view.point("purchases", order_id="o-1")
receipt = view.resolve(purchase["receipt_uri"])
token = view.token
```

A view pins one manifest. Commits and compactions that happen afterward do not change
its results. Create a new view to observe a newer token.

`scan` supports column projection and `==`, `!=`, and `in` filters. FineStore unifies
compatible Parquet schemas across active shards and keeps the row from the latest
user commit for each primary key. Compaction preserves each row's original commit
sequence, so reorganizing files never makes an older value outrank a concurrent
write.

## Named objects and caches

`DataStore.write_object(name, data, metadata)` writes bytes into the reserved `blobs`
table and returns `finestore://blobs/<name>`. Named objects participate in the same
transaction and commit token as ordinary table rows.

Two adapters build cache behavior on this primitive:

- `finestore.cache.PersistentKvCache` stores serialized autotuning and compiled-kernel
  results by key. The process memory tier serves repeated reads. Remote writes queue in
  the background, and each burst shares a bounded multi-object transaction. Normal
  interpreter shutdown gives queued writes up to 10 seconds to finish. A stalled write
  is then abandoned so cache storage cannot hold process shutdown. `cache.close()` waits
  for queued writes when a caller wants deterministic resource cleanup.
- `finestore.fileset.FineStoreDirectory` materializes a committed named-object set
  into a local directory and publishes newly created files in bounded transactions.
  Iris uses it for XLA's per-fusion autotune directory, replacing the ZIP mirror.

Both adapters are caches: failures may be treated as misses by their callers. Cache
cleanup does not acknowledge persistence or return a commit token. Use
`DataStore.transaction()` when a write must become visible before the caller continues.
The core `DataStore` and `ReadView` APIs propagate storage and commit errors.

## Compaction

The store's maintenance thread flushes on its configured interval and compacts a
table after it crosses `compaction_shards` active files. `store.maintain()` exposes
the same operation for deterministic callers. `compact(root, table)` performs one
explicit compaction.

Compaction pins a manifest, streams an ordered merge of its exact input shards, writes
one next-generation shard, and replaces those paths through the same `HEAD` compare
and swap. If another compactor removes an input first, the losing output remains
unreferenced and the operation is a benign no-op. If another writer adds a shard,
the replacement can rebase while preserving the new shard.

Compaction does not delete source objects. An older read view may still reference
them. Physical garbage collection needs a separate retention rule, such as sealed
archive retention or reader leases; it is outside the current API.

## Layout

```text
{root}/
    _archive.json                         format marker changed only by migrations
    HEAD                                  conditional commit pointer
    manifests/{commit_id}.json            immutable complete snapshots
    schemas/{content_hash}.json            immutable table contracts
    data/{table}/w={writer}/g={level}/... immutable Parquet shards
```

Only `HEAD` is mutable. The `HEAD` document names a manifest, commit id, and logical
sequence. A manifest names every active shard, its writer and compaction generation,
row and sequence bounds, each table's metadata object, and optional seal state.

Writable archives require real conditional object writes:

- local disk: `flock` plus a content version;
- GCS: object generations and `if_generation_match`;
- S3: ETags with `If-None-Match` or `If-Match`.

Other fsspec schemes are rejected for writes because last-writer-wins publication
cannot provide transaction atomicity. They may be added after their backend exposes
an equivalent compare-and-swap primitive.

## Recovery and lifecycle

- A process that stops before advancing `HEAD` leaves the previous commit intact.
- A process that stops after advancing `HEAD` completed the transaction.
- A cache writer may stop at either point. The named value is absent or fully committed;
  a partial transaction is never visible.
- A compare-and-swap loss reloads `HEAD` and rebases disjoint additions. A compaction
  rebases only while all of its exact inputs remain active.
- Opening a sealed archive for writing publishes a commit that clears its seal.
  `store.seal()` flushes, compacts, and records seal metadata in a final manifest.

Format migrations live in `finestore.migrations` as a linear revision chain. Each
`mNNNN_<name>.py` revision declares its source and target format versions and must be
safe to retry. `_archive.json` is the applied-version ledger, so a separate migration
table would duplicate the archive's visibility state.

Each revision also declares whether it is safe to run while opening an archive for
writing. `DataStore.open()` applies a complete chain only when every required revision
opts in. `ReadView()` never mutates storage; an older archive raises
`FormatVersionError` with instructions to call `finestore.migrations.migrate(root)`.
An archive from a newer FineStore build instructs either caller to upgrade the package.

The application that owns an archive runs migrations after quiescing its writers. For
evaluation archives, the fleet operator runs
`experiments.evaluation.migrations.cli upgrade-format` before deploying format-v2
readers. The v1-to-v2 revision is write-open-safe when the archive is sealed. A writable
open then publishes existing Parquet shards through an initial manifest and keeps the
legacy objects without copying payload data. An unsealed archive still requires the
owner to quiesce and seal its writers first.

## Package boundary

`marin-finestore` contains storage, transactions, read views, compaction, and generic
cache adapters. Evaluation record types and `EvaluationStore` live in
`marin.evaluation.archive`. FineStore has no evaluation or Marin pipeline dependency.
