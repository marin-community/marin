# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Global document shuffle of a tokenized Levanter cache.

The hero flat caches under ``hero_tok/v*/train`` (built by the Datakit tokenization pipeline; see
``experiments/datakit/hero_data.py`` for the source/tokenizer/artifact-version pins) were written
cell-by-cell, so their on-disk document order is domain-contiguous: a training run reading them
sequentially sweeps one domain at a time, which the block shuffle in the data loader only partly
hides (window-scale loss bumps).
This job rewrites the cache with a globally-shuffled document order so sequential reads already
interleave domains and the loader's cheap block shuffle suffices.

Each cache entry is exactly one document, so this is a document-level permutation of the jagged
entries -- no re-tokenization. It is a standard two-pass scatter/gather shuffle:

* ``scatter``: each task reads a slice of the input shards and routes every document to one of
  ``--num-buckets`` output buckets by a per-shard seeded draw (deterministic, independent of task
  partitioning), spilling to per-(task, bucket) parquet. No cross-task coordination.
* ``gather``: each task owns a slice of buckets; for each it reads every scatter spill, shuffles
  the concatenated documents, and writes one Levanter cache shard via ``write_bucket_cache``.
* ``merge``: the driver stitches the per-shard ledgers into the top-level ``shard_ledger.json``
  and writes ``.stats.json``.
* ``validate``: load the result, assert document/token counts match the input, and confirm the
  document-length autocorrelation collapsed (domain contiguity broken).

Everything stays in ``marin-us-east-02a``; point ``MARIN_PREFIX`` there. Launched as independent
(non-gang) Iris tasks -- one submission per ``--task-index`` -- so one OOM does not kill the rest::

    for k in $(seq 0 63); do
      uv run iris --config lib/iris/config/marin.yaml job run --enable-extra-resources \\
        --target-cluster cw-us-east-08a --priority production --cpu 4 --memory 64GB --disk 16GB \\
        --max-retries 2 --no-wait --job-name shuf-v16k-scatter-$k \\
        -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
        -- python -m experiments.grug.fast_track.shuffle_cache --phase scatter \\
           --in-cache s3://marin-us-east-02a/marin/datakit/hero_tok/v16384_617680a7/train \\
           --out s3://marin-us-east-02a/marin/datakit/hero_tok/v16384_shuf \\
           --num-buckets 128 --num-tasks 64 --task-index $k
    done
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from levanter.store.cache import CacheLedger, CacheMetadata, TreeCache, _merge_sharded_ledgers
from levanter.store.jagged_array import set_jagged_array_read_cache_bytes
from levanter.store.tree_store import TreeStore
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.datakit.store.bucket_writer import write_bucket_cache

logger = logging.getLogger(__name__)

_EXEMPLAR = {"input_ids": np.zeros(0, dtype=np.int32)}
READ_CACHE_BYTES = 256_000_000  # cap the shared TensorStore read cache (default is 1 GB)
READ_BATCH_DOCS = 2_000  # docs per batched store read
FLUSH_DOCS = 1_500_000  # scatter: flush buffered docs to spills once this many accumulate
_STATS_FILE = ".stats.json"


def _input_shards(in_cache: str) -> list[str]:
    """Relative shard names of the input cache, in ledger order."""
    return list(TreeCache.load(in_cache, _EXEMPLAR).ledger.finished_shards)


def _shard_buckets(shard_name: str, n_docs: int, num_buckets: int, seed: int) -> np.ndarray:
    """Deterministic per-document bucket ids for one shard.

    Seeded by the shard name (not the task), so a document lands in the same bucket on every run
    and regardless of how shards are partitioned across scatter tasks.
    """
    digest = hashlib.blake2b(shard_name.encode(), digest_size=8, key=str(seed).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "big"))
    return rng.integers(0, num_buckets, size=n_docs, dtype=np.int32)


def _bucket_dir(out: str, bucket: int) -> str:
    return prefix_join(out, f"scatter/bucket={bucket:05d}")


def _spill_docs(bucket_dir: str, task_index: int, part: int, docs: list[np.ndarray]) -> None:
    """Write one parquet spill of variable-length int32 documents."""
    lengths = np.fromiter((len(d) for d in docs), dtype=np.int64, count=len(docs))
    values = np.concatenate(docs) if docs else np.zeros(0, dtype=np.int32)
    offsets = np.zeros(len(docs) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    array = pa.LargeListArray.from_arrays(pa.array(offsets, type=pa.int64()), pa.array(values, type=pa.int32()))
    table = pa.table({"input_ids": array})
    path = prefix_join(bucket_dir, f"task-{task_index:04d}-part-{part:04d}.parquet")
    with StoragePath(path).open("wb") as fh:
        pq.write_table(table, fh, compression="zstd")


def _scatter(in_cache: str, out: str, num_buckets: int, seed: int, num_tasks: int, task_index: int) -> None:
    """Route this task's slice of input shards into per-bucket parquet spills."""
    set_jagged_array_read_cache_bytes(READ_CACHE_BYTES)
    shards = _input_shards(in_cache)[task_index::num_tasks]
    logger.info("scatter task %d/%d: %d shards -> %d buckets", task_index, num_tasks, len(shards), num_buckets)

    buffers: list[list[np.ndarray]] = [[] for _ in range(num_buckets)]
    buffered = 0
    parts = [0] * num_buckets

    def flush() -> None:
        nonlocal buffered
        for bucket, docs in enumerate(buffers):
            if docs:
                _spill_docs(_bucket_dir(out, bucket), task_index, parts[bucket], docs)
                parts[bucket] += 1
                docs.clear()
        buffered = 0

    for shard_name in shards:
        store = TreeStore.open(_EXEMPLAR, prefix_join(in_cache, shard_name), mode="r")
        n = len(store)
        assign = _shard_buckets(shard_name, n, num_buckets, seed)
        for base in range(0, n, READ_BATCH_DOCS):
            hi = min(base + READ_BATCH_DOCS, n)
            for offset, item in enumerate(store.get_batch_sync(range(base, hi))):
                bucket = int(assign[base + offset])
                buffers[bucket].append(np.asarray(item["input_ids"], dtype=np.int32))
                buffered += 1
            if buffered >= FLUSH_DOCS:
                flush()
        logger.info("scatter task %d: shard %s (%d docs) done", task_index, shard_name, n)
    flush()
    logger.info("scatter task %d complete", task_index)


def _gather_bucket(out: str, bucket: int, num_buckets: int, seed: int) -> tuple[str, int, int]:
    """Shuffle one bucket's spills into a single cache shard; return (path, rows, tokens)."""
    fs, _ = url_to_fs(out)
    spill_glob = prefix_join(_bucket_dir(out, bucket), "*.parquet")
    spill_paths = sorted(fs.unstrip_protocol(p) for p in fs.glob(spill_glob))
    if not spill_paths:
        raise FileNotFoundError(f"bucket {bucket}: no spills at {spill_glob}")

    docs: list[np.ndarray] = []
    for path in spill_paths:
        with StoragePath(path).open("rb") as fh:
            column = pq.read_table(fh, columns=["input_ids"]).column("input_ids")
        for chunk in column.chunks:
            for value in chunk:
                docs.append(value.values.to_numpy(zero_copy_only=False).astype(np.int32, copy=False))

    rng = np.random.default_rng(np.random.SeedSequence([seed, bucket]))
    perm = rng.permutation(len(docs))
    lengths = [len(docs[i]) for i in perm]
    shard_path = prefix_join(out, f"train/part-{bucket:05d}-of-{num_buckets:05d}")
    ledger = write_bucket_cache(shard_path, (docs[i] for i in perm), lengths)
    tokens = ledger.field_counts.get("input_ids", 0)
    logger.info("gather bucket %d: %d docs, %d tokens -> %s", bucket, ledger.total_num_rows, tokens, shard_path)
    return shard_path, ledger.total_num_rows, tokens


def _gather(out: str, num_buckets: int, seed: int, num_tasks: int, task_index: int) -> None:
    """Materialize this task's slice of buckets into cache shards, one sidecar each."""
    done_dir = prefix_join(out, "_gather_done")
    for bucket in range(task_index, num_buckets, num_tasks):
        sidecar = prefix_join(done_dir, f"bucket-{bucket:05d}.json")
        if StoragePath(sidecar).exists():
            logger.info("gather bucket %d already done; skipping", bucket)
            continue
        shard_path, rows, tokens = _gather_bucket(out, bucket, num_buckets, seed)
        StoragePath(sidecar).write_text(json.dumps({"path": shard_path, "rows": rows, "tokens": tokens}))


def _merge(out: str, num_buckets: int) -> None:
    """Stitch per-shard ledgers into the top-level sharded ledger and write .stats.json."""
    done_dir = prefix_join(out, "_gather_done")
    parts = []
    for bucket in range(num_buckets):
        sidecar = prefix_join(done_dir, f"bucket-{bucket:05d}.json")
        if not StoragePath(sidecar).exists():
            raise FileNotFoundError(f"missing gather sidecar for bucket {bucket}: {sidecar}")
        parts.append(json.loads(StoragePath(sidecar).read_text()))

    metadata = CacheMetadata.empty()
    train = prefix_join(out, "train")
    shard_paths = [p["path"] for p in parts]
    stubs = [
        CacheLedger(total_num_rows=p["rows"], shard_rows={}, finished_shards=[], field_counts={}, metadata=metadata)
        for p in parts
    ]
    field_counts = [{"input_ids": p["tokens"]} for p in parts]
    ledger = _merge_sharded_ledgers(train, shard_paths, stubs, field_counts, metadata)
    stats = {"total_tokens": ledger.field_counts.get("input_ids", 0), "total_elements": ledger.total_num_rows}
    StoragePath(prefix_join(train, _STATS_FILE)).write_text(json.dumps(stats))
    logger.info("merge complete: %d docs, %d tokens -> %s", stats["total_elements"], stats["total_tokens"], train)


def _length_autocorr(train: str, n_docs: int) -> float:
    """Lag-1 autocorrelation of the first ``n_docs`` document lengths.

    Domain-contiguous order keeps neighboring lengths correlated; a global shuffle drives this to ~0.
    """
    store = TreeCache.load(train, _EXEMPLAR)
    n = min(n_docs, len(store))
    lengths = np.array([len(item["input_ids"]) for item in store.get_batch_sync(range(n))], dtype=np.float64)
    a, b = lengths[:-1], lengths[1:]
    return float(np.corrcoef(a, b)[0, 1])


def _validate(in_cache: str, out: str) -> None:
    train = prefix_join(out, "train")
    want = json.loads(StoragePath(prefix_join(in_cache, _STATS_FILE)).read_text())
    got = json.loads(StoragePath(prefix_join(train, _STATS_FILE)).read_text())
    loaded = len(TreeCache.load(train, _EXEMPLAR))
    logger.info("input stats:  %s", want)
    logger.info("output stats: %s (loaded rows=%d)", got, loaded)
    if got["total_elements"] != want["total_elements"] or loaded != want["total_elements"]:
        raise ValueError(
            f"doc count mismatch: input {want['total_elements']} vs output {got['total_elements']}/{loaded}"
        )
    if got["total_tokens"] != want["total_tokens"]:
        raise ValueError(f"token count mismatch: input {want['total_tokens']} vs output {got['total_tokens']}")

    in_ac = _length_autocorr(in_cache, 200_000)
    out_ac = _length_autocorr(train, 200_000)
    logger.info("doc-length lag-1 autocorr: input=%.4f output=%.4f", in_ac, out_ac)
    if out_ac > 0.05:
        raise ValueError(f"output still correlated (autocorr={out_ac:.4f}); shuffle ineffective")
    logger.info("VALIDATE OK: counts match and domain contiguity broken")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True, choices=["scatter", "gather", "merge", "validate"])
    ap.add_argument("--in-cache", required=True, help="input cache train dir (…/train)")
    ap.add_argument("--out", required=True, help="output root; writes scatter/, train/, _gather_done/")
    ap.add_argument("--num-buckets", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-tasks", type=int, default=1)
    ap.add_argument("--task-index", type=int, default=0)
    args = ap.parse_args()

    if args.phase == "scatter":
        _scatter(args.in_cache, args.out, args.num_buckets, args.seed, args.num_tasks, args.task_index)
    elif args.phase == "gather":
        _gather(args.out, args.num_buckets, args.seed, args.num_tasks, args.task_index)
    elif args.phase == "merge":
        _merge(args.out, args.num_buckets)
    else:
        _validate(args.in_cache, args.out)


if __name__ == "__main__":
    main()
