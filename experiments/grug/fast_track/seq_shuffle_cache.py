# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sequence-level shuffle of a flat Levanter token cache.

Training reads a flat cache as one concatenated token stream cut into ``SEQ_LEN`` sequences
(``TokenSeqDataset``). A document longer than one sequence fills several consecutive sequences, and the
loader's block shuffle keeps neighboring sequences in one window, so a long document's pieces all land
in the same ~128 steps at batch 1024. The model adapts to those documents within the window and the
train loss shows a sawtooth at the window period (s7l-d1280L16, 2026-10-20).

This job rewrites the cache so its sequence ``j`` is source sequence ``perm[j]``, where ``perm`` is a
fixed seeded permutation of all ``N = tokens // SEQ_LEN`` source sequences. Each output row is one
sequence, and the source's last partial sequence (fewer than ``SEQ_LEN`` tokens, never trained on) is kept
as the final row so token counts match. Every example the loader builds is unchanged (segment ids come
from EOS tokens inside the sequence); only the order changes, so the cheap block shuffle then gives
near-iid batches.

Two passes, as in ``shuffle_cache.py``:

* ``scatter``: task ``k`` reads its contiguous slice of source sequences sequentially and routes each to
  output part ``perm_position // part_size``, spilling ``(position, tokens)`` parquet per (task, part).
* ``gather``: each task owns a slice of output parts; for each it reads that part's spills, orders the
  sequences by position and writes one cache shard (one row per sequence).
* ``merge``: stitches the shard ledgers (``shuffle_cache._merge``) and writes ``seq_shuffle.json``.
* ``validate``: loads the output with the flat-cache reader, checks sequence and token counts, and
  checks sampled output sequences are identical to their source sequences.

Run on CPU in ``cw-us-east-02a`` next to the data (point ``MARIN_PREFIX`` at ``s3://marin-us-east-02a/marin``)::

    for k in $(seq 0 63); do
      uv run iris --cluster marin job run --enable-extra-resources --target-cluster cw-us-east-02a \\
        --priority interactive --cpu 4 --memory 32GB --disk 16GB --max-retries 2 --no-wait \\
        --job-name seqshuf-scatter-$k -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
        -- python -m experiments.grug.fast_track.seq_shuffle_cache --phase scatter \\
           --in-cache s3://marin-us-east-02a/marin/datakit/hero_tok/v16384_shuf/train \\
           --out s3://marin-us-east-02a/marin/datakit/hero_tok/v16384_seqshuf \\
           --num-parts 128 --num-tasks 64 --task-index $k
    done

then ``gather`` with the same flags, then one ``merge`` and one ``validate`` task. ``--limit-seqs`` restricts
every phase to the first that many source sequences (sample runs).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from levanter.data.text.datasets import TokenSeqDataset
from levanter.store.cache import TreeCache
from levanter.store.jagged_array import set_jagged_array_read_cache_bytes
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.datakit.store.bucket_writer import write_bucket_cache
from experiments.grug.fast_track.shuffle_cache import _merge

logger = logging.getLogger(__name__)

SEQ_LEN = 4096
PERMUTATION_SEED = 20261020
"""Seed of the fixed sequence permutation: ``np.random.default_rng(PERMUTATION_SEED).permutation(N)``."""
READ_BATCH_SEQS = 2048  # source sequences per sequential read (32 MB of int32)
READ_CACHE_BYTES = 256_000_000
FLUSH_SEQS = 131_072  # scatter: spill once this many sequences (2 GB of int32) are buffered
METADATA_FILE = "seq_shuffle.json"
NUM_VALIDATE_SAMPLES = 512

_EXEMPLAR = {"input_ids": np.zeros(0, dtype=np.int32)}


def _source(in_cache: str) -> TokenSeqDataset:
    return TokenSeqDataset(TreeCache.load(in_cache, _EXEMPLAR), SEQ_LEN)


def _num_seqs(source: TokenSeqDataset, limit_seqs: int | None) -> int:
    n = asyncio.run(source.async_len())
    return n if limit_seqs is None else min(n, limit_seqs)


def _permutation(num_seqs: int) -> np.ndarray:
    """``perm[j]`` = source sequence written at output position ``j``."""
    return np.random.default_rng(PERMUTATION_SEED).permutation(num_seqs)


def _part_size(num_seqs: int, num_parts: int) -> int:
    return -(-num_seqs // num_parts)


def _spill_dir(out: str, part: int) -> str:
    return prefix_join(out, f"scatter/part={part:05d}")


def _read_seqs(source: TokenSeqDataset, start: int, stop: int) -> np.ndarray:
    return np.stack([row["input_ids"] for row in asyncio.run(source.get_batch(list(range(start, stop))))])


def _scatter(in_cache: str, out: str, num_parts: int, num_tasks: int, task_index: int, limit_seqs: int | None) -> None:
    """Route this task's contiguous slice of source sequences into per-part spills."""
    set_jagged_array_read_cache_bytes(READ_CACHE_BYTES)
    source = _source(in_cache)
    num_seqs = _num_seqs(source, limit_seqs)
    position = np.empty(num_seqs, dtype=np.int64)
    position[_permutation(num_seqs)] = np.arange(num_seqs, dtype=np.int64)
    part_size = _part_size(num_seqs, num_parts)
    lo = task_index * num_seqs // num_tasks
    hi = (task_index + 1) * num_seqs // num_tasks
    logger.info("scatter task %d/%d: source sequences [%d, %d) of %d", task_index, num_tasks, lo, hi, num_seqs)

    buffers: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(num_parts)]
    buffered = 0
    flushes = 0

    def flush() -> None:
        nonlocal buffered, flushes
        for p, chunks in enumerate(buffers):
            if chunks:
                _spill(out, p, task_index, flushes, chunks)
                chunks.clear()
        buffered = 0
        flushes += 1

    for start in range(lo, hi, READ_BATCH_SEQS):
        stop = min(start + READ_BATCH_SEQS, hi)
        tokens = _read_seqs(source, start, stop)
        pos = position[start:stop]
        part = pos // part_size
        for p in np.unique(part):
            sel = part == p
            buffers[int(p)].append((pos[sel], tokens[sel]))
        buffered += stop - start
        if buffered >= FLUSH_SEQS:
            flush()
        if (start - lo) // READ_BATCH_SEQS % 50 == 0:
            logger.info("scatter task %d: read %d / %d sequences", task_index, stop - lo, hi - lo)
    flush()
    logger.info("scatter task %d complete", task_index)


def _spill(out: str, part: int, task_index: int, flush_index: int, chunks: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Write one parquet spill of ``(position, tokens)`` rows for output ``part``."""
    pos = np.concatenate([c[0] for c in chunks])
    tokens = np.concatenate([c[1] for c in chunks]).reshape(-1)
    table = pa.table(
        {
            "position": pa.array(pos, type=pa.int64()),
            "input_ids": pa.FixedSizeListArray.from_arrays(pa.array(tokens, type=pa.int32()), SEQ_LEN),
        }
    )
    path = prefix_join(_spill_dir(out, part), f"task-{task_index:04d}-{flush_index:04d}.parquet")
    with StoragePath(path).open("wb") as fh:
        pq.write_table(table, fh, compression="zstd")


def _source_tail(source: TokenSeqDataset, limit_seqs: int | None) -> np.ndarray:
    """The tokens after the last full sequence (empty for a ``--limit-seqs`` sample)."""
    if limit_seqs is not None:
        return np.zeros(0, dtype=np.int32)
    total = asyncio.run(source.doc_cache.async_flat_field_length("input_ids"))
    full = (total // SEQ_LEN) * SEQ_LEN
    if total == full:
        return np.zeros(0, dtype=np.int32)
    return np.asarray(asyncio.run(source.doc_cache.get_flat_field_batch("input_ids", [full], total - full))[0])


def _gather_part(in_cache: str, out: str, part: int, num_parts: int, limit_seqs: int | None) -> tuple[str, int, int]:
    """Order one part's spilled sequences by output position and write its cache shard."""
    source = _source(in_cache)
    num_seqs = _num_seqs(source, limit_seqs)
    part_size = _part_size(num_seqs, num_parts)
    want = range(part * part_size, min((part + 1) * part_size, num_seqs))

    fs, _ = url_to_fs(out)
    spill_glob = prefix_join(_spill_dir(out, part), "*.parquet")
    spill_paths = sorted(fs.unstrip_protocol(p) for p in fs.glob(spill_glob))
    if not spill_paths:
        raise FileNotFoundError(f"part {part}: no spills at {spill_glob}")
    positions, tokens = [], []
    for path in spill_paths:
        with StoragePath(path).open("rb") as fh:
            table = pq.read_table(fh)
        positions.append(table.column("position").to_numpy())
        flat = table.column("input_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False)
        tokens.append(flat.astype(np.int32, copy=False).reshape(-1, SEQ_LEN))
    pos = np.concatenate(positions)
    order = np.argsort(pos, kind="stable")
    if not np.array_equal(pos[order], np.arange(want.start, want.stop)):
        raise ValueError(f"part {part}: spilled positions do not cover [{want.start}, {want.stop}) exactly once")
    seqs = np.concatenate(tokens)[order]
    rows = [seqs[i] for i in range(len(seqs))]
    if part == num_parts - 1:
        tail = _source_tail(source, limit_seqs)
        if len(tail):
            rows.append(tail.astype(np.int32))
    lengths = [len(r) for r in rows]
    shard_path = prefix_join(out, f"train/part-{part:05d}-of-{num_parts:05d}")
    ledger = write_bucket_cache(shard_path, iter(rows), lengths)
    n_tokens = ledger.field_counts.get("input_ids", 0)
    logger.info("gather part %d: %d rows, %d tokens -> %s", part, ledger.total_num_rows, n_tokens, shard_path)
    return shard_path, ledger.total_num_rows, n_tokens


def _gather(in_cache: str, out: str, num_parts: int, num_tasks: int, task_index: int, limit_seqs: int | None) -> None:
    set_jagged_array_read_cache_bytes(READ_CACHE_BYTES)
    done_dir = prefix_join(out, "_gather_done")
    for part in range(task_index, num_parts, num_tasks):
        sidecar = prefix_join(done_dir, f"bucket-{part:05d}.json")
        if StoragePath(sidecar).exists():
            logger.info("gather part %d already done; skipping", part)
            continue
        shard_path, rows, n_tokens = _gather_part(in_cache, out, part, num_parts, limit_seqs)
        StoragePath(sidecar).write_text(json.dumps({"path": shard_path, "rows": rows, "tokens": n_tokens}))


def _write_metadata(in_cache: str, out: str, num_parts: int, limit_seqs: int | None) -> None:
    source = _source(in_cache)
    num_seqs = _num_seqs(source, limit_seqs)
    train = prefix_join(out, "train")
    stats = json.loads(StoragePath(prefix_join(train, ".stats.json")).read_text())
    meta = {
        "source": in_cache,
        "seq_len": SEQ_LEN,
        "permutation": f"numpy.random.default_rng({PERMUTATION_SEED}).permutation({num_seqs})",
        "permutation_seed": PERMUTATION_SEED,
        "num_sequences": num_seqs,
        "num_parts": num_parts,
        "part_size": _part_size(num_seqs, num_parts),
        "tail_tokens": stats["total_tokens"] - num_seqs * SEQ_LEN,
        "total_tokens": stats["total_tokens"],
        "total_rows": stats["total_elements"],
        "limit_seqs": limit_seqs,
    }
    StoragePath(prefix_join(train, METADATA_FILE)).write_text(json.dumps(meta, indent=2))
    logger.info("metadata: %s", meta)


def _validate(in_cache: str, out: str, limit_seqs: int | None) -> None:
    set_jagged_array_read_cache_bytes(READ_CACHE_BYTES)
    source = _source(in_cache)
    num_seqs = _num_seqs(source, limit_seqs)
    output = _source(prefix_join(out, "train"))
    got_seqs = asyncio.run(output.async_len())
    if got_seqs != num_seqs:
        raise ValueError(f"sequence count mismatch: source {num_seqs} vs output {got_seqs}")
    src_tokens = (
        asyncio.run(source.doc_cache.async_flat_field_length("input_ids")) if limit_seqs is None else num_seqs * SEQ_LEN
    )
    out_tokens = asyncio.run(output.doc_cache.async_flat_field_length("input_ids"))
    if out_tokens != src_tokens:
        raise ValueError(f"token count mismatch: source {src_tokens} vs output {out_tokens}")
    perm = _permutation(num_seqs)
    rng = np.random.default_rng(0)
    sample = np.concatenate([[0, num_seqs - 1], rng.choice(num_seqs, NUM_VALIDATE_SAMPLES, replace=False)])
    got = asyncio.run(output.get_batch([int(j) for j in sample]))
    want = asyncio.run(source.get_batch([int(perm[j]) for j in sample]))
    for j, g, w in zip(sample, got, want, strict=True):
        if not np.array_equal(g["input_ids"], w["input_ids"]):
            raise ValueError(f"output sequence {j} differs from source sequence {perm[j]}")
    if limit_seqs is None:
        tail = np.asarray(
            asyncio.run(
                output.doc_cache.get_flat_field_batch("input_ids", [num_seqs * SEQ_LEN], out_tokens - num_seqs * SEQ_LEN)
            )[0]
        )
        if not np.array_equal(tail, _source_tail(source, None)):
            raise ValueError("output tail differs from source tail")
    logger.info(
        "VALIDATE OK: %d sequences, %d tokens, %d sampled sequences identical to source",
        num_seqs,
        out_tokens,
        len(sample),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=["scatter", "gather", "merge", "validate"])
    ap.add_argument("--in-cache", required=True, help="source flat cache (…/train)")
    ap.add_argument("--out", required=True, help="output root; writes scatter/, train/, _gather_done/")
    ap.add_argument("--num-parts", type=int, default=128)
    ap.add_argument("--num-tasks", type=int, default=1)
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--limit-seqs", type=int, default=None, help="only the first N source sequences (sample runs)")
    args = ap.parse_args()

    if args.phase == "scatter":
        _scatter(args.in_cache, args.out, args.num_parts, args.num_tasks, args.task_index, args.limit_seqs)
    elif args.phase == "gather":
        _gather(args.in_cache, args.out, args.num_parts, args.num_tasks, args.task_index, args.limit_seqs)
    elif args.phase == "merge":
        _merge(args.out, args.num_parts)
        _write_metadata(args.in_cache, args.out, args.num_parts, args.limit_seqs)
    else:
        _validate(args.in_cache, args.out, args.limit_seqs)


if __name__ == "__main__":
    main()
