# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bulk writers for one materialized Datakit bucket cache."""

import dataclasses
import os
from collections.abc import Iterable, Iterator, Sequence

import numpy as np
from levanter.store.cache import CacheLedger, CacheMetadata
from levanter.store.jagged_array import DEFAULT_WRITE_CHUNK_SIZE, JaggedArrayStore
from rigging.filesystem.storage_path import prefix_join


@dataclasses.dataclass(frozen=True)
class BucketSpillRun:
    """One append-only local run for a bucket's token data and document lengths."""

    data_path: str
    lengths_path: str
    rows: int
    tokens: int


def _aligned_token_chunks(token_chunks: Iterable[np.ndarray], chunk_elements: int) -> Iterator[np.ndarray]:
    """Yield a flat token stream in full write-shard chunks and one final tail."""
    if chunk_elements < 1:
        raise ValueError(f"chunk_elements must be positive, got {chunk_elements}")

    buffer = np.empty(chunk_elements, dtype=np.int32)
    filled = 0
    for chunk in token_chunks:
        tokens = np.asarray(chunk, dtype=np.int32).reshape(-1)
        position = 0
        while position < len(tokens):
            if filled == 0 and len(tokens) - position >= chunk_elements:
                stop = position + chunk_elements
                yield tokens[position:stop]
                position = stop
                continue

            copied = min(chunk_elements - filled, len(tokens) - position)
            buffer[filled : filled + copied] = tokens[position : position + copied]
            filled += copied
            position += copied
            if filled == chunk_elements:
                yield buffer
                filled = 0

    if filled:
        yield buffer[:filled]


def _spill_token_chunks(runs: Sequence[BucketSpillRun]) -> Iterator[np.ndarray]:
    for run in runs:
        expected_bytes = run.tokens * np.dtype(np.int32).itemsize
        if os.path.getsize(run.data_path) != expected_bytes:
            raise ValueError(f"{run.data_path}: expected {expected_bytes} bytes for {run.tokens} tokens")
        yield np.memmap(run.data_path, mode="r", dtype=np.int32, shape=(run.tokens,))


def _write_store(
    cache_dir: str,
    *,
    token_chunks: Iterator[np.ndarray],
    stored_offsets: np.ndarray,
    total_tokens: int,
    max_pending_commits: int,
) -> CacheLedger:
    if max_pending_commits < 1:
        raise ValueError(f"max_pending_commits must be positive, got {max_pending_commits}")

    store = JaggedArrayStore.open(
        prefix_join(cache_dir, "input_ids"),
        mode="w",
        item_rank=1,
        dtype=np.int32,
        cache_metadata=True,
    )
    pending_commits = []

    def submit(write_futures) -> None:
        # Once copy completes, TensorStore no longer borrows the source buffer.
        write_futures.copy.result()
        pending_commits.append(write_futures.commit)
        if len(pending_commits) >= max_pending_commits:
            pending_commits.pop(0).result()

    position = 0
    for chunk in token_chunks:
        stop = position + len(chunk)
        submit(store.data[position:stop].write(chunk))
        position = stop
    assert position == total_tokens

    # offsets[0] is the row count in the JaggedArrayStore layout. Writing the
    # complete array in one operation avoids rewriting its first indexed shard.
    submit(store.offsets[: len(stored_offsets)].write(stored_offsets))
    for commit in pending_commits:
        commit.result()

    metadata = CacheMetadata.empty()
    total_rows = len(stored_offsets) - 1
    ledger = CacheLedger(
        total_num_rows=total_rows,
        is_finished=True,
        shard_rows={cache_dir: total_rows},
        finished_shards=[cache_dir],
        field_counts={"input_ids": total_tokens},
        metadata=metadata,
    )
    ledger._serialize_and_commit(cache_dir)
    return ledger


def write_bucket_cache(
    cache_dir: str,
    token_chunks: Iterable[np.ndarray],
    document_lengths: Sequence[int] | np.ndarray,
    *,
    write_chunk_elements: int = DEFAULT_WRITE_CHUNK_SIZE,
    max_pending_commits: int = 4,
) -> CacheLedger:
    """Write a token stream and its document lengths to an ``input_ids`` cache."""
    lengths = np.asarray(document_lengths, dtype=np.int64)
    if not len(lengths):
        raise ValueError("write_bucket_cache requires at least one document")
    total_tokens = int(lengths.sum())
    stored_offsets = np.empty(len(lengths) + 1, dtype=np.int64)
    stored_offsets[0] = len(lengths)
    np.cumsum(lengths, out=stored_offsets[1:])
    return _write_store(
        cache_dir,
        token_chunks=_aligned_token_chunks(token_chunks, write_chunk_elements),
        stored_offsets=stored_offsets,
        total_tokens=total_tokens,
        max_pending_commits=max_pending_commits,
    )


def write_bucket_cache_from_spills(
    cache_dir: str,
    runs: Sequence[BucketSpillRun],
    *,
    write_chunk_elements: int = DEFAULT_WRITE_CHUNK_SIZE,
    max_pending_commits: int = 4,
) -> CacheLedger:
    """Write local bucket runs to a materialized ``input_ids`` cache."""
    if not runs:
        raise ValueError("write_bucket_cache_from_spills requires at least one run")

    lengths_by_run: list[np.ndarray] = []
    for run in runs:
        lengths = np.fromfile(run.lengths_path, dtype=np.int64)
        if len(lengths) != run.rows:
            raise ValueError(f"{run.lengths_path}: expected {run.rows} lengths, found {len(lengths)}")
        if int(lengths.sum()) != run.tokens:
            raise ValueError(f"{run.lengths_path}: lengths do not sum to {run.tokens} tokens")
        lengths_by_run.append(lengths)
    document_lengths = np.concatenate(lengths_by_run)
    if int(document_lengths.sum()) != sum(run.tokens for run in runs):
        raise ValueError("spill lengths do not match declared token counts")
    return write_bucket_cache(
        cache_dir,
        _spill_token_chunks(runs),
        document_lengths,
        write_chunk_elements=write_chunk_elements,
        max_pending_commits=max_pending_commits,
    )
