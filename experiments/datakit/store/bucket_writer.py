# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bulk writers for one materialized Datakit bucket cache."""

import dataclasses
import os
from collections.abc import Iterator, Sequence

import numpy as np
from levanter.store.cache import CacheLedger, CacheMetadata
from levanter.store.jagged_array import DEFAULT_WRITE_CHUNK_SIZE, JaggedArrayStore


@dataclasses.dataclass(frozen=True)
class BucketSpillRun:
    """One append-only local run for a bucket's token data and document lengths."""

    data_path: str
    lengths_path: str
    rows: int
    tokens: int


def _aligned_token_chunks(documents: Sequence[np.ndarray], chunk_elements: int) -> Iterator[np.ndarray]:
    """Yield a flat token stream in full write-shard chunks and one final tail."""
    if chunk_elements < 1:
        raise ValueError(f"chunk_elements must be positive, got {chunk_elements}")

    buffer = np.empty(chunk_elements, dtype=np.int32)
    filled = 0
    for document in documents:
        tokens = np.asarray(document, dtype=np.int32).reshape(-1)
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


def _aligned_spill_chunks(runs: Sequence[BucketSpillRun], chunk_elements: int) -> Iterator[np.ndarray]:
    """Read local token runs into full write-shard chunks and one final tail."""
    if chunk_elements < 1:
        raise ValueError(f"chunk_elements must be positive, got {chunk_elements}")

    buffer = np.empty(chunk_elements, dtype=np.int32)
    filled = 0
    for run in runs:
        expected_bytes = run.tokens * np.dtype(np.int32).itemsize
        if os.path.getsize(run.data_path) != expected_bytes:
            raise ValueError(f"{run.data_path}: expected {expected_bytes} bytes for {run.tokens} tokens")
        with open(run.data_path, "rb") as stream:
            while True:
                count = stream.readinto(buffer[filled:].data)
                if count == 0:
                    break
                if count % np.dtype(np.int32).itemsize:
                    raise ValueError(f"{run.data_path}: token data is not int32-aligned")
                filled += count // np.dtype(np.int32).itemsize
                if filled == chunk_elements:
                    yield buffer
                    filled = 0
    if filled:
        yield buffer[:filled]


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
        f"{cache_dir.rstrip('/')}/input_ids",
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
    documents: Sequence[np.ndarray],
    *,
    write_chunk_elements: int = DEFAULT_WRITE_CHUNK_SIZE,
    max_pending_commits: int = 4,
) -> CacheLedger:
    """Write one `input_ids` cache without revisiting a TensorStore write shard."""
    if not documents:
        raise ValueError("write_bucket_cache requires at least one document")

    lengths = np.fromiter((np.asarray(document).size for document in documents), dtype=np.int64, count=len(documents))
    total_tokens = int(lengths.sum())
    stored_offsets = np.empty(len(documents) + 1, dtype=np.int64)
    stored_offsets[0] = len(documents)
    np.cumsum(lengths, out=stored_offsets[1:])
    return _write_store(
        cache_dir,
        token_chunks=_aligned_token_chunks(documents, write_chunk_elements),
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
    """Finalize append-only local bucket runs into one aligned TensorStore cache."""
    if not runs:
        raise ValueError("write_bucket_cache_from_spills requires at least one run")

    total_rows = sum(run.rows for run in runs)
    total_tokens = sum(run.tokens for run in runs)
    stored_offsets = np.empty(total_rows + 1, dtype=np.int64)
    stored_offsets[0] = total_rows
    row_position = 1
    token_position = 0
    for run in runs:
        lengths = np.fromfile(run.lengths_path, dtype=np.int64)
        if len(lengths) != run.rows:
            raise ValueError(f"{run.lengths_path}: expected {run.rows} lengths, found {len(lengths)}")
        stop = row_position + run.rows
        np.cumsum(lengths, out=stored_offsets[row_position:stop])
        stored_offsets[row_position:stop] += token_position
        token_position += run.tokens
        row_position = stop
    if token_position != total_tokens or (total_rows and stored_offsets[-1] != total_tokens):
        raise ValueError("spill lengths do not match declared token counts")

    return _write_store(
        cache_dir,
        token_chunks=_aligned_spill_chunks(runs, write_chunk_elements),
        stored_offsets=stored_offsets,
        total_tokens=total_tokens,
        max_pending_commits=max_pending_commits,
    )
