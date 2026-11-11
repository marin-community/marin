# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Read document-length buckets as virtual token streams from one cache."""

import asyncio
import threading
from collections import OrderedDict
from collections.abc import Sequence
from enum import StrEnum
from functools import lru_cache

import numpy as np
from pydantic import BaseModel
from rigging.filesystem.storage_path import StoragePath, prefix_join

from levanter.data.dataset import AsyncDataset
from levanter.data.text.formats import TokenSeqDict
from levanter.store.cache import TreeCache, _await_shared_open

DOCUMENT_CHUNK_ROWS = 65_536
LENGTH_STATS_FILE = "length_stats.json"
_CACHED_OFFSET_CHUNKS = 4


class LengthBucket(StrEnum):
    SHORT = "short"
    LONG = "long"


class DocumentLengthStats(BaseModel):
    """Token totals for one cache, with chunk summaries for virtual reads."""

    num_documents: int
    total_tokens: int
    tokens_above: dict[int, int]
    chunk_total_tokens: list[int]
    chunk_tokens_above: dict[int, list[int]]


class BucketLengthStats(BaseModel):
    """Document-length statistics for named caches."""

    chunk_rows: int
    buckets: dict[str, DocumentLengthStats]


@lru_cache(maxsize=8)
def load_bucket_length_stats(path: str) -> BucketLengthStats:
    """Load an immutable statistics artifact once per process."""
    return BucketLengthStats.model_validate_json(StoragePath(prefix_join(path, LENGTH_STATS_FILE)).read_text())


async def document_length_stats(
    cache: TreeCache[dict], *, thresholds: Sequence[int], chunk_rows: int
) -> DocumentLengthStats:
    """Count tokens in documents strictly above each threshold, reading only offsets."""
    chunk_totals: list[int] = []
    chunk_above: dict[int, list[int]] = {threshold: [] for threshold in thresholds}
    row_count = cache.flat_field_num_rows("input_ids")
    for start in range(0, row_count, chunk_rows):
        offsets = await cache.flat_field_document_offsets("input_ids", start, min(start + chunk_rows, row_count))
        lengths = np.diff(offsets)
        chunk_totals.append(int(lengths.sum()))
        for threshold in thresholds:
            chunk_above[threshold].append(int(lengths[lengths > threshold].sum()))
    total_tokens = sum(chunk_totals)
    if total_tokens != cache.flat_field_length("input_ids"):
        raise ValueError(f"Document offsets disagree with the cache ledger at {cache.cache_dir}")
    return DocumentLengthStats(
        num_documents=row_count,
        total_tokens=total_tokens,
        tokens_above={threshold: sum(counts) for threshold, counts in chunk_above.items()},
        chunk_total_tokens=chunk_totals,
        chunk_tokens_above=chunk_above,
    )


class LengthBucketIndex:
    """Compact in-memory token totals for chunks of documents in a cache."""

    def __init__(
        self,
        cache: TreeCache[dict],
        threshold: int,
        *,
        statistics: DocumentLengthStats,
        chunk_rows: int,
        cached_chunks: int = _CACHED_OFFSET_CHUNKS,
    ):
        if cached_chunks <= 0:
            raise ValueError("cached_chunks must be positive")
        self.cache = cache
        self.threshold = threshold
        self.chunk_rows = chunk_rows
        self.cached_chunks = cached_chunks
        row_count = cache.flat_field_num_rows("input_ids")
        if statistics.num_documents != row_count:
            raise ValueError(f"Length statistics have the wrong document count for {cache.cache_dir}")
        if len(statistics.chunk_total_tokens) != (row_count + chunk_rows - 1) // chunk_rows:
            raise ValueError(f"Length statistics have the wrong number of chunks for {cache.cache_dir}")
        chunk_totals = np.asarray(statistics.chunk_total_tokens, dtype=np.int64)
        chunk_long = np.asarray(statistics.chunk_tokens_above[threshold], dtype=np.int64)
        if chunk_long.shape != chunk_totals.shape or np.any(chunk_long < 0) or np.any(chunk_long > chunk_totals):
            raise ValueError(f"Invalid length statistics for {cache.cache_dir}")
        short = np.concatenate(([0], np.cumsum(chunk_totals - chunk_long, dtype=np.int64)))
        long = np.concatenate(([0], np.cumsum(chunk_long, dtype=np.int64)))
        if short[-1] + long[-1] != cache.flat_field_length("input_ids"):
            raise ValueError(f"Length statistics disagree with the cache ledger at {cache.cache_dir}")
        expected_counts = (
            statistics.total_tokens - statistics.tokens_above[threshold],
            statistics.tokens_above[threshold],
        )
        if (int(short[-1]), int(long[-1])) != expected_counts:
            raise ValueError(f"Length statistics disagree within {cache.cache_dir}")
        self._prefixes = {LengthBucket.SHORT: short, LengthBucket.LONG: long}
        self._lock = threading.Lock()
        self._offset_chunks: OrderedDict[int, np.ndarray] = OrderedDict()
        self._pending_chunks: dict[tuple[asyncio.AbstractEventLoop, int], asyncio.Task[np.ndarray]] = {}

    def token_count(self, bucket: LengthBucket) -> int:
        return int(self._prefixes[bucket][-1])

    async def _chunk_offsets(self, chunk: int) -> np.ndarray:
        offsets = await _await_shared_open(
            self._offset_chunks, self._pending_chunks, self._lock, chunk, lambda: self._read_chunk_offsets(chunk)
        )
        with self._lock:
            self._offset_chunks[chunk] = offsets
            self._offset_chunks.move_to_end(chunk)
            while len(self._offset_chunks) > self.cached_chunks:
                self._offset_chunks.popitem(last=False)
        return offsets

    async def _read_chunk_offsets(self, chunk: int) -> np.ndarray:
        start = chunk * self.chunk_rows
        stop = min(start + self.chunk_rows, self.cache.flat_field_num_rows("input_ids"))
        offsets = await self.cache.flat_field_document_offsets("input_ids", start, stop)
        lengths = np.diff(offsets)
        long_tokens = int(lengths[lengths > self.threshold].sum())
        for bucket, count in (
            (LengthBucket.SHORT, int(lengths.sum()) - long_tokens),
            (LengthBucket.LONG, long_tokens),
        ):
            prefixes = self._prefixes[bucket]
            if count != int(prefixes[chunk + 1] - prefixes[chunk]):
                raise ValueError(
                    f"Length statistics disagree with document offsets at {self.cache.cache_dir}, chunk {chunk}"
                )
        return offsets

    async def source_spans(self, *, bucket: LengthBucket, start: int, length: int) -> list[tuple[int, int]]:
        prefixes = self._prefixes[bucket]
        if start < 0 or length < 0 or start + length > int(prefixes[-1]):
            raise ValueError("Requested tokens beyond the end of the length bucket")
        spans: list[tuple[int, int]] = []
        cursor = start
        while cursor < start + length:
            chunk = int(np.searchsorted(prefixes, cursor, side="right") - 1)
            offsets = await self._chunk_offsets(chunk)
            lengths = np.diff(offsets)
            selected = (lengths > self.threshold) if bucket == LengthBucket.LONG else (lengths <= self.threshold)
            selected_rows = np.flatnonzero(selected)
            selected_lengths = lengths[selected_rows]
            local_prefixes = np.concatenate(([0], np.cumsum(selected_lengths, dtype=np.int64)))
            local_cursor = cursor - int(prefixes[chunk])
            row = int(np.searchsorted(local_prefixes, local_cursor, side="right") - 1)
            while row < len(selected_rows) and cursor < start + length:
                within = local_cursor - int(local_prefixes[row])
                available = int(selected_lengths[row]) - within
                take = min(available, start + length - cursor)
                source = int(offsets[selected_rows[row]]) + within
                if spans and spans[-1][0] + spans[-1][1] == source:
                    prior_source, prior_length = spans[-1]
                    spans[-1] = prior_source, prior_length + take
                else:
                    spans.append((source, take))
                cursor += take
                local_cursor += take
                row += 1
        return spans


class LengthBucketTokenSeqDataset(AsyncDataset[TokenSeqDict]):
    """Fixed-length sequences from a virtual short or long document stream."""

    def __init__(self, index: LengthBucketIndex, seq_len: int, *, bucket: LengthBucket):
        self.index = index
        self.seq_len = seq_len
        self.bucket = bucket

    async def async_len(self) -> int:
        return self.index.token_count(self.bucket) // self.seq_len

    def is_finite(self) -> bool:
        return True

    async def get_batch(self, indices: Sequence[int]) -> Sequence[TokenSeqDict]:
        if not indices:
            return []
        dataset_len = await self.async_len()
        if min(indices) < 0 or max(indices) >= dataset_len:
            raise ValueError("Requested indices beyond the end of the dataset")

        spans_by_sequence = await asyncio.gather(
            *(
                self.index.source_spans(bucket=self.bucket, start=index * self.seq_len, length=self.seq_len)
                for index in indices
            )
        )

        async def read(spans: list[tuple[int, int]]) -> TokenSeqDict:
            pieces = await asyncio.gather(
                *(self.index.cache.get_flat_field_batch("input_ids", [start], length) for start, length in spans)
            )
            return {"input_ids": np.concatenate([piece[0] for piece in pieces])}

        return await asyncio.gather(*(read(spans) for spans in spans_by_sequence))
