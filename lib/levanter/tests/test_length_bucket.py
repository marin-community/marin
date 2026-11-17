# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from levanter.data.text.length_bucket import (
    LengthBucket,
    LengthBucketIndex,
    LengthBucketTokenSeqDataset,
    document_length_stats,
)
from levanter.store.cache import TreeCache, consolidate_shard_cache_ledgers, write_levanter_cache


@dataclass(frozen=True)
class SourceCache:
    cache: TreeCache[dict]
    docs: list[np.ndarray]
    path: Path


@pytest.fixture(params=["consolidated", "sharded"])
def source_cache(tmp_path, request):
    lengths = [4, 9, 5, 20, 0, 4, 11, 5]
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    docs = [np.arange(start, stop, dtype=np.int32) for start, stop in zip(offsets[:-1], offsets[1:], strict=True)]
    path = tmp_path / "cache"
    exemplar = {"input_ids": np.empty(0, dtype=np.int32)}
    if request.param == "consolidated":
        write_levanter_cache([{"input_ids": doc} for doc in docs], str(path), metadata={})
    else:
        shard_paths = []
        for shard, shard_docs in enumerate((docs[:4], docs[4:])):
            shard_path = str(path / f"shard-{shard}")
            write_levanter_cache([{"input_ids": doc} for doc in shard_docs], shard_path, metadata={})
            shard_paths.append(shard_path)
        consolidate_shard_cache_ledgers(shard_paths, str(path), exemplar)
    return SourceCache(TreeCache.load(str(path), exemplar), docs, path)


@pytest.mark.asyncio
async def test_virtual_reads_match_document_filtering_without_cache_writes(source_cache):
    cache, docs, path = source_cache.cache, source_cache.docs, source_cache.path
    before = {file.relative_to(path): file.read_bytes() for file in path.rglob("*") if file.is_file()}
    stats = await document_length_stats(cache, thresholds=(5, 9, 20), chunk_rows=3)
    assert stats.tokens_above == {5: 40, 9: 31, 20: 0}
    index = LengthBucketIndex(cache, 5, statistics=stats, chunk_rows=3)
    for bucket, seq_len, indices in (
        (LengthBucket.SHORT, 6, [2, 0, 2, 1]),
        (LengthBucket.LONG, 10, [3, 0, 2, 1]),
    ):
        expected = np.concatenate([doc for doc in docs if (len(doc) > 5) == (bucket == LengthBucket.LONG)])
        dataset = LengthBucketTokenSeqDataset(index, seq_len, bucket=bucket)
        assert await dataset.async_len() == len(expected) // seq_len
        rows = await dataset.get_batch(indices)
        for row, sequence in zip(rows, indices, strict=True):
            np.testing.assert_array_equal(row["input_ids"], expected[sequence * seq_len : (sequence + 1) * seq_len])
    after = {file.relative_to(path): file.read_bytes() for file in path.rglob("*") if file.is_file()}
    assert after == before


@pytest.mark.asyncio
async def test_stale_chunk_statistics_fail_on_read(source_cache):
    cache = source_cache.cache
    stats = await document_length_stats(cache, thresholds=(5,), chunk_rows=3)
    stale = stats.model_copy(update={"chunk_tokens_above": {5: [10, 19, 11]}})
    index = LengthBucketIndex(cache, 5, statistics=stale, chunk_rows=3)
    dataset = LengthBucketTokenSeqDataset(index, 6, bucket=LengthBucket.SHORT)

    with pytest.raises(ValueError, match="disagree with document offsets"):
        await dataset.get_batch([0])
