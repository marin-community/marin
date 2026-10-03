# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from levanter.store.cache import TreeCache, consolidate_shard_cache_ledgers

from experiments.datakit.store.bucket_writer import write_bucket_cache
from experiments.datakit.store.datakit_store import BucketCacheStats
from experiments.grug.fast_track.negative_control import build_shuffled_store, shuffle_bucket_tokens


@pytest.mark.parametrize("layout", ["materialized", "sharded"])
def test_token_shuffle_preserves_documents_and_special_positions(tmp_path, monkeypatch, layout):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "scratch"))
    source = str(tmp_path / "source")
    exemplar = {"input_ids": np.zeros(0, dtype=np.int32)}
    documents = [
        np.array([0, *range(10, 25), 1, *range(25, 32), 2], dtype=np.int32),
        np.array([0, *range(32, 53), 2], dtype=np.int32),
    ]
    if layout == "sharded":
        shards = [f"{source}/part-{i}" for i in range(len(documents))]
        for shard, document in zip(shards, documents, strict=True):
            write_bucket_cache(shard, [document], [len(document)])
        consolidate_shard_cache_ledgers(shards, source, exemplar)
    else:
        write_bucket_cache(source, documents, [len(document) for document in documents])
    bucket = BucketCacheStats(
        cluster_id=3,
        quality_bucket=1,
        path=source,
        total_elements=len(documents),
        total_tokens=sum(map(len, documents)),
        n_shards=len(documents) if layout == "sharded" else 1,
    )

    results = []
    for name in ("shuffled", "repeat"):
        result = shuffle_bucket_tokens(bucket, output_path=str(tmp_path / name), seed=7, special_ids=(0, 1, 2))
        cache = TreeCache.load(result.path, exemplar)
        assert len(cache) == len(documents)
        assert cache.flat_field_length("input_ids") == sum(map(len, documents))
        results.append([row["input_ids"] for row in cache.get_batch_sync(range(len(documents)))])

    original = TreeCache.load(source, exemplar).get_batch_sync(range(len(documents)))
    for before, after, repeat, retained in zip(documents, *results, original, strict=True):
        np.testing.assert_array_equal(np.sort(before), np.sort(after))
        special = np.isin(before, (0, 1, 2))
        np.testing.assert_array_equal(before[special], after[special])
        np.testing.assert_array_equal(after, repeat)
        np.testing.assert_array_equal(before, retained["input_ids"])
        assert not np.array_equal(before, after)


def test_shuffled_store_cache_separates_sources_and_permutations(tmp_path):
    source = str(tmp_path / "baseline")
    first = build_shuffled_store(source, 0, version="test-dev")
    repeated = build_shuffled_store(source, 0, version="test-dev")
    changed_seed = build_shuffled_store(source, 1, version="test-dev")
    changed_source = build_shuffled_store(str(tmp_path / "other-baseline"), 0, version="test-dev")

    assert first.path(str(tmp_path)) == repeated.path(str(tmp_path))
    assert len({step.path(str(tmp_path)) for step in (first, changed_seed, changed_source)}) == 3
