import os
import tempfile

import numpy as np
import pytest
import ray
from levanter.data import BatchProcessor, ShardedDataSource
from levanter.store.cache import CacheLedger, CacheOptions
from levanter.store.tree_store import TreeStore

from marin.processing.independent_tokenize import (
    _tokenize_one_shard,
    concatenate_stores,
    tokenize_all_shards,
    tokenize_and_concatenate_shards,
)


class SimpleProcessor(BatchProcessor):
    def __call__(self, batch):
        return [{"data": x} for x in batch]

    @property
    def num_cpus(self):
        return 1

    @property
    def num_gpus(self):
        return 0

    @property
    def resources(self):
        return {}

    @property
    def output_exemplar(self):
        return {"data": [0]}

    @property
    def metadata(self):
        return {}


def simple_process(processor, source, shard_names=None):
    if shard_names is None:
        shard_names = source.shard_names
    result = []
    for shard_name in shard_names:
        for batch in source.open_shard(shard_name):
            result.append(processor([batch])[0])

    return result


class SimpleShardSource(ShardedDataSource):
    def __init__(self, num_shards=4, rows_per_shard=10):
        self._num_shards = num_shards
        self._rows_per_shard = rows_per_shard

    @property
    def shard_names(self):
        return [f"shard_{i}" for i in range(self._num_shards)]

    def open_shard_at_row(self, shard_name, row):
        shard_num = int(shard_name.split("_")[1])
        return ([shard_num * 10 + i] * 10 for i in range(row, self._rows_per_shard))


@pytest.fixture(scope="module")
def ray_init():
    ray.init(ignore_reinit_error=True)
    try:
        yield
    finally:
        ray.shutdown()


def test_tokenize_one_shard(ray_init):
    with tempfile.TemporaryDirectory() as tmpdir:
        source = SimpleShardSource()
        processor = SimpleProcessor()
        options = CacheOptions(batch_size=2, target_size_per_flush=1024)

        shard_name = source.shard_names[0]
        temporary_cache_path = os.path.join(tmpdir, shard_name)

        result = _tokenize_one_shard(temporary_cache_path, source, shard_name, processor, options)

        assert result[0] == temporary_cache_path
        assert isinstance(result[1], CacheLedger)

        # make sure the cache has the right data
        cache = TreeStore.open(processor.output_exemplar, temporary_cache_path, mode="r")

        assert len(cache) == 10

        processed = simple_process(processor, source, [shard_name])

        for i, item in enumerate(cache):
            assert np.all(item["data"] == processed[i]["data"])


def test_tokenize_all_shards(ray_init):
    with tempfile.TemporaryDirectory() as tmpdir:
        source = SimpleShardSource()
        processor = SimpleProcessor()
        options = CacheOptions(batch_size=2, target_size_per_flush=1024)

        ledgers = tokenize_all_shards(tmpdir, source, processor, options)

        assert len(ledgers) == len(source.shard_names)
        for path, ledger in ledgers:
            assert os.path.exists(path)
            assert isinstance(ledger, CacheLedger)


@pytest.mark.asyncio
async def test_concatenate_stores():
    with tempfile.TemporaryDirectory() as tmpdir:
        source = SimpleShardSource()
        processor = SimpleProcessor()
        options = CacheOptions(batch_size=2, target_size_per_flush=1024)

        caches = []
        for shard_name in source.shard_names:
            temporary_cache_path = os.path.join(tmpdir, shard_name)
            caches.append(_tokenize_one_shard(temporary_cache_path, source, shard_name, processor, options))

        permanent_cache_path = os.path.join(tmpdir, "permanent")
        ledger = await concatenate_stores(permanent_cache_path, source, processor, caches)

        assert os.path.exists(permanent_cache_path)
        assert isinstance(ledger, CacheLedger)
        assert ledger.is_finished

        # let's make sure it's right
        cache = TreeStore.open(processor.output_exemplar, permanent_cache_path, mode="r")
        assert len(cache) == 40

        processed = simple_process(processor, source)

        for i, item in enumerate(cache):
            assert np.all(item["data"] == processed[i]["data"])


def test_tokenize_and_concatenate_shards(ray_init):
    with tempfile.TemporaryDirectory() as tmpdir:
        source = SimpleShardSource()
        processor = SimpleProcessor()
        options = CacheOptions(batch_size=2, target_size_per_flush=1024)

        ledgers = tokenize_and_concatenate_shards(source, processor, tmpdir, options)

        assert len(ledgers) == len(source.shard_names)
        for path, ledger in ledgers:
            assert os.path.exists(path)
            assert isinstance(ledger, CacheLedger)

        permanent_cache_path = os.path.join(tmpdir, "permanent")
        assert os.path.exists(permanent_cache_path)
        permanent_ledger = CacheLedger.load_or_initialize(permanent_cache_path, source, processor, options)
        assert permanent_ledger.is_finished


if __name__ == "__main__":
    pytest.main()
