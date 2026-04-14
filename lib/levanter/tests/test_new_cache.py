# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from typing import Any, Dict, Iterator, Sequence

import numpy as np
import pytest
from zephyr.execution import ZephyrWorkerError

from levanter.data import BatchProcessor, ShardedDataSource, batched
from levanter.data.sharded_datasource import TextUrlDataSource
from levanter.store.cache import (
    CacheLedger,
    CacheMetadata,
    SerialCacheWriter,
    ShardedTreeCache,
    TreeStore,
    build_or_load_cache,
)


class TestProcessor(BatchProcessor[Sequence[int], dict[str, np.ndarray]]):
    def __call__(self, batch: Sequence[Sequence[int]]) -> Sequence[dict[str, np.ndarray]]:
        # return pa.RecordBatch.from_arrays([pa.array(batch)], ["test"])
        return [{"test": np.asarray(x)} for x in batch]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {}

    @property
    def output_exemplar(self):
        return {"test": np.array([0], dtype=np.int64)}

    @property
    def num_cpus(self) -> int:
        return 1


def simple_process(processor, source):
    result = []
    for shard_name in source.shard_names:
        for batch in source.open_shard(shard_name):
            result.append(processor([batch])[0])

    return result


def process_interleave(processor, source, batch_size):
    shard_iterators = {
        shard_name: batched(iter(source.open_shard(shard_name)), batch_size) for shard_name in source.shard_names
    }
    finished = 0

    while finished < len(shard_iterators):
        for shard_name, shard_iter in shard_iterators.items():
            if shard_iter is None:
                continue
            try:
                batch = next(shard_iter)
                yield from processor(batch)
            except StopIteration:
                shard_iterators[shard_name] = None
                finished += 1


class SimpleProcessor(BatchProcessor[Sequence[int], dict[str, np.ndarray]]):
    def __call__(self, batch: Sequence[Sequence[int]]) -> Sequence[dict[str, Sequence[int]]]:
        return [{"data": x} for x in batch]

    @property
    def num_cpus(self) -> int:
        return 1

    @property
    def output_exemplar(self) -> dict[str, np.ndarray]:
        return {"data": np.array([0], dtype=np.int64)}

    @property
    def metadata(self) -> Dict[str, Any]:
        return {}


class SimpleShardSource(ShardedDataSource[list[int]]):
    def __init__(self, num_shards: int = 4, rows_per_shard: int = 10):
        self._num_shards = num_shards
        self._rows_per_shard = rows_per_shard

    @property
    def shard_names(self) -> Sequence[str]:
        return [f"shard_{i}" for i in range(self._num_shards)]

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[list[int]]:
        # parse the shard name to get the shard number
        shard_num = int(shard_name.split("_")[1])
        return ([shard_num * 10 + i] * 10 for i in range(row, self._rows_per_shard))


def test_serial_cache_writer():
    with tempfile.TemporaryDirectory() as tmpdir1:
        source = SimpleShardSource(num_shards=4)
        processor = SimpleProcessor()

        exemplar = {"data": np.array([0], dtype=np.int64)}

        with SerialCacheWriter(tmpdir1, exemplar) as writer:
            for shard_name in source.shard_names:
                for ex in batched(source.open_shard(shard_name), 32):
                    writer.write_batch(processor(ex))

        _ = writer.result()
        data_path = writer._tree_store.path

        builder = TreeStore.open(exemplar, data_path, mode="r")

        assert len(builder) == 40

        for i, x in enumerate(builder):
            np.testing.assert_array_equal(x["data"], np.asarray([i % 10 + i // 10 * 10] * 10))


def test_full_end_to_end_cache():
    td = tempfile.TemporaryDirectory()
    with td as tmpdir:
        cache = build_or_load_cache(
            tmpdir,
            SimpleShardSource(num_shards=15),
            TestProcessor(),
        )

        expected = simple_process(TestProcessor(), SimpleShardSource(num_shards=15))

        all_data = cache[:]

        check_datasets_equal(all_data, expected)


def test_full_end_to_end_cache_with_groups():
    td = tempfile.TemporaryDirectory()
    with td as tmpdir:
        cache = build_or_load_cache(
            tmpdir,
            SimpleShardSource(num_shards=5),
            TestProcessor(),
        )

        expected = simple_process(TestProcessor(), SimpleShardSource(num_shards=5))

        all_data = cache[:]

        check_datasets_equal(all_data, expected)


def test_cache_remembers_its_cached():
    directory = tempfile.TemporaryDirectory()
    with directory as tmpdir:
        ds1 = build_or_load_cache(tmpdir, SimpleShardSource(), TestProcessor())

        class ThrowingProcessor(TestProcessor):
            def __call__(self, batch: Sequence[Sequence[int]]):
                raise RuntimeError("This should not be called")

        # testing this doesn't throw
        ds2 = build_or_load_cache(tmpdir, SimpleShardSource(), ThrowingProcessor())

        check_datasets_equal(ds1, ds2)


def check_datasets_equal(ds1, ds2):
    ds1 = list(ds1)
    ds2 = list(ds2)
    assert len(ds1) == len(ds2)
    for r1, r2 in zip(ds1, ds2):
        assert r1.keys() == r2.keys()
        for key in r1.keys():
            np.testing.assert_array_equal(r1[key], r2[key])


class _CustomException(Exception):
    pass


def test_cache_recover_from_crash():
    class CrashingShardSource(ShardedDataSource[list[int]]):
        def __init__(self, crash_point: int):
            self.crash_point = crash_point

        @property
        def shard_names(self) -> Sequence[str]:
            return [f"shard_{i}" for i in range(4)]

        def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[list[int]]:
            # parse the shard name to get the shard number
            shard_num = int(shard_name.split("_")[1])
            for i in range(10):
                if i == self.crash_point:
                    raise _CustomException(f"Crashing at {shard_num} {i} {self.crash_point}")
                if i >= row:
                    yield [shard_num * 10 + i] * 10

    with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as tmpdir2:
        source = CrashingShardSource(4)
        with pytest.raises(ZephyrWorkerError):
            build_or_load_cache(tmpdir, source, TestProcessor())

        source = CrashingShardSource(5)
        with pytest.raises(ZephyrWorkerError):
            build_or_load_cache(tmpdir, source, TestProcessor())

        # testing this doesn't throw
        source = CrashingShardSource(100000)
        reader1 = build_or_load_cache(tmpdir, source, TestProcessor())

        # compare to the original with no crash
        reader2 = build_or_load_cache(tmpdir2, SimpleShardSource(num_shards=4), TestProcessor())

        check_datasets_equal(reader1, reader2)


def test_no_hang_if_empty_shard_source():
    class EmptyShardSource(ShardedDataSource[list[int]]):
        @property
        def shard_names(self) -> Sequence[str]:
            return []

        def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[list[int]]:
            raise RuntimeError("This should not be called")

    with tempfile.TemporaryDirectory() as tmpdir:
        reader = build_or_load_cache(tmpdir, EmptyShardSource(), TestProcessor())
        assert list(reader) == []


def test_shard_cache_crashes_if_processor_throws():
    class ThrowingProcessor(SimpleProcessor):
        def __call__(self, batch: Sequence[Sequence[int]]):
            raise RuntimeError("exc")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        with pytest.raises(ZephyrWorkerError):
            build_or_load_cache(tmpdir, SimpleShardSource(), ThrowingProcessor())


def test_shard_cache_fails_with_multiple_shards_with_the_same_name():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/data.txt", "w") as f:
            f.write("")

        with pytest.raises(ValueError):
            TextUrlDataSource(
                [f"{tmpdir}/data.txt", f"{tmpdir}/data.txt"],
            )

        with open(f"{tmpdir}/data.txt.1", "w") as f:
            f.write("")

            dataset = TextUrlDataSource(
                [f"{tmpdir}/data.txt", f"{tmpdir}/data.txt.1"],
            )

            build_or_load_cache(tmpdir, dataset, TestProcessor())


@pytest.mark.asyncio
async def test_shard_cache_fails_gracefully_with_unknown_file_type_async():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/data.not_a_real_extension", "w") as f:
            f.write("")

        dataset = TextUrlDataSource(
            [f"{tmpdir}/data.not_a_real_extension"],
        )

        with pytest.raises(ZephyrWorkerError):
            build_or_load_cache(tmpdir, dataset, TestProcessor())

        with pytest.raises(ZephyrWorkerError):
            build_or_load_cache(tmpdir, dataset, TestProcessor())


def test_shard_cache_fails_gracefully_with_unknown_file_type():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/data.not_a_real_extension", "w") as f:
            f.write("")

        dataset = TextUrlDataSource(
            [f"{tmpdir}/data.not_a_real_extension"],
        )

        with pytest.raises(ZephyrWorkerError):
            build_or_load_cache(tmpdir, dataset, TestProcessor())

        with pytest.raises(ZephyrWorkerError):
            build_or_load_cache(tmpdir, dataset, TestProcessor())


def test_sharded_tree_cache_reads_across_shards():
    """Write independent shard caches, then read them via ShardedTreeCache."""
    source = SimpleShardSource(num_shards=4, rows_per_shard=10)
    processor = SimpleProcessor()
    exemplar = processor.output_exemplar

    with tempfile.TemporaryDirectory() as tmpdir:
        shard_paths = []
        all_expected = []

        for shard_idx, shard_name in enumerate(source.shard_names):
            shard_path = os.path.join(tmpdir, f"shard_{shard_idx}")
            with SerialCacheWriter(shard_path, exemplar, shard_name=shard_name) as writer:
                for batch in batched(source.open_shard(shard_name), 32):
                    processed = processor(batch)
                    writer.write_batch(processed)
                    all_expected.extend(processed)
            shard_paths.append(shard_path)

        # Build a ledger that references the shard paths
        shard_rows = {}
        for path in shard_paths:
            shard_ledger = CacheLedger.load(path)
            shard_rows[os.path.basename(path)] = shard_ledger.total_num_rows

        ledger = CacheLedger(
            total_num_rows=sum(shard_rows.values()),
            shard_rows=shard_rows,
            is_finished=True,
            finished_shards=list(shard_rows.keys()),
            field_counts={},
            metadata=CacheMetadata.empty(),
            shard_paths=shard_paths,
        )

        cache = ShardedTreeCache(shard_paths, exemplar, ledger)

        assert len(cache) == 40

        # Sequential read
        for i in range(40):
            row = cache[i]
            np.testing.assert_array_equal(row["data"], all_expected[i]["data"])

        # Batch read (sync)
        batch = cache.get_batch_sync(list(range(0, 40, 3)))
        for idx, b in zip(range(0, 40, 3), batch):
            np.testing.assert_array_equal(b["data"], all_expected[idx]["data"])

        # Cross-shard batch (indices spanning multiple shards)
        cross_indices = [0, 10, 20, 30, 5, 15, 25, 35]
        batch = cache.get_batch_sync(cross_indices)
        for idx, b in zip(cross_indices, batch):
            np.testing.assert_array_equal(b["data"], all_expected[idx]["data"])


def test_sharded_tree_cache_via_load():
    """TreeCache.load returns ShardedTreeCache when ledger has shard_paths."""
    source = SimpleShardSource(num_shards=3, rows_per_shard=5)
    processor = SimpleProcessor()
    exemplar = processor.output_exemplar

    with tempfile.TemporaryDirectory() as tmpdir:
        shard_paths = []
        for shard_idx, shard_name in enumerate(source.shard_names):
            shard_path = os.path.join(tmpdir, f"shard_{shard_idx}")
            with SerialCacheWriter(shard_path, exemplar, shard_name=shard_name) as writer:
                for batch in batched(source.open_shard(shard_name), 32):
                    writer.write_batch(processor(batch))
            shard_paths.append(shard_path)

        shard_rows = {}
        for path in shard_paths:
            shard_ledger = CacheLedger.load(path)
            shard_rows[os.path.basename(path)] = shard_ledger.total_num_rows

        ledger = CacheLedger(
            total_num_rows=sum(shard_rows.values()),
            shard_rows=shard_rows,
            is_finished=True,
            finished_shards=list(shard_rows.keys()),
            field_counts={},
            metadata=CacheMetadata.empty(),
            shard_paths=shard_paths,
        )
        ledger._serialize_and_commit(tmpdir)

        # Load via TreeCache.load — should return ShardedTreeCache
        from levanter.store.cache import TreeCache

        loaded = TreeCache.load(tmpdir, exemplar)
        assert isinstance(loaded, ShardedTreeCache)
        assert len(loaded) == 15
