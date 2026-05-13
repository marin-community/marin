# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, Sequence

import numpy as np
import pytest
from zephyr.execution import ZephyrWorkerError

from levanter.data import BatchProcessor, ShardedDataSource, batched
from levanter.data.sharded_datasource import TextUrlDataSource
from levanter.store.cache import (
    CACHE_LAYOUT_SHARDED,
    CacheLedger,
    SerialCacheWriter,
    TreeCache,
    TreeStore,
    build_or_load_cache,
    write_levanter_cache,
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


@pytest.mark.asyncio
async def test_tree_store_open_async_reads_cache():
    exemplar = {"data": np.array([0], dtype=np.int64)}

    with tempfile.TemporaryDirectory() as tmpdir:
        with SerialCacheWriter(tmpdir, exemplar) as writer:
            writer.write_batch([{"data": np.asarray([1, 2])}, {"data": np.asarray([3])}])

        store = await TreeStore.open_async(exemplar, tmpdir, mode="r", cache_metadata=True)

        assert len(store) == 2
        np.testing.assert_array_equal(store[0]["data"], np.asarray([1, 2]))
        np.testing.assert_array_equal(store[1]["data"], np.asarray([3]))


def test_sharded_flat_field_offsets_read_shards_concurrently(monkeypatch):
    shard_names = ["shard_0", "shard_1", "shard_2"]
    field_counts_by_shard = {
        "shard_0": {"data": 2},
        "shard_1": {"data": 3},
        "shard_2": {"data": 4},
    }
    ledger = CacheLedger(
        total_num_rows=3,
        shard_rows={shard_name: 1 for shard_name in shard_names},
        is_finished=True,
        finished_shards=shard_names,
        field_counts={"data": 9},
        field_counts_by_shard=field_counts_by_shard,
        layout=CACHE_LAYOUT_SHARDED,
    )
    cache = TreeCache("/unused", {"data": np.array([0], dtype=np.int64)}, ledger)
    active_reads = 0
    max_active_reads = 0

    class FakeRead:
        def __init__(self, value: int):
            self.value = value

        def __await__(self):
            async def read():
                nonlocal active_reads, max_active_reads
                active_reads += 1
                max_active_reads = max(max_active_reads, active_reads)
                await asyncio.sleep(0)
                active_reads -= 1
                return np.array([self.value], dtype=np.int64)

            return read().__await__()

    class FakeOffsets:
        def __init__(self, value: int):
            self.value = value

        def __getitem__(self, item):
            return self

        def read(self):
            return FakeRead(self.value)

    class FakeFieldStore:
        def __init__(self, value: int):
            self.offsets = FakeOffsets(value)

    async def shard_field_store(shard_name: str, field: str):
        assert field == "data"
        return FakeFieldStore(field_counts_by_shard[shard_name][field])

    monkeypatch.setattr(cache, "_shard_field_store_async", shard_field_store)

    offsets = cache.jagged_array_tree()["data"].offsets[0:4].read().result()

    np.testing.assert_array_equal(offsets, np.array([3, 2, 5, 9], dtype=np.int64))
    assert max_active_reads == len(shard_names)


@pytest.mark.asyncio
async def test_sharded_flat_field_offsets_share_in_flight_build(monkeypatch):
    shard_names = ["shard_0", "shard_1", "shard_2"]
    ledger = CacheLedger(
        total_num_rows=3,
        shard_rows={shard_name: 1 for shard_name in shard_names},
        is_finished=True,
        finished_shards=shard_names,
        field_counts={"data": 9},
        field_counts_by_shard={
            "shard_0": {"data": 2},
            "shard_1": {"data": 3},
            "shard_2": {"data": 4},
        },
        layout=CACHE_LAYOUT_SHARDED,
    )
    cache = TreeCache("/unused", {"data": np.array([0], dtype=np.int64)}, ledger)
    build_count = 0
    build_started = asyncio.Event()
    release_build = asyncio.Event()
    expected_offsets = np.array([3, 2, 5, 9], dtype=np.int64)

    async def build_offsets(field: str):
        nonlocal build_count
        assert field == "data"
        build_count += 1
        build_started.set()
        await release_build.wait()
        return expected_offsets

    monkeypatch.setattr(cache, "_build_flat_field_offsets_async", build_offsets)

    first = asyncio.create_task(cache._ensure_flat_field_offsets_async("data"))
    await build_started.wait()
    second = asyncio.create_task(cache._ensure_flat_field_offsets_async("data"))

    await asyncio.sleep(0)
    release_build.set()
    first_offsets, second_offsets = await asyncio.gather(first, second)

    np.testing.assert_array_equal(first_offsets, expected_offsets)
    np.testing.assert_array_equal(second_offsets, expected_offsets)
    assert build_count == 1

    cached_offsets = await cache._ensure_flat_field_offsets_async("data")
    np.testing.assert_array_equal(cached_offsets, expected_offsets)
    assert build_count == 1


def test_sharded_cache_rejects_drifted_aggregate_field_counts():
    ledger = CacheLedger(
        total_num_rows=2,
        shard_rows={"shard_0": 1, "shard_1": 1},
        is_finished=True,
        finished_shards=["shard_0", "shard_1"],
        field_counts={"data": 4},
        field_counts_by_shard={"shard_0": {"data": 2}, "shard_1": {"data": 3}},
        layout=CACHE_LAYOUT_SHARDED,
    )

    with pytest.raises(ValueError, match="field count mismatch"):
        TreeCache("/unused", {"data": np.array([0], dtype=np.int64)}, ledger)


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


def _make_levanter_records(n: int) -> list[dict[str, list[int]]]:
    return [{"input_ids": [i, i + 100], "attention_mask": [1, 1]} for i in range(n)]


def test_write_levanter_cache_end_to_end():
    """Write records and verify they can be read back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "cache")
        records = _make_levanter_records(8)

        result = write_levanter_cache(iter(records), output_path, metadata={})

        assert result["path"] == output_path
        assert result["count"] == len(records)
        assert Path(output_path, ".success").exists()

        store = TreeStore.open(records[0], output_path, mode="r", cache_metadata=False)
        assert len(store) == len(records)
        assert store[0]["input_ids"].tolist() == records[0]["input_ids"]
        assert store[len(records) - 1]["input_ids"].tolist() == records[len(records) - 1]["input_ids"]
