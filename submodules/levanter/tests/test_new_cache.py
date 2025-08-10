import asyncio
import tempfile
from typing import Any, Dict, Iterator, Sequence

import numpy as np
import pytest
import ray

from levanter.data import BatchProcessor, ShardedDataSource, batched
from levanter.data.sharded_datasource import TextUrlDataSource
from levanter.store.cache import CacheOptions, SerialCacheWriter, TreeStore, _get_builder_actor, build_or_load_cache
from levanter.utils.py_utils import logical_cpu_core_count


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


def setup_module(module):
    ray.init(
        "local", num_cpus=max(2 * logical_cpu_core_count(), 8), ignore_reinit_error=True
    )  # 2x cpu count is faster on my m1


def teardown_module(module):
    ray.shutdown()


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


@pytest.mark.ray
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


@pytest.mark.ray
def test_full_end_to_end_cache():
    td = tempfile.TemporaryDirectory()
    with td as tmpdir:
        ray_ds = build_or_load_cache(
            tmpdir,
            SimpleShardSource(num_shards=15),
            TestProcessor(),
            await_finished=True,
            options=CacheOptions(num_shard_groups=3, batch_size=8),
        )

        expected = simple_process(TestProcessor(), SimpleShardSource(num_shards=15))

        all_data = ray_ds[:]

        check_datasets_equal(all_data, expected)


@pytest.mark.ray
def test_full_end_to_end_cache_with_groups():
    td = tempfile.TemporaryDirectory()
    with td as tmpdir:
        ray_ds = build_or_load_cache(
            tmpdir,
            SimpleShardSource(num_shards=5),
            TestProcessor(),
            await_finished=True,
            options=CacheOptions(num_shard_groups=2, batch_size=8),
        )

        expected = simple_process(TestProcessor(), SimpleShardSource(num_shards=5))

        all_data = ray_ds[:]

        check_datasets_equal(all_data, expected)


@pytest.mark.ray
def test_cache_remembers_its_cached():
    directory = tempfile.TemporaryDirectory()
    with directory as tmpdir:
        ds1 = build_or_load_cache(tmpdir, SimpleShardSource(), TestProcessor(), await_finished=True)

        class ThrowingProcessor(TestProcessor):
            def __call__(self, batch: Sequence[Sequence[int]]):
                raise RuntimeError("This should not be called")

        # testing this doesn't throw
        ds2 = build_or_load_cache(tmpdir, SimpleShardSource(), ThrowingProcessor(), await_finished=True)

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


@pytest.mark.ray
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
        with pytest.raises(_CustomException):
            build_or_load_cache(tmpdir, source, TestProcessor(), CacheOptions(target_size_per_flush=1))

        # kill the broker actor so that we can test recovery
        ray.kill(
            _get_builder_actor(tmpdir, source, TestProcessor()),
            no_restart=True,
        )

        source = CrashingShardSource(5)
        with pytest.raises(_CustomException):
            build_or_load_cache(tmpdir, source, TestProcessor())

        ray.kill(
            _get_builder_actor(tmpdir, source, TestProcessor()),
            no_restart=True,
        )

        # testing this doesn't throw
        source = CrashingShardSource(100000)
        reader1 = build_or_load_cache(tmpdir, source, TestProcessor(), await_finished=True)

        # compare to the original with no crash
        reader2 = build_or_load_cache(tmpdir2, SimpleShardSource(num_shards=4), TestProcessor(), await_finished=True)

        check_datasets_equal(reader1, reader2)


@pytest.mark.ray
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


@pytest.mark.ray
def test_chunk_ordering_is_correct_with_slow_shards():
    class SlowShardSource(ShardedDataSource[list[int]]):
        @property
        def shard_names(self) -> Sequence[str]:
            return ["shard_0", "shard_1"]

        def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[list[int]]:
            assert shard_name in self.shard_names
            max_count = 40 if shard_name == "shard_1" else 20
            shard_id = int(shard_name.split("_")[1])
            for i in range(0, max_count):
                yield [i * 10 + shard_id] * 10

    with tempfile.TemporaryDirectory() as tmpdir:
        processor = TestProcessor()
        cache = build_or_load_cache(
            tmpdir,
            SlowShardSource(),
            processor,
            await_finished=False,
            options=CacheOptions.no_fanciness(16),
        )

        # now block until the cache is done
        cache.await_finished(timeout=30)

        expected = simple_process(processor, SlowShardSource())

        check_datasets_equal(list(cache[:]), expected)


@pytest.mark.asyncio
@pytest.mark.ray
async def test_can_get_elems_before_finished():
    @ray.remote(num_cpus=0)
    class Blocker:
        def __init__(self):
            self.future = asyncio.Future()

        async def block(self):
            await self.future

        def unblock(self):
            self.future.set_result(None)

    blocker_to_wait_on_test = Blocker.remote()

    class SlowShardSource(ShardedDataSource[list[int]]):
        @property
        def shard_names(self) -> Sequence[str]:
            return ["shard_0"]

        def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[list[int]]:
            for i in range(10):
                yield [i] * 10
            ray.get(blocker_to_wait_on_test.block.remote())
            for i in range(10, 20):
                yield [i] * 10

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = build_or_load_cache(
            tmpdir,
            SlowShardSource(),
            TestProcessor(),
            await_finished=False,
            options=CacheOptions(target_size_per_flush=1, batch_size=1),
        )

        # read the first 10 elements
        # ensure the first 10 elements are [{"test": np.array([i] * 10)} for i in range(10)]
        first_10 = list(await asyncio.wait_for(cache.get_batch(range(0, 10)), timeout=30.0))

        for i, x in enumerate(first_10):
            np.testing.assert_array_equal(x["test"], np.array([i] * 10))

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(cache.get_batch(range(10, 20)), timeout=0.1)

        # then unblock:
        ray.get(blocker_to_wait_on_test.unblock.remote())

        # now ensure we can get the next 10 elements, which will be
        # [{"test": np.array([i] * 10)} for i in range(10, 20)]
        batch = await asyncio.wait_for(cache.get_batch(range(10, 20)), timeout=10.0)

        for i, x in enumerate(batch):
            np.testing.assert_array_equal(x["test"], np.array([i + 10] * 10))

        ray.get(blocker_to_wait_on_test.block.remote())

        # now wait until the cache is finished. mostly so that the tempdir cleanup works
        cache.await_finished(timeout=10)


@pytest.mark.ray
def test_shard_cache_crashes_if_processor_throws():
    class ThrowingProcessor(SimpleProcessor):
        def __call__(self, batch: Sequence[Sequence[int]]):
            raise RuntimeError("exc")

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(RuntimeError):
            build_or_load_cache(tmpdir, SimpleShardSource(), ThrowingProcessor(), await_finished=True)


@pytest.mark.ray
@pytest.mark.skip("This test segfaults in CI. I think a ray bug")
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

            build_or_load_cache(tmpdir, dataset, TestProcessor(), await_finished=True)


@pytest.mark.ray
@pytest.mark.asyncio
async def test_shard_cache_fails_gracefully_with_unknown_file_type_async():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/data.not_a_real_extension", "w") as f:
            f.write("")

        dataset = TextUrlDataSource(
            [f"{tmpdir}/data.not_a_real_extension"],
        )

        with pytest.raises(ValueError):
            build_or_load_cache(tmpdir, dataset, TestProcessor(), await_finished=True)

        # now make sure it works in non-blocking mode

        cache = build_or_load_cache(tmpdir, dataset, TestProcessor(), await_finished=False)

        with pytest.raises(ValueError):
            await cache.get_batch([0])

        with pytest.raises(ValueError):
            cache.await_finished(timeout=10)

        del cache


@pytest.mark.skip("This test segfaults in CI. I think a ray bug")
@pytest.mark.ray
def test_shard_cache_fails_gracefully_with_unknown_file_type():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/data.not_a_real_extension", "w") as f:
            f.write("")

        dataset = TextUrlDataSource(
            [f"{tmpdir}/data.not_a_real_extension"],
        )

        with pytest.raises(ValueError):
            build_or_load_cache(tmpdir, dataset, TestProcessor(), await_finished=True)

        # now make sure it works in non-blocking mode

        cache = build_or_load_cache(tmpdir, dataset, TestProcessor(), await_finished=False)

        with pytest.raises(ValueError):
            cache.get_batch_sync([0])

        with pytest.raises(ValueError):
            cache.await_finished(timeout=10)

        del cache
