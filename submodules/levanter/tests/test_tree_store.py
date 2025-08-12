import tempfile
from typing import Any, Dict, Iterator, List, Sequence

import numpy as np
import pytest
import tensorstore as ts

from levanter.data import BatchProcessor, ShardedDataSource
from levanter.data.utils import batched
from levanter.store.tree_store import TreeStore


class SimpleProcessor(BatchProcessor[Sequence[int], dict[str, np.ndarray]]):
    def __call__(self, batch: Sequence[Sequence[int]]) -> Sequence[dict[str, Sequence[int]]]:
        return [{"data": x} for x in batch]

    @property
    def output_exemplar(self) -> dict[str, Sequence[int]]:
        return {"data": np.array([0], dtype=np.int64)}

    @property
    def num_cpus(self) -> int:
        return 1

    @property
    def metadata(self) -> Dict[str, Any]:
        return {}


class SimpleShardSource(ShardedDataSource[List[int]]):
    def __init__(self, num_shards: int = 4):
        self._num_shards = num_shards

    @property
    def shard_names(self) -> Sequence[str]:
        return [f"shard_{i}" for i in range(self._num_shards)]

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[List[int]]:
        # parse the shard name to get the shard number
        shard_num = int(shard_name.split("_")[1])
        return ([shard_num * 10 + i] * 10 for i in range(row, 10))


def test_tree_builder_with_processor():
    with tempfile.TemporaryDirectory() as tempdir:
        exemplar = {"data": np.array([0], dtype=np.int64)}

        builder = TreeStore.open(exemplar, tempdir, mode="w")
        processor = SimpleProcessor()
        source = SimpleShardSource()

        for batch in batched(source, 8):
            processed = processor(batch)
            builder.extend(processed)

        assert len(builder) == 40

        for i, x in enumerate(builder):
            assert len(x) == 1

            np.testing.assert_array_equal(x["data"], np.asarray([i % 10 + i // 10 * 10] * 10))

        assert i == 39

        # now test random access
        for i in range(40):
            x = builder[i]
            assert len(x) == 1
            np.testing.assert_array_equal(x["data"], np.asarray([i % 10 + i // 10 * 10] * 10))

        # double check columnar access
        assert builder.tree["data"].data_size == 10 * 40


def test_append_batch():
    with tempfile.TemporaryDirectory() as tmpdir:
        exemplar = {"a": np.array([0], dtype=np.float64), "b": np.array([0], dtype=np.float64)}
        builder = TreeStore.open(exemplar, tmpdir)

        batch1 = [
            {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
            {"a": np.array([5.0, 6.0]), "b": np.array([7.0, 8.0])},
        ]
        builder.extend(batch1)

        assert len(builder) == 2

        result1 = builder[0]
        assert np.all(result1["a"] == np.array([1.0, 2.0]))
        assert np.all(result1["b"] == np.array([3.0, 4.0]))

        result2 = builder[1]
        assert np.all(result2["a"] == np.array([5.0, 6.0]))
        assert np.all(result2["b"] == np.array([7.0, 8.0]))


def test_append_batch_different_shapes():
    with tempfile.TemporaryDirectory() as tmpdir:

        def _f32(x):
            return np.asarray(x, dtype=np.float32)

        exemplar = {"a": _f32([0]), "b": _f32([0])}
        builder = TreeStore.open(exemplar, tmpdir)
        batch1 = [
            {"a": _f32([1.0, 2.0]), "b": _f32([3.0, 4.0])},
            {"a": _f32([5.0, 6.0]), "b": _f32([7.0, 8.0])},
        ]
        builder.extend(batch1)

        batch2 = [
            {"a": _f32([9.0]), "b": _f32([10.0])},
            {"a": _f32([11.0, 12.0, 13.0]), "b": _f32([14.0, 15.0, 16.0])},
        ]
        builder.extend(batch2)

        assert len(builder) == 4

        result3 = builder[2]
        assert np.all(result3["a"] == np.array([9.0]))
        assert np.all(result3["b"] == np.array([10.0]))

        result4 = builder[3]
        assert np.all(result4["a"] == np.array([11.0, 12.0, 13.0]))
        assert np.all(result4["b"] == np.array([14.0, 15.0, 16.0]))


def test_extend_batch_different_shapes():
    with tempfile.TemporaryDirectory() as tmpdir:
        exemplar = {"a": np.array([0], dtype=np.float64), "b": np.array([0], dtype=np.float64)}
        builder = TreeStore.open(exemplar, tmpdir)

        batch1 = {"a": [np.array([1.0, 2.0]), np.array([5.0, 6.0])], "b": [np.array([3.0, 4.0]), np.array([7.0, 8.0])]}
        builder.extend_with_batch(batch1)

        batch2 = {
            "a": [np.array([9.0]), np.array([11.0, 12.0, 13.0])],
            "b": [np.array([10.0]), np.array([14.0, 15.0, 16.0])],
        }
        builder.extend_with_batch(batch2)

        assert len(builder) == 4

        result3 = builder[2]
        assert np.all(result3["a"] == np.array([9.0]))
        assert np.all(result3["b"] == np.array([10.0]))

        result4 = builder[3]
        assert np.all(result4["a"] == np.array([11.0, 12.0, 13.0]))
        assert np.all(result4["b"] == np.array([14.0, 15.0, 16.0]))


def test_len():
    with tempfile.TemporaryDirectory() as tmpdir:
        exemplar = {"a": np.array([0], dtype=np.float64), "b": np.array([0], dtype=np.float64)}
        builder = TreeStore.open(exemplar, tmpdir)

        assert len(builder) == 0

        batch = [
            {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
            {"a": np.array([5.0, 6.0]), "b": np.array([7.0, 8.0])},
        ]
        builder.extend(batch)

        assert len(builder) == 2


def test_getitem():
    with tempfile.TemporaryDirectory() as tmpdir:
        exemplar = {"a": np.array([0], dtype=np.float64), "b": np.array([0], dtype=np.float64)}
        builder = TreeStore.open(exemplar, tmpdir)

        batch = [
            {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
            {"a": np.array([5.0, 6.0]), "b": np.array([7.0, 8.0])},
        ]
        builder.extend(batch)

        result = builder[0]
        assert np.all(result["a"] == np.array([1.0, 2.0]))
        assert np.all(result["b"] == np.array([3.0, 4.0]))

        result = builder[1]
        assert np.all(result["a"] == np.array([5.0, 6.0]))
        assert np.all(result["b"] == np.array([7.0, 8.0]))

        # test slice
        # result = builder[0:2]
        # assert isinstance(result["a"], JaggedArray)
        # assert isinstance(result["b"], JaggedArray)


def test_getitem_out_of_bounds():
    with tempfile.TemporaryDirectory() as tmpdir:
        exemplar = {"a": np.array([0], dtype=np.float64), "b": np.array([0], dtype=np.float64)}
        builder = TreeStore.open(exemplar, tmpdir)

        batch = [
            {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
            {"a": np.array([5.0, 6.0]), "b": np.array([7.0, 8.0])},
        ]
        builder.extend(batch)

        with pytest.raises(IndexError):
            builder[2]


def test_iter():
    with tempfile.TemporaryDirectory() as tmpdir:
        exemplar = {"a": np.array([0], dtype=np.float64), "b": np.array([0], dtype=np.float64)}
        builder = TreeStore.open(exemplar, tmpdir)

        batch = [
            {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
            {"a": np.array([5.0, 6.0]), "b": np.array([7.0, 8.0])},
        ]
        builder.extend(batch)

        for i, result in enumerate(builder):
            if i == 0:
                assert np.all(result["a"] == np.array([1.0, 2.0]))
                assert np.all(result["b"] == np.array([3.0, 4.0]))
            elif i == 1:
                assert np.all(result["a"] == np.array([5.0, 6.0]))
                assert np.all(result["b"] == np.array([7.0, 8.0]))
            else:
                pytest.fail("Unexpected index")


def test_reading_from_written():
    with tempfile.TemporaryDirectory() as tmpdir:
        exemplar = {"a": np.array([0], dtype=np.float64), "b": np.array([0], dtype=np.float64)}
        builder = TreeStore.open(exemplar, tmpdir, mode="w")

        batch = [
            {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
            {"a": np.array([5.0, 6.0]), "b": np.array([7.0, 8.0])},
        ]
        builder.extend(batch)

        del builder

        builder2 = TreeStore.open(exemplar, tmpdir, mode="r")

        for i, result in enumerate(builder2):
            if i == 0:
                assert np.all(result["a"] == np.array([1.0, 2.0]))
                assert np.all(result["b"] == np.array([3.0, 4.0]))
            elif i == 1:
                assert np.all(result["a"] == np.array([5.0, 6.0]))
                assert np.all(result["b"] == np.array([7.0, 8.0]))
            else:
                pytest.fail("Unexpected index")


def test_using_prepared_batches():
    with tempfile.TemporaryDirectory() as tmpdir:
        exemplar = {"a": np.array([0], dtype=np.float64), "b": np.array([0], dtype=np.float64)}
        builder = TreeStore.open(exemplar, tmpdir, mode="w")
        preparer = builder.batch_preparer

        batch = [
            {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
            {"a": np.array([5.0, 6.0]), "b": np.array([7.0, 8.0])},
        ]
        batch = preparer(batch)
        builder.extend_with_batch(batch)

        del builder

        builder2 = TreeStore.open(exemplar, tmpdir, mode="r")

        for i, result in enumerate(builder2):
            if i == 0:
                assert np.all(result["a"] == np.array([1.0, 2.0]))
                assert np.all(result["b"] == np.array([3.0, 4.0]))
            elif i == 1:
                assert np.all(result["a"] == np.array([5.0, 6.0]))
                assert np.all(result["b"] == np.array([7.0, 8.0]))
            else:
                pytest.fail("Unexpected index")


def test_resolve_changed_cache_size():
    with tempfile.TemporaryDirectory() as tmpdir:
        exemplar = {"a": np.array([0], dtype=np.float64), "b": np.array([0], dtype=np.float64)}
        builder = TreeStore.open(exemplar, tmpdir, mode="w")
        follower = TreeStore.open(exemplar, tmpdir, mode="r")

        batch = [
            {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
            {"a": np.array([5.0, 6.0]), "b": np.array([7.0, 8.0])},
        ]
        builder.extend(batch)

        follower = follower.reload()
        follower2 = TreeStore.open(exemplar, tmpdir, mode="r")

        assert len(follower2) == 2
        assert len(follower) == 2

        builder.extend(batch)
        follower = follower.reload()

        assert len(follower) == 4


# this test mostly exists to help me remember the API


def test_simple_resize_bounds():
    with tempfile.TemporaryDirectory() as tmpdir:
        store1 = ts.open(
            {
                "driver": "zarr",
                "kvstore": {
                    "driver": "file",
                    "path": tmpdir,
                },
            },
            create=True,
            dtype=ts.int32,
            shape=[1000, 2000, 3000],
            chunk_layout=ts.ChunkLayout(inner_order=[2, 1, 0]),
        ).result()

        store2 = ts.open(
            {
                "driver": "zarr",
                "kvstore": {
                    "driver": "file",
                    "path": tmpdir,
                },
            },
            dtype=ts.int32,
        ).result()

        assert store2.shape == (1000, 2000, 3000)
        assert store2.chunk_layout.inner_order == (2, 1, 0)

        store1 = store1.resize(exclusive_max=[2000, 3000, 4000]).result()

        assert store1.shape == (2000, 3000, 4000)

        # store2 = store2[ts.d[0].mark_bounds_implicit[True]].resolve().result()
        spec = store2.spec(retain_context=True, minimal_spec=True)
        # spec.update(transform={})
        store2 = ts.open(spec).result()

        # store2 = store2.resolve(fix_resizable_bounds=False).result()

        assert store2.shape == (2000, 3000, 4000)  # nope?


@pytest.mark.asyncio
async def test_get_batch_single_item():
    with tempfile.TemporaryDirectory() as tmpdir:
        exemplar = {"a": np.array([0], dtype=np.float64), "b": np.array([0], dtype=np.float64)}
        builder = TreeStore.open(exemplar, tmpdir)

        batch1 = [
            {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
            {"a": np.array([5.0, 6.0]), "b": np.array([7.0, 8.0])},
        ]
        builder.extend(batch1)

        # Retrieve a single item using get_batch
        batch = await builder.get_batch([0])
        result = batch[0]

        expected_data = builder[0]
        assert np.array_equal(result["a"], expected_data["a"])
        assert np.array_equal(result["b"], expected_data["b"])


@pytest.mark.asyncio
async def test_get_batch_multiple_items():
    with tempfile.TemporaryDirectory() as tmpdir:
        exemplar = {"a": np.array([0], dtype=np.float64), "b": np.array([0], dtype=np.float64)}
        builder = TreeStore.open(exemplar, tmpdir)

        batch1 = [
            {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
            {"a": np.array([5.0, 6.0]), "b": np.array([7.0, 8.0])},
            {"a": np.array([9.0, 10.0]), "b": np.array([11.0, 12.0])},
        ]
        builder.extend(batch1)

        # Retrieve multiple items using get_batch
        indices = [0, 2]
        batch = await builder.get_batch(indices)

        for idx, result in zip(indices, batch):
            expected_data = builder[idx]
            assert np.array_equal(result["a"], expected_data["a"])
            assert np.array_equal(result["b"], expected_data["b"])


@pytest.mark.asyncio
async def test_get_batch_out_of_order():
    with tempfile.TemporaryDirectory() as tmpdir:
        exemplar = {"a": np.array([0], dtype=np.float64), "b": np.array([0], dtype=np.float64)}
        builder = TreeStore.open(exemplar, tmpdir)

        batch1 = [
            {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
            {"a": np.array([5.0, 6.0]), "b": np.array([7.0, 8.0])},
            {"a": np.array([9.0, 10.0]), "b": np.array([11.0, 12.0])},
        ]
        builder.extend(batch1)

        # Retrieve items out of order using get_batch
        indices = [2, 0, 1]
        batch = await builder.get_batch(indices)

        for idx, result in zip(indices, batch):
            expected_data = builder[idx]
            assert np.array_equal(result["a"], expected_data["a"])
            assert np.array_equal(result["b"], expected_data["b"])


@pytest.mark.asyncio
async def test_get_batch_with_shapes():
    with tempfile.TemporaryDirectory() as tmpdir:
        exemplar = {"a": np.array([[0]], dtype=np.float64), "b": np.array([[0]], dtype=np.float64)}
        builder = TreeStore.open(exemplar, tmpdir)

        batch1 = [
            {"a": np.array([[1.0, 2.0], [3.0, 4.0]]), "b": np.array([[5.0, 6.0], [7.0, 8.0]])},
            {"a": np.array([[9.0, 10.0], [11.0, 12.0]]), "b": np.array([[13.0, 14.0], [15.0, 16.0]])},
        ]
        builder.extend(batch1)

        # Retrieve multiple items using get_batch
        indices = [0, 1]
        batch = await builder.get_batch(indices)

        for idx, result in zip(indices, batch):
            expected_data = builder[idx]
            assert np.array_equal(result["a"], expected_data["a"])
            assert np.array_equal(result["b"], expected_data["b"])


@pytest.mark.asyncio
async def test_get_batch_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        exemplar = {"a": np.array([0], dtype=np.float64), "b": np.array([0], dtype=np.float64)}
        builder = TreeStore.open(exemplar, tmpdir)

        batch1 = [
            {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
            {"a": np.array([5.0, 6.0]), "b": np.array([7.0, 8.0])},
        ]
        builder.extend(batch1)

        # Retrieve an empty batch
        batch = await builder.get_batch([])

        assert batch == []
