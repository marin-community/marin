# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import math
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.store.jagged_array import JaggedArrayStore, PreparedBatch


@pytest.mark.parametrize("cache_metadata", [True, False])
def test_append_and_get(cache_metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = JaggedArrayStore.open(tmpdir, item_rank=2, dtype=jnp.float32, cache_metadata=cache_metadata)

        data1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        data2 = jnp.array([[5.0]])

        builder.append(data1)
        builder.append(data2)

        assert len(builder) == 2

        result1 = builder[0]
        assert jnp.all(result1 == data1)

        result2 = builder[1]
        assert jnp.all(result2 == data2)

        # result_slice = builder[0:2]
        # assert isinstance(result_slice, JaggedArray)


@pytest.mark.parametrize("cache_metadata", [True, False])
def test_extend_with_multiple(cache_metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = JaggedArrayStore.open(tmpdir, item_rank=2, dtype=jnp.float32, cache_metadata=cache_metadata)

        data1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        data2 = jnp.array([[5.0]])

        builder.extend([data1, data2])

        assert len(builder) == 2

        result1 = builder[0]
        assert jnp.all(result1 == data1)

        result2 = builder[1]
        assert jnp.all(result2 == data2)


@pytest.mark.parametrize("cache_metadata", [True, False])
def test_extend_with_prepared_batch(cache_metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = JaggedArrayStore.open(tmpdir, item_rank=2, dtype=jnp.float32, cache_metadata=cache_metadata)

        data1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        data2 = np.array([[5.0]], dtype=jnp.float32)
        prepared = PreparedBatch.from_batch([data1, data2])

        builder.extend(prepared)

        assert len(builder) == 2

        result1 = builder[0]
        assert jnp.all(result1 == data1)

        result2 = builder[1]
        assert jnp.all(result2 == data2)

        # extendd with more data
        data3 = jnp.array([[6.0, 7.0], [8.0, 9.0]])
        data4 = jnp.array([[10.0]])
        prepared2 = PreparedBatch.from_batch([data3, data4])

        builder.extend(prepared2)

        assert len(builder) == 4

        result3 = builder[2]
        assert jnp.all(result3 == data3)

        result4 = builder[3]
        assert jnp.all(result4 == data4)


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_metadata", [True, False])
async def test_extend_with_prepared_batch_async(cache_metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = JaggedArrayStore.open(tmpdir, item_rank=2, dtype=jnp.float32, cache_metadata=cache_metadata)

        data1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        data2 = np.array([[5.0]], dtype=jnp.float32)
        prepared = PreparedBatch.from_batch([data1, data2])

        await builder.extend_async(prepared)

        assert len(builder) == 2

        result1 = builder[0]
        assert jnp.all(result1 == data1)

        result2 = builder[1]
        assert jnp.all(result2 == data2)

        # extendd with more data
        data3 = jnp.array([[6.0, 7.0], [8.0, 9.0]])
        data4 = jnp.array([[10.0]])
        prepared2 = PreparedBatch.from_batch([data3, data4])

        await builder.extend_async(prepared2)

        assert len(builder) == 4

        result3 = builder[2]
        assert jnp.all(result3 == data3)

        result4 = builder[3]
        assert jnp.all(result4 == data4)


def test_append_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = JaggedArrayStore.open(tmpdir, item_rank=1, dtype=jnp.float32)
        with pytest.raises(ValueError):
            builder.append(jnp.array([[1.0, 2.0]]))


@pytest.mark.parametrize("cache_metadata", [True, False])
def test_append_single_rank(cache_metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = JaggedArrayStore.open(tmpdir, item_rank=1, dtype=jnp.float32, cache_metadata=cache_metadata)

        data = jnp.array([1.0, 2.0, 3.0])
        builder.append(data)

        assert len(builder) == 1

        result = builder[0]
        assert jnp.all(result == data)


@pytest.mark.parametrize("cache_metadata", [True, False])
def test_append_multi_rank(cache_metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = JaggedArrayStore.open(tmpdir, item_rank=2, dtype=jnp.float32, cache_metadata=cache_metadata)

        data1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        data2 = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        builder.append(data1)
        builder.append(data2)

        assert len(builder) == 2

        result1 = builder[0]
        assert jnp.all(result1 == data1)

        result2 = builder[1]
        assert jnp.all(result2 == data2)


def test_getitem_out_of_bounds():
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = JaggedArrayStore.open(tmpdir, item_rank=2, dtype=jnp.float32)

        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        builder.append(data)

        with pytest.raises(IndexError):
            builder[2]


def test_step_slicing():
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = JaggedArrayStore.open(tmpdir, item_rank=2, dtype=jnp.float32)

        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        builder.append(data)

        # with pytest.raises(ValueError):
        #     builder[::2]


async def create_builder_with_data(
    directory, num_sequences: int, sequence_length: int | tuple[int, ...], cache_metadata: bool = True
) -> JaggedArrayStore:
    if isinstance(sequence_length, int):
        sequence_length = (sequence_length,)

    """Helper function to create a JaggedArrayStore with specific data."""
    seed = jax.random.PRNGKey(num_sequences * math.prod(sequence_length))

    builder = await JaggedArrayStore.open_async(
        directory, item_rank=len(sequence_length), dtype=jnp.int64, cache_metadata=cache_metadata
    )
    for i in range(num_sequences):
        key, seed = jax.random.split(seed)
        data = jax.random.randint(key, sequence_length, 0, 100)
        await builder.append_async(data)

    return builder


def create_builder_with_data_sync(
    directory, num_sequences: int, sequence_length: int | tuple[int, ...], cache_metadata: bool = True
) -> JaggedArrayStore:
    if isinstance(sequence_length, int):
        sequence_length = (sequence_length,)

    """Helper function to create a JaggedArrayStore with specific data."""
    seed = jax.random.PRNGKey(num_sequences * math.prod(sequence_length))

    builder = JaggedArrayStore.open(
        directory, item_rank=len(sequence_length), dtype=jnp.int64, cache_metadata=cache_metadata
    )
    for i in range(num_sequences):
        key, seed = jax.random.split(seed)
        data = jax.random.randint(key, sequence_length, 0, 100)
        builder.append(data)

    return builder


@pytest.mark.asyncio
async def test_trim_to_size_async():
    tmpdir = tempfile.TemporaryDirectory().name
    builder = await create_builder_with_data(tmpdir, num_sequences=10, sequence_length=1000)

    # Initial size
    initial_size = len(builder)
    assert initial_size == 10

    expected_data = list([builder[i] for i in range(10)])

    # Trim to smaller size
    await builder.trim_to_size_async(5)
    new_size = len(builder)
    assert new_size == 5

    # Verify the data integrity
    trimmed_data = await builder.data[0:5000].read()
    assert jnp.all(trimmed_data == jnp.concatenate(expected_data[:5]))

    # Trim to zero size
    await builder.trim_to_size_async(0)
    new_size = len(builder)
    assert new_size == 0

    # Verify the data integrity
    trimmed_data = await builder.data[0:5000].read()
    assert jnp.all(trimmed_data == 0)


@pytest.mark.asyncio
async def test_trim_to_size_larger_than_current():
    tmpdir = tempfile.TemporaryDirectory().name
    builder = await create_builder_with_data(tmpdir, num_sequences=10, sequence_length=1000)
    expected_data = list([builder[i] for i in range(10)])

    # Initial size
    initial_size = len(builder)
    assert initial_size == 10

    # Trim to a larger size than current (should not change)
    await builder.trim_to_size_async(15)
    new_size = len(builder)
    assert new_size == 10

    # Verify the data integrity
    trimmed_data = await builder.data[0:10000].read()
    assert np.array_equal(trimmed_data, jnp.concatenate(expected_data[:10]))


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_metadata", [True, False])
async def test_trim_to_size_with_shapes_async(cache_metadata):
    tmpdir = tempfile.TemporaryDirectory().name
    builder = await create_builder_with_data(
        tmpdir, num_sequences=10, sequence_length=(10, 100), cache_metadata=cache_metadata
    )
    expected_shapes = list(await builder.shapes[0:10].read())

    # Trim to smaller size
    await builder.trim_to_size_async(5)
    new_size = len(builder)
    assert new_size == 5

    # Verify the shapes integrity
    trimmed_shapes = await builder.shapes[0:5].read()
    assert np.array_equal(trimmed_shapes, jnp.stack(expected_shapes[:5]))


@pytest.mark.parametrize("cache_metadata", [True, False])
def test_trim_to_size_sync(cache_metadata):
    tmpdir = tempfile.TemporaryDirectory().name
    builder = create_builder_with_data_sync(
        tmpdir, num_sequences=10, sequence_length=1000, cache_metadata=cache_metadata
    )

    # Initial size
    initial_size = len(builder)
    assert initial_size == 10

    expected_data = list([builder[i] for i in range(10)])

    # Trim to smaller size
    builder.trim_to_size(5)
    new_size = len(builder)
    assert new_size == 5

    # Verify the data integrity
    trimmed_data = builder.data[0:5000].read().result()
    assert jnp.all(trimmed_data == jnp.concatenate(expected_data[:5]))

    # Trim to zero size
    builder.trim_to_size(0)
    new_size = len(builder)
    assert new_size == 0

    # Verify the data integrity
    trimmed_data = builder.data[0:10000].read().result()
    assert jnp.all(trimmed_data == 0)


@pytest.mark.asyncio
async def test_get_batch_single_item():
    tmpdir = tempfile.TemporaryDirectory().name
    builder = await create_builder_with_data(tmpdir, num_sequences=10, sequence_length=1000)

    # Retrieve a single item using get_batch
    batch = await builder.get_batch([3])
    result = batch[0]

    expected_data = await builder.get_item_async(3)

    assert np.array_equal(result, expected_data)


@pytest.mark.asyncio
async def test_get_batch_multiple_items():
    tmpdir = tempfile.TemporaryDirectory().name
    builder = await create_builder_with_data(tmpdir, num_sequences=10, sequence_length=1000)

    # Retrieve multiple items using get_batch
    indices = [1, 4, 7]
    batch = await builder.get_batch(indices)

    for idx, result in zip(indices, batch):
        expected_data = await builder.get_item_async(idx)
        assert np.array_equal(result, expected_data)


@pytest.mark.asyncio
async def test_get_batch_out_of_order():
    tmpdir = tempfile.TemporaryDirectory().name
    builder = await create_builder_with_data(tmpdir, num_sequences=10, sequence_length=1000)

    # Retrieve items out of order using get_batch
    indices = [7, 2, 5]
    batch = await builder.get_batch(indices)

    for idx, result in zip(indices, batch):
        expected_data = await builder.get_item_async(idx)
        assert np.array_equal(result, expected_data)


@pytest.mark.asyncio
async def test_get_batch_with_shapes():
    tmpdir = tempfile.TemporaryDirectory().name
    builder = await create_builder_with_data(tmpdir, num_sequences=10, sequence_length=(10, 100))

    # Retrieve multiple items using get_batch
    indices = [0, 3, 6]
    batch = await builder.get_batch(indices)

    for idx, result in zip(indices, batch):
        expected_data = await builder.get_item_async(idx)
        assert np.array_equal(result, expected_data)


@pytest.mark.asyncio
async def test_get_batch_empty():
    tmpdir = tempfile.TemporaryDirectory().name
    builder = await create_builder_with_data(tmpdir, num_sequences=10, sequence_length=1000)

    # Retrieve an empty batch
    batch = await builder.get_batch([])

    assert batch == []


if __name__ == "__main__":
    pytest.main()
