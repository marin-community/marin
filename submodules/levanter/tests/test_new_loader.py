import asyncio
from typing import Optional, Sequence

import jax
import numpy as np
import pytest
from jax.sharding import Mesh

import haliax
from haliax import Axis
from haliax.partitioning import ResourceAxis

from levanter.data.dataset import AsyncDataset, ListAsyncDataset
from levanter.data.loader import DataLoader, check_sharded_consistency
from levanter.schedule import ScheduleStep

from .test_utils import skip_if_not_enough_devices


def _small_dataset(seq_len=128, num_sequences=200) -> AsyncDataset[Sequence[int]]:
    sequences = [np.arange(seq_len) + 1000 * i for i in range(num_sequences)]

    return ListAsyncDataset(sequences, is_complete=True)


@skip_if_not_enough_devices(2)
def test_local_batched_data_loading_model_axis_2():
    devices = jax.devices()
    model_axis_size = 2

    mesh = Mesh(
        np.array(devices).reshape(-1, model_axis_size),
        (ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, haliax.axis_mapping({"batch": ResourceAxis.DATA}):

        seq_len = 128
        cache = _small_dataset(seq_len)
        loader = DataLoader(cache, len(devices), max_buffered_batches=10, mesh=mesh, axis_resources=None)

        batches = list(loader)
        for batch in batches:
            check_sharded_consistency(batch, check_disjoint_indices_are_different=True)


def test_local_batched_data_loading_model_axis_1():
    devices = jax.devices()
    model_axis_size = 1

    mesh = Mesh(
        np.array(devices).reshape(-1, model_axis_size),
        (ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, haliax.axis_mapping({"batch": ResourceAxis.DATA}):

        seq_len = 128
        cache = _small_dataset(seq_len)
        loader = DataLoader(cache, len(devices), max_buffered_batches=10, mesh=mesh, axis_resources=None)

        batches = list(loader)
        for batch in batches:
            check_sharded_consistency(batch, check_disjoint_indices_are_different=True)


class StructuredDataset(AsyncDataset):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.begin = 0
        self.end = 256
        self.stride = 1

    async def async_len(self) -> int:
        return (self.end - self.begin) // self.stride

    async def getitem_async(self, index: int) -> dict:
        index = self.begin + index * self.stride
        return {
            "input_ids": np.arange(self.seq_len, dtype=np.int32) + index * 1000,
            "labels": np.arange(self.seq_len, dtype=np.int32) + index * 1000,
            "extra": {
                "input_ids": np.arange(self.seq_len, dtype=np.int32) + index * 1000,
                "mask": np.arange(self.seq_len * 2, dtype=np.int32).reshape(-1, 2) + index * 1000,
            },
        }

    async def final_length_is_known(self) -> bool:
        return True

    def is_finite(self) -> bool:
        return True

    async def current_len(self) -> Optional[int]:
        return await self.async_len()

    async def get_batch(self, indices: Sequence[int]):
        out = await asyncio.gather(*(self.getitem_async(i) for i in indices))
        return out


def test_structured_batches_model_axis_1():
    devices = jax.devices()
    model_axis_size = 1

    mesh = Mesh(
        np.array(devices).reshape(-1, model_axis_size),
        (ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, haliax.axis_mapping({"batch": ResourceAxis.DATA}):
        seq_len = 128
        dataset = StructuredDataset(seq_len)
        loader = DataLoader(dataset, len(devices), max_buffered_batches=10, mesh=mesh, axis_resources=None)

        batches = list(loader)
        for batch in batches:
            check_sharded_consistency(batch, check_disjoint_indices_are_different=True)


@skip_if_not_enough_devices(2)
def test_structured_batches_model_axis_2():
    devices = jax.devices()
    model_axis_size = 2

    mesh = Mesh(
        np.array(devices).reshape(-1, model_axis_size),
        (ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, haliax.axis_mapping({"batch": ResourceAxis.DATA}):
        seq_len = 128
        dataset = StructuredDataset(seq_len)
        loader = DataLoader(dataset, len(devices), max_buffered_batches=10, mesh=mesh, axis_resources=None)

        batches = list(loader)
        for batch in batches:
            check_sharded_consistency(batch, check_disjoint_indices_are_different=True)


class StructuredDatasetWithNames(AsyncDataset):
    def __init__(self, Height: Axis, Width: Axis, begin, end, stride):
        super().__init__()
        self.Height = Height
        self.Width = Width
        self.begin = begin
        self.end = end
        self.stride = stride

    async def final_length_is_known(self) -> bool:
        return True

    def is_finite(self) -> bool:
        return True

    async def current_len(self) -> Optional[int]:
        return True

    async def get_batch(self, indices: Sequence[int]):
        out = await asyncio.gather(*(self.getitem_async(i) for i in indices))
        return out

    async def async_len(self) -> int:
        return (self.end - self.begin) // self.stride

    async def getitem_async(self, index: int) -> dict:
        index = self.begin + index * self.stride
        return {
            "input_ids": self._gen_image(index),
            "labels": self._gen_image(index),
            "extra": {
                "input_ids": self._gen_image(index),
                "mask": haliax.arange(self.Height) + index * 1000,
            },
        }

    def _gen_image(self, index):
        image = (
            np.arange(self.Height.size * self.Width.size, dtype=np.int32).reshape(self.Height.size, self.Width.size)
            + index * 1000
        )

        return haliax.named(image, (self.Height, self.Width))

    def __iter__(self):
        for i in range(self.begin, self.end, self.stride):
            yield self[i]


def test_structured_batches_model_axis_1_with_names():
    devices = jax.devices()
    model_axis_size = 1

    mesh = Mesh(
        np.array(devices).reshape(-1, model_axis_size),
        (ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, haliax.axis_mapping({"batch": ResourceAxis.DATA}):
        Height = Axis("Height", 16)
        Width = Axis("Width", 16)
        dataset = StructuredDatasetWithNames(Height, Width, 0, len(devices) * 10, 1)
        loader = DataLoader(dataset, len(devices), max_buffered_batches=0, mesh=mesh, axis_resources=None)

        batches = list(loader)
        for batch in batches:
            check_sharded_consistency(batch, check_disjoint_indices_are_different=True)

        assert len(batches) == 10


@skip_if_not_enough_devices(2)
def test_structured_batches_model_axis_1_non_divisible_batch_size():
    devices = jax.devices()
    model_axis_size = 1

    mesh = Mesh(
        np.array(devices).reshape(-1, model_axis_size),
        (ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, haliax.axis_mapping({"batch": ResourceAxis.DATA}):
        Height = Axis("Height", 16)
        Width = Axis("Width", 16)
        dataset = StructuredDatasetWithNames(Height, Width, 0, 10, 1)
        loader = DataLoader(
            dataset, 1, max_buffered_batches=0, mesh=mesh, axis_resources=None, allow_nondivisible_batch_size=True
        )

        batches = list(loader)
        for batch in batches:
            check_sharded_consistency(batch, check_disjoint_indices_are_different=False)

        # check that it crashes if allow_non_divisible_batch_size is False
        with pytest.raises(ValueError):
            DataLoader(
                dataset, 1, max_buffered_batches=0, mesh=mesh, axis_resources=None, allow_nondivisible_batch_size=False
            )

        assert len(batches) == 10


@skip_if_not_enough_devices(2)
def test_structured_batches_model_axis_2_with_names():
    devices = jax.devices()
    model_axis_size = 2

    mesh = Mesh(
        np.array(devices).reshape(-1, model_axis_size),
        (ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, haliax.axis_mapping({"batch": ResourceAxis.DATA}):
        Height = Axis("Height", 16)
        Width = Axis("Width", 16)
        dataset = StructuredDatasetWithNames(Height, Width, 0, 256, 1)
        loader = DataLoader(dataset, len(devices), max_buffered_batches=10, mesh=mesh, axis_resources=None)

        batches = list(loader)
        for batch in batches:
            check_sharded_consistency(batch, check_disjoint_indices_are_different=True)


@skip_if_not_enough_devices(4)
def test_structured_batches_model_axis_2_subsharded():
    """This tests data loading if individual datums are sharded too"""
    devices = jax.devices()
    model_axis_size = 2

    mesh = Mesh(
        np.array(devices).reshape(-1, model_axis_size),
        (ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    Height = Axis("Height", 16)
    Width = Axis("Width", 16)
    with mesh, haliax.axis_mapping({"batch": ResourceAxis.DATA, Height.name: ResourceAxis.MODEL}):
        dataset = StructuredDatasetWithNames(Height, Width, 0, 256, 1)
        loader = DataLoader(dataset, len(devices), max_buffered_batches=10, mesh=mesh, axis_resources=None)

        for batch in iter(loader):
            check_sharded_consistency(batch, check_disjoint_indices_are_different=True)


@pytest.mark.parametrize("model_axis_size", [1, 2])
def test_loader_with_batch_scheduler(model_axis_size):
    schedule = [ScheduleStep(start=0, value=8), ScheduleStep(start=10, value=16), ScheduleStep(start=20, value=32)]

    if len(jax.devices()) % model_axis_size != 0:
        pytest.skip("This test requires the number of devices to divide model_axis_size")

    if 32 % (len(jax.devices()) // model_axis_size) != 0:
        pytest.skip("This test requires the number of devices to divide 32")

    devices = jax.devices()

    mesh = Mesh(
        np.array(devices).reshape(-1, model_axis_size),
        (ResourceAxis.DATA, ResourceAxis.MODEL),
    )

    with mesh, haliax.axis_mapping({"batch": ResourceAxis.DATA}):
        seq_len = 128
        cache = _small_dataset(seq_len, num_sequences=1000)
        loader = DataLoader(
            cache, schedule, max_buffered_batches=10, mesh=mesh, axis_resources=None, pad_final_batch=False
        )

        for step, batch in enumerate(loader):
            if step < 10:
                assert len(batch) == 8
            elif step < 20:
                assert len(batch) == 16
            else:
                assert len(batch) == 32, f"step: {step} len: {len(batch)}"

        # total steps: 10 * 8 + 10 * 16 = 240, (1000 - 240) // 32 = 23
        assert step == 20 + 22


@pytest.mark.parametrize("model_axis_size", [1, 2])
def test_padded_final_batch(model_axis_size):
    schedule = [ScheduleStep(start=0, value=8), ScheduleStep(start=10, value=16), ScheduleStep(start=20, value=32)]

    if len(jax.devices()) % model_axis_size != 0:
        pytest.skip("This test requires the number of devices to divide model_axis_size")

    if 32 % (len(jax.devices()) // model_axis_size) != 0:
        pytest.skip("This test requires the number of devices to divide 32")

    devices = jax.devices()

    mesh = Mesh(
        np.array(devices).reshape(-1, model_axis_size),
        (ResourceAxis.DATA, ResourceAxis.MODEL),
    )

    with mesh, haliax.axis_mapping({"batch": ResourceAxis.DATA}):
        seq_len = 128
        cache = _small_dataset(seq_len, num_sequences=1007)
        loader = DataLoader(
            cache, schedule, max_buffered_batches=10, mesh=mesh, axis_resources=None, pad_final_batch=True
        )

        for step, batch in enumerate(loader):
            if step < 10:
                assert len(batch) == 8
            elif step < 20:
                assert len(batch) == 16
            else:
                assert len(batch) == 32

        # total steps: 10 * 8 + 10 * 16 = 240, (1007 - 240) // 32 = 23
        assert step == 20 + 23

        # last batch should be padded
        assert len(batch) == 32
        # ensure all the padded examples are all 0's
        num_padding = 32 - (1007 - 240) % 32
        assert np.all(batch[-num_padding:] == 0)
