# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
import functools
import logging
import time
import warnings
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Generic, TypeVar

import haliax.partitioning
import jax
import numpy
from jax import Array
from jax import numpy as jnp
from jax import tree_util as jtu
from jax._src.mesh import get_concrete_mesh
from jax.experimental import multihost_utils
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import PyTree
from optax.tree_utils import tree_zeros_like

import haliax as hax
from haliax import is_named_array
from haliax._src.util import index_where
from haliax.partitioning import ResourceMapping

from levanter.data.dataset import AsyncDataset
from levanter.data.utils import batched
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.schedule import BatchSchedule, IntSchedule
from levanter.shapes import NamedShapeSpec, ShapeSpec, to_raw_shape
from levanter.utils.background_iterable import BackgroundIterator
from levanter.utils.jax_utils import local_cpu_mesh
from levanter.utils.thread_utils import AsyncIteratorWrapper, blocking_wait


Ex = TypeVar("Ex")

_TensorSliceIndex = tuple[slice, ...]
logger = logging.getLogger(__name__)


# NOTE: In general there are a lot of different indices flying around. Here's a quick guide:
# - `step` or `batch_number` or `bn` is the training step or batch number
# - `global` indices refer to the index into the datastore
# - `local` indices refer to the index into the batch. These start at 0 for every batch and go up to the batch size.

# That is, for each batch, we have:
# global_offset = global_offset_at_step(bn)
# batch_size = batch_size_at_step(bn)
# local_data_to_global_map = {i: global_offset + i for i in range(batch_size)}

# But note that we only load the data we need for our local devices, so we only load a subset of the data for each batch.


@dataclass
class _Batch(Generic[Ex]):
    index: int
    global_data_offset: int
    global_size: int
    data_by_local_index: dict[int, Ex]


class DataLoader(Iterable[Ex]):
    def __init__(
        self,
        data: AsyncDataset[Ex],
        batch_size: int | IntSchedule | hax.Axis,
        *,
        batch_axis_name: str | None = None,
        max_buffered_batches: int | None = 64,
        mesh: Mesh | None = None,
        axis_resources: ResourceMapping | None = None,
        prefetch_size: int = 32,
        pad_final_batch: bool = True,
        allow_nondivisible_batch_size: bool = False,
    ):
        """
        Batch- and NamedArray-aware data loader. This class works with an [AsyncDataset][], a Mesh,
        and a resource mapping to load data in a way that is aware of the batch axis and the sharding
        of the data. In general, each machine only loads the data that it needs to process.

        (At the moment, it's not smart enough to load parts of examples if examples are sharded across machines,
        but whole examples are handled correctly.)

        Args:
            batch_size (int | IntSchedule | None): The size of the batch or a schedule for the size of the batch
            data (AsyncDataset[Ex]): The dataset to load from
            max_buffered_batches (Optional[int]): The maximum number of batches to buffer. If None, the buffer is unbounded.
             If <0, the buffer is disabled and single threaded operation is used.
            axis_resources (Optional[ResourceMapping]): axis mapping
            prefetch_size (int): The number of batches to prefetch at once
            mesh (Mesh): The mesh to use
            batch_axis_name (str | None): The name of the batch axis. If None, defaults to "batch" unless batch_size is an Axis.
            pad_final_batch (bool): If True, the final batch will be padded to the size of the previous batch.
            allow_nondivisible_batch_size (bool): All the batch size to be non-divisible by the data axis size (typically the number of devices).
        """
        self.max_buffered_batches = max_buffered_batches
        self.prefetch_size = prefetch_size
        self.axis_resources = axis_resources
        self.data_store = data

        if mesh is None:
            mesh = get_concrete_mesh()
        self.mesh = mesh

        if isinstance(batch_size, hax.Axis):
            assert batch_axis_name is None
            self.batch_axis_name = batch_size.name
            self.scheduler = BatchSchedule(batch_size.size)
        else:
            self.batch_axis_name = batch_axis_name or "batch"
            self.scheduler = BatchSchedule(batch_size)

        self._batch_sharding = hax.partitioning.sharding_for_axis(self.batch_axis_name, axis_resources, mesh)
        with haliax.partitioning.set_mesh(mesh):
            self._data_axis_size = hax.partitioning.physical_axis_size(self.batch_axis_name, axis_resources)

        assert self._data_axis_size is not None, "Data axis size must be known. Make sure you're passing in a mesh"

        self._allow_non_divisible_batch_size = allow_nondivisible_batch_size
        self._pad_final_batch = pad_final_batch

        with local_cpu_mesh():
            # It's important that all data loading happens CPU side. We might relax this one day.
            current_len = blocking_wait(self.data_store.current_len())
            if current_len is not None and current_len <= 0:
                logger.warning("Data store currently has no data. We will block until data is available.")

            initial_example = blocking_wait(self.data_store.getitem_async(0))
            self._ex_leaves, self._ex_structure = jax.tree.flatten(initial_example, is_leaf=is_named_array)
            self._padding_example = _make_padding_example(initial_example)

        if not self._allow_non_divisible_batch_size:
            self._check_batch_size_divisibility()

    def _check_batch_size_divisibility(self):
        if self._data_axis_size is None:
            return
        for size in self.scheduler.unique_batch_sizes():
            if size % self._data_axis_size != 0:
                raise ValueError(
                    f"Batch size {size} is not divisible by data axis size {self._data_axis_size}. "
                    "This will lead to incorrect sharding. Set allow_non_divisible_batch_size=True to allow this."
                )

    def local_data_indices_by_device_for_step(self, step: int) -> dict[jax.Device, range]:
        """

        Returns the 0-based indices of the data that should be loaded by each device for a given step.

        That is, if we are on step i with a fixed batch size of b, then device d should load the data at indices
        [step * bs + (d * local_bs), step * bs + ((d + 1) * local_bs) ) where local_bs is the number of data points
        that device d should load.

        In the case where the batch size is not divisible by the number of devices, the last device(s) will
        have fewer specified indices. The caller is responsible for handling this case.
        """
        batch_size = self.scheduler.batch_size_at_step(step)
        local_indices = self.compute_local_device_indices_for_bs(batch_size)

        return local_indices

    def global_data_indices_by_device_for_step(self, step: int) -> dict[jax.Device, range]:
        local_indices = self.local_data_indices_by_device_for_step(step)
        offset = self.scheduler.global_data_offset_by_step(step)

        return {device: range(offset + r.start, offset + r.stop, r.step) for device, r in local_indices.items()}

    def rounded_batch_size_at_step(self, step: int) -> int:
        return self._round_batch_size(self.scheduler.batch_size_at_step(step))

    def batch_axis_at_step(self, step: int):
        size = self.rounded_batch_size_at_step(step)

        return hax.Axis(self.batch_axis_name, size)

    def _round_batch_size(self, size: int) -> int:
        if self._data_axis_size is None:
            return size

        if size % self._data_axis_size == 0:
            return size

        if not self._allow_non_divisible_batch_size and not self._pad_final_batch:
            raise ValueError(f"Batch size {size} is not divisible by data axis size {self._data_axis_size}")

        out = _round_to_nearest_multiple(size, self._data_axis_size)

        warnings.warn(
            f"Padding batch size {size} to {out} to be divisible by data axis size {self._data_axis_size}."
            f"\n This results in a per-device batch size of {out // self._data_axis_size}. (Extra data is zeros.)"
        )

        return out

    @lru_cache
    def compute_local_device_indices_for_bs(self, bs) -> dict[jax.Device, range]:
        """Return the local indices for each device for a given batch size and sharding."""
        rounded_bs = self._round_batch_size(bs)
        local_indices_map = self._batch_sharding.addressable_devices_indices_map((rounded_bs,))
        # we just want all the indices
        local_device_indices: dict[jax.Device, range] = {
            device: range(*idx[0].indices(rounded_bs)) for device, idx in local_indices_map.items() if idx is not None
        }
        local_device_indices_clipped = {
            device: range(max(0, r.start), min(rounded_bs, r.stop)) for device, r in local_device_indices.items()
        }
        return local_device_indices_clipped

    def __iter__(self):
        return self.iter_from_step(None)

    def iter_from_step(self, start_from_batch: int | None = None):
        # sometimes we pass in an array for the start_from_batch, so we need to check for that
        start_from_batch = int(start_from_batch) if start_from_batch is not None else None
        return DataLoaderIterator(self, start_from_batch=start_from_batch)

    def has_len(self):
        return self.data_store.is_finite()

    def __len__(self):
        if not self.has_len():
            raise ValueError("DataLoader has no length")
        total_length = blocking_wait(self.data_store.current_len())
        step = self.scheduler.find_step_containing_offset(total_length) + 1
        return step

    def reversed(
        self,
        num_train_steps: int | None = None,
        segment_starts: Iterable[int] | None = None,
    ):
        """Return a reversed-order loader.

        If ``segment_starts`` is provided, returns a segment-aware reversed loader
        that iterates segments in reverse and supports ``iter_segment``.
        Otherwise returns the basic reversed loader.
        """
        if segment_starts is not None:
            return ReversedSegmentedDataLoader(self, segment_starts, num_train_steps=num_train_steps)
        return ReversedDataLoader(self, num_train_steps=num_train_steps)

    def reversed_segmented(
        self,
        segment_starts: Iterable[int],
        num_train_steps: int | None = None,
    ):
        """Segment-aware reverse traversal.

        Returns a dataloader that iterates over batches by reversing the order of
        segments while iterating forward within each segment. If
        ``segment_starts = [0, s1, ..., sk]`` (sorted, increasing), and the
        forward order is batches ``[0, 1, ..., N-1]``, then iteration order will be
        ``[sk, sk+1, ..., N-1, s{k-1}, ..., sk-1, ..., 0, ..., s1-1]``.

        Args:
            segment_starts: Sorted increasing list of segment start batch numbers.
                             0 will be added if not present.
            num_train_steps: Optional cap on the number of batches to iterate.
        """
        return ReversedSegmentedDataLoader(self, segment_starts, num_train_steps=num_train_steps)


class DataLoaderIterator(Iterator[Ex]):
    def __init__(self, data_loader: DataLoader, start_from_batch: int | None = None):
        self.dl = data_loader
        self._start_from_batch = start_from_batch
        self.mapping = self.dl.axis_resources
        if self.mapping is None:
            self.mapping = hax.partitioning.current_thread_local_mapping()

        buffered_batches = self.dl.max_buffered_batches
        self._batches: Iterator[Ex]
        if buffered_batches == 0:
            self._batches = AsyncIteratorWrapper(self._produce_batches())
        else:
            self._batches = _JaxCpuBackgroundIterator(self._produce_batches, max_capacity=buffered_batches)

    def __next__(self):
        time_start = time.time()
        batch = next(self._batches)
        time_mid = time.time()

        time_end = time.time()
        time_batch = time_end - time_mid
        if (time_end - time_start) > 0.5:
            if time_batch > 0.1:
                logger.info(f"Prefetch wasn't fast enough: {time_end - time_start:.3f}. {time_batch:.3f} in batchify")
            else:
                logger.info(f"Prefetch wasn't fast enough: {time_end - time_start:.3f}.")
        return batch

    def __del__(self):
        if hasattr(self, "_batches") and hasattr(self._batches, "stop"):
            self._batches.stop()

    async def _produce_batches(self):
        with local_cpu_mesh():
            batch_number = self._start_from_batch or 0
            done = False
            while not done:
                # we try to prefetch multiple batches at a time
                target_next_batch_number = batch_number + self.dl.prefetch_size
                max_achievable_batch_number, final_batch_size = await self._dataset_get_available_batch_number(
                    target_next_batch_number
                )

                assert batch_number <= max_achievable_batch_number <= target_next_batch_number

                if max_achievable_batch_number < target_next_batch_number:
                    done = True

                next_batch_numbers = list(
                    range(batch_number, min(target_next_batch_number, max_achievable_batch_number))
                )

                if len(next_batch_numbers) == 0:
                    logger.debug(f"Breaking because no more data available at batch number {batch_number}")
                    break

                batches = [
                    _Batch(
                        bn,
                        self.dl.scheduler.global_data_offset_by_step(bn),
                        self.dl.scheduler.batch_size_at_step(bn),
                        {},
                    )
                    for bn in next_batch_numbers
                ]

                if final_batch_size is not None:
                    batches[-1] = dataclasses.replace(batches[-1], global_size=final_batch_size)

                batch_of_batches: list[_Batch[Ex]] = await self._do_retrieve_batch_of_batches(batches)

                for batch in batch_of_batches:
                    batch = self._batchify_local_data(batch)
                    yield batch

                batch_number = next_batch_numbers[-1] + 1

            logger.debug(f"DataLoaderIterator finished at batch number {batch_number}")

    async def _dataset_get_available_batch_number(self, target_max_batch_number: int) -> tuple[int, int | None]:
        """
        Wait until the data store has enough data to support the given batch number. If
        the data store is finite, this will wait until the data store has at least `target_max_batch_number` batches
        or until the data store is exhausted, in which case it will return the last batch number that the data store
        has data for.

        Returns:
            int: The batch number that the data store has data for.
            int | None: The size of the final batch if the final batch is partial and needs to be padded.
        """
        if self.dl.data_store.is_finite():
            next_end = self.dl.scheduler.global_data_offset_by_step(target_max_batch_number)
            available_len = await self.dl.data_store.wait_until_len_at_least(next_end)

            at_the_end = available_len < next_end

            if available_len < next_end:
                target_max_batch_number = self.dl.scheduler.find_step_containing_offset(available_len)
                next_end = self.dl.scheduler.global_data_offset_by_step(target_max_batch_number)
                logger.debug(f"Data store exhausted after {target_max_batch_number} batches.")

            # if we are padding the final batch, we want to see if there is data past the end of the last batch
            if at_the_end and self.dl._pad_final_batch:
                if available_len > next_end:
                    partial_batch_size = available_len - next_end
                    logger.debug(f"Partial batch size: {partial_batch_size}")
                    return target_max_batch_number + 1, partial_batch_size
                else:
                    # exact match
                    return target_max_batch_number, None

        return target_max_batch_number, None

    def _batchify_local_data(self, batch: _Batch[Ex]) -> Ex:
        """
        Stacks the individual examples (pytrees) into a single example (pytree) with the batch axis added
        and creates a global array for each leaf of the example.
        """
        cache: dict[tuple[int, int], list[Array | hax.NamedArray]] = {}
        padded_batch_size = self.dl.rounded_batch_size_at_step(batch.index)
        Batch = hax.Axis(self.dl.batch_axis_name, padded_batch_size)

        def get_local_batch(begin: int, end: int) -> list:
            if (begin, end) in cache:
                return cache[(begin, end)]

            local_data = []
            for i in range(begin, end):
                try:
                    local_data.append(batch.data_by_local_index[i])
                except KeyError:
                    assert self.dl._allow_non_divisible_batch_size or self.dl._pad_final_batch
                    local_data.append(self.dl._padding_example)

            # TODO: if we ever do "big data" (i.e. huge examples) we might want to be able to load part of an example
            # which will require support from the datastore (i.e. tensorstore)
            device_batch = stack_tree(self.dl.batch_axis_name, local_data)
            batch_leaves = hax.tree_util.tree_leaves(device_batch)

            cache[(begin, end)] = batch_leaves

            return batch_leaves

        def get_local_data_for_leaf(indices: _TensorSliceIndex, leaf_index: int) -> Array:
            batch_slice = indices[0]
            begin, end, stride = batch_slice.indices(padded_batch_size)
            if stride != 1:
                raise ValueError("Stride must be 1")

            leaf_data = get_local_batch(begin, end)[leaf_index]

            if isinstance(leaf_data, hax.NamedArray):
                # select out the batch axis
                batch_index = index_where(lambda ax: ax.name == Batch.name, leaf_data.axes)
                new_indices = list(indices)
                new_indices[batch_index] = slice(None)
                return leaf_data.array[tuple(new_indices)]
            else:
                other_indices = indices[1:]
                if all(idx == slice(None) for idx in other_indices):
                    return leaf_data
                else:
                    return leaf_data[(..., *other_indices)]

        def make_global_array_for_leaf(leaf_index, item_leaf_shape: ShapeSpec | NamedShapeSpec):
            def get_data(indices):
                return get_local_data_for_leaf(indices, leaf_index)

            raw_array = jax.make_array_from_callback(
                to_raw_shape(item_leaf_shape),
                jax.sharding.NamedSharding(self.dl.mesh, self._pspec_for(item_leaf_shape)),
                get_data,
            )
            if isinstance(item_leaf_shape, NamedShapeSpec):
                return hax.NamedArray(raw_array, item_leaf_shape.shape)
            else:
                return raw_array

        gda_leaves = [
            make_global_array_for_leaf(leaf_index, _batchified_shape(Batch, item_leaf))
            for leaf_index, item_leaf in enumerate(self.dl._ex_leaves)
        ]
        gda_tree = jax.tree.unflatten(self.dl._ex_structure, gda_leaves)
        return gda_tree

    async def _do_retrieve_batch_of_batches(self, batch_specs: list[_Batch[None]]) -> list[_Batch[Ex]]:
        """
        Retrieve the data for a batch of batches.

        - For each batch number, we get the mapping from our local devices to the indices in the data store that we need
          to load.
        - We then load the data for all the devices for all the batch numbers.
        - we then return each batch as a dictionary from device to the data for that device.
        """
        global_indices_for_each_batch = []

        for batch in batch_specs:
            global_offset = batch.global_data_offset
            local_indices_for_device = self.dl.local_data_indices_by_device_for_step(batch.index)

            distinct_local_indices_this_batch = set()
            for indices in local_indices_for_device.values():
                for local_index in indices:
                    if local_index >= batch.global_size:
                        assert self.dl._pad_final_batch or self.dl._allow_non_divisible_batch_size
                        continue

                    distinct_local_indices_this_batch.add(local_index)

            global_indices_for_this_batch = [global_offset + i for i in distinct_local_indices_this_batch]
            global_indices_for_each_batch.append(global_indices_for_this_batch)

        # flattened view so we can load all the data at once
        indices_for_this_batch_of_batches: list[int] = [
            i for indices in global_indices_for_each_batch for i in indices
        ]
        individual_datums = await self.run_and_report_slowness(
            self.dl.data_store.get_batch(indices_for_this_batch_of_batches),
            f"Waiting for {len(indices_for_this_batch_of_batches)} items.",
        )

        # unflatten
        global_map: dict[int, Ex] = {}
        for i, datum in enumerate(individual_datums):
            global_map[indices_for_this_batch_of_batches[i]] = datum

        out: list[_Batch[Ex]] = []

        for batch, global_indices_batch in zip(batch_specs, global_indices_for_each_batch, strict=False):
            local_index_to_example = {}
            for global_index in global_indices_batch:
                local_index = global_index - batch.global_data_offset
                local_index_to_example[local_index] = global_map[global_index]

            out.append(dataclasses.replace(batch, data_by_local_index=local_index_to_example))  # type: ignore

        return out

    def _pspec_for(self, shape_spec: ShapeSpec | NamedShapeSpec) -> PartitionSpec:
        if isinstance(shape_spec, ShapeSpec):  # type: ignore
            batch_name = hax.partitioning.physical_axis_name(self.dl.batch_axis_name, self.dl.axis_resources)
            return PartitionSpec(batch_name, *((None,) * (len(shape_spec.shape) - 1)))
        else:
            return hax.partitioning.pspec_for_axis(shape_spec.shape, self.dl.axis_resources)  # type: ignore

    async def run_and_report_slowness(self, coro, description: str):
        threshold = 10.0
        task = asyncio.create_task(coro)

        async def watchdog():
            total = 0.0

            while not task.done():
                await asyncio.sleep(threshold)
                total += threshold
                if not task.done():
                    logging.warning(f"Data loading is taking a long time: {total:.1f} seconds. {description}")

        watchdog_task = asyncio.create_task(watchdog())

        try:
            result = await task
            return result
        finally:
            watchdog_task.cancel()


def _make_dummy_instance(batch, Pos):
    """
    Creates a dummy instance matching the shape of the provided batch.
    If the dataset is exhausted and a full batch is needed, this function returns a dummy instance
    with all elements set to zero and a segment mask filled with -1. This design ensures that the dummy
    instance does not contribute to the loss during training.
    """
    dummy_instance: LmExample = tree_zeros_like(batch[0])
    dummy_segment_mask = hax.full(Pos, -1, dtype=jnp.int32)
    dummy_attn = AttentionMask.causal().with_segment_ids(dummy_segment_mask)
    dummy_instance = dataclasses.replace(dummy_instance, attn_mask=dummy_attn)
    return dummy_instance


def stack_batches(example_iterator, Pos, Batch):
    """
    Stack examples from an iterator into a batch.

    Args:
        Batch: The batch axis.
        Pos: The position axis.
        example_iterator: An iterator of examples.

    Returns:
        A batch of examples.
    """
    # add timer here as well and profile
    with local_cpu_mesh():
        for batch in batched(example_iterator, Batch.size):
            if len(batch) < Batch.size:
                dummy_instance = _make_dummy_instance(batch, Pos)
                batch.extend([dummy_instance] * (Batch.size - len(batch)))
            yield stack_tree(Batch, batch)


def _batchified_shape(Batch, leaf: hax.NamedArray | Array) -> ShapeSpec | NamedShapeSpec:
    if is_named_array(leaf):
        return NamedShapeSpec((Batch,) + leaf.axes, leaf.dtype)
    else:
        return ShapeSpec((Batch.size,) + tuple(leaf.shape), leaf.dtype)


class _JaxCpuBackgroundIterator(BackgroundIterator[Ex]):
    """
    We want the thread to only use the CPU device.
    """

    def __init__(self, producer_fn: Callable[[], Iterator[Ex] | AsyncIterator[Ex]], max_capacity: int | None):
        super().__init__(producer_fn, max_capacity)

    def _fill_queue_with_batches(self):
        with local_cpu_mesh():
            super()._fill_queue_with_batches()


# @equinox.filter_jit
@functools.partial(jax.jit, static_argnums=(0,))
def stack_tree(batch_name, individual_datums):
    def _stack_leaves_unchecked(*leaves):
        if is_named_array(leaves[0]):
            return hax.stack(batch_name, leaves)
        else:
            return jnp.stack(leaves)

    return jax.tree.map(_stack_leaves_unchecked, *individual_datums, is_leaf=is_named_array)


def check_sharded_consistency(tree: PyTree, check_disjoint_indices_are_different: bool = False):
    """Checks the following consistency conditions on an array:
    - all replicas have the same data
    - if check_disjoint_indices_are_different is True, then all shards with disjoint indices have different data
    """
    # index is a tuple[slice, ...], slices are obnoxiously not hashable so we have to convert to tuple

    def check_array(array):
        def _to_tuple(index: tuple[slice, ...]) -> tuple[tuple[int, int], ...]:
            my_indices: tuple[tuple[int, int], ...] = tuple(
                s.indices(axis_size)[0:2] for axis_size, s in zip(array.shape, index, strict=False)
            )

            return my_indices

        replicas_by_index = defaultdict(list)
        for shard in array.global_shards:
            replicas_by_index[_to_tuple(shard.index)].append(shard)

        # global shards is not necessarily sorted consistently, so we have to sort the indices
        sorted_indices = sorted(replicas_by_index.keys())

        # ok now get canonical versions of each index
        replica_0_arrays = {}

        for index in sorted_indices:
            shards = replicas_by_index[index]
            try:
                leader = next(s for s in shards if s.replica_id == 0)
            except StopIteration:
                raise ValueError("No replica 0 for index", index)

            data = leader.data
            if data is None:
                shard_shape = [s[1] - s[0] for s in index]
                data = jnp.zeros(shard_shape, dtype=array.dtype)

            replica_0_array = multihost_utils.broadcast_one_to_all(data, is_source=leader.data is not None)
            replica_0_arrays[index] = replica_0_array

        for shard in array.addressable_shards:
            replica_0_array = replica_0_arrays[_to_tuple(shard.index)]
            assert shard.data is not None

            if not numpy.array_equal(shard.data, replica_0_array, equal_nan=True):
                raise ValueError("Shard data does not match replica 0 data", shard, replica_0_array)

            if check_disjoint_indices_are_different:
                for other_index, other_array in replica_0_arrays.items():
                    if other_index == _to_tuple(shard.index):
                        continue

                    if shard.index != other_index:
                        if jnp.array_equal(shard.data, other_array, equal_nan=True):
                            raise ValueError(
                                "Shard data is the same as another shard with disjoint indices", shard, other_array
                            )

    for leaf in jtu.tree_leaves(tree):
        check_array(leaf)


def _make_padding_example(ex: Ex) -> Ex:
    with local_cpu_mesh():
        return tree_zeros_like(ex)


def _round_to_nearest_multiple(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


class ReversedDataLoader(Iterable[Ex]):
    """A thin wrapper that exposes the same interface as :class:DataLoader but iterates in reverse batch order."""

    def __init__(self, dl: DataLoader, num_train_steps: int | None = None):
        self._dl = dl
        self._num_train_steps = num_train_steps

    # ---------------- Iterable interface ----------------
    def __iter__(self):
        return _ReversedDataLoaderIterator(self._dl, num_train_steps=self._num_train_steps)

    # --------------- convenience passthroughs ------------
    def __getattr__(self, item):
        """Delegate attribute access to the wrapped DataLoader for convenience (read-only)."""
        return getattr(self._dl, item)

    # Helpful hint for callers that need per-segment iteration
    def iter_segment(self, segment_start: int):  # pragma: no cover - simple runtime hint
        raise AttributeError(
            "iter_segment is only available on ReversedSegmentedDataLoader. "
            "Construct via DataLoader.reversed_segmented(segment_starts, num_train_steps)."
        )

    def get_segment_loader(self, segment_start: int):  # pragma: no cover - simple runtime hint
        return self.iter_segment(segment_start)


class _ReversedDataLoaderIterator(DataLoaderIterator):
    """
    Iterator that yields batches in the opposite order compared to
    :class:`DataLoaderIterator`, but only over the first `num_train_steps`
    batches if that argument is supplied.
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        data_loader: DataLoader,
        *,
        num_train_steps: int | None = None,   # ← NEW
    ):
        # -------------------------------------------------------------
        # 1.  Work out the last batch index we are allowed to visit.
        #     If the underlying dataset is infinite, require
        #     `num_train_steps` and use it to bound iteration.
        # -------------------------------------------------------------
        if not data_loader.data_store.is_finite():
            if num_train_steps is None:
                raise ValueError(
                    "ReversedDataLoader requires a finite dataset with a known length, or provide num_train_steps."
                )
            if num_train_steps <= 0:
                raise ValueError("num_train_steps must be > 0 for reversed iteration.")

            # Bound by the requested training horizon
            self._last_batch_number = int(num_train_steps) - 1
            # For infinite datasets, treat the final batch as full-sized
            self._final_batch_start = data_loader.scheduler.global_data_offset_by_step(self._last_batch_number)
            self._final_batch_expected_size = data_loader.scheduler.batch_size_at_step(self._last_batch_number)
            self._final_batch_size = self._final_batch_expected_size
            self._num_train_steps = num_train_steps

            super().__init__(data_loader, start_from_batch=None)  # type: ignore[arg-type]
            return

        total_len = blocking_wait(data_loader.data_store.async_len())

        # “Natural” last batch in the dataset.
        last_bn_dataset = 0
        while data_loader.scheduler.global_data_offset_by_step(last_bn_dataset) < total_len:
            last_bn_dataset += 1
        last_bn_dataset -= 1
        if last_bn_dataset < 0:
            raise ValueError("Dataset appears to be empty – nothing to iterate over in reverse.")

        # If the caller limits us, respect it (but never exceed the dataset’s end).
        if num_train_steps is not None:
            self._last_batch_number = min(num_train_steps - 1, last_bn_dataset)
        else:
            self._last_batch_number = last_bn_dataset

        # Figure out sizing info for that *new* last batch.
        self._final_batch_start = data_loader.scheduler.global_data_offset_by_step(
            self._last_batch_number
        )
        self._final_batch_expected_size = data_loader.scheduler.batch_size_at_step(
            self._last_batch_number
        )
        self._final_batch_size = total_len - self._final_batch_start
        if self._final_batch_size == 0:
            # Perfectly divisible – treat as a full batch.
            self._final_batch_size = self._final_batch_expected_size

        # Remember the limit so we know when to stop in `_produce_batches`.
        self._num_train_steps = num_train_steps

        # -------------------------------------------------------------
        # 2.  Now it’s safe to invoke the parent, which launches the
        #     background pre-fetch thread that calls `_produce_batches`.
        # -------------------------------------------------------------
        super().__init__(data_loader, start_from_batch=None)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Core producer
    # ------------------------------------------------------------------
    async def _produce_batches(self):  # type: ignore[override]
        bn = self._last_batch_number
        prefetch = self.dl.prefetch_size

        # How many batches we’ve yielded so far (needed if we are limited).
        yielded = 0
        to_yield_total = (
            self._num_train_steps if self._num_train_steps is not None else None
        )

        while bn >= 0 and (to_yield_total is None or yielded < to_yield_total):
            lower_bn = max(0, bn - prefetch + 1)
            next_batch_numbers = list(range(lower_bn, bn + 1))[::-1]

            # Trim window so we don’t overshoot `num_train_steps`.
            if to_yield_total is not None:
                remaining = to_yield_total - yielded
                next_batch_numbers = next_batch_numbers[:remaining]

            batches: list[_Batch[None]] = []
            for batch_number in next_batch_numbers:
                global_offset = self.dl.scheduler.global_data_offset_by_step(batch_number)
                global_size = self.dl.scheduler.batch_size_at_step(batch_number)
                if (
                    batch_number == self._last_batch_number
                    and self._final_batch_size != global_size
                ):
                    global_size = self._final_batch_size
                batches.append(_Batch(batch_number, global_offset, global_size, {}))

            batch_of_batches = await self._do_retrieve_batch_of_batches(batches)

            for batch in batch_of_batches:
                yield self._batchify_local_data(batch)
                yielded += 1

            # Move cursor to the preceding window.
            bn = lower_bn - 1


class ReversedSegmentedDataLoader(Iterable[Ex]):
    """Wrapper mirroring :class:`DataLoader` but iterating with reversed segments.

    Within each segment, batch order is forward; the order of segments is reversed.
    Accepts the same convenience attribute accessors as :class:`ReversedDataLoader`.
    """

    def __init__(
        self,
        dl: DataLoader,
        segment_starts: Iterable[int],
        num_train_steps: int | None = None,
    ):
        self._dl = dl
        self._segment_starts = list(segment_starts)
        self._num_train_steps = num_train_steps

    def __iter__(self):
        return _ReversedSegmentedDataLoaderIterator(
            self._dl,
            segment_starts=self._segment_starts,
            num_train_steps=self._num_train_steps,
        )

    def __getattr__(self, item):
        return getattr(self._dl, item)

    def iter_segment(self, segment_start: int):
        """Return an iterator over a single forward-ordered segment starting at ``segment_start``.

        The iterator yields batches ``segment_start, segment_start+1, ...`` up to (but not including)
        the next ``segment_start`` in the original list, or up to the final batch if this is the last
        segment within the available training range.
        """
        return _SegmentOnlyDataLoaderIterator(
            self._dl,
            segment_start=segment_start,
            segment_starts=self._segment_starts,
            num_train_steps=self._num_train_steps,
        )

    # Backward-compatible alias if callers prefer this name
    def get_segment_loader(self, segment_start: int):
        return self.iter_segment(segment_start)


class _ReversedSegmentedDataLoaderIterator(DataLoaderIterator):
    """Iterator that reverses segments but iterates forward within each segment.

    If the forward batch order is ``[0, 1, ..., N-1]`` and
    ``segment_starts = [0, s1, ..., sk]`` (sorted, increasing), then iteration
    order is ``sk, sk+1, ..., N-1, s{k-1}, ..., sk-1, ..., 0, ..., s1-1``.
    Optionally limits to the first ``num_train_steps`` batches.
    """

    def __init__(
        self,
        data_loader: DataLoader,
        *,
        segment_starts: Iterable[int],
        num_train_steps: int | None = None,
    ):
        if not data_loader.data_store.is_finite():
            if num_train_steps is None:
                raise ValueError(
                    "ReversedSegmentedDataLoader requires a finite dataset or num_train_steps to bound iteration."
                )
            if num_train_steps <= 0:
                raise ValueError("num_train_steps must be > 0 for reversed segmented iteration.")

            self._last_batch_number = int(num_train_steps) - 1
            self._final_batch_start = data_loader.scheduler.global_data_offset_by_step(self._last_batch_number)
            self._final_batch_expected_size = data_loader.scheduler.batch_size_at_step(self._last_batch_number)
            self._final_batch_size = self._final_batch_expected_size
        else:
            total_len = blocking_wait(data_loader.data_store.async_len())

            # Determine the last reachable batch index from dataset size
            last_bn_dataset = 0
            while data_loader.scheduler.global_data_offset_by_step(last_bn_dataset) < total_len:
                last_bn_dataset += 1
            last_bn_dataset -= 1
            if last_bn_dataset < 0:
                raise ValueError("Dataset appears to be empty – nothing to iterate over in reverse-segmented order.")

            if num_train_steps is not None:
                self._last_batch_number = min(num_train_steps - 1, last_bn_dataset)
            else:
                self._last_batch_number = last_bn_dataset

            # Final batch size (handle partial at dataset end)
            self._final_batch_start = data_loader.scheduler.global_data_offset_by_step(self._last_batch_number)
            self._final_batch_expected_size = data_loader.scheduler.batch_size_at_step(self._last_batch_number)
            self._final_batch_size = total_len - self._final_batch_start
            if self._final_batch_size == 0:
                self._final_batch_size = self._final_batch_expected_size

        # Normalize and store segment starts for iteration planning
        starts = sorted(set(int(s) for s in segment_starts if int(s) >= 0))
        if not starts or starts[0] != 0:
            starts = [0] + starts
            starts = sorted(set(starts))

        # Clip to the last batch we're going to visit
        max_bn = self._last_batch_number
        starts = [s for s in starts if s <= max_bn]
        if not starts:
            starts = [0]

        # Build [start, end) segments, where the final end is max_bn + 1
        ends_exclusive = starts[1:] + [max_bn + 1]
        self._segments: list[tuple[int, int]] = [
            (s, e) for s, e in zip(starts, ends_exclusive, strict=False) if s < e
        ]

        self._num_train_steps = num_train_steps

        # Launch background machinery; it will call our _produce_batches
        super().__init__(data_loader, start_from_batch=None)  # type: ignore[arg-type]

    async def _produce_batches(self):  # type: ignore[override]
        # Precompute the exact batch-number sequence: reverse the segments,
        # forward within each segment.
        batch_numbers: list[int] = []
        for start, end in reversed(self._segments):
            batch_numbers.extend(range(start, end))

        # If for any reason a num_train_steps smaller than the computed
        # range is supplied, respect it.
        if self._num_train_steps is not None:
            batch_numbers = batch_numbers[: self._num_train_steps]

        prefetch = self.dl.prefetch_size
        i = 0
        N = len(batch_numbers)
        while i < N:
            window = batch_numbers[i : min(i + prefetch, N)]

            batches: list[_Batch[None]] = []
            for bn in window:
                global_offset = self.dl.scheduler.global_data_offset_by_step(bn)
                global_size = self.dl.scheduler.batch_size_at_step(bn)
                if bn == self._last_batch_number and self._final_batch_size != global_size:
                    global_size = self._final_batch_size
                batches.append(_Batch(bn, global_offset, global_size, {}))

            batch_of_batches = await self._do_retrieve_batch_of_batches(batches)

            for batch in batch_of_batches:
                yield self._batchify_local_data(batch)

            i += len(window)


class _SegmentOnlyDataLoaderIterator(DataLoaderIterator):
    """Iterates forward only within a single segment [start, next_start) or to the last batch.

    Handles partial final batch sizing if this segment contains the dataset's final batch.
    """

    def __init__(
        self,
        data_loader: DataLoader,
        *,
        segment_start: int,
        segment_starts: Iterable[int],
        num_train_steps: int | None = None,
    ):

        # If an explicit training-step bound is provided, avoid waiting on dataset length (which would
        # call cache.finished()). Bound iteration by the training schedule instead.
        if num_train_steps is not None:
            if num_train_steps <= 0:
                raise ValueError("num_train_steps must be > 0 for segment iteration.")

            self._max_bn = int(num_train_steps) - 1
            # Without dataset length, treat the final batch as full-sized
            expected_size = data_loader.scheduler.batch_size_at_step(self._max_bn)
            self._partial_last_bn = self._max_bn
            self._partial_last_bn_size = expected_size
        else:
            if not data_loader.data_store.is_finite():
                raise ValueError(
                    "Segment iterator requires a finite dataset or num_train_steps to bound iteration."
                )

            total_len = blocking_wait(data_loader.data_store.async_len())

            # Compute the last permissible batch number given dataset length
            last_bn_dataset = 0
            while data_loader.scheduler.global_data_offset_by_step(last_bn_dataset) < total_len:
                last_bn_dataset += 1
            last_bn_dataset -= 1
            if last_bn_dataset < 0:
                raise ValueError("Dataset appears to be empty – nothing to iterate.")

            self._max_bn = last_bn_dataset

            # Partial final batch sizing for the true last batch within training range
            final_bn_start = data_loader.scheduler.global_data_offset_by_step(self._max_bn)
            expected_size = data_loader.scheduler.batch_size_at_step(self._max_bn)
            partial_size = total_len - final_bn_start
            if partial_size == 0:
                partial_size = expected_size
            self._partial_last_bn = self._max_bn
            self._partial_last_bn_size = partial_size
        # Normalize provided segment starts, ensure 0 present, then clip to range
        starts = sorted(set(int(s) for s in segment_starts if int(s) >= 0))
        if not starts or starts[0] != 0:
            starts = sorted(set([0, *starts]))
        starts = [s for s in starts if s <= self._max_bn]
        if not starts:
            starts = [0]

        if segment_start not in starts:
            raise ValueError(f"segment_start {segment_start} not found in segment_starts {starts}")

        idx = starts.index(segment_start)
        end_exclusive = starts[idx + 1] if idx + 1 < len(starts) else (self._max_bn + 1)

        if segment_start >= end_exclusive:
            raise ValueError(f"Empty segment derived for start {segment_start}")

        self._range_start = segment_start
        self._range_end_exclusive = end_exclusive

        super().__init__(data_loader, start_from_batch=None)  # type: ignore[arg-type]

    async def _produce_batches(self):  # type: ignore[override]
        prefetch = self.dl.prefetch_size
        i = self._range_start
        N = self._range_end_exclusive
        while i < N:
            window = list(range(i, min(i + prefetch, N)))

            batches: list[_Batch[None]] = []
            for bn in window:
                global_offset = self.dl.scheduler.global_data_offset_by_step(bn)
                global_size = self.dl.scheduler.batch_size_at_step(bn)
                if bn == self._partial_last_bn and self._partial_last_bn_size != global_size:
                    global_size = self._partial_last_bn_size
                batches.append(_Batch(bn, global_offset, global_size, {}))

            batch_of_batches = await self._do_retrieve_batch_of_batches(batches)
            for batch in batch_of_batches:
                yield self._batchify_local_data(batch)

            i += len(window)
