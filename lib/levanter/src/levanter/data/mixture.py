# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import warnings
from typing import List, Mapping, Optional, Sequence, Tuple, TypeVar

import jax
import numpy as np
from async_lru import alru_cache
from jax.random import PRNGKey
from jaxtyping import PRNGKeyArray

from haliax.util import StringHolderEnum

from levanter.data import AsyncDataset
from levanter.schedule import BatchSchedule
from levanter.utils.index import Index
from levanter.utils.thread_utils import future_from_value


T = TypeVar("T")


class StopStrategy(metaclass=StringHolderEnum):
    FIRST_STOP_STRATEGY = "first_exhausted"
    ALL_STOP_STRATEGY = "all_exhausted"
    RESTART_STRATEGY = "restart"


class MixtureDataset(AsyncDataset[T]):
    """
    MixtureDataset supports loading data from multiple datasets. It takes a list of datasets and yields from them
    according to the weights.

    Creating a random-access MixtureDataset is challenging because we need to keep track of the current index of each
    dataset. To solve this, we instead use "block-deterministic" mixtures, where the number of samples from each dataset
    in each block is always identical (and we shuffle the order of the dataset ids in each block). To handle the case where the dataset mixture changes over time, we use a list of stages and precompute statistics to accurately compute the index of each dataset in each block.

    Args:
        datasets: A dict of datasets, where the key is the name of the dataset and the value is the dataset itself
        weights: Weights for each dataset. This can be provided in a list of stages, where each stage is a tuple of (start_seq_index, weights). Note that start_seq_index corresponds to the sequence index at which the weights should change, not the training batch index.
        stop_strategy: strategy for stopping the iteration, by default RESTART_STRATEGY. (Currently only RESTART_STRATEGY is supported)
            - FIRST_STOP_STRATEGY: stop when one dataset has been exhausted
            - ALL_STOP_STRATEGY: stop when all datasets have been exhausted
            - RESTART_STRATEGY: restart the dataset when it has been exhausted
        key: random key for datasets sampling
    """

    def __init__(
        self,
        datasets: Mapping[str, AsyncDataset[T]],
        weights: dict[str, float] | List[Tuple[int, dict[str, float]]],
        block_size: int,
        *,
        randomize_blocks: bool = True,
        key: PRNGKeyArray | int,
        stop_strategy: str = StopStrategy.RESTART_STRATEGY,
    ):
        super().__init__()
        if isinstance(weights, dict):
            weight_stages = [(0, weights)]
        else:
            weight_stages = weights

        if len(datasets) == 0:
            raise ValueError("Datasets cannot be empty")

        # assert that steps are in sorted order and that the start index of each stage is a multiple of block_size
        for i, (start_seq_index, _) in enumerate(weight_stages):
            if i == 0:
                assert start_seq_index == 0
            else:
                assert start_seq_index % block_size == 0, (
                    f"start_seq_index for a stage must be a multiple of block_size, got {start_seq_index=} and"
                    f" {block_size=}"
                )
                assert start_seq_index > weight_stages[i - 1][0], f"Weights list must be sorted, got {weight_stages}"

        self.weight_stages = [
            (start_seq_index, self._normalize_weights(weights)) for start_seq_index, weights in weight_stages
        ]
        self.datasets = {
            name: dataset
            for name, dataset in datasets.items()
            if any(weights.get(name, 0) > 0 for _, weights in self.weight_stages)
        }
        self.dataset_index = Index(self.datasets.keys())
        self.block_size = block_size
        # we pack index and ds id into a single 32 bit, so block size must be at most 2^16
        if block_size >= 2**16:
            raise ValueError(f"Block size must be at most 2^16, got {block_size}")

        self.randomize_blocks = randomize_blocks

        if isinstance(key, int):
            key = PRNGKey(key)

        self.key = key

        if stop_strategy not in StopStrategy:  # type: ignore
            raise ValueError(f"Stop strategy {stop_strategy} is not supported.")

        # for now, just support restart strategy
        if stop_strategy != StopStrategy.RESTART_STRATEGY:
            raise NotImplementedError("Only restart strategy is supported for now.")

        self.stop_strategy = stop_strategy

        # Initialize stage-related counts and IDs
        (
            self._counts_per_block_per_stage,
            self._counts_after_stage,
            self._unpermuted_ids_per_stage,
        ) = self._initialize_stage_counts()

    def _initialize_stage_counts(self):
        counts_per_block_per_stage = []
        counts_after_stage = []
        unpermuted_ids_per_stage = []

        cumulative_counts = np.zeros(len(self.datasets), dtype=np.int32)

        for stage_idx, (start_seq_index, stage_weights) in enumerate(self.weight_stages):
            counts_this_stage = self._compute_expected_counts_per_block(stage_weights, self.block_size)
            counts_per_block_per_stage.append(counts_this_stage)
            unpermuted_ids_per_stage.append(self._compute_unpermuted_ids(counts_this_stage))

            if stage_idx < len(self.weight_stages) - 1:
                next_start = self.weight_stages[stage_idx + 1][0]
                num_blocks_in_stage = (next_start - start_seq_index) // self.block_size
                stage_total_counts = counts_this_stage * num_blocks_in_stage
                cumulative_counts += stage_total_counts
                counts_after_stage.append(cumulative_counts.copy())

        return counts_per_block_per_stage, counts_after_stage, unpermuted_ids_per_stage

    def _compute_expected_counts_per_block(self, weights: dict[str, float], block_size: int):
        _expected_values_per_block = np.zeros(len(self.datasets), dtype=np.int32)
        for i, dsname in enumerate(self.dataset_index):
            _expected_values_per_block[i] = weights.get(dsname, 0) * block_size

        # handle remainder by adding to the largest dataset
        largest_dataset = np.argmax(_expected_values_per_block)
        _expected_values_per_block[largest_dataset] += block_size - _expected_values_per_block.sum()

        # check if any dataset has 0 samples (and nonzero weight)
        for i, dsname in enumerate(self.dataset_index):
            if _expected_values_per_block[i] == 0 and weights.get(dsname, 0) > 0:
                warnings.warn(
                    f"Dataset {dsname} has 0 samples in the block, but weight of {weights[dsname]}."
                    " Recommend increasing block size."
                )

        return _expected_values_per_block

    def _compute_unpermuted_ids(self, counts_per_block):
        unpermuted_ids = np.zeros(int(counts_per_block.sum()), dtype=np.int64)
        start = 0
        for i, dsname in enumerate(self.dataset_index):
            count = counts_per_block[i]
            unpermuted_ids[start : start + count] = (i << 16) + np.arange(count)
            start += count
        return unpermuted_ids

    @staticmethod
    def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
        """Normalize the weights to sum to 1"""
        total = sum(weights.values())
        if total == 0:
            raise ValueError(f"Datasets' weights cannot sum to 0, got {weights}")

        out_weights = {}
        for name, weight in weights.items():
            if weight < 0:
                raise ValueError(f"Dataset weights cannot be negative, got {weights}")
            elif weight > 0:
                out_weights[name] = weight / total

        return out_weights

    async def async_len(self) -> int:
        if self.stop_strategy == StopStrategy.RESTART_STRATEGY:
            raise ValueError("Length is infinite for restart strategy")

        raise NotImplementedError("Length is not implemented for other strategies")

    async def final_length_is_known(self) -> bool:
        if self.stop_strategy == StopStrategy.RESTART_STRATEGY:
            return False

        raise NotImplementedError("Length is not known for other strategies")

    def is_finite(self) -> bool:
        if self.stop_strategy == StopStrategy.RESTART_STRATEGY:
            return False

        return True

    async def current_len(self) -> Optional[int]:
        if self.stop_strategy == StopStrategy.RESTART_STRATEGY:
            return None

        raise NotImplementedError("Length is not known for other strategies")

    def _get_stage_for_block(self, block_id: int) -> int:
        block_start = block_id * self.block_size
        stage_starts = np.array([start for start, _ in self.weight_stages])
        return max(0, np.searchsorted(stage_starts, block_start, side="right") - 1)

    @alru_cache(maxsize=32)
    async def _get_block(self, index: int) -> Optional[np.ndarray]:
        stage = self._get_stage_for_block(index)
        if not self.randomize_blocks:
            return self._unpermuted_ids_per_stage[stage]

        return np.array(_compute_block_assignment(self._unpermuted_ids_per_stage[stage], index, self.key))

    def _index_into_dataset_for_id(self, id: int, block_id: int) -> tuple[int, int]:
        stage = self._get_stage_for_block(block_id)
        dataset_id = id >> 16
        dataset_index = id & 0xFFFF

        # Get the base offset from previous stages
        base_offset = self._counts_after_stage[stage - 1][dataset_id] if stage > 0 else 0
        # Add offset within current stage
        offset_in_stage = (block_id * self.block_size - self.weight_stages[stage][0]) // self.block_size
        current_stage_offset = offset_in_stage * self._counts_per_block_per_stage[stage][dataset_id]

        return dataset_id, dataset_index + base_offset + current_stage_offset

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T]:
        block_ids = np.array([idx // self.block_size for idx in indices])

        blocks = [self._get_block(block_id) for block_id in block_ids]
        blocks = await asyncio.gather(*blocks)

        # split the indices into batches for each dataset
        batches_per_dataset: list[list[int]] = [[] for _ in range(len(self.datasets))]
        indices_in_final_batch: list[list[int]] = [[] for _ in range(len(self.datasets))]

        assert len(indices) == len(blocks) == len(block_ids)

        for batch_index, (idx, block, block_id) in enumerate(zip(indices, blocks, block_ids)):
            index_within_block = idx % self.block_size  # which element of the block to get
            id = block[index_within_block]  # for this block, which dataset+base dataset offset
            dataset_id, dataset_index = self._index_into_dataset_for_id(id, block_id)
            batches_per_dataset[dataset_id].append(dataset_index)
            indices_in_final_batch[dataset_id].append(batch_index)

        # get the batches from each dataset
        batch_futures = []
        for dataset_id, indices_for_dataset in enumerate(batches_per_dataset):
            if len(indices_for_dataset) == 0:
                batch_futures.append(future_from_value([]))
            else:
                dataset = self._dataset_of_id(dataset_id)
                indices_for_dataset = await self._remap_indices(dataset, indices_for_dataset)
                batch_futures.append(dataset.get_batch(indices_for_dataset))

        batches = await asyncio.gather(*batch_futures)

        # reassemble the final batch
        final_batch = [None] * len(indices)

        for dataset_id, indices_into_batch in enumerate(indices_in_final_batch):
            for i, idx in enumerate(indices_into_batch):
                assert final_batch[idx] is None
                assert len(final_batch) > idx
                final_batch[idx] = batches[dataset_id][i]

        return final_batch  # type: ignore

    async def getitem_async(self, index: int) -> T:
        # simpler implementation because there's only one
        block_id = index // self.block_size
        index = index % self.block_size
        permuted_ids = await self._get_block(block_id)
        dataset_id, dataset_index = self._index_into_dataset_for_id(permuted_ids[index], block_id)

        dataset = self._dataset_of_id(dataset_id)
        dataset_index = (await self._remap_indices(dataset, [dataset_index]))[0]

        return await dataset.getitem_async(dataset_index)

    async def _remap_indices(self, ds, indices_into_ds):
        """
        Handles wrap around for datasets that have finite length
        """
        if self.stop_strategy == StopStrategy.RESTART_STRATEGY:
            if ds.is_finite():
                max_elem = max(indices_into_ds)
                length_of_dataset = await ds.wait_until_len_at_least(max_elem + 1)
                indices_into_ds = [idx % length_of_dataset for idx in indices_into_ds]

            return indices_into_ds

        raise NotImplementedError("Length is not known for other strategies")

    def _dataset_of_id(self, id):
        return self.datasets[self.dataset_index[id]]


def _compute_block_assignment(base_ids, index, key):
    rng = jax.random.fold_in(key, index)
    permuted_ids = jax.random.permutation(rng, base_ids)
    return permuted_ids


def rescale_mixture_schedule_for_batch_schedule(
    mixture_schedule: Sequence[Tuple[int, dict[str, float]]], batch_schedule: BatchSchedule
) -> List[Tuple[int, dict[str, float]]]:
    """
    Rescale the mixture schedule to match the batch schedule. MixtureDataset expects the mixture schedule to be in terms
     of example indices, but the batch schedule is in terms of batch indices/steps. So, given a mixture schedule
     that is in terms of *batch* indices, this function will rescale it to be in terms of example indices suitable for
        MixtureDataset.

    Args:
        mixture_schedule: The mixture schedule to rescale
        batch_schedule: The batch schedule to rescale to

    Returns:
        The rescaled mixture schedule in terms of example indices
    """

    # for each step in mixture_schedule, we want to compute its data offset in the batch schedule
    out = []
    for i, (step, weights) in enumerate(mixture_schedule):
        # find the batch index that corresponds to this step
        if step < 0:
            if i != len(mixture_schedule) - 1:
                raise ValueError("Negative step indices are only allowed for the last step")
            data_offset = -1
        else:
            data_offset = batch_schedule.global_data_offset_by_step(step)

        out.append((data_offset, weights))

    return out
