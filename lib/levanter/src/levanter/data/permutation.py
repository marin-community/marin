# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import functools
from dataclasses import dataclass
from typing import Optional, Sequence

import jax.random
from async_lru import alru_cache
from jaxtyping import PRNGKeyArray

from levanter.data._prp import PermType, Permutation
from levanter.data.dataset import AsyncDataset, T_co
from levanter.utils.jax_utils import local_cpu_mesh


def _key_on_local_cpu(key: PRNGKeyArray) -> PRNGKeyArray:
    """Canonicalize a PRNG key onto the local CPU device."""
    with local_cpu_mesh():
        return jax.device_put(jax.device_get(key))


def _fold_in_on_local_cpu(key: PRNGKeyArray, data: int) -> PRNGKeyArray:
    """Fold in an integer into a PRNG key while pinned to local CPU mesh."""
    with local_cpu_mesh():
        key = jax.device_put(jax.device_get(key))
        return jax.device_put(jax.device_get(jax.random.fold_in(key, data)))


class PermutationDataset(AsyncDataset[T_co]):
    """A permutation dataset that wraps another dataset and applies a permutation to the indices."""

    # TODO: add epoch reshuffling

    def __init__(self, dataset: AsyncDataset[T_co], key: PRNGKeyArray, perm_type: PermType = "feistel"):
        self.dataset = dataset
        self.key = _key_on_local_cpu(key)
        self._permutation: Optional[Permutation] = None
        self._perm_type = perm_type

    async def async_len(self) -> int:
        return await self.dataset.async_len()

    def is_finite(self) -> bool:
        return self.dataset.is_finite()

    async def getitem_async(self, index: int) -> T_co:
        permutation = await self._get_permutation()
        return await self.dataset.getitem_async(permutation(index))

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        permutation = await self._get_permutation()
        return await self.dataset.get_batch(
            [int(permutation(i)) for i in indices]
        )  # cast to int to be sure it's python int

    async def _get_permutation(self):
        if self._permutation is None:
            self._permutation = Permutation.make(self._perm_type, await self.async_len(), self.key)
        return self._permutation


class EraShufflingDataset(AsyncDataset[T_co]):
    r"""
    A dataset that shuffles the data in "eras" of fixed length. Era shuffling is somewhere in between a shuffle buffer
    and a permutation. It's a "local" permutation where pi(i) \in [ (i//L) * L, (i//L + 1) * L ) for some era length L.

    The advantages of era shuffling are:
    - It's stateless, so resumes are easy
    - Like shuffle buffers, it's a decent compromise between full shuffling and no shuffling
    - Like a shuffle buffer, it's streaming: we don't need to know the length of the data in advance

    The disadvantages are:
    - It's not as good as full shuffling
    - It distributes less well than a shuffle buffer does. It's more like a "local" shuffle buffer.
    - You have to wait for an era to fill before you can start shuffling it. With prefetching, this is less of an issue.


    # TODO: given the way tokenization works (where it runs way ahead of training), we can probably increase the era
    length # over time. This would be a nice feature to have.
    """

    def __init__(
        self, dataset: AsyncDataset[T_co], era_length: int, *, key: jax.random.PRNGKey, perm_type: PermType = "feistel"
    ):
        self.dataset = dataset
        self.era_length = era_length

        # Force key to CPU (like MixtureDataset does) to prevent JAX device placement errors
        with local_cpu_mesh():
            if isinstance(key, int):
                key = jax.random.PRNGKey(key)
            else:
                key = _key_on_local_cpu(key)

        self.key = key
        self._perm_type = perm_type

        @alru_cache(maxsize=4)  # we're mostly going to be going sequentially
        async def gen_era_permutation(era: int) -> Permutation:
            # TODO: support epochs
            if self.dataset.is_finite():
                # edge case: final era may be shorter than era_length
                dataset_len = await self.dataset.async_len()
                remaining = dataset_len - era * self.era_length
                if remaining <= 0:
                    raise IndexError(
                        f"Era {era} is out of bounds for dataset length {dataset_len} with era length {self.era_length}"
                    )
                era_length_val = min(self.era_length, remaining)
            else:
                era_length_val = self.era_length

            mix_key = _fold_in_on_local_cpu(key, era)
            return Permutation.make(self._perm_type, era_length_val, mix_key)

        self.gen_era_permutation = gen_era_permutation

    async def _get_index(self, idx: int) -> int:
        if idx < 0:
            raise ValueError("Negative indices are not supported")
        era = idx // self.era_length
        permutation = await self.gen_era_permutation(era)
        out = permutation(idx - era * self.era_length) + era * self.era_length

        return out

    async def async_len(self) -> int:
        return await self.dataset.async_len()

    def is_finite(self) -> bool:
        return self.dataset.is_finite()

    async def getitem_async(self, index: int) -> T_co:
        return await self.dataset.getitem_async(await self._get_index(index))

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        return await self.dataset.get_batch([await self._get_index(i) for i in indices])

    def __repr__(self):
        return f"EraShufflingDataset({repr(self.dataset)}, era_length={self.era_length})"

    def __str__(self):
        return f"EraShufflingDataset({str(self.dataset)})"


@dataclass(frozen=True)
class _BlockShuffleState:
    dataset_len: int
    num_full_blocks: int
    tail_size: int
    total_blocks: int
    num_windows: int
    last_window_id: int
    length_before_last_window: int
    full_window_size: int


@dataclass(frozen=True)
class _WindowLayout:
    """
    Layout of one logical shuffle window.

    `full_blocks` are permuted blocks of size `io_block_size`.
    `tail_size` is the optional tiny block at the end of the final window.
    """

    full_blocks: tuple[int, ...]
    full_region_size: int
    tail_size: int

    @property
    def window_size(self) -> int:
        return self.full_region_size + self.tail_size


class BlockShufflingDataset(AsyncDataset[T_co]):
    """
    A dataset that applies hierarchical block shuffling for better I/O locality.

    It works in two stages:
    1. Permute full blocks of size `io_block_size`.
    2. Within each window of `window_blocks`, permute examples from full blocks.

    If the dataset has a final partial (tiny) block, that tail block is always kept at
    the very end and is not mixed into earlier positions. This keeps the edge case
    simple and deterministic.
    """

    def __init__(
        self,
        dataset: AsyncDataset[T_co],
        io_block_size: int,
        *,
        window_blocks: int,
        key: PRNGKeyArray | int,
        perm_type: PermType = "feistel",
    ):
        if io_block_size <= 0:
            raise ValueError(f"io_block_size must be positive, got {io_block_size}")
        if window_blocks <= 0:
            raise ValueError(f"window_blocks must be positive, got {window_blocks}")

        self.dataset = dataset
        self.io_block_size = io_block_size
        self.window_blocks = window_blocks
        self._perm_type = perm_type

        with local_cpu_mesh():
            if isinstance(key, int):
                key = jax.random.PRNGKey(key)
            else:
                key = _key_on_local_cpu(key)
            block_key, window_full_key, window_tail_key = jax.random.split(key, 3)
            self.key = _key_on_local_cpu(key)
            self._block_key = _key_on_local_cpu(block_key)
            self._window_full_key = _key_on_local_cpu(window_full_key)
            self._window_tail_key = _key_on_local_cpu(window_tail_key)

        self._state: _BlockShuffleState | None = None
        self._full_block_permutation: Optional[Permutation] = None

    def is_finite(self) -> bool:
        return self.dataset.is_finite()

    async def async_len(self) -> int:
        state = await self._ensure_initialized()
        return state.dataset_len

    async def getitem_async(self, index: int) -> T_co:
        mapped = await self._map_index(index)
        return await self.dataset.getitem_async(mapped)

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        if not indices:
            return []
        mapped = [await self._map_index(i) for i in indices]
        return await self.dataset.get_batch(mapped)

    async def _ensure_initialized(self) -> _BlockShuffleState:
        if self._state is not None:
            return self._state

        if not self.dataset.is_finite():
            raise ValueError("BlockShufflingDataset only supports finite datasets")

        dataset_len = await self.dataset.async_len()
        if dataset_len < 0:
            raise ValueError(f"Dataset length must be non-negative, got {dataset_len}")

        num_full_blocks = dataset_len // self.io_block_size
        tail_size = dataset_len % self.io_block_size
        total_blocks = num_full_blocks + int(tail_size > 0)
        full_window_size = self.io_block_size * self.window_blocks
        num_windows = (total_blocks + self.window_blocks - 1) // self.window_blocks if total_blocks > 0 else 0
        last_window_id = max(0, num_windows - 1)
        length_before_last_window = 0 if num_windows <= 1 else (num_windows - 1) * full_window_size

        if num_full_blocks > 1:
            self._full_block_permutation = Permutation.make(self._perm_type, num_full_blocks, self._block_key)

        self._state = _BlockShuffleState(
            dataset_len=dataset_len,
            num_full_blocks=num_full_blocks,
            tail_size=tail_size,
            total_blocks=total_blocks,
            num_windows=num_windows,
            last_window_id=last_window_id,
            length_before_last_window=length_before_last_window,
            full_window_size=full_window_size,
        )
        return self._state

    def _state_or_error(self) -> _BlockShuffleState:
        if self._state is None:
            raise RuntimeError("BlockShufflingDataset is not initialized")
        return self._state

    @functools.lru_cache(maxsize=256)
    def _window_layout(self, window_id: int) -> _WindowLayout:
        state = self._state_or_error()
        if window_id < 0 or window_id >= state.num_windows:
            raise IndexError(f"Window id {window_id} out of bounds for {state.num_windows} windows")

        block_start = window_id * self.window_blocks
        blocks_remaining = state.total_blocks - block_start
        blocks_in_window = min(self.window_blocks, blocks_remaining)

        has_tail_in_window = state.tail_size > 0 and (block_start + blocks_in_window == state.total_blocks)
        full_blocks_in_window = blocks_in_window - int(has_tail_in_window)

        physical_full_blocks: list[int] = []
        for block_position in range(block_start, block_start + full_blocks_in_window):
            if self._full_block_permutation is None:
                # Identity for 0 or 1 full blocks.
                physical_full_blocks.append(block_position)
            else:
                physical_full_blocks.append(int(self._full_block_permutation(block_position)))

        tail_size = state.tail_size if has_tail_in_window else 0
        full_region_size = len(physical_full_blocks) * self.io_block_size
        return _WindowLayout(
            full_blocks=tuple(physical_full_blocks), full_region_size=full_region_size, tail_size=tail_size
        )

    @functools.lru_cache(maxsize=256)
    def _window_full_region_permutation(self, window_id: int) -> Optional[Permutation]:
        layout = self._window_layout(window_id)
        if layout.full_region_size <= 1:
            return None
        key = _fold_in_on_local_cpu(self._window_full_key, window_id)
        return Permutation.make(self._perm_type, layout.full_region_size, key)

    @functools.lru_cache(maxsize=256)
    def _window_tail_region_permutation(self, window_id: int) -> Optional[Permutation]:
        layout = self._window_layout(window_id)
        if layout.tail_size <= 1:
            return None
        key = _fold_in_on_local_cpu(self._window_tail_key, window_id)
        return Permutation.make(self._perm_type, layout.tail_size, key)

    async def _map_index(self, index: int) -> int:
        if index < 0:
            raise ValueError("Negative indices are not supported")

        state = await self._ensure_initialized()

        if index >= state.dataset_len:
            raise IndexError(f"Index {index} out of bounds for dataset length {state.dataset_len}")

        if state.num_windows == 0:
            raise IndexError(f"Index {index} out of bounds for empty dataset")

        if state.num_windows == 1:
            window_id = 0
            offset_in_window = index
        elif index < state.length_before_last_window:
            window_id = index // state.full_window_size
            offset_in_window = index % state.full_window_size
        else:
            window_id = state.last_window_id
            offset_in_window = index - state.length_before_last_window

        layout = self._window_layout(window_id)

        if offset_in_window >= layout.window_size:
            raise IndexError(
                f"Offset {offset_in_window} out of bounds for window {window_id} with length {layout.window_size}"
            )

        if offset_in_window < layout.full_region_size:
            perm = self._window_full_region_permutation(window_id)
            permuted_offset = offset_in_window if perm is None else int(perm(offset_in_window))
            block_offset, offset_in_block = divmod(permuted_offset, self.io_block_size)
            physical_full_block = layout.full_blocks[block_offset]
            return physical_full_block * self.io_block_size + offset_in_block

        # Final tiny tail block stays at the very end, but can be locally permuted.
        tail_offset = offset_in_window - layout.full_region_size
        tail_perm = self._window_tail_region_permutation(window_id)
        permuted_tail_offset = tail_offset if tail_perm is None else int(tail_perm(tail_offset))
        return state.num_full_blocks * self.io_block_size + permuted_tail_offset

    def __repr__(self):
        return (
            "BlockShufflingDataset("
            f"{repr(self.dataset)}, io_block_size={self.io_block_size}, "
            f"window_blocks={self.window_blocks})"
        )

    def __str__(self):
        return (
            "BlockShufflingDataset("
            f"{str(self.dataset)}, io_block_size={self.io_block_size}, "
            f"window_blocks={self.window_blocks})"
        )
