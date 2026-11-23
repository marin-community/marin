# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Cache implementations for paged attention."""

import dataclasses
import functools
from typing import Generic, Iterable, Iterator, Self, TypeVar

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call
from jax import lax

from levanter.inference.page_table import PageBatchInfo, PageTableSpec

_SEQUENTIAL_UPDATE_TOKEN_LIMIT = 8


class PageCache(eqx.Module):
    """Abstract base for paged attention caches."""

    def copy_page(self, src_page: int, dst_page: int) -> Self:
        """Return a copy of this cache with ``src_page`` cloned into ``dst_page``."""
        raise NotImplementedError

    def copy_sequence(self, src_slot: int, dst_slot: int) -> Self:
        """Clone request-owned state; page-only caches have no such state."""
        return self

    def reset(self) -> Self:
        """Return a reset version of this cache."""
        raise NotImplementedError


class KvPageCache(PageCache):
    """Concrete KV cache storing interleaved key/value pages for paged attention."""

    kv_pages: NamedArray  # [Page, Slot, 2 * KVHeads, Embed]

    @staticmethod
    def init(spec: PageTableSpec, kv_heads: Axis, head_size: Axis, dtype=jnp.float32) -> "KvPageCache":
        """
        Initialize a KvPageCache with the given page table specification and dimensions.

        Args:
            spec: The layout specification for KV pages.
            kv_heads: Axis for key/value heads.
            head_size: Axis for head size.
            dtype: Data type for the cache.
        """
        kv_pages = hax.zeros(
            {
                "page": spec.num_pages,
                "slot": spec.page_size,
                "kv_head": 2 * kv_heads.size,
                head_size.name: head_size.size,
            },
            dtype=dtype,
        )
        return KvPageCache(kv_pages)

    def reset(self) -> "KvPageCache":
        """Return a reset version of this cache."""
        reset_pages = jnp.zeros_like(self.kv_pages.array)
        return dataclasses.replace(self, kv_pages=NamedArray(reset_pages, self.kv_pages.axes))

    @named_call
    def update(
        self,
        batch_info: PageBatchInfo,
        new_k: NamedArray,  # [Tok, KvHeads, HeadDim]
        new_v: NamedArray,  # [Tok, KvHeads, HeadDim]
        *,
        max_active_tokens: int | None = None,
    ) -> "KvPageCache":
        """Append keys and values to the cache based on *batch_info*."""
        page_size = self.kv_pages.array.shape[1]

        assert page_size == batch_info.page_size, (
            f"Page size mismatch: {page_size} != {batch_info.page_size}. "
            "Ensure that the page size in batch_info matches the kv_pages."
        )

        K = jnp.asarray(batch_info.num_new_tokens, jnp.int32)
        t_pages, t_slots = batch_info.pages_and_slots()  # [T] int32 (first K valid)

        updated = kv_update_unified_prefix(
            self.kv_pages.array,
            t_pages.astype(jnp.int32).array,
            t_slots.astype(jnp.int32).array,
            new_k.array,
            new_v.array,
            K,
            max_active_tokens=max_active_tokens,
        )
        updated = NamedArray(updated, self.kv_pages.axes)
        return dataclasses.replace(self, kv_pages=updated)

    def copy_page(self, src_page: int, dst_page: int) -> "KvPageCache":
        """Copy the entire contents of page ``src_page`` into ``dst_page``.

        This is used when creating clones that should have an identical last partial page, but mapped to a fresh page.
        """
        new_k = self.kv_pages.at["page", dst_page].set(self.kv_pages["page", src_page])
        return dataclasses.replace(self, kv_pages=new_k)


class StackedKvPageCache(PageCache):
    """Interleaved KV pages for a model that scans homogeneous layers."""

    kv_pages: NamedArray  # [Layer, Page, Slot, 2 * KVHeads, Embed]

    def reset(self) -> "StackedKvPageCache":
        return StackedKvPageCache(hax.named(jnp.zeros_like(self.kv_pages.array), self.kv_pages.axes))

    def copy_page(self, src_page: int, dst_page: int) -> "StackedKvPageCache":
        pages = self.kv_pages.at["page", dst_page].set(self.kv_pages["page", src_page])
        return StackedKvPageCache(pages)


PageCacheT = TypeVar("PageCacheT", bound=PageCache)


class ListCache(PageCache, Generic[PageCacheT]):
    """Container cache that delegates operations to a sequence of caches."""

    caches: tuple[PageCacheT, ...]

    def __post_init__(self):
        object.__setattr__(self, "caches", tuple(self.caches))

    def reset(self) -> "ListCache[PageCacheT]":
        return ListCache(tuple(cache.reset() for cache in self.caches))

    @staticmethod
    def from_iterable(caches: Iterable[PageCacheT]) -> "ListCache[PageCacheT]":
        return ListCache(tuple(caches))

    def __len__(self) -> int:
        return len(self.caches)

    def __iter__(self) -> Iterator[PageCacheT]:
        return iter(self.caches)

    def __getitem__(self, idx: int) -> PageCacheT:
        return self.caches[idx]

    def copy_page(self, src_page: int, dst_page: int) -> "ListCache[PageCacheT]":
        return ListCache(tuple(cache.copy_page(src_page, dst_page) for cache in self.caches))

    def copy_sequence(self, src_slot: int, dst_slot: int) -> "ListCache[PageCacheT]":
        return ListCache(tuple(cache.copy_sequence(src_slot, dst_slot) for cache in self.caches))

    def replace(self, idx: int, value: PageCacheT) -> "ListCache[PageCacheT]":
        caches = list(self.caches)
        caches[idx] = value
        return ListCache(tuple(caches))


@named_call
def _interleave_kv(new_k, new_v):
    # [T, H, D] x2 -> [T, 2H, D] with (k0,v0,k1,v1,...) along heads
    T, H, D = new_k.shape
    return jnp.stack([new_k, new_v], axis=2).reshape(T, 2 * H, D)


@named_call
@functools.partial(jax.jit, donate_argnums=(0,), static_argnames=("max_active_tokens",))
def kv_update_unified_prefix(kv_pages, t_pages, t_slots, new_k, new_v, K, *, max_active_tokens: int | None = None):
    """
    Update interleaved key/value pages with new tokens.

    kv_pages: [P, S, 2H, D]  (unified K/V buffer, donated)
    t_pages, t_slots: [T] int32  (only first K are valid)
    new_k, new_v: [T, H, D]
    K: int32 scalar = number of valid updates (num_new_tokens)
    max_active_tokens: static upper bound on K
    """
    kv_ev = _interleave_kv(new_k.astype(kv_pages.dtype), new_v.astype(kv_pages.dtype))  # [T, 2H, D]
    cache_sharding = jax.typeof(kv_pages).sharding
    if not cache_sharding.mesh.empty:
        # Interleaving can erase Explicit head sharding, even for a size-one mesh axis.
        # Match the cache's trailing dimensions before selecting a token and inserting page/slot axes.
        update_spec = cache_sharding.spec.update(partitions=(None, *cache_sharding.spec[2:]))
        kv_ev = jax.reshard(kv_ev, cache_sharding.update(spec=update_spec))

    if max_active_tokens == 1:
        page = jnp.where(K > 0, t_pages[0], kv_pages.shape[0])
        return kv_pages.at[page, t_slots[0]].set(kv_ev[0], mode="drop")

    if t_pages.shape[0] <= _SEQUENTIAL_UPDATE_TOKEN_LIMIT:
        # On TPU, a scatter can reformat the whole cache even for one decode token.
        def update_token(i, cache):
            update = kv_ev[i][None, None, :, :]
            return lax.dynamic_update_slice(cache, update, (t_pages[i], t_slots[i], 0, 0))

        return lax.fori_loop(0, K, update_token, kv_pages)

    rows = jnp.arange(t_pages.shape[0], dtype=jnp.int32)
    active = rows < K
    invalid_page = kv_pages.shape[0]
    pages = jnp.where(active, t_pages, invalid_page)
    slots = jnp.where(active, t_slots, 0)
    # Resolve repeated destinations before the scatter so the last write wins.
    last_write = jnp.full(kv_pages.shape[:2], -1, dtype=jnp.int32).at[pages, slots].max(rows, mode="drop")
    keep = active & (last_write.at[pages, slots].get(mode="fill", fill_value=-1) == rows)
    return kv_pages.at[jnp.where(keep, pages, invalid_page), slots].set(kv_ev, mode="drop")
