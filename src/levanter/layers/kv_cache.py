# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Cache implementations for paged attention."""

from __future__ import annotations

import dataclasses
import functools
from typing import Generic, Iterable, Iterator, TypeVar, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

import haliax as hax
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call

from levanter.inference.page_table import PageBatchInfo, PageTable


class PageCache(eqx.Module):
    """Abstract base for paged attention caches."""

    def copy_page(self, src_page: int, dst_page: int) -> Self:
        """Return a copy of this cache with ``src_page`` cloned into ``dst_page``."""
        raise NotImplementedError


class KvPageCache(PageCache):
    """Concrete KV cache storing interleaved key/value pages for paged attention."""

    kv_pages: NamedArray  # [Page, Slot, 2 * KVHeads, Embed]

    @staticmethod
    def init(page_table: PageTable, kv_heads: Axis, head_size: Axis, dtype=jnp.float32) -> "KvPageCache":
        """Allocate an empty KV cache matching *page_table* and head axes."""
        kv_pages = hax.zeros(
            {
                "page": page_table.num_pages,
                "slot": page_table.page_size,
                "kv_head": 2 * kv_heads.size,
                head_size.name: head_size.size,
            },
            dtype=dtype,
        )
        return KvPageCache(kv_pages)

    @named_call
    def update(
        self,
        batch_info: PageBatchInfo,
        new_k: NamedArray,  # [Tok, KvHeads, HeadDim]
        new_v: NamedArray,  # [Tok, KvHeads, HeadDim]
    ) -> "KvPageCache":
        """Append keys and values to the cache based on *batch_info*."""
        page_size = self.kv_pages.array.shape[1]

        _ = page_size  # keeps JIT-shapes available; retained for future assertions.

        K = jnp.asarray(batch_info.num_new_tokens, jnp.int32)
        t_pages, t_slots = batch_info.pages_and_slots()  # [T] int32 (first K valid)

        updated = kv_update_unified_prefix(
            self.kv_pages.array,
            t_pages.astype(jnp.int32).array,
            t_slots.astype(jnp.int32).array,
            new_k.array,
            new_v.array,
            K,
        )
        updated = NamedArray(updated, self.kv_pages.axes)
        return dataclasses.replace(self, kv_pages=updated)

    def copy_page(self, src_page: int, dst_page: int) -> "KvPageCache":
        """Copy an entire page of cached keys/values."""
        new_k = self.kv_pages.at["page", dst_page].set(self.kv_pages["page", src_page])
        return dataclasses.replace(self, kv_pages=new_k)


PageCacheT = TypeVar("PageCacheT", bound=PageCache)


class ListCache(PageCache, Generic[PageCacheT]):
    """Container cache that delegates operations to a sequence of caches."""

    caches: tuple[PageCacheT, ...]

    def __post_init__(self):
        object.__setattr__(self, "caches", tuple(self.caches))

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
@functools.partial(jax.jit, donate_argnums=(0,))
def kv_update_unified_prefix(kv_pages, t_pages, t_slots, new_k, new_v, K):
    """
    Update interleaved key/value pages with new tokens.

    kv_pages: [P, S, 2H, D]  (unified K/V buffer, donated)
    t_pages, t_slots: [T] int32  (only first K are valid)
    new_k, new_v: [T, H, D]
    K: int32 scalar = number of valid updates (num_new_tokens)
    """
    kv_ev = _interleave_kv(new_k.astype(kv_pages.dtype), new_v.astype(kv_pages.dtype))  # [T, 2H, D]

    def body(i, buf):
        p = t_pages[i]
        s = t_slots[i]
        ins = kv_ev[i][None, None, :, :]
        return lax.dynamic_update_slice(buf, ins, (p, s, 0, 0))

    return lax.fori_loop(0, K, body, kv_pages)
