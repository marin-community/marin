# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import haliax as hax
import haliax.haxtyping as ht
from haliax import NamedArray

from levanter.inference.utils import INVALID, is_valid

__all__ = ["PageTable", "PageTableSpec", "PageBatchInfo"]


@dataclasses.dataclass(frozen=True)
class PageTableSpec:
    """Lightweight description of the layout required for allocating paged KV caches."""

    num_pages: int
    page_size: int


class PageTable(eqx.Module):
    """Global KV page allocator tracking only per-page reference counts."""

    page_ref_counts: NamedArray  # i32[Page]
    page_size: int = eqx.field(static=True)
    _max_seqs: int = eqx.field(static=True)
    _pages_per_seq: int = eqx.field(static=True)

    @staticmethod
    def init(max_pages: int, max_seqs: int, page_size: int, max_pages_per_seq: int) -> "PageTable":
        ref_counts = hax.full({"page": max_pages}, 0, dtype=jnp.int32)
        return PageTable(ref_counts, page_size, max_seqs, max_pages_per_seq)

    @property
    def num_pages(self) -> int:
        return self.page_ref_counts.axis_size("page")

    @property
    def pages_per_seq(self) -> int:
        return self._pages_per_seq

    @property
    def max_seqs(self) -> int:
        return self._max_seqs

    @property
    def max_len_per_seq(self) -> int:
        return self.page_size * self.pages_per_seq

    def spec(self) -> PageTableSpec:
        return PageTableSpec(num_pages=self.num_pages, page_size=self.page_size)


class PageBatchInfo(eqx.Module):
    """
    Page and length information for a batch of sequences.

    NOTE: the "sequence" indices here are not the same as the sequence indices in DecodeState. That is,
    page_indices[0] does not in general correspond to the first sequence in DecodeState, but rather the first sequence
    that has tokens **in this batch**.

    To recover the mapping, use slot_ids to map from batch sequence index to DecodeState sequence index.
    """

    slot_ids: ht.i32[NamedArray, " seq"]  # type: ignore[name-defined]
    page_indices: ht.i32[NamedArray, " seq page"]  # type: ignore[name-defined]
    seq_lens: ht.i32[NamedArray, " seq"]  # type: ignore[name-defined]
    cu_q_lens: ht.i32[NamedArray, " seq"]  # type: ignore[name-defined]
    num_seqs: ht.i32[jnp.ndarray, ""]
    new_token_dests: ht.i32[NamedArray, "position"]  # type: ignore[name-defined]
    page_size: int = eqx.field(static=True)

    @property
    def num_new_tokens(self) -> jnp.ndarray:
        return self.cu_q_lens["seq", self.num_seqs].scalar()

    def __post_init__(self):
        assert isinstance(self.num_seqs, jnp.ndarray), "num_seqs must be a JAX ndarray"

    def pages_and_slots(self):
        token_dests = self.new_token_dests

        t_pages = hax.where(is_valid(token_dests), token_dests // self.page_size, INVALID)
        t_slots = hax.where(is_valid(token_dests), token_dests % self.page_size, INVALID)

        return t_pages, t_slots
