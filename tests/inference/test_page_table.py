# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax.numpy as jnp

import haliax as hax

from levanter.inference.utils import INVALID
from levanter.inference.page_table import PageBatchInfo, PageTable


def _make_table(pages=8, seqs=4, page_size=2, pages_per_seq=2):
    return PageTable.init(pages, seqs, page_size, pages_per_seq)


def test_page_table_max_len_per_seq():
    pt = _make_table(page_size=2, pages_per_seq=3)
    assert pt.max_len_per_seq == 6


def test_page_table_free_pages():
    pt = _make_table()
    # manually set some ownership
    pt = dataclasses.replace(
        pt,
        page_ref_counts=hax.named(jnp.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=jnp.int32), "page"),
        page_indices=pt.page_indices.at["seq", 0, "page", 0].set(0).at["seq", 0, "page", 1].set(1),
        seq_lens=pt.seq_lens.at["seq", 0].set(4),
    )

    freed = PageTable.free_pages(pt, 0)

    assert jnp.all(freed.page_ref_counts.array[:2] == 0)
    assert jnp.all(freed.page_indices.array[0] == INVALID)
    assert freed.seq_lens.array[0] == INVALID


def test_page_batch_info_shapes():
    seq = hax.Axis("seq", 2)
    page = hax.Axis("page", 3)
    pb = PageBatchInfo(
        page_indices=hax.full((seq, page), INVALID, dtype=jnp.int32),
        seq_lens=hax.full((seq,), INVALID, dtype=jnp.int32),
        cu_q_lens=hax.named(jnp.array([0, 1, 2], dtype=jnp.int32), hax.Axis("seq_plus_one", 3)),
        num_seqs=jnp.array(2, dtype=jnp.int32),
        new_token_dests=hax.full((hax.Axis("position", 2),), INVALID, dtype=jnp.int32),
        page_size=2,
    )

    assert pb.page_indices.axes == (seq, page)
    assert pb.seq_lens.axes == (seq,)
    assert pb.cu_q_lens.array.shape[0] == pb.num_seqs + 1


def test_allocate_for_seqs_with_padding():
    pt = _make_table()
    pt, seq_id = pt.assign_seq_id_to_seq()

    new_pt, batch_info = pt.allocate_for_seq(hax.zeros({"position": 1}, dtype=jnp.int32))

    assert batch_info.new_token_dests.array[0] == 0

    assert new_pt.seq_lens.array[0] == 1
    assert batch_info.num_seqs == 1


def test_allocate_for_seqs_updates_only_valid_ids():
    pt = _make_table(seqs=8, pages=16)
    axis = pt.seq_lens.axes[0]
    seq_lens = hax.named(jnp.array([0, 0, 0, 0, 0, 0, INVALID, INVALID], dtype=jnp.int32), axis)
    pt = dataclasses.replace(pt, seq_lens=seq_lens)

    tokens = hax.named(jnp.array([2, 3, 3, 5, 5, 5], dtype=jnp.int32), hax.Axis("position", 6))
    new_pt, batch_info = eqx.filter_jit(pt.allocate_for_seq)(tokens)

    assert jnp.all(new_pt.seq_lens.array[:6] == jnp.array([0, 0, 1, 2, 0, 3]))
    assert jnp.all(new_pt.seq_lens.array[6:] == INVALID)
    assert batch_info.num_seqs == 3


def test_free_pages_invalid_seq_id_noop():
    pt = _make_table()
    freed = PageTable.free_pages(pt, -1)
    fresh = _make_table()
    assert jnp.array_equal(freed.page_ref_counts.array, fresh.page_ref_counts.array)
    assert jnp.array_equal(freed.page_indices.array, fresh.page_indices.array)
    assert jnp.array_equal(freed.seq_lens.array, fresh.seq_lens.array)
