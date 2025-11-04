# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax.numpy as jnp
import numpy as np

import haliax as hax

from levanter.inference.jit_scheduler import SequenceTable
from levanter.inference.page_table import PageTable
from levanter.layers.kv_cache import KvPageCache


def _make_allocator(max_pages=8, max_seqs=2, page_size=4, pages_per_seq=3):
    pt = PageTable.init(max_pages, max_seqs, page_size, pages_per_seq)
    sequences = SequenceTable.init(max_seqs, pages_per_seq, page_size)
    return sequences, pt


def test_clone_pages_from_partial_last_page_allocates_fresh_page():
    sequences, pt = _make_allocator(max_pages=10, max_seqs=2, page_size=4, pages_per_seq=3)

    parent = 0
    child = 1

    sequences, parent_id = sequences.reserve_slot(parent)
    sequences, child_id = sequences.reserve_slot(child)

    seq_lens = sequences.seq_lens.at["seq", parent_id].set(5)
    page_indices = sequences.page_indices.at["seq", parent_id, "page", 0].set(2)
    page_indices = page_indices.at["seq", parent_id, "page", 1].set(3)
    sequences = dataclasses.replace(sequences, seq_lens=seq_lens, page_indices=page_indices)
    ref_counts = pt.page_ref_counts.at["page", 2].set(1)
    ref_counts = ref_counts.at["page", 3].set(1)
    pt = PageTable(ref_counts, pt.page_size, pt._max_seqs, pt._pages_per_seq)

    sequences, new_pt = sequences.clone_pages_from(pt, parent_id, child_id)

    assert int(sequences.page_indices["seq", child_id, "page", 0].scalar()) == 2
    assert int(sequences.page_indices["seq", child_id, "page", 1].scalar()) != 3
    assert int(new_pt.page_ref_counts["page", 2].scalar()) == 2
    assert int(new_pt.page_ref_counts["page", 3].scalar()) == 1
    assert int(sequences.seq_lens["seq", child_id].scalar()) == 5


def test_clone_pages_from_boundary_shares_last_page():
    sequences, pt = _make_allocator(max_pages=10, max_seqs=2, page_size=4, pages_per_seq=3)

    parent = 0
    child = 1

    sequences, parent_id = sequences.reserve_slot(parent)
    sequences, child_id = sequences.reserve_slot(child)

    seq_lens = sequences.seq_lens.at["seq", parent_id].set(8)
    page_indices = sequences.page_indices.at["seq", parent_id, "page", 0].set(4)
    page_indices = page_indices.at["seq", parent_id, "page", 1].set(5)
    sequences = dataclasses.replace(sequences, seq_lens=seq_lens, page_indices=page_indices)
    ref_counts = pt.page_ref_counts.at["page", 4].set(1)
    ref_counts = ref_counts.at["page", 5].set(1)
    pt = PageTable(ref_counts, pt.page_size, pt._max_seqs, pt._pages_per_seq)

    sequences, new_pt = sequences.clone_pages_from(pt, parent_id, child_id)

    assert int(sequences.page_indices["seq", child_id, "page", 0].scalar()) == 4
    assert int(sequences.page_indices["seq", child_id, "page", 1].scalar()) == 5
    assert int(new_pt.page_ref_counts["page", 4].scalar()) == 2
    assert int(new_pt.page_ref_counts["page", 5].scalar()) == 2
    assert int(sequences.seq_lens["seq", child_id].scalar()) == 8


def test_kv_cache_copy_page():
    sequences, pt = _make_allocator(max_pages=3, max_seqs=1, page_size=2, pages_per_seq=1)
    kv = KvPageCache.init(pt.spec(), kv_heads=hax.Axis("kv_head", 2), head_size=hax.Axis("head", 3), dtype=jnp.float32)

    src_page = 1
    dst_page = 2
    k_pattern = hax.full_like(kv.kv_pages["page", src_page, "kv_head", 0::2], 7.0)
    v_pattern = hax.full_like(kv.kv_pages["page", src_page, "kv_head", 1::2], 3.0)
    kv_pattern = hax.stack("inter", [k_pattern, v_pattern]).rearrange(
        "{inter kv_head} -> ... (kv_head: kv_head inter)"
    )
    kv = dataclasses.replace(kv, kv_pages=kv.kv_pages.at["page", src_page].set(kv_pattern))

    kv2 = kv.copy_page(src_page, dst_page)
    np.testing.assert_allclose(
        np.asarray(kv2.kv_pages["page", dst_page, "kv_head", 0::2].array),
        np.asarray(kv.kv_pages["page", src_page, "kv_head", 0::2].array),
    )
    np.testing.assert_allclose(
        np.asarray(kv2.kv_pages["page", dst_page, "kv_head", 1::2].array),
        np.asarray(kv.kv_pages["page", src_page, "kv_head", 1::2].array),
    )
