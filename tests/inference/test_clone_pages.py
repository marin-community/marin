# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax.numpy as jnp
import numpy as np

import haliax as hax

from levanter.inference.page_table import PageTable
from levanter.layers.kv_cache import KvPageCache


def _make_table(max_pages=8, max_seqs=2, page_size=4, pages_per_seq=3):
    return PageTable.init(max_pages, max_seqs, page_size, pages_per_seq)


def test_clone_pages_from_partial_last_page_allocates_fresh_page():
    pt = _make_table(max_pages=10, max_seqs=2, page_size=4, pages_per_seq=3)

    # Parent uses two pages with a partial last page: length = 5 (pages 0 and 1 used)
    parent = 0
    child = 1

    pt = dataclasses.replace(
        pt,
        page_indices=pt.page_indices.at["seq", parent, "page", 0]
        .set(jnp.array(2, dtype=jnp.int32))
        .at["seq", parent, "page", 1]
        .set(jnp.array(3, dtype=jnp.int32))
        .at["seq", parent, "page", 2]
        .set(jnp.array(-1, dtype=jnp.int32)),
        seq_lens=pt.seq_lens.at["seq", parent].set(jnp.array(5, dtype=jnp.int32)),
    )

    # Clone
    new_pt = pt.clone_pages_from(parent, child)

    # Fully used pages (all but last partial) should be shared: page 0 mapping identical
    assert int(new_pt.page_indices["seq", child, "page", 0].scalar()) == 2

    # Last page must be a fresh allocation, different from parent's last page (3)
    assert int(new_pt.page_indices["seq", child, "page", 1].scalar()) != 3

    # Refcounts: +1 for shared full page (page 2) and +1 for newly allocated page; parent's partial last page unchanged
    ref_shared = int(new_pt.page_ref_counts["page", 2].scalar())
    ref_parent_last = int(new_pt.page_ref_counts["page", 3].scalar())
    assert ref_shared == 1
    assert ref_parent_last == 0

    # Lengths equal (no rounding)
    assert int(new_pt.seq_lens["seq", child].scalar()) == 5


def test_clone_pages_from_boundary_shares_last_page():
    pt = _make_table(max_pages=10, max_seqs=2, page_size=4, pages_per_seq=3)

    # Parent uses exactly 2 full pages: length = 8 (pages 4 and 5 used)
    parent = 0
    child = 1

    pt = dataclasses.replace(
        pt,
        page_indices=pt.page_indices.at["seq", parent, "page", 0]
        .set(jnp.array(4, dtype=jnp.int32))
        .at["seq", parent, "page", 1]
        .set(jnp.array(5, dtype=jnp.int32))
        .at["seq", parent, "page", 2]
        .set(jnp.array(-1, dtype=jnp.int32)),
        seq_lens=pt.seq_lens.at["seq", parent].set(jnp.array(8, dtype=jnp.int32)),
    )

    new_pt = pt.clone_pages_from(parent, child)

    # Child should share both pages
    assert int(new_pt.page_indices["seq", child, "page", 0].scalar()) == 4
    assert int(new_pt.page_indices["seq", child, "page", 1].scalar()) == 5

    # Refcounts incremented for both pages
    assert int(new_pt.page_ref_counts["page", 4].scalar()) == 1
    assert int(new_pt.page_ref_counts["page", 5].scalar()) == 1

    # Lengths equal
    assert int(new_pt.seq_lens["seq", child].scalar()) == 8


def test_kv_cache_copy_page():
    # Minimal KvPageCache with 3 pages and small dims
    from levanter.inference.page_table import PageTable as _PT

    pt = _PT.init(max_pages=3, max_seqs=1, page_size=2, max_pages_per_seq=1)
    kv = KvPageCache.init(pt, kv_heads=hax.Axis("kv_head", 2), head_size=hax.Axis("head", 3), dtype=jnp.float32)

    # Write identifiable values into page 1 for both K and V
    src_page = 1
    dst_page = 2
    k_pattern = hax.full_like(kv.kv_pages["page", src_page, "kv_head", 0::2], 7.0)
    v_pattern = hax.full_like(kv.kv_pages["page", src_page, "kv_head", 1::2], 3.0)
    # need to interleave k and v along kv_heads
    kv_pattern = hax.stack("inter", [k_pattern, v_pattern]).rearrange(
        "{inter kv_head} -> ... (kv_head: kv_head inter)"
    )
    kv = dataclasses.replace(
        kv,
        kv_pages=kv.kv_pages.at["page", src_page].set(kv_pattern),
    )

    kv2 = kv.copy_page(src_page, dst_page)
    np.testing.assert_allclose(
        np.asarray(kv2.kv_pages["page", dst_page, "kv_head", 0::2].array),
        np.asarray(kv.kv_pages["page", src_page, "kv_head", 0::2].array),
    )
    np.testing.assert_allclose(
        np.asarray(kv2.kv_pages["page", dst_page, "kv_head", 1::2].array),
        np.asarray(kv.kv_pages["page", src_page, "kv_head", 1::2].array),
    )
