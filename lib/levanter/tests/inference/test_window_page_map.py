# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np

from levanter.inference.jit_scheduler import SequenceTable
from levanter.inference.page_table import PageBatchInfo
from levanter.inference.utils import INVALID
from levanter.inference.window_page_map import WindowPageMap


def test_window_pool_reuses_expired_page_and_preserves_live_suffix():
    page_size = 4
    logical_pages = 3
    state = SequenceTable.init(max_seqs=1, pages_per_seq=logical_pages, page_size=page_size)
    state = replace(state, used_mask=hax.named(jnp.array([True]), "seq"))
    compact = WindowPageMap.init(logical_pages=logical_pages, window_pages=2, window=4)
    remap = jax.jit(WindowPageMap.remap)
    translated = []
    for logical in range(logical_pages):
        pages = hax.named(
            jnp.where(jnp.arange(logical_pages)[None] <= logical, jnp.arange(logical_pages)[None], INVALID),
            ("seq", "page"),
        )
        total = (logical + 1) * page_size
        live = replace(state, seq_lens=hax.named(jnp.array([total]), "seq"), page_indices=pages)
        work = PageBatchInfo(
            slot_ids=hax.named(jnp.array([0]), "seq"),
            page_indices=pages,
            seq_lens=live.seq_lens,
            cu_q_lens=hax.named(jnp.array([0, page_size]), "seq"),
            num_seqs=jnp.asarray(1),
            new_token_dests=hax.named(logical * page_size + jnp.arange(page_size), "position"),
            page_size=page_size,
        )
        compact, result = remap(compact, live, work)
        translated.append((result.page_indices.array, result.new_token_dests.array))

    first_slot, second_slot, _ = np.asarray(translated[1][0])[0]
    assert first_slot != second_slot
    np.testing.assert_array_equal(translated[2][0], [[INVALID, second_slot, first_slot]])
    np.testing.assert_array_equal(translated[2][1], first_slot * page_size + np.arange(page_size))
