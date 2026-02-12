# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp

import haliax as hax

from levanter.inference.utils import INVALID
from levanter.inference.page_table import PageBatchInfo, PageTable
from levanter.inference.jit_scheduler import SequenceTable


def _make_table(pages=8, seqs=4, page_size=2, pages_per_seq=2):
    return PageTable.init(pages, seqs, page_size, pages_per_seq)


def test_page_table_max_len_per_seq():
    pt = _make_table(page_size=2, pages_per_seq=3)
    assert pt.max_len_per_seq == 6


def test_sequence_table_reserve_and_release_slot():
    pt = _make_table()
    sequences = SequenceTable.init(pt.max_seqs, pt.pages_per_seq, pt.page_size)

    sequences, slot_arr = sequences.reserve_slot()
    slot = int(slot_arr)
    assert slot == 0
    assert bool(sequences.used_mask.array[slot])

    sequences = sequences.release_slot(slot)
    assert not bool(sequences.used_mask.array[0])


def test_page_batch_info_shapes():
    seq = hax.Axis("seq", 2)
    page = hax.Axis("page", 3)
    pb = PageBatchInfo(
        slot_ids=hax.arange(seq),
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


def test_sequence_table_allocate_and_free_pages():
    pt = _make_table()
    sequences = SequenceTable.init(pt.max_seqs, pt.pages_per_seq, pt.page_size)

    # reserve two slots
    sequences, seq0_arr = sequences.reserve_slot(0)
    sequences, seq1_arr = sequences.reserve_slot(1)
    seq0 = int(seq0_arr)
    seq1 = int(seq1_arr)
    assert seq0 == 0 and seq1 == 1

    slot_ids = hax.named(jnp.array([seq0, seq1], dtype=jnp.int32), axis=("position",))
    pos_ids = hax.named(jnp.array([0, 0], dtype=jnp.int32), axis=("position",))
    sequences, pt, batch = sequences.allocate_for_seq(pt, slot_ids, pos_ids)

    assert batch.num_seqs == 2
    assert sequences.seq_lens["seq", seq0].scalar() == 1

    # ensure pages are allocated/ref counts updated
    assert (pt.page_ref_counts.array > 0).sum() == 2

    sequences, pt = sequences.free_pages(pt, seq0)
    assert sequences.seq_lens["seq", seq0].scalar() == 0
    assert not bool(sequences.used_mask["seq", seq0].scalar())


def test_free_pages_for_finished_respects_clone_refcounts():
    pt = _make_table(pages=8, seqs=2, page_size=2, pages_per_seq=4)
    sequences = SequenceTable.init(pt.max_seqs, pt.pages_per_seq, pt.page_size)

    sequences, seq0_arr = sequences.reserve_slot(0)
    seq0 = int(seq0_arr)

    slot_ids = hax.named(jnp.array([seq0, seq0, seq0], dtype=jnp.int32), axis=("position",))
    pos_ids = hax.named(jnp.array([0, 1, 2], dtype=jnp.int32), axis=("position",))
    sequences, pt, _ = sequences.allocate_for_seq(pt, slot_ids, pos_ids)

    sequences, pt = sequences.clone_pages_from(pt, src_seq_id=seq0, dst_seq_id=1)

    ref_counts = pt.page_ref_counts.array
    assert int((ref_counts > 0).sum()) == 3
    assert int(ref_counts.max()) == 2

    finished_mask = jnp.array([False, True], dtype=bool)
    sequences, pt = sequences.free_pages_for_finished(pt, finished_mask)

    ref_counts = pt.page_ref_counts.array
    assert int((ref_counts > 0).sum()) == 2
    assert int(ref_counts.max()) == 1
    assert not bool(sequences.used_mask["seq", 1].scalar())
