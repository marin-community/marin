# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import haliax as hax
import jax
import jax.numpy as jnp
import pytest

from levanter.inference.jit_scheduler import DecodeState, SequenceTable, TokenQueue
from levanter.inference.page_table import PageTable
from levanter.inference.utils import INVALID


def test_pack_next_sequence_single_seq_boundary_at_last_token():
    # Queue entirely filled with a single sequence id; pack exactly all tokens.
    capacity = 8
    tq = TokenQueue.init(capacity)

    tokens = hax.named(jnp.arange(capacity, dtype=jnp.int32), axis=("position",))
    slot_ids = hax.named(jnp.full((capacity,), 0, dtype=jnp.int32), axis=("position",))

    # Absolute pos_ids for a single sequence: 0..capacity-1
    pos_ids = hax.named(jnp.arange(capacity, dtype=jnp.int32), axis=("position",))
    tq = tq.enqueue_tokens(tokens, slot_ids, pos_ids, capacity)

    tq2, packed = tq.pack_next_sequence(capacity)

    # Queue should be empty now
    assert tq2.num_queued_tokens == 0

    # Determine boundaries based on PageTable seq_lens after allocation
    assert int(packed.num_tokens) == capacity
    pt = PageTable.init(max_pages=16, max_seqs=4, page_size=8, max_pages_per_seq=4)
    sequences = SequenceTable.init(pt.max_seqs, pt.pages_per_seq, pt.page_size)
    sequences, _ = sequences.reserve_slot(0)
    sequences, pt, binfo = sequences.allocate_for_seq(pt, packed.slot_ids, packed.pos_ids)
    seq_lens_after = binfo.seq_lens["seq", packed.slot_ids]
    boundary_mask = packed.pos_ids == (seq_lens_after - 1)
    # Expect exactly one boundary at the last token
    bm = boundary_mask.array
    assert bm.dtype == jnp.bool_
    assert bool(bm[-1]) is True
    assert int(bm.sum()) == 1


@pytest.mark.parametrize("seq_ids", [[0, 1], [1, 0]])
def test_pack_next_sequence_boundaries_between_sequences(seq_ids):
    # Two sequences back-to-back; boundaries at the last token of each sequence in the packed slice.
    capacity = 7
    tq = TokenQueue.init(capacity)
    seq1, seq2 = seq_ids

    tokens = hax.named(jnp.array([10, 11, 12, 20, 21, 22, 23], dtype=jnp.int32), axis=("position",))
    slot_ids = hax.named(jnp.array([seq1, seq1, seq1, seq2, seq2, seq2, seq2], dtype=jnp.int32), axis=("position",))

    # Absolute pos_ids are per-sequence; start fresh at 0 for each sequence in this test
    pos_ids = hax.named(jnp.array([0, 1, 2, 0, 1, 2, 3], dtype=jnp.int32), axis=("position",))
    tq = tq.enqueue_tokens(tokens, slot_ids, pos_ids, capacity)

    tq2, packed = tq.pack_next_sequence(capacity)

    assert int(packed.num_tokens) == capacity
    pt = PageTable.init(max_pages=16, max_seqs=4, page_size=8, max_pages_per_seq=4)
    sequences = SequenceTable.init(pt.max_seqs, pt.pages_per_seq, pt.page_size)
    sequences, assigned1 = sequences.reserve_slot(seq1)
    assert int(assigned1) == seq1
    sequences, assigned2 = sequences.reserve_slot(seq2)
    assert int(assigned2) == seq2
    sequences, pt, binfo = sequences.allocate_for_seq(pt, packed.slot_ids, packed.pos_ids)
    seq_lens_after = binfo.seq_lens["seq", packed.slot_ids]

    boundary_mask = packed.pos_ids == (seq_lens_after - 1)
    bm = boundary_mask.array
    if seq1 == 0:
        # Boundaries at positions 2 and 6
        assert bool(bm[2]) is True
        assert bool(bm[6]) is True
    else:
        # Boundaries at positions 2 and 5
        assert bool(bm[3]) is True
        assert bool(bm[6]) is True

    assert int(bm.sum()) == 2


def test_allocate_for_seq_ignores_out_of_range_slots():
    pt = PageTable.init(max_pages=8, max_seqs=2, page_size=4, max_pages_per_seq=2)
    sequences = SequenceTable.init(pt.max_seqs, pt.pages_per_seq, pt.page_size)

    # Slot ids that are >= max_seqs should be ignored without allocating pages.
    slot_ids = hax.named(jnp.array([5, 5], dtype=jnp.int32), axis=("position",))
    pos_ids = hax.named(jnp.array([0, 1], dtype=jnp.int32), axis=("position",))

    new_sequences, new_pt, batch = sequences.allocate_for_seq(pt, slot_ids, pos_ids)

    assert jnp.array_equal(new_sequences.seq_lens.array, jnp.zeros((pt.max_seqs,), dtype=jnp.int32))
    assert jnp.all(new_sequences.page_indices.array == INVALID)
    assert jnp.all(new_sequences.kv_pages.array == INVALID)
    assert jnp.all(new_pt.page_ref_counts.array == 0)
    assert int(batch.num_seqs) == 0
    assert jnp.all(batch.new_token_dests.array == INVALID)


def test_allocate_for_seq_ignores_invalid_padding_tokens():
    pt = PageTable.init(max_pages=8, max_seqs=2, page_size=4, max_pages_per_seq=2)
    sequences = SequenceTable.init(pt.max_seqs, pt.pages_per_seq, pt.page_size)
    sequences, seq0 = sequences.reserve_slot(0)
    assert int(seq0) == 0

    slot_ids = hax.named(jnp.array([0, INVALID], dtype=jnp.int32), axis=("position",))
    pos_ids = hax.named(jnp.array([0, INVALID], dtype=jnp.int32), axis=("position",))

    new_sequences, new_pt, batch = sequences.allocate_for_seq(pt, slot_ids, pos_ids)

    seq_lens_arr = new_sequences.seq_lens.array
    assert int(seq_lens_arr[0]) == 1
    assert jnp.all(seq_lens_arr[1:] == 0)
    assert jnp.all(new_sequences.page_indices.array[1:] == INVALID)
    # Only one sequence should be considered in the batch info
    assert int(batch.num_seqs) == 1
    # The real token should map to a concrete KV destination
    assert int(batch.new_token_dests.array[0]) == 0
    # Padding token should not produce a KV destination
    assert int(batch.new_token_dests.array[1]) == INVALID
    assert int(new_pt.page_ref_counts.array.sum()) == 1


def test_decode_state_stats_tracks_pages():
    pt = PageTable.init(max_pages=4, max_seqs=2, page_size=4, max_pages_per_seq=2)
    ds = DecodeState.init(pt, max_stop_seqs=0, max_stop_tokens=0, max_queued_tokens=4)

    stats = jax.device_get(ds.stats())
    assert int(stats.active_seqs) == 0
    assert int(stats.pages_in_use) == 0
    assert int(stats.free_pages) == pt.num_pages
    assert int(stats.max_refcount) == 0

    ds, slot = ds.reserve_slot(0)
    slot_id = int(slot)
    slot_ids = hax.named(jnp.array([slot_id], dtype=jnp.int32), axis=("position",))
    pos_ids = hax.named(jnp.array([0], dtype=jnp.int32), axis=("position",))
    ds, _ = ds.allocate_for_seq(slot_ids, pos_ids)

    stats = jax.device_get(ds.stats())
    assert int(stats.active_seqs) == 1
    assert int(stats.pages_in_use) == 1
    assert int(stats.free_pages) == pt.num_pages - 1
    assert int(stats.max_refcount) == 1
