# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import jax.numpy as jnp
import haliax as hax

from levanter.data.splice_dataset import SpliceDocumentDataset


def _arr(x):
    return np.asarray(x, dtype=np.int32)


def test_anchor_start_with_k_moves_doc_left_to_right():
    # S=5, L=5, but anchor_start with K=3 should yield s in {0,1,2}
    Pos = hax.Axis("position", 5)
    doc = _arr([0, 1, 2, 3, 4])
    pad = -1

    ds = SpliceDocumentDataset(
        Pos=Pos,
        doc_tokens=doc,
        pad_token_id=pad,
        content_length=3,
        content_start_mode="anchor_start",
    ).as_sync_dataset()

    assert len(ds) == 3

    ex0 = ds[0].tokens.array
    ex1 = ds[1].tokens.array
    ex2 = ds[2].tokens.array

    np.testing.assert_array_equal(ex0, _arr([0, 1, 2, pad, pad]))
    np.testing.assert_array_equal(ex1, _arr([pad, 0, 1, 2, pad]))
    np.testing.assert_array_equal(ex2, _arr([pad, pad, 0, 1, 2]))

    # loss mask excludes final in-span token
    np.testing.assert_array_equal(ds[0].loss_mask.array, _arr([1, 1, 0, 0, 0]))
    np.testing.assert_array_equal(ds[1].loss_mask.array, _arr([0, 1, 1, 0, 0]))
    np.testing.assert_array_equal(ds[2].loss_mask.array, _arr([0, 0, 1, 1, 0]))


def test_slide_within_k3_matches_example_grid():
    # S=5, L=5, K=3, slide t in {0,1,2,3} with pruning for copy_len>=2
    Pos = hax.Axis("position", 5)
    doc = _arr([0, 1, 2, 3, 4])
    pad = -1

    ds = SpliceDocumentDataset(
        Pos=Pos,
        doc_tokens=doc,
        pad_token_id=pad,
        content_length=3,
        content_start_mode="slide_within",
    ).as_sync_dataset()

    # From the design doc: 13 total examples
    assert len(ds) == 13

    # Spot-check a few indices based on enumeration order (t outer, s inner):
    # (t=0,s=0)
    np.testing.assert_array_equal(ds[0].tokens.array, _arr([0, 1, 2, pad, pad]))
    # (t=0,s=2)
    np.testing.assert_array_equal(ds[2].tokens.array, _arr([pad, pad, 0, 1, 2]))
    # (t=3,s=0)
    np.testing.assert_array_equal(ds[9].tokens.array, _arr([3, 4, pad, pad, pad]))
    # (t=3,s=3)
    np.testing.assert_array_equal(ds[12].tokens.array, _arr([pad, pad, pad, 3, 4]))

    # Check loss masks for a couple
    # (t=1,s=1): [pad,1,2,3,pad] → loss at positions 1,2
    lm = ds[4].loss_mask.array
    np.testing.assert_array_equal(lm, _arr([0, 1, 1, 0, 0]))

    # Segment IDs: s>0 splits segments at s; s=0 yields a single segment (all ones here)
    seg0 = ds[0].attn_mask.segment_ids[0].array
    seg1 = ds[1].attn_mask.segment_ids[0].array
    seg9 = ds[9].attn_mask.segment_ids[0].array
    np.testing.assert_array_equal(seg0, _arr([1, 1, 1, 1, 1]))
    np.testing.assert_array_equal(seg1, _arr([0, 1, 1, 1, 1]))
    np.testing.assert_array_equal(seg9, _arr([1, 1, 1, 1, 1]))  # s=0 → single segment


def test_content_length_none_clamps_to_fit():
    # S=5, L=3, slide_within with K=None should use all that fits per (t,s)
    Pos = hax.Axis("position", 5)
    doc = _arr([10, 20, 30])
    pad = -1

    ds = SpliceDocumentDataset(
        Pos=Pos,
        doc_tokens=doc,
        pad_token_id=pad,
        content_length=None,
        content_start_mode="slide_within",
    ).as_sync_dataset()

    # t=0: s in 0..2 (3 ex), t=1: s in 0..3 (4 ex), t=2: remaining=1 → pruned (copy_len<2)
    assert len(ds) == 7

    # (t=0,s=2) → copy_len=min(3,3,3)=3
    np.testing.assert_array_equal(ds[2].tokens.array, _arr([pad, pad, 10, 20, 30]))

    # (t=1,s=3) → copy_len=min(2,2,2)=2
    np.testing.assert_array_equal(ds[6].tokens.array, _arr([pad, pad, pad, 20, 30]))


def test_slide_within_multi_stride_counts_and_samples():
    # S=6, L=6, K=4, t stride=2 → t in {0,2,4}; s stride=2
    # Expected pairs: (0,0),(0,2),(2,0),(2,2),(4,0),(4,2),(4,4) → total 7
    Pos = hax.Axis("position", 6)
    doc = _arr([0, 1, 2, 3, 4, 5])
    pad = -1

    ds = SpliceDocumentDataset(
        Pos=Pos,
        doc_tokens=doc,
        pad_token_id=pad,
        content_length=4,
        content_stride=2,
        offset_stride=2,
        content_start_mode="slide_within",
    ).as_sync_dataset()

    assert len(ds) == 7

    # (t=0,s=0)
    np.testing.assert_array_equal(ds[0].tokens.array, _arr([0, 1, 2, 3, pad, pad]))
    # (t=0,s=2)
    np.testing.assert_array_equal(ds[1].tokens.array, _arr([pad, pad, 0, 1, 2, 3]))
    # (t=4,s=4)
    np.testing.assert_array_equal(ds[-1].tokens.array, _arr([pad, pad, pad, pad, 4, 5]))

    # Loss mask spot-check for (t=2,s=0, copy_len=4) → ones on 0..2
    lm = ds[2].loss_mask.array
    np.testing.assert_array_equal(lm, _arr([1, 1, 1, 0, 0, 0]))


def test_min_copy_len_prunes_short_spans():
    # S=6, L=6, K=4, min_copy_len=4 should prune later t where remaining < 4
    # Remaining valid pairs: t in {0,1,2}, s in {0,1,2} → 9 examples
    Pos = hax.Axis("position", 6)
    doc = _arr([0, 1, 2, 3, 4, 5])
    pad = -1

    ds = SpliceDocumentDataset(
        Pos=Pos,
        doc_tokens=doc,
        pad_token_id=pad,
        content_length=4,
        min_copy_len=4,
        content_start_mode="slide_within",
    ).as_sync_dataset()

    assert len(ds) == 9
    # Last should correspond to (t=2,s=2): [pad,pad,2,3,4,5]
    np.testing.assert_array_equal(ds[-1].tokens.array, _arr([pad, pad, 2, 3, 4, 5]))


def test_offset_stride_is_respected():
    # S=6, L=4, K=3, offset_stride=2 → s only in {0,2,...}
    Pos = hax.Axis("position", 6)
    doc = _arr([0, 1, 2, 3])
    pad = -1

    ds = SpliceDocumentDataset(
        Pos=Pos,
        doc_tokens=doc,
        pad_token_id=pad,
        content_length=3,
        offset_stride=2,
        content_start_mode="slide_within",
    ).as_sync_dataset()

    # Count: t=0 → s in {0,2}; t=1 → {0,2}; t=2 (remaining=2) → s in {0,2,4} → total 7
    assert len(ds) == 7
    # Ensure we do not see s=1 left-anchored pattern for t=0
    toks = [ex.tokens.array.tolist() for ex in ds]
    forbidden = [pad, 0, 1, 2, pad, pad]
    assert forbidden not in toks
    # But we should see s=2 variant for t=0
    assert [pad, pad, 0, 1, 2, pad] in toks


def test_loss_mask_bounds_and_segments():
    # S=6, L=6, K=4, pick (t=1,s=1) → tokens [pad,1,2,3,4,pad]
    Pos = hax.Axis("position", 6)
    doc = _arr([0, 1, 2, 3, 4, 5])
    pad = -1

    ds = SpliceDocumentDataset(
        Pos=Pos,
        doc_tokens=doc,
        pad_token_id=pad,
        content_length=4,
        content_start_mode="slide_within",
    ).as_sync_dataset()

    # Locate (t=1,s=1): we know enumeration is (t outer, s inner)
    # t=0 → s=0,1,2 (3 items), so index 3 is (t=1,s=0), index 4 is (t=1,s=1)
    ex = ds[4]
    np.testing.assert_array_equal(ex.tokens.array, _arr([pad, 1, 2, 3, 4, pad]))
    # Loss mask on 1,2,3 (exclude last in-span)
    np.testing.assert_array_equal(ex.loss_mask.array, _arr([0, 1, 1, 1, 0, 0]))
    # Segment ids: 0 before s, 1 from s onward
    np.testing.assert_array_equal(ex.attn_mask.segment_ids[0].array, _arr([0, 1, 1, 1, 1, 1]))
