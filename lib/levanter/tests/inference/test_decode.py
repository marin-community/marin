# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# tests/test_decode.py
import jax
import math

import jax.numpy as jnp
import jax.random as jr
import jax.random as jrandom
import numpy as np
import pytest
from chex import assert_trees_all_close

import haliax as hax
from haliax import NamedArray, Axis

from levanter.inference.page_table import PageBatchInfo, PageTable
from levanter.inference.jit_scheduler import SequenceTable
from levanter.inference.utils import INVALID
from levanter.layers import AttentionConfig, AttentionBackend, Attention
from levanter.layers.attention import AttentionMask, simple_attention_with_dropout
from levanter.layers.kv_cache import KvPageCache
from test_utils import use_test_mesh

SLOT = hax.Axis("slot", 4)  # page size
NUM_SLOTS = SLOT.size
KV_HEADS = hax.Axis("kv_head", 1)
QH = hax.Axis("q_heads_per_group", 1)
D = hax.Axis("head_size", 128)

KV_BS = 32  # must match constant inside kernel
SM_SCALE = 1 / math.sqrt(D.size)


def _tol() -> float:
    devices = jax.devices()
    return 2e-3 if any(device.platform == "tpu" for device in devices) else 1e-4


# -----------------------------------------------------------------------------
# Tests for decode
# -----------------------------------------------------------------------------


@jax.jit
def _jit_decode(attn, x, pos_ids, cache: KvPageCache, binfo: PageBatchInfo) -> tuple[NamedArray, KvPageCache]:
    return attn.decode(x, cache, binfo, pos_ids=pos_ids, key=jrandom.PRNGKey(2))


def _run_attention_decode_matches_full_ar(pos_size: int, embed_size: int, attn_backend: AttentionBackend):
    """Helper to test attention decode matches full autoregressive attention."""
    Pos = Axis("position", pos_size)
    Embed = Axis("embed", embed_size)

    cfg = AttentionConfig(Embed=Embed, num_heads=2, num_kv_heads=2, rope=None, attn_backend=attn_backend)
    attn_key, x_key = jrandom.split(jrandom.PRNGKey(0))
    attn = Attention.init(cfg, key=attn_key)

    x = hax.random.normal(x_key, (Pos, Embed)) * 0.2
    full_out = attn(x, AttentionMask.causal(), key=jrandom.PRNGKey(1))

    # page size must be equal to max_seq_len
    pt = PageTable.init(max_pages=4, max_seqs=2, page_size=Pos.size, max_pages_per_seq=1)
    sequences = SequenceTable.init(pt.max_seqs, pt.pages_per_seq, pt.page_size)
    sequences, seq_id_arr = sequences.reserve_slot()
    seq_id = int(seq_id_arr)
    kv_cache = attn.empty_page_cache(pt.spec(), dtype=jnp.float32)
    out_chunks = []
    for i in range(Pos.size):
        # Compute pos_ids for this allocation using current seq_lens before allocation
        seg_ids = hax.named([seq_id], "position")
        # relative position inside this seg is 0 for this single token; absolute pos is current len
        abs_pos = sequences.seq_lens["seq", seg_ids].array
        pos_ids = hax.named(abs_pos, "position")

        sequences, pt, binfo = sequences.allocate_for_seq(pt, seg_ids, pos_ids)

        x_tok = x[Pos, hax.dslice(i, 1)]
        out_tok, kv_cache = _jit_decode(attn, x_tok, pos_ids, kv_cache, binfo)
        out_chunks.append(out_tok.array)

    decoded_arr = jnp.concatenate(out_chunks, axis=0)
    tol = _tol()
    assert_trees_all_close(full_out.array, decoded_arr, atol=tol, rtol=tol)


def test_attention_decode_matches_full_ar():
    _run_attention_decode_matches_full_ar(pos_size=4, embed_size=8, attn_backend=AttentionBackend.VANILLA)


def test_attention_decode_matches_full_ar_splash():
    # Splash requires head_dim to be a multiple of 128; 256 / 2 heads = 128 per head
    with use_test_mesh():
        _run_attention_decode_matches_full_ar(pos_size=128, embed_size=256, attn_backend=AttentionBackend.SPLASH)


def _run_attention_decode_full_prefill(pos_size: int, embed_size: int, attn_backend: AttentionBackend):
    """Helper to test attention decode matches full prefill with ragged sequences."""
    Pos = Axis("position", pos_size)
    Embed = Axis("embed", embed_size)

    cfg = AttentionConfig(Embed=Embed, num_heads=2, num_kv_heads=2, rope=None, attn_backend=attn_backend)
    attn_key, x_key = jrandom.split(jrandom.PRNGKey(0))
    attn = Attention.init(cfg, key=attn_key)

    # page size must be equal to max_seq_len
    pt = PageTable.init(max_pages=4, max_seqs=2, page_size=Pos.size, max_pages_per_seq=1)
    sequences = SequenceTable.init(pt.max_seqs, pt.pages_per_seq, pt.page_size)
    sequences, seq1_arr = sequences.reserve_slot(0)
    sequences, seq2_arr = sequences.reserve_slot(1)

    x = hax.random.normal(x_key, (Pos, Embed)) * 0.2
    
    # Create segment_ids with two sequences and padding, scaled to pos_size
    # Keep the same proportions: seq0=4, seq1=3, padding=9 (out of 16) -> scale proportionally
    seq0_len = max(1, pos_size // 4)
    seq1_len = max(1, (pos_size * 3) // 16)
    padding_len = pos_size - seq0_len - seq1_len
    
    seg_ids = hax.named([0] * seq0_len + [1] * seq1_len + [INVALID] * padding_len, Pos)
    pos_ids = hax.named(
        jnp.array(list(range(seq0_len)) + list(range(seq1_len)) + [INVALID] * padding_len, dtype=jnp.int32),
        Pos
    )
    sequences, pt, binfo = sequences.allocate_for_seq(pt, seg_ids, pos_ids)

    causal = AttentionMask.causal().with_segment_ids(seg_ids)
    full_out = attn(x, causal, key=jrandom.PRNGKey(1))

    kv_cache = attn.empty_page_cache(pt.spec(), dtype=jnp.float32)

    # Compute absolute pos ids for this batch from current seq_lens
    def _relative_positions(seg_ids):
        idx = jnp.arange(seg_ids.shape[0])
        is_start = jnp.concatenate([jnp.array([True]), seg_ids[1:] != seg_ids[:-1]])
        start_idx = idx * is_start.astype(idx.dtype)
        seg_start = jax.lax.associative_scan(jnp.maximum, start_idx)
        return idx - seg_start

    rel_pos = _relative_positions(seg_ids.array)
    starts = sequences.seq_lens["seq", seg_ids].array
    pos_ids = hax.named(starts + rel_pos, "position")

    decode_out, _ = _jit_decode(attn, x, pos_ids, kv_cache, binfo)

    # we only care about the non-padding positions
    valid_len = seq0_len + seq1_len
    full_out = full_out["position", hax.dslice(0, valid_len)]
    decode_out = decode_out["position", hax.dslice(0, valid_len)]

    tol = _tol()
    assert_trees_all_close(full_out.array, decode_out.array, atol=tol, rtol=tol)


def test_attention_decode_matches_full_prefill():
    _run_attention_decode_full_prefill(pos_size=16, embed_size=16, attn_backend=AttentionBackend.VANILLA)


def test_attention_decode_matches_full_prefill_splash():
    # Splash requires head_dim to be a multiple of 128; 256 / 2 heads = 128 per head
    with use_test_mesh():
        _run_attention_decode_full_prefill(pos_size=128, embed_size=256, attn_backend=AttentionBackend.SPLASH)


def test_attention_decode_ragged_fill_in_chunks():
    B = Axis("batch", 2)
    Pos = Axis("position", 8)
    Embed = Axis("embed", 16)

    cfg = AttentionConfig(Embed=Embed, num_heads=2, num_kv_heads=2, attn_backend=AttentionBackend.VANILLA)
    attn_key, x_key = jrandom.split(jrandom.PRNGKey(0))
    attn = Attention.init(cfg, key=attn_key)
    tol = _tol()
    # x = hax.random.normal(x_key, (B, Pos, Embed)) * 0.2
    x = hax.arange((B, Pos, Embed), start=-2, step=0.1, dtype=jnp.float32)
    full_out = attn(x, AttentionMask.causal(), key=jrandom.PRNGKey(1))

    # page size must be equal to max_seq_len
    pt = PageTable.init(max_pages=8, max_seqs=2, page_size=Pos.size, max_pages_per_seq=1)
    sequences = SequenceTable.init(pt.max_seqs, pt.pages_per_seq, pt.page_size)
    sequences, seq1_arr = sequences.reserve_slot(0)
    sequences, seq2_arr = sequences.reserve_slot(1)
    seq1 = int(seq1_arr)
    seq2 = int(seq2_arr)
    kv_cache = attn.empty_page_cache(pt.spec(), dtype=jnp.float32)

    x0 = x[B, 0]
    x1 = x[B, 1]

    chunk_sizes = [[4, 2], [0, 1], [0, 1], [2, 1], [1, 2], [1, 1]]
    off0 = off1 = 0
    outputs0 = []
    outputs1 = []

    for step0, step1 in chunk_sizes:
        tok_axis = Axis("position", step0 + step1)
        seg_ids = hax.named([seq1] * step0 + [seq2] * step1, tok_axis)
        pos_ids = hax.concatenate(
            "position",
            [
                hax.arange({"position": step0}, start=off0, dtype=jnp.int32),
                hax.arange({"position": step1}, start=off1, dtype=jnp.int32),
            ],
        )

        sequences, pt, binfo = sequences.allocate_for_seq(pt, seg_ids, pos_ids)

        x_chunk = hax.concatenate(
            "position",
            [x0[Pos, hax.dslice(off0, step0)], x1[Pos, hax.dslice(off1, step1)]],
        )

        output, kv_cache = _jit_decode(attn, x_chunk, pos_ids=pos_ids, cache=kv_cache, binfo=binfo)
        outputs0.append(output["position", hax.dslice(0, step0)])
        outputs1.append(output["position", hax.dslice(step0, step1)])

        # check each chunk individually
        assert_trees_all_close(
            outputs0[-1].array,
            full_out[B, 0, "position", hax.dslice(off0, step0)].array,
            atol=1e-1,
            rtol=1e-2
        )
        assert_trees_all_close(
            outputs1[-1].array,
            full_out[B, 1, "position", hax.dslice(off1, step1)].array,
            atol=1e-1,
            rtol=1e-2
        )

        off0 += step0
        off1 += step1

    outputs0_cat = hax.concatenate("position", outputs0)
    outputs1_cat = hax.concatenate("position", outputs1)

    decoded_arr = hax.stack("batch", [outputs0_cat, outputs1_cat])
    assert_trees_all_close(full_out.array, decoded_arr.array, atol=1e-1, rtol=1e-2)
