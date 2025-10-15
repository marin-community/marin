# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# tests/test_ragged_paged_attention.py
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
from levanter.inference.utils import INVALID
from levanter.layers import AttentionConfig, AttentionBackend, Attention
from levanter.layers.attention import AttentionMask, ragged_paged_attention, simple_attention_with_dropout
from levanter.layers.kv_cache import KvPageCache

SLOT = hax.Axis("slot", 4)  # page size
NUM_SLOTS = SLOT.size
KV_HEADS = hax.Axis("kv_head", 1)
QH = hax.Axis("q_heads_per_group", 1)
D = hax.Axis("head_size", 128)

KV_BS = 32  # must match constant inside kernel
SM_SCALE = 1 / math.sqrt(D.size)


def _build_random_case(rng, seq_lens):
    """Seq‑by‑seq contiguous paging with padding to KV_BS."""
    num_seqs = len(seq_lens)

    # token → sequence mapping and cumulative Q lengths
    tok_offsets = np.cumsum([0] + seq_lens)
    cu_q_lens = jnp.asarray(tok_offsets, dtype=jnp.int32)
    cu_q_lens = hax.named(cu_q_lens, "seq")

    # allocate enough pages for the longest sequence, but give each sequence its own slice
    pages_per_seq_raw = [(length + NUM_SLOTS - 1) // NUM_SLOTS for length in seq_lens]
    max_pages = KV_BS * ((max(pages_per_seq_raw) + KV_BS - 1) // KV_BS)  # pad to mult of 32
    page_indices = -jnp.ones((num_seqs, max_pages), dtype=jnp.int32)

    # generate actual page contents
    next_free_page = 0
    kv_data = []
    for sid, (qlen, p_raw) in enumerate(zip(seq_lens, pages_per_seq_raw)):
        pages = jnp.arange(next_free_page, next_free_page + p_raw, dtype=jnp.int32)
        page_indices = page_indices.at[sid, :p_raw].set(pages)
        next_free_page += p_raw

        # fill those pages with random k/v
        kv_pages_for_seq = hax.random.normal(
            jr.fold_in(rng, sid),
            # (hax.Axis("page", p_raw), SLOT, KV_HEADS.resize(2 * KV_HEADS.size), D),
            {"page": p_raw, "slot": NUM_SLOTS, "kv_head": 2 * KV_HEADS.size, "head_size": 128},
        )
        kv_data.append(kv_pages_for_seq)

    # stack pages, pad out to PAGE axis
    # ensure we have enough pages
    # kv_pages = hax.zeros(({PAGE, SLOT, KV_HEADS.resize(2 * KV_HEADS.size), D))
    num_pages_needed = next_free_page + 4
    kv_pages = hax.zeros({"page": num_pages_needed, "slot": 4, "kv_head": 2, "head_size": 128})
    kv_pages = kv_pages.at["page", :next_free_page].set(hax.concatenate("page", kv_data))

    # random queries for whole token axis
    total_tokens = tok_offsets[-1]
    this_position = hax.Axis("position", int(total_tokens))
    q = hax.random.normal(rng, (this_position, KV_HEADS, QH, D))

    kv_lens = jnp.asarray(seq_lens, dtype=jnp.int32)
    kv_lens = hax.named(kv_lens, "seq")

    page_indices = hax.named(page_indices, ("seq", "page"))

    return q, kv_pages, kv_lens, page_indices, cu_q_lens, jnp.array(num_seqs, dtype=jnp.int32)


def _build_incremental_case(rng, seq_lens, k_lens):
    """Like ``_build_random_case`` but query only contains the last ``k`` tokens.

    ``seq_lens`` gives the total tokens already in the KV cache for each
    sequence. ``k_lens`` is how many query tokens each sequence has. The KV
    cache still contains ``seq_lens`` tokens for every sequence.
    """
    q_full, kv_pages, kv_lens, page_indices, full_cu_q_lens, num_seqs = _build_random_case(rng, seq_lens)

    assert len(seq_lens) == len(k_lens)

    chunks = []
    new_offsets = [0]
    for sid, (total_len, k) in enumerate(zip(seq_lens, k_lens)):
        start = int(full_cu_q_lens["seq", sid]) + total_len - k
        chunks.append(q_full["position", hax.ds(start, k)])
        new_offsets.append(new_offsets[-1] + k)

    q = hax.concatenate("position", chunks)
    cu_q_lens = hax.named(jnp.asarray(new_offsets, dtype=jnp.int32), "seq")

    return q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs


def _reference_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, seq_lens):
    """Naïve per‑sequence causal soft‑max (slow but tiny)."""
    out_chunks = []
    for sid, qlen in enumerate(seq_lens):
        # slice query tokens for this sequence
        start = int(cu_q_lens["seq", sid])
        q_seq = q["position", hax.ds(start, qlen)]
        TOK_S = hax.Axis("position", qlen)

        # gather kv for the sequence
        n_pages = (kv_lens["seq", sid] + SLOT.size - 1) // SLOT.size
        pages = page_indices["seq", sid, "page", : n_pages.scalar()]
        kv_flat = kv_pages["page", pages, "slot", :].flatten_axes(("page", "slot"), "kv_position")
        kv_flat = kv_flat["kv_position", hax.ds(0, kv_lens["seq", sid].scalar())]

        k_seq = kv_flat["kv_head", 0::2]
        v_seq = kv_flat["kv_head", 1::2]

        # rename axes so they line up with dot_product_attention sig
        q_seq = q_seq.rename({"position": TOK_S.name})

        offset = kv_lens["seq", sid].scalar() - qlen
        mask = AttentionMask.causal(offset=offset)
        ref = simple_attention_with_dropout(
            "position", "kv_position", D, q_seq, k_seq, v_seq, mask=mask, scaling_factor=SM_SCALE
        )

        out_chunks.append(ref)

    return hax.concatenate("position", out_chunks)


# very loose tolerance. JAX uses a very loose tolerance for their ragged_attention tests.
def _rpa_tol() -> float:
    devices = jax.devices()
    # 2%?!?!
    return 2e-2 if any(device.platform == "tpu" for device in devices) else 1e-4


def test_ragged_paged_attention_single_seq():
    with jax.make_mesh((len(jax.devices()),), ("dp",)):
        rng = jr.PRNGKey(0)
        seq_lens = [1]  # one sequence
        q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs = _build_random_case(rng, seq_lens)

    ragged = ragged_paged_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, sm_scale=SM_SCALE)
    ref = _reference_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, seq_lens)

    assert ragged.axes == ref.axes
    tol = _rpa_tol()
    for i in range(len(ragged.array)):
        assert_trees_all_close(ragged.array[i], ref.array[i], atol=tol, rtol=tol, custom_message=f" at index {i}")
    # assert_trees_all_close(ragged.array[:-1], ref.array[:-1], atol=_rpa_tol(), rtol=_rpa_tol())
    # assert_trees_all_close(ragged.array[-1], ref.array[-1], atol=_rpa_tol(), rtol=_rpa_tol())
    # assert_trees_all_close(ragged.array, ref.array, atol=_rpa_tol(), rtol=_rpa_tol())


jit_rpa = jax.jit(ragged_paged_attention)


@pytest.mark.parametrize(
    "seq_lens", [[8], [8, 32, 16], [10, 37, 64], [34, 17], [9, 10, 34, 17], [64, 10, 37], [5, 15, 25, 35, 45]]
)
def test_ragged_paged_attention_multi_seq(seq_lens):
    rng = jr.PRNGKey(hash(tuple(seq_lens)))
    q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs = _build_random_case(rng, seq_lens)

    ragged = jit_rpa(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, sm_scale=SM_SCALE)
    ref = _reference_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, seq_lens)

    assert ragged.axes == ref.axes
    tol = _rpa_tol()
    assert_trees_all_close(ragged.array, ref.array, atol=tol, rtol=tol)


def test_ragged_paged_attention_incremental_single_seq():
    rng = jr.PRNGKey(2)
    seq_lens = [47]
    k_lens = [5]
    q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs = _build_incremental_case(rng, seq_lens, k_lens)

    ragged = jit_rpa(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, sm_scale=SM_SCALE)
    ref = _reference_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, k_lens)

    assert ragged.axes == ref.axes
    tol = _rpa_tol()
    assert_trees_all_close(ragged.array, ref.array, atol=tol, rtol=tol)


def test_ragged_paged_attention_incremental_multi_seq():
    rng = jr.PRNGKey(3)
    seq_lens = [10, 37, 64]
    k_lens = [1, 3, 9]
    q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs = _build_incremental_case(rng, seq_lens, k_lens)

    ragged = jit_rpa(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, sm_scale=SM_SCALE)
    ref = _reference_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, k_lens)

    assert ragged.axes == ref.axes
    tol = _rpa_tol()
    assert_trees_all_close(ragged.array, ref.array, atol=tol, rtol=tol)


# -----------------------------------------------------------------------------
# Tests moved from tests/test_paged_attention.py
# -----------------------------------------------------------------------------


@jax.jit
def _jit_paged_decode(attn, x, pos_ids, cache: KvPageCache, binfo: PageBatchInfo) -> tuple[NamedArray, KvPageCache]:
    return attn.paged_decode(x, cache, binfo, pos_ids=pos_ids, key=jrandom.PRNGKey(2))


def test_attention_paged_decode_matches_full_ar():
    Pos = Axis("position", 4)
    Embed = Axis("embed", 8)

    cfg = AttentionConfig(Embed=Embed, num_heads=2, num_kv_heads=2, rope=None, attn_backend=AttentionBackend.VANILLA)
    attn_key, x_key = jrandom.split(jrandom.PRNGKey(0))
    attn = Attention.init(cfg, key=attn_key)

    x = hax.random.normal(x_key, (Pos, Embed)) * 0.2
    full_out = attn(x, AttentionMask.causal(), key=jrandom.PRNGKey(1))

    pt = PageTable.init(max_pages=4, max_seqs=2, page_size=4, max_pages_per_seq=4)
    pt, seq_id = pt.assign_seq_id_to_seq()
    kv_cache = attn.empty_page_cache(pt, dtype=jnp.float32)
    out_chunks = []
    for i in range(Pos.size):
        # Compute pos_ids for this allocation using current seq_lens before allocation
        seg_ids = hax.named([seq_id], "position")
        # relative position inside this seg is 0 for this single token; absolute pos is current len
        abs_pos = pt.seq_lens["seq", seg_ids].array
        pos_ids = hax.named(abs_pos, "position")

        pt, binfo = pt.allocate_for_seq(seg_ids, pos_ids)

        x_tok = x[Pos, hax.dslice(i, 1)]
        out_tok, kv_cache = _jit_paged_decode(attn, x_tok, pos_ids, kv_cache, binfo)
        out_chunks.append(out_tok.array)

    decoded_arr = jnp.concatenate(out_chunks, axis=0)
    tol = _rpa_tol()
    assert_trees_all_close(full_out.array, decoded_arr, atol=tol, rtol=tol)


def test_attention_paged_decode_matches_full_prefill():
    Pos = Axis("position", 16)
    Embed = Axis("embed", 16)

    cfg = AttentionConfig(Embed=Embed, num_heads=2, num_kv_heads=2, rope=None, attn_backend=AttentionBackend.VANILLA)
    attn_key, x_key = jrandom.split(jrandom.PRNGKey(0))
    attn = Attention.init(cfg, key=attn_key)

    pt = PageTable.init(max_pages=4, max_seqs=2, page_size=4, max_pages_per_seq=4)
    pt, seq1 = pt.assign_seq_id_to_seq()
    pt, seq2 = pt.assign_seq_id_to_seq()

    x = hax.random.normal(x_key, (Pos, Embed)) * 0.2
    seg_ids = hax.named([0] * 4 + [1] * 3 + [INVALID] * 9, "position")
    pos_ids = hax.named(jnp.array([0, 1, 2, 3, 0, 1, 2] + [INVALID] * 9, dtype=jnp.int32), "position")
    pt, binfo = pt.allocate_for_seq(seg_ids, pos_ids)

    causal = AttentionMask.causal().with_segment_ids(seg_ids)
    full_out = attn(x, causal, key=jrandom.PRNGKey(1))

    kv_cache = attn.empty_page_cache(pt, dtype=jnp.float32)

    # Compute absolute pos ids for this batch from current seq_lens
    def _relative_positions(seg_ids):
        idx = jnp.arange(seg_ids.shape[0])
        is_start = jnp.concatenate([jnp.array([True]), seg_ids[1:] != seg_ids[:-1]])
        start_idx = idx * is_start.astype(idx.dtype)
        seg_start = jax.lax.associative_scan(jnp.maximum, start_idx)
        return idx - seg_start

    rel_pos = _relative_positions(seg_ids.array)
    starts = pt.seq_lens["seq", seg_ids].array
    pos_ids = hax.named(starts + rel_pos, "position")

    decode_out, _ = _jit_paged_decode(attn, x, pos_ids, kv_cache, binfo)

    # we only care about the first 7 positions, since the rest are padding
    full_out = full_out["position", hax.dslice(0, 7)]
    decode_out = decode_out["position", hax.dslice(0, 7)]

    tol = _rpa_tol()
    assert_trees_all_close(full_out.array, decode_out.array, atol=tol, rtol=tol)


@pytest.mark.parametrize("prefix_size", [1, 2, 3])
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 8])
@pytest.mark.parametrize("seq_ids", [[0, 1], [1, 0], [2, 0], [0, 2], [2, 1]])
def test_attention_paged_decode_prefill_in_chunks(prefix_size, chunk_size, seq_ids):
    Pos = Axis("position", prefix_size + 4 * chunk_size)
    Embed = Axis("embed", 16)

    cfg = AttentionConfig(Embed=Embed, num_heads=2, num_kv_heads=2, rope=None, attn_backend=AttentionBackend.VANILLA)
    attn_key, x_key = jrandom.split(jrandom.PRNGKey(0))
    attn = Attention.init(cfg, key=attn_key)
    tol = _rpa_tol()

    max_pages_per_seq = math.ceil((prefix_size + 4 * chunk_size) / NUM_SLOTS)

    pt = PageTable.init(max_pages=max_pages_per_seq * 3, max_seqs=3, page_size=4, max_pages_per_seq=max_pages_per_seq)
    seq1 = seq_ids[0]
    seq2 = seq_ids[1]
    pt, _seq1 = pt.assign_seq_id_to_seq(seq1)
    pt, _seq2 = pt.assign_seq_id_to_seq(seq2)
    assert _seq1 == seq1
    assert _seq2 == seq2
    kv_cache = attn.empty_page_cache(pt, dtype=jnp.float32)

    x = hax.random.normal(x_key, (Pos, Embed)) * 0.2
    x0 = x["position", hax.dslice(0, Pos.size)]
    x1 = x0
    full_out = attn(hax.stack("batch", [x0, x1]), AttentionMask.causal(), key=jrandom.PRNGKey(1))

    # prefill prefix
    outputs0 = []
    outputs1 = []
    x_prefill = hax.concatenate("position", [x0[Pos, hax.dslice(0, prefix_size)], x1[Pos, hax.dslice(0, prefix_size)]])
    tokens = hax.named([seq1] * prefix_size + [seq2] * prefix_size, Axis("position", 2 * prefix_size))
    pos_ids = hax.concatenate(
        "position",
        [
            hax.arange({"position": prefix_size}, dtype=jnp.int32),
            hax.arange({"position": prefix_size}, dtype=jnp.int32),
        ],
    )
    pt, binfo = pt.allocate_for_seq(tokens, pos_ids)

    with jax.sharding.set_mesh(jax.make_mesh((len(jax.devices()),), ("model",))):
        out, kv_cache = _jit_paged_decode(attn, x_prefill, pos_ids, kv_cache, binfo)
        outputs0.append(out["position", hax.dslice(0, prefix_size)])
        outputs1.append(out["position", hax.dslice(prefix_size, prefix_size)])

        start0 = start1 = prefix_size

        # decode rest in chunks
        for i in range(prefix_size, Pos.size, chunk_size):
            tok_axis = Axis("position", 2 * chunk_size)
            tokens = hax.named([seq1] * chunk_size + [seq2] * chunk_size, tok_axis)

            pos_ids = hax.concatenate(
                "position",
                [
                    hax.arange({"position": chunk_size}, start=start0, dtype=jnp.int32),
                    hax.arange({"position": chunk_size}, start=start1, dtype=jnp.int32),
                ],
            )

            start0 += chunk_size
            start1 += chunk_size

            pt, binfo = pt.allocate_for_seq(tokens, pos_ids)

            x_chunk = hax.concatenate(
                "position",
                [x0[Pos, hax.dslice(i, chunk_size)], x1[Pos, hax.dslice(i, chunk_size)]],
            )
            out_chunk, kv_cache = _jit_paged_decode(attn, x_chunk, pos_ids, kv_cache, binfo)
            outputs0.append(out_chunk["position", hax.dslice(0, chunk_size)])
            outputs1.append(out_chunk["position", hax.dslice(chunk_size, chunk_size)])

        outputs0_cat = hax.concatenate("position", outputs0)
        outputs1_cat = hax.concatenate("position", outputs1)
        decoded_arr = hax.stack("batch", [outputs0_cat, outputs1_cat])
        assert_trees_all_close(full_out.array, decoded_arr.array, atol=tol, rtol=tol)


def test_attention_paged_decode_ragged_fill_in_chunks():
    B = Axis("batch", 2)
    Pos = Axis("position", 8)
    Embed = Axis("embed", 16)

    cfg = AttentionConfig(Embed=Embed, num_heads=2, num_kv_heads=2, attn_backend=AttentionBackend.VANILLA)
    attn_key, x_key = jrandom.split(jrandom.PRNGKey(0))
    attn = Attention.init(cfg, key=attn_key)
    tol = _rpa_tol()
    # x = hax.random.normal(x_key, (B, Pos, Embed)) * 0.2
    x = hax.arange((B, Pos, Embed), start=-2, step=0.1, dtype=jnp.float32)
    full_out = attn(x, AttentionMask.causal(), key=jrandom.PRNGKey(1))

    pt = PageTable.init(max_pages=8, max_seqs=2, page_size=4, max_pages_per_seq=4)
    pt, seq1 = pt.assign_seq_id_to_seq()
    pt, seq2 = pt.assign_seq_id_to_seq()
    kv_cache = attn.empty_page_cache(pt, dtype=jnp.float32)

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

        pt, binfo = pt.allocate_for_seq(seg_ids, pos_ids)

        x_chunk = hax.concatenate(
            "position",
            [x0[Pos, hax.dslice(off0, step0)], x1[Pos, hax.dslice(off1, step1)]],
        )

        output, kv_cache = _jit_paged_decode(attn, x_chunk, pos_ids=pos_ids, cache=kv_cache, binfo=binfo)
        outputs0.append(output["position", hax.dslice(0, step0)])
        outputs1.append(output["position", hax.dslice(step0, step1)])

        # check each chunk individually
        assert_trees_all_close(
            outputs0[-1].array,
            full_out[B, 0, "position", hax.dslice(off0, step0)].array,
            atol=tol,
            rtol=tol,
        )
        assert_trees_all_close(
            outputs1[-1].array,
            full_out[B, 1, "position", hax.dslice(off1, step1)].array,
            atol=tol,
            rtol=tol,
        )

        off0 += step0
        off1 += step1

    outputs0_cat = hax.concatenate("position", outputs0)
    outputs1_cat = hax.concatenate("position", outputs1)

    decoded_arr = hax.stack("batch", [outputs0_cat, outputs1_cat])
    assert_trees_all_close(full_out.array, decoded_arr.array, atol=tol, rtol=tol)
