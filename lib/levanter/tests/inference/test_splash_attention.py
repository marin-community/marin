# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Splash attention decode with paged KV cache."""

import math

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from chex import assert_trees_all_close

import haliax as hax
from haliax import Axis, NamedArray

from levanter.inference.jit_scheduler import SequenceTable
from levanter.inference.page_table import PageBatchInfo, PageTable
from levanter.inference.utils import INVALID
from levanter.layers import Attention, AttentionBackend, AttentionConfig
from levanter.layers.attention import AttentionMask
from test_utils import use_test_mesh


def _splash_tol() -> float:
    """Tolerance for Splash attention tests.
    
    Splash attention uses different numerics than vanilla, so we use relaxed tolerances.
    """
    devices = jax.devices()
    # Use 2% tolerance on TPU, tighter on CPU/GPU
    return 2e-2 if any(device.platform == "tpu" for device in devices) else 1e-4


@pytest.mark.parametrize("seq_len", [128, 256, 384])
@pytest.mark.parametrize("num_heads", [2, 4])
def test_splash_decode_single_sequence(seq_len, num_heads):
    """Test splash_decode matches paged_decode for a single sequence."""
    # Skip on non-TPU since Splash may not be available
    if jax.default_backend() != "tpu":
        pytest.skip("Splash attention only available on TPU")
    
    # Setup: page_size = seq_len so the entire sequence fits in one page
    Pos = Axis("position", seq_len)
    Embed = Axis("embed", 64)
    page_size = seq_len
    
    cfg = AttentionConfig(
        Embed=Embed,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        rope=None,
        attn_backend=AttentionBackend.VANILLA,
    )
    
    attn_key, x_key = jrandom.split(jrandom.PRNGKey(0))
    attn = Attention.init(cfg, key=attn_key)
    
    # Create page table with page_size = seq_len
    pt = PageTable.init(max_pages=4, max_seqs=2, page_size=page_size, max_pages_per_seq=1)
    sequences = SequenceTable.init(pt.max_seqs, pt.pages_per_seq, pt.page_size)
    
    # Reserve a slot
    sequences, assigned = sequences.reserve_slot(0)
    assert int(assigned) == 0
    
    with use_test_mesh():
        # Create input tokens
        x = hax.random.normal(x_key, (Pos, Embed)) * 0.2
        
        # Reference: use regular attention on the full sequence
        full_out = attn(x, AttentionMask.causal(), key=jrandom.PRNGKey(1))
        
        # Test paged_decode: pack all tokens into one prefill
        kv_cache = attn.empty_page_cache(pt.spec(), dtype=jnp.float32)
        
        # Allocate pages for the sequence
        pages_needed = (seq_len + page_size - 1) // page_size
        page_list = list(range(pages_needed))
        page_indices = hax.full({"seq": pt.max_seqs, "page": pt.pages_per_seq}, INVALID, dtype=jnp.int32)
        page_indices = page_indices.at["seq", 0, "page", :pages_needed].set(jnp.array(page_list, dtype=jnp.int32))
        
        slot_ids = hax.full({"seq": pt.max_seqs}, INVALID, dtype=jnp.int32)
        slot_ids = slot_ids.at["seq", 0].set(0)
        
        seq_lens = hax.full({"seq": pt.max_seqs}, 0, dtype=jnp.int32)
        seq_lens = seq_lens.at["seq", 0].set(seq_len)
        
        cu_q_lens = hax.full({"seq": pt.max_seqs + 1}, 0, dtype=jnp.int32)
        cu_q_lens = cu_q_lens.at["seq", 1].set(seq_len)
        
        new_token_dests = hax.arange(Pos, dtype=jnp.int32)
        
        batch_info = PageBatchInfo(
            slot_ids=slot_ids,
            page_indices=page_indices,
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            num_seqs=jnp.array(1, dtype=jnp.int32),
            new_token_dests=new_token_dests,
            page_size=page_size,
        )
        
        pos_ids = hax.arange(Pos, dtype=jnp.int32)
        
        # Run paged_decode
        paged_out, kv_cache_paged = attn.paged_decode(
            x, kv_cache, batch_info, pos_ids=pos_ids, key=jrandom.PRNGKey(1)
        )
        
        # Run splash_decode with same cache
        kv_cache_splash = attn.empty_page_cache(pt.spec(), dtype=jnp.float32)
        splash_out, kv_cache_splash = attn.splash_decode(
            x, kv_cache_splash, batch_info, pos_ids=pos_ids, key=jrandom.PRNGKey(1)
        )
        
        tol = _splash_tol()
        # Compare outputs
        assert_trees_all_close(splash_out, paged_out, rtol=tol, atol=tol)
        # Also check against reference
        assert_trees_all_close(splash_out, full_out, rtol=tol, atol=tol)


@pytest.mark.parametrize("prefix_size", [64, 128])
@pytest.mark.parametrize("decode_steps", [1, 2, 4])
def test_splash_decode_incremental(prefix_size, decode_steps):
    """Test splash_decode with incremental decoding after prefill."""
    if jax.default_backend() != "tpu":
        pytest.skip("Splash attention only available on TPU")
    
    max_seq_len = prefix_size + decode_steps
    Pos = Axis("position", max_seq_len)
    Embed = Axis("embed", 64)
    page_size = max_seq_len
    
    cfg = AttentionConfig(
        Embed=Embed,
        num_heads=2,
        num_kv_heads=2,
        rope=None,
        attn_backend=AttentionBackend.VANILLA,
    )
    
    attn_key, x_key = jrandom.split(jrandom.PRNGKey(0))
    attn = Attention.init(cfg, key=attn_key)
    
    pt = PageTable.init(max_pages=4, max_seqs=2, page_size=page_size, max_pages_per_seq=1)
    
    with use_test_mesh():
        # Generate full sequence
        x_full = hax.random.normal(x_key, (Pos, Embed)) * 0.2
        
        # Reference: full attention
        full_out = attn(x_full, AttentionMask.causal(), key=jrandom.PRNGKey(1))
        
        # Initialize caches
        kv_cache_splash = attn.empty_page_cache(pt.spec(), dtype=jnp.float32)
        kv_cache_paged = attn.empty_page_cache(pt.spec(), dtype=jnp.float32)
        
        page_indices = hax.full({"seq": pt.max_seqs, "page": pt.pages_per_seq}, INVALID, dtype=jnp.int32)
        page_indices = page_indices.at["seq", 0, "page", 0].set(0)
        
        slot_ids = hax.full({"seq": pt.max_seqs}, INVALID, dtype=jnp.int32)
        slot_ids = slot_ids.at["seq", 0].set(0)
        
        # Prefill with prefix
        x_prefix = x_full["position", hax.dslice(0, prefix_size)]
        PosPrefix = Axis("position", prefix_size)
        
        seq_lens = hax.full({"seq": pt.max_seqs}, 0, dtype=jnp.int32)
        seq_lens = seq_lens.at["seq", 0].set(prefix_size)
        
        cu_q_lens = hax.full({"seq": pt.max_seqs + 1}, 0, dtype=jnp.int32)
        cu_q_lens = cu_q_lens.at["seq", 1].set(prefix_size)
        
        new_token_dests = hax.arange(PosPrefix, dtype=jnp.int32)
        
        batch_info = PageBatchInfo(
            slot_ids=slot_ids,
            page_indices=page_indices,
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            num_seqs=jnp.array(1, dtype=jnp.int32),
            new_token_dests=new_token_dests,
            page_size=page_size,
        )
        
        pos_ids_prefix = hax.arange(PosPrefix, dtype=jnp.int32)
        
        splash_out_prefix, kv_cache_splash = attn.splash_decode(
            x_prefix, kv_cache_splash, batch_info, pos_ids=pos_ids_prefix, key=jrandom.PRNGKey(1)
        )
        
        paged_out_prefix, kv_cache_paged = attn.paged_decode(
            x_prefix, kv_cache_paged, batch_info, pos_ids=pos_ids_prefix, key=jrandom.PRNGKey(1)
        )
        
        # Incrementally decode remaining tokens
        outputs_splash = [splash_out_prefix]
        outputs_paged = [paged_out_prefix]
        
        for step in range(decode_steps):
            token_pos = prefix_size + step
            x_token = x_full["position", token_pos : token_pos + 1]
            PosOne = Axis("position", 1)
            
            seq_lens = seq_lens.at["seq", 0].set(token_pos + 1)
            cu_q_lens = hax.full({"seq": pt.max_seqs + 1}, 0, dtype=jnp.int32)
            cu_q_lens = cu_q_lens.at["seq", 1].set(1)
            
            new_token_dests = hax.full(PosOne, token_pos, dtype=jnp.int32)
            
            batch_info = PageBatchInfo(
                slot_ids=slot_ids,
                page_indices=page_indices,
                seq_lens=seq_lens,
                cu_q_lens=cu_q_lens,
                num_seqs=jnp.array(1, dtype=jnp.int32),
                new_token_dests=new_token_dests,
                page_size=page_size,
            )
            
            pos_ids_one = hax.full(PosOne, token_pos, dtype=jnp.int32)
            
            out_splash, kv_cache_splash = attn.splash_decode(
                x_token, kv_cache_splash, batch_info, pos_ids=pos_ids_one, key=jrandom.PRNGKey(2 + step)
            )
            
            out_paged, kv_cache_paged = attn.paged_decode(
                x_token, kv_cache_paged, batch_info, pos_ids=pos_ids_one, key=jrandom.PRNGKey(2 + step)
            )
            
            outputs_splash.append(out_splash)
            outputs_paged.append(out_paged)
        
        # Concatenate outputs
        full_splash = hax.concatenate("position", outputs_splash)
        full_paged = hax.concatenate("position", outputs_paged)
        
        tol = _splash_tol()
        # Compare splash vs paged
        assert_trees_all_close(full_splash, full_paged, rtol=tol, atol=tol)
        # Compare against reference
        assert_trees_all_close(full_splash, full_out, rtol=tol, atol=tol)


@pytest.mark.parametrize("seq_lens", [[128, 256], [256, 128], [128, 128]])
def test_splash_decode_batched(seq_lens):
    """Test splash_decode with multiple sequences in a batch."""
    if jax.default_backend() != "tpu":
        pytest.skip("Splash attention only available on TPU")
    
    max_seq_len = max(seq_lens)
    total_tokens = sum(seq_lens)
    Pos = Axis("position", total_tokens)
    Embed = Axis("embed", 64)
    page_size = max_seq_len
    
    cfg = AttentionConfig(
        Embed=Embed,
        num_heads=2,
        num_kv_heads=2,
        rope=None,
        attn_backend=AttentionBackend.VANILLA,
    )
    
    attn_key, x_key = jrandom.split(jrandom.PRNGKey(0))
    attn = Attention.init(cfg, key=attn_key)
    
    num_seqs = len(seq_lens)
    pt = PageTable.init(max_pages=num_seqs + 2, max_seqs=num_seqs, page_size=page_size, max_pages_per_seq=1)
    
    with use_test_mesh():
        # Generate packed input
        x = hax.random.normal(x_key, (Pos, Embed)) * 0.2
        
        # Setup page indices and metadata
        page_indices = hax.full({"seq": pt.max_seqs, "page": pt.pages_per_seq}, INVALID, dtype=jnp.int32)
        for i in range(num_seqs):
            page_indices = page_indices.at["seq", i, "page", 0].set(i)
        
        slot_ids = hax.full({"seq": pt.max_seqs}, INVALID, dtype=jnp.int32)
        for i in range(num_seqs):
            slot_ids = slot_ids.at["seq", i].set(i)
        
        seq_lens_arr = hax.full({"seq": pt.max_seqs}, 0, dtype=jnp.int32)
        for i, slen in enumerate(seq_lens):
            seq_lens_arr = seq_lens_arr.at["seq", i].set(slen)
        
        cu_q_lens = hax.full({"seq": pt.max_seqs + 1}, 0, dtype=jnp.int32)
        cumsum = 0
        for i, slen in enumerate(seq_lens):
            cumsum += slen
            cu_q_lens = cu_q_lens.at["seq", i + 1].set(cumsum)
        
        # Build new_token_dests for packed sequences
        new_token_dests_list = []
        for slen in seq_lens:
            new_token_dests_list.extend(range(slen))
        new_token_dests = hax.named(jnp.array(new_token_dests_list, dtype=jnp.int32), Pos)
        
        batch_info = PageBatchInfo(
            slot_ids=slot_ids,
            page_indices=page_indices,
            seq_lens=seq_lens_arr,
            cu_q_lens=cu_q_lens,
            num_seqs=jnp.array(num_seqs, dtype=jnp.int32),
            new_token_dests=new_token_dests,
            page_size=page_size,
        )
        
        # Position IDs for packed sequences
        pos_ids_list = []
        for slen in seq_lens:
            pos_ids_list.extend(range(slen))
        pos_ids = hax.named(jnp.array(pos_ids_list, dtype=jnp.int32), Pos)
        
        # Run both decode methods
        kv_cache_splash = attn.empty_page_cache(pt.spec(), dtype=jnp.float32)
        kv_cache_paged = attn.empty_page_cache(pt.spec(), dtype=jnp.float32)
        
        splash_out, _ = attn.splash_decode(
            x, kv_cache_splash, batch_info, pos_ids=pos_ids, key=jrandom.PRNGKey(1)
        )
        
        paged_out, _ = attn.paged_decode(
            x, kv_cache_paged, batch_info, pos_ids=pos_ids, key=jrandom.PRNGKey(1)
        )
        
        tol = _splash_tol()
        assert_trees_all_close(splash_out, paged_out, rtol=tol, atol=tol)


def test_splash_decode_fallback():
    """Test that splash_decode falls back gracefully when Splash is unavailable."""
    # This test should pass on all platforms
    Pos = Axis("position", 128)
    Embed = Axis("embed", 64)
    page_size = 128
    
    cfg = AttentionConfig(
        Embed=Embed,
        num_heads=2,
        num_kv_heads=2,
        rope=None,
        attn_backend=AttentionBackend.VANILLA,
    )
    
    attn_key, x_key = jrandom.split(jrandom.PRNGKey(0))
    attn = Attention.init(cfg, key=attn_key)
    
    pt = PageTable.init(max_pages=4, max_seqs=2, page_size=page_size, max_pages_per_seq=1)
    
    with use_test_mesh():
        x = hax.random.normal(x_key, (Pos, Embed)) * 0.2
        
        page_indices = hax.full({"seq": pt.max_seqs, "page": pt.pages_per_seq}, INVALID, dtype=jnp.int32)
        page_indices = page_indices.at["seq", 0, "page", 0].set(0)
        
        slot_ids = hax.full({"seq": pt.max_seqs}, INVALID, dtype=jnp.int32)
        slot_ids = slot_ids.at["seq", 0].set(0)
        
        seq_lens = hax.full({"seq": pt.max_seqs}, 0, dtype=jnp.int32)
        seq_lens = seq_lens.at["seq", 0].set(128)
        
        cu_q_lens = hax.full({"seq": pt.max_seqs + 1}, 0, dtype=jnp.int32)
        cu_q_lens = cu_q_lens.at["seq", 1].set(128)
        
        new_token_dests = hax.arange(Pos, dtype=jnp.int32)
        
        batch_info = PageBatchInfo(
            slot_ids=slot_ids,
            page_indices=page_indices,
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            num_seqs=jnp.array(1, dtype=jnp.int32),
            new_token_dests=new_token_dests,
            page_size=page_size,
        )
        
        pos_ids = hax.arange(Pos, dtype=jnp.int32)
        
        kv_cache = attn.empty_page_cache(pt.spec(), dtype=jnp.float32)
        
        # This should not raise, even on non-TPU platforms (will use fallback)
        out, _ = attn.splash_decode(x, kv_cache, batch_info, pos_ids=pos_ids, key=jrandom.PRNGKey(1))
        
        # Verify output shape is correct
        assert out.axis_size("position") == 128
        assert out.axis_size("embed") == 64
