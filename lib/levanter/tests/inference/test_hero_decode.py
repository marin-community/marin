# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax import Axis
from levanter.grug.sharding import compact_grug_mesh
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.models.hero import GlobalQueryLogScaling, HeroConfig
from levanter.models.hero_model import HeroLMHeadModel
from levanter.testing.devices import use_test_devices


def _assert_mixed_paged_decode_matches_forward(model):
    """Check packed chunked requests, noncontiguous pages, and padding against full forward."""
    # Keep token buffers divisible by the data mesh, including padding after the valid prefix.
    capacity = 8 * jax.device_count()
    token_axis = Axis("position", capacity)
    sequences = [np.array([2, 8, 3, 7, 1, 9]), np.array([6, 4, 10, 5, 11])]
    page_indices = np.array([[2, 4, 1], [3, 0, 5]], dtype=np.int32)
    phases = [((0, 3), (0, 2)), ((3, 2), (2, 1)), ((5, 1), (3, 2))]
    full_forward = hax.named_jit(lambda m, ids: m(ids))
    reference = []
    for seq in sequences:
        ids = hax.named(
            jnp.broadcast_to(jnp.asarray(seq, dtype=jnp.int32), (jax.device_count(), len(seq))),
            (Axis("batch", jax.device_count()), Axis("position", len(seq))),
        )
        reference.append(np.asarray(full_forward(model, ids).array)[0])
    cache = model.initial_cache(PageTableSpec(num_pages=6, page_size=2, max_seqs=2), dtype=jnp.float32)
    decode = hax.named_jit(
        lambda m, ids, state, info, pos, select=None: m.decode(ids, state, info, pos, logit_positions=select)
    )
    for phase_index, phase in enumerate(phases):
        lengths = [length for _, length in phase]
        n = sum(lengths)
        ids = np.zeros(capacity, dtype=np.int32)
        positions = np.zeros(capacity, dtype=np.int32)
        dests = np.full(capacity, -1, dtype=np.int32)
        expected = []
        offset = 0
        for seq_id, (start, length) in enumerate(phase):
            pos = np.arange(start, start + length)
            ids[offset : offset + length] = sequences[seq_id][pos]
            positions[offset : offset + length] = pos
            dests[offset : offset + length] = page_indices[seq_id, pos // 2] * 2 + pos % 2
            expected.append(reference[seq_id][pos])
            offset += length
        info = PageBatchInfo(
            slot_ids=hax.named(jnp.array([0, 1], dtype=jnp.int32), "seq"),
            page_indices=hax.named(jnp.asarray(page_indices), ("seq", "page")),
            seq_lens=hax.named(jnp.array([start + length for start, length in phase], dtype=jnp.int32), "seq"),
            cu_q_lens=hax.named(jnp.array([0, lengths[0], n], dtype=jnp.int32), "seq"),
            num_seqs=jnp.array(2, dtype=jnp.int32),
            new_token_dests=hax.named(jnp.asarray(dests), token_axis),
            page_size=2,
        )
        token_ids = hax.named(jnp.asarray(ids), token_axis)
        position_ids = hax.named(jnp.asarray(positions), token_axis)
        logits, full_cache = decode(model, token_ids, cache, info, position_ids)
        np.testing.assert_allclose(np.asarray(logits.array)[:n], np.concatenate(expected), rtol=1e-4, atol=1e-4)
        if phase_index == 0:
            indices = np.array([n - 1, 0])
            selection = hax.named(jnp.asarray(indices), Axis("sample", 2))
            selected, cache = decode(model, token_ids, cache, info, position_ids, selection)
            assert selected.axes == (*selection.axes, model.Vocab)
            np.testing.assert_allclose(
                np.asarray(selected.array), np.asarray(logits.array)[indices], rtol=1e-4, atol=1e-4
            )
            for full, sparse in zip(jax.tree.leaves(full_cache), jax.tree.leaves(cache), strict=True):
                np.testing.assert_array_equal(full, sparse)
        else:
            cache = full_cache


@pytest.mark.parametrize("sliding_window", [2, 16])
@pytest.mark.parametrize("rope_fused", [False, True])
@jax.default_matmul_precision("highest")
def test_hero_paged_decode_matches_full_forward(sliding_window, rope_fused):
    """Mixed chunked prefill/decode preserves per-request positions and Hero attention and convolution history."""
    cfg = HeroConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=12,
        shared_expert_intermediate_dim=12,
        num_shared_experts=2,
        num_experts=4,
        num_experts_per_token=2,
        latent_dim=8,
        num_layers=3,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=1,
        head_dim=8,
        max_seq_len=16,
        sliding_window=sliding_window,
        global_every=2,
        qk_mult_long_scale=1.2,
        global_query_log_scaling=GlobalQueryLogScaling(2, 0.4, 2.0),
        sconv_kernel=3,
        rope_fused=rope_fused,
        initializer_std=0.2,
        attention_implementation="reference",
        inference_attention_implementation="reference",
    )
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = HeroLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(17))
        # Exercise the learned head gates, whose initializer is identically zero.
        blocks = model.transformer.stacked_blocks.stacked
        for site in [
            lambda b: b.attn.attn_gate,
            lambda b: b.attn.sconv_k.weight,
            lambda b: b.sconv_attn.weight,
            lambda b: b.sconv_mlp.weight,
        ]:
            array = site(blocks)
            blocks = eqx.tree_at(site, blocks, jax.random.normal(jax.random.key(42), array.shape) * 0.2)
        model = eqx.tree_at(lambda m: m.transformer.stacked_blocks.stacked, model, blocks)
        _assert_mixed_paged_decode_matches_forward(model)


# The first engine.generate call compiles the generation loop on each CI worker.
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "prompt_length, model_axis_size, expert_axis_size",
    [(3, 1, 1), (4, 1, 1), pytest.param(3, 2, 2, marks=pytest.mark.multidevice)],
)
@jax.default_matmul_precision("highest")
def test_hero_engine_clones_preserve_convolution_history(prompt_length, model_axis_size, expert_axis_size):
    """Cloning partial and complete KV pages preserves all emitted tokens and scores."""
    cfg = HeroConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=12,
        shared_expert_intermediate_dim=12,
        num_shared_experts=2,
        num_experts=4,
        num_experts_per_token=2,
        latent_dim=8,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=1,
        head_dim=8,
        max_seq_len=16,
        sliding_window=3,
        global_every=2,
        sconv_kernel=3,
        initializer_std=0.2,
        attention_implementation="reference",
        inference_attention_implementation="reference",
        moe_implementation="ring",
        capacity_factor=float(expert_axis_size),
    )
    device_count = model_axis_size * expert_axis_size
    with (
        use_test_devices(device_count),
        jax.set_mesh(compact_grug_mesh(model_axis_size=model_axis_size, expert_axis_size=expert_axis_size)),
    ):
        model = HeroLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(29))
        blocks = model.transformer.stacked_blocks.stacked
        for site in [lambda b: b.attn.sconv_k.weight, lambda b: b.sconv_attn.weight, lambda b: b.sconv_mlp.weight]:
            weight = site(blocks)
            blocks = eqx.tree_at(site, blocks, jax.random.normal(jax.random.key(31), weight.shape) * 0.2)
        model = eqx.tree_at(lambda m: m.transformer.stacked_blocks.stacked, model, blocks)
        config = InferenceEngineConfig(
            max_seq_len=16,
            max_pages=16,
            max_seqs=2,
            page_size=4,
            max_rounds=1,
            compute_dtype=jnp.float32,
            max_queued_tokens=16,
            max_seqs_in_prefill=2,
            max_prefill_size=8 * jax.device_count(),
            max_tokens_per_round=2 * jax.device_count(),
        )
        engine = InferenceEngine.from_model_with_config(model, None, config)
        params = dataclasses.replace(SeqDecodingParams.default(), max_num_tokens=jnp.asarray(prompt_length + 4))
        request = Request([2, 7, 9, 3][:prompt_length], 0, params, n_generations=2)
        cloned = engine.generate([request])
        assert cloned.logprobs is not None
        assert len(cloned.tokens) == 2
        for choice, logprobs in zip(cloned.tokens, cloned.logprobs, strict=True):
            assert len(choice) == 4
            full = jnp.asarray(request.prompt_tokens + choice)
            batch = jax.device_count() // model_axis_size
            ids = hax.named(jnp.broadcast_to(full, (batch, full.size)), ("batch", "position"))
            logits = model(ids).array[0, prompt_length - 1 : -1]
            expected = jnp.argmax(logits, axis=-1)
            np.testing.assert_array_equal(choice, expected)
            scores = jax.nn.log_softmax(logits, axis=-1)
            np.testing.assert_allclose(logprobs, scores[jnp.arange(len(choice)), expected], rtol=1e-4, atol=1e-4)
