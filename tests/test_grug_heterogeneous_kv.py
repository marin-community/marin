# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, set_mesh
from levanter.grug.attention import AttentionMask

from experiments.grug.moe import model as grug_model


def _one_device_mesh() -> Mesh:
    return Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def _block_at(stacked: grug_model.Block, index: int) -> grug_model.Block:
    return jax.tree.map(lambda leaf: leaf[index], stacked)


def _unrolled_heterogeneous_forward(
    model: grug_model.Transformer,
    token_ids: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    cfg = model.config
    assert model.stacked_blocks is not None

    hidden = grug_model._embedding_gather(model.token_embed, token_ids)
    hidden = model.embed_gated_norm(model.embed_norm(hidden))
    global_mask = AttentionMask.causal()
    stats: list[dict[str, jax.Array]] = []

    for layer_index in range(cfg.num_layers):
        is_global = layer_index % cfg.global_layer_period == cfg.global_layer_period - 1
        layer = _block_at(model.stacked_blocks.stacked, layer_index)
        disable_rope = is_global and cfg.disable_long_rope
        # The production FA4 path reads precomputed bounds attached to this full-causal
        # mask. The CPU reference backend ignores those FA4-only bounds, so use the same
        # base mask here; the slicing behavior is covered independently below.
        hidden, layer_stats = layer(hidden, global_mask, False, disable_rope, is_global)
        stats.append(layer_stats)

    hidden = model.final_gated_norm(model.final_norm(hidden))
    router_metrics = {f"{key}_per_layer": jnp.stack([layer_stats[key] for layer_stats in stats]) for key in stats[0]}
    return hidden, router_metrics


def test_heterogeneous_kv_scan_matches_unrolled_layers():
    cfg = grug_model.GrugModelConfig(
        vocab_size=64,
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=32,
        num_layers=6,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=1,
        global_layer_period=3,
        head_dim=8,
        max_seq_len=6,
        sliding_window=3,
        num_experts=4,
        num_experts_per_token=2,
        use_array_stacked_blocks=True,
        attention_implementation="reference",
    )
    token_ids = jnp.arange(12, dtype=jnp.int32).reshape(2, 6)

    with set_mesh(_one_device_mesh()):
        model = grug_model.Transformer.init(cfg, key=jax.random.PRNGKey(0))
        scanned = eqx.filter_jit(lambda current_model, tokens: current_model(tokens, AttentionMask.causal()))
        unrolled = eqx.filter_jit(_unrolled_heterogeneous_forward)
        scanned_hidden, scanned_metrics = scanned(model, token_ids)
        unrolled_hidden, unrolled_metrics = unrolled(model, token_ids)

    np.testing.assert_allclose(scanned_hidden, unrolled_hidden, rtol=1e-5, atol=1e-5)
    assert scanned_metrics.keys() == unrolled_metrics.keys()
    for key in scanned_metrics:
        np.testing.assert_allclose(scanned_metrics[key], unrolled_metrics[key], rtol=1e-5, atol=1e-5)


def test_heterogeneous_kv_global_attention_matches_narrow_reference_and_ignores_padding():
    cfg = grug_model.GrugModelConfig(
        vocab_size=64,
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=32,
        num_layers=3,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=1,
        global_layer_period=3,
        head_dim=8,
        max_seq_len=6,
        sliding_window=3,
        num_experts=4,
        num_experts_per_token=2,
        use_array_stacked_blocks=True,
        attention_implementation="reference",
    )

    with set_mesh(_one_device_mesh()):
        attention = grug_model.CausalSelfAttention.init(cfg, key=jax.random.PRNGKey(0))
        x = jax.random.normal(jax.random.PRNGKey(1), (2, 6, cfg.hidden_dim))
        mask = AttentionMask.causal()
        global_output = attention(x, mask, is_global=True)
        local_output = attention(x, mask, is_global=False)

        assert cfg.global_kv_heads is not None
        global_width = cfg.global_kv_heads * cfg.inferred_head_dim
        uniform_global_cfg = dataclasses.replace(
            cfg,
            num_kv_heads=cfg.global_kv_heads,
            local_kv_heads=None,
            global_kv_heads=None,
        )
        uniform_global_attention = grug_model.CausalSelfAttention(
            w_q=attention.w_q,
            w_k=attention.w_k[:, :global_width],
            w_v=attention.w_v[:, :global_width],
            w_o=attention.w_o,
            attn_gate=attention.attn_gate,
            cfg=uniform_global_cfg,
        )
        uniform_global_output = uniform_global_attention(x, mask)
        np.testing.assert_allclose(global_output, uniform_global_output, rtol=1e-5, atol=1e-5)

        padded_delta = jnp.broadcast_to(
            jnp.concatenate(
                [
                    jnp.zeros((global_width,), dtype=attention.w_k.dtype),
                    jnp.ones((attention.w_k.shape[1] - global_width,), dtype=attention.w_k.dtype),
                ]
            ),
            attention.w_k.shape,
        )
        perturbed = eqx.tree_at(
            lambda current: (current.w_k, current.w_v),
            attention,
            (
                attention.w_k + padded_delta,
                attention.w_v + padded_delta,
            ),
        )
        perturbed_global_output = perturbed(x, mask, is_global=True)
        perturbed_local_output = perturbed(x, mask, is_global=False)

    np.testing.assert_allclose(global_output, perturbed_global_output, rtol=1e-5, atol=1e-5)
    assert not np.allclose(local_output, perturbed_local_output, rtol=1e-5, atol=1e-5)
