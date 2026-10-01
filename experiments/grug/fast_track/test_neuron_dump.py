# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Neuron-health dump: per-unit ReLU² activity of the routed (conditioned on routing) and shared experts."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from levanter.grug.attention import AttentionMask

import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.train import _expert_grad_step, _neuron_counts_step, _unit_grad_stats

_RELU2 = dict(moe_ungated_relu2=True, shared_ungated_relu2=True, num_experts=8, num_experts_per_token=2)
_MP = jmp.get_policy("params=float32,compute=float32,output=float32")


def test_neuron_counts_match_a_direct_count():
    mesh, model = t._model(**_RELU2)
    cfg = model.config
    tokens = jax.random.randint(jax.random.PRNGKey(2), (2, t._SEQ), 0, t._VOCAB)
    segments = np.zeros(tokens.shape, np.int32)
    segments[1, -3:] = -1  # padding is not counted
    seg = jnp.asarray(segments)
    with jax.set_mesh(mesh):
        counts = jax.device_get(
            _neuron_counts_step(_MP)(model, jnp.zeros((cfg.num_layers, cfg.num_experts)), tokens, seg)
        )
        mask = AttentionMask.causal().with_segment_ids(seg)
        selected, routed_input, shared_pre = jax.device_get(
            eqx.filter_jit(lambda m: m.neuron_inputs(tokens, mask))(model)
        )
        w_ups = [np.asarray(layer.mlp.expert_mlp.w_up) for layer in model.layers()]
    valid = (segments >= 0).reshape(-1)
    k = cfg.num_experts_per_token
    for layer in (0, 3):
        sel, x, w = selected[layer], routed_input[layer], w_ups[layer]
        for e in range(cfg.num_experts):
            routed = np.any(sel == e, axis=-1) & valid
            pre = x[routed] @ w[e]
            np.testing.assert_array_equal(counts["routed_active"][layer, e], np.sum(pre > 0, axis=0))
            assert counts["routed_tokens"][layer, e] == routed.sum()
        assert counts["routed_tokens"][layer].sum() == valid.sum() * k
        assert counts["routed_hist"][layer].sum() == valid.sum() * k
        np.testing.assert_array_equal(counts["shared_active"][layer], np.sum(shared_pre[layer][valid] > 0, axis=0))
        assert counts["shared_hist"][layer].sum() == counts["shared_tokens"][layer] == valid.sum()
    # A ReLU² unit is neither always off nor always on for random inputs.
    assert 0 < counts["shared_active"].mean() < counts["shared_tokens"].mean()


def test_unit_gradient_stats_match_the_gradient():
    mesh, model = t._model(**_RELU2)
    cfg = model.config
    tokens = jax.random.randint(jax.random.PRNGKey(3), (2, t._SEQ), 0, t._VOCAB)
    seg = jnp.zeros(tokens.shape, jnp.int32)
    betas = jnp.zeros((cfg.num_layers, cfg.num_experts))
    with jax.set_mesh(mesh):
        grads = _expert_grad_step(_MP, None)(model, betas, tokens, seg)
        stats = jax.device_get(_unit_grad_stats(model, grads))
        grads = jax.device_get(grads)
    g_up = np.asarray(grads[2][0])  # layer 2, routed w_up: [E, in, I]
    np.testing.assert_allclose(stats["routed_grad_up"][2], np.linalg.norm(g_up, axis=-2), rtol=1e-5)
    assert stats["routed_grad_up"].shape == (cfg.num_layers, cfg.num_experts, cfg.intermediate_dim)
    assert stats["shared_grad_down"].shape == (cfg.num_layers, cfg.shared_expert_intermediate_dim)
    # Newton-Schulz output spreads the step: each expert's update has a near-unit Frobenius norm per direction.
    assert np.all(np.isfinite(stats["routed_ns_up"])) and stats["routed_ns_up"].max() <= 1.5
