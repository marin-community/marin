# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh
from levanter.grug.attention import AttentionMask

from experiments.grug.fast_track.model import Transformer
from experiments.grug.noise_belay.candidate11 import load_model_config
from experiments.grug.noise_belay.candidate11_boost import (
    reconstruct_latent_expansions,
    reconstruct_mlp_convolutions,
    reconstruct_routed_experts,
    reconstruct_router_matrices,
)
from experiments.grug.noise_belay.candidate11_forward import capture_forward


def make_small_model():
    config = dataclasses.replace(
        load_model_config(),
        hidden_dim=64,
        intermediate_dim=48,
        shared_expert_intermediate_dim=96,
        num_experts=16,
        num_experts_per_token=2,
        head_dim=16,
        num_heads=4,
        latent_dim=32,
        embed2_rows=256,
        vocab_size=128,
        max_seq_len=128,
        mla_kv_latent_dim=32,
        rel_dim=4,
        rel_extent=8,
    )
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    with jax.set_mesh(mesh):
        model = eqx.filter_jit(Transformer.init)(config, key=jax.random.PRNGKey(0))
        leaves, tree = jax.tree_util.tree_flatten_with_path(model)
        keys = jax.random.split(jax.random.PRNGKey(3), len(leaves))
        leaves = [
            (leaf + 0.1 * jax.random.normal(key, leaf.shape) if "attn_res_query" in jax.tree_util.keystr(path) else leaf)
            for (path, leaf), key in zip(leaves, keys, strict=True)
        ]
        model = jax.tree_util.tree_unflatten(tree, leaves)
    return model, mesh


def assert_routed_reconstruction(
    model_and_mesh,
):
    model, mesh = model_and_mesh
    with jax.set_mesh(mesh):
        tokens = jax.random.randint(jax.random.PRNGKey(31), (2, 128), 0, 128)
        segments = jnp.broadcast_to(jnp.repeat(jnp.arange(2), 64)[None], tokens.shape)
        mask = AttentionMask(is_causal=True, segment_ids=(segments, segments))
        _, capture = eqx.filter_jit(capture_forward)(model, tokens, mask)
        keys = jax.random.split(jax.random.PRNGKey(5), 12)
        hub_credits = tuple(jax.random.normal(k, (2, 128, 64)) * 0.1 for k in keys)
        actual = eqx.filter_jit(reconstruct_routed_experts)(model, capture, hub_credits, relative_step=0.0001)
        routers = eqx.filter_jit(reconstruct_router_matrices)(model, capture, hub_credits, sigma=0.001)
        expansions = eqx.filter_jit(reconstruct_latent_expansions)(model, capture, hub_credits)
        convolutions = eqx.filter_jit(reconstruct_mlp_convolutions)(model, capture, hub_credits)
        for index, layer in enumerate(model.layers()):

            def objective(layer, index=index):
                output, _ = layer.mlp_branch(capture.stages[2 * index + 1].input, mask)
                return jnp.sum(output * hub_credits[2 * index + 1])

            expected = eqx.filter_jit(eqx.filter_grad(objective))(layer)
            np.testing.assert_allclose(actual[index].up, expected.mlp.expert_mlp.w_up, atol=2e-4, rtol=2e-4)
            np.testing.assert_allclose(actual[index].down, expected.mlp.expert_mlp.w_down, atol=2e-4, rtol=2e-4)
            np.testing.assert_allclose(routers[index], expected.mlp.router, atol=2e-5, rtol=2e-4)
            np.testing.assert_allclose(convolutions[index], expected.sconv_mlp.weight, atol=2e-5, rtol=2e-4)
            if expected.mlp.w_latent_up is None:
                assert expansions[index] is None
            else:
                np.testing.assert_allclose(expansions[index], expected.mlp.w_latent_up, atol=2e-5, rtol=2e-4)
