# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track: fusing an MoE block's input projections (router, latent down, shared experts) into one
GEMM gives the per-projection outputs, router stats and gradients."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh
from levanter.utils.activation import ActivationFunctionEnum

from experiments.grug.fast_track.model import DenseMLP, GrugModelConfig, MoEMLP, moe_and_shared_fused


def test_fused_mlp_input_projections_match_per_projection_path():
    cfg = GrugModelConfig(
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=8,
        num_experts=4,
        num_experts_per_token=2,
        latent_dim=16,
        capacity_factor=8.0,
        pooled_transport_capacity_factor=8.0,
    )
    mesh = Mesh(
        np.array(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    x = jax.random.normal(jax.random.key(1), (2, 16, cfg.hidden_dim), jnp.float32)
    cotangent = jax.random.normal(jax.random.key(2), x.shape, jnp.float32)

    def per_projection(mlp: MoEMLP, shared: tuple[DenseMLP, ...], x: jax.Array):
        out, stats = mlp(x)
        for expert in shared:
            out = out + expert(x, activation=ActivationFunctionEnum.silu)
        return out, stats

    def loss_and_grads(fn, mlp, shared):
        def loss(modules, x):
            out, stats = fn(*modules, x)
            return jnp.sum(out * cotangent), stats

        return eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))((mlp, shared), x)

    with jax.set_mesh(mesh):
        keys = jax.random.split(jax.random.key(0), 3)
        mlp = MoEMLP.init(cfg, key=keys[0])
        shared = tuple(DenseMLP.init(cfg.hidden_dim, 8, 0.1, key=k) for k in keys[1:])
        (loss, stats), grads = loss_and_grads(moe_and_shared_fused, mlp, shared)
        (ref_loss, ref_stats), ref_grads = loss_and_grads(per_projection, mlp, shared)

    np.testing.assert_allclose(loss, ref_loss, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(stats["qb_beta"]), np.asarray(ref_stats["qb_beta"]), rtol=1e-6, atol=1e-6)
    flat = jax.tree_util.tree_flatten_with_path(eqx.filter(grads, eqx.is_array))[0]
    ref_flat = jax.tree_util.tree_leaves(eqx.filter(ref_grads, eqx.is_array))
    assert len(flat) == len(ref_flat)
    for (path, grad), ref in zip(flat, ref_flat, strict=True):
        scale = float(jnp.max(jnp.abs(ref))) + 1e-12
        assert float(jnp.max(jnp.abs(grad - ref))) / scale < 1e-5, jax.tree_util.keystr(path)
