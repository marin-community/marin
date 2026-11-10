# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The June model's custom-VJP RMS norm (``rms_norm_impl="custom_vjp"``) against the barrier path and a float64
reference."""

import collections
import contextlib
import dataclasses
import io

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.sharding import compact_grug_mesh

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, RMSNorm, Transformer, rms_norm

# The custom-VJP RMS norm (rms_norm_impl="custom_vjp") computes the barrier path's float32 forward expression, but XLA
# may fuse its reduction with the users the barrier keeps apart, which rounds differently: a few float32 ulps, at most
# one ulp after the cast to bfloat16. Both paths are compared with a float64 reference, to which they are equally close.
_RMS_SHAPES = [(4, 16), (2, 8, 128), (2, 3, 4, 128), (3, 7, 2560)]
# Largest error of each path against the float64 reference, in units of the output dtype's epsilon times the
# reference's largest magnitude.
_RMS_EPS_MULTIPLE = {jnp.float32: 64, jnp.bfloat16: 2}
_RMS_FORWARD_RTOL = {
    jnp.float32: 4 * float(jnp.finfo(jnp.float32).eps),
    jnp.bfloat16: float(jnp.finfo(jnp.bfloat16).eps),
}


def _rms_inputs(shape, dtype, seed=0):
    k_x, k_g, k_w = jax.random.split(jax.random.key(seed), 3)
    # Rows of different scales, so eps matters for some of them.
    scale = jnp.exp(jax.random.uniform(k_w, (*shape[:-1], 1), minval=-6.0, maxval=3.0))
    x = (jax.random.normal(k_x, shape) * scale).astype(dtype)
    g = jax.random.normal(k_g, shape, dtype=jnp.float32)
    return x, g


def _rms_reference(x, g, eps, weight=None):
    # float64 value and VJP of x * rsqrt(mean(x^2) + eps) * weight.
    x = np.asarray(x, np.float64)
    g = np.asarray(g, np.float64)
    w = np.ones(x.shape[-1]) if weight is None else np.asarray(weight, np.float64)
    r = 1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    gw = g * w
    dx = r * gw - x * r**3 * np.mean(gw * x, axis=-1, keepdims=True)
    dw = np.sum((g * x * r).reshape(-1, x.shape[-1]), axis=0)
    return x * r * w, dx, dw


def _max_scaled_error(actual, expected):
    actual = np.asarray(actual, np.float64)
    return float(np.max(np.abs(actual - expected)) / max(np.max(np.abs(expected)), 1e-30))


@pytest.mark.parametrize("eps", [1e-6, 1e-5])
@pytest.mark.parametrize("shape", _RMS_SHAPES)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_custom_vjp_rms_norm_matches_barrier_rms_norm(dtype, shape, eps):
    x, g = _rms_inputs(shape, dtype)

    @jax.jit
    def run(x, g):
        out = {}
        for impl in ("barrier", "custom_vjp"):
            y, vjp = jax.vjp(lambda v, i=impl: rms_norm(v, eps, impl=i), x)
            out[impl] = (y, vjp(g.astype(y.dtype))[0])
        return out

    out = run(x, g)
    (y_barrier, dx_barrier), (y_custom, dx_custom) = out["barrier"], out["custom_vjp"]
    assert y_custom.dtype == dx_custom.dtype == dtype
    np.testing.assert_allclose(y_custom, y_barrier, rtol=_RMS_FORWARD_RTOL[dtype], atol=0)
    tolerance = _RMS_EPS_MULTIPLE[dtype] * float(jnp.finfo(dtype).eps)
    y_ref, dx_ref, _ = _rms_reference(x, g, eps)
    for name, dx in (("barrier", dx_barrier), ("custom_vjp", dx_custom)):
        assert _max_scaled_error(dx, dx_ref) <= tolerance, name
    assert _max_scaled_error(y_custom, y_ref) <= tolerance
    diff = np.max(np.abs(np.asarray(dx_custom, np.float64) - np.asarray(dx_barrier, np.float64)))
    assert diff <= tolerance * np.max(np.abs(dx_ref))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_custom_vjp_rms_norm_layer_matches_reference(dtype):
    # The weight multiplies the normalized activations outside the custom VJP, so its gradient comes from AD.
    eps = 1e-5
    x, g = _rms_inputs((2, 64, 256), dtype, seed=3)
    weight = jax.random.normal(jax.random.key(4), (256,)) * 0.3 + 1
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):

        @jax.jit
        def run(w, v, g):
            y, vjp = jax.vjp(lambda w, v: RMSNorm(w, eps=eps, impl="custom_vjp")(v), w, v)
            dw, dx = vjp(g.astype(y.dtype))
            return y, dx, dw

        y, dx, dw = run(weight, x, g)
    y_ref, dx_ref, dw_ref = _rms_reference(x, np.asarray(g.astype(dtype), np.float64), eps, weight)
    tolerance = _RMS_EPS_MULTIPLE[dtype] * float(jnp.finfo(dtype).eps)
    assert _max_scaled_error(y, y_ref) <= tolerance
    assert _max_scaled_error(dx, dx_ref) <= tolerance
    assert _max_scaled_error(dw, dw_ref) <= 64 * float(jnp.finfo(jnp.float32).eps)


def test_bf16_custom_vjp_rms_norm_scoring_is_within_one_ulp_of_differentiated_forward():
    # Unlike the barrier path (test_bf16_rms_norm_scoring_matches_differentiated_forward in the root June model tests),
    # the scoring forward does not output the inverse RMS, so XLA may fuse the reduction with its users there and not in
    # the differentiated forward.
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        inputs = jax.random.normal(jax.random.PRNGKey(10), (2, 2048, 2560)).astype(jnp.bfloat16)
        weight = (jax.random.normal(jax.random.PRNGKey(11), (2560,)) * 0.3 + 1).astype(jnp.bfloat16)
        norm = RMSNorm(weight, eps=1e-6, impl="custom_vjp")

        def objective(module, values):
            output = module(values)
            return jnp.mean(jnp.sin(output.astype(jnp.float32))), output

        scored = jax.jit(lambda module, values: module(values))(norm, inputs)
        (_, trained), gradients = jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True))(norm, inputs)
        jax.block_until_ready(gradients)
        np.testing.assert_allclose(scored, trained, rtol=float(jnp.finfo(jnp.bfloat16).eps), atol=0)


@pytest.mark.parametrize("stacked", [True, False])
@pytest.mark.parametrize("remat_mode", ["recompute_all", "save_moe"])
def test_custom_vjp_rms_norm_saves_fewer_residuals(remat_mode, stacked):
    # The custom VJP's residuals are unnamed, so the block checkpoints recompute them like the barrier path's: the
    # blocks save the same residuals at any depth. Outside the checkpoints, the embedding and final norms each save one
    # float32 activation and one float32 statistic fewer than AD of the barrier path.
    small = GrugModelConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_experts=4,
        num_experts_per_token=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        capacity_factor=8.0,
        attention_implementation="reference",
        disable_pko=True,
    )
    tokens = jax.random.randint(jax.random.key(1), (2, 8), 0, small.vocab_size)

    def saved(num_layers, impl):
        cfg = dataclasses.replace(
            small, num_layers=num_layers, remat_mode=remat_mode, use_array_stacked_blocks=stacked, rms_norm_impl=impl
        )
        with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
            params, static = eqx.partition(Transformer.init(cfg, key=jax.random.key(0)), eqx.is_array)

            def loss(p):
                hidden, _ = eqx.combine(p, static)(tokens)
                return jnp.mean(jnp.square(hidden))

            report = io.StringIO()
            with contextlib.redirect_stdout(report):
                jax.ad_checkpoint.print_saved_residuals(loss, params)
        # One line per residual: its type, then where it comes from.
        return collections.Counter(line.split()[0] for line in report.getvalue().splitlines())

    for num_layers in (2, 4):
        barrier, custom = saved(num_layers, "barrier"), saved(num_layers, "custom_vjp")
        assert custom <= barrier, num_layers
        # Rows: a batch of 2 sharded over the batch axes.
        activation, statistic = "f32[2@(replica_dcn,data,expert),8,16]", "f32[2@(replica_dcn,data,expert),8,1]"
        assert barrier - custom == collections.Counter({activation: 2, statistic: 2}), num_layers
