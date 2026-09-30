# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Parity of the fused RMSNorm + GatedNorm kernels against the vanilla-JAX reference.

On CPU the Pallas interpreter executes the kernel bodies (tiling, the fused down projection,
the gate) with plain XLA ops, so these tests cover the algorithm and the hand-written backward.
In f32 the only difference from the reference is where the row scale enters the down
projection, so values and gradients agree to f32 rounding. In bf16 the kernel rounds
``x * w`` rather than ``x * rstd * w`` before that projection, so it is held to the reference's
own error against an f32 evaluation instead of to the reference itself.
`test_pallas_matches_reference_on_gpu` covers lowering when a GPU is present.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.kernels.pallas.gated_rms_norm import GatedRmsNormBlockSizes, gated_rms_norm, gated_rms_norm_reference
from levanter.kernels.pallas.gated_rms_norm.pallas_gpu import interpret_mode
from levanter.testing.cpu_devices import run_on_cpu_devices

pytestmark = pytest.mark.skipif(jax.default_backend() == "tpu", reason="Triton kernel")

EPS = 1e-5
BLOCKS = GatedRmsNormBlockSizes(t_block_size=16, stats_d_block_size=32, out_d_block_size=64)


def _inputs(shape, rank, dtype, seed):
    k = jax.random.split(jax.random.key(seed), 5)
    d = shape[-1]
    x = jax.random.normal(k[0], shape, jnp.float32).astype(dtype)
    norm_weight = 1.0 + 0.2 * jax.random.normal(k[1], (d,), jnp.float32)
    w_down = (jax.random.normal(k[2], (d, rank)) / np.sqrt(d)).astype(dtype)
    w_up = (2.0 * jax.random.normal(k[3], (rank, d)) / np.sqrt(rank)).astype(dtype)
    cotangent = jax.random.normal(k[4], shape, jnp.float32).astype(dtype)
    return (x, norm_weight, w_down, w_up), cotangent


def _value_and_grads(fn, args, cotangent):
    out, vjp = jax.vjp(fn, *args)
    return [np.asarray(a, np.float64) for a in (out, *vjp(cotangent))]


def _reference(x, norm_weight, w_down, w_up):
    return gated_rms_norm_reference(x, norm_weight, w_down, w_up, eps=EPS)


def _kernel(x, norm_weight, w_down, w_up):
    return gated_rms_norm(x, norm_weight, w_down, w_up, eps=EPS, implementation="pallas_gpu", block_sizes=BLOCKS)


# (activation shape, gate rank); 2 * 37 tokens is not a multiple of the 16-token tile.
SHAPES = [((2, 32, 64), 16), ((2, 37, 128), 32), ((64, 192), 16)]


@pytest.mark.parametrize("shape,rank", SHAPES, ids=lambda v: str(v))
def test_float32_values_and_gradients_match_reference(shape, rank):
    args, cotangent = _inputs(shape, rank, jnp.float32, seed=sum(shape))
    with interpret_mode():
        got = _value_and_grads(_kernel, args, cotangent)
    want = _value_and_grads(_reference, args, cotangent)
    for name, g, w in zip(("out", "dx", "dnorm_weight", "dw_down", "dw_up"), got, want, strict=True):
        np.testing.assert_allclose(g, w, rtol=2e-5, atol=2e-5 * np.max(np.abs(w)), err_msg=name)


@pytest.mark.parametrize("shape,rank", SHAPES[:2], ids=lambda v: str(v))
def test_bfloat16_error_is_within_the_references_own_rounding(shape, rank):
    args, cotangent = _inputs(shape, rank, jnp.bfloat16, seed=7 + sum(shape))
    exact = _value_and_grads(_reference, [a.astype(jnp.float32) for a in args], cotangent.astype(jnp.float32))
    with interpret_mode():
        got = _value_and_grads(_kernel, args, cotangent)
    want = _value_and_grads(_reference, args, cotangent)
    for name, g, w, e in zip(("out", "dx", "dnorm_weight", "dw_down", "dw_up"), got, want, exact, strict=True):
        kernel_error = np.sqrt(np.mean((g - e) ** 2))
        reference_error = np.sqrt(np.mean((w - e) ** 2))
        assert kernel_error <= 1.5 * reference_error, (name, kernel_error, reference_error)


def test_explicit_pallas_request_fails_fast_off_gpu():
    if jax.default_backend() == "gpu":
        pytest.skip("asserts the non-GPU behavior")
    args, _ = _inputs((2, 16, 64), 16, jnp.float32, seed=1)
    with pytest.raises(RuntimeError, match="unusable"):
        _kernel(*args)


def test_kernel_runs_inside_a_shard_map_without_collectives():
    run_on_cpu_devices(
        """
        import jax, jax.numpy as jnp, numpy as np
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
        from levanter.kernels.pallas.gated_rms_norm import GatedRmsNormBlockSizes, gated_rms_norm, gated_rms_norm_reference
        from levanter.kernels.pallas.gated_rms_norm.pallas_gpu import interpret_mode

        mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("data", "context"), axis_types=(AxisType.Explicit,) * 2)
        blocks = GatedRmsNormBlockSizes(t_block_size=16, stats_d_block_size=32, out_d_block_size=64)
        k = jax.random.split(jax.random.key(0), 5)
        x = jax.random.normal(k[0], (4, 32, 64))
        w = 1.0 + 0.2 * jax.random.normal(k[1], (64,))
        wd = jax.random.normal(k[2], (64, 16)) / 8.0
        wu = jax.random.normal(k[3], (16, 64)) / 2.0
        cot = jax.random.normal(k[4], (4, 32, 64))

        def objective(fn, x, w, wd, wu):
            return jnp.sum(fn(x, w, wd, wu) * cot)

        kernel = lambda x, w, wd, wu: gated_rms_norm(x, w, wd, wu, eps=1e-5, implementation="pallas_gpu", block_sizes=blocks)
        reference = lambda x, w, wd, wu: gated_rms_norm_reference(x, w, wd, wu, eps=1e-5)
        with jax.set_mesh(mesh), interpret_mode():
            xs = jax.device_put(x, NamedSharding(mesh, P("data", "context", None)))
            text = str(jax.make_jaxpr(kernel)(xs, w, wd, wu))
            assert "shard_map" in text
            for banned in ("all_gather", "psum", "all_to_all", "reduce_scatter"):
                assert banned not in text, banned
            got = jax.jit(jax.value_and_grad(lambda *a: objective(kernel, *a), argnums=(0, 1, 2, 3)))(xs, w, wd, wu)
        want = jax.jit(jax.value_and_grad(lambda *a: objective(reference, *a), argnums=(0, 1, 2, 3)))(x, w, wd, wu)
        np.testing.assert_allclose(got[0], want[0], rtol=2e-5)
        for g, r in zip(got[1], want[1]):
            np.testing.assert_allclose(np.asarray(g), np.asarray(r), rtol=2e-5, atol=2e-5 * float(np.max(np.abs(r))))
        """,
        device_count=4,
    )


def test_pallas_matches_reference_on_gpu():
    """The compiled kernels at a tile-aligned shape; the interpreter cannot show that they lower."""
    if jax.default_backend() != "gpu":
        pytest.skip("requires the JAX GPU backend")
    args, cotangent = _inputs((4, 256, 1024), 128, jnp.bfloat16, seed=3)
    exact = _value_and_grads(_reference, [a.astype(jnp.float32) for a in args], cotangent.astype(jnp.float32))

    def kernel(x, norm_weight, w_down, w_up):
        return gated_rms_norm(x, norm_weight, w_down, w_up, eps=EPS, implementation="pallas_gpu")

    got = _value_and_grads(jax.jit(kernel), args, cotangent)
    want = _value_and_grads(jax.jit(_reference), args, cotangent)
    for name, g, w, e in zip(("out", "dx", "dnorm_weight", "dw_down", "dw_up"), got, want, exact, strict=True):
        assert np.all(np.isfinite(g)), name
        assert np.sqrt(np.mean((g - e) ** 2)) <= 1.5 * np.sqrt(np.mean((w - e) ** 2)), name
