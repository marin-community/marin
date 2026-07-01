# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from haliax.nn.ragged_dot import ragged_dot
from haliax.quantization import Fp8RaggedDotOp, apply_updates, partition_for_grad_overwrite

gpu_only = pytest.mark.skipif(jax.default_backend() != "gpu", reason="fp8 wgmma only lowers on GPU")


def _rel_fro(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)


def _nonuniform(T, K, E, N, seed=0):
    # N is a multiple of 128 so the bf16 output store's TMA descriptor is
    # swizzle-aligned in the Mosaic ragged wgmma kernel.
    rng = np.random.default_rng(seed)
    lhs = jnp.asarray(rng.standard_normal((T, K)) * 0.1, jnp.bfloat16)
    rhs = jnp.asarray(rng.standard_normal((E, K, N)) * 0.1, jnp.bfloat16)
    # genuine non-uniform groups summing to T
    parts = rng.multinomial(T, np.ones(E) / E)
    return lhs, rhs, jnp.asarray(parts, jnp.int32)


@gpu_only
def test_fp8_forward_rel_fro():
    lhs, rhs, gs = _nonuniform(64, 128, 4, 128)
    op = Fp8RaggedDotOp.init()
    out = ragged_dot(lhs, rhs, gs, op=op)
    ref = ragged_dot(lhs, rhs, gs, op=None)
    assert _rel_fro(out, ref) < 5e-2


@gpu_only
def test_fp8_grad_lhs_rel_fro():
    # The input gradient (grad_lhs) now runs on the FP8 tensor cores: the output
    # gradient is quantized to e4m3 (uniform, stock-jaxlib-safe) and contracted
    # against the pre-cast weight, so grad_lhs is approximate vs the bf16 grad.
    # The weight gradient (grad_rhs) stays bf16, so it still matches to roundoff.
    # T is a multiple of 64 so the ragged wgmma accumulator tile is valid; the
    # groups are still genuinely non-uniform (including a small 5-token expert).
    lhs, rhs, _ = _nonuniform(64, 128, 4, 128, seed=3)
    gs = jnp.asarray([13, 5, 17, 29], jnp.int32)  # genuine non-uniform, sums to T=64
    op = Fp8RaggedDotOp.init()  # rev_dtype defaults to e4m3 (uniform dgrad)

    def loss(l, r, o):
        return ragged_dot(l, r, gs, op=o).astype(jnp.float32).sum()

    g_lhs_fp8, g_rhs_fp8 = jax.grad(lambda l, r: loss(l, r, op), argnums=(0, 1))(lhs, rhs)
    g_lhs_ref, g_rhs_ref = jax.grad(lambda l, r: loss(l, r, None), argnums=(0, 1))(lhs, rhs)

    assert _rel_fro(g_lhs_fp8, g_lhs_ref) < 6e-2, "FP8 dgrad grad_lhs out of tolerance"
    assert _rel_fro(g_rhs_fp8, g_rhs_ref) < 1e-3, "bf16 wgrad grad_rhs should match bf16 to roundoff"


@gpu_only
def test_output_grad_scale_updates_across_steps():
    """output_grad_scale is now live: it updates from the gradient magnitudes.

    The FP8 dgrad quantizes the output gradient with delayed scaling, so
    _qrd_bwd returns the rolled grad_scale/grad_amax_history as the
    OverwriteWithGradient cotangents. output_grad_scale therefore moves away from
    1.0 once the amax history accumulates a non-zero gradient magnitude — exactly
    like input_scale/kernel_scale. It must still never be zeroed.
    """
    lhs, rhs, gs = _nonuniform(64, 128, 4, 128)
    op = Fp8RaggedDotOp.init()

    def loss(op_, l, r):
        return ragged_dot(l, r, gs, op=op_).astype(jnp.float32).sum()

    def step(op_):
        grads = eqx.filter_grad(loss)(op_, lhs, rhs)
        overwrites, non_overwrites = partition_for_grad_overwrite(grads)
        updates = jax.tree_util.tree_map(lambda g: jnp.zeros_like(g), non_overwrites)
        return apply_updates(op_, updates, overwrites)

    op1 = step(op)
    op2 = step(op1)

    # output_grad_scale must not be zeroed (the delayed-scaling state stays valid).
    assert not np.allclose(np.asarray(op2.output_grad_scale), 0.0), "output_grad_scale was zeroed (state corruption)"
    # input_scale, kernel_scale, and output_grad_scale must all have moved away from
    # 1.0 — the input/kernel via in_q, the output gradient via the FP8 dgrad's delayed
    # scaling. Require all three so a regression in any delayed-scaling path is caught.
    assert (
        not np.allclose(np.asarray(op2.input_scale), 1.0, atol=1e-6)
        and not np.allclose(np.asarray(op2.kernel_scale), 1.0, atol=1e-6)
        and not np.allclose(np.asarray(op2.output_grad_scale), 1.0, atol=1e-6)
    ), "input_scale, kernel_scale, and output_grad_scale should all update from 1.0 via delayed scaling"


def _time_forward(fn, warmup: int = 3, iters: int = 10) -> float:
    """Steady-state wall-clock timer (seconds) for a JIT-compiled JAX forward.

    Warms up ``warmup`` times to compile and prime caches, then returns the
    minimum over ``iters`` timed calls (``jax.block_until_ready`` ensures device
    completion before the clock stops).
    """
    for _ in range(warmup):
        jax.block_until_ready(fn())
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - t0)
    return float(np.min(times))


@gpu_only
def test_fp8_forward_faster_than_bf16():
    # w13-like shape: T=4096 tokens, K=2560, E=8 experts, N=2560 (multiple of 128).
    lhs, rhs, gs = _nonuniform(4096, 2560, 8, 2560)
    op = Fp8RaggedDotOp.init()
    f8 = _time_forward(lambda: ragged_dot(lhs, rhs, gs, op=op))
    bf = _time_forward(lambda: ragged_dot(lhs, rhs, gs, op=None))
    assert (
        bf / f8 > 1.5
    ), f"expected FP8 forward >1.5x faster than bf16; got {bf / f8:.3f}x (bf16={bf * 1e3:.2f}ms, fp8={f8 * 1e3:.2f}ms)"
