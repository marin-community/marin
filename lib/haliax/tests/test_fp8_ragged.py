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
    # Both gradients now run on the FP8 tensor cores: the output gradient is
    # quantized to e4m3 (uniform, stock-jaxlib-safe) and contracted against the
    # pre-cast weight (grad_lhs) and the pre-cast transposed activation (grad_rhs),
    # so both are approximate vs the bf16 grad and held to the 6e-2 FP8 tolerance.
    # T is a multiple of 128 (the wgrad's token-dim TMA tile); the groups are still
    # genuinely non-uniform (including a small 5-token expert).
    lhs, rhs, _ = _nonuniform(128, 128, 4, 128, seed=3)
    gs = jnp.asarray([13, 5, 47, 63], jnp.int32)  # genuine non-uniform, sums to T=128
    op = Fp8RaggedDotOp.init()  # rev_dtype defaults to e4m3 (uniform backward)

    def loss(l, r, o):
        return ragged_dot(l, r, gs, op=o).astype(jnp.float32).sum()

    g_lhs_fp8, g_rhs_fp8 = jax.grad(lambda l, r: loss(l, r, op), argnums=(0, 1))(lhs, rhs)
    g_lhs_ref, g_rhs_ref = jax.grad(lambda l, r: loss(l, r, None), argnums=(0, 1))(lhs, rhs)

    assert _rel_fro(g_lhs_fp8, g_lhs_ref) < 6e-2, "FP8 dgrad grad_lhs out of tolerance"
    assert _rel_fro(g_rhs_fp8, g_rhs_ref) < 6e-2, "FP8 wgrad grad_rhs out of tolerance"


@gpu_only
def test_fp8_grad_rhs_rel_fro_incl_boundaries():
    # The weight gradient (grad_rhs) contracts the ragged token dim in block_k=128
    # tiles via mgpu_dwgrad. This exercises the f16-upcast group-boundary mask:
    # the token groups are chosen so boundaries fall mid-tile and one group
    # straddles the 128-token block boundary (start 100, end 180). T is a multiple
    # of 128 (the wgrad's contracting-dim TMA tile), N a multiple of 128 (the
    # ragged-kernel alignment), K a multiple of 128 (the dgrad alignment).
    lhs, rhs, _ = _nonuniform(256, 128, 4, 128, seed=4)
    gs = jnp.asarray([60, 40, 80, 76], jnp.int32)  # boundaries 60, 100, 180 (mid-block_k)
    assert int(gs.sum()) == 256
    op = Fp8RaggedDotOp.init()

    def loss(w, o):
        return ragged_dot(lhs, w, gs, op=o).astype(jnp.float32).sum()

    g_rhs_fp8 = jax.grad(lambda w: loss(w, op))(rhs)
    g_rhs_ref = jax.grad(lambda w: loss(w, None))(rhs)

    assert _rel_fro(g_rhs_fp8, g_rhs_ref) < 6e-2, "FP8 wgrad grad_rhs out of tolerance at mid-tile boundaries"


@gpu_only
def test_output_grad_scale_updates_across_steps():
    """output_grad_scale is now live: it updates from the gradient magnitudes.

    The FP8 dgrad quantizes the output gradient with delayed scaling, so
    _qrd_bwd returns the rolled grad_scale/grad_amax_history as the
    OverwriteWithGradient cotangents. output_grad_scale therefore moves away from
    1.0 once the amax history accumulates a non-zero gradient magnitude — exactly
    like input_scale/kernel_scale. It must still never be zeroed.
    """
    lhs, rhs, gs = _nonuniform(128, 128, 4, 128)
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


@gpu_only
def test_fp8_fwd_bwd_throughput_speedup_w13_1024():
    """Throughput fwd+bwd speedup at the operating point: w13/E=64/1024 tok, ≥1.2×.

    Operating point: w13 shape (K=2560, N=2560), E_local=64, 1024 tokens/expert
    (T=65536 total), genuine non-uniform group_sizes.

    Uses the throughput timer (enqueue N calls, block_until_ready once) which
    represents a pipelined training step.  Measured 1.28× on H100; the 1.2× floor
    provides margin.  Note: per-call latency (block after each call) measures ~1.14-1.23×
    at the same point -- both are valid, the gate uses throughput to match how the
    target was defined.
    """
    lhs, rhs, gs = _nonuniform(65536, 2560, 64, 2560, seed=7)
    op = Fp8RaggedDotOp.init(amax_history_length=16)

    jfp8 = jax.jit(lambda a, b: jax.value_and_grad(lambda a_, b_: ragged_dot(a_, b_, gs, op=op).sum(), (0, 1))(a, b))
    jbf16 = jax.jit(lambda a, b: jax.value_and_grad(lambda a_, b_: ragged_dot(a_, b_, gs).sum(), (0, 1))(a, b))

    # Compile and warm up both before timing to avoid cross-contamination.
    for _ in range(5):
        jax.block_until_ready(jbf16(lhs, rhs))
    for _ in range(5):
        jax.block_until_ready(jfp8(lhs, rhs))

    def _tput(jfn, n=20):
        t0 = time.perf_counter()
        for _ in range(n):
            out = jfn(lhs, rhs)
        jax.block_until_ready(out)
        return (time.perf_counter() - t0) / n

    t_bf16 = _tput(jbf16)
    t_fp8 = _tput(jfp8)
    speedup = t_bf16 / t_fp8
    assert speedup >= 1.2, (
        f"FP8 fwd+bwd throughput speedup {speedup:.3f}x < 1.2x floor at operating point "
        f"(bf16={t_bf16 * 1e3:.2f}ms, fp8={t_fp8 * 1e3:.2f}ms)"
    )
