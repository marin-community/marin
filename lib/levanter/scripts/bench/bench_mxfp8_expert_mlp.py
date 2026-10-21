# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Memory and time of the MXFP8 expert MLP against QuACK's bf16 expert MLP at one hero expert chunk.

For both expert MLPs this compiles the calls the ragged all-to-all backend makes -- ``apply``,
``forward``, ``backward``, and ``forward`` then ``backward`` in one program -- and the
rematerialized forward and backward the hero's ``recompute_all`` runs, and reports each program's
``memory_analysis()`` in GiB, its median time, and its count of custom calls (kernel launches).
``peak`` is arguments plus outputs minus aliased bytes plus temporaries. It also reports the bytes
of the residuals ``forward`` keeps for ``backward``.

Each expert MLP takes the receiver buffer the ragged backend lays out for it: QuACK's packs the
experts' rows back to back in ``capacity`` rows, and the MXFP8 one starts each expert on a
GROUP_PAD-row boundary, in the few more rows that takes. Rows that were not received are NaN.

The MXFP8 forward and backward are then compiled stage by stage, as the expert MLP runs them, so
the transient memory of the Triton quantizer, the weight quantizers, and the kernels can be told
apart.

Every program compiles with XLA GPU command buffers off, as the hero's do.

Defaults are one hero expert chunk: H = I = 3072, three experts, a 302,336-row capacity::

    python lib/levanter/scripts/bench/bench_mxfp8_expert_mlp.py
"""

import argparse
import json
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp

from levanter.grug._moe.ep_common import _aligned_group_sizes, _aligned_receiver_rows
from levanter.grug._moe.ep_ragged_all_to_all import _CuteExpertMlp, _RoutingWeightGradient
from levanter.grug._moe.mxfp8.adapter import (
    mxfp8_fused_dswiglu,
    mxfp8_fused_gemm,
    mxfp8_fused_swiglu,
    mxfp8_fused_wgrad,
)
from levanter.grug._moe.mxfp8.expert_mlp import (
    _aligned_layout,
    _Mxfp8ExpertMlp,
    _quantized_weights,
)
from levanter.grug._moe.mxfp8.layout import (
    aligned_dispatch_layout,
    deinterleave_gate_up_blocks,
    interleave_gate_up_blocks,
)
from levanter.grug._moe.mxfp8.triton_quant import dual_quantize_padded

_GIB = 2**30
_WARMUP = 3
# experiments/grug/moe_hero_ep/train.py: XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG.
_HERO_COMPILER_OPTIONS = {"xla_gpu_enable_command_buffer": ""}

_DEFAULT_CAPACITY = 302_336
_DEFAULT_ACTIVE_GROUP_SIZES = "140000,100000,62000"
_DEFAULT_HIDDEN = 3072
_DEFAULT_INTERMEDIATE = 3072


def _memory_gib(compiled) -> dict[str, float]:
    stats = compiled.memory_analysis()
    sizes = {
        "argument": stats.argument_size_in_bytes,
        "output": stats.output_size_in_bytes,
        "alias": stats.alias_size_in_bytes,
        "temp": stats.temp_size_in_bytes,
    }
    sizes["peak"] = sizes["argument"] + sizes["output"] - sizes["alias"] + sizes["temp"]
    return {name: round(size / _GIB, 3) for name, size in sizes.items()}


def _median_ms(compiled, args, iters: int) -> float:
    for _ in range(_WARMUP):
        jax.block_until_ready(compiled(*args))
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        times.append(time.perf_counter() - start)
    return round(1e3 * sorted(times)[len(times) // 2], 3)


def _profile(label: str, fn: Callable, args: tuple, iters: int) -> tuple[dict, object]:
    """Compile ``fn``, report its memory and median time, and return one call's output."""
    compiled = jax.jit(fn, compiler_options=_HERO_COMPILER_OPTIONS).lower(*args).compile()
    report = {"program": label, **_memory_gib(compiled), "custom_calls": compiled.as_text().count(" custom-call(")}
    if iters:
        report["ms"] = _median_ms(compiled, args, iters)
    print(json.dumps(report), flush=True)
    return report, compiled(*args)


def _nbytes_gib(tree) -> float:
    return round(sum(leaf.nbytes for leaf in jax.tree.leaves(tree)) / _GIB, 3)


def _receiver_inputs(
    alignment: int, capacity: int, group_sizes: tuple[int, ...], hidden: int, intermediate: int
) -> dict:
    """Operands laid out as the ragged backend lays out ``capacity`` rows for a receiver alignment."""
    experts = len(group_sizes)
    rows = _aligned_receiver_rows(capacity, experts, alignment)
    active = jnp.asarray(group_sizes, jnp.int32)
    spans = _aligned_group_sizes(active, alignment)
    physical = spans.at[-1].add(rows - jnp.sum(spans))
    received = aligned_dispatch_layout(physical, active, rows=rows).source_valid[:, None]
    kx, k13, k2, kct = jax.random.split(jax.random.key(0), 4)
    weights = jax.random.uniform(jax.random.key(1), (rows,), jnp.float32, 0.1, 1.0)
    return {
        "x": jnp.where(received, jax.random.normal(kx, (rows, hidden), jnp.bfloat16), jnp.nan),
        "w13": (jax.random.normal(k13, (experts, hidden, 2 * intermediate)) / hidden**0.5).astype(jnp.bfloat16),
        "w2": (jax.random.normal(k2, (experts, intermediate, hidden)) / intermediate**0.5).astype(jnp.bfloat16),
        "cotangent": jnp.where(received, jax.random.normal(kct, (rows, hidden), jnp.bfloat16), jnp.nan),
        # Each row's routing weight; unreceived rows carry none, as dropped assignments do.
        "weights": jnp.where(received[:, 0], weights, 0.0),
        "physical": physical,
        "active": active,
    }


def _remat_forward_backward(expert_mlp) -> Callable:
    """One rematerialized forward and backward of the expert MLP, as the hero's ``recompute_all`` runs it.

    ``routed`` mirrors `ep_ragged_all_to_all._routed_experts` without the transports: a custom_vjp
    with ``optimize_remat`` whose primal runs ``apply`` and whose fwd rule runs ``forward``, keeping
    no expert outputs (the expert-side routing-weight gradient), and whose bwd rule takes the
    routing-weight gradient from the row dot. Under `jax.checkpoint` with nothing saved, the forward
    pass runs the primal and the backward pass reruns the fwd rule for its residuals.
    """

    @jax.custom_vjp
    def routed(x, w13, w2, weights, physical, active):
        out, _ready = expert_mlp.apply(x, w13, w2, physical, active)
        return out * weights[:, None].astype(out.dtype)

    def routed_fwd(x, w13, w2, weights, physical, active):
        out, residuals = expert_mlp.forward(x, w13, w2, physical, active)
        return out * weights[:, None].astype(out.dtype), (residuals, weights)

    def routed_bwd(saved, cotangent):
        residuals, weights = saved
        dy = (cotangent.astype(jnp.float32) * weights[:, None]).astype(cotangent.dtype)
        dx, dw13, dw2, row_dot = expert_mlp.backward(residuals, dy)
        divisible = weights != 0
        dweights = jnp.where(divisible, row_dot / jnp.where(divisible, weights, 1), 0)
        return dx, dw13, dw2, dweights.astype(weights.dtype), None, None

    routed.defvjp(routed_fwd, routed_bwd, optimize_remat=True)
    layer = jax.checkpoint(routed)

    def remat_forward_backward(x, w13, w2, weights, cotangent, physical, active):
        y, pullback = jax.vjp(lambda *operands: layer(*operands, physical, active), x, w13, w2, weights)
        return y, pullback(cotangent)

    return remat_forward_backward


def _expert_mlp_programs(name: str, expert_mlp, inputs: dict, iters: int) -> list[dict]:
    x, w13, w2, cotangent = inputs["x"], inputs["w13"], inputs["w2"], inputs["cotangent"]
    physical, active = inputs["physical"], inputs["active"]

    def forward_backward(x, w13, w2, cotangent, physical, active):
        out, residuals = expert_mlp.forward(x, w13, w2, physical, active)
        return out, expert_mlp.backward(residuals, cotangent)

    reports = []
    report, _ = _profile(f"{name} apply", lambda *a: expert_mlp.apply(*a)[0], (x, w13, w2, physical, active), iters)
    reports.append(report)
    report, (_, residuals) = _profile(f"{name} forward", expert_mlp.forward, (x, w13, w2, physical, active), iters)
    report["residuals"] = _nbytes_gib(residuals)
    report["residuals_without_weights"] = round(report["residuals"] - _nbytes_gib((w13, w2)), 3)
    reports.append(report)
    report, _ = _profile(f"{name} backward", expert_mlp.backward, (residuals, cotangent), iters)
    reports.append(report)
    del residuals
    report, _ = _profile(
        f"{name} forward+backward", forward_backward, (x, w13, w2, cotangent, physical, active), iters
    )
    reports.append(report)
    report, _ = _profile(
        f"{name} remat forward+backward",
        _remat_forward_backward(expert_mlp),
        (x, w13, w2, inputs["weights"], cotangent, physical, active),
        iters,
    )
    reports.append(report)
    return reports


def _mxfp8_stages(inputs: dict) -> list[dict]:
    """The MXFP8 forward and backward one stage per program, in the order the expert MLP runs them."""
    x, w13, w2, cotangent = inputs["x"], inputs["w13"], inputs["w2"], inputs["cotangent"]
    physical, active = inputs["physical"], inputs["active"]
    rows, hidden = x.shape
    intermediate = w2.shape[1]

    def layout(physical, active):
        return _aligned_layout(physical, active, rows=rows, hidden=hidden)

    def group_ends(physical, active):
        return layout(physical, active)[0].group_ends

    reports = []

    def stage(label, fn, *args):
        report, out = _profile(label, fn, args, iters=0)
        reports.append(report)
        return out

    x_rows, x_rows_scales, x_col, x_col_scales = stage(
        "fwd: dual-quantize x", lambda x, *sizes: dual_quantize_padded(x, *layout(*sizes)), x, physical, active
    )
    w13_q, w13_scales, w2_q, w2_scales = stage(
        "fwd: quantize weights",
        lambda w13, w2: (
            *_quantized_weights(jnp.swapaxes(interleave_gate_up_blocks(w13), 1, 2)),
            *_quantized_weights(jnp.swapaxes(w2, 1, 2)),
        ),
        w13,
        w2,
    )
    c13, h, h_col, h_scales, h_col_scales = stage(
        "fwd: gate/up GEMM + SwiGLU + quantize h",
        lambda *a: mxfp8_fused_swiglu(*a[:-2], group_ends(*a[-2:])),
        x_rows,
        x_rows_scales,
        w13_q,
        w13_scales,
        physical,
        active,
    )
    del x_rows, x_rows_scales, w13_q, w13_scales
    stage(
        "fwd: down GEMM",
        lambda *a: mxfp8_fused_gemm(*a[:-2], group_ends(*a[-2:])),
        h,
        h_scales,
        w2_q,
        w2_scales,
        physical,
        active,
    )
    del h, h_scales, w2_q, w2_scales

    g_rows, g_rows_scales, g_col, g_col_scales = stage(
        "bwd: dual-quantize cotangent",
        lambda cotangent, *sizes: dual_quantize_padded(cotangent, *layout(*sizes)),
        cotangent,
        physical,
        active,
    )
    w2_q, w2_scales, w13_q, w13_scales = stage(
        "bwd: quantize weights",
        lambda w13, w2: (*_quantized_weights(w2), *_quantized_weights(interleave_gate_up_blocks(w13))),
        w13,
        w2,
    )
    dc13, dc13_col, dc13_scales, dc13_col_scales, _row_dot = stage(
        "bwd: down dgrad GEMM + dSwiGLU + quantize dc13",
        lambda *a: mxfp8_fused_dswiglu(*a[:-2], group_ends(*a[-2:])),
        g_rows,
        g_rows_scales,
        w2_q,
        w2_scales,
        c13,
        physical,
        active,
    )
    del g_rows, g_rows_scales, w2_q, w2_scales, c13
    stage(
        "bwd: gate/up dgrad GEMM",
        lambda *a: mxfp8_fused_gemm(*a[:-2], group_ends(*a[-2:])),
        dc13,
        dc13_scales,
        w13_q,
        w13_scales,
        physical,
        active,
    )
    stage(
        "bwd: weight-gradient GEMMs",
        lambda xc, xcs, dcc, dccs, hc, hcs, gc, gcs, *sizes: (
            deinterleave_gate_up_blocks(
                mxfp8_fused_wgrad(xc, xcs, dcc, dccs.reshape(2 * intermediate, -1), group_ends(*sizes))
            ),
            mxfp8_fused_wgrad(hc, hcs.reshape(intermediate, -1), gc, gcs, group_ends(*sizes)),
        ),
        x_col,
        x_col_scales,
        dc13_col,
        dc13_col_scales,
        h_col,
        h_col_scales,
        g_col,
        g_col_scales,
        physical,
        active,
    )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capacity", type=int, default=_DEFAULT_CAPACITY)
    parser.add_argument("--active-group-sizes", default=_DEFAULT_ACTIVE_GROUP_SIZES)
    parser.add_argument("--hidden", type=int, default=_DEFAULT_HIDDEN)
    parser.add_argument("--intermediate", type=int, default=_DEFAULT_INTERMEDIATE)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    group_sizes = tuple(int(size) for size in args.active_group_sizes.split(","))
    if sum(group_sizes) > args.capacity:
        raise ValueError(f"active group sizes {group_sizes} exceed the capacity {args.capacity}")
    print(
        json.dumps(
            {
                "capacity": args.capacity,
                "active_group_sizes": group_sizes,
                "hidden": args.hidden,
                "intermediate": args.intermediate,
            }
        ),
        flush=True,
    )

    mxfp8 = _Mxfp8ExpertMlp(routing_weight_gradient=_RoutingWeightGradient.EXPERT_SIDE)
    quack = _CuteExpertMlp(routing_weight_gradient=_RoutingWeightGradient.EXPERT_SIDE)
    reports = []
    for name, expert_mlp in (("mxfp8", mxfp8), ("quack", quack)):
        inputs = _receiver_inputs(
            expert_mlp.receiver_alignment, args.capacity, group_sizes, args.hidden, args.intermediate
        )
        print(json.dumps({"expert_mlp": name, "receiver_rows": inputs["x"].shape[0]}), flush=True)
        reports += _expert_mlp_programs(name, expert_mlp, inputs, args.iters)
        if expert_mlp is mxfp8:
            reports += _mxfp8_stages(inputs)
    print("SUMMARY " + json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
