#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grouped-GEMM baselines at the hero's real expert shapes and token distribution.

Times the six grouped GEMMs one expert-MLP chunk runs per layer -- gate/up forward, down
forward, the two backward activation contractions (dh, dx), and the two weight gradients (dw13,
dw2) -- on every implementation reachable from this stack, and checks each against an fp32
per-group reference computed from the same bf16 operands:

- ``xla``: `jax.lax.ragged_dot_general` as XLA lowers it. On NVIDIA GPUs the `RaggedDotRewriter`
  expands it to a dense dot over a group-masked, group-broadcast LHS (E times the FLOPs and an
  [rows, E, K] temporary), unless the cuDNN ragged-dot fusion is enabled.
- ``xla-cudnn``: the same call with ``--xla_gpu_experimental_use_ragged_dot_fusion=true``, which
  lowers the non-contracting-ragged (forward-shaped) calls to a cuDNN grouped GEMM. Needs cuDNN
  >= 9.22; the weight gradients (ragged contracting dimension) always take the expansion. The run
  records the cuDNN version and whether the flag could take effect.
- ``triton``: haliax's Pallas-Triton ragged dot (the tokamax kernel at tokamax's untuned default
  config), plus a small block-size sweep with ``--sweep``.
- ``quack``: the shipped QuACK configuration from `sonic_cute`, plus the same tile/cluster/CLC
  sweep `bench_grouped_wgrad.py` walks, with ``--sweep``.

cuBLASLt grouped GEMM is not reachable from JAX on NVIDIA: XLA's
``xla_gpu_experimental_use_ragged_dot_grouped_gemm`` path is hipBLASLt-only.

Group sizes come from ``--sizes-json`` when given (a list of per-expert active row counts for one
chunk, e.g. dumped from a training step), otherwise from a skewed draw that matches the hero's
observed drop regime: three experts sharing ``rows`` at capacity factor 1.15 with one expert
oversubscribed and clipped to the chunk capacity.

Run each implementation in its own process: the cuDNN flag is process-global, and a CUDA fault in
one kernel poisons the context for every later variant. ``run_all.sh`` next to this file does that.

    uv run python bench_grouped_gemm_baselines.py --impl quack --sweep --json quack.json
    XLA_FLAGS=--xla_gpu_experimental_use_ragged_dot_fusion=true \\
        uv run python bench_grouped_gemm_baselines.py --impl xla-cudnn --json cudnn.json
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np

HERO_ROWS = 301_466
HERO_EXPERTS = 3
HERO_HIDDEN = 3_072
HERO_INTERMEDIATE = 3_072
CAPACITY_FACTOR = 1.15

_WARMUP = 3
_ITERS = 10
_GROUP_SEED = 1


@dataclass
class Result:
    gemm: str
    impl: str
    variant: str
    rows: int
    active_rows: int
    sizes: list[int]
    m: int
    n: int
    k_or_rows: int
    seconds: float
    tflops: float
    max_abs_over_scale: float
    mean_abs_over_scale: float
    note: str = ""


def _hero_like_sizes(rows: int, experts: int, seed: int) -> np.ndarray:
    """Skewed per-expert counts for one receiver chunk, clipped to the chunk capacity.

    The hero's router at capacity 1.15 and two chunks drops well under 0.1% of assignments at
    steady state, so most steps look like: mean load ``rows / 1.15 / experts`` per expert with
    a lognormal spread, one expert occasionally over capacity. The draw keeps every boundary off
    the 256-row alignment a tile scheduler would otherwise get for free.
    """
    rng = np.random.default_rng(seed)
    mean = rows / CAPACITY_FACTOR / experts
    sizes = np.floor(mean * rng.lognormal(0.0, 0.35, size=experts)).astype(np.int64)
    sizes = sizes - (sizes % 256) + 137
    overshoot = int(sizes.sum()) - rows
    if overshoot > 0:
        sizes[np.argmax(sizes)] -= overshoot
    assert sizes.min() >= 1 and sizes.sum() <= rows
    return sizes


def _time(fn, *args) -> float:
    for _ in range(_WARMUP):
        jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    for _ in range(_ITERS):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / _ITERS


def _errors(got, want: np.ndarray) -> tuple[float, float]:
    delta = np.abs(np.asarray(got, dtype=np.float64) - want)
    scale = max(float(np.abs(want).max()), 1e-30)
    return float(delta.max() / scale), float(delta.mean() / scale)


def _ref_rowgrouped(a, b_per_group, sizes: np.ndarray) -> np.ndarray:
    """fp32 per-group ``a[rows_e] @ b[e]`` written into a [rows, n] buffer (zeros past the last group)."""
    out = np.zeros((a.shape[0], b_per_group.shape[-1]), dtype=np.float64)
    start = 0
    for e, size in enumerate(sizes):
        stop = start + int(size)
        ae = a[start:stop].astype(jnp.float32)
        be = b_per_group[e].astype(jnp.float32)
        out[start:stop] = np.asarray(jax.block_until_ready(ae @ be), dtype=np.float64)
        start = stop
    return out


def _ref_wgrad(lhs, rhs, sizes: np.ndarray) -> np.ndarray:
    out = np.zeros((len(sizes), lhs.shape[1], rhs.shape[1]), dtype=np.float64)
    start = 0
    for e, size in enumerate(sizes):
        stop = start + int(size)
        a = lhs[start:stop].astype(jnp.float32)
        b = rhs[start:stop].astype(jnp.float32)
        out[e] = np.asarray(jax.block_until_ready(a.T @ b), dtype=np.float64)
        start = stop
    return out


# --------------------------------------------------------------------------------------------
# Implementations. Each returns dict gemm-name -> (jitted fn, args, reference, flops).
# --------------------------------------------------------------------------------------------


def _xla_variants(impl: str, x, w13, w2, dy, sizes: np.ndarray, rows: int):
    """`ragged_dot_general` for the four row-grouped GEMMs and its transpose for the weight gradients."""
    from haliax.nn.ragged_dot import ragged_dot  # noqa: PLC0415

    physical = jnp.asarray(sizes, jnp.int32).at[-1].add(rows - int(sizes.sum()))
    forced = "xla" if impl.startswith("xla") else "triton"

    def rd(a, b, gs):
        return ragged_dot(a, b, gs, implementation=forced)

    def wgrad(lhs, rhs, gs):
        weights = jnp.zeros((len(sizes), lhs.shape[1], rhs.shape[1]), dtype=lhs.dtype)
        return jax.vjp(lambda w: rd(lhs, w, gs), weights)[1](rhs)[0]

    h = jnp.zeros((rows, w2.shape[1]), x.dtype)
    variants = {
        "gate_up_fwd": (jax.jit(rd), (x, w13, physical), lambda: _ref_rowgrouped(x, w13, sizes)),
        "down_fwd": (jax.jit(rd), (h, w2, physical), lambda: _ref_rowgrouped(h, w2, sizes)),
        "dh_bwd": (
            jax.jit(rd),
            (dy, jnp.swapaxes(w2, 1, 2), physical),
            lambda: _ref_rowgrouped(dy, jnp.swapaxes(w2, 1, 2), sizes),
        ),
        "dw2_bwd": (jax.jit(wgrad), (h, dy, physical), lambda: _ref_wgrad(h, dy, sizes)),
    }
    return variants


def _quack_variants(config: dict, x, w13, w2, dy, sizes: np.ndarray, rows: int):
    from levanter.grug._moe.common import _interleave_gate_up  # noqa: PLC0415
    from levanter.grug._moe.quack_moe_cute import (  # noqa: PLC0415
        quack_gated_grouped_gemm,
        quack_grouped_gemm,
        quack_grouped_wgrad,
    )

    cu = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(jnp.asarray(sizes, jnp.int32))])
    moe_dim = w2.shape[1]
    w13_il = _interleave_gate_up(w13, moe_dim)
    gated_kw = config["gated"]
    grouped_kw = config["grouped"]
    wgrad_kw = config["wgrad"]

    def gated(a, b, c):
        return quack_gated_grouped_gemm(a, b, c, return_preact=True, **gated_kw)[0]

    h = jnp.zeros((rows, moe_dim), x.dtype)
    variants = {
        # The gated kernel also applies SwiGLU; the reference here is the pre-activation, which the
        # kernel returns as its first output (interleaved gate/up columns).
        "gate_up_fwd": (
            jax.jit(gated),
            (x, w13_il, cu),
            lambda: _ref_rowgrouped(x, w13_il, sizes),
        ),
        "down_fwd": (
            jax.jit(functools.partial(quack_grouped_gemm, b_major="n", **grouped_kw)),
            (h, w2, cu),
            lambda: _ref_rowgrouped(h, w2, sizes),
        ),
        "dh_bwd": (
            jax.jit(functools.partial(quack_grouped_gemm, b_major="k", **grouped_kw)),
            (dy, w2, cu),
            lambda: _ref_rowgrouped(dy, jnp.swapaxes(w2, 1, 2), sizes),
        ),
        "dw2_bwd": (
            jax.jit(functools.partial(quack_grouped_wgrad, **wgrad_kw)),
            (h, dy, cu),
            lambda: _ref_wgrad(h, dy, sizes),
        ),
    }
    return variants


def _quack_sweep_configs():
    tiles = [(128, 128), (128, 256), (256, 128), (256, 256)]
    clusters = [(2, 1, 1), (2, 2, 1)]
    for tile, cluster, clc in itertools.product(tiles, clusters, [False, True]):
        kw = dict(tile_mn=tile, cluster_mnk=cluster, use_clc_persistence=clc)
        yield f"tile={tile[0]}x{tile[1]} cluster={cluster[0]}x{cluster[1]} clc={int(clc)}", {
            "gated": kw,
            "grouped": kw,
            "wgrad": dict(tile_mn=tile, cluster_mnk=cluster, use_clc_persistence=clc),
        }


def _triton_sweep(x, w13, sizes, rows):
    """Block-size sweep for haliax's Pallas-Triton kernel (its defaults are tokamax's untuned ones)."""
    from haliax.nn import ragged_dot as rd_module  # noqa: PLC0415

    physical = jnp.asarray(sizes, jnp.int32).at[-1].add(rows - int(sizes.sum()))
    for block_m, block_n, block_k in itertools.product([64, 128], [128, 256], [32, 64, 128]):

        def sizes_fn(m, k, n, bm=block_m, bn=block_n, bk=block_k):
            return bm, bn, bk

        original = rd_module._triton_default_block_sizes
        rd_module._triton_default_block_sizes = sizes_fn
        rd_module._triton_default_matmul.cache_clear()
        try:
            fn = jax.jit(lambda a, b, gs: rd_module.ragged_dot(a, b, gs, implementation="triton"))
            yield f"block={block_m}x{block_n}x{block_k}", fn, (x, w13, physical)
        finally:
            rd_module._triton_default_block_sizes = original
            rd_module._triton_default_matmul.cache_clear()


def _cudnn_version() -> str:
    try:
        from jax._src.lib import cuda_versions  # noqa: PLC0415

        return str(cuda_versions.cudnn_get_version())
    except Exception as exc:
        return f"unknown ({exc})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--impl", choices=["xla", "xla-cudnn", "triton", "quack"], required=True)
    parser.add_argument("--rows", type=int, default=HERO_ROWS)
    parser.add_argument("--experts", type=int, default=HERO_EXPERTS)
    parser.add_argument("--hidden", type=int, default=HERO_HIDDEN)
    parser.add_argument("--intermediate", type=int, default=HERO_INTERMEDIATE)
    parser.add_argument("--sizes-json", type=str, default=None, help="per-expert active rows for one chunk")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    if args.impl == "xla-cudnn" and "xla_gpu_experimental_use_ragged_dot_fusion=true" not in os.environ.get(
        "XLA_FLAGS", ""
    ):
        print(
            "xla-cudnn needs XLA_FLAGS=--xla_gpu_experimental_use_ragged_dot_fusion=true in the environment",
            file=sys.stderr,
        )
        return 2

    rows, experts = args.rows, args.experts
    if args.sizes_json:
        with open(args.sizes_json) as fh:
            sizes = np.asarray(json.load(fh), dtype=np.int64)
        if len(sizes) != experts or sizes.sum() > rows:
            raise ValueError(f"sizes {sizes.tolist()} do not fit experts={experts} rows={rows}")
    else:
        sizes = _hero_like_sizes(rows, experts, _GROUP_SEED)
    active = int(sizes.sum())
    print(f"device: {jax.devices()[0].device_kind}  cudnn: {_cudnn_version()}")
    print(f"sizes: {sizes.tolist()} active {active}/{rows}")

    rng = np.random.default_rng(0)
    hidden, inter = args.hidden, args.intermediate
    x = jnp.asarray(rng.standard_normal((rows, hidden), dtype=np.float32), jnp.bfloat16)
    dy = jnp.asarray(rng.standard_normal((rows, hidden), dtype=np.float32), jnp.bfloat16)
    w13 = jnp.asarray(
        rng.standard_normal((experts, hidden, 2 * inter), dtype=np.float32) / np.sqrt(hidden), jnp.bfloat16
    )
    w2 = jnp.asarray(rng.standard_normal((experts, inter, hidden), dtype=np.float32) / np.sqrt(inter), jnp.bfloat16)

    flops = {
        "gate_up_fwd": 2.0 * active * hidden * 2 * inter,
        "down_fwd": 2.0 * active * inter * hidden,
        "dh_bwd": 2.0 * active * hidden * inter,
        "dw2_bwd": 2.0 * active * inter * hidden,
    }
    dims = {
        "gate_up_fwd": (hidden, 2 * inter),
        "down_fwd": (inter, hidden),
        "dh_bwd": (hidden, inter),
        "dw2_bwd": (inter, hidden),
    }

    results: list[Result] = []

    def record(gemm, impl, variant, fn, fn_args, reference, note=""):
        try:
            out = jax.block_until_ready(fn(*fn_args))
        except Exception as exc:
            print(f"  {gemm:<12} {variant:<40} unsupported ({type(exc).__name__}: {str(exc)[:100]})")
            return
        want = reference()
        out_np = np.asarray(out, dtype=np.float32)
        if out_np.shape[0] == rows and want.ndim == 2:
            # Row-grouped outputs: compare active rows only. What the kernels leave in padding
            # rows is masked by the backend and is not part of the contract.
            out_np, want = out_np[:active], want[:active]
        max_err, mean_err = _errors(out_np, want)
        seconds = _time(fn, *fn_args)
        m, n = dims[gemm]
        results.append(
            Result(
                gemm=gemm,
                impl=impl,
                variant=variant,
                rows=rows,
                active_rows=active,
                sizes=sizes.tolist(),
                m=m,
                n=n,
                k_or_rows=active,
                seconds=seconds,
                tflops=flops[gemm] / seconds / 1e12,
                max_abs_over_scale=max_err,
                mean_abs_over_scale=mean_err,
                note=note,
            )
        )
        tflops = results[-1].tflops
        print(f"  {gemm:<12} {variant:<40} {seconds * 1e3:9.3f} ms {tflops:7.1f} TFLOP/s  max_err {max_err:.2e}")

    if args.impl in ("xla", "xla-cudnn", "triton"):
        note = ""
        if args.impl == "xla-cudnn":
            note = "requires cuDNN>=9.22; weight gradients always take the masked expansion"
        for gemm, (fn, fn_args, ref) in _xla_variants(args.impl, x, w13, w2, dy, sizes, rows).items():
            record(gemm, args.impl, "default", fn, fn_args, ref, note)
        if args.impl == "triton" and args.sweep:
            for variant, fn, fn_args in _triton_sweep(x, w13, sizes, rows):
                record("gate_up_fwd", "triton", variant, fn, fn_args, lambda: _ref_rowgrouped(x, w13, sizes))
    else:
        from levanter.grug._moe.sonic_cute import _QUACK_GATED_KW, _QUACK_GROUPED_KW, _QUACK_WGRAD_KW  # noqa: PLC0415

        shipped = {"gated": _QUACK_GATED_KW, "grouped": _QUACK_GROUPED_KW, "wgrad": _QUACK_WGRAD_KW}
        for gemm, (fn, fn_args, ref) in _quack_variants(shipped, x, w13, w2, dy, sizes, rows).items():
            record(gemm, "quack", "shipped", fn, fn_args, ref)
        if args.sweep:
            for variant, config in _quack_sweep_configs():
                for gemm, (fn, fn_args, ref) in _quack_variants(config, x, w13, w2, dy, sizes, rows).items():
                    record(gemm, "quack", variant, fn, fn_args, ref)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
