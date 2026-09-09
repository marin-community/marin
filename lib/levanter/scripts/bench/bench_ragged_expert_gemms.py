# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Time every candidate grouped-GEMM lowering for the ragged EP expert MLP on one GPU.

The ragged all-to-all EP backend runs six grouped GEMMs per expert chunk: the gate/up and down
forwards, the two activation gradients, and the two weight gradients. This script times each
leg, plus the whole MLP forward-and-backward, for each lowering that can compute it:

    quack        the shipped QuACK SM100 kernels (``sonic_cute``), for reference
    xla          ``jax.lax.ragged_dot`` at XLA's defaults (masked dense expansion on CUDA)
    xla-cudnn    ``jax.lax.ragged_dot`` with ``xla_gpu_experimental_use_ragged_dot_fusion``
    xla-triton   ``jax.lax.ragged_dot`` with ``xla_gpu_experimental_triton_ragged_dot`` (fork build)
    jax-pallas   ``jax.lax.ragged_dot`` under ``jax_ragged_dot_use_gpu_pallas_triton_lowering``
    haliax       haliax's Pallas-Triton grouped kernel (``implementation="triton"``)
    dense        one dense cuBLAS dot over the whole buffer, an upper bound on any grouping

For each XLA variant it also prints which kernels the compiled HLO contains, so a flag that did
not engage is visible rather than reported as a slow result.

Shapes default to one EP64 chunk of the d6144 LatentMoE hero. Group sizes are uneven and sum to
the capacity factor's fill, with zeros past the last group as in production.

Example::

    python lib/levanter/scripts/bench/bench_ragged_expert_gemms.py --json results.json
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np

_DEFAULT_ROWS = 301_466
_DEFAULT_EXPERTS = 3
_DEFAULT_HIDDEN = 3_072
_DEFAULT_INTERMEDIATE = 3_072
_DEFAULT_FILL = 1.0 / 1.15

_WARMUP = 3
_ITERS = 10
_GROUP_SEED = 1

_RAGGED_FWD_DIMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((1,), (1,)), ((), ())),
    lhs_ragged_dimensions=(0,),
    rhs_group_dimensions=(0,),
)


@dataclass
class Result:
    variant: str
    leg: str
    seconds: float
    tflops: float
    max_rel_err: float
    kernels: str


def _group_sizes(rows: int, experts: int, fill: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.7, 1.3, size=experts)
    sizes = np.floor(weights / weights.sum() * rows * fill).astype(np.int64)
    # Off every tile boundary, so a kernel that reads past its group shows up as an error.
    sizes = np.maximum(sizes - (sizes % 256) + 137, 1)
    assert sizes.sum() <= rows
    return sizes


def _time(fn: Callable, *args) -> float:
    for _ in range(_WARMUP):
        jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    for _ in range(_ITERS):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / _ITERS


def _rel_err(got, want, active: int) -> float:
    """Max abs error over the active rows, relative to the reference's mean magnitude.

    Row-shaped outputs are compared on the rows inside a group only: kernels driven by segment
    boundaries leave the trailing rows unwritten, and production never reads them.
    """
    got = np.asarray(jax.device_get(got), dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.ndim == 2:
        got, want = got[:active], want[:active]
    return float(np.abs(got - want).max() / (np.abs(want).mean() + 1e-30))


def _kernel_summary(text: str) -> str:
    """Name the fusion backends and library calls in a compiled HLO, to show which lowering engaged."""
    counts: dict[str, int] = {}
    for kind in re.findall(r'"kind":"([^"]+)"', text):
        counts[kind] = counts.get(kind, 0) + 1
    for target in re.findall(r'custom_call_target="([^"]+)"', text):
        counts[target] = counts.get(target, 0) + 1
    if "ragged-dot(" in text:
        counts["ragged-dot"] = text.count("ragged-dot(")
    return ",".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "none"


def _compiled_text(fn, *args) -> str:
    try:
        return fn.lower(*args).compile().as_text()
    except Exception:  # noqa: BLE001 -- engagement is advisory
        return ""


# --- lowerings -------------------------------------------------------------------------------


def _xla_ragged_dot(lhs, rhs, group_sizes):
    return jax.lax.ragged_dot_general(lhs, rhs, group_sizes, ragged_dot_dimension_numbers=_RAGGED_FWD_DIMS)


def _make_xla_variant(name: str, compiler_options: dict | None, ragged_dot):
    """Build the leg functions for a ``ragged_dot``-shaped callable under given XLA options."""

    def jit(fn):
        return jax.jit(fn, compiler_options=compiler_options)

    def mlp(x, w13, w2, gs):
        gu = ragged_dot(x, w13, gs)
        moe_dim = w2.shape[1]
        gate, up = gu[:, :moe_dim], gu[:, moe_dim:]
        return ragged_dot(jax.nn.silu(gate) * up, w2, gs)

    legs = {
        "fwd_w13": jit(lambda x, w13, gs: ragged_dot(x, w13, gs)),
        "fwd_w2": jit(lambda h, w2, gs: ragged_dot(h, w2, gs)),
        "dx_w13": jit(lambda x, w13, gs, dgu: jax.vjp(lambda a: ragged_dot(a, w13, gs), x)[1](dgu)[0]),
        "dx_w2": jit(lambda h, w2, gs, dy: jax.vjp(lambda a: ragged_dot(a, w2, gs), h)[1](dy)[0]),
        "dw13": jit(lambda x, w13, gs, dgu: jax.vjp(lambda w: ragged_dot(x, w, gs), w13)[1](dgu)[0]),
        "dw2": jit(lambda h, w2, gs, dy: jax.vjp(lambda w: ragged_dot(h, w, gs), w2)[1](dy)[0]),
        "mlp_fwd": jit(mlp),
        "mlp_fwd_bwd": jit(lambda x, w13, w2, gs, dy: jax.vjp(lambda a, b, c: mlp(a, b, c, gs), x, w13, w2)[1](dy)),
    }
    return name, legs


def _dense_variant():
    def mlp(x, w13, w2):
        gu = x @ w13
        moe_dim = w2.shape[1]
        gate, up = gu[:, :moe_dim], gu[:, moe_dim:]
        return (jax.nn.silu(gate) * up) @ w2

    legs = {
        "fwd_w13": jax.jit(lambda x, w13, gs: x @ w13[0]),
        "fwd_w2": jax.jit(lambda h, w2, gs: h @ w2[0]),
        "dx_w13": jax.jit(lambda x, w13, gs, dgu: dgu @ w13[0].T),
        "dx_w2": jax.jit(lambda h, w2, gs, dy: dy @ w2[0].T),
        "dw13": jax.jit(lambda x, w13, gs, dgu: (x.T @ dgu)[None]),
        "dw2": jax.jit(lambda h, w2, gs, dy: (h.T @ dy)[None]),
        "mlp_fwd": jax.jit(lambda x, w13, w2, gs: mlp(x, w13[0], w2[0])),
        "mlp_fwd_bwd": jax.jit(
            lambda x, w13, w2, gs, dy: jax.vjp(lambda a, b, c: mlp(a, b, c), x, w13[0], w2[0])[1](dy)
        ),
    }
    return "dense", {leg: (fn, None) for leg, fn in legs.items()}


def _quack_variant():
    # QuACK ships only with the CUDA 13 GPU extra, so the reference stays importable without it.
    from levanter.grug._moe.common import _interleave_gate_up, _unpack_pairs_u32  # noqa: PLC0415
    from levanter.grug._moe.quack_moe_cute import (  # noqa: PLC0415
        quack_gated_grouped_gemm,
        quack_grouped_gemm,
        quack_grouped_wgrad,
    )
    from levanter.grug._moe.sonic_cute import (  # noqa: PLC0415
        _QUACK_GATED_KW,
        _QUACK_GROUPED_KW,
        _QUACK_WGRAD_KW,
        _expert_mlp_quack_wgrad,
    )

    def cu_of(gs):
        return jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(gs).astype(jnp.int32)])

    def fwd_w13(x, w13, gs):
        # Fused SwiGLU epilogue: returns the pre-activation too, like the shipped path.
        w13_il = _interleave_gate_up(w13, w13.shape[2] // 2)
        gu, _h = quack_gated_grouped_gemm(x, w13_il, cu_of(gs), return_preact=True, **_QUACK_GATED_KW)
        return gu

    def deinterleave(gu):
        return jnp.concatenate(_unpack_pairs_u32(gu), axis=-1)

    def dx_w13(x, w13, gs, dgu):
        # Interleaving both operands' 2I axis is a shared permutation of the contraction, so
        # this equals ``dgu @ w13[g].T`` and checks against the same reference.
        moe_dim = w13.shape[2] // 2
        w13_il = _interleave_gate_up(w13, moe_dim)
        dgu_il = _interleave_gate_up(dgu[None], moe_dim)[0]
        return quack_grouped_gemm(dgu_il, w13_il, cu_of(gs), b_major="k", **_QUACK_GROUPED_KW)

    def mlp_fwd_bwd(x, w13, w2, gs, dy):
        w13_il = _interleave_gate_up(w13, w2.shape[1])
        return jax.vjp(lambda a, b, c: _expert_mlp_quack_wgrad(a, b, c, cu_of(gs)), x, w13_il, w2)[1](dy)

    legs = {
        "fwd_w13": (jax.jit(fwd_w13), deinterleave),
        "fwd_w2": jax.jit(lambda h, w2, gs: quack_grouped_gemm(h, w2, cu_of(gs), b_major="n", **_QUACK_GROUPED_KW)),
        "dx_w13": jax.jit(dx_w13),
        "dx_w2": jax.jit(
            lambda h, w2, gs, dy: quack_grouped_gemm(dy, w2, cu_of(gs), b_major="k", **_QUACK_GROUPED_KW)
        ),
        "dw13": jax.jit(lambda x, w13, gs, dgu: quack_grouped_wgrad(x, dgu, cu_of(gs), **_QUACK_WGRAD_KW)),
        "dw2": jax.jit(lambda h, w2, gs, dy: quack_grouped_wgrad(h, dy, cu_of(gs), **_QUACK_WGRAD_KW)),
        "mlp_fwd": jax.jit(
            lambda x, w13, w2, gs: _expert_mlp_quack_wgrad(x, _interleave_gate_up(w13, w2.shape[1]), w2, cu_of(gs))
        ),
        "mlp_fwd_bwd": jax.jit(mlp_fwd_bwd),
    }
    return "quack", legs


# --- references ------------------------------------------------------------------------------


def _ref_fwd(lhs, rhs, sizes) -> np.ndarray:
    """Per-group ``lhs @ rhs[g]`` in float32; rows past the last group are zero."""
    out = np.zeros((lhs.shape[0], rhs.shape[2]), dtype=np.float32)
    start = 0
    for g, size in enumerate(sizes):
        stop = start + int(size)
        out[start:stop] = np.asarray(
            jax.block_until_ready(lhs[start:stop].astype(jnp.float32) @ rhs[g].astype(jnp.float32))
        )
        start = stop
    return out


def _ref_wgrad(lhs, rhs, sizes) -> np.ndarray:
    out = []
    start = 0
    for size in sizes:
        stop = start + int(size)
        out.append(
            np.asarray(
                jax.block_until_ready(lhs[start:stop].astype(jnp.float32).T @ rhs[start:stop].astype(jnp.float32))
            )
        )
        start = stop
    return np.stack(out)


# --- driver ----------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=_DEFAULT_ROWS)
    parser.add_argument("--experts", type=int, default=_DEFAULT_EXPERTS)
    parser.add_argument("--hidden", type=int, default=_DEFAULT_HIDDEN)
    parser.add_argument("--intermediate", type=int, default=_DEFAULT_INTERMEDIATE)
    parser.add_argument("--fill", type=float, default=_DEFAULT_FILL, help="fraction of rows inside a group")
    parser.add_argument(
        "--group-sizes",
        default=None,
        help="Comma-separated explicit group sizes (overrides --experts/--fill); a 0 tests an empty expert.",
    )
    parser.add_argument("--variants", default="quack,xla,xla-cudnn,xla-triton,haliax,dense")
    parser.add_argument("--legs", default="fwd_w13,fwd_w2,dx_w13,dx_w2,dw13,dw2,mlp_fwd,mlp_fwd_bwd")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    rows, experts, hidden, inter = args.rows, args.experts, args.hidden, args.intermediate
    print(f"device: {jax.devices()[0].device_kind}  jax {jax.__version__}")
    if args.group_sizes:
        sizes = np.array([int(v) for v in args.group_sizes.split(",")], dtype=np.int64)
        experts = len(sizes)
        assert sizes.sum() <= rows
    else:
        sizes = _group_sizes(rows, experts, args.fill, _GROUP_SEED)
    active = int(sizes.sum())
    print(f"rows {rows} experts {experts} hidden {hidden} intermediate {inter} sizes {sizes.tolist()} active {active}")

    key = jax.random.key(0)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    x = jax.random.normal(k1, (rows, hidden), jnp.bfloat16)
    w13 = (jax.random.normal(k2, (experts, hidden, 2 * inter), jnp.float32) / np.sqrt(hidden)).astype(jnp.bfloat16)
    w2 = (jax.random.normal(k3, (experts, inter, hidden), jnp.float32) / np.sqrt(inter)).astype(jnp.bfloat16)
    h = jax.random.normal(k4, (rows, inter), jnp.bfloat16)
    dy = jax.random.normal(k5, (rows, hidden), jnp.bfloat16)
    dgu = jax.random.normal(k5, (rows, 2 * inter), jnp.bfloat16)
    # Zero the rows past the last group, as the receiver buffer's zero init leaves them.
    mask = (jnp.arange(rows) < active)[:, None]
    x, h, dy, dgu = (jnp.where(mask, a, 0) for a in (x, h, dy, dgu))
    gs = jnp.asarray(sizes, dtype=jnp.int32)

    refs = {
        "fwd_w13": _ref_fwd(x, w13, sizes),
        "fwd_w2": _ref_fwd(h, w2, sizes),
        "dx_w13": _ref_fwd(dgu, jnp.swapaxes(w13, 1, 2), sizes),
        "dx_w2": _ref_fwd(dy, jnp.swapaxes(w2, 1, 2), sizes),
        "dw13": _ref_wgrad(x, dgu, sizes),
        "dw2": _ref_wgrad(h, dy, sizes),
    }
    flops = {
        "fwd_w13": 2.0 * active * hidden * 2 * inter,
        "fwd_w2": 2.0 * active * inter * hidden,
        "dx_w13": 2.0 * active * hidden * 2 * inter,
        "dx_w2": 2.0 * active * inter * hidden,
        "dw13": 2.0 * active * hidden * 2 * inter,
        "dw2": 2.0 * active * inter * hidden,
    }
    flops["mlp_fwd"] = flops["fwd_w13"] + flops["fwd_w2"]
    flops["mlp_fwd_bwd"] = 3 * flops["mlp_fwd"]
    leg_args = {
        "fwd_w13": (x, w13, gs),
        "fwd_w2": (h, w2, gs),
        "dx_w13": (x, w13, gs, dgu),
        "dx_w2": (h, w2, gs, dy),
        "dw13": (x, w13, gs, dgu),
        "dw2": (h, w2, gs, dy),
        "mlp_fwd": (x, w13, w2, gs),
        "mlp_fwd_bwd": (x, w13, w2, gs, dy),
    }

    def build(name: str):
        if name == "quack":
            return _quack_variant()
        if name == "dense":
            return _dense_variant()
        if name == "xla":
            return _make_xla_variant("xla", None, _xla_ragged_dot)
        if name == "xla-cudnn":
            return _make_xla_variant(
                "xla-cudnn", {"xla_gpu_experimental_use_ragged_dot_fusion": True}, _xla_ragged_dot
            )
        if name == "xla-triton":
            return _make_xla_variant(
                "xla-triton",
                {
                    "xla_gpu_experimental_triton_ragged_dot": True,
                    "xla_gpu_experimental_enable_tiling_propagation": True,
                },
                _xla_ragged_dot,
            )
        if name == "xla-cudnn-triton":
            # cuDNN takes the forward-shaped contractions, Triton the rest (XLA's priority order).
            return _make_xla_variant(
                "xla-cudnn-triton",
                {
                    "xla_gpu_experimental_use_ragged_dot_fusion": True,
                    "xla_gpu_experimental_triton_ragged_dot": True,
                    "xla_gpu_experimental_enable_tiling_propagation": True,
                },
                _xla_ragged_dot,
            )
        if name == "jax-pallas":
            return _make_xla_variant("jax-pallas", None, _xla_ragged_dot)
        if name == "haliax":
            from haliax.nn.ragged_dot import ragged_dot as hx_ragged_dot  # noqa: PLC0415

            return _make_xla_variant("haliax", None, functools.partial(hx_ragged_dot, implementation="triton"))
        raise ValueError(name)

    results: list[Result] = []
    for variant_name in args.variants.split(","):
        print(f"\n== {variant_name}")
        try:
            name, legs = build(variant_name)
        except Exception:  # noqa: BLE001 -- an unavailable lowering is a datum, not an abort
            print(f"  build failed:\n{traceback.format_exc()}")
            continue
        pallas = variant_name == "jax-pallas"
        if pallas:
            jax.config.update("jax_ragged_dot_use_gpu_pallas_triton_lowering", True)
        try:
            for leg in args.legs.split(","):
                fn = legs[leg]
                post = lambda out: out  # noqa: E731
                if isinstance(fn, tuple):
                    fn, post = fn
                a = leg_args[leg]
                try:
                    text = _compiled_text(fn, *a) if variant_name != "quack" else ""
                    out = jax.block_until_ready(fn(*a))
                    err = _rel_err(post(out), refs[leg], active) if leg in refs and post is not None else float("nan")
                    seconds = _time(fn, *a)
                except Exception as exc:  # noqa: BLE001 -- an OOM or unsupported mode is a datum
                    msg = str(exc).splitlines()[0][:160] if str(exc) else type(exc).__name__
                    print(f"  {leg:<12} FAILED {type(exc).__name__}: {msg}")
                    results.append(
                        Result(name, leg, float("nan"), float("nan"), float("nan"), f"error:{type(exc).__name__}")
                    )
                    continue
                kernels = _kernel_summary(text) if text else "-"
                r = Result(name, leg, seconds, flops[leg] / seconds / 1e12, err, kernels)
                results.append(r)
                print(f"  {leg:<12} {seconds * 1e3:9.3f} ms  {r.tflops:8.1f} TFLOP/s  err {err:.2e}  [{kernels}]")
        finally:
            if pallas:
                jax.config.update("jax_ragged_dot_use_gpu_pallas_triton_lowering", False)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
