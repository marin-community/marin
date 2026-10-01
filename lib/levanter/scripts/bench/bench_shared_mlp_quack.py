# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Time the hero's shared-expert SwiGLU MLP on XLA/cuBLAS against QuACK GEMMs with fused SwiGLU.

The hero runs two shared experts per layer as ordinary einsums: two ``[T, D] x [D, I]`` GEMMs for
gate and up, an elementwise SwiGLU, and the down GEMM, with the SwiGLU backward as more elementwise
passes. QuACK can instead run gate and up as one GEMM against interleaved weights with SwiGLU in
the epilogue, and the down projection's input gradient with the SwiGLU backward in the epilogue,
as the routed experts already do. Those GEMMs only pay if QuACK matches cuBLAS at these dense
shapes, which is what this script measures, per leg and for the complete MLP.

QuACK's grouped kernels take one group spanning every row here. Each leg's inputs that come from
an earlier leg are the float32 reference's values rounded to bf16, identical for both variants.
Timing follows ``bench_expert_mlp_legs.py``, whose helpers this script imports.

Example::

    python lib/levanter/scripts/bench/bench_shared_mlp_quack.py --tokens 65536 --model-dim 6144 --intermediate 3072
"""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from bench_expert_mlp_legs import Timing, _errors, _peak_jax, _t, _time_jax, _versions

from levanter.grug._moe.common import _deinterleave_gate_up, _interleave_gate_up
from levanter.grug._moe.quack_moe_cute import quack_gated_grouped_gemm, quack_grouped_dswiglu_gemm
from levanter.grug._moe.sonic_cute import (
    _QUACK_GATED_KW,
    _QUACK_GROUPED_KW,
    shared_gate_up,
    shared_swiglu_down,
)

# Per-GPU shared-expert shapes of the GB200 hero: 1024 x 4096 tokens over 64 GPUs, d6144, I3072.
_DEFAULT_TOKENS = 65_536
_DEFAULT_MODEL_DIM = 6144
_DEFAULT_INTERMEDIATE = 3072

# Tile, cluster and CLC settings to try for the two fused GEMMs, beyond the routed path's shipped ones.
_GATED_CONFIGS = {
    "shipped": _QUACK_GATED_KW,
    "c221": dict(tile_mn=(256, 256), cluster_mnk=(2, 2, 1), use_clc_persistence=True),
    "t128": dict(tile_mn=(256, 128), cluster_mnk=(2, 1, 1), use_clc_persistence=True),
}
_DSWIGLU_CONFIGS = {
    "shipped": _QUACK_GROUPED_KW,
    "c211": dict(tile_mn=(256, 256), cluster_mnk=(2, 1, 1), use_clc_persistence=True),
    "t128": dict(tile_mn=(256, 128), cluster_mnk=(2, 1, 1), use_clc_persistence=True),
}


def _xla_mlp(x, w_gate, w_up, w_down):
    """``DenseMLP.__call__``'s math on the flattened tokens."""
    gate = jnp.einsum("td,dm->tm", x, w_gate)
    up = jnp.einsum("td,dm->tm", x, w_up)
    return jnp.einsum("tm,md->td", jax.nn.silu(gate) * up, w_down)


def _quack_mlp(x, w_gate, w_up, w_down):
    """The shipped shared-expert op pair: QuACK gate/up+SwiGLU, then the down op with its dSwiGLU backward."""
    preact, h = shared_gate_up(x, w_gate, w_up)
    return shared_swiglu_down(preact, h, w_down)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, default=_DEFAULT_TOKENS)
    parser.add_argument("--model-dim", type=int, default=_DEFAULT_MODEL_DIM)
    parser.add_argument("--intermediate", type=int, default=_DEFAULT_INTERMEDIATE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    T, D, I = args.tokens, args.model_dim, args.intermediate
    versions = _versions()
    print(json.dumps(versions["packages"]))
    print(f"shape: T={T} D={D} I={I} seed={args.seed}")

    rng = np.random.default_rng(args.seed)
    x = jnp.asarray(rng.standard_normal((T, D), dtype=np.float32), jnp.bfloat16)
    w_gate = jnp.asarray(rng.standard_normal((D, I), dtype=np.float32) / np.sqrt(D), jnp.bfloat16)
    w_up = jnp.asarray(rng.standard_normal((D, I), dtype=np.float32) / np.sqrt(D), jnp.bfloat16)
    w_down = jnp.asarray(rng.standard_normal((I, D), dtype=np.float32) / np.sqrt(I), jnp.bfloat16)
    dy = jnp.asarray(rng.standard_normal((T, D), dtype=np.float32), jnp.bfloat16)
    w13 = jax.block_until_ready(_interleave_gate_up(jnp.concatenate([w_gate, w_up], axis=-1)[None], I))
    cu = jnp.array([0, T], jnp.int32)

    print("building float32 reference")
    xt, dyt, wgt, wut, wdt = (_t(a).float() for a in (x, dy, w_gate, w_up, w_down))
    ref = {"g": xt @ wgt, "u": xt @ wut}
    ref["h"] = F.silu(ref["g"]) * ref["u"]
    ref["y"] = ref["h"] @ wdt
    ref["dh"] = dyt @ wdt.T
    g_c, u_c = ref["g"].bfloat16(), ref["u"].bfloat16()
    sg = torch.sigmoid(g_c.float())
    silu_c = g_c.float() * sg
    ref["dgate_leg"] = ref["dh"] * u_c.float() * (sg + silu_c * (1 - sg))
    ref["dup_leg"] = ref["dh"] * silu_c
    dgate_c, dup_c = ref["dgate_leg"].bfloat16(), ref["dup_leg"].bfloat16()
    ref["dx_leg"] = dgate_c.float() @ wgt.T + dup_c.float() @ wut.T
    ref["dwgate_leg"] = xt.T @ dgate_c.float()
    ref["dwup_leg"] = xt.T @ dup_c.float()
    sg = torch.sigmoid(ref["g"])
    silu = ref["g"] * sg
    dgate = ref["dh"] * ref["u"] * (sg + silu * (1 - sg))
    dup = ref["dh"] * silu
    ref["dx"] = dgate @ wgt.T + dup @ wut.T
    ref["dwgate"] = xt.T @ dgate
    ref["dwup"] = xt.T @ dup
    ref["dwdown"] = ref["h"].T @ dyt
    del sg, silu, dgate, dup, silu_c, xt, dyt
    torch.cuda.synchronize()

    def jx(t: torch.Tensor) -> jax.Array:
        return jnp.from_dlpack(t.contiguous())

    g_cj, u_cj = jx(g_c), jx(u_c)
    gu_il_c = jx(torch.stack([g_c, u_c], dim=-1).reshape(T, 2 * I))
    dgate_cj, dup_cj = jx(dgate_c), jx(dup_c)
    dgu_il_c = jx(torch.stack([dgate_c, dup_c], dim=-1).reshape(T, 2 * I))
    del g_c, u_c, dgate_c, dup_c

    flops = {
        "fwd_gate_up": 4.0 * T * D * I,
        "bwd_dh": 2.0 * T * D * I,
        "bwd_dx": 4.0 * T * D * I,
        "wgrad_w13": 4.0 * T * D * I,
        "mlp_fwd": 6.0 * T * D * I,
        "mlp_fwd_bwd": 18.0 * T * D * I,
    }
    results: list[Timing] = []

    def run(variant, leg, fn, fn_args, errors_fn):
        print(f"  {variant:22s} {leg:12s}", end="", flush=True)
        out = jax.block_until_ready(fn(*fn_args))
        errs = errors_fn(out)
        del out
        peak = _peak_jax(fn, fn_args)
        tm = _time_jax(fn, fn_args, args.warmup, args.reps)
        t = Timing(
            variant,
            leg,
            flops.get(leg, 0.0),
            tm["op_ms"],
            tm["pipelined_ms"],
            tm["kernel_ms"],
            tm["pipelined_kernel_ms"],
            tm["kernels"],
            peak,
            errs,
        )
        s = t.summary()
        print(
            f" op {s['op_ms']['median']:7.3f} pipelined {s['pipelined_ms']['median']:7.3f}"
            f" kernel {s['kernel_ms'].get('median', float('nan')):7.3f} steady {s['pipelined_kernel_ms'] or 0:7.3f} ms"
            f" {s['tflops_pipelined'] or 0:6.0f} TF/s peak {peak / 2**30:5.2f} GiB "
            + " ".join(f"{k}={v[0]:.2e}" for k, v in errs.items())
        )
        results.append(t)

    # ---------------------------------------------------------------- XLA, the hero's DenseMLP math
    print("XLA/cuBLAS (DenseMLP einsums)")
    run(
        "xla",
        "fwd_gate_up",
        jax.jit(lambda x, wg, wu: jax.nn.silu(x @ wg) * (x @ wu)),
        (x, w_gate, w_up),
        lambda h: dict(h=_errors(_t(h), ref["h"])),
    )

    def xla_dh(dy, wd, g, u):
        dh = jnp.einsum("td,md->tm", dy, wd)
        return jax.vjp(lambda g, u: jax.nn.silu(g) * u, g, u)[1](dh)

    run(
        "xla",
        "bwd_dh",
        jax.jit(xla_dh),
        (dy, w_down, g_cj, u_cj),
        lambda o: dict(dgate=_errors(_t(o[0]), ref["dgate_leg"]), dup=_errors(_t(o[1]), ref["dup_leg"])),
    )
    run(
        "xla",
        "bwd_dx",
        jax.jit(lambda dg, du, wg, wu: jnp.einsum("tm,dm->td", dg, wg) + jnp.einsum("tm,dm->td", du, wu)),
        (dgate_cj, dup_cj, w_gate, w_up),
        lambda o: dict(dx=_errors(_t(o), ref["dx_leg"])),
    )
    run(
        "xla",
        "wgrad_w13",
        jax.jit(lambda x, dg, du: (jnp.einsum("td,tm->dm", x, dg), jnp.einsum("td,tm->dm", x, du))),
        (x, dgate_cj, dup_cj),
        lambda o: dict(dwgate=_errors(_t(o[0]), ref["dwgate_leg"]), dwup=_errors(_t(o[1]), ref["dwup_leg"])),
    )

    def full(mlp):
        def fwd_bwd(x, wg, wu, wd, dy):
            y, vjp = jax.vjp(mlp, x, wg, wu, wd)
            return (y, *vjp(dy))

        return fwd_bwd

    def err_full(o):
        y, dx, dwg, dwu, dwd = o
        return dict(
            y=_errors(_t(y), ref["y"]),
            dx=_errors(_t(dx), ref["dx"]),
            dwgate=_errors(_t(dwg), ref["dwgate"]),
            dwup=_errors(_t(dwu), ref["dwup"]),
            dwdown=_errors(_t(dwd), ref["dwdown"]),
        )

    run("xla", "mlp_fwd", jax.jit(_xla_mlp), (x, w_gate, w_up, w_down), lambda y: dict(y=_errors(_t(y), ref["y"])))
    run("xla", "mlp_fwd_bwd", jax.jit(full(_xla_mlp)), (x, w_gate, w_up, w_down, dy), err_full)

    # ---------------------------------------------------------------- QuACK fused SwiGLU GEMMs
    for name, kw in _GATED_CONFIGS.items():

        def err_gu(o):
            gu, h = o
            gu = _t(gu)
            return dict(
                gate=_errors(gu[:, 0::2], ref["g"]), up=_errors(gu[:, 1::2], ref["u"]), h=_errors(_t(h), ref["h"])
            )

        run(
            f"quack_{name}",
            "fwd_gate_up",
            jax.jit(lambda x, w, cu, kw=kw: quack_gated_grouped_gemm(x, w, cu, return_preact=True, **kw)),
            (x, w13, cu),
            err_gu,
        )
    for name, kw in _DSWIGLU_CONFIGS.items():

        def err_dh(o):
            dgu = _t(o[0])
            return dict(dgate=_errors(dgu[:, 0::2], ref["dgate_leg"]), dup=_errors(dgu[:, 1::2], ref["dup_leg"]))

        run(
            f"quack_{name}",
            "bwd_dh",
            jax.jit(lambda dy, w, gu, cu, kw=kw: quack_grouped_dswiglu_gemm(dy, w, gu, cu, **kw)),
            (dy, w_down[None], gu_il_c, cu),
            err_dh,
        )
    run(
        "packed_xla",
        "bwd_dx",
        jax.jit(lambda d, w: jnp.einsum("tn,dn->td", d, w[0])),
        (dgu_il_c, w13),
        lambda o: dict(dx=_errors(_t(o), ref["dx_leg"])),
    )

    def err_w13(o):
        o = _t(o)
        return dict(dwgate=_errors(o[:, :I], ref["dwgate_leg"]), dwup=_errors(o[:, I:], ref["dwup_leg"]))

    run(
        "packed_xla",
        "wgrad_w13",
        jax.jit(lambda x, d: _deinterleave_gate_up(jnp.einsum("td,tn->dn", x, d))),
        (x, dgu_il_c),
        err_w13,
    )
    run(
        "quack_no_preact",
        "fwd_gate_up",
        jax.jit(lambda x, w, cu: quack_gated_grouped_gemm(x, w, cu, **_QUACK_GATED_KW)),
        (x, w13, cu),
        lambda h: dict(h=_errors(_t(h), ref["h"])),
    )
    run("quack", "mlp_fwd", jax.jit(_quack_mlp), (x, w_gate, w_up, w_down), lambda y: dict(y=_errors(_t(y), ref["y"])))
    run("quack", "mlp_fwd_bwd", jax.jit(full(_quack_mlp)), (x, w_gate, w_up, w_down, dy), err_full)
    run("xla_repeat", "mlp_fwd_bwd", jax.jit(full(_xla_mlp)), (x, w_gate, w_up, w_down, dy), err_full)

    record = dict(
        config=dict(tokens=T, model_dim=D, intermediate=I, seed=args.seed, warmup=args.warmup, reps=args.reps),
        versions=versions,
        results=[r.summary() for r in results],
    )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(record, fh, indent=1)
    print("JSON " + json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
