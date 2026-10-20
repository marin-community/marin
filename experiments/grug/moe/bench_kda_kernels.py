# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GB200 microbench of the fused KDA kernels at the hero per-GPU shape, kernel by kernel.

Shape: B=16 x L=4096 tokens, 48 heads x 128 (the gb200-d6144 rung's per-GPU KDA layer), 32-token chunks in
one decay tile, the on-chip -5 sigmoid gate, packed documents, bf16 activations. Times each Pallas call
(current kernels and the variants) and the two whole passes the training step runs per KDA layer: the
primal forward (scan forward) and the recompute + backward (scan backward's ``jax.vjp``).

    python -m experiments.grug.moe.bench_kda_kernels [--parity] [--sweep]
"""

import argparse
import functools
import statistics
import time

import jax
import jax.experimental.pallas
import jax.experimental.pallas.triton
import jax.numpy as jnp
import numpy as np

from experiments.grug.moe import kda_prep_pallas as prep_mod
from experiments.grug.moe import kda_state_pallas as state_mod
from experiments.grug.moe.kda import doc_starts, kda_fused
from experiments.grug.moe.kda_prep_pallas import ChunkPrep, GateFn, GateKind, PrepConfig
from experiments.grug.moe.kda_state_pallas import StateConfig, StateGrads, StateLayout

B, L, H, D = 16, 4096, 48, 128
C = 32
ITERS = 20


def _inputs(seed: int = 0):
    rng = np.random.RandomState(seed)
    bf = jnp.bfloat16
    q, k, v, a = (jnp.asarray(rng.randn(B, L, H, D), bf) for _ in range(4))
    beta = jnp.asarray(rng.rand(B, L, H), jnp.float32)
    rate = jnp.asarray(rng.uniform(0.5, 2.0, (H, D)), jnp.float32)
    bias = jnp.asarray(rng.randn(H, D), jnp.float32)
    lengths = rng.geometric(1 / 900, size=64)
    seg = np.repeat(np.arange(64), lengths)[:L]
    seg = np.pad(seg, (0, L - len(seg)), mode="edge")
    segment_ids = jnp.asarray(np.broadcast_to(seg, (B, L)), jnp.int32)
    return q, k, v, a, beta, (rate, bias), segment_ids


def _time(fn, *args) -> float:
    out = fn(*args)
    jax.block_until_ready(out)
    out = fn(*args)
    jax.block_until_ready(out)
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return 1e3 * statistics.median(times)


def _prep_cfg(**kw) -> PrepConfig:
    base = dict(
        chunk_size=C,
        gate_kind=GateKind.SIGMOID.value,
        gate_floor=5.0,
        erase_input="gate",
        sub_chunk=C,
        mm_dtype="bfloat16",
        l2norm=True,
        num_warps=4,
        bwd_num_warps=4,
        tail_block_d=32,
        tail_num_warps=4,
        interpret=False,
    )
    base.update(kw)
    return PrepConfig(**base)


def _state_cfg(**kw) -> StateConfig:
    base = dict(
        num_heads=H,
        block_v=64,
        num_warps=4,
        num_stages=2,
        interpret=False,
        save_states=False,
        grads=StateGrads.PALLAS,
        primal_block_v=0,
        primal_num_warps=4,
        layout=StateLayout.DK_ROWS,
        grads_block_v=32,
        grads_num_warps=4,
    )
    base.update(kw)
    return StateConfig(**base)


def _report(name: str, fn, *args):
    try:
        ms = _time(fn, *args)
        print(f"BENCH {name:58s} {ms:8.3f} ms", flush=True)
        return ms
    except Exception as e:  # a candidate config can fail to compile (registers, shared memory)
        print(f"BENCH {name:58s} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
        return None


def _ds_dvp(qi, attn, kw, kcd, decay, h, vnew, dout, dst, cfg: StateConfig):
    """The state backward kernel's per-chunk dS and dV_pseudo (the grads kernel's inputs)."""
    gb, n, c, dk = kw.shape
    dv = dout.shape[-1]
    transposed = cfg.layout == StateLayout.DV_ROWS
    bv = cfg.block_v
    return jax.experimental.pallas.pallas_call(
        state_mod._state_bwd_t_kernel if transposed else state_mod._state_bwd_kernel,
        grid=(gb, dv // bv),
        in_specs=[
            state_mod._full(n, c, dk),
            state_mod._full(n, c, c),
            state_mod._full(n, c, dk),
            state_mod._full(n, c, dk),
            state_mod._full(n, dk),
            state_mod._seq_slab(n * c, H, bv),
            state_mod._state_slab(dk, bv),
        ],
        out_specs=[
            state_mod._tslab(n, bv, dk) if transposed else state_mod._vslab(n, dk, bv),
            state_mod._vslab(n, c, bv),
            state_mod._state_slab(dk, bv),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((gb, n, dv, dk) if transposed else (gb, n, dk, dv), jnp.float32),
            jax.ShapeDtypeStruct((gb, n, c, dv), jnp.float32),
            jax.ShapeDtypeStruct((gb, dk, dv), jnp.float32),
        ],
        compiler_params=jax.experimental.pallas.triton.CompilerParams(num_warps=4, num_stages=2),
    )(qi, attn, kw, kcd, decay, dout, dst)[:2]


def _kernel_benches(sweep: bool):
    q, k, v, a, beta, gate, segment_ids = _inputs()
    starts = doc_starts(segment_ids)
    pcfg = _prep_cfg()
    gseg = jax.jit(lambda a, p, b: prep_mod._seg_decay_fwd_call(a, (p, b), starts, pcfg))(a, *gate)

    def prep_fwd(cfg, save_t):
        return jax.jit(
            lambda q, k, v, g, beta: prep_mod._prep_fwd_call(q, k, v, g, beta, None, starts, None, cfg, save_t)
        )

    _report("seg_decay_fwd", jax.jit(lambda a, p, b: prep_mod._seg_decay_fwd_call(a, (p, b), starts, pcfg)), a, *gate)
    _report("prep_fwd (primal, w4)", prep_fwd(pcfg, False), q, k, v, gseg, beta)
    _report("prep_fwd (save_t, w4)", prep_fwd(pcfg, True), q, k, v, gseg, beta)

    *outs, t = prep_fwd(pcfg, True)(q, k, v, gseg, beta)
    prep = ChunkPrep(*outs)
    s0 = jnp.zeros((B * H, D, D), jnp.float32)
    sargs = (prep.q_inflate, prep.attn, prep.kw, prep.k_cumdecay, prep.v_pseudo, prep.decay, s0)

    def state_fwd(cfg):
        return jax.jit(lambda *x: state_mod._state_fwd_call(*x, jnp.bfloat16, cfg))

    def state_lean(cfg):
        return jax.jit(lambda *x: state_mod._state_fwd_lean_call(*x, jnp.bfloat16, cfg))

    dk_rows, dv_rows = StateLayout.DK_ROWS, StateLayout.DV_ROWS
    _report("state_fwd (saving, dk_rows bv64 w4 s2)", state_fwd(_state_cfg()), *sargs)
    _report("state_fwd_lean (dk_rows bv64 w4 s2)", state_lean(_state_cfg(primal_block_v=64)), *sargs)
    fwd_grid = [(64, 4, 2), (64, 4, 3), (32, 4, 2), (128, 4, 2), (128, 8, 2), (64, 8, 2)] if sweep else [(64, 4, 2)]
    for bv, w, st in fwd_grid:
        cfg = _state_cfg(layout=dv_rows, block_v=bv, num_warps=w, num_stages=st)
        _report(f"state_fwd (saving, dv_rows bv{bv} w{w} s{st})", state_fwd(cfg), *sargs)
        cfg = cfg._replace(primal_block_v=bv, primal_num_warps=w)
        _report(f"state_fwd_lean (dv_rows bv{bv} w{w} s{st})", state_lean(cfg), *sargs)

    rng = np.random.RandomState(1)
    for layout in (dk_rows, dv_rows):
        out, h, vnew, _ = state_fwd(_state_cfg(layout=layout))(*sargs)
        dout = jnp.asarray(rng.randn(*out.shape), jnp.bfloat16)
        dst = jnp.zeros_like(s0)
        bargs = (prep.q_inflate, prep.attn, prep.kw, prep.k_cumdecay, prep.decay, h, vnew, dout, dst)

        def state_bwd(cfg):
            return jax.jit(lambda *x: state_mod._state_bwd_call(*x, cfg))

        def state_grads(cfg):
            return jax.jit(lambda *x: state_mod._state_grads_call(*x, cfg))

        bwd_cfg = _state_cfg(layout=layout, grads=StateGrads.XLA if layout == dk_rows else StateGrads.PALLAS)
        ds_dvp = jax.jit(functools.partial(_ds_dvp, cfg=bwd_cfg))(*bargs)
        gargs = (prep.q_inflate, prep.attn, prep.kw, prep.k_cumdecay, h, vnew, dout, *ds_dvp)

        bwd_grid = [(64, 4, 2), (64, 4, 3), (128, 4, 2), (128, 8, 2), (32, 4, 2)] if sweep else [(64, 4, 2)]
        for bv, w, st in bwd_grid:
            cfg = _state_cfg(layout=layout, block_v=bv, num_warps=w, num_stages=st)
            _report(f"state_bwd + grads ({layout} bv{bv} w{w} s{st}; grads b32 w4)", state_bwd(cfg), *bargs)
        grads_grid = [(32, 4), (64, 4), (64, 8), (128, 8), (16, 4)] if sweep else [(32, 4)]
        for gbv, gw in grads_grid:
            cfg = _state_cfg(layout=layout, grads_block_v=gbv, grads_num_warps=gw)
            _report(f"state_grads alone ({layout} b{gbv} w{gw})", state_grads(cfg), *gargs)

    cts = ChunkPrep(*(jnp.asarray(rng.randn(*x.shape) * 0.1, x.dtype) for x in prep))

    def prep_bwd(cfg):
        return jax.jit(
            lambda q, k, v, a, beta, p, b, t, cts: prep_mod._prep_bwd_call(
                q, k, v, a, beta, (p, b), starts, None, t, cts, cfg
            )
        )

    _report("prep_bwd (wy + tail + seg, w4/w4 tail b32)", prep_bwd(pcfg), q, k, v, a, beta, *gate, t, cts)
    if sweep:
        for tb, tw in ((64, 4), (128, 4), (64, 8), (16, 4)):
            cfg = _prep_cfg(tail_block_d=tb, tail_num_warps=tw)
            _report(f"prep_bwd (wy w4 + tail b{tb} w{tw} + seg)", prep_bwd(cfg), q, k, v, a, beta, *gate, t, cts)


def _layer_benches():
    """The two KDA passes of one training step per layer: primal forward, and recompute + backward."""
    q, k, v, a, beta, gate, segment_ids = _inputs()
    rng = np.random.RandomState(2)
    dout = jnp.asarray(rng.randn(B, L, H, D), jnp.bfloat16)
    base = dict(
        gate=gate,
        gate_fn=GateFn(GateKind.SIGMOID, 5.0),
        segment_ids=segment_ids,
        chunk_size=C,
        sub_chunk_size=C,
        state_grads=StateGrads.PALLAS,
        state_block_v=64,
    )
    variants = {
        "current": {},
        "lean primal dk_rows bv64": dict(state_primal_block_v=64),
        "dv_rows": dict(state_layout=StateLayout.DV_ROWS),
        "dv_rows + lean primal bv64": dict(state_layout=StateLayout.DV_ROWS, state_primal_block_v=64),
    }
    for name, extra in variants.items():
        kw = dict(base, **extra)

        def fwd(q, k, v, a, beta, kw=kw):
            return kda_fused(q, k, v, a, beta, **kw)

        def fwd_bwd(q, k, v, a, beta, dout, kw=kw):
            out, vjp = jax.vjp(lambda *x: kda_fused(*x, **kw), q, k, v, a, beta)
            return out, vjp(dout)

        _report(f"LAYER {name}: primal forward", jax.jit(fwd), q, k, v, a, beta)
        _report(f"LAYER {name}: recompute + backward", jax.jit(fwd_bwd), q, k, v, a, beta, dout)


def _parity():
    """Variants vs the current kernels at the hero shape on GPU (bf16 GEMM operands): output and every
    input cotangent, as max |diff| relative to max |reference|."""
    q, k, v, a, beta, gate, segment_ids = _inputs(3)
    rng = np.random.RandomState(4)
    dout = jnp.asarray(rng.randn(B, L, H, D), jnp.bfloat16)
    base = dict(
        gate_fn=GateFn(GateKind.SIGMOID, 5.0),
        segment_ids=segment_ids,
        chunk_size=C,
        sub_chunk_size=C,
        state_grads=StateGrads.PALLAS,
        state_block_v=64,
    )

    args = (q, k, v, a, beta, *gate)

    def run(kw):
        def f(q, k, v, a, beta, p, b):
            return kda_fused(q, k, v, a, beta, gate=(p, b), **kw)

        def f_vjp(dout, *x):
            out, vjp = jax.vjp(f, *x)
            return out, vjp(dout)

        primal = jax.jit(f)(*args)
        out, grads = jax.jit(f_vjp)(dout, *args)
        return primal, out, grads

    ref_primal, ref_out, ref_grads = run(base)
    names = ("dq", "dk", "dv", "da", "dbeta", "drate", "dbias")
    for vname, extra in (
        ("lean primal", dict(state_primal_block_v=64)),
        ("dv_rows", dict(state_layout=StateLayout.DV_ROWS)),
        ("dv_rows + lean", dict(state_layout=StateLayout.DV_ROWS, state_primal_block_v=64)),
    ):
        primal, out, grads = run(dict(base, **extra))

        def rel(x, y):
            x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
            return float(np.max(np.abs(x - y)) / max(np.max(np.abs(y)), 1e-30))

        parts = [f"primal {rel(primal, ref_primal):.2e}", f"out {rel(out, ref_out):.2e}"]
        parts += [f"{n} {rel(x, y):.2e}" for n, x, y in zip(names, grads, ref_grads, strict=True)]
        finite = all(np.isfinite(np.asarray(x, np.float32)).all() for x in (primal, out, *grads))
        print(f"PARITY {vname:18s} finite={finite} " + " ".join(parts), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--skip-kernels", action="store_true")
    parser.add_argument("--skip-layer", action="store_true")
    args = parser.parse_args()
    print("devices", jax.devices(), flush=True)
    if args.parity:
        _parity()
    if not args.skip_kernels:
        _kernel_benches(args.sweep)
    if not args.skip_layer:
        _layer_benches()


if __name__ == "__main__":
    main()
