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
from experiments.grug.moe.kda_state_pallas import StateConfig, StateGrads

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
        chunks=1,
        chunk_stages=1,
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
        grads_block_v=32,
        grads_num_warps=4,
        grads_chunks=1,
        grads_num_stages=1,
        saved_dtype="float32",
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
    for cpb, st in ((2, 2), (4, 2), (4, 3), (8, 2), (8, 3)):
        cfg = _prep_cfg(chunks=cpb, chunk_stages=st)
        _report(f"prep_fwd (save_t, w4, c{cpb} s{st})", prep_fwd(cfg, True), q, k, v, gseg, beta)

    *outs, t = prep_fwd(pcfg, True)(q, k, v, gseg, beta)
    prep = ChunkPrep(*outs)
    s0 = jnp.zeros((B * H, D, D), jnp.float32)
    sargs = (prep.q_inflate, prep.attn, prep.kw, prep.k_cumdecay, prep.v_pseudo, prep.decay, s0)

    def state_fwd(cfg):
        return jax.jit(lambda *x: state_mod._state_fwd_call(*x, jnp.bfloat16, cfg))

    def state_lean(cfg):
        return jax.jit(lambda *x: state_mod._state_fwd_lean_call(*x, jnp.bfloat16, cfg))

    _report("state_fwd (saving, bv64 w4 s2)", state_fwd(_state_cfg()), *sargs)
    _report("state_fwd (saving, bf16 v_new)", state_fwd(_state_cfg(saved_dtype="bfloat16")), *sargs)
    _report("state_fwd_lean (bv64 w4 s2)", state_lean(_state_cfg(primal_block_v=64)), *sargs)
    _report("state_fwd_lean (bv64 w4 s3)", state_lean(_state_cfg(primal_block_v=64, num_stages=3)), *sargs)

    rng = np.random.RandomState(1)
    for saved in ("float32", "bfloat16"):
        out, h, vnew, _ = state_fwd(_state_cfg(saved_dtype=saved))(*sargs)
        dout = jnp.asarray(rng.randn(*out.shape), jnp.bfloat16)
        dst = jnp.zeros_like(s0)
        bargs = (prep.q_inflate, prep.attn, prep.kw, prep.k_cumdecay, prep.decay, h, vnew, dout, dst)

        def state_bwd(cfg):
            return jax.jit(lambda *x: state_mod._state_bwd_call(*x, cfg))

        def state_bwd_only(cfg):
            def f(qi, attn, kw, kcd, decay, h_, vnew_, dout_, dst_):
                return state_mod._state_bwd_call(
                    qi, attn, kw, kcd, decay, h_, vnew_, dout_, dst_, cfg._replace(grads=StateGrads.XLA)
                )

            return jax.jit(f)

        _report(f"state_bwd alone ({saved} dS; + xla grads)", state_bwd_only(_state_cfg(saved_dtype=saved)), *bargs)
        grid = [(1, 1, 32, 4), (2, 2, 32, 4), (4, 2, 32, 4), (8, 2, 32, 4), (4, 3, 32, 4), (8, 3, 32, 4)]
        if sweep:
            grid += [(2, 2, 16, 4), (4, 2, 16, 4), (4, 3, 16, 4), (2, 2, 32, 8), (4, 2, 32, 8), (8, 2, 16, 8)]
        for cpb, st, gbv, gw in grid:
            cfg = _state_cfg(
                saved_dtype=saved, grads_chunks=cpb, grads_num_stages=st, grads_block_v=gbv, grads_num_warps=gw
            )
            _report(f"state_bwd + grads ({saved}; grads c{cpb} s{st} b{gbv} w{gw})", state_bwd(cfg), *bargs)

    cts = ChunkPrep(*(jnp.asarray(rng.randn(*x.shape) * 0.1, x.dtype) for x in prep))

    def prep_bwd(cfg):
        return jax.jit(
            lambda q, k, v, a, beta, p, b, t, cts: prep_mod._prep_bwd_call(
                q, k, v, a, beta, (p, b), starts, None, t, cts, cfg
            )
        )

    _report("prep_bwd (wy + tail + seg, w4/w4 tail b32)", prep_bwd(pcfg), q, k, v, a, beta, *gate, t, cts)
    for cpb, st in ((2, 2), (4, 2), (4, 3), (8, 2), (8, 3)):
        cfg = _prep_cfg(chunks=cpb, chunk_stages=st)
        _report(f"prep_bwd (c{cpb} s{st})", prep_bwd(cfg), q, k, v, a, beta, *gate, t, cts)
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
    lean = dict(state_primal_block_v=64, state_num_stages=3)
    bf16 = dict(state_saved_dtype=jnp.bfloat16, state_grads_chunks=8, state_grads_num_stages=2)
    variants = {
        "current": {},
        "lean s3": lean,
        "lean s3 + xla grads": dict(lean, state_grads=StateGrads.XLA),
        "lean s3 + bf16 saved grads c8 s2": dict(lean, **bf16),
        "lean s3 + bf16 saved + xla grads": dict(lean, state_saved_dtype=jnp.bfloat16, state_grads=StateGrads.XLA),
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
        ("lean s3", dict(state_primal_block_v=64, state_num_stages=3)),
        ("prep c4 s2", dict(prep_chunks=4, prep_chunk_stages=2)),
        ("bf16 saved", dict(state_saved_dtype=jnp.bfloat16)),
    ):
        primal, out, grads = run(dict(base, **extra))

        def rel(x, y):
            x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
            return float(np.max(np.abs(x - y)) / max(np.max(np.abs(y)), 1e-30))

        def rel_l2(x, y):
            x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
            return float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-30))

        parts = [f"primal {rel(primal, ref_primal):.2e}", f"out {rel(out, ref_out):.2e}"]
        parts += [
            f"{n} maxrel {rel(x, y):.2e} relL2 {rel_l2(x, y):.2e}"
            for n, x, y in zip(names, grads, ref_grads, strict=True)
        ]
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
