# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""M0 numerics probe: does the all-E4M3 backward (E4M3 gradients) cost much accuracy vs the
hybrid E5M2-gradient recipe? (S5 Grug FP8, logbook GFP8-023.)

The forward is E4M3 either way; only the backward output-grad format differs. Quantization
error is backend-independent, so this runs on CPU via the XLA ragged path and still measures
the real recipe error -- the H100 question is purely speed. We backprop the Grug MoE expert
MLP (two grouped GEMMs + gated activation) and report the relative Frobenius error of each
gradient (dx, dw13, dw2) vs a bf16 reference, for grad_dtype in {E5M2, E4M3}, across a few
injected output-cotangent distributions (the "upstream gradient" whose dynamic range is what
stresses E4M3's narrower range).

CAVEAT: the cotangents are synthetic. Real training gradients can be heavier-tailed, so a
small E4M3-vs-E5M2 gap here is a lower bound on the real-training gap -- a large gap is the
decisive signal, a small one is suggestive-not-conclusive.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from haliax.nn.ragged_dot import ragged_dot
from haliax.quantization import Fp8RaggedDotOp

_GRAD_DTYPES = [("e5m2", jnp.float8_e5m2), ("e4m3", jnp.float8_e4m3fn)]


def _expert_mlp(x, w13, w2, group_sizes, dot13, dot2):
    """Grug MoE expert MLP: gated up-projection, SiLU, down-projection, all grouped."""
    h = dot13(x, w13, group_sizes)
    gate, up = jnp.split(h, 2, axis=-1)
    return dot2(jax.nn.silu(gate) * up, w2, group_sizes)


def _make_inputs(tokens, hidden, intermediate, experts, dtype, seed=0):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((tokens, hidden)), dtype)
    w13 = jnp.asarray(rng.standard_normal((experts, hidden, 2 * intermediate)) * 0.08, dtype)
    w2 = jnp.asarray(rng.standard_normal((experts, intermediate, hidden)) * 0.08, dtype)
    counts = rng.multinomial(tokens, np.ones(experts) / experts)
    group_sizes = jnp.asarray(counts, jnp.int32)
    return x, w13, w2, group_sizes


def _cotangents(shape, kind, seed=1):
    """Injected output gradient. 'gaussian' is benign; 'heavytail' mimics the wide-dynamic-range
    gradients that stress E4M3 (a few large entries among many small via a lognormal scale)."""
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        return jnp.asarray(rng.standard_normal(shape), jnp.float32)
    if kind == "moderate":
        base = rng.standard_normal(shape)
        scale = np.exp(rng.standard_normal(shape) * 1.0)  # lognormal, ~1 decade of spread
        return jnp.asarray(base * scale, jnp.float32)
    if kind == "heavytail":
        base = rng.standard_normal(shape)
        scale = np.exp(rng.standard_normal(shape) * 3.0)  # lognormal, ~3 decades (extreme stress)
        return jnp.asarray(base * scale, jnp.float32)
    raise ValueError(kind)


def _rel_frob(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _grads(dot13, dot2, x, w13, w2, group_sizes, cotangent):
    def loss(x, w13, w2):
        out = _expert_mlp(x, w13, w2, group_sizes, dot13, dot2)
        return jnp.sum(out.astype(jnp.float32) * cotangent)

    return jax.grad(loss, argnums=(0, 1, 2))(x, w13, w2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--intermediate", type=int, default=128)
    ap.add_argument("--experts", type=int, default=8)
    args = ap.parse_args()

    dtype = jnp.bfloat16
    x, w13, w2, gs = _make_inputs(args.tokens, args.hidden, args.intermediate, args.experts, dtype)
    print(
        f"backend={jax.default_backend()}  shape: T={args.tokens} D={args.hidden} F={args.intermediate} E={args.experts}"
    )

    # bf16 reference (no fp8): plain ragged_dot on both projections.
    bf16_dot = lambda a, b, g: ragged_dot(a, b, g, implementation="xla")  # noqa: E731

    for ck in ("gaussian", "moderate", "heavytail"):
        cot = _cotangents((args.tokens, args.hidden), ck)
        ref = _grads(bf16_dot, bf16_dot, x, w13, w2, gs, cot)
        print(f"\n=== output-cotangent: {ck} ===")
        print(f"  {'grad_dtype':10}  {'dx relerr':>12}  {'dw13 relerr':>12}  {'dw2 relerr':>12}")
        per_dtype = {}
        for name, gdt in _GRAD_DTYPES:
            op13 = Fp8RaggedDotOp.init(compute_dtype=dtype, implementation="xla", grad_dtype=gdt)
            op2 = Fp8RaggedDotOp.init(compute_dtype=dtype, implementation="xla", grad_dtype=gdt)
            g = _grads(op13, op2, x, w13, w2, gs, cot)
            per_dtype[name] = g
            errs = [_rel_frob(g[i], ref[i]) for i in range(3)]
            print(f"  {name:10}  {errs[0]:12.4e}  {errs[1]:12.4e}  {errs[2]:12.4e}")
        # Direct E4M3-vs-E5M2 gradient divergence (isolates the backward-format effect).
        d = [_rel_frob(per_dtype["e4m3"][i], per_dtype["e5m2"][i]) for i in range(3)]
        print(f"  {'e4m3 vs e5m2':10}  {d[0]:12.4e}  {d[1]:12.4e}  {d[2]:12.4e}")


if __name__ == "__main__":
    main()
