# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Time the gated SwiGLU expert MLP in MXFP8 against bf16 on QuACK's SM100 grouped GEMMs.

Both sides run the same pipeline as the ``ragged_all_to_all`` backend: the bf16 path is
``_expert_mlp_quack_wgrad_fwd``/``_backward``, the MXFP8 path ``swiglu_expert_mlp_mxfp8_fwd``/
``_backward_input`` plus its bf16 weight gradients. Weight interleaving and quantization are timed
with them, as the layer pays them every step. Each case reports the forward, the input-gradient
half of the backward, and the full forward + backward (weight gradients included).

The defaults are one EP64 chunk of the d6144 LatentMoE hero (latent 3072, expert width 3072,
three local experts per chunk, 262144 routed rows in a 301466-row receiver buffer)::

    python lib/levanter/scripts/bench/bench_mxfp8_expert_mlp.py
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug._moe.common import _deinterleave_gate_up, _interleave_gate_up
from levanter.grug._moe.quack_mxfp8 import swiglu_expert_mlp_mxfp8_backward_input, swiglu_expert_mlp_mxfp8_fwd
from levanter.grug._moe.sonic_cute import (
    _expert_mlp_quack_wgrad_backward,
    _expert_mlp_quack_wgrad_fwd,
    _swiglu_expert_mlp_quack_wgrads,
)


def _bf16_fwd(x, w13, w2, cu):
    return _expert_mlp_quack_wgrad_fwd(x, _interleave_gate_up(w13, w2.shape[1]), w2, cu)


def _bf16_input_grad(x, w13, w2, cu, dy):
    y, res = _bf16_fwd(x, w13, w2, cu)
    return y, _expert_mlp_quack_wgrad_backward(res, dy)[0]


def _bf16_train(x, w13, w2, cu, dy):
    y, res = _bf16_fwd(x, w13, w2, cu)
    dx, dw13_il, dw2, _ = _expert_mlp_quack_wgrad_backward(res, dy)
    return y, dx, _deinterleave_gate_up(dw13_il), dw2


def _mx_fwd(x, w13, w2, cu):
    return swiglu_expert_mlp_mxfp8_fwd(x, _interleave_gate_up(w13, w2.shape[1]), w2, cu)


def _mx_input_grad(x, w13, w2, cu, dy):
    y, res = _mx_fwd(x, w13, w2, cu)
    return y, swiglu_expert_mlp_mxfp8_backward_input(res, dy)[0]


def _mx_train(x, w13, w2, cu, dy):
    y, res = _mx_fwd(x, w13, w2, cu)
    dx, _, wgrad_args = swiglu_expert_mlp_mxfp8_backward_input(res, dy)
    dw13_il, dw2 = _swiglu_expert_mlp_quack_wgrads(*wgrad_args)
    return y, dx, _deinterleave_gate_up(dw13_il), dw2


def _time(fn, args, iters: int) -> float:
    jitted = jax.jit(fn)
    jax.block_until_ready(jitted(*args))
    jax.block_until_ready(jitted(*args))
    start = time.perf_counter()
    for _ in range(iters):
        out = jitted(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / iters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=262_144, help="routed rows (active)")
    parser.add_argument("--capacity", type=int, default=301_466, help="receiver buffer rows")
    parser.add_argument("--experts", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=3_072, help="expert input width")
    parser.add_argument("--intermediate", type=int, default=3_072, help="expert width")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    h, i, e = args.hidden, args.intermediate, args.experts
    x = jnp.asarray(rng.normal(0, 1, (args.capacity, h)), jnp.bfloat16)
    dy = jnp.asarray(rng.normal(0, 1, (args.capacity, h)), jnp.bfloat16)
    w13 = jnp.asarray(rng.normal(0, h**-0.5, (e, h, 2 * i)), jnp.bfloat16)
    w2 = jnp.asarray(rng.normal(0, i**-0.5, (e, i, h)), jnp.bfloat16)
    sizes = np.full(e, args.rows // e)
    sizes[-1] += args.rows - sizes.sum()
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(sizes)]), jnp.int32)

    # Activation-path GEMM FLOPs: gate/up (2I wide) and down, forward; dh and dx, backward. Wgrads add the same again.
    fwd_flops = 2 * args.rows * h * 2 * i + 2 * args.rows * i * h
    results = {"shape": vars(args)}
    for name, fn, extra, flops in [
        ("forward", (_bf16_fwd, _mx_fwd), (), fwd_flops),
        ("forward+input_grad", (_bf16_input_grad, _mx_input_grad), (dy,), 2 * fwd_flops),
        ("forward+backward", (_bf16_train, _mx_train), (dy,), 3 * fwd_flops),
    ]:
        bf16_t = _time(fn[0], (x, w13, w2, cu, *extra), args.iters)
        mx_t = _time(fn[1], (x, w13, w2, cu, *extra), args.iters)
        results[name] = {
            "bf16_ms": bf16_t * 1e3,
            "mxfp8_ms": mx_t * 1e3,
            "speedup": bf16_t / mx_t,
            "bf16_tflops": flops / bf16_t / 1e12,
            "mxfp8_tflops": flops / mx_t / 1e12,
        }
        print(f"{name}: {json.dumps(results[name])}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
