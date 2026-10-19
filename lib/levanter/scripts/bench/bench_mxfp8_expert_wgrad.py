# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Time the MXFP8 routed-expert weight gradients against the bf16 ones on QuACK's SM100 grouped GEMMs.

Both sides run the MXFP8 activation path (``*_expert_mlp_mxfp8_fwd``); the bf16 side runs ``_backward_input`` plus
bf16 ``quack_grouped_wgrad``, the MXFP8 side the fused ``*_expert_mlp_mxfp8_backward`` (token-axis quantization of
both operands per expert, block-scaled varlen-k GEMM). Reported: the weight gradients alone from bf16 operands
(``expert_mlp_mxfp8_wgrads``, quantization included), the token-axis quantizers alone, the full forward + backward,
and the two wgrad GEMMs on pre-quantized operands per tile/cluster configuration.

The defaults are one EP64 chunk of the d6144 LatentMoE hero (latent 3072, three local experts, 262144 routed rows in
a 301466-row receiver buffer); SwiGLU experts are 3072 wide and ungated ReLU^2 experts 4608::

    python lib/levanter/scripts/bench/bench_mxfp8_expert_wgrad.py --activation swiglu --intermediate 3072
    python lib/levanter/scripts/bench/bench_mxfp8_expert_wgrad.py --activation relu2 --intermediate 4608
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug._moe.common import _interleave_gate_up
from levanter.grug._moe.quack_mxfp8 import (
    expert_mlp_mxfp8_wgrads,
    mxfp8_token_grouped_gemm,
    quantize_token_blocks,
    relu2_expert_mlp_mxfp8_backward,
    relu2_expert_mlp_mxfp8_backward_input,
    relu2_expert_mlp_mxfp8_fwd,
    swiglu_expert_mlp_mxfp8_backward,
    swiglu_expert_mlp_mxfp8_backward_input,
    swiglu_expert_mlp_mxfp8_fwd,
)
from levanter.grug._moe.sonic_cute import _relu2_expert_mlp_quack_wgrads, _swiglu_expert_mlp_quack_wgrads


def _paths(activation: str):
    if activation == "swiglu":
        return (
            swiglu_expert_mlp_mxfp8_fwd,
            swiglu_expert_mlp_mxfp8_backward_input,
            _swiglu_expert_mlp_quack_wgrads,
            swiglu_expert_mlp_mxfp8_backward,
        )
    return (
        relu2_expert_mlp_mxfp8_fwd,
        relu2_expert_mlp_mxfp8_backward_input,
        _relu2_expert_mlp_quack_wgrads,
        relu2_expert_mlp_mxfp8_backward,
    )


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
    parser.add_argument("--activation", choices=["swiglu", "relu2"], default="swiglu")
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
    gated = args.activation == "swiglu"
    up_width = 2 * i if gated else i
    x = jnp.asarray(rng.normal(0, 1, (args.capacity, h)), jnp.bfloat16)
    dy = jnp.asarray(rng.normal(0, 1, (args.capacity, h)), jnp.bfloat16)
    w_up = jnp.asarray(rng.normal(0, h**-0.5, (e, h, up_width)), jnp.bfloat16)
    w_down = jnp.asarray(rng.normal(0, i**-0.5, (e, i, h)), jnp.bfloat16)
    if gated:
        w_up = _interleave_gate_up(w_up, i)
    sizes = np.full(e, args.rows // e)
    sizes[-1] += args.rows - sizes.sum()
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(sizes)]), jnp.int32)
    fwd, backward_input, bf16_wgrads, mxfp8_backward = _paths(args.activation)

    def wgrad_args(x, w_up, w_down, dy):
        _, residuals = fwd(x, w_up, w_down, cu)
        return backward_input(residuals, dy)[2][:4]

    operands = jax.block_until_ready(jax.jit(wgrad_args)(x, w_up, w_down, dy))

    def bf16_train(x, w_up, w_down, dy):
        y, residuals = fwd(x, w_up, w_down, cu)
        dx, _, grad_args = backward_input(residuals, dy)
        return y, dx, *bf16_wgrads(*grad_args)

    def mxfp8_train(x, w_up, w_down, dy):
        y, residuals = fwd(x, w_up, w_down, cu)
        return y, *mxfp8_backward(residuals, dy)[:3]

    def quantizers(x, d_up, act, dy):
        return [quantize_token_blocks(a, cu) for a in (x, d_up, act, dy)]

    quantized = jax.block_until_ready(jax.jit(quantizers)(*operands))

    def gemms(config):
        tile_mn, cluster_mnk = config

        def run(qx, qd, qa, qdy):
            kw = dict(out_dtype=jnp.bfloat16, tile_mn=tile_mn, cluster_mnk=cluster_mnk)
            return mxfp8_token_grouped_gemm(qx, qd, cu, **kw), mxfp8_token_grouped_gemm(qa, qdy, cu, **kw)

        return run

    wgrad_flops = 2 * args.rows * (h * up_width + i * h)
    results = {"shape": vars(args)}
    cases = [
        ("wgrads", lambda *a: bf16_wgrads(*a, cu), lambda *a: expert_mlp_mxfp8_wgrads(*a, cu), operands, wgrad_flops),
        ("token_quantizers", None, quantizers, operands, None),
        ("forward+backward", bf16_train, mxfp8_train, (x, w_up, w_down, dy), None),
    ]
    for config in [((256, 256), (2, 1, 1)), ((256, 256), (2, 2, 1)), ((256, 128), (2, 1, 1)), ((256, 128), (2, 2, 1))]:
        cases.append((f"gemms tile {config[0]} cluster {config[1]}", None, gemms(config), quantized, wgrad_flops))
    for name, bf16_fn, mx_fn, inputs, flops in cases:
        mx_t = _time(mx_fn, inputs, args.iters)
        result = {"mxfp8_ms": mx_t * 1e3}
        if bf16_fn is not None:
            bf16_t = _time(bf16_fn, inputs, args.iters)
            result |= {"bf16_ms": bf16_t * 1e3, "speedup": bf16_t / mx_t}
        if flops is not None:
            result["mxfp8_tflops"] = flops / mx_t / 1e12
        if flops is not None and bf16_fn is not None:
            result["bf16_tflops"] = flops / bf16_t / 1e12
        results[name] = result
        print(f"{args.activation} {name}: {json.dumps(result)}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
