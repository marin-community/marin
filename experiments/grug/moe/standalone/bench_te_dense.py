# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA Transformer Engine dense MXFP8 control at Grug production shapes.

Installs through ``run_te_bench.sh`` and compares TE's supported
``MXFP8BlockScaling`` JAX Dense path directly with Haliax's delayed per-tensor
``Fp8DotGeneralOp``. Both timed graphs return forward output, input gradient,
and weight gradient for the same explicit BF16 cotangent.
"""

import argparse
import json
import os
import re
import time

import jax
import jax.numpy as jnp

from experiments.grug.moe.standalone.bench_mxfp8_dense import (
    DNUMS,
    custom_call_count,
    dot_fp8_tensor,
    rel_frob,
    timed_samples,
    weighted_production_ratio,
)

SHAPES = [
    ("q_o_shared_5120x5120", 5120, 5120, 5),
    ("kv_5120x1280", 5120, 1280, 2),
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--tokens", type=int, default=65536)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--shape", action="append", choices=[label for label, _, _, _ in SHAPES])
    parser.add_argument("--out", default="bench_te_dense.json")
    return parser.parse_args(argv)


def main():
    args = parse_args()

    import flax.linen as nn  # noqa: PLC0415
    from transformer_engine.common.recipe import MXFP8BlockScaling  # noqa: PLC0415
    from transformer_engine.jax import flax as te_flax  # noqa: PLC0415

    recipe = MXFP8BlockScaling()
    te_dot_general_cls = te_flax.make_dot_general_cls(recipe)

    class DenseBlock(nn.Module):
        features: int

        @nn.compact
        def __call__(self, x):
            return nn.Dense(
                features=self.features,
                use_bias=False,
                dtype=jnp.bfloat16,
                param_dtype=jnp.bfloat16,
                dot_general=te_dot_general_cls(),
            )(x)

    device = jax.devices()[0]
    selected_shapes = [shape for shape in SHAPES if args.shape is None or shape[0] in args.shape]
    results = {
        "device": str(device.device_kind),
        "backend": str(device.client.platform_version),
        "jax": jax.__version__,
        "git_sha": args.git_sha,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "tokens": args.tokens,
        "warmup": args.warmup,
        "iters": args.iters,
        "shapes": {},
    }
    production_measurements = []

    for label, k, n, production_weight in selected_shapes:
        key_init, key_x, key_g = jax.random.split(jax.random.PRNGKey(0), 3)
        x = jax.random.normal(key_x, (args.tokens, k), dtype=jnp.bfloat16)
        g = jax.random.normal(key_g, (args.tokens, n), dtype=jnp.bfloat16)
        model = DenseBlock(features=n)
        variables = model.init(key_init, x)
        w = variables["params"]["Dense_0"]["kernel"]

        def te_fwd(variables, x, model=model):
            return model.apply(variables, x)

        def te_fwdbwd(variables, x, g):
            y, pullback = jax.vjp(te_fwd, variables, x)
            return y, pullback(g)

        def tensor_fwdbwd(x, w, g):
            y, pullback = jax.vjp(dot_fp8_tensor, x, w)
            return y, pullback(g)

        def compile_and_measure(fn, fn_args):
            lowered = jax.jit(fn).lower(*fn_args)
            compile_started = time.perf_counter()
            compiled = lowered.compile()
            compile_ms = (time.perf_counter() - compile_started) * 1e3
            median, mad = timed_samples(compiled, fn_args, args.iters, args.warmup)
            compiled_text = compiled.as_text()
            return compiled, {
                "fwdbwd_ms": median * 1e3,
                "fwdbwd_mad_ms": mad * 1e3,
                "compile_ms": compile_ms,
                "custom_call_targets": sorted(set(re.findall(r'custom_call_target="([^"]+)"', compiled_text))),
                "block_scaled_dot_call_count": custom_call_count(compiled_text, "__cudnn$blockScaledDot"),
            }

        te_compiled, te_result = compile_and_measure(te_fwdbwd, (variables, x, g))
        tensor_compiled, tensor_result = compile_and_measure(tensor_fwdbwd, (x, w, g))
        te_out, (te_variable_grads, te_gx) = te_compiled(variables, x, g)
        tensor_out, (tensor_gx, tensor_gw) = tensor_compiled(x, w, g)
        te_gw = te_variable_grads["params"]["Dense_0"]["kernel"]

        ref_out = jax.lax.dot_general(x.astype(jnp.float32), w.astype(jnp.float32), DNUMS)
        ref_gx = jax.lax.dot_general(g.astype(jnp.float32), w.T.astype(jnp.float32), DNUMS)
        ref_gw = jax.lax.dot_general(x.T.astype(jnp.float32), g.astype(jnp.float32), DNUMS)
        for result, out, gx, gw in (
            (te_result, te_out, te_gx, te_gw),
            (tensor_result, tensor_out, tensor_gx, tensor_gw),
        ):
            result["err_out"] = rel_frob(out, ref_out)
            result["err_gx"] = rel_frob(gx, ref_gx)
            result["err_gw"] = rel_frob(gw, ref_gw)

        results["shapes"][label] = {"te_mxfp8": te_result, "fp8_tensor": tensor_result}
        production_measurements.append((production_weight, te_result["fwdbwd_ms"], tensor_result["fwdbwd_ms"]))
        print(f"\n== {label} (T={args.tokens}) ==")
        for arm, row in results["shapes"][label].items():
            print(
                f"  {arm:14s} {row['fwdbwd_ms']:.3f} +/- {row['fwdbwd_mad_ms']:.3f} ms "
                f"err {row['err_out']:.2e}/{row['err_gx']:.2e}/{row['err_gw']:.2e}"
            )

    covered_weight = sum(weight for weight, _, _ in production_measurements)
    results["covered_production_weight"] = covered_weight
    results["weighted_production_ratio"] = weighted_production_ratio(production_measurements)
    results["complete_production_mix"] = covered_weight == 7
    print(f"\nTE MXFP8 / per-tensor weighted production time: {results['weighted_production_ratio']:.4f}x")
    with open(args.out, "w") as output:
        json.dump(results, output, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
