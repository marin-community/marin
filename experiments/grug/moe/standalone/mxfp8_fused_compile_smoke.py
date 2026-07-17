# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MXFP8-004a: CPU-only compile smoke for the fused cudnn-frontend kernels.

Traces + compiles all four ``mxfp8_fused`` launchers through the exact
FunctionSpec + get_or_compile_kernel path the FFI would run, at small shapes,
with CUTE_DSL_ARCH=sm_100a. Runs on a GPU-less x86 box (catches trace/layout
errors in seconds); it does NOT exercise the pod's libNVVM/SASS stage.

Usage: python mxfp8_fused_compile_smoke.py [swiglu|gemm|dswiglu|wgrad ...]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import jax
import jax.numpy as jnp
import numpy as np
from cutlass.jax import compile as cjc
from mxfp8_fused.adapter import (
    dswiglu_plan,
    gemm_plan,
    sf_col_bytes,
    sf_row_bytes,
    swiglu_plan,
    wgrad_plan,
)

E, M, D, F = 4, 1024, 256, 128
N2 = 2 * F
rng = np.random.default_rng(0)


def u8(shape):
    return jnp.asarray(rng.integers(0, 100, shape, dtype=np.uint8))


def e4m3(shape):
    return u8(shape).view(jnp.float8_e4m3fn)


def e8m0(shape):
    return jnp.asarray(rng.integers(120, 130, shape, dtype=np.uint8)).view(jnp.float8_e8m0fnu)


def compile_plan(name: str, plan, ins):
    ins_flat, in_tree = jax.tree.flatten(ins)
    outs_flat, out_tree = jax.tree.flatten(plan.out_shapes)
    spec = cjc.build_function_spec(
        ins_flat, in_tree, outs_flat, out_tree, plan.input_spec, plan.output_spec, {}, None, True, {}
    )
    result = cjc.get_or_compile_kernel(plan.launcher, spec)
    print(f"{name}: OK, module bytes {len(result.module)}", flush=True)


def main():
    which = set(sys.argv[1:]) or {"swiglu", "gemm", "dswiglu", "wgrad"}
    offs = jnp.cumsum(jnp.full((E,), M // E, jnp.int32)).astype(jnp.int32)
    ones_e = jnp.ones((E,), jnp.float32)
    ones_m = jnp.ones((M,), jnp.float32)
    one = jnp.ones((1,), jnp.float32)

    if "swiglu" in which:
        plan = swiglu_plan(M, D, N2, E)
        ins = (
            e4m3((M, D)),
            e4m3((E, N2, D)),
            e8m0((sf_row_bytes(M, D),)),
            e8m0((E * sf_row_bytes(N2, D),)),
            offs,
            ones_e,
            ones_m,
            one,
        )
        compile_plan("swiglu", plan, ins)

    if "gemm" in which:
        plan = gemm_plan(M, F, D, E)
        ins = (
            e4m3((M, F)),
            e4m3((E, D, F)),
            e8m0((sf_row_bytes(M, F),)),
            e8m0((E * sf_row_bytes(D, F),)),
            offs,
            ones_e,
            ones_m,
        )
        compile_plan("gemm", plan, ins)

    if "dswiglu" in which:
        plan = dswiglu_plan(M, D, F, E)
        ins = (
            e4m3((M, D)),
            e4m3((E, F, D)),
            u8((M, N2)).astype(jnp.bfloat16),
            e8m0((sf_row_bytes(M, D),)),
            e8m0((E * sf_row_bytes(F, D),)),
            offs,
            ones_e,
            ones_e,
            ones_m,
            one,
        )
        compile_plan("dswiglu", plan, ins)

    if "wgrad" in which:
        sf_cols = sf_col_bytes(M, D) // ((D + 127) // 128 * 128)
        plan = wgrad_plan(M, D, N2, E, sf_cols)
        ins = (
            e4m3((M, D)),
            e4m3((M, N2)),
            e8m0((D, sf_cols)),
            e8m0((N2, sf_cols)),
            offs,
        )
        compile_plan("wgrad", plan, ins)


if __name__ == "__main__":
    main()
