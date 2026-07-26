# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check whether XLA lowers the expert GEMM to a cuBLASLt fp8 custom call on this stack.

The fp8-wire idea needs the expert GEMM to consume fp8 activations without a separate dequant
pass. XLA's GEMM rewriter folds fp8 operands into ``__cublas$lt$matmul$f8`` only under conditions
that interact badly with per-token scaling: operand scales must be *scalars*, so a per-token scale
applied to a GEMM input blocks the rewrite, while an operand carrying no scale at all is accepted.

This probe compiles the operating point's expert-GEMM shapes four ways and reports, for each,
whether the fp8 custom call appears in the optimized HLO:

* ``bf16``            -- baseline, for the kernel name it lowers to.
* ``fp8_unscaled``    -- fp8 x fp8, no scales anywhere.
* ``fp8_out_scaled``  -- fp8 x fp8 with the per-token scale applied to the GEMM OUTPUT (the
  proposed design: row-linearity moves dequant past the dot, where it fuses into the SwiGLU).
* ``fp8_in_scaled``   -- fp8 x fp8 with the per-token scale applied to the input (the shape the
  earlier fp8-wire attempt had); expected to fall back.

Usage: python -m experiments.grug.moe.fp8_gemm_probe
"""

import jax
import jax.numpy as jnp

# Per local expert at the operating point: capacity buckets from every sender, model width, and
# the fused gate+up projection width.
TOKENS = 64 * 2048
HIDDEN = 5120
FFN = 2 * 1280
FP8_CALL = "__cublas$lt$matmul$f8"


def bf16(x, w, scale):
    return x.astype(jnp.bfloat16) @ w.astype(jnp.bfloat16)


def fp8_unscaled(x, w, scale):
    return x.astype(jnp.float8_e4m3fn).astype(jnp.bfloat16) @ w.astype(jnp.float8_e4m3fn).astype(jnp.bfloat16)


def fp8_out_scaled(x, w, scale):
    out = x.astype(jnp.float8_e4m3fn).astype(jnp.bfloat16) @ w.astype(jnp.float8_e4m3fn).astype(jnp.bfloat16)
    return out * scale[:, None].astype(jnp.bfloat16)


def fp8_in_scaled(x, w, scale):
    lhs = x.astype(jnp.float8_e4m3fn).astype(jnp.bfloat16) * scale[:, None].astype(jnp.bfloat16)
    return lhs @ w.astype(jnp.float8_e4m3fn).astype(jnp.bfloat16)


VARIANTS = (bf16, fp8_unscaled, fp8_out_scaled, fp8_in_scaled)


def main() -> None:
    print(f"jax {jax.__version__} devices: {jax.devices()}")
    x = jnp.ones((TOKENS, HIDDEN), jnp.bfloat16)
    w = jnp.ones((HIDDEN, FFN), jnp.bfloat16)
    scale = jnp.ones((TOKENS,), jnp.float32)
    for variant in VARIANTS:
        text = jax.jit(variant).lower(x, w, scale).compile().as_text()
        calls = text.count(FP8_CALL)
        gemms = text.count("__cublas$lt$matmul")
        kernels = sorted({line.split('"')[1] for line in text.splitlines() if 'custom_call_target="' in line})
        print(f"{variant.__name__:<16} fp8_calls={calls:<3} total_cublaslt={gemms:<3} targets={kernels}")


if __name__ == "__main__":
    main()
