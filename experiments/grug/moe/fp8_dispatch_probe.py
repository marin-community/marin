# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate 2: does the REAL dispatch path still lower to a cuBLASLt fp8 GEMM?

`fp8_gemm_probe` showed that a dot over genuine fp8 operands lowers to
``__cublas$lt$matmul$f8`` when the per-token scale is applied to the GEMM output. That probe fed
the operands as clean jit parameters. The real dispatch path does not: the payload is quantized,
**bitcast to uint8**, carried across a tiled ``all_to_all`` inside ``shard_map``, bitcast back to
fp8 and reshaped before it reaches the dot. Any of those steps could break the rewriter's operand
pattern and silently fall back to bf16, which is the failure this gate exists to catch.

This probe reproduces that graph — quantize, bitcast, a2a, bitcast, reshape, dot, output-side
scale, SwiGLU — at operating-point shapes on a 4-GPU expert mesh, and reports whether the fp8
custom call survives. It touches no production code.

Usage (single node, 4 GPUs): python -m experiments.grug.moe.fp8_dispatch_probe
"""

import jax
import jax.numpy as jnp
import numpy as np

EXPERT_SHARDS = 4
CAPACITY = 2048
HIDDEN = 5120
FFN = 1280
E4M3_MAX = 448.0
FP8_CALL = "__cublas$lt$matmul$f8"


def quantize_rows(x):
    """Per-token scale over the hidden dim, as the wire quantizer does."""
    xf = x.astype(jnp.float32)
    amax = jnp.max(jnp.abs(xf), axis=-1)
    scale = jnp.maximum(amax, 1e-12) / E4M3_MAX
    return (xf / scale[..., None]).astype(jnp.float8_e4m3fn), scale


def quantize_weight(w):
    """Per-expert scalar scale: what the rewriter requires of an operand scale."""
    scale = jnp.maximum(jnp.max(jnp.abs(w.astype(jnp.float32))), 1e-12) / E4M3_MAX
    return (w.astype(jnp.float32) / scale).astype(jnp.float8_e4m3fn), scale


def dispatch_then_gemm(x, w13, fp8_gemm: bool):
    """Quantize -> bitcast -> a2a -> bitcast -> reshape -> dot, then scale and SwiGLU."""
    q, scale = quantize_rows(x)
    bits = jax.lax.all_to_all(
        jax.lax.bitcast_convert_type(q, jnp.uint8), "expert", split_axis=0, concat_axis=0, tiled=True
    )
    scales = jax.lax.all_to_all(scale[..., None], "expert", split_axis=0, concat_axis=0, tiled=True)
    received = jax.lax.bitcast_convert_type(bits, jnp.float8_e4m3fn)
    row_scale = scales.reshape(-1)

    if fp8_gemm:
        # Unscaled fp8 into the dot; the per-token scale rides the output.
        wq, w_scale = quantize_weight(w13)
        expert_input = received.reshape(-1, HIDDEN)
        hidden = expert_input.astype(jnp.bfloat16) @ wq.astype(jnp.bfloat16)
        hidden = hidden * (row_scale * w_scale)[:, None].astype(jnp.bfloat16)
    else:
        # The arrangement shipped today: dequantize on arrival, then a bf16 dot.
        expert_input = (received.astype(jnp.float32) * row_scale[:, None]).astype(jnp.bfloat16)
        hidden = expert_input.reshape(-1, HIDDEN) @ w13.astype(jnp.bfloat16)

    gate, up = jnp.split(hidden, [FFN], axis=-1)
    return jax.nn.silu(gate) * up


def main() -> None:
    devices = jax.devices()
    print(f"jax {jax.__version__} devices: {devices}")
    mesh = jax.sharding.Mesh(np.array(devices[:EXPERT_SHARDS]), ("expert",))
    partition = jax.sharding.PartitionSpec("expert")

    x = jnp.ones((EXPERT_SHARDS, EXPERT_SHARDS * CAPACITY, HIDDEN), jnp.bfloat16)
    w13 = jnp.ones((EXPERT_SHARDS, HIDDEN, 2 * FFN), jnp.bfloat16)

    for fp8_gemm in (False, True):
        def sharded(x, w13, fp8_gemm=fp8_gemm):
            return jax.shard_map(
                lambda xb, wb: dispatch_then_gemm(xb[0], wb[0], fp8_gemm)[None],
                mesh=mesh,
                in_specs=(partition, partition),
                out_specs=partition,
            )(x, w13)

        text = jax.jit(sharded).lower(x, w13).compile().as_text()
        label = "fp8_gemm" if fp8_gemm else "dequant_on_arrival"
        targets = sorted({line.split('"')[1] for line in text.splitlines() if 'custom_call_target="' in line})
        print(f"{label:<20} fp8_calls={text.count(FP8_CALL):<3} cublaslt={text.count('__cublas$lt$matmul'):<3}")
        print(f"{'':<20} targets={targets}")


if __name__ == "__main__":
    main()
