# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FP8W-003: can a quantized dispatch payload cross the expert-MLP op boundary
as a differentiated argument?

Runs on CPU. See `.agents/logbooks/fp8-dispatch-wire.md` and issue #7665.

The design has the EP backend quantize before the dispatch collective, permute
the payload, and hand it to the op, so the gradient must flow back to the bf16
source through a quantized intermediate. This checks which carrier dtypes
actually propagate a cotangent.

The answer decides the implementation shape, and the failure mode is silent:
`levanter.grug._moe.fp8_wire` deliberately bitcasts its wire payload to uint8
("permutation collectives move bytes"), which is safe there only because that
module dequantizes to bf16 inside its own custom_vjp, so uint8 never crosses an
autodiff boundary.

Usage: uv run python experiments/grug/moe/standalone/study_fp8_wire_ad.py
"""

import jax
import jax.numpy as jnp

PAYLOAD_SHAPE = (4,)


def carrier_gradient(cast_fn, uncast_fn):
    """Gradient reaching a bf16 source through a quantized intermediate.

    Models the design: the op is a custom_vjp that consumes the payload and
    hands back a straight-through bf16 cotangent, exactly as
    ``fp8_all_gather`` does today.
    """

    @jax.custom_vjp
    def consume(payload):
        return uncast_fn(payload).sum()

    def consume_bwd(_res, cotangent):
        return (jnp.full(PAYLOAD_SHAPE, cotangent, jnp.bfloat16),)

    consume.defvjp(lambda p: (consume(p), None), consume_bwd)

    x = jnp.arange(PAYLOAD_SHAPE[0], dtype=jnp.bfloat16) + 1
    return jax.grad(lambda v: consume(cast_fn(v)))(x)


def main():
    carriers = {
        "bfloat16 (control)": (lambda x: x, lambda p: p.astype(jnp.float32)),
        "float8_e4m3fn": (
            lambda x: x.astype(jnp.float8_e4m3fn),
            lambda p: p.astype(jnp.float32),
        ),
        "uint8 (bitcast)": (
            lambda x: jax.lax.bitcast_convert_type(x.astype(jnp.float8_e4m3fn), jnp.uint8),
            lambda p: jax.lax.bitcast_convert_type(p, jnp.float8_e4m3fn).astype(jnp.float32),
        ),
    }

    print("=== cotangent reaching the bf16 source through a quantized carrier ===")
    for name, (cast_fn, uncast_fn) in carriers.items():
        grad = carrier_gradient(cast_fn, uncast_fn)
        verdict = "propagates" if bool(jnp.any(grad != 0)) else "SILENTLY ZERO"
        print(f"  {name:22s} grad={grad}  {verdict}")

    print(
        "\nConclusion: the payload must stay float8-typed where it crosses the op\n"
        "boundary. Bitcasting to uint8 for the collective is fine only inside a\n"
        "custom_vjp that returns a float8-typed value."
    )


if __name__ == "__main__":
    main()
