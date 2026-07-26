# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FP8W-001: does an MXFP8 forward-dispatch wire change the expert-MLP numerics?

Runs on CPU. See `.agents/logbooks/fp8-dispatch-wire.md` and issue #7665.

Two questions, corresponding to the two operands the expert MLP builds from the
dispatch buffer:

H1 (forward operand, blocked along features). The dispatch permutes the token
axis while MXFP8 blocks lie along the feature axis, so a row's scale should
travel with its row and quantize-then-gather should equal gather-then-quantize.

H2 (wgrad operand, blocked along tokens). This orientation cannot cross the
permutation -- its scale is a property of a set of 32 rows, and routing dissolves
that set -- so it must be rebuilt from the arrived payload rather than from a
bf16 original. Because e8m0 scales are exact powers of two and e4m3 is a
floating-point grid, the rebuild should be a mantissa-preserving exponent shift
rather than a second rounding, diverging only where a value shifts into e4m3
subnormals.

Usage: uv run python experiments/grug/moe/standalone/study_fp8_wire_numerics.py
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))
from mxfp8_grouped.quantize import (  # noqa: E402
    dequantize_mxfp8,
    dequantize_mxfp8_tokens,
    quantize_mxfp8,
    quantize_mxfp8_tokens,
)

REGIMES = {
    "benign": dict(token_spread=0.3, outlier_frac=0.0, outlier_gain=1.0),
    "typical": dict(token_spread=0.8, outlier_frac=0.01, outlier_gain=20.0),
    "harsh": dict(token_spread=1.5, outlier_frac=0.02, outlier_gain=100.0),
    "adversarial": dict(token_spread=3.0, outlier_frac=0.05, outlier_gain=1000.0),
}


def synth_dispatch(key, rows, features, *, token_spread, outlier_frac, outlier_gain):
    """Dispatch-buffer-shaped activations with LLM-like structure.

    Two effects drive the row-vs-column scale mismatch this study is about:
    per-token magnitude spread (log-normal) and a minority of very large feature
    channels, which is the documented shape of transformer activations.
    """
    k_base, k_tok, k_feat = jax.random.split(key, 3)
    base = jax.random.normal(k_base, (rows, features), dtype=jnp.float32)
    per_token = jnp.exp(jax.random.normal(k_tok, (rows, 1), dtype=jnp.float32) * token_spread)
    per_feature = jnp.where(
        jax.random.uniform(k_feat, (1, features), dtype=jnp.float32) < outlier_frac,
        outlier_gain,
        1.0,
    )
    return (base * per_token * per_feature).astype(jnp.bfloat16)


def column_operand_paths(x):
    """Token-orientation operand built the current way (A) and via the wire (B)."""
    xf = x.astype(jnp.float32)
    a = quantize_mxfp8_tokens(xf)
    row_q, row_sf = quantize_mxfp8(xf)
    arrived = dequantize_mxfp8(row_q, row_sf)  # exactly what comes off the wire
    b = quantize_mxfp8_tokens(arrived)
    return a, b


def check_forward_operand_commutes():
    """H1: quantize(gather(x)) vs gather(quantize(x)) through a routing-shaped gather."""
    print("=== H1: forward operand, does quantize commute with the dispatch gather? ===")
    tokens, features, capacity = 1024, 512, 2048
    x = synth_dispatch(jax.random.PRNGKey(7), tokens, features, **REGIMES["harsh"])

    # Non-order-preserving, non-injective (top-k replication), non-surjective
    # (drops), plus the validity mask the backends apply.
    k_src, k_valid = jax.random.split(jax.random.PRNGKey(11))
    token_sources = jax.random.randint(k_src, (capacity,), 0, tokens)
    valid = jax.random.uniform(k_valid, (capacity,)) < 0.85

    gathered = jnp.take(x.astype(jnp.float32), token_sources, axis=0)
    after_q, after_sf = quantize_mxfp8(jnp.where(valid[:, None], gathered, 0.0))

    src_q, src_sf = quantize_mxfp8(x.astype(jnp.float32))
    wire_q = jnp.where(valid[:, None], jnp.take(src_q, token_sources, axis=0), jnp.float8_e4m3fn(0))
    wire_sf = jnp.where(valid[:, None], jnp.take(src_sf, token_sources, axis=0), jnp.uint8(0))

    same_bytes = bool(jnp.all(after_q.view(jnp.uint8)[valid] == wire_q.view(jnp.uint8)[valid]))
    same_scales = bool(jnp.all(after_sf[valid] == wire_sf[valid]))
    same_values = bool(
        jnp.all(dequantize_mxfp8(after_q, after_sf)[valid] == dequantize_mxfp8(wire_q, wire_sf)[valid])
    )
    print(f"  valid rows: {int(jnp.sum(valid))} of {capacity}")
    print(f"  identical e4m3 bytes : {same_bytes}")
    print(f"  identical e8m0 scales: {same_scales}")
    print(f"  identical values     : {same_values}")

    # Dropped slots: the wire masks bytes and scales; the bf16 path quantizes a
    # zeroed row, which hits the all-zero-block division below.
    after_deq = dequantize_mxfp8(after_q, after_sf)
    wire_deq = dequantize_mxfp8(wire_q, wire_sf)
    print(f"  dropped slots -- bf16 path exact zero: {bool(jnp.all(after_deq[~valid] == 0))}"
          f" (nan: {bool(jnp.any(jnp.isnan(after_deq[~valid])))})")
    print(f"  dropped slots -- wire path exact zero: {bool(jnp.all(wire_deq[~valid] == 0))}"
          f" (nan: {bool(jnp.any(jnp.isnan(wire_deq[~valid])))})")


def check_wgrad_operand():
    """H2: added wgrad error from rebuilding, against the existing noise floor."""
    print("\n=== H2: wgrad operand, added error vs. the accepted noise floor ===")
    rows, features, intermediate = 2048, 1024, 512
    for i, (name, kw) in enumerate(REGIMES.items()):
        x = synth_dispatch(jax.random.PRNGKey(i), rows, features, **kw).astype(jnp.float32)
        dh = synth_dispatch(jax.random.PRNGKey(100 + i), rows, intermediate, **kw).astype(jnp.float32)

        (a_q, a_sf), (b_q, b_sf) = column_operand_paths(x)
        dh_deq = dequantize_mxfp8_tokens(*quantize_mxfp8_tokens(dh))

        reference = x.T @ dh
        dw_a = dequantize_mxfp8_tokens(a_q, a_sf).T @ dh_deq
        dw_b = dequantize_mxfp8_tokens(b_q, b_sf).T @ dh_deq

        err_a = float(jnp.linalg.norm(dw_a - reference))
        norm = float(jnp.linalg.norm(reference))
        added = float(jnp.linalg.norm(dw_b - dw_a))

        a_deq = dequantize_mxfp8_tokens(a_q, a_sf)
        b_deq = dequantize_mxfp8_tokens(b_q, b_sf)
        agree = float(jnp.mean((a_deq == b_deq).astype(jnp.float32)))
        nonzero = jnp.abs(x) > 0
        extra_zeroed = float(jnp.mean(((b_deq == 0) & nonzero).astype(jnp.float32))) - float(
            jnp.mean(((a_deq == 0) & nonzero).astype(jnp.float32))
        )

        print(
            f"  {name:12s} relerr(A) {err_a / norm:.4e}   added ||B-A||/||A-ref|| {added / err_a:.3e}"
            f"   values agree {agree * 100:6.2f}%   extra flush-to-zero {extra_zeroed * 100:+.4f}pp"
        )


def check_zero_block():
    """Incidental: all-zero blocks divide by a subnormal scale."""
    print("\n=== incidental: all-zero 32-element block ===")
    q, sf = quantize_mxfp8(jnp.zeros((32, 64), jnp.float32))
    deq = dequantize_mxfp8(q, sf)
    print(f"  scale byte: {int(jnp.unique(sf)[0])}  -> decodes to 2^-127 (subnormal f32)")
    print(f"  dequantized block is all-NaN: {bool(jnp.all(jnp.isnan(deq)))}")
    print(f"  jit(0.0 / 2**-127) = {float(jax.jit(lambda a, b: a / b)(jnp.float32(0.0), jnp.float32(2.0**-127)))}")


if __name__ == "__main__":
    check_forward_operand_commutes()
    check_wgrad_operand()
    check_zero_block()
