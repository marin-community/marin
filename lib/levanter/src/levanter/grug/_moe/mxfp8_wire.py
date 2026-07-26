# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 forward-dispatch wire for expert-parallel MoE (issue #7665).

Where [levanter.grug._moe.fp8_wire][] halves the payload and then dequantizes
back to bf16 before the expert GEMMs, this module hands the quantized payload
straight through to an expert-MLP op that already wants MXFP8 operands. The
quantization is therefore relocated rather than added: the op quantizes the
arrived activations today, and this moves that pass in front of the collective,
where it also covers ``tokens`` rows instead of the capacity buffer's
``cf * tokens * topk``.

Two constraints are load-bearing, both established on #7665.

*Blocking axis.* MXFP8 scale blocks run along the feature axis while the
dispatch permutes the token axis, so a row's scales are a per-row attribute that
travels with its row. Quantizing before the gather is bit-exact against
quantizing after. The token-axis orientation the wgrads need has the opposite
property -- its scale is a property of a *set* of 32 rows, which routing
dissolves -- so it stays a post-arrival rebuild and is not carried here.

*The quantized payload must not be an autodiff boundary.* Neither carrier dtype
works for a value that a downstream ``custom_vjp`` hands a cotangent back
through. A ``uint8`` payload has a float0 tangent type, so the cotangent is
dropped silently and the model trains on a zeroed dispatch gradient with every
finiteness guard passing. A ``float8_e4m3fn`` payload does propagate, but JAX
matches the cotangent to the primal's tangent type, so a bf16 ``dx`` is
downcast to unscaled e4m3 on the way back -- saturating above 448 and flushing
below the subnormal floor. Both failures are silent.

The consequence is that the caller must own the whole chain, quantize through
expert MLP, under a single ``custom_vjp`` whose only differentiable boundary is
bf16 in, bf16 out -- the shape the Hopper fused-dispatch path used (#6911). That
wrapper has to drive the expert-MLP op's forward and backward pipelines
directly rather than its ``custom_vjp``, since routing a cotangent through the
op's quantized primal input reintroduces the same downcast. It is not written
yet; this module currently provides the verified primitive only.

Payload and scales share one collective: rows are ``D`` payload bytes followed
by ``D/32`` e8m0 scale bytes, a stride of ``33D/32`` against bf16's ``2D``. The
EP64 profile attributes cost to many small per-layer dispatch legs, so adding a
second collective for the scales would work against the transport effort.
"""

import jax
import jax.numpy as jnp

SF_VEC_SIZE = 32  # MXFP8 block size along the feature axis
E4M3_MAX = 448.0


def cast_to_e8m0_with_rounding_up(x):
    """f32 -> e8m0 exponent byte with round-UP.

    Mirrors ``jax.nn.scaled_matmul``'s stablehlo lowering. The vendored
    ``mxfp8_grouped.quantize`` reference the expert kernels use carries the same
    math; ``experiments/grug/moe/test_mxfp8_wire_parity.py`` pins the two to
    bit-exact agreement from the side of the dependency that is allowed to
    import both.
    """
    bits = x.astype(jnp.float32).view(jnp.uint32)
    exponent = bits >> 23
    mantissa = bits & 0x7FFFFF
    round_up = jnp.logical_and(
        jnp.logical_and(mantissa > 0, exponent != 0xFE),
        ~jnp.logical_and(exponent == 0, mantissa <= 0x400000),
    )
    return jnp.where(round_up, exponent + 1, exponent).astype(jnp.uint8)


def e8m0_to_f32(scale_u8):
    """Exact e8m0 decode: bit-construct ``2^(e - 127)`` (byte 0 -> ``2^-127``).

    Not ``jnp.exp2``: on GPU that lowers to the approximate ``ex2`` path, which
    misses the exact power of two on 217 of the 256 exponent bytes.
    """
    e = scale_u8.astype(jnp.uint32)
    bits = jnp.where(e == 0, jnp.uint32(0x00400000), e << 23)
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def quantize_mxfp8_rows(x):
    """Quantize ``x[T, D]`` along the feature axis. Returns ``(q, scales[T, D//32])``.

    Row-local by construction, which is what makes the payload survive the
    dispatch permutation unchanged.
    """
    d = x.shape[-1]
    if d % SF_VEC_SIZE != 0:
        raise ValueError(f"MXFP8 wire needs a feature dim divisible by {SF_VEC_SIZE}; got {d}")
    blocks = x.astype(jnp.float32).reshape(*x.shape[:-1], d // SF_VEC_SIZE, SF_VEC_SIZE)
    amax = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True)
    scales = cast_to_e8m0_with_rounding_up(amax / E4M3_MAX)
    scaled = blocks / e8m0_to_f32(scales)
    # An all-zero block gives amax 0 -> scale byte 0 -> a subnormal 2^-127
    # divisor. Backends that flush denormals turn that into 0/0; dropped slots
    # and pad rows make all-zero rows routine, so mask rather than divide.
    scaled = jnp.where(amax > 0, scaled, 0.0)
    q = jnp.clip(scaled, -E4M3_MAX, E4M3_MAX).astype(jnp.float8_e4m3fn)
    return q.reshape(x.shape), scales.squeeze(-1)


def dequantize_mxfp8_rows(q, scales):
    """Inverse of [quantize_mxfp8_rows][] (f32 out); for tests and the wgrad rebuild."""
    d = q.shape[-1]
    blocks = q.astype(jnp.float32).reshape(*q.shape[:-1], d // SF_VEC_SIZE, SF_VEC_SIZE)
    return (blocks * e8m0_to_f32(scales)[..., None]).reshape(q.shape)


def _pack(q, scales):
    """[T, D] e4m3 + [T, D/32] e8m0 -> [T, D + D/32] uint8, one buffer, one collective."""
    return jnp.concatenate([jax.lax.bitcast_convert_type(q, jnp.uint8), scales], axis=-1)


def _unpack(packed, feature_dim):
    payload = jax.lax.bitcast_convert_type(packed[..., :feature_dim], jnp.float8_e4m3fn)
    return payload, packed[..., feature_dim:]


def _mxfp8_all_gather_impl(x, axis_name):
    q, scales = quantize_mxfp8_rows(x)
    gathered = jax.lax.all_gather(_pack(q, scales), axis_name, tiled=True)
    return _unpack(gathered, x.shape[-1])


def mxfp8_all_gather(x, axis_name):
    """Quantize ``x[Tlocal, D]`` and all-gather it as MXFP8, one collective.

    Returns ``(payload, scales)``: ``payload`` is ``float8_e4m3fn`` of shape
    ``[S*Tlocal, D]``, ``scales`` is ``uint8`` e8m0 of shape
    ``[S*Tlocal, D//32]``, raw and unswizzled. The Blackwell scale swizzle stays
    with the consumer because it is parameterised by ``group_sizes``, which
    routing produces.

    NOT DIFFERENTIABLE, by design. Both plausible payload carriers fail
    silently across an autodiff boundary (see the module docstring), so this
    must be called inside a ``custom_vjp`` that spans quantize through the
    expert MLP and exposes only bf16 on both sides. That wrapper has to drive
    the expert-MLP op's forward and backward pipelines directly rather than its
    ``custom_vjp``, because routing a cotangent through the op's quantized
    primal input reintroduces the same downcast. Tracked on #7665.
    """
    return _mxfp8_all_gather_impl(x, axis_name)
