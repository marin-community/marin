# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Behaviour of the MXFP8 forward-dispatch wire (issue #7665)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import shard_map
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug._moe.mxfp8_wire import (
    SF_VEC_SIZE,
    _pack,
    _unpack,
    dequantize_mxfp8_rows,
    e8m0_to_f32,
    mxfp8_all_gather,
    quantize_mxfp8_rows,
)

TOKENS, FEATURES = 64, 128


def _activations(seed=0, scale=1.0):
    key = jax.random.PRNGKey(seed)
    k_base, k_tok = jax.random.split(key)
    base = jax.random.normal(k_base, (TOKENS, FEATURES), dtype=jnp.float32)
    per_token = jnp.exp(jax.random.normal(k_tok, (TOKENS, 1), dtype=jnp.float32))
    return (base * per_token * scale).astype(jnp.bfloat16)


def _expert_mesh_or_skip(min_shards: int = 2) -> Mesh:
    devices = jax.devices()
    if len(devices) < min_shards:
        pytest.skip(f"needs >= {min_shards} devices")
    return Mesh(np.array(devices), axis_names=("expert",), axis_types=(AxisType.Explicit,))


def test_quantization_is_row_local_so_it_survives_the_dispatch_permutation():
    """Quantizing before the dispatch gather equals quantizing after it.

    This is what makes the forward payload free: the gather replicates and
    reorders whole rows, and a row's scales belong to that row.
    """
    x = _activations()
    sources = jax.random.randint(jax.random.PRNGKey(3), (200,), 0, TOKENS)

    after_q, after_sf = quantize_mxfp8_rows(jnp.take(x.astype(jnp.float32), sources, axis=0))
    src_q, src_sf = quantize_mxfp8_rows(x.astype(jnp.float32))
    wire_q = jnp.take(src_q, sources, axis=0)
    wire_sf = jnp.take(src_sf, sources, axis=0)

    assert jnp.array_equal(after_q.view(jnp.uint8), wire_q.view(jnp.uint8))
    assert jnp.array_equal(after_sf, wire_sf)


def test_all_zero_rows_quantize_to_zero_not_nan():
    """Dropped slots and pad rows are all-zero; an amax-0 block must not divide by
    the subnormal 2^-127 scale, which flushes to 0/0 on some backends."""
    q, scales = quantize_mxfp8_rows(jnp.zeros((SF_VEC_SIZE, FEATURES), jnp.float32))
    out = dequantize_mxfp8_rows(q, scales)
    assert not bool(jnp.any(jnp.isnan(out)))
    assert bool(jnp.all(out == 0))


def test_masking_a_row_after_quantization_matches_quantizing_a_masked_row():
    """Backends zero invalid slots; doing it on the payload must agree."""
    x = _activations(seed=5)
    keep = jax.random.uniform(jax.random.PRNGKey(6), (TOKENS,)) < 0.7

    masked_first = quantize_mxfp8_rows(jnp.where(keep[:, None], x.astype(jnp.float32), 0.0))
    q, sf = quantize_mxfp8_rows(x.astype(jnp.float32))
    masked_after = (
        jnp.where(keep[:, None], q, jnp.zeros_like(q)),
        jnp.where(keep[:, None], sf, jnp.zeros_like(sf)),
    )

    assert jnp.array_equal(dequantize_mxfp8_rows(*masked_first), dequantize_mxfp8_rows(*masked_after))


def test_pack_round_trips_and_costs_33_of_64_bytes_against_bf16():
    x = _activations(seed=7)
    q, scales = quantize_mxfp8_rows(x.astype(jnp.float32))
    packed = _pack(q, scales)

    assert packed.dtype == jnp.uint8
    assert packed.shape == (TOKENS, FEATURES + FEATURES // SF_VEC_SIZE)
    assert packed.nbytes * 64 == x.nbytes * 33

    payload, recovered = _unpack(packed, FEATURES)
    assert payload.dtype == jnp.float8_e4m3fn
    assert jnp.array_equal(payload.view(jnp.uint8), q.view(jnp.uint8))
    assert jnp.array_equal(recovered, scales)


def test_payload_leaves_the_wire_float8_typed_not_uint8():
    """Regression guard for the silent-gradient failure.

    A uint8 payload has a float0 tangent type, so a cotangent crossing this
    boundary is dropped without error and the dispatch gradient becomes zero.
    """
    mesh = _expert_mesh_or_skip()
    x = _activations(seed=9)
    with jax.set_mesh(mesh):
        xs = jax.device_put(x, NamedSharding(mesh, P("expert")))
        payload, scales = jax.jit(
            shard_map(
                lambda v: mxfp8_all_gather(v, "expert"),
                mesh=mesh,
                in_specs=(P("expert"),),
                out_specs=(P(), P()),
                check_vma=False,
            )
        )(xs)

    assert payload.dtype == jnp.float8_e4m3fn
    assert scales.dtype == jnp.uint8
    assert payload.shape == (TOKENS, FEATURES)
    assert scales.shape == (TOKENS, FEATURES // SF_VEC_SIZE)


def test_a_cotangent_crossing_the_payload_is_silently_corrupted():
    """Why [mxfp8_all_gather][] is not a differentiable seam.

    A float8 payload does propagate a cotangent, unlike a uint8 one, but JAX
    matches the cotangent to the primal's tangent type. A bf16 dx handed back
    through this boundary is therefore downcast to unscaled e4m3 -- saturating
    above 448 and flushing below the subnormal floor -- with nothing raised.
    """
    mesh = _expert_mesh_or_skip()
    x = _activations(seed=11)

    @jax.custom_vjp
    def consume(payload):
        return payload.astype(jnp.float32).sum()

    def consume_bwd(_res, ct):
        # A downstream op's honest bf16 gradient, well outside e4m3's range.
        return (jnp.full((TOKENS, FEATURES), 1e-6, jnp.bfloat16) * ct,)

    consume.defvjp(lambda p: (consume(p), None), consume_bwd)

    def loss(v):
        payload, _ = mxfp8_all_gather(v, "expert")
        return consume(payload)

    with jax.set_mesh(mesh):
        xs = jax.device_put(x, NamedSharding(mesh, P("expert")))
        grad = jax.jit(
            shard_map(jax.grad(loss), mesh=mesh, in_specs=(P("expert"),), out_specs=P("expert"), check_vma=False)
        )(xs)

    # 1e-6 is below e4m3's smallest subnormal (2^-9), so the gradient is gone.
    assert bool(jnp.all(np.asarray(grad) == 0)), "expected the e4m3 downcast to flush this gradient"


@pytest.mark.parametrize("byte", [0, 1, 127, 128, 200, 254])
def test_e8m0_decode_is_an_exact_power_of_two(byte):
    """jnp.exp2 lowers to an approximate path on GPU and misses most of these."""
    decoded = float(e8m0_to_f32(jnp.uint8(byte)))
    expected = 2.0**-127 if byte == 0 else 2.0 ** (byte - 127)
    assert decoded == expected


def test_e8m0_top_byte_overflows_f32():
    """Byte 255 is 2^128, above f32 max; the OCP spec reserves it as NaN."""
    assert float(e8m0_to_f32(jnp.uint8(255))) == float("inf")
