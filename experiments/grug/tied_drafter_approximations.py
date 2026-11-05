# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded drafter approximations; FP4 arithmetic is emulated on H100."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from levanter.grug._moe.ep_fixed_pooled_wave_all_to_all import _fp8_pack_rows, _fp8_unpack_rows
from levanter.grug.sharding import unshard

FP4_BLOCK = 16
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def fp4_pack_rows(x):
    """Pack E2M1 values, E4M3 scales per 16 values, and an FP32 scale per row.

    This is NVFP4-inspired row scaling, not the native tensor-wide NVFP4 recipe.
    The wire layout is packed nibbles, FP8 block scales, then FP32 row scale.
    """
    width = x.shape[-1]
    blocks = x.astype(jnp.float32).reshape(*x.shape[:-1], width // FP4_BLOCK, FP4_BLOCK)
    block_max = jnp.max(jnp.abs(blocks), axis=-1)
    row_scale = jnp.maximum(jnp.max(block_max, axis=-1, keepdims=True) / (6.0 * 448.0), 1e-12)
    block_scale = (block_max / (6.0 * row_scale)).astype(jnp.float8_e4m3fn)
    scale = jnp.maximum(block_scale.astype(jnp.float32) * row_scale, 1e-12)
    normalized = blocks / scale[..., None]
    magnitude = jnp.abs(normalized)
    thresholds = jnp.asarray(FP4_MIDPOINTS)
    # Round ties toward the even E2M1 code.
    above = (magnitude[..., None] > thresholds) | (
        (magnitude[..., None] == thresholds) & jnp.asarray((False, True, False, True, False, True, False))
    )
    codes = jnp.sum(above, axis=-1).astype(jnp.uint8)
    codes = codes | (jnp.signbit(normalized).astype(jnp.uint8) << 3)
    codes = codes.reshape(*x.shape[:-1], width)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    block_bytes = jax.lax.bitcast_convert_type(block_scale, jnp.uint8)
    row_bytes = jax.lax.bitcast_convert_type(row_scale[..., 0], jnp.uint8)
    return jnp.concatenate((packed, block_bytes, row_bytes), axis=-1)


def fp4_unpack_rows(packed, *, width, dtype):
    leading = packed.shape[:-1]
    codes = packed[..., : width // 2]
    nibbles = jnp.stack((codes & 15, codes >> 4), axis=-1).reshape(*leading, width)
    magnitude = jnp.take(jnp.asarray(FP4_VALUES), nibbles & 7)
    values = jnp.where((nibbles & 8) != 0, -magnitude, magnitude)
    blocks = width // FP4_BLOCK
    scales = jax.lax.bitcast_convert_type(packed[..., width // 2 : width // 2 + blocks], jnp.float8_e4m3fn)
    row_scale = jax.lax.bitcast_convert_type(packed[..., -4:], jnp.float32)
    decoded = values.reshape(*leading, blocks, FP4_BLOCK) * scales.astype(jnp.float32)[..., None]
    return (decoded * row_scale[..., None, None]).reshape(*leading, width).astype(dtype)


def fp4_roundtrip(x):
    return fp4_unpack_rows(fp4_pack_rows(x), width=x.shape[-1], dtype=x.dtype)


class FixedScale(eqx.Module):
    scale: float = eqx.field(static=True)

    def __call__(self, x):
        return x * self.scale


class GainOnly(eqx.Module):
    weight: jax.Array

    def __call__(self, x):
        return (x.astype(jnp.float32) * unshard(self.weight)).astype(x.dtype)


class Fp4Norm(eqx.Module):
    base: eqx.Module

    def __call__(self, x):
        value = self.base(x)
        spec = jax.typeof(value).sharding.spec
        return jax.shard_map(fp4_roundtrip, in_specs=spec, out_specs=spec)(value)


def fp4_expert_weight(path, value):
    if "expert_mlp" not in jax.tree_util.keystr(path):
        return value
    spec = jax.typeof(value).sharding.spec
    return jax.shard_map(fp4_roundtrip, in_specs=spec, out_specs=spec)(value)


def modify_blocks(model, mode):
    """Apply one post-restriction ablation without changing the source checkpoint."""
    if mode == "none":
        return model
    if mode == "fp4_expert_weights":
        return jax.tree_util.tree_map_with_path(fp4_expert_weight, model)
    if mode == "drop_shared":
        return eqx.tree_at(
            lambda m: [s.stacked.shared for s in m.layer_stacks()], model, [None for _ in model.layer_stacks()]
        )
    if mode in ("gate_half", "gate_identity"):
        scale = 0.5 if mode == "gate_half" else 1.0
        return eqx.tree_at(
            lambda m: [n for s in m.layer_stacks() for n in (s.stacked.attn_gated_norm, s.stacked.mlp_gated_norm)],
            model,
            [FixedScale(scale) for _ in range(2 * len(model.layer_stacks()))],
        )
    if mode == "rms_gain_only":
        return eqx.tree_at(
            lambda m: [n for s in m.layer_stacks() for n in (s.stacked.rms_attn, s.stacked.rms_mlp)],
            model,
            [GainOnly(n.weight) for s in model.layer_stacks() for n in (s.stacked.rms_attn, s.stacked.rms_mlp)],
        )
    if mode == "fp4_expert_inputs":
        return eqx.tree_at(
            lambda m: [s.stacked.mlp.latent_norm for s in m.layer_stacks()],
            model,
            [Fp4Norm(s.stacked.mlp.latent_norm) for s in model.layer_stacks()],
        )
    raise ValueError(f"Unknown approximation: {mode}")


def packed_exchange(x, *, encoding):
    """Forward-only communication microbenchmark; no training backward is claimed."""
    if encoding == "fp4":
        payload = fp4_pack_rows(x)
    elif encoding == "fp8":
        payload = _fp8_pack_rows(x)
    elif encoding == "bf16":
        payload = x
    else:
        raise ValueError(encoding)
    received = jax.lax.all_to_all(payload, "expert", split_axis=0, concat_axis=0, tiled=True)
    if encoding == "fp4":
        return fp4_unpack_rows(received, width=x.shape[-1], dtype=x.dtype)
    if encoding == "fp8":
        return _fp8_unpack_rows(received, hidden_dim=x.shape[-1], dtype=x.dtype)
    return received


def exchange_function(spec, encoding):
    return jax.jit(jax.shard_map(partial(packed_exchange, encoding=encoding), in_specs=spec, out_specs=spec))
