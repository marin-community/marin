# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Array-native default, linear, Llama3 and YaRN rotary embeddings."""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


@dataclass(frozen=True)
class RotaryConfig:
    theta: float = 10000.0
    factor: float = 1.0


@dataclass(frozen=True)
class Llama3RotaryConfig:
    theta: float = 500000.0
    factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192


@dataclass(frozen=True)
class YarnRotaryConfig:
    theta: float = 10000.0
    factor: float = 1.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    original_max_position_embeddings: int = 2048
    mscale: float = 1.0


NativeRotaryConfig = RotaryConfig | Llama3RotaryConfig | YarnRotaryConfig


def yarn_attention_factor(factor: float, mscale: float) -> float:
    """Return YaRN's multiplier for the rotary cosine and sine values."""
    return 1.0 if factor <= 1.0 else 0.1 * mscale * math.log(factor) + 1.0


def _inverse_frequencies(dim: int, config: NativeRotaryConfig) -> Array:
    inv_freq = 1.0 / (config.theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    if isinstance(config, Llama3RotaryConfig):
        low_wavelength = config.original_max_position_embeddings / config.low_freq_factor
        high_wavelength = config.original_max_position_embeddings / config.high_freq_factor
        wavelength = 2 * jnp.pi / inv_freq
        inv_freq = jnp.where(wavelength > low_wavelength, inv_freq / config.factor, inv_freq)
        smooth = (config.original_max_position_embeddings / wavelength - config.low_freq_factor) / (
            config.high_freq_factor - config.low_freq_factor
        )
        interpolated = (1 - smooth) * inv_freq / config.factor + smooth * inv_freq
        medium = ~(wavelength < high_wavelength) * ~(wavelength > low_wavelength)
        return jnp.where(medium, interpolated, inv_freq)
    if isinstance(config, YarnRotaryConfig):
        low = math.floor(
            dim
            * math.log(config.original_max_position_embeddings / (config.beta_fast * 2 * math.pi))
            / (2 * math.log(config.theta))
        )
        high = math.ceil(
            dim
            * math.log(config.original_max_position_embeddings / (config.beta_slow * 2 * math.pi))
            / (2 * math.log(config.theta))
        )
        low, high = max(low, 0), min(high, dim - 1)
        ramp = jnp.clip((jnp.arange(dim // 2, dtype=jnp.float32) - low) / max(high - low, 1), 0.0, 1.0)
        return (inv_freq / config.factor) * ramp + inv_freq * (1 - ramp)
    return inv_freq / config.factor


def apply_rotary(
    x: Float[Array, "B S H D"], position_ids: Int[Array, "B S"], config: NativeRotaryConfig
) -> Float[Array, "B S H D"]:
    """Apply split-half RoPE using caller-supplied token positions."""
    dim = x.shape[-1]
    with jax.ensure_compile_time_eval():
        inv_freq = _inverse_frequencies(dim, config)
    frequencies = position_ids[..., None] * inv_freq
    angles = jnp.concatenate((frequencies, frequencies), axis=-1)[..., None, :]
    cos = jnp.cos(angles).astype(x.dtype)
    sin = jnp.sin(angles).astype(x.dtype)
    if isinstance(config, YarnRotaryConfig):
        temperature = yarn_attention_factor(config.factor, config.mscale)
        cos = cos * temperature
        sin = sin * temperature
    first, second = jnp.split(x, 2, axis=-1)
    rotated = jnp.concatenate((-second, first), axis=-1)
    return (x * cos + rotated * sin).astype(x.dtype)
