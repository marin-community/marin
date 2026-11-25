# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import abc
import math
from dataclasses import dataclass

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp

import haliax as hax
from haliax import Axis, NamedArray


def _rotate_half(x: NamedArray, HeadSize: Axis) -> NamedArray:
    """Rotates half of the hidden dims of the input and concatenates them."""
    x1 = x[HeadSize, : HeadSize.size // 2]
    x2 = x[HeadSize, HeadSize.size // 2 :]
    out = hax.concatenate(HeadSize, (-x2, x1))
    return out


class RotaryEmbeddings(eqx.Module):
    HeadDim: Axis = eqx.field(static=True)
    config: "RotaryEmbeddingsConfig" = eqx.field(static=True)

    def __call__(self, q: NamedArray, position_ids: NamedArray) -> NamedArray:
        return _apply_named_rotary(q, position_ids, self.HeadDim, self.config)


def _theta_inverse_frequencies(theta: float, head_dim: int) -> jax.Array:
    channels = 2 * jnp.arange(head_dim // 2, dtype=jnp.float32)
    return 1.0 / (theta ** (channels / head_dim))


@dataclass(frozen=True)
class RotaryEmbeddingsConfig(abc.ABC, draccus.ChoiceRegistry):
    theta: float = 10000.0

    @abc.abstractmethod
    def inverse_frequencies(self, head_dim: int) -> jax.Array:
        """Return one inverse frequency per rotated pair."""

    def scale_coefficients(self, cos, sin):
        """Apply variant-specific scaling after rounding coefficients to the activation dtype."""
        return cos, sin

    def build(self, HeadSize: Axis) -> RotaryEmbeddings:
        return RotaryEmbeddings(HeadSize, self)

    @staticmethod
    def from_hf_config(rope_theta, config: dict | None) -> "RotaryEmbeddingsConfig":
        if config is None:
            return DefaultRotaryEmbeddingsConfig(theta=rope_theta)
        tpe = config.get("rope_type") or config.get("type") or "default"
        return RotaryEmbeddingsConfig.get_choice_class(tpe).make_from_hf_config(rope_theta, config)

    @classmethod
    @abc.abstractmethod
    def make_from_hf_config(cls, rope_theta: float, config: dict) -> "RotaryEmbeddingsConfig":
        pass

    @abc.abstractmethod
    def to_hf_config(self) -> tuple[float, dict | None]:
        """Returns the rope_theta and config dict for the HF config."""
        pass


@dataclass(frozen=True)
class DefaultRotaryEmbeddingsConfig(RotaryEmbeddingsConfig):
    theta: float = 10000
    factor: float = 1.0  # this should have been called scale_factor, but for hf compat

    def inverse_frequencies(self, head_dim: int) -> jax.Array:
        return _theta_inverse_frequencies(self.theta, head_dim) / self.factor

    @classmethod
    def make_from_hf_config(cls, rope_theta: float, config: dict) -> "RotaryEmbeddingsConfig":
        return DefaultRotaryEmbeddingsConfig(theta=rope_theta, factor=config.get("factor", 1.0))

    def to_hf_config(self) -> tuple[float, dict | None]:
        if self.factor == 1.0:
            return self.theta, None
        return self.theta, {"factor": self.factor}


RotaryEmbeddingsConfig.register_subclass("default", DefaultRotaryEmbeddingsConfig)
RotaryEmbeddingsConfig.register_subclass("linear", DefaultRotaryEmbeddingsConfig)


@dataclass(frozen=True)
class Llama3RotaryEmbeddingsConfig(RotaryEmbeddingsConfig):
    """
    To match this from HF:
                "rope_scaling": {
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
              },
    """

    theta: float = 500000
    factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192

    def inverse_frequencies(self, head_dim: int) -> jax.Array:
        inv_freq = _theta_inverse_frequencies(self.theta, head_dim)
        low_wavelength = self.original_max_position_embeddings / self.low_freq_factor
        high_wavelength = self.original_max_position_embeddings / self.high_freq_factor
        wavelength = 2 * jnp.pi / inv_freq
        inv_freq = jnp.where(wavelength > low_wavelength, inv_freq / self.factor, inv_freq)
        smooth = (self.original_max_position_embeddings / wavelength - self.low_freq_factor) / (
            self.high_freq_factor - self.low_freq_factor
        )
        interpolated = (1 - smooth) * inv_freq / self.factor + smooth * inv_freq
        medium = ~(wavelength < high_wavelength) * ~(wavelength > low_wavelength)
        return jnp.where(medium, interpolated, inv_freq)

    @classmethod
    def make_from_hf_config(cls, rope_theta: float, config: dict) -> "RotaryEmbeddingsConfig":
        return Llama3RotaryEmbeddingsConfig(
            theta=rope_theta,
            factor=config.get("factor", 8.0),
            low_freq_factor=config.get("low_freq_factor", 1.0),
            high_freq_factor=config.get("high_freq_factor", 4.0),
            original_max_position_embeddings=config.get("original_max_position_embeddings", 8192),
        )

    def to_hf_config(self) -> tuple[float, dict]:
        return self.theta, {
            "factor": self.factor,
            "low_freq_factor": self.low_freq_factor,
            "high_freq_factor": self.high_freq_factor,
            "original_max_position_embeddings": self.original_max_position_embeddings,
            "rope_type": "llama3",
        }


RotaryEmbeddingsConfig.register_subclass("llama3", Llama3RotaryEmbeddingsConfig)


@dataclass(frozen=True)
class YarnRotaryEmbeddingsConfig(RotaryEmbeddingsConfig):
    theta: float = 10000
    factor: float = 1.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    original_max_position_embeddings: int = 2048
    mscale: float = 1.0

    def inverse_frequencies(self, head_dim: int) -> jax.Array:
        inv_freq = _theta_inverse_frequencies(self.theta, head_dim)

        def correction_dimension(rotations: float) -> float:
            return (
                head_dim
                * math.log(self.original_max_position_embeddings / (rotations * 2 * math.pi))
                / (2 * math.log(self.theta))
            )

        half_dim = head_dim // 2
        low = max(math.floor(correction_dimension(self.beta_fast)), 0)
        high = min(math.ceil(correction_dimension(self.beta_slow)), half_dim - 1)
        ramp = jnp.clip((jnp.arange(half_dim, dtype=jnp.float32) - low) / max(high - low, 1), 0.0, 1.0)
        return (inv_freq / self.factor) * ramp + inv_freq * (1 - ramp)

    @property
    def attention_factor(self) -> float:
        """Multiplier applied to the rounded sine and cosine coefficients."""
        return 1.0 if self.factor <= 1.0 else 0.1 * self.mscale * math.log(self.factor) + 1.0

    def scale_coefficients(self, cos, sin):
        return cos * self.attention_factor, sin * self.attention_factor

    @classmethod
    def make_from_hf_config(cls, rope_theta: float, config: dict) -> "YarnRotaryEmbeddingsConfig":
        return YarnRotaryEmbeddingsConfig(
            theta=rope_theta,
            factor=float(config.get("factor", 1.0)),
            beta_fast=float(config.get("beta_fast", 32.0)),
            beta_slow=float(config.get("beta_slow", 1.0)),
            original_max_position_embeddings=int(config.get("original_max_position_embeddings", 2048)),
            mscale=float(config.get("mscale", 1.0)),
        )

    def to_hf_config(self) -> tuple[float, dict]:
        return self.theta, {
            "rope_type": "yarn",
            "factor": self.factor,
            "beta_fast": self.beta_fast,
            "beta_slow": self.beta_slow,
            "original_max_position_embeddings": self.original_max_position_embeddings,
            "mscale": self.mscale,
        }


RotaryEmbeddingsConfig.register_subclass("yarn", YarnRotaryEmbeddingsConfig)


def _apply_named_rotary(
    q: NamedArray, position_ids: NamedArray, head: Axis, config: RotaryEmbeddingsConfig
) -> NamedArray:
    with jax.ensure_compile_time_eval():
        inv_freq = hax.named(config.inverse_frequencies(head.size), head.resize(head.size // 2))
    frequencies = inv_freq.broadcast_axis(position_ids.axes) * position_ids
    angles = hax.concatenate(head, (frequencies, frequencies))
    cos, sin = config.scale_coefficients(hax.cos(angles).astype(q.dtype), hax.sin(angles).astype(q.dtype))
    return q * cos + _rotate_half(q, head) * sin
