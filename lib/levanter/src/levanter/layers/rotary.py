# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import abc
from dataclasses import dataclass
from typing import Tuple

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
    def __call__(self, q: NamedArray, position_ids: NamedArray) -> NamedArray:
        raise NotImplementedError("This is an abstract base class for RotaryEmbeddings. Use a subclass instead.")


class DefaultRotaryEmbeddings(RotaryEmbeddings):
    HeadDim: Axis = eqx.field(static=True)
    config: "DefaultRotaryEmbeddingsConfig" = eqx.field(static=True)

    def __call__(self, q: NamedArray, position_ids: NamedArray) -> NamedArray:
        with jax.ensure_compile_time_eval():
            HeadHalfSize = self.HeadDim.resize(self.HeadDim.size // 2)
            inv_freq: NamedArray = 1.0 / (self.config.theta ** (hax.arange(HeadHalfSize, step=2) / self.HeadDim.size))
            inv_freq = inv_freq / self.config.factor

        freqs = inv_freq.broadcast_axis(position_ids.axes) * position_ids
        emb = hax.concatenate(self.HeadDim, (freqs, freqs))
        cos = hax.cos(emb)
        sin = hax.sin(emb)

        q_embed = q * cos + _rotate_half(q, self.HeadDim) * sin
        return q_embed


@dataclass(frozen=True)
class RotaryEmbeddingsConfig(abc.ABC, draccus.ChoiceRegistry):
    theta: float = 10000.0

    @abc.abstractmethod
    def build(self, HeadSize: Axis) -> RotaryEmbeddings:
        pass

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

    def build(self, HeadSize: Axis) -> RotaryEmbeddings:
        return DefaultRotaryEmbeddings(HeadSize, self)

    @classmethod
    def make_from_hf_config(cls, rope_theta: float, config: dict) -> "RotaryEmbeddingsConfig":
        return DefaultRotaryEmbeddingsConfig(theta=rope_theta, factor=config.get("factor", 1.0))

    def to_hf_config(self) -> tuple[float, dict | None]:
        if self.factor == 1.0:
            return self.theta, None
        return self.theta, {"factor": self.factor}


RotaryEmbeddingsConfig.register_subclass("default", DefaultRotaryEmbeddingsConfig)
RotaryEmbeddingsConfig.register_subclass("linear", DefaultRotaryEmbeddingsConfig)


class Llama3RotaryEmbeddings(RotaryEmbeddings):
    HeadDim: Axis = eqx.field(static=True)
    config: "Llama3RotaryEmbeddingsConfig" = eqx.field(static=True)

    def __call__(self, q: NamedArray, position_ids: NamedArray) -> NamedArray:
        inv_freq_llama = self._compute_inv_freq_llama()
        freqs = position_ids * inv_freq_llama.broadcast_axis(position_ids.axes)
        emb = hax.concatenate(self.HeadDim, (freqs, freqs))
        cos = hax.cos(emb)
        sin = hax.sin(emb)

        q_embed = q * cos + _rotate_half(q, self.HeadDim) * sin
        return q_embed

    @staticmethod
    def init(HeadDim, config):
        return Llama3RotaryEmbeddings(HeadDim, config)

    def _compute_inv_freq_llama(self):
        with jax.ensure_compile_time_eval():
            # This is the Llama3 implementation of rotary embeddings.
            # It uses a different scaling factor and frequency calculation.
            HeadHalfSize = self.HeadDim.resize(self.HeadDim.size // 2)
            inv_freq: NamedArray = 1.0 / (self.config.theta ** (hax.arange(HeadHalfSize, step=2) / self.HeadDim.size))

            old_context_len = self.config.original_max_position_embeddings
            low_freq_wavelen = old_context_len / self.config.low_freq_factor
            high_freq_wavelen = old_context_len / self.config.high_freq_factor

            wavelen = 2 * jnp.pi / inv_freq
            inv_freq_llama = hax.where(wavelen > low_freq_wavelen, inv_freq / self.config.factor, inv_freq)
            smooth_factor = (old_context_len / wavelen - self.config.low_freq_factor) / (
                self.config.high_freq_factor - self.config.low_freq_factor
            )
            smoothed_inv_freq = (
                1 - smooth_factor
            ) * inv_freq_llama / self.config.factor + smooth_factor * inv_freq_llama
            is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
            inv_freq_llama = hax.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        return inv_freq_llama


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

    def build(self, HeadSize: Axis) -> RotaryEmbeddings:
        # https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py#L307
        # Porting that to JAX/Haliax:
        return Llama3RotaryEmbeddings.init(HeadSize, self)

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


class YarnRotaryEmbeddings(RotaryEmbeddings):
    HeadDim: Axis = eqx.field(static=True)
    config: "YarnRotaryEmbeddingsConfig" = eqx.field(static=True)

    def __call__(self, q: NamedArray, position_ids: NamedArray) -> NamedArray:
        import math

        with jax.ensure_compile_time_eval():
            half_dim = self.HeadDim.size // 2
            head_dim = self.HeadDim.size
            HeadHalfSize = self.HeadDim.resize(self.HeadDim.size // 2)
            inv_freq: NamedArray = 1.0 / (
                self.config.theta ** (hax.arange(HeadHalfSize, step=2, dtype=jnp.float32) / head_dim)
            )

            # YaRN β-ramp
            def _find_dim(n_rot: float):
                return (half_dim * math.log(self.config.original_max_position_embeddings / (n_rot * 2 * math.pi))) / (
                    2 * math.log(self.config.theta)
                )

            low = int(math.floor(_find_dim(self.config.beta_fast)))
            high = int(math.ceil(_find_dim(self.config.beta_slow)))
            low, high = max(low, 0), min(high, half_dim - 1)
            ramp = hax.clip((hax.arange(HeadHalfSize, dtype=jnp.float32) - low) / max(high - low, 1), 0.0, 1.0)
            inv_extrap = inv_freq
            inv_interp = inv_freq / self.config.factor
            inv_freq = inv_interp * ramp + inv_extrap * (1 - ramp)
            position_ids_scaled = position_ids / self.config.factor

        freqs = position_ids_scaled * inv_freq.broadcast_axis(position_ids.axes)
        emb = hax.concatenate(self.HeadDim, (freqs, freqs))

        # temperature scaling
        if self.config.factor < 1.0:
            temperature = 1.0
        else:
            temperature = math.sqrt(0.1 * self.config.mscale * math.log(self.config.factor) + 1.0)

        cos = hax.cos(emb) * temperature
        sin = hax.sin(emb) * temperature

        q_embed = q * cos + _rotate_half(q, self.HeadDim) * sin
        return q_embed

    @staticmethod
    def init(HeadDim, config):
        return YarnRotaryEmbeddings(HeadDim, config)


@dataclass(frozen=True)
class YarnRotaryEmbeddingsConfig(RotaryEmbeddingsConfig):
    theta: float = 10000
    factor: float = 1.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    original_max_position_embeddings: int = 2048
    mscale: float = 1.0

    def build(self, HeadSize: Axis) -> RotaryEmbeddings:
        return YarnRotaryEmbeddings.init(HeadSize, self)

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
            "type": "yarn",
            "factor": self.factor,
            "beta_fast": self.beta_fast,
            "beta_slow": self.beta_slow,
            "original_max_position_embeddings": self.original_max_position_embeddings,
            "mscale": self.mscale,
        }


RotaryEmbeddingsConfig.register_subclass("yarn", YarnRotaryEmbeddingsConfig)


def rotary_pos_emb(
    HeadSize: Axis, Pos: Axis, theta: float = 10000, scale: float = 1.0
) -> Tuple[NamedArray, NamedArray]:
    with jax.ensure_compile_time_eval():
        HeadHalfSize = HeadSize.resize(HeadSize.size // 2)
        inv_freq: NamedArray = 1.0 / (theta ** (hax.arange(HeadHalfSize, step=2) / HeadSize.size)) / scale

        position_ids: NamedArray = hax.arange(Pos)

        freqs = position_ids * inv_freq.broadcast_axis(Pos)
        # This is different from the paper but aligns with HF implementation:
        # It uses a different permutation in order to obtain the same calculation
        emb = hax.concatenate(HeadSize, (freqs, freqs))
        cos = hax.cos(emb)
        sin = hax.sin(emb)
        # This is different from the paper but aligns with HF implementation:
        return cos, sin
