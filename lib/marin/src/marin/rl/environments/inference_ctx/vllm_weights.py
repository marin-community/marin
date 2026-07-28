# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

_VLLM_ATTENTION_HEAD_DIM_MULTIPLE = 128


@dataclass(frozen=True)
class VLLMWeight:
    """One source weight exposed through TPU-inference's sync protocol."""

    value: jax.Array

    def get_value(self) -> jax.Array:
        return self.value


@dataclass(frozen=True)
class VLLMWeightState:
    """Flat source weights accepted by TPU-inference's ``sync_weights``."""

    weights: tuple[tuple[tuple[str, ...], VLLMWeight], ...]

    def flat_state(self) -> tuple[tuple[tuple[str, ...], VLLMWeight], ...]:
        return self.weights

    def __len__(self) -> int:
        return len(self.weights)


def _vllm_weight_path(parts: tuple[str, ...]) -> tuple[str, ...]:
    parameter_name = parts[-2]
    if parts[-1] == "bias":
        parameter_name = f"{parameter_name}_bias"
    return (*parts[:-2], parameter_name)


def _pad_head_dim_to_multiple_of_128(value: jax.Array, axis: int) -> jax.Array:
    head_size = value.shape[axis]
    padded_head_size = (
        (head_size + _VLLM_ATTENTION_HEAD_DIM_MULTIPLE - 1) // _VLLM_ATTENTION_HEAD_DIM_MULTIPLE
    ) * _VLLM_ATTENTION_HEAD_DIM_MULTIPLE
    if head_size == padded_head_size:
        return value

    padding = [(0, 0)] * value.ndim
    padding[axis] = (0, padded_head_size - head_size)
    return jnp.pad(value, padding)


def _reshape_and_pad_attention_weight(value: jax.Array, parts: tuple[str, ...], is_bias: bool) -> jax.Array:
    if "self_attn" not in parts:
        return value

    if "q_proj" in parts:
        if value.ndim == 4:
            kv_heads, q_heads_per_group, head_size, embed = value.shape
            value = value.reshape(kv_heads * q_heads_per_group, head_size, embed)
        elif value.ndim == 3 and is_bias:
            kv_heads, q_heads_per_group, head_size = value.shape
            value = value.reshape(kv_heads * q_heads_per_group, head_size)

    if ("q_proj" in parts or "k_proj" in parts or "v_proj" in parts) and value.ndim >= 2:
        return _pad_head_dim_to_multiple_of_128(value, axis=1)
    if "o_proj" in parts and not is_bias and value.ndim == 3:
        return _pad_head_dim_to_multiple_of_128(value, axis=2)
    return value


def levanter_state_dict_to_vllm_weights_on_cpu(state_dict: dict[str, ArrayLike]) -> VLLMWeightState:
    """Prepare a Levanter state dict for TPU-inference's weight-sync API."""
    with jax.default_device(jax.devices("cpu")[0]):
        weights = []
        for key, raw_value in state_dict.items():
            parts = tuple(key.split("."))
            value = _reshape_and_pad_attention_weight(
                jnp.asarray(raw_value),
                parts[:-1],
                is_bias=parts[-1] == "bias",
            )
            weights.append((_vllm_weight_path(parts), VLLMWeight(value)))
    return VLLMWeightState(tuple(weights))
