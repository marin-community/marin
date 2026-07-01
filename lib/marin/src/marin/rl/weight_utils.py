# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from flax import nnx
from levanter.models.lm_model import LmHeadModel


def _get_nnx_key_name(split_key: list[str]) -> str:
    """
    Determine the NNX key name from the split Levanter key.
    If the key ends in 'bias', append '_bias' to the parameter name.
    Otherwise (e.g. 'weight'), use the parameter name directly.
    """
    key_name = split_key[-2]
    if split_key[-1] == "bias":
        key_name = f"{key_name}_bias"
    return key_name


def _pad_head_dim_to_multiple_of_128(value: jax.Array, axis: int) -> jax.Array:
    """Pad ``value`` along ``axis`` up to the next multiple of 128.

    vLLM expects attention weights padded to a multiple of 128, since its Pallas
    kernels require it.
    """
    head_size = value.shape[axis]
    next_multiple_of_128 = ((head_size + 127) // 128) * 128
    if head_size >= next_multiple_of_128:
        return value
    padding = [(0, 0)] * value.ndim
    padding[axis] = (0, next_multiple_of_128 - head_size)
    return jnp.pad(value, padding)


def _reshape_and_pad_attention_param(value: jax.Array, parts: list[str], is_bias: bool) -> jax.Array:
    """Prepare a Levanter attention parameter for vLLM.

    Flattens grouped query heads to ``(Total Heads, Head Dim, [Embed])`` and pads the
    head dimension to a multiple of 128. Non-attention parameters pass through unchanged.

    Args:
        value: The parameter array.
        parts: The key components excluding the trailing ``weight``/``bias`` token.
        is_bias: Whether this parameter is a bias (1D-per-head) rather than a weight.
    """
    if "self_attn" not in parts:
        return value

    if "q_proj" in parts:
        if value.ndim == 4:
            # Weight: (KV, Group, HeadSize, Embed) -> (Heads, HeadSize, Embed)
            kv_heads, q_heads_per_group, head_size, embed = value.shape
            value = value.reshape(kv_heads * q_heads_per_group, head_size, embed)
        elif value.ndim == 3 and is_bias:
            # Bias: (KV, Group, HeadSize) -> (Heads, HeadSize)
            kv_heads, q_heads_per_group, head_size = value.shape
            value = value.reshape(kv_heads * q_heads_per_group, head_size)

    if "q_proj" in parts or "k_proj" in parts or "v_proj" in parts:
        # Q/K/V projections carry the head dimension on axis 1.
        if value.ndim >= 2:
            value = _pad_head_dim_to_multiple_of_128(value, axis=1)
    elif "o_proj" in parts and not is_bias and value.ndim == 3:
        # o_proj weight is (Embed, Heads, HeadSize); pad the head dimension on axis 2.
        value = _pad_head_dim_to_multiple_of_128(value, axis=2)

    return value


def _insert_nested_param(nested_state_dict: dict, split_key: list[str], value: jax.Array) -> None:
    """Insert ``value`` into ``nested_state_dict`` following the dotted ``split_key``.

    The final dotted component (``weight``/``bias``) is folded into the parameter name by
    :func:`_get_nnx_key_name` instead of becoming its own nesting level, because the leaf
    name is what ``transpose_keys`` consults to pick a transpose.
    """
    current = nested_state_dict
    for part in split_key[:-2]:
        current = current.setdefault(part, {})
    current[_get_nnx_key_name(split_key)] = nnx.Param(value)


def levanter_to_nnx_state(levanter_model: LmHeadModel) -> nnx.State:
    """Convert a Levanter model's flat state dict into the nested ``nnx.State`` vLLM expects.

    Levanter state dicts are flat (``model.layers.0.self_attn.q_proj.weight -> array``); vLLM
    consumes a nested mapping (``{model: {layers: {0: {self_attn: {q_proj: array}}}}}``).
    """
    nested_state_dict: dict = {}
    for key, value in levanter_model.to_state_dict().items():
        split_key = key.split(".")
        value = _reshape_and_pad_attention_param(value, split_key[:-1], is_bias=split_key[-1] == "bias")
        _insert_nested_param(nested_state_dict, split_key, value)
    return nnx.State(nested_state_dict)


def levanter_state_dict_to_nnx_state_on_cpu(state_dict: dict) -> nnx.State:
    """Like :func:`levanter_to_nnx_state`, but for an already-extracted state dict, on CPU.

    Used on the inference side where weights arrive (possibly as numpy) over the wire and must
    be placed on the host before being synced into vLLM.
    """
    with jax.default_device(jax.devices("cpu")[0]):
        nested_state_dict: dict = {}
        for key, value in state_dict.items():
            value = jnp.asarray(value)
            split_key = key.split(".")
            value = _reshape_and_pad_attention_param(value, split_key[:-1], is_bias=split_key[-1] == "bias")
            _insert_nested_param(nested_state_dict, split_key, value)
        return nnx.State(nested_state_dict)
