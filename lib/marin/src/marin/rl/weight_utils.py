# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
from flax import nnx
import jax.numpy as jnp
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


def levanter_to_nnx_state(levanter_model: LmHeadModel) -> dict:
    # The format of this state dict is flat like:
    # model.layers.0.self_attn.q_proj.weight -> jax array
    # We are creating a new state dict that is nested because
    # that's the format that is expected by nnx.State.
    # This converts the dictionary to something of the form of
    #  model: {layers: {0: {self_attn: {q_proj: {weight: jax array}}}}}
    # We do not include the "weight" part of the key to the nested state dict
    # e.g. we store model: {layers: {0: {self_attn: {q_proj: jax array}}}}
    # instead of model: {layers: {0: {self_attn: {q_proj: {weight: jax array}}}}}
    # The reason why is the last part of the key is used to determine the
    # type of transpose used in transpose_keys. Since normally the weights would all end
    # in .weight, this would not be informative about how to transpose the weight.
    state_dict = levanter_model.to_state_dict()
    nested_state_dict = {}
    for key, value in state_dict.items():
        split_key = key.split(".")
        current = nested_state_dict
        split_key_without_weight = split_key[:-1]
        for part in split_key_without_weight[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # for q, k, v projections, we need to pad the 2nd dimension to next multiple of 128
        # vLLM expects the weights to be padded to the next multiple of 128. I assume this is
        # because they want to use Pallas kernels which have this requirement.
        if "self_attn" in split_key_without_weight:
            if "q_proj" in split_key_without_weight and len(value.shape) == 4:
                kv_heads, q_heads_per_group, head_size, embed = value.shape
                value = value.reshape(kv_heads * q_heads_per_group, head_size, embed)

            if (
                "q_proj" in split_key_without_weight
                or "k_proj" in split_key_without_weight
                or "v_proj" in split_key_without_weight
            ):
                _heads, head_size, embed = value.shape
                next_multiple_of_128 = ((head_size + 127) // 128) * 128
                if head_size < next_multiple_of_128:
                    # pad 2nd dimension to 128 (e.g., (8, 64, 2048) -> (8, 128, 2048))
                    value = jnp.pad(value, ((0, 0), (0, next_multiple_of_128 - head_size), (0, 0)))
            elif "o_proj" in split_key_without_weight:
                embed, _heads, head_size = value.shape
                next_multiple_of_128 = ((head_size + 127) // 128) * 128
                if head_size < next_multiple_of_128:
                    # pad 3rd dimension to 128 (e.g., (8, 2048, 64) -> (8, 2048, 128))
                    value = jnp.pad(value, ((0, 0), (0, 0), (0, next_multiple_of_128 - head_size)))

        current[_get_nnx_key_name(split_key)] = nnx.Param(value)
    return nnx.State(nested_state_dict)


def levanter_state_dict_to_nnx_state_on_cpu(state_dict: dict) -> dict:
    with jax.default_device(jax.devices("cpu")[0]):
        nested_state_dict = {}
        for key, value in state_dict.items():
            # Convert from numpy to jax array here
            try:
                value = jax.numpy.asarray(value)
            except Exception as e:
                print(f"ConversionError converting {key} to jax array {type(value)}, {value}: {e}")

            split_key = key.split(".")
            current = nested_state_dict
            split_key_without_weight = split_key[:-1]
            for part in split_key_without_weight[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # vLLM requires weights/biases to be padded to the nearest multiple of 128 for Pallas kernels.
            if "self_attn" in split_key_without_weight:
                is_bias = split_key[-1] == "bias"

                # Flatten grouped query heads -> (Total Heads, Head Dim, [Embed]) for vLLM
                if "q_proj" in split_key_without_weight:
                    if len(value.shape) == 4:
                        # Weight: (KV, Group, HeadSize, Embed) -> (Heads, HeadSize, Embed)
                        kv_heads, q_heads_per_group, head_size, embed = value.shape
                        value = value.reshape(kv_heads * q_heads_per_group, head_size, embed)
                    elif len(value.shape) == 3 and is_bias:
                        # Bias: (KV, Group, HeadSize) -> (Heads, HeadSize)
                        kv_heads, q_heads_per_group, head_size = value.shape
                        value = value.reshape(kv_heads * q_heads_per_group, head_size)

                # Pad the head dimension (dim 1) for Q/K/V projections
                if (
                    "q_proj" in split_key_without_weight
                    or "k_proj" in split_key_without_weight
                    or "v_proj" in split_key_without_weight
                ):
                    pad_axis = 1
                    if len(value.shape) >= 2:
                        head_size = value.shape[pad_axis]
                        next_multiple_of_128 = ((head_size + 127) // 128) * 128

                        if head_size < next_multiple_of_128:
                            padding = [(0, 0)] * len(value.shape)
                            padding[pad_axis] = (0, next_multiple_of_128 - head_size)
                            value = jnp.pad(value, padding)

                # Pad o_proj weights along the head dimension (dim 2)
                elif "o_proj" in split_key_without_weight:
                    # Weight: (Embed, Heads, HeadSize). Skip bias as it is 1D (Embed,) or handled differently.
                    if not is_bias and len(value.shape) == 3:
                        embed, _heads, head_size = value.shape
                        next_multiple_of_128 = ((head_size + 127) // 128) * 128
                        if head_size < next_multiple_of_128:
                            value = jnp.pad(value, ((0, 0), (0, 0), (0, next_multiple_of_128 - head_size)))

            current[_get_nnx_key_name(split_key)] = nnx.Param(value)

        return nnx.State(nested_state_dict)
