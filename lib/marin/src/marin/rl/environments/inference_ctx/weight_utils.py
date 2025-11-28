# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
from flax import nnx
import jax.numpy as jnp
from levanter.models.lm_model import LmHeadModel
import numpy as np

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
            if "q_proj" in split_key_without_weight:
                kv_heads, q_heads_per_group, head_size, embed = value.shape
                value = value.reshape(kv_heads * q_heads_per_group, head_size, embed)

            if (
                "q_proj" in split_key_without_weight
                or "k_proj" in split_key_without_weight
                or "v_proj" in split_key_without_weight
            ):
                heads, head_size, embed = value.shape
                next_multiple_of_128 = ((head_size + 127) // 128) * 128
                if head_size < next_multiple_of_128:
                    # pad 2nd dimension to 128 (e.g., (8, 64, 2048) -> (8, 128, 2048))
                    value = jnp.pad(value, ((0, 0), (0, next_multiple_of_128 - head_size), (0, 0)))
            elif "o_proj" in split_key_without_weight:
                embed, heads, head_size = value.shape
                next_multiple_of_128 = ((head_size + 127) // 128) * 128
                if head_size < next_multiple_of_128:
                    # pad 3rd dimension to 128 (e.g., (8, 2048, 64) -> (8, 2048, 128))
                    value = jnp.pad(value, ((0, 0), (0, 0), (0, next_multiple_of_128 - head_size)))

        current[split_key_without_weight[-1]] = nnx.Param(value)
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

            # for q, k, v projections, we need to pad the 2nd dimension to next multiple of 128
            # vLLM expects the weights to be padded to the next multiple of 128. I assume this is
            # because they want to use Pallas kernels which have this requirement.
            if "self_attn" in split_key_without_weight:
                if "q_proj" in split_key_without_weight:
                    kv_heads, q_heads_per_group, head_size, embed = value.shape
                    value = value.reshape(kv_heads * q_heads_per_group, head_size, embed)

                if (
                    "q_proj" in split_key_without_weight
                    or "k_proj" in split_key_without_weight
                    or "v_proj" in split_key_without_weight
                ):
                    heads, head_size, embed = value.shape
                    next_multiple_of_128 = ((head_size + 127) // 128) * 128
                    if head_size < next_multiple_of_128:
                        # pad 2nd dimension to 128 (e.g., (8, 64, 2048) -> (8, 128, 2048))
                        value = jnp.pad(value, ((0, 0), (0, next_multiple_of_128 - head_size), (0, 0)))
                elif "o_proj" in split_key_without_weight:
                    embed, heads, head_size = value.shape
                    next_multiple_of_128 = ((head_size + 127) // 128) * 128
                    if head_size < next_multiple_of_128:
                        # pad 3rd dimension to 128 (e.g., (8, 2048, 64) -> (8, 2048, 128))
                        value = jnp.pad(value, ((0, 0), (0, 0), (0, next_multiple_of_128 - head_size)))

            current[split_key_without_weight[-1]] = nnx.Param(value)

        return nnx.State(nested_state_dict)
