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

from flax import nnx
import jax.numpy as jnp


def levanter_llama_to_vllm_mapping():
    return {
        "model.lm_head": ("model.lm_head", (None, "model")),
        "model.embed_tokens": (
            "model.embed.embedding",
            ("model", None),
        ),
        "model.layers.*.input_layernorm": (
            "model.layers.*.input_layernorm.scale",
            (None,),
        ),
        "model.layers.*.mlp.down_proj": (
            "model.layers.*.mlp.down_proj.kernel",
            ("model", None),
        ),
        "model.layers.*.mlp.gate_proj": (
            "model.layers.*.mlp.gate_proj.kernel",
            (None, "model"),
        ),
        "model.layers.*.mlp.up_proj": (
            "model.layers.*.mlp.up_proj.kernel",
            (None, "model"),
        ),
        "model.layers.*.post_attention_layernorm": (
            "model.layers.*.post_attention_layernorm.scale",
            (None,),
        ),
        "model.layers.*.self_attn.k_proj": (
            "model.layers.*.self_attn.k_proj.kernel",
            (None, "model", None),
        ),
        "model.layers.*.self_attn.o_proj": (
            "model.layers.*.self_attn.o_proj.kernel",
            ("model", None, None),
        ),
        "model.layers.*.self_attn.q_proj": (
            "model.layers.*.self_attn.q_proj.kernel",
            (None, "model", None),
        ),
        "model.layers.*.self_attn.v_proj": (
            "model.layers.*.self_attn.v_proj.kernel",
            (None, "model", None),
        ),
        "model.norm": ("model.norm.scale", (None,)),
    }


llama_transpose_keys = {
    "lm_head": (1, 0),
    "gate_proj": (1, 0),
    "up_proj": (1, 0),
    "down_proj": (1, 0),
    "q_proj": (2, 0, 1),
    "k_proj": (2, 0, 1),
    "v_proj": (2, 0, 1),
    "o_proj": (1, 2, 0),
}

MODEL_MAPPINGS: dict[str, dict[str, tuple[str, tuple[str, ...]]]] = {
    "meta-llama/Llama-3.2-1B-Instruct": levanter_llama_to_vllm_mapping(),
    "meta-llama/Llama-3.2-3B-Instruct": levanter_llama_to_vllm_mapping(),
}

MODEL_TRANSPOSE_KEYS: dict[str, tuple[int, ...]] = {
    "meta-llama/Llama-3.2-1B-Instruct": llama_transpose_keys,
    "meta-llama/Llama-3.2-3B-Instruct": llama_transpose_keys,
}


def levanter_to_nnx_state(levanter_model):
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
                    # pad 2nd dimension to 128 (e.g., (8, 64, 2048) -> (8, 128, 2048))
                    value = jnp.pad(value, ((0, 0), (0, 0), (0, next_multiple_of_128 - head_size)))

        current[split_key_without_weight[-1]] = nnx.Param(value)
    return nnx.State(nested_state_dict)
