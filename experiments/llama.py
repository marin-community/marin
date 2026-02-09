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

"""
Specifies a sequence of Llama 3 models from small to large.
"""

from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.llama import LlamaConfig
from levanter.utils.activation import ActivationFunctionEnum

from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig

from levanter.data.text import ChatLmDatasetFormat

llama3_tokenizer = "meta-llama/Meta-Llama-3.1-8B"
llama3_tokenizer_vocab_size = 128_256
llama3_instruct_tokenizer = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Llama3 instruct trainable chat template
# Slight modification of https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/tokenizer_config.json
# to add {% generation %} so we can create the assistant_mask
llama3_instruct_trainable_chat_template = """{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- set date_string = "26 Jul 2024" %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message + builtin tools #}
{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}
{%- if builtin_tools is defined or tools is not none %}
    {{- "Environment: ipython\\n" }}
{%- endif %}
{%- if builtin_tools is defined %}
    {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\\n\\n"}}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\\n" }}
{{- "Today Date: " + date_string + "\\n\\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\\n\\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\\n\\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
    {%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|}\\n\\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\\n\\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\\n\\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>" }}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {%- if message.role == "assistant" %}
            {{- "<|start_header_id|>assistant<|end_header_id|>\\n\\n" }}
            {%- generation %}
            {{- message['content'] | trim }}
            {%- endgeneration %}
            {{- "<|eot_id|>" }}
        {%- else %}
            {{- "<|start_header_id|>" + message['role'] + "<|end_header_id|>\\n\\n" + message['content'] | trim + "<|eot_id|>" }}
        {%- endif %}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
            {{- "<|start_header_id|>assistant<|end_header_id|>\\n\\n" -}}
            {{- "<|python_tag|>" + tool_call.name + ".call(" }}
            {%- for arg_name, arg_val in tool_call.arguments | items %}
                {{- arg_name + '="' + arg_val + '"' }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- endif %}
            {%- endfor %}
            {{- ")" }}
        {%- else  %}
            {{- "<|start_header_id|>assistant<|end_header_id|>\\n\\n" -}}
            {{- '{"name": "' + tool_call.name + '", ' }}
            {{- '"parameters": ' }}
            {{- tool_call.arguments | tojson }}
            {{- "}" }}
        {%- endif %}
        {%- if builtin_tools is defined %}
            {{- "<|eom_id|>" }}
        {%- else %}
            {{- "<|eot_id|>" }}
        {%- endif %}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- "<|start_header_id|>assistant<|end_header_id|>\\n\\n" }}
{%- endif %}"""  # noqa: E501

# Chat format compatible with the llama3 instruct tokenizer and Levanter's chat format
llama3_instruct_chat_format = ChatLmDatasetFormat(
    messages_field="messages",
    chat_template=llama3_instruct_trainable_chat_template,
    pack=True,
    mask_user_turns=True,
)

llama_nano = LlamaConfig(
    max_seq_len=512,
    hidden_dim=32,
    intermediate_dim=128,
    num_heads=2,
    num_kv_heads=2,
    num_layers=2,
)

llama_30m = LlamaConfig(
    max_seq_len=1024,
    hidden_dim=128,
    intermediate_dim=448,
    num_heads=2,
    num_kv_heads=2,
    num_layers=4,
)

llama_50m = LlamaConfig(
    max_seq_len=1024,
    hidden_dim=192,
    intermediate_dim=448,
    num_heads=2,
    num_kv_heads=2,
    num_layers=4,
)


llama_75m = LlamaConfig(
    max_seq_len=1024,
    hidden_dim=256,
    intermediate_dim=896,
    num_heads=4,
    num_kv_heads=4,
    num_layers=8,
)

llama_150m = LlamaConfig(
    max_seq_len=1024,
    hidden_dim=512,
    intermediate_dim=1792,
    num_heads=8,
    num_kv_heads=8,
    num_layers=6,
)

llama_300m = LlamaConfig(
    max_seq_len=1024,
    hidden_dim=768,
    intermediate_dim=2688,
    num_heads=12,
    num_kv_heads=12,
    num_layers=12,
)

llama_600m = LlamaConfig(
    max_seq_len=1024,
    hidden_dim=1024,
    intermediate_dim=3584,
    num_heads=16,
    num_kv_heads=8,
    num_layers=24,
)

llama_1_4b = LlamaConfig(
    max_seq_len=4096,
    hidden_dim=2048,
    intermediate_dim=7168,
    num_heads=16,
    num_kv_heads=8,
    num_layers=16,
)

llama_1_9b = LlamaConfig(
    max_seq_len=4096,
    hidden_dim=2048,
    intermediate_dim=7168,
    num_heads=16,
    num_kv_heads=8,
    num_layers=24,
)

# Llama-3.2-1B
llama_3_2_1b = LlamaConfig(
    max_seq_len=4096,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=16,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
)

# Llama-3.1-8B
llama_3_1_8b_tokenizer = "meta-llama/Llama-3.1-8B"
llama_3_1_8b = LlamaConfig(
    # Matching defaults in https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
    max_seq_len=4096,
    hidden_dim=4096,
    intermediate_dim=14336,
    num_heads=32,
    num_kv_heads=8,
    num_layers=32,
    activation_function=ActivationFunctionEnum.silu,
    initializer_range=0.02,
    layer_norm_epsilon=1e-5,
    reference_checkpoint="meta-llama/Llama-3.1-8B",
    rope=Llama3RotaryEmbeddingsConfig(
        theta=500000.0,
        factor=8.0,
        high_freq_factor=4.0,
        low_freq_factor=1.0,
        original_max_position_embeddings=8192,
    ),
    tie_word_embeddings=False,
)

# Llama-3.1-8B-Instruct
llama_3_1_8b_instruct_tokenizer = (
    "meta-llama/Llama-3.1-8B-Instruct"  # NOTE: Instruct and base eos_token_id values are different
)
llama_3_1_8b_instruct = LlamaConfig(
    # Matching defaults in https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/config.json
    max_seq_len=4096,
    hidden_dim=4096,
    intermediate_dim=14336,
    num_heads=32,
    num_kv_heads=8,
    num_layers=32,
    activation_function=ActivationFunctionEnum.silu,
    initializer_range=0.02,
    layer_norm_epsilon=1e-5,
    reference_checkpoint="meta-llama/Llama-3.1-8B-Instruct",
    rope=Llama3RotaryEmbeddingsConfig(
        theta=500000.0,
        factor=8.0,
        high_freq_factor=4.0,
        low_freq_factor=1.0,
        original_max_position_embeddings=8192,
    ),
    tie_word_embeddings=False,
)

llama_3_5b = LlamaConfig(
    max_seq_len=4096,
    hidden_dim=2560,
    intermediate_dim=8960,
    num_heads=20,
    num_kv_heads=10,
    num_layers=32,
)

llama_8b = LlamaConfig(
    max_seq_len=4096,
    hidden_dim=4096,
    intermediate_dim=14336,
    num_heads=32,
    num_kv_heads=8,
    num_layers=32,
    rope=Llama3RotaryEmbeddingsConfig(),
)


llama_8b_old_rotary = LlamaConfig(
    max_seq_len=4096,
    hidden_dim=4096,
    intermediate_dim=14336,
    num_heads=32,
    num_kv_heads=8,
    num_layers=32,
    # Levanter defaults to Llama2 rotary
    # rope=Llama3RotaryEmbeddingsConfig(),
)


llama_13b = LlamaConfig(
    max_seq_len=4096,
    hidden_dim=5120,
    intermediate_dim=13824,
    num_heads=40,
    num_kv_heads=8,
    num_layers=40,
    rope=Llama3RotaryEmbeddingsConfig(),
)


# With Llama 3 tokenizer, this is 24B
llama_24b = LlamaConfig(
    max_seq_len=4096,
    hidden_dim=6144,
    intermediate_dim=16384,
    num_heads=48,
    num_kv_heads=16,
    num_layers=56,
    rope=Llama3RotaryEmbeddingsConfig(),
)


# same as olmo 32b
llama_32b = LlamaConfig(
    max_seq_len=4096,
    hidden_dim=5120,
    intermediate_dim=27648,
    num_heads=40,
    num_kv_heads=8,
    num_layers=64,
    rope=Llama3RotaryEmbeddingsConfig(),
)


llama_56b = LlamaConfig(
    max_seq_len=4096,
    hidden_dim=8192,
    intermediate_dim=28672,
    num_heads=64,
    num_kv_heads=8,
    num_layers=64,
    rope=Llama3RotaryEmbeddingsConfig(),
)


llama_70b = LlamaConfig(
    max_seq_len=4096,
    hidden_dim=8192,
    intermediate_dim=28672,
    num_heads=64,
    num_kv_heads=8,
    num_layers=80,
    rope=Llama3RotaryEmbeddingsConfig(),
)


llama_150m_train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu("v4-32"),
    train_batch_size=512,
    num_train_steps=20000,  # 1024 * 1024 * 20000 = 20B tokens
    learning_rate=3e-3,
    weight_decay=0.1,
)
# (18B is way overtrained, but...)

llama_300m_train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu("v4-64"),
    train_batch_size=1024,
    num_train_steps=18000,  # 1024 * 1024 * 18000 = 18B tokens
    learning_rate=3e-3,
    weight_decay=0.1,
)
# (18B is way overtrained, but...)

llama_1_4b_train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu("v4-128"),
    train_batch_size=1024,
    num_train_steps=10000,  # 4096 * 1024 * 10000 = 42B tokens
    learning_rate=3e-4,
    weight_decay=0.1,
)

llama_8b_train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu("v4-128", slice_count=4),
    train_batch_size=1024,
    num_train_steps=40000,  # 4096 * 1024 * 40000 = 167B tokens
    # these hypers from Table 12 in https://arxiv.org/html/2406.11794v1#A6
    learning_rate=2e-3,
    weight_decay=0.05,
)


def compute_num_parameters(config: LlamaConfig, vocab_size: int) -> int:
    head_size = config.hidden_dim // config.num_heads
    q_params = config.num_heads * head_size * config.hidden_dim
    k_params = config.num_kv_heads * head_size * config.hidden_dim
    v_params = config.num_kv_heads * head_size * config.hidden_dim
    o_params = config.num_heads * head_size * config.hidden_dim
    attention_params = q_params + k_params + v_params + o_params

    layer_norm_params = 2 * config.hidden_dim

    gate_params = config.hidden_dim * config.intermediate_dim
    up_params = config.hidden_dim * config.intermediate_dim
    down_params = config.intermediate_dim * config.hidden_dim
    mlp_params = gate_params + up_params + down_params

    nonembedding_params = config.num_layers * (attention_params + mlp_params + layer_norm_params)
    embedding_params = 2 * vocab_size * config.hidden_dim

    return nonembedding_params + embedding_params


# For scaling laws
scaling_llamas = [llama_30m, llama_50m, llama_150m, llama_300m, llama_600m, llama_1_4b, llama_1_9b, llama_3_5b, llama_8b]


if __name__ == "__main__":
    for llama in scaling_llamas:
        print(f"{compute_num_parameters(llama, llama3_tokenizer_vocab_size):,}")
