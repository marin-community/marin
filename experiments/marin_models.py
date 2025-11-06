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
# flake8: noqa

"""
Various models and templates for Marin.
"""

marin_tokenizer = "marin-community/marin-tokenizer"
"""
The HF Hub name for the Marin tokenizer.
The Marin tokenizer is (currently) just the Llama 3 tokenizer with a custom chat template (MARIN_CHAT_TEMPLATE).
"""

# inspired by the smollm3 template and the Olmo 2 template, using llama3's special tokens
MARIN_CHAT_TEMPLATE = """
{{ bos_token }}
{%- if enable_thinking is defined -%}
  {%- if enable_thinking is sameas true -%}
    {%- set _reasoning_mode = "/think" -%}
  {%- elif enable_thinking is sameas false -%}
    {%- set _reasoning_mode = "/nothink" -%}
  {%- else -%}
    {%- set _reasoning_mode = enable_thinking -%}
  {%- endif -%}
{%- else -%}
  {%- set _reasoning_mode = none -%}
{%- endif -%}
{%- set _custom_instructions = custom_instructions | default(None, true) -%}
{%- set _xml_tools_list = xml_tools | default([], true) -%}
{%- if tools is defined and tools -%}
  {%- set _xml_tools_list = tools -%}
{%- endif -%}
{%- set _python_tools = python_tools | default([], true) -%}
{%- set _has_aux_header = (_reasoning_mode is not none) or _custom_instructions or (_xml_tools_list) or (_python_tools) -%}
{%- if _has_aux_header -%}
<|start_header_id|>system<|end_header_id|>
{%- if _reasoning_mode is not none -%}
Reasoning: {{ _reasoning_mode }}
{%- endif %}
{%- if _custom_instructions %}
{{ _custom_instructions | trim }}
{%- endif %}
{% if _xml_tools_list or _python_tools %}
{{ "\n### Tools\n" }}
You may call one or more functions to assist with the user query.
{% if _xml_tools_list %}
You are provided with function signatures within <tools> </tools> tags:

<tools>
{% for tool in _xml_tools_list %}
{{ tool | string }}{% if not loop.last %}
{% endif %}
{% endfor %}
</tools>

For each function call, pass a json object with function name and arguments within <tool_call> </tool_call> tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

{% endif %}
{% if _python_tools %}
When you send a message containing Python code between <|python_tag|> and <|eom_id|> tags, it will be executed in a stateful Jupyter notebook environment, and you will then be given the output.

You can use the following tools in your python code like regular functions:
<tools>
{% for tool in _python_tools %}
{{ tool | string }}{% if not loop.last %}
{% endif %}
{% endfor %}
</tools>
{% endif %}
{% endif %}
<|eot_id|>
{%- endif -%}
{%- for message in messages -%}
{%- if message['role'] == 'assistant' -%}
    <|start_header_id|>{{ message['role'] }}<|end_header_id|>
{% generation %}{{- message['content'] | trim }}<|eot_id|>{% endgeneration %}\n
{% else %}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>
{{ message['content'] | trim }}<|eot_id|>
{% endif %}
{%- endfor -%}
{%- if add_generation_prompt -%}
<|start_header_id|>assistant<|end_header_id|>\n{% endif -%}
""".strip()

"""
The chat template for the Marin tokenizer.

This template is used to generate the chat template for the Marin tokenizer.
It is a modified version of the Olmo 2 template.

The modifications are:
- Use the Llama 3 special tokens rather than Olmo (which doesn't use special tokens but rather just literally <|, etc.)
- Add the {% generation -%} tag stuff that makes the assistant_mask in Levanter work.

See [Levanter's documentation on Chat Templates](https://levanter.readthedocs.io/en/latest/reference/Data-Formats/#chat-format)
for more information on how this works.
"""

MARIN_CUSTOM_SPECIAL_TOKENS = {
    128002: "<|start_think|>",  # Originally "<|reserved_special_token_0|>"
    128003: "<|end_think|>",  # Originally "<|reserved_special_token_1|>"
}
