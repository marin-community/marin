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
Various models and templates for Marin.
"""

marin_tokenizer = "stanford-crfm/marin-tokenizer"
"""
The HF Hub name for the Marin tokenizer.
The Marin tokenizer is (currently) just the Llama 3 tokenizer with a custom chat template (MARIN_CHAT_TEMPLATE).
"""

# to be clear this is the Olmo 2 template except we use llama3's special tokens
MARIN_CHAT_TEMPLATE = """
{{ bos_token }}
{%- for message in messages -%}
  {%- set has_tool_calls = message.get('tool_calls') is not none and message.get('tool_calls') -%}
  {%- if not (message.get('role') in ['tool', 'ipython'] or has_tool_calls) -%}
    {%- if message.get('role') == 'assistant' -%}
<|start_header_id|>assistant<|end_header_id|>
{% set content = message.get('content') %}
{% if content is string %}
{% generation %}{{- content | trim }}<|eot_id|>{% endgeneration %}
{% elif content is mapping %}
{% generation %}{{- content.get('text', '') | trim }}<|eot_id|>{% endgeneration %}
{% elif content is iterable %}
{% generation %}
{%- for chunk in content -%}
  {%- if chunk.get('type') == 'text' -%}
    {{ chunk.get('text', '') | trim }}
  {%- endif -%}
{%- endfor -%}
<|eot_id|>
{% endgeneration %}
{% else %}
{% generation %}{% endgeneration %}<|eot_id|>
{% endif %}
    {%- else -%}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>
{% set content = message.get('content') %}
{% if content is string %}
{{ content | trim }}<|eot_id|>
{% elif content is mapping %}
{{ content.get('text', '') | trim }}<|eot_id|>
{% elif content is iterable %}
{%- for chunk in content -%}
  {%- if chunk.get('type') == 'text' -%}
    {{ chunk.get('text', '') | trim }}
  {%- endif -%}
{%- endfor -%}<|eot_id|>
{% else %}
<|eot_id|>
{% endif %}
    {%- endif -%}

  {%- elif message.get('role') == 'tool' -%}
    {%- set _tool_name = message.get('name') -%}
    {%- set _tool_id = message.get('tool_call_id') -%}
    {%- set _attr_name = ' name=\"' ~ _tool_name ~ '\"' if _tool_name else '' -%}
    {%- set _attr_id = ' id=\"' ~ _tool_id ~ '\"' if _tool_id else '' -%}
<|start_header_id|>tool<|end_header_id|>
<tool_response{{ _attr_name }}{{ _attr_id }}>
{%- set tool_content = message.get('content') -%}
{%- if tool_content is mapping or (tool_content is iterable and tool_content is not string) -%}
{{- tool_content | tojson }}
{%- else -%}
{{- tool_content if tool_content is not none else '' }}
{%- endif -%}
</tool_response><|eot_id|>
{{- "\n" -}}
  {%- elif message.get('role') == 'ipython' -%}
<|start_header_id|>ipython<|end_header_id|>
{% set ipy_content = message.get('content') %}
{% if ipy_content is string %}
{{- { "output": ipy_content } | tojson -}}
{% elif ipy_content is iterable %}
{%- for chunk in ipy_content -%}
  {%- if chunk.get('type') == 'text' -%}
    {{- { "output": chunk.get('text', '') } | tojson -}}
  {%- endif -%}
{%- endfor -%}
{% else %}
{{- { "output": ipy_content } | tojson -}}
{% endif %}
<|eot_id|>
{% elif has_tool_calls -%}
    {%- if message.tool_calls|length != 1 -%}
      {{- raise_exception("This template expects exactly one tool call per assistant turn.") -}}
    {%- endif -%}
    {%- set tool_call = message.tool_calls[0].function -%}
<|start_header_id|>assistant<|end_header_id|>
{% generation %}
{{- '{\"name\": \"' + tool_call.name + '\", \"arguments\": ' -}}
{{- tool_call.arguments | tojson -}}
{{- \"}\" -}}<|eot_id|>
{% endgeneration %}
  {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
<|start_header_id|>assistant<|end_header_id|>
{% endif -%}
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
