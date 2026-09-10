# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E501 -- Inference prompt text preserves its original line breaks.

"""Shared Marin chat template for tokenizer export and Datakit rendering.

The template must remain self-contained so Hugging Face and vLLM can load it
from tokenizer artifacts without importing Marin. Generation blocks identify
assistant output without changing the rendered text.
"""

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
{{ tool if tool is string else tool | tojson }}{{ "\n" }}
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
{{ tool if tool is string else tool | tojson }}{{ "\n" }}
{% endfor %}
</tools>
{% endif %}
{% endif %}
<|eot_id|>
{%- endif -%}
{%- macro text(content) -%}
  {%- if content is string -%}
    {{- content | trim -}}
  {%- elif content is mapping -%}
    {{- content.get('text', '') | trim -}}
  {%- elif content is iterable -%}
    {%- for chunk in content if chunk.get('type') == 'text' -%}
      {{- chunk.text | trim -}}
    {%- endfor -%}
  {%- endif -%}
{%- endmacro -%}

{%- set tool_names = namespace(by_id={}) -%}
{%- macro tool_calls(calls) -%}
  {%- for call in calls -%}
    {%- set function = call.function -%}
    {%- if call.get('id') -%}
      {%- set tool_names.by_id = dict(tool_names.by_id, **{call.id: function.name}) -%}
    {%- endif -%}
    {{- '<tool_call>\n{"name": ' -}}
    {{- function.name | tojson -}}
    {{- ', "arguments": ' -}}
    {{- function.arguments if function.arguments is string else function.arguments | tojson -}}
    {{- '}\n</tool_call>' -}}
  {%- endfor -%}
{%- endmacro -%}

{%- for message in messages -%}
  {{- '<|start_header_id|>' ~ message.role ~ '<|end_header_id|>\n' -}}
  {%- if message.role == 'assistant' -%}
    {% generation %}
    {%- if message.get('reasoning_content') -%}
      {{- '<|start_think|>' ~ message.reasoning_content ~ '<|end_think|>' -}}
    {%- endif -%}
    {{- text(message.get('content')) -}}
    {{- tool_calls(message.get('tool_calls') or []) -}}
    {{- '<|eot_id|>' -}}
    {% endgeneration %}
  {%- elif message.role == 'tool' -%}
    {{- '<tool_response' -}}
    {%- set name = message.get('name') or tool_names.by_id.get(message.get('tool_call_id')) -%}
    {%- if name -%}{{- ' name="' ~ name ~ '"' -}}{%- endif -%}
    {{- '>' -}}
    {%- set content = message.get('content') -%}
    {{- content if content is string else content | tojson if content is not none else '' -}}
    {{- '</tool_response><|eot_id|>\n' -}}
  {%- elif message.role == 'ipython' -%}
    {%- set content = message.get('content') -%}
    {%- if content is iterable and content is not string and content is not mapping -%}
      {%- for chunk in content if chunk.get('type') == 'text' -%}
        {{- {"output": chunk.text} | tojson -}}
      {%- endfor -%}
    {%- else -%}
      {{- {"output": content} | tojson -}}
    {%- endif -%}
    {{- '<|eot_id|>\n' -}}
  {%- else -%}
    {{- text(message.get('content')) ~ '<|eot_id|>\n' -}}
  {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
  {{- '<|start_header_id|>assistant<|end_header_id|>\n' -}}
{%- endif -%}
""".strip()
