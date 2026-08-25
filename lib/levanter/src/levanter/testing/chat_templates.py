# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

MULTI_TOOL_TEMPLATE = """{{ bos_token }}
{%- for message in messages -%}
  {%- if message['role'] == 'assistant' -%}
<|start_header_id|>assistant<|end_header_id|>
{% generation %}{{ message['content'] | trim }}
{%- for tool_call in message.get('tool_calls', []) -%}
  {%- set call = tool_call['function'] -%}
{{ '{"name": "' + call['name'] + '", "arguments": ' }}{{ call['arguments'] | tojson }}{{ '}' }}
{%- endfor -%}
<|eot_id|>{% endgeneration %}
  {%- elif message['role'] == 'tool' -%}
<|start_header_id|>tool<|end_header_id|>
{{ message['content'] | tojson }}<|eot_id|>
  {%- else -%}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>
{{ message['content'] | trim }}<|eot_id|>
  {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
<|start_header_id|>assistant<|end_header_id|>
{%- endif -%}
"""
