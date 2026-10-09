# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

"""Chat templates for PivotRL SFT baselines that supervise only a pivot's final, expert turn.

Each template renders exactly what the policy's own tokenizer template renders when vLLM serves it,
so SFT examples have the prompts pass rates and RL sample from. Levanter masks loss to
``{% generation %}`` spans; :func:`final_turn_only` keeps that span on the last message alone, so
the expert turns already in a pivot's prompt are context, not targets.
"""

import re

from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE

_GENERATION = re.compile(
    r"(?P<open>\{%-?\s*generation\s*-?%\})(?P<body>.*?)(?P<close>\{%-?\s*endgeneration\s*-?%\})", re.DOTALL
)


def final_turn_only(template: str) -> str:
    """Restrict a template's one ``{% generation %}`` block to the last message of the conversation.

    The block must sit directly in the template's loop over messages, so ``loop.last`` names the
    final message. Earlier assistant turns render the same text without a generation span.
    """
    blocks = list(_GENERATION.finditer(template))
    if len(blocks) != 1:
        raise ValueError(f"expected one generation block, found {len(blocks)}")
    (block,) = blocks
    body = block["body"]
    supervised = f"{{%- if loop.last %}}{block['open']}{body}{block['close']}"
    return template[: block.start()] + f"{supervised}{{%- else %}}{body}{{%- endif %}}" + template[block.end() :]


# Qwen/Qwen3-0.6B@c1899de's template, with the assistant header moved out of each branch so one
# generation block covers the turn: optional thinking, content, tool calls, and <|im_end|>.
QWEN3_CHAT_TEMPLATE = r"""{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- set think = loop.index0 > ns.last_query_index and (loop.last or reasoning_content) %}
        {{- '<|im_start|>' + message.role + '\n' }}
        {%- generation %}
            {%- if think %}
                {{- '<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- content }}
            {%- endif %}
            {%- if message.tool_calls %}
                {%- for tool_call in message.tool_calls %}
                    {%- if (loop.first and content) or (not loop.first) %}
                        {{- '\n' }}
                    {%- endif %}
                    {%- if tool_call.function %}
                        {%- set tool_call = tool_call.function %}
                    {%- endif %}
                    {{- '<tool_call>\n{"name": "' }}
                    {{- tool_call.name }}
                    {{- '", "arguments": ' }}
                    {%- if tool_call.arguments is string %}
                        {{- tool_call.arguments }}
                    {%- else %}
                        {{- tool_call.arguments | tojson }}
                    {%- endif %}
                    {{- '}\n</tool_call>' }}
                {%- endfor %}
            {%- endif %}
            {{- '<|im_end|>' }}
        {%- endgeneration %}
        {{- '\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}"""

# The Grug Datakit SFT exports carry MARIN_CHAT_TEMPLATE; examples set its thinking mode explicitly.
GRUG_FINAL_TURN_TEMPLATE = final_turn_only(MARIN_CHAT_TEMPLATE)
QWEN3_FINAL_TURN_TEMPLATE = final_turn_only(QWEN3_CHAT_TEMPLATE)
