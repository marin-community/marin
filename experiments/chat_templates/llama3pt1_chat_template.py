# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

"""
Updated chat template for Llama3.1 that has {% generation %} tags inserted (required by Levanter).
NOTE (moojink): Handling None content values by, for example, changing `message['content'] | trim` to `(message['content'] or '') | trim` to handle None content values.
"""

LLAMA_3_1_CHAT_TEMPLATE = """{{- bos_token }}
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
    {%- set system_message = (messages[0]['content'] or '')|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message + builtin tools #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if builtin_tools is defined or tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{%- if builtin_tools is defined %}
    {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\n\n"}}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = (messages[0]['content'] or '')|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {%- set first_user_message = "" %}
    {%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or (message.tool_calls is defined and message.tool_calls)) %}
        {%- if message.role == 'assistant' %}
            {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}{% generation %}{{- (message['content'] or '') | trim + '<|eot_id|>' }}{% endgeneration %}
        {%- else %}
            {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ (message['content'] or '') | trim + '<|eot_id|>' }}
        {%- endif %}
    {%- elif message.tool_calls is defined and message.tool_calls %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
        {% generation %}
        {%- if message.content %}
            {{- message.content | trim }}
            {{- "\n" }}
        {%- endif %}
        {%- for raw_tool_call in message.tool_calls %}
            {%- if raw_tool_call.function is defined and raw_tool_call.function is not none %}
                {%- set tool_call = raw_tool_call.function %}
            {%- else %}
                {%- set tool_call = raw_tool_call %}
            {%- endif %}
            {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
                {{- "<|python_tag|>" + tool_call.name + ".call(" }}
                {%- for arg_name, arg_val in tool_call.arguments | items %}
                    {{- arg_name + '="' + arg_val + '"' }}
                    {%- if not loop.last %}
                        {{- ", " }}
                    {%- endif %}
                {%- endfor %}
                {{- ")" }}
            {%- else  %}
                {{- '{"name": "' + tool_call.name + '", ' }}
                {{- '"parameters": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- "}" }}
            {%- endif %}
            {%- if not loop.last %}
                {{- "\n" }}
            {%- endif %}
        {%- endfor %}
        {%- if builtin_tools is defined %}
            {#- This means we're in ipython mode #}
            {{- "<|eom_id|>" }}
        {%- else %}
            {{- "<|eot_id|>" }}
        {%- endif %}
        {% endgeneration %}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is none %}
            {{- '' }}
        {%- elif message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"""
