# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

"""
Delphi v0 chat template ported into marin's / Levanter's chat-template mechanism
(pattern = experiments/chat_templates/{llama3pt1,qwen2pt5_instruct}_chat_template.py).

WHAT THIS IS
------------
The `delphi` chat template — the exact Llama-3-family think/tool token protocol the Delphi
models are trained to emit — ported from the LLaMA-Factory fork so it can be consumed by
Levanter's `ChatLmDatasetFormat` (lib/levanter/src/levanter/data/text/formats.py:158).

Parity source (byte-exact ground truth):
    OpenThoughts-Agent/sft/llamafactory/src/llamafactory/data/template.py
        :473  DELPHI_V0_JINJA_TEMPLATE  (== chat_templates/delphi_v0.jinja2, 4109 bytes)

The ONLY delta vs. that source is the insertion of Levanter's `{% generation %}` /
`{% endgeneration %}` markers around the assistant-generated span (header EXCLUDED,
think-block + content + tool-calls + `<|eot_id|>` INCLUDED). Those markers are REQUIRED by
Levanter for completions-only masking: `ChatProcessor` (formats.py:283) hard-errors if
`mask_user_turns=True` and the template has no `{%generation%}` block, and it derives the
assistant-token loss mask from HF's `return_assistant_tokens_mask` on exactly that span.
This mirrors LF's ReasoningTemplate, which supervises only `{{content}}<|eot_id|>` (the
assistant header is prompt-side) — see the llama3pt1 marin port for the identical placement.

TOKEN PROTOCOL (Llama-3 tokenizer, vocab 128256; verified against the shipped Delphi
tokenizers laion/delphi-{3e18,1e22}-* on 2026-07-09):
    NATIVE Llama-3 (resolve to single ids in the raw checkpoint tokenizer):
        <|begin_of_text|>    = 128000   (bos)
        <|start_header_id|>  = 128006
        <|end_header_id|>    = 128007
        <|eot_id|>           = 128009   (turn terminator; the chat "eos")
    REPURPOSED reserved slots (canonical strings; single ids ONLY after
    sft/delphi/prepare_delphi_tokenizer.py renames the reserved_special_token_* slots —
    see CONFIG_NOTES.md "tokenizer prerequisite"):
        <|start_think|>      = 128002   (was <|reserved_special_token_0|>)
        <|end_think|>        = 128003   (was <|reserved_special_token_1|>)
        <|tool_call|>        = 128005   (was <|reserved_special_token_2|>)
        <|tool_call_end|>    = 128011   (was <|reserved_special_token_3|>)
        <|tool_result|>      = 128012   (was <|reserved_special_token_4|>)
        <|tool_result_end|>  = 128013   (was <|reserved_special_token_5|>)
    ⚠ ChatML <|im_start|> / <|im_end|> are ABSENT from this tokenizer (they fragment into
    ~6 bytes each) — this is a Llama-3 template, NOT ChatML. Do not "qwen"-ify it.

The template keeps the think/tool tokens INLINE inside the generation span, so they are
SUPERVISED (trained), exactly as in LF. For the magpie+warmup SFT mix specifically: magpie
turns are plain (no think); the `delphi_warmup` slice
(laion/llama-nemotron-science-reasoning-on-le3000tok-100k-canonical-think) already carries
the literal `<|start_think|>…<|end_think|>` tokens INSIDE assistant `content` (see
sft/delphi/canonicalize_warmup_think.py), so the template's reasoning-extraction branches
(reasoning_content / inline <think>) are inert for this data and content is emitted verbatim
— identical to the LF path.
"""

# The reserved Llama-3 slots repurposed as single-id think/tool tokens (id -> canonical string).
# The raw laion/delphi-* checkpoints ship these as ``<|reserved_special_token_N|>``; the SFT
# checkpoint preparation renames them so the canonical strings above tokenize to one id each and
# reinitializes their embedding rows. Native Llama-3 control tokens (bos/eot/header ids) are
# already single ids and are left untouched. Verified against laion/delphi-{3e18,1e22}-* on
# 2026-07-09; see the TOKEN PROTOCOL block above.
DELPHI_RESERVED_TOKEN_RENAMES = {
    128002: "<|start_think|>",  # was <|reserved_special_token_0|>
    128003: "<|end_think|>",  # was <|reserved_special_token_1|>
    128005: "<|tool_call|>",  # was <|reserved_special_token_2|>
    128011: "<|tool_call_end|>",  # was <|reserved_special_token_3|>
    128012: "<|tool_result|>",  # was <|reserved_special_token_4|>
    128013: "<|tool_result_end|>",  # was <|reserved_special_token_5|>
}

DELPHI_V0_CHAT_TEMPLATE = r"""{#- ===================================================================== -#}
{#- Delphi v0 chat template  (Llama-3 tokenizer; static; MVP)              -#}
{#- Ported for Levanter: {% generation %} wraps the supervised assistant   -#}
{#- span (think + content + tool_calls + <|eot_id|>); header is prompt.    -#}
{#- ===================================================================== -#}
{{- bos_token }}
{%- if messages[0].role == 'system' %}
    {%- set system_message = messages[0].content %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set system_message = '' %}
    {%- set loop_messages = messages %}
{%- endif %}
{{- '<|start_header_id|>system<|end_header_id|>\n\n' }}
{%- if system_message %}
    {{- system_message }}
{%- endif %}
{%- if tools is defined and tools %}
    {%- if system_message %}{{- '\n\n' }}{%- endif %}
    {{- '# Tools\n\nYou may call one or more of the following functions. Emit each call as a JSON object {"name": ..., "arguments": ...} between <|tool_call|> and <|tool_call_end|>:\n' }}
    {%- for tool in tools %}
        {{- '\n' }}{{- tool | tojson }}
    {%- endfor %}
{%- endif %}
{{- '<|eot_id|>' }}
{%- for message in loop_messages %}
    {%- set content = message.content if message.content is string else '' %}
    {%- if message.role == 'user' %}
        {{- '<|start_header_id|>user<|end_header_id|>\n\n' + content + '<|eot_id|>' }}
    {%- elif message.role == 'assistant' %}
        {%- set reasoning = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning = message.reasoning_content %}
        {%- elif '</think>' in content %}
            {%- set reasoning = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
            {%- set content = content.split('</think>')[-1].lstrip('\n') %}
        {%- endif %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% generation %}
        {%- if reasoning %}
            {{- '<|start_think|>\n' + reasoning.strip('\n') + '\n<|end_think|>\n\n' }}
        {%- endif %}
        {{- content }}
        {%- if message.tool_calls %}
            {%- for tc in message.tool_calls %}
                {%- set fn = tc.function if tc.function is defined else tc %}
                {{- '\n<|tool_call|>\n{"name": "' + fn.name + '", "arguments": ' }}
                {%- if fn.arguments is string %}{{- fn.arguments }}{%- else %}{{- fn.arguments | tojson }}{%- endif %}
                {{- '}\n<|tool_call_end|>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|eot_id|>' }}{% endgeneration %}
    {%- elif message.role == 'tool' %}
        {{- '<|start_header_id|>tool<|end_header_id|>\n\n<|tool_result|>\n' + content + '\n<|tool_result_end|><|eot_id|>' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<|start_think|>\n\n<|end_think|>\n\n' }}
    {%- endif %}
{%- endif %}
"""
