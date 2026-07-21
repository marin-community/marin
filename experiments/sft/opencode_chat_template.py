# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

"""Opencode tools-aware chat template + Levanter chat-format wiring for agentic (tool-calling) SFT.

WHAT THIS IS
------------
``OPENCODE_TOOLS_CHAT_TEMPLATE`` is the Qwen3 tools-aware chat template (the ``{% if tools %}`` +
``<tools>`` system block, structured ``<tool_call>``/``"arguments"``, ``<tool_response>`` framing,
and inline ``<think>``) with Levanter's ``{% generation %}`` / ``{% endgeneration %}`` markers
inserted around the assistant-generated span. It is the checked-in resource for the opencode
agentic-SFT path so a run can supply it to ``ChatLmDatasetFormat`` instead of relying on a
tokenizer's built-in template. Pattern mirrors ``experiments/sft/delphi_chat_template.py``.

The ONLY delta vs. the base Qwen3-30B-A3B-Thinking tools-aware template (``tokenizer.chat_template``,
4049 chars) is the ``{% generation %}`` insertion: it wraps the assistant think-block + content +
tool-calls + ``<|im_end|>`` (the assistant header is EXCLUDED, i.e. prompt-side). Those markers are
REQUIRED by Levanter for completions-only masking -- ``ChatProcessor`` hard-errors if
``mask_user_turns=True`` and the template has no ``{% generation %}`` block
(lib/levanter/src/levanter/data/text/formats.py) -- and are stripped before tokenization, so the
rendered text (hence ``input_ids``) is byte-identical to the base template.

PARITY (byte-exact ground truth)
--------------------------------
Validated against Axolotl's ``ChatTemplateStrategy`` (the default diff-masking path) on the
``Qwen/Qwen3-30B-A3B-Thinking-2507`` tokenizer over 60 real rows of the densemoe SFT dataset
``laion/nemotron-code-oracle-opencode-sft-serveparity``:

  * ``input_ids`` BYTE-EXACT 60/60 -- incl. tool-schema ``| tojson`` serialization, tool-call
    ``arguments`` serialization, ``role: tool`` / ``<tool_response>`` framing, ``<think>``, special
    tokens and BOS/EOS (both encode with ``add_special_tokens=False``; specials come only from the
    template -- no double BOS).
  * loss mask is NOT bit-identical, but the residual is an Axolotl artifact, not a Levanter
    deficiency: Axolotl masks via ``find_turn``, a runtime diff of the real render vs the turn's
    content replaced by a sentinel, so its span boundaries jitter +/-~3 tokens per assistant turn
    (0.38% of supervised tokens -- it trains a few next-turn ``<|im_start|>user`` header tokens and
    skips a few leading assistant tokens). Levanter's static ``{% generation %}`` mask is the exact,
    intended assistant-only signal (think + content + tool_calls + ``<|im_end|>``; everything else
    ``-100``) and provably cannot reproduce a runtime-diff boundary -- it is the cleaner realization
    of the documented intent.

DATA WIRING (the two data-shape adaptations this template needs)
----------------------------------------------------------------
Levanter's ``ChatLmDatasetFormat`` reads OpenAI ``messages`` and, for template kwargs, a
``chat_template_kwargs`` column. It has no ``tools`` column wiring. So the upstream conversation
transform (see ``marin.transform.conversation.adapters.tools_column_to_chat_template_kwargs``) must:

  1. relocate the per-row ``tools`` schema list into a ``chat_template_kwargs`` column
     (``{"tools": [...]}``) so the ``{% if tools %}`` / ``<tools>`` block renders per row; and
  2. parse assistant ``tool_calls.function.arguments`` from a JSON string to a dict before templating
     (done by ``transform_conversation._normalize_tool_structures``) so the template's
     ``arguments | tojson`` path runs identically to Axolotl.

EXAMPLE
-------
::

    from experiments.sft.opencode_chat_template import opencode_chat_lm_format

    fmt = opencode_chat_lm_format()          # ChatLmDatasetFormat, pack=True, mask_user_turns=True
    # ... plug `fmt` into your LmDataConfig component alongside the Qwen3 tokenizer, over a dataset
    # produced with `extra_metadata_fn=tools_column_to_chat_template_kwargs` (see that helper).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from levanter.data.text.formats import ChatLmDatasetFormat

# The base Qwen3 tools-aware template with ``{% generation %}`` wrapping the supervised assistant span
# (think + content + tool_calls + ``<|im_end|>``; header EXCLUDED). Byte-exact input_ids vs Axolotl.
OPENCODE_TOOLS_CHAT_TEMPLATE: str = r"""{%- if tools %}
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
        {%- if loop.index0 > ns.last_query_index and (loop.last or (not loop.last and reasoning_content)) %}
            {{- '<|im_start|>' + message.role + '\n<think>' }}{% generation %}{{- '\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
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
            {{- '<|im_end|>' }}{% endgeneration %}{{- '\n' }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' }}{% generation %}{{- content }}
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
            {{- '<|im_end|>' }}{% endgeneration %}{{- '\n' }}
        {%- endif %}
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
    {{- '<|im_start|>assistant\n<think>\n' }}
{%- endif %}
"""


def opencode_chat_lm_format(
    *,
    messages_field: str = "messages",
    chat_template_kwargs_field: str = "chat_template_kwargs",
    pack: bool = True,
    mask_user_turns: bool = True,
) -> "ChatLmDatasetFormat":
    """Build the Levanter ``ChatLmDatasetFormat`` for the opencode agentic-SFT path.

    Wires ``OPENCODE_TOOLS_CHAT_TEMPLATE`` with completions-only masking and packing. The dataset
    it consumes must carry per-row ``tools`` relocated into the ``chat_template_kwargs`` column
    (``marin.transform.conversation.adapters.tools_column_to_chat_template_kwargs``) so the
    ``<tools>`` block renders. Import is lazy so this module is readable without a levanter install.

      * ``chat_template``      = the ``{% generation %}``-annotated tools-aware template
      * ``chat_template_kwargs`` = per-row column holding ``{"tools": [...]}``
      * ``mask_user_turns``    = True  (supervise assistant turns only)
      * ``pack``               = True  (greedy packing; analogue of axolotl ``sample_packing``)
    """
    from levanter.data.text.formats import ChatLmDatasetFormat

    return ChatLmDatasetFormat(
        messages_field=messages_field,
        chat_template=OPENCODE_TOOLS_CHAT_TEMPLATE,
        chat_template_kwargs=chat_template_kwargs_field,
        mask_user_turns=mask_user_turns,
        pack=pack,
    )
