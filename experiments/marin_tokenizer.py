# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E501  -- MARIN_CHAT_TEMPLATE has long literal lines that cannot be wrapped

"""
Saves a modified version of the llama3 tokenizer with:
1) a simple Olmo2-inspired chat format and
2) special tokens defined in this module.

The script uses temporary in-memory storage for intermediate operations.

"""

import json
import os
import tempfile
from typing import cast

from transformers import AutoTokenizer, PreTrainedTokenizer

from experiments.llama import llama3_tokenizer as llama3_tokenizer_hf_path

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


def inject_special_tokens(
    tokenizer: PreTrainedTokenizer,
    new_tokens: dict[int, str],
) -> PreTrainedTokenizer:
    """
    Rename existing vocabulary slots to new special-token strings.

    Rewrites ``added_tokens_decoder`` (``tokenizer_config.json``) and the fast tokenizer's
    ``added_tokens`` (``tokenizer.json``) so each ``token_id`` decodes to ``token_str`` and the
    string round-trips to that single id. A slot that does not exist yet is added.

    Args:
        tokenizer: The tokenizer to modify
        new_tokens: A dictionary of token_id -> token_str

    Returns:
        A new tokenizer instance
    """
    # Create a temporary directory that may be RAM-based
    with tempfile.TemporaryDirectory() as temp_path:
        # Save the original tokenizer to temp directory
        tokenizer.save_pretrained(temp_path)

        # Modify and save tokenizer_config.json
        tokenizer_config_path = os.path.join(temp_path, "tokenizer_config.json")
        with open(tokenizer_config_path, "r") as f:
            tokenizer_config = json.load(f)
        added_tokens_decoder = tokenizer_config.setdefault("added_tokens_decoder", {})
        for token_id, token_str in new_tokens.items():
            token_config = added_tokens_decoder.setdefault(
                str(token_id),
                {
                    "content": token_str,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                },
            )
            token_config["content"] = token_str
        with open(tokenizer_config_path, "w") as f:
            json.dump(tokenizer_config, f)

        # Also update tokenizer.json so the underlying fast tokenizer knows about the new strings
        tokenizer_json_path = os.path.join(temp_path, "tokenizer.json")
        if os.path.exists(tokenizer_json_path):
            with open(tokenizer_json_path, "r") as f:
                tokenizer_json = json.load(f)
            # Update the added_tokens list (id -> content)
            if "added_tokens" in tokenizer_json:
                seen_token_ids = set()
                for at in tokenizer_json["added_tokens"]:
                    tid = at.get("id")
                    if tid in new_tokens:
                        seen_token_ids.add(tid)
                        at["content"] = new_tokens[tid]
                for tid, token_str in new_tokens.items():
                    if tid not in seen_token_ids:
                        tokenizer_json["added_tokens"].append(
                            {
                                "id": tid,
                                "content": token_str,
                                "single_word": False,
                                "lstrip": False,
                                "rstrip": False,
                                "normalized": False,
                                "special": True,
                            }
                        )
            # Persist the file back
            with open(tokenizer_json_path, "w") as f:
                json.dump(tokenizer_json, f)

        # Load the modified tokenizer
        return cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(temp_path))


def create_marin_tokenizer(
    tokenizer: PreTrainedTokenizer,
) -> PreTrainedTokenizer:
    """
    Create a modified version of tokenizer with custom chat format and special tokens.

    Args:
        tokenizer: The base tokenizer to modify (typically llama3)

    Returns:
        A new tokenizer instance
    """
    # Inject special tokens
    marin_tokenizer = inject_special_tokens(tokenizer, MARIN_CUSTOM_SPECIAL_TOKENS)

    # Assign marin template
    marin_tokenizer.chat_template = MARIN_CHAT_TEMPLATE

    return marin_tokenizer


def load_llama3_tokenizer() -> PreTrainedTokenizer:
    """
    Load the base llama3 tokenizer.

    Returns:
        The llama3 tokenizer instance

    Raises:
        OSError, GatedRepoError, HTTPError: If access to the tokenizer is not available
    """
    return cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(llama3_tokenizer_hf_path))


def main(dry_run: bool = False):
    """
    Create the Marin tokenizer and push it to the Hugging Face Hub.

    Loads the base llama3 tokenizer, applies the custom chat format and special
    tokens, performs a roundtrip write-read to ensure consistency, and (unless
    ``dry_run``) pushes the result to the Hub.
    """
    llama3_tokenizer = load_llama3_tokenizer()
    tokenizer = create_marin_tokenizer(llama3_tokenizer)

    # Roundtrip write-read to ensure consistency.
    with tempfile.TemporaryDirectory() as temp_path:
        tokenizer.save_pretrained(temp_path)
        tokenizer = AutoTokenizer.from_pretrained(temp_path, local_files_only=True)

    if not dry_run:
        tokenizer.push_to_hub(marin_tokenizer)


if __name__ == "__main__":
    import sys

    main(dry_run="--dry-run" in sys.argv)
