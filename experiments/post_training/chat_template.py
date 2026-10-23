# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The shared chat-template structure and per-family token maps.

One Jinja template renders every architecture's conversations; a family supplies
only its delimiters. The template is built by baking a :class:`ChatTokens` map's
strings into the skeleton as Jinja string literals, so the result is self-contained
and can be frozen into a tokenizer config without importing this module at serve
time.

The contract the template owns:

- Reasoning renders from ``reasoning_content`` on every assistant turn. There is no
  last-query gate, no history truncation, and no injected empty think block.
- Tool calls render as ``<tool_call>\\n{"name": ..., "arguments": ...}\\n</tool_call>``
  with identical JSON serialization on every family.
- Tool results use bare ``<tool_response>`` without a ``name=`` attribute, so the
  marker stays one token wherever the vocabulary has one.
- A system turn renders only when a system message is present; none is injected.

Every structural delimiter in a token map must encode to exactly one token of its
pinned tokenizer (:func:`validate_token_map`), so masking, stop conditions and the
protocol gate are all token-exact.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# Placeholders baked out by shared_template(); double underscores cannot collide
# with Jinja syntax or tokenizer vocabulary.
_TURN_OPEN_PREFIX = "__TURN_OPEN_PREFIX__"
_TURN_OPEN_SUFFIX = "__TURN_OPEN_SUFFIX__"
_TURN_CLOSE = "__TURN_CLOSE__"
_THINK_OPEN = "__THINK_OPEN__"
_THINK_CLOSE = "__THINK_CLOSE__"
_BOS = "__BOS__"

_TEMPLATE_SKELETON = """{%- if '__BOS__' -%}{{ '__BOS__' }}{%- endif -%}{%- for message in messages -%}
{%- if message.role == 'assistant' -%}
{{- '__TURN_OPEN_PREFIX__assistant__TURN_OPEN_SUFFIX__\n' -}}
{%- if message.reasoning_content -%}
{{- '__THINK_OPEN__' ~ message.reasoning_content ~ '__THINK_CLOSE__' -}}
{%- endif -%}
{{- message.content if message.content else '' -}}
{%- for call in message.tool_calls or [] -%}
{%- set function = call.function if call.function is defined else call -%}
{{- '<tool_call>\\n{"name": ' -}}
{{- function.name | tojson -}}
{{- ', "arguments": ' -}}
{{- function.arguments if function.arguments is string else function.arguments | tojson -}}
{{- '}\\n</tool_call>' -}}
{%- endfor -%}
{{- '__TURN_CLOSE__' -}}
{%- elif message.role == 'tool' -%}
{{- '__TURN_OPEN_PREFIX__tool__TURN_OPEN_SUFFIX__\n<tool_response>\\n' -}}
{{- message.content if message.content else '' -}}
{{- '\\n</tool_response>__TURN_CLOSE__' -}}
{%- else -%}
{{- '__TURN_OPEN_PREFIX__' ~ message.role ~ '__TURN_OPEN_SUFFIX__\\n' -}}
{{- message.content if message.content else '' -}}
{{- '__TURN_CLOSE__' -}}
{%- endif -%}
{{- '\\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '__TURN_OPEN_PREFIX__assistant__TURN_OPEN_SUFFIX__\n' -}}
{%- endif -%}"""


@dataclass(frozen=True)
class ChatTokens:
    """One family's delimiters. ``bos`` may be empty where the family has none."""

    bos: str
    turn_open_prefix: str
    turn_open_suffix: str
    turn_close: str
    think_open: str
    think_close: str


GRUG_TOKENS = ChatTokens(
    bos="<|begin_of_text|>",
    turn_open_prefix="<|start_header_id|>",
    turn_open_suffix="<|end_header_id|>",
    turn_close="<|eot_id|>",
    think_open="<|start_think|>",
    think_close="<|end_think|>",
)

# Qwen3 and Qwen3.5 share one convention, as does Nemotron-H apart from its BOS.
QWEN_TOKENS = ChatTokens(
    bos="",
    turn_open_prefix="<|im_start|>",
    turn_open_suffix="",
    turn_close="<|im_end|>",
    think_open="<think>",
    think_close="</think>",
)
NEMOTRON_TOKENS = ChatTokens(
    bos="<s>",
    turn_open_prefix="<|im_start|>",
    turn_open_suffix="",
    turn_close="<|im_end|>",
    think_open="<think>",
    think_close="</think>",
)

# Keys match the trainer-side architecture names (Axolotl's config strings; "grug"
# for the Levanter Snowball path).
FAMILY_TOKEN_MAPS: dict[str, ChatTokens] = {
    "grug": GRUG_TOKENS,
    "qwen3_moe": QWEN_TOKENS,
    "qwen3_5_moe": QWEN_TOKENS,
    "nemotron_h": NEMOTRON_TOKENS,
}


def _jinja_literal(value: str) -> str:
    """Escape ``value`` for use inside a single-quoted Jinja string literal."""
    return value.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "\\r")


def shared_template(tokens: ChatTokens) -> str:
    """The shared Jinja template with one family's delimiters baked in as literals."""
    template = _TEMPLATE_SKELETON
    for placeholder, value in (
        (_BOS, tokens.bos),
        (_TURN_OPEN_PREFIX, tokens.turn_open_prefix),
        (_TURN_OPEN_SUFFIX, tokens.turn_open_suffix),
        (_TURN_CLOSE, tokens.turn_close),
        (_THINK_OPEN, tokens.think_open),
        (_THINK_CLOSE, tokens.think_close),
    ):
        template = template.replace(placeholder, _jinja_literal(value))
    assert _TURN_OPEN_PREFIX not in template, "skeleton placeholders must all be baked"
    return template


def template_sha256(tokens: ChatTokens) -> str:
    """The sha256 of the rendered template, for manifests and integrity checks."""
    return hashlib.sha256(shared_template(tokens).encode()).hexdigest()


def validate_token_map(tokens: ChatTokens, tokenizer) -> dict[str, int]:
    """Check every delimiter encodes to exactly one token of ``tokenizer``.

    Returns the token lengths by slot; raises ``ValueError`` naming the offending
    slot otherwise. ``bos`` is exempt when empty (Qwen has none).
    """
    lengths: dict[str, int] = {}
    for slot in ("turn_open_prefix", "turn_open_suffix", "turn_close", "think_open", "think_close"):
        value = getattr(tokens, slot)
        if not value:
            continue
        length = len(tokenizer.encode(value, add_special_tokens=False))
        if length != 1:
            raise ValueError(f"{slot}={value!r} encodes to {length} tokens; the shared template needs one")
        lengths[slot] = length
    if tokens.bos:
        bos_length = len(tokenizer.encode(tokens.bos, add_special_tokens=False))
        if bos_length != 1:
            raise ValueError(f"bos={tokens.bos!r} encodes to {bos_length} tokens; the shared template needs one")
        lengths["bos"] = bos_length
    slots = [tokens.turn_open_prefix, tokens.turn_open_suffix, tokens.turn_close, tokens.think_open, tokens.think_close]
    non_empty = [slot for slot in slots if slot]
    if len(set(non_empty)) != len(non_empty):
        raise ValueError("token map delimiters must be distinct")
    return lengths
