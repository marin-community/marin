# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pytest
from transformers import AutoTokenizer

from levanter.data.text import ChatProcessor


MODEL_NAME = "stanford-crfm/marin-tokenizer"

ALT_TEMPLATE = """{{ bos_token }}
{%- if enable_thinking is defined -%}
  {%- if enable_thinking is sameas true -%}
    {%- set reasoning_mode = "/think" -%}
  {%- elif enable_thinking is sameas false -%}
    {%- set reasoning_mode = "/nothink" -%}
  {%- else -%}
    {%- set reasoning_mode = enable_thinking -%}
  {%- endif -%}
{%- else -%}
  {%- set reasoning_mode = "/think" -%}
{%- endif -%}
{% if custom_instructions is defined and custom_instructions %}{{ custom_instructions }}{% endif %}
{%- set xml_tools_list = xml_tools | default([], true) -%}
<|im_start|>system
ALT Reasoning Mode: {{ reasoning_mode }}
{%- if xml_tools_list %}
\\nALT Tools:
{%- for tool in xml_tools_list -%}
\\n* {{ tool }}
{%- endfor -%}
{%- endif -%}
\\n<|im_end|>
{%- for message in messages -%}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>
{%- if message['role'] == 'assistant' -%}
{% generation %}[ALT] {{ message['content'] | trim }}<|eot_id|>{% endgeneration %}
{%- else -%}
[ALT] {{ message['content'] | trim }}<|eot_id|>
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt -%}
<|start_header_id|>assistant<|end_header_id|>
{% endif -%}
"""


@pytest.fixture(scope="module")
def tokenizer_path() -> Path:
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        return Path(tokenizer.name_or_path)
    except Exception as e:  # noqa
        pytest.skip(f"Could not load tokenizer {MODEL_NAME}: {e}", allow_module_level=True)
        raise NotImplementedError("unreachable")


def load_tokenizer(tokenizer_path: Path):
    return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)


def decode_sequence(tokenizer, tensor: Sequence[int]) -> str:
    return tokenizer.decode(list(tensor), skip_special_tokens=False)


def assert_messages_in_order(rendered: str, roles: Iterable[str]) -> None:
    search_pos = 0
    for role in roles:
        marker = f"<|start_header_id|>{role}<|end_header_id|>"
        pos = rendered.find(marker, search_pos)
        assert pos != -1, f"Did not find role {role!r} after position {search_pos}"
        search_pos = pos + 1


def test_chat_processor_injects_system_prompt(tokenizer_path: Path):
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer.chat_template = ALT_TEMPLATE
    processor = ChatProcessor(tokenizer, chat_template=ALT_TEMPLATE, mask_user_turns=False)

    batch = [
        {
            "messages": [
                {"role": "user", "content": "Hi there."},
                {"role": "assistant", "content": "Hello!"},
            ],
            "system": "You are a helpful assistant.",
        }
    ]

    result = processor(batch)
    assert len(result) == 1

    rendered = decode_sequence(tokenizer, result[0]["input_ids"])
    # Default template should remain unchanged beyond the injected system prompt.
    # Confirm the injected system message appears before the user turn.
    assert rendered.index("You are a helpful assistant.") < rendered.index("Hi there.")
    assert_messages_in_order(rendered, ["system", "user", "assistant"])
    assert "You are a helpful assistant." in rendered
    assert "Hi there." in rendered
    assert "Hello!" in rendered
    assert result[0]["assistant_masks"].sum() > 0


def test_chat_processor_respects_thinking_kwarg(tokenizer_path: Path):
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer.chat_template = ALT_TEMPLATE
    processor = ChatProcessor(tokenizer, chat_template=ALT_TEMPLATE, mask_user_turns=False)

    batch = [
        {
            "messages": [
                {"role": "user", "content": "Please reason carefully."},
                {"role": "assistant", "content": "Thoughtful answer."},
            ],
            "chat_template_kwargs": {
                "enable_thinking": True,
                "custom_instructions": "Follow best practices.",
            },
        }
    ]

    result = processor(batch)
    rendered = decode_sequence(tokenizer, result[0]["input_ids"])
    assert "Reasoning Mode: /think" in rendered
    assert "Follow best practices." in rendered


def test_chat_processor_handles_disable_thinking_kwarg(tokenizer_path: Path):
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer.chat_template = ALT_TEMPLATE
    processor = ChatProcessor(tokenizer, chat_template=ALT_TEMPLATE, mask_user_turns=False)

    batch = [
        {
            "messages": [
                {"role": "user", "content": "Don't think."},
                {"role": "assistant", "content": "Direct answer."},
            ],
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ]

    rendered = decode_sequence(tokenizer, processor(batch)[0]["input_ids"])
    assert "Reasoning Mode: /nothink" in rendered
    assert "<|start_think|>" not in rendered


def test_chat_processor_accepts_custom_reasoning_mode_value(tokenizer_path: Path):
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer.chat_template = ALT_TEMPLATE
    processor = ChatProcessor(tokenizer, chat_template=ALT_TEMPLATE, mask_user_turns=False)

    batch = [
        {
            "messages": [
                {"role": "user", "content": "Custom mode."},
                {"role": "assistant", "content": "Responding."},
            ],
            "chat_template_kwargs": {"enable_thinking": "experimental"},
        }
    ]

    rendered = decode_sequence(tokenizer, processor(batch)[0]["input_ids"])
    assert "Reasoning Mode: experimental" in rendered


def test_chat_processor_renders_tool_spec(tokenizer_path: Path):
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer.chat_template = ALT_TEMPLATE
    processor = ChatProcessor(tokenizer, chat_template=ALT_TEMPLATE, mask_user_turns=False)

    batch = [
        {
            "messages": [
                {"role": "user", "content": "What tools do you have?"},
                {"role": "assistant", "content": "Listing tools."},
            ],
            "chat_template_kwargs": {
                "xml_tools": [
                    '{"type": "function", "function": {"name": "final_answer"}}',
                ]
            },
        }
    ]

    rendered = decode_sequence(tokenizer, processor(batch)[0]["input_ids"])
    assert "ALT Tools" in rendered
    assert '{"type": "function", "function": {"name": "final_answer"}}' in rendered


def test_chat_processor_supports_per_example_chat_template_kwargs(tokenizer_path: Path):
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer.chat_template = ALT_TEMPLATE
    processor = ChatProcessor(tokenizer, chat_template=ALT_TEMPLATE, mask_user_turns=False)

    batch: list = [
        {
            "messages": [
                {"role": "user", "content": "First prompt"},
                {"role": "assistant", "content": "First reply"},
            ],
            "chat_template_kwargs": {
                "chat_template": ALT_TEMPLATE,
                "enable_thinking": False,
                "xml_tools": [
                    '{"type": "function", "function": {"name": "final_answer"}}',
                    '{"type": "function", "function": {"name": "web_search"}}',
                ],
                "add_generation_prompt": False,
            },
        },
        {
            "messages": [
                {"role": "user", "content": "Second prompt"},
                {"role": "assistant", "content": "Second reply"},
            ],
        },
    ]

    result = processor(batch)
    assert len(result) == 2

    rendered_override = decode_sequence(tokenizer, result[0]["input_ids"])

    assert "ALT Reasoning Mode: /nothink" in rendered_override
    assert "ALT Tools" in rendered_override
    assert '* {"type": "function", "function": {"name": "web_search"' in rendered_override
    assert "[ALT] First prompt" in rendered_override
    assert "[ALT] First reply" in rendered_override



def test_chat_processor_custom_system_field_name(tokenizer_path: Path):
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer.chat_template = ALT_TEMPLATE
    processor = ChatProcessor(
        tokenizer,
        chat_template=ALT_TEMPLATE,
        system_prompt_field="instructions",
        mask_user_turns=False,
    )

    batch = [
        {
            "messages": [
                {"role": "user", "content": "Request"},
                {"role": "assistant", "content": "Response"},
            ],
            "instructions": "Follow these instructions carefully.",
        }
    ]

    rendered = decode_sequence(tokenizer, processor(batch)[0]["input_ids"])
    assert_messages_in_order(rendered, ["system", "user", "assistant"])
    assert "Follow these instructions carefully." in rendered


def test_chat_processor_rejects_system_mapping_without_content(tokenizer_path: Path):
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer.chat_template = ALT_TEMPLATE
    processor = ChatProcessor(tokenizer, chat_template=ALT_TEMPLATE, mask_user_turns=False)

    batch = [
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            "system": {"role": "still system"},
        }
    ]

    with pytest.raises(ValueError, match="System prompt mapping must include a 'content' field"):
        processor(batch)
