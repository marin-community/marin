from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pytest
from transformers import AutoTokenizer

from levanter.data.text import ChatProcessor


MODEL_NAME = "stanford-crfm/marin-tokenizer"

BASE_TEMPLATE = """{{ bos_token }}
{%- set enable_thinking = enable_thinking | default(true) -%}
{%- set xml_tools_list = xml_tools | default([]) -%}
{%- if enable_thinking -%}
  {%- set reasoning_mode = "/think" -%}
{%- else -%}
  {%- set reasoning_mode = "/no_think" -%}
{%- endif -%}
<|im_start|>system
Reasoning Mode: {{ reasoning_mode }}
{%- if xml_tools_list | length > 0 -%}
\\nTools:
{%- for tool in xml_tools_list -%}
\\n- {{ tool }}
{%- endfor -%}
{%- endif -%}
\\n<|im_end|>
{%- for message in messages -%}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>
{%- if message['role'] == 'assistant' -%}
{% generation %}{{ message['content'] | trim }}<|eot_id|>{% endgeneration %}
{%- else -%}
{{ message['content'] | trim }}<|eot_id|>
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt -%}
<|start_header_id|>assistant<|end_header_id|>
{% endif -%}
"""

ALT_TEMPLATE = """{{ bos_token }}
{%- set enable_thinking = enable_thinking | default(true) -%}
{%- set xml_tools_list = xml_tools | default([]) -%}
{%- if enable_thinking -%}
  {%- set reasoning_mode = "/think" -%}
{%- else -%}
  {%- set reasoning_mode = "/no_think" -%}
{%- endif -%}
<|im_start|>system
ALT Reasoning Mode: {{ reasoning_mode }}
{%- if xml_tools_list | length > 0 -%}
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
    tokenizer.chat_template = BASE_TEMPLATE
    processor = ChatProcessor(tokenizer, chat_template=BASE_TEMPLATE, mask_user_turns=False)

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
    assert "Reasoning Mode: /think" in rendered
    assert "Tools:" not in rendered
    # Confirm the injected system message appears before the user turn.
    assert rendered.index("You are a helpful assistant.") < rendered.index("Hi there.")
    assert_messages_in_order(rendered, ["system", "user", "assistant"])
    assert "You are a helpful assistant." in rendered
    assert "Hi there." in rendered
    assert "Hello!" in rendered
    assert result[0]["assistant_masks"].sum() > 0


def test_chat_processor_supports_per_example_chat_template_kwargs(tokenizer_path: Path):
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer.chat_template = BASE_TEMPLATE
    processor = ChatProcessor(tokenizer, chat_template=BASE_TEMPLATE, mask_user_turns=False)

    batch = [
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
    rendered_default = decode_sequence(tokenizer, result[1]["input_ids"])

    assert "ALT Reasoning Mode: /no_think" in rendered_override
    assert "ALT Tools" in rendered_override
    assert '* {"type": "function", "function": {"name": "web_search"' in rendered_override
    assert "[ALT] First prompt" in rendered_override
    assert "[ALT] First reply" in rendered_override
    assert "[ALT]" not in rendered_default
    assert "Second prompt" in rendered_default
    assert "Reasoning Mode: /think" in rendered_default


def test_chat_processor_custom_system_field_name(tokenizer_path: Path):
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer.chat_template = BASE_TEMPLATE
    processor = ChatProcessor(
        tokenizer,
        chat_template=BASE_TEMPLATE,
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
    tokenizer.chat_template = BASE_TEMPLATE
    processor = ChatProcessor(tokenizer, chat_template=BASE_TEMPLATE, mask_user_turns=False)

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
