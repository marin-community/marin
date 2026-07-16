# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest
from haliax import Axis

from levanter.data.text.formats import ChatProcessor
from levanter.data.text.trace_chat import TraceChatProcessor
from levanter.data.text.trace_chat import (
    TRACE_LABEL_ASSISTANT_TEXT,
    TRACE_LABEL_ASSISTANT_TOOL_CALL,
    TRACE_LABEL_FINAL_ASSISTANT,
    TRACE_LABEL_OBSERVATION,
    TRACE_LABEL_PATCH,
    TraceChatDataset,
    TraceChatEvaluationFormat,
    dataset_for_trace_chat_format,
)
from levanter.store.cache import SerialCacheWriter
from levanter.tokenizers import MarinTokenizer, load_tokenizer


MODEL_NAME = "marin-community/marin-tokenizer"

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

TOOL_TEMPLATE = """{{ bos_token }}
{%- for message in messages -%}
  {%- if message['role'] == 'assistant' and message.get('tool_calls') -%}
    {%- set call = message['tool_calls'][0]['function'] -%}
<|start_header_id|>assistant<|end_header_id|>
{% generation %}{{ '{\"name\": \"' + call['name'] + '\", \"arguments\": ' }}{{ call['arguments'] | tojson }}{{ '}' }}<|eot_id|>{% endgeneration %}
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


@pytest.fixture(scope="module")
def tokenizer() -> MarinTokenizer:
    try:
        return load_tokenizer(MODEL_NAME)
    except Exception as e:  # noqa
        pytest.skip(f"Could not load tokenizer {MODEL_NAME}: {e}", allow_module_level=True)
        raise NotImplementedError("unreachable")


def decode_sequence(tokenizer: MarinTokenizer, tensor: Sequence[int]) -> str:
    return tokenizer.decode(list(tensor), skip_special_tokens=False)


def assert_messages_in_order(rendered: str, roles: list[str]) -> None:
    search_pos = 0
    for role in roles:
        marker = f"<|start_header_id|>{role}<|end_header_id|>"
        pos = rendered.find(marker, search_pos)
        assert pos != -1, f"Did not find role {role!r} after position {search_pos}"
        search_pos = pos + 1


def test_chat_processor_injects_system_prompt(tokenizer: MarinTokenizer):
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
    assert rendered.index("You are a helpful assistant.") < rendered.index("Hi there.")
    assert_messages_in_order(rendered, ["system", "user", "assistant"])
    assert "You are a helpful assistant." in rendered
    assert "Hi there." in rendered
    assert "Hello!" in rendered
    assert result[0]["assistant_masks"].sum() > 0


def test_chat_processor_respects_thinking_kwarg(tokenizer: MarinTokenizer):
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


def test_chat_processor_handles_disable_thinking_kwarg(tokenizer: MarinTokenizer):
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


def test_chat_processor_accepts_custom_reasoning_mode_value(tokenizer: MarinTokenizer):
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


def test_chat_processor_renders_tool_spec(tokenizer: MarinTokenizer):
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


def test_chat_processor_supports_per_example_chat_template_kwargs(tokenizer: MarinTokenizer):
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


def test_chat_processor_tool_call_support(tokenizer: MarinTokenizer):
    processor = ChatProcessor(tokenizer, chat_template=TOOL_TEMPLATE, mask_user_turns=True)

    batch = [
        {
            "messages": [
                {"role": "user", "content": "Call the adder."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "add", "arguments": {"a": 2, "b": 3}},
                        }
                    ],
                },
                {"role": "tool", "content": {"result": 5}},
                {"role": "assistant", "content": "The sum is 5."},
            ]
        }
    ]

    result = processor(batch)[0]
    rendered = decode_sequence(tokenizer, result["input_ids"])
    assert '{"name": "add", "arguments": {"a": 2, "b": 3}}' in rendered
    assert "<|start_header_id|>tool<|end_header_id|>" in rendered
    assert '{"result": 5}' in rendered
    assert result["assistant_masks"].sum() > 0


def test_chat_template_with_masks_returns_message_spans(tokenizer: MarinTokenizer):
    conversation = [
        {"role": "user", "content": "Call the adder."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "add", "arguments": {"a": 2, "b": 3}},
                }
            ],
        },
        {"role": "tool", "content": {"result": 5}},
    ]

    result = tokenizer.apply_chat_template_with_masks(
        [conversation],
        chat_template=TOOL_TEMPLATE,
        return_message_spans=True,
    )

    spans = result["message_spans"][0]
    assert len(spans) == len(conversation)
    assert all(start <= end for start, end in spans)
    tool_start, tool_end = spans[2]
    tool_text = decode_sequence(tokenizer, result["input_ids"][0][tool_start:tool_end])
    assert '{"result": 5}' in tool_text


def test_trace_chat_processor_labels_generation_masked_tool_spans(tokenizer: MarinTokenizer):
    processor = TraceChatProcessor(
        tokenizer,
        chat_template=TOOL_TEMPLATE,
        loss_tags=("assistant", "assistant_text", "tool_call", "observation", "final_assistant"),
    )

    result = processor(
        [
            {
                "messages": [
                    {"role": "user", "content": "Call the adder."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "add", "arguments": {"a": 2, "b": 3}},
                            }
                        ],
                    },
                    {"role": "tool", "content": {"result": 5}},
                    {"role": "assistant", "content": "The sum is 5."},
                ]
            }
        ]
    )[0]

    labels = result["loss_labels"]
    input_ids = result["input_ids"]

    tool_call_text = decode_sequence(tokenizer, input_ids[labels == TRACE_LABEL_ASSISTANT_TOOL_CALL])
    observation_text = decode_sequence(tokenizer, input_ids[labels == TRACE_LABEL_OBSERVATION])

    assert '"name": "add"' in tool_call_text
    assert '"arguments": {"a": 2, "b": 3}' in tool_call_text
    assert '{"result": 5}' in observation_text
    assert (labels == TRACE_LABEL_FINAL_ASSISTANT).sum() == 0


def test_trace_chat_processor_can_label_only_explicit_message_tags(tokenizer: MarinTokenizer):
    processor = TraceChatProcessor(
        tokenizer,
        chat_template=ALT_TEMPLATE,
        loss_tags=("assistant_text", "observation", "patch"),
        include_role_tags=False,
    )

    result = processor(
        [
            {
                "messages": [
                    {"role": "user", "content": "Call the adder."},
                    {"role": "assistant", "content": "I will inspect the repo."},
                    {"role": "tool", "content": {"result": 5}},
                    {"role": "assistant", "content": "diff --git a/a.py b/a.py", "loss_tags": ["patch"]},
                ]
            }
        ]
    )[0]

    labels = result["loss_labels"]
    patch_text = decode_sequence(tokenizer, result["input_ids"][labels == TRACE_LABEL_PATCH])
    assert (labels == TRACE_LABEL_PATCH).sum() > 0
    assert (labels == TRACE_LABEL_ASSISTANT_TEXT).sum() == 0
    assert (labels == TRACE_LABEL_OBSERVATION).sum() == 0
    assert "diff --git" in patch_text


def test_trace_chat_processor_labels_multi_tool_and_interleaved_user_turns(tokenizer: MarinTokenizer):
    processor = TraceChatProcessor(
        tokenizer,
        chat_template=MULTI_TOOL_TEMPLATE,
        loss_tags=("assistant", "assistant_text", "tool_call", "observation", "final_assistant"),
    )

    result = processor(
        [
            {
                "messages": [
                    {"role": "user", "content": "Find the failing test."},
                    {"role": "user", "content": "Use the repo tools before answering."},
                    {
                        "role": "assistant",
                        "content": "I will inspect two files.",
                        "tool_calls": [
                            {
                                "id": "call_search",
                                "type": "function",
                                "function": {"name": "search", "arguments": {"query": "failing test"}},
                            },
                            {
                                "id": "call_open",
                                "type": "function",
                                "function": {"name": "open_file", "arguments": {"path": "tests/test_eval.py"}},
                            },
                        ],
                    },
                    {"role": "user", "content": "Check the eval path first."},
                    {"role": "tool", "content": {"matches": ["tests/test_eval.py"]}},
                    {"role": "tool", "content": {"path": "tests/test_eval.py", "line": 345}},
                    {"role": "assistant", "content": "The issue is in the eval callback test."},
                ]
            }
        ]
    )[0]

    labels = result["loss_labels"]
    input_ids = result["input_ids"]
    assistant_text = decode_sequence(tokenizer, input_ids[labels == TRACE_LABEL_ASSISTANT_TEXT])
    tool_call_text = decode_sequence(tokenizer, input_ids[labels == TRACE_LABEL_ASSISTANT_TOOL_CALL])
    observation_text = decode_sequence(tokenizer, input_ids[labels == TRACE_LABEL_OBSERVATION])
    final_text = decode_sequence(tokenizer, input_ids[labels == TRACE_LABEL_FINAL_ASSISTANT])

    assert "I will inspect two files." in assistant_text
    assert '"name": "search"' in tool_call_text
    assert '"name": "open_file"' in tool_call_text
    assert "failing test" not in observation_text
    assert "Check the eval path first." not in observation_text
    assert '"matches": ["tests/test_eval.py"]' in observation_text
    assert '"line": 345' in observation_text
    assert "The issue is in the eval callback test." in final_text


def test_trace_chat_processor_splits_text_tool_calls(tokenizer: MarinTokenizer):
    processor = TraceChatProcessor(
        tokenizer,
        chat_template=MULTI_TOOL_TEMPLATE,
        loss_tags=("assistant", "assistant_text", "tool_call"),
    )

    result = processor(
        [
            {
                "messages": [
                    {"role": "user", "content": "Read the file."},
                    {
                        "role": "assistant",
                        "content": (
                            "I will read it.\n"
                            '<tool_call>{"name": "open_file", "arguments": {"path": "README.md"}}</tool_call>'
                        ),
                    },
                ]
            }
        ]
    )[0]

    labels = result["loss_labels"]
    input_ids = result["input_ids"]
    assistant_text = decode_sequence(tokenizer, input_ids[labels == TRACE_LABEL_ASSISTANT_TEXT])
    tool_call_text = decode_sequence(tokenizer, input_ids[labels == TRACE_LABEL_ASSISTANT_TOOL_CALL])

    assert "I will read it." in assistant_text
    assert '"name": "open_file"' in tool_call_text
    assert '"path": "README.md"' in tool_call_text


@pytest.mark.asyncio
async def test_trace_chat_dataset_shifts_labels_without_cross_document_bleed(tmp_path):
    exemplar = {
        "input_ids": np.zeros((0,), dtype=np.int32),
        "loss_labels": np.zeros((0,), dtype=np.int32),
    }
    with SerialCacheWriter(str(tmp_path), exemplar) as writer:
        writer.write_batch(
            [
                {
                    "input_ids": np.array([10, 11, 12], dtype=np.int32),
                    "loss_labels": np.array([TRACE_LABEL_ASSISTANT_TEXT] * 3, dtype=np.int32),
                },
                {
                    "input_ids": np.array([20, 21, 22], dtype=np.int32),
                    "loss_labels": np.array([TRACE_LABEL_ASSISTANT_TOOL_CALL] * 3, dtype=np.int32),
                },
            ]
        )

    dataset = TraceChatDataset(
        writer.result(),
        Axis("position", 6),
        max_segments_per_example=2,
        slice_strategy="raise",
        block_cross_document_attention=True,
    )
    example = await dataset.getitem_async(0)

    np.testing.assert_array_equal(example.tokens, np.array([10, 11, 12, 20, 21, 22], dtype=np.int32))
    np.testing.assert_array_equal(
        example.loss_labels,
        np.array(
            [
                TRACE_LABEL_ASSISTANT_TEXT,
                TRACE_LABEL_ASSISTANT_TEXT,
                0,
                TRACE_LABEL_ASSISTANT_TOOL_CALL,
                TRACE_LABEL_ASSISTANT_TOOL_CALL,
                0,
            ],
            dtype=np.int32,
        ),
    )
    assert example.attn_mask.segment_ids is not None
    query_segment_ids, key_segment_ids = example.attn_mask.segment_ids
    np.testing.assert_array_equal(query_segment_ids, np.array([0, 0, 0, 1, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(key_segment_ids, np.array([0, 0, 0, 1, 1, 1], dtype=np.int32))


def test_tool_call_masking_behavior(tokenizer: MarinTokenizer):
    processor = ChatProcessor(tokenizer, chat_template=TOOL_TEMPLATE, mask_user_turns=True)

    batch = [
        {
            "messages": [
                {"role": "user", "content": "Add two numbers."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_add",
                            "type": "function",
                            "function": {"name": "add", "arguments": {"a": 1, "b": 2}},
                        }
                    ],
                },
                {"role": "tool", "content": {"result": 3}},
                {"role": "assistant", "content": "3"},
            ]
        }
    ]

    result = processor(batch)[0]
    mask = result["assistant_masks"]
    ids = list(result["input_ids"])

    # Decode to find boundaries. The tool call content and final assistant
    # reply are inside {% generation %} blocks, so their tokens should be masked.
    # Tool response tokens should NOT be masked.
    rendered = decode_sequence(tokenizer, ids)

    # Find the tool response section — it should have mask=0
    tool_header = "<|start_header_id|>tool<|end_header_id|>"
    assert tool_header in rendered

    # The rendered text has clear structure. Verify that masked tokens exist
    # (from generation blocks) and unmasked tokens exist (user + tool turns).
    assert mask.sum() > 0, "Expected some masked (assistant) tokens"
    assert (mask == 0).sum() > 0, "Expected some unmasked (non-assistant) tokens"

    # Verify tool response content is not in the masked region by checking
    # that the tokens for the tool response decode to unmasked content.
    # Build unmasked text from tokens where mask==0
    unmasked_ids = [tok_id for tok_id, m in zip(ids, mask) if m == 0]
    unmasked_text = tokenizer.decode(unmasked_ids, skip_special_tokens=False)
    assert "tool" in unmasked_text.lower() or '{"result": 3}' in unmasked_text


def test_chat_processor_custom_system_field_name(tokenizer: MarinTokenizer):
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


def test_chat_processor_rejects_system_mapping_without_content(tokenizer: MarinTokenizer):
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

    with pytest.raises(ValueError, match="System prompt mapping must include 'content'"):
        processor(batch)


def _write_trace_cache(tmp_path, docs):
    exemplar = {
        "input_ids": np.zeros((0,), dtype=np.int32),
        "loss_labels": np.zeros((0,), dtype=np.int32),
    }
    with SerialCacheWriter(str(tmp_path), exemplar) as writer:
        writer.write_batch(
            [
                {
                    "input_ids": np.array(ids, dtype=np.int32),
                    "loss_labels": np.array(labels, dtype=np.int32),
                }
                for ids, labels in docs
            ]
        )
    return writer.result()


@pytest.mark.parametrize("pack", [False, 1])
def test_dataset_for_trace_chat_format_unpacked_yields_one_padded_example_per_trace(tmp_path, pack):
    cache = _write_trace_cache(
        tmp_path,
        [
            ([10, 11, 12], [TRACE_LABEL_ASSISTANT_TEXT] * 3),
            ([20, 21], [TRACE_LABEL_ASSISTANT_TOOL_CALL] * 2),
        ],
    )
    trace_format = TraceChatEvaluationFormat(pack=pack)
    Pos = Axis("position", 8)
    ds = dataset_for_trace_chat_format(trace_format, Pos, cache).as_sync_dataset()

    # one example per trace; unpacked mode never packs two traces together
    assert len(ds) == 2
    first = ds[0]
    np.testing.assert_array_equal(np.asarray(first.tokens)[:3], np.array([10, 11, 12], dtype=np.int32))
    for ex in ds:
        assert ex.tokens.shape == (Pos.size,)
        segment_ids = np.asarray(ex.attn_mask.segment_ids[0])
        padding = segment_ids == -1
        assert padding.any(), "trace should be shorter than Pos, leaving padding"
        # neither padding nor the position predicting the first pad token may carry a loss label
        predicts_padding = np.roll(segment_ids, -1) == -1
        np.testing.assert_array_equal(np.asarray(ex.loss_labels)[padding | predicts_padding], 0)


def test_dataset_for_trace_chat_format_unpacked_raises_on_document_longer_than_pos(tmp_path):
    cache = _write_trace_cache(
        tmp_path,
        [([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [TRACE_LABEL_ASSISTANT_TEXT] * 10)],
    )
    trace_format = TraceChatEvaluationFormat(pack=False)
    Pos = Axis("position", 4)
    with pytest.raises(ValueError, match="exceeds"):
        dataset_for_trace_chat_format(trace_format, Pos, cache)


# ---------------------------------------------------------------------------
# HuggingFace parity: Levanter's chat renderer vs transformers.apply_chat_template
#
# Levanter reimplements apply_chat_template(..., return_assistant_tokens_mask=True)
# without transformers. These tests pin that reimplementation against the real HF
# tokenizer as an independent oracle: rendered string, input_ids, and assistant_masks
# must all match. HF renders the chat, tokenizes the full string once, and maps
# {% generation %} char spans onto tokens via char_to_token; Levanter must do the same
# so that BPE merges spanning a generation boundary and JSON key ordering agree.
# ---------------------------------------------------------------------------

# Minimal generation-bearing template. Generation content is delimited by special
# tokens (role header + <|eot_id|>), so no BPE merge crosses the mask boundary.
HF_SIMPLE_TEMPLATE = """\
{%- for message in messages -%}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>
{%- if message['role'] == 'assistant' -%}
{% generation %}{{ message['content'] }}{% endgeneration %}
{%- else -%}
{{ message['content'] }}
{%- endif -%}
<|eot_id|>
{%- endfor -%}
"""

# Adversarial: the {% generation %} block abuts ordinary text (no special token,
# no whitespace) on both sides, so the correct tokenization BPE-merges across the
# mask boundary ("Bot says: <gen>greetings" and "friend</gen> done"). Levanter's
# former per-segment encoding split those merges; tokenizing once fixes it. This
# case fails against HF unless the renderer maps char spans onto a single encoding.
HF_BOUNDARY_MERGE_TEMPLATE = """\
{%- for message in messages -%}
{%- if message['role'] == 'assistant' -%}
Bot says: {% generation %}{{ message['content'] }}{% endgeneration %} done.
{%- else -%}
User: {{ message['content'] }}
{%- endif -%}
{%- endfor -%}
"""


# A llama3-instruct-style trainable template exercising the constructs that separate real
# SFT templates from the toy ones above: a hoisted system message, a `tools` parameter
# serialized with `tojson(indent=4)`, tool_calls rendered with `tojson`, tool-role
# responses, and `{% generation %}` assistant spans. Self-contained so the levanter test
# suite stays independent of the marin layer where the production template lives.
LLAMA3_STYLE_TEMPLATE = """{{- bos_token }}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set system_message = '' %}
    {%- set loop_messages = messages %}
{%- endif %}
{{- '<|start_header_id|>system<|end_header_id|>\\n\\n' + system_message }}
{%- if tools is defined and tools %}
    {{- '\\n\\nYou have access to the following functions:\\n' }}
    {%- for tool in tools %}
        {{- tool | tojson(indent=4) }}
        {{- '\\n' }}
    {%- endfor %}
{%- endif %}
{{- '<|eot_id|>' }}
{%- for message in loop_messages %}
    {%- if message['role'] == 'assistant' and message.get('tool_calls') %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}
        {%- generation %}
        {%- for tool_call in message['tool_calls'] %}
            {{- '{"name": "' + tool_call['function']['name'] + '", "parameters": ' }}
            {{- tool_call['function']['arguments'] | tojson }}
            {{- '}' }}
        {%- endfor %}
        {%- endgeneration %}
        {{- '<|eot_id|>' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}
        {%- generation %}{{ message['content'] }}{%- endgeneration %}
        {{- '<|eot_id|>' }}
    {%- elif message['role'] == 'tool' %}
        {{- '<|start_header_id|>ipython<|end_header_id|>\\n\\n' + (message['content'] | tojson) + '<|eot_id|>' }}
    {%- else %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}
{%- endif %}"""

# Serializes the `tools` list, forcing canonical key order with `tojson(sort_keys=True)`
# when the `sort_tools` kwarg is set. Both HF's tojson filter and Levanter's accept the
# sort_keys flag, so the canonical rendering must agree byte-for-byte — this is the
# escape hatch a deployment uses for prefix-cache-stable prompts, and it must stay a
# template choice rather than a hardcoded renderer policy.
HF_SORT_KEYS_TEMPLATE = """\
<|start_header_id|>system<|end_header_id|>
{% for tool in tools %}{% if sort_tools %}{{ tool | tojson(sort_keys=True) }}{% else %}{{ tool | tojson }}{% endif %}
{% endfor %}<|eot_id|>
{%- for message in messages -%}
{%- if message['role'] == 'assistant' -%}
<|start_header_id|>assistant<|end_header_id|>
{% generation %}{{ message['content'] }}{% endgeneration %}<|eot_id|>
{%- else -%}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>
{{ message['content'] }}<|eot_id|>
{%- endif -%}
{%- endfor -%}
"""

_WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }
]

_MULTI_TURN = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "And 3+3?"},
    {"role": "assistant", "content": "It is 6."},
]

_WITH_SYSTEM = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hi there."},
    {"role": "assistant", "content": "Hello! How can I help you today?"},
]

_TOOL_CALL_CONV = [
    {"role": "user", "content": "Call the adder."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "add", "arguments": {"a": 2, "b": 3}}}
        ],
    },
    {"role": "tool", "content": {"result": 5}},
    {"role": "assistant", "content": "The sum is 5."},
]

_MULTI_TOOL_CONV = [
    {"role": "user", "content": "Search then open."},
    {
        "role": "assistant",
        "content": "On it.",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "search", "arguments": {"query": "cats"}}},
            {"id": "c2", "type": "function", "function": {"name": "open", "arguments": {"path": "a.py"}}},
        ],
    },
    {"role": "assistant", "content": "Done."},
]

_LLAMA3_TOOL_CONV = [
    {"role": "system", "content": "You are a weather bot."},
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
    },
    {"role": "tool", "content": {"temp": 20}},
    {"role": "assistant", "content": "It is 20 degrees in Paris."},
]

_ADVERSARIAL_CONV = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "greetings friend"},
    {"role": "user", "content": "bye"},
    {"role": "assistant", "content": "farewell traveller"},
]

_PARITY_CASES = [
    pytest.param(HF_SIMPLE_TEMPLATE, _MULTI_TURN, {}, id="simple-multi-turn"),
    pytest.param(HF_SIMPLE_TEMPLATE, _WITH_SYSTEM, {}, id="simple-system"),
    pytest.param(TOOL_TEMPLATE, _TOOL_CALL_CONV, {}, id="tool-calls-and-response"),
    pytest.param(MULTI_TOOL_TEMPLATE, _MULTI_TOOL_CONV, {}, id="multi-tool-calls"),
    pytest.param(
        ALT_TEMPLATE,
        [{"role": "user", "content": "Reason please."}, {"role": "assistant", "content": "Sure, here goes."}],
        {"enable_thinking": True, "custom_instructions": "Be careful.", "xml_tools": ['{"name": "final_answer"}']},
        id="alt-chat-template-kwargs",
    ),
    pytest.param(HF_BOUNDARY_MERGE_TEMPLATE, _ADVERSARIAL_CONV, {}, id="boundary-merge-crosses-generation"),
    pytest.param(LLAMA3_STYLE_TEMPLATE, _LLAMA3_TOOL_CONV, {"tools": _WEATHER_TOOLS}, id="llama3-style-with-tools"),
    pytest.param(
        LLAMA3_STYLE_TEMPLATE,
        _WITH_SYSTEM + [{"role": "user", "content": "Bye"}, {"role": "assistant", "content": "Goodbye."}],
        {},
        id="llama3-style-plain",
    ),
]


@pytest.fixture(scope="module")
def hf_tokenizer(tokenizer: MarinTokenizer):
    try:
        hf = tokenizer.as_hf_tokenizer()
    except Exception as e:  # noqa: BLE001 - network/optional-dep failure should skip, not error
        pytest.skip(f"Could not construct HF tokenizer: {e}")
    if not hf.is_fast:
        pytest.skip("assistant-mask parity requires a fast HF tokenizer for char_to_token")
    return hf


@pytest.mark.parametrize("template, conversation, kwargs", _PARITY_CASES)
def test_chat_template_matches_hf(tokenizer: MarinTokenizer, hf_tokenizer, template, conversation, kwargs):
    lev = tokenizer.apply_chat_template_with_masks([conversation], chat_template=template, **kwargs)
    lev_ids = lev["input_ids"][0]
    lev_mask = lev["assistant_masks"][0]
    lev_rendered = tokenizer.with_chat_template(template).apply_chat_template(conversation, tokenize=False, **kwargs)

    hf_out = hf_tokenizer.apply_chat_template(
        conversation,
        chat_template=template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
        **kwargs,
    )
    hf_rendered = hf_tokenizer.apply_chat_template(
        conversation, chat_template=template, tokenize=False, add_generation_prompt=False, **kwargs
    )

    assert lev_rendered == hf_rendered
    assert lev_ids == list(hf_out["input_ids"])
    assert lev_mask == list(hf_out["assistant_masks"])
    # A template with an assistant turn must charge at least one assistant token,
    # otherwise "masks match" could hold vacuously on two all-zero masks.
    assert sum(lev_mask) > 0


def test_tojson_sort_keys_flag_matches_hf(tokenizer: MarinTokenizer, hf_tokenizer):
    """An explicit `tojson(sort_keys=True)` in a template canonicalizes JSON key order,
    identically in Levanter and HF.

    The renderer must honor the template's sort choice rather than hardcode one: the
    default (insertion order) keeps parity with the base model's native tool format,
    while `sort_keys=True` is the opt-in a deployment uses for prefix-cache-stable
    prompts. The tool keys here are in insertion order (`type` before `function`), which
    is not alphabetical, so sorting must change the output — asserted below so the parity
    check is not vacuously satisfied by both engines ignoring the flag.
    """
    conversation = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

    def lev_render(sort_tools: bool) -> str:
        return tokenizer.with_chat_template(HF_SORT_KEYS_TEMPLATE).apply_chat_template(
            conversation, tokenize=False, tools=_WEATHER_TOOLS, sort_tools=sort_tools
        )

    sorted_rendered = lev_render(sort_tools=True)
    assert sorted_rendered != lev_render(sort_tools=False), "sort_keys=True must reorder non-alphabetical keys"

    lev = tokenizer.apply_chat_template_with_masks(
        [conversation], chat_template=HF_SORT_KEYS_TEMPLATE, tools=_WEATHER_TOOLS, sort_tools=True
    )
    hf_out = hf_tokenizer.apply_chat_template(
        conversation,
        chat_template=HF_SORT_KEYS_TEMPLATE,
        tools=_WEATHER_TOOLS,
        sort_tools=True,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )
    hf_rendered = hf_tokenizer.apply_chat_template(
        conversation,
        chat_template=HF_SORT_KEYS_TEMPLATE,
        tools=_WEATHER_TOOLS,
        sort_tools=True,
        tokenize=False,
        add_generation_prompt=False,
    )

    assert sorted_rendered == hf_rendered
    assert lev["input_ids"][0] == list(hf_out["input_ids"])
    assert lev["assistant_masks"][0] == list(hf_out["assistant_masks"])
    assert sum(lev["assistant_masks"][0]) > 0
