# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from itertools import pairwise

import pytest
from levanter.data.text.trace_chat import (
    TRACE_LABEL_ASSISTANT_TOOL_CALL,
    TRACE_LABEL_FINAL_ASSISTANT,
    TRACE_LABEL_OBSERVATION,
    TraceChatProcessor,
)
from levanter.tokenizers import MarinTokenizer, load_tokenizer

from experiments.chat_templates.llama3pt1_chat_template import LLAMA_3_1_CHAT_TEMPLATE
from experiments.chat_templates.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from experiments.marin_tokenizer import MARIN_CHAT_TEMPLATE

_MESSAGE_SPAN_REAL_TEMPLATE_CASES = [
    (
        "marin",
        "marin-community/marin-tokenizer",
        MARIN_CHAT_TEMPLATE,
        {},
    ),
    (
        "llama3.1",
        "meta-llama/Llama-3.1-8B",
        LLAMA_3_1_CHAT_TEMPLATE,
        {"tools": None},
    ),
    (
        "qwen3",
        "Qwen/Qwen3-8B",
        QWEN_3_CHAT_TEMPLATE,
        {"tools": None},
    ),
    (
        "gemma3",
        "google/gemma-3-4b-it",
        None,
        {},
    ),
    (
        "gpt-oss",
        "openai/gpt-oss-20b",
        None,
        {"tools": None, "builtin_tools": None},
    ),
]


def _load_optional_tokenizer(name: str) -> MarinTokenizer:
    try:
        return load_tokenizer(name)
    except Exception as e:
        pytest.skip(f"Cannot load tokenizer {name}: {e}")


@pytest.mark.parametrize(
    "case_name,tokenizer_name,chat_template,template_kwargs",
    _MESSAGE_SPAN_REAL_TEMPLATE_CASES,
    ids=[case[0] for case in _MESSAGE_SPAN_REAL_TEMPLATE_CASES],
)
def test_apply_chat_template_message_spans_real_templates(case_name, tokenizer_name, chat_template, template_kwargs):
    tokenizer = _load_optional_tokenizer(tokenizer_name)
    conversation = [
        {"role": "user", "content": f"{case_name} alpha prompt."},
        {"role": "assistant", "content": f"{case_name} beta answer."},
        {"role": "user", "content": f"{case_name} gamma prompt."},
        {"role": "assistant", "content": f"{case_name} delta answer."},
    ]

    result = tokenizer.apply_chat_template_with_masks(
        [conversation],
        chat_template=chat_template,
        return_message_spans=True,
        **template_kwargs,
    )

    input_ids = result["input_ids"][0]
    spans = result["message_spans"][0]
    assert len(spans) == len(conversation)
    assert all(start < end for start, end in spans)
    assert all(left[1] <= right[0] for left, right in pairwise(spans))

    for message, (start, end) in zip(conversation, spans, strict=True):
        span_text = tokenizer.decode(input_ids[start:end], skip_special_tokens=False)
        assert message["content"] in span_text


@pytest.mark.parametrize(
    "case_name,tokenizer_name,chat_template,template_kwargs",
    _MESSAGE_SPAN_REAL_TEMPLATE_CASES[:3],
    ids=[case[0] for case in _MESSAGE_SPAN_REAL_TEMPLATE_CASES[:3]],
)
def test_trace_chat_processor_labels_real_templates(case_name, tokenizer_name, chat_template, template_kwargs):
    tokenizer = _load_optional_tokenizer(tokenizer_name)
    processor = TraceChatProcessor(
        tokenizer,
        chat_template=chat_template,
        loss_tags=("assistant", "tool_call", "observation", "final_assistant"),
    )

    result = processor(
        [
            {
                "messages": [
                    {"role": "user", "content": f"{case_name} call the lookup tool."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_lookup",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": {"key": case_name}},
                            }
                        ],
                    },
                    {"role": "tool", "content": '{"result": 3}'},
                    {"role": "assistant", "content": f"{case_name} final answer is done."},
                ],
                "chat_template_kwargs": template_kwargs,
            }
        ]
    )[0]

    labels = result["loss_labels"]
    input_ids = result["input_ids"]
    tool_call_text = tokenizer.decode(input_ids[labels == TRACE_LABEL_ASSISTANT_TOOL_CALL].tolist())
    observation_text = tokenizer.decode(input_ids[labels == TRACE_LABEL_OBSERVATION].tolist())
    final_text = tokenizer.decode(input_ids[labels == TRACE_LABEL_FINAL_ASSISTANT].tolist())

    assert "lookup" in tool_call_text
    assert case_name in tool_call_text
    assert "result" in observation_text
    assert "3" in observation_text
    assert f"{case_name} final answer is done." in final_text
