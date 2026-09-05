# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from typing import cast

from levanter.data.text.formats import ChatLmDatasetFormat
from levanter.tokenizers import MarinTokenizer
from marin.datakit.chat import _render_messages


class _Tokenizer:
    bos_token = "<bos>"

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False
        assert add_generation_prompt is False
        assert kwargs["enable_thinking"] is False
        blocks = ["<|start_header_id|>system<|end_header_id|>\nReasoning: /nothink<|eot_id|>"]
        blocks.extend(
            f"<|start_header_id|>{message['role']}<|end_header_id|>\n{message['content']}<|eot_id|>"
            for message in conversation
        )
        return "<bos>" + "".join(blocks)


class _ToolTokenizer:
    bos_token = "<bos>"

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt, **kwargs):
        arguments = conversation[1]["tool_calls"][0]["function"]["arguments"]
        assert arguments == {"city": "Paris"}
        assert kwargs["tools"][0]["name"] == "weather"
        assert kwargs["enable_thinking"] is False
        return (
            "<bos><|start_header_id|>system<|end_header_id|>\nReasoning: /nothink<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\nWeather?<|eot_id|>"
            '<|start_header_id|>assistant<|end_header_id|>\n<tool_call>{"name":"weather",'
            '"arguments":{"city":"Paris"}}</tool_call><|eot_id|>'
        )


def test_render_messages_leaves_bos_insertion_to_text_tokenizer():
    [document] = _render_messages(
        {
            "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
            "source": "test",
        },
        cast(MarinTokenizer, _Tokenizer()),
        ChatLmDatasetFormat(mask_user_turns=False),
    )

    assert not document["text"].startswith("<bos>")
    assert "<|start_header_id|>assistant<|end_header_id|>\nHello<|eot_id|>" in document["text"]


def test_render_messages_decodes_tool_arguments_for_template():
    [document] = _render_messages(
        {
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_weather", "function": {"name": "weather", "arguments": '{"city":"Paris"}'}}
                    ],
                },
            ],
            "chat_template_kwargs": {"tools": [{"name": "weather"}]},
        },
        cast(MarinTokenizer, _ToolTokenizer()),
        ChatLmDatasetFormat(mask_user_turns=False, chat_template_kwargs="chat_template_kwargs"),
    )

    assert '"arguments":{"city":"Paris"}' in document["text"]


def test_render_messages_labels_reasoning_mode():
    class ThinkingTokenizer:
        bos_token = "<bos>"

        def apply_chat_template(self, conversation, *, enable_thinking, **kwargs):
            assert enable_thinking is True
            assert conversation[1]["content"] == "<|start_think|>plan<|end_think|>answer"
            return (
                "<bos><|start_header_id|>system<|end_header_id|>\nReasoning: /think<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\nQuestion<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n"
                "<|start_think|>plan<|end_think|>answer<|eot_id|>"
            )

    [document] = _render_messages(
        {
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "<|start_think|>plan<|end_think|>answer"},
            ]
        },
        cast(MarinTokenizer, ThinkingTokenizer()),
        ChatLmDatasetFormat(mask_user_turns=False),
    )

    assert "Reasoning: /think" in document["text"]
