# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Sequence

import pytest

from experiments.create_marin_tokenizer import (
    create_marin_tokenizer,
    load_llama3_tokenizer,
    run_all_tests,
)
from levanter.data.text import ChatProcessor

pytestmark = pytest.mark.skipif("CI" in os.environ, reason="Requires HF tokenizer download")


def _load_marin_tokenizer():
    return create_marin_tokenizer(load_llama3_tokenizer())


def decode_sequence(tokenizer, tensor: Sequence[int]) -> str:
    return tokenizer.decode(list(tensor), skip_special_tokens=False)


def test_marin_chat_template_handles_tool_calls():
    tokenizer = _load_marin_tokenizer()
    processor = ChatProcessor(tokenizer, mask_user_turns=True)

    batch = [
        {
            "messages": [
                {"role": "user", "content": "Run the VIN check."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "check_valid_vin", "arguments": {"vin": "1FMXK92W8YPA12345"}},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "check_valid_vin",
                    "tool_call_id": "call_abc",
                    "content": {"valid": True},
                },
                {"role": "assistant", "content": "VIN 1FMXK92W8YPA12345 is valid."},
            ]
        }
    ]

    result = processor(batch)
    rendered = decode_sequence(tokenizer, result[0]["input_ids"])
    assert '{"name": "check_valid_vin", "arguments": {"vin": "1FMXK92W8YPA12345"}}' in rendered
    assert '<tool_response name="check_valid_vin" id="call_abc">' in rendered
    assert result[0]["assistant_masks"].sum() > 0


def test_marin_tokenizer_integration_checks():
    tokenizer = _load_marin_tokenizer()
    run_all_tests(tokenizer)


def test_marin_chat_template_ipython_output():
    tokenizer = _load_marin_tokenizer()
    processor = ChatProcessor(tokenizer, mask_user_turns=True)

    batch = [
        {
            "messages": [
                {"role": "user", "content": "Show me the result."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_output",
                            "type": "function",
                            "function": {"name": "python_exec", "arguments": {"code": "print(1+1)"}},
                        }
                    ],
                },
                {
                    "role": "ipython",
                    "content": [{"type": "text", "text": "4\n"}],
                },
                {"role": "assistant", "content": "The result is 4."},
            ]
        }
    ]

    result = processor(batch)[0]
    rendered = decode_sequence(tokenizer, result["input_ids"])
    assert '{"name": "python_exec", "arguments": {"code": "print(1+1)"}}' in rendered
    assert "<|start_header_id|>ipython<|end_header_id|>" in rendered
    assert '{"output": "4\\n"}' in rendered
    assert result["assistant_masks"].sum() > 0
