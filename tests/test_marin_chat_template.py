# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence

import pytest

from experiments.create_marin_tokenizer import (
    create_marin_tokenizer,
    load_llama3_tokenizer,
    run_all_tests,
)
from levanter.data.text import ChatProcessor


@pytest.fixture()
def fresh_marin_tokenizer():
    try:
        base = load_llama3_tokenizer()
    except Exception as exc:
        pytest.skip(f"Could not load llama3 tokenizer: {exc}", allow_module_level=True)
    return create_marin_tokenizer(base)


def decode_sequence(tokenizer, tensor: Sequence[int]) -> str:
    return tokenizer.decode(list(tensor), skip_special_tokens=False)


def test_marin_chat_template_handles_tool_calls(fresh_marin_tokenizer):
    tokenizer = fresh_marin_tokenizer
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


def test_marin_tokenizer_integration_checks(fresh_marin_tokenizer):
    tokenizer = fresh_marin_tokenizer
    run_all_tests(tokenizer)


def test_marin_chat_template_ipython_output(fresh_marin_tokenizer):
    tokenizer = fresh_marin_tokenizer
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
