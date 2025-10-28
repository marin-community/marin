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

"""Tests for vLLM result processing functions."""

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs, ChatCompletionTokenLogprob
from openai.types.completion_usage import CompletionUsage
from transformers import AutoTokenizer

from marin.rl.environments.process_vllm_results import (
    parse_chat_completion_tokens_from_bytes,
    parse_chat_completion_logprobs,
)


@pytest.fixture
def tokenizer():
    """Create a tokenizer for testing."""
    return AutoTokenizer.from_pretrained("gpt2")


def create_mock_chat_completion_with_logprobs(
    response_text: str, token_strs: list[str], logprobs: list[float]
) -> ChatCompletion:
    """Create a mock ChatCompletion with logprobs for testing."""
    assert len(token_strs) == len(logprobs), "Token strings and logprobs must have same length"

    logprob_content = [
        ChatCompletionTokenLogprob(
            token=token_str,
            logprob=logprob,
            bytes=list(token_str.encode("utf-8")),
            top_logprobs=[],
        )
        for token_str, logprob in zip(token_strs, logprobs, strict=False)
    ]

    return ChatCompletion(
        id="chatcmpl-test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=response_text),
                logprobs=ChoiceLogprobs(content=logprob_content, refusal=None),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=len(token_strs), prompt_tokens=10, total_tokens=10 + len(token_strs)),
    )


def test_parse_chat_completion_tokens_special_tokens(tokenizer):
    """Test token parsing with special tokens like newlines."""
    # Ċ is how GPT-2 represents newlines in BPE
    token_strs = ["Hello", "Ċ", "World"]
    logprobs = [-0.1, -0.2, -0.3]

    chat_completion = create_mock_chat_completion_with_logprobs("Hello\nWorld", token_strs, logprobs)

    parsed_tokens = parse_chat_completion_tokens_from_bytes(chat_completion, tokenizer)

    assert len(parsed_tokens) == 3
    assert all(isinstance(t, int) for t in parsed_tokens)

    expected_ids = [tokenizer.convert_tokens_to_ids(t) for t in token_strs]
    assert parsed_tokens == expected_ids


def test_parse_chat_completion_tokens_token_id_format(tokenizer):
    """Test token parsing when tokens are in 'token_id:<int>' format."""
    # Some vLLM configurations might return tokens in this format
    token_strs = ["token_id:123", "token_id:456", "token_id:789"]
    logprobs = [-0.1, -0.2, -0.3]

    chat_completion = create_mock_chat_completion_with_logprobs("test", token_strs, logprobs)

    parsed_tokens = parse_chat_completion_tokens_from_bytes(chat_completion, tokenizer)

    assert parsed_tokens == [123, 456, 789]


def test_parse_chat_completion_logprobs(tokenizer):
    """Test logprob extraction."""
    token_strs = ["Hello", "Ġworld"]
    logprobs = [-0.123, -0.456]

    chat_completion = create_mock_chat_completion_with_logprobs("Hello world", token_strs, logprobs)

    parsed_logprobs = parse_chat_completion_logprobs(chat_completion)

    assert len(parsed_logprobs) == 2
    assert parsed_logprobs == logprobs


def test_parse_chat_completion_tokens_empty_response():
    """Test handling of empty response."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    chat_completion = create_mock_chat_completion_with_logprobs("", [], [])

    parsed_tokens = parse_chat_completion_tokens_from_bytes(chat_completion, tokenizer)
    assert parsed_tokens == []
