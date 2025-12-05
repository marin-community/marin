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

"""Tests for InferenceContext utilities and chat template handling."""

from dataclasses import dataclass

import numpy as np
import pytest
from levanter.inference.openai import ChatMessage
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletionTokenLogprob, Choice, ChoiceLogprobs
from transformers import AutoTokenizer

from marin.rl.environments.inference_ctx import LevanterInferenceContext, LevanterInferenceContextConfig


@dataclass
class DummyInferenceServer:
    """Minimal inference server for testing."""

    host: str = "localhost"
    port: int = 8000

    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def config(self):
        @dataclass
        class Config:
            model_name: str = "test-model"

        return Config()


@pytest.fixture
def llama3_tokenizer():
    """Llama 3 tokenizer with chat template (uses tiktoken, not sentencepiece)."""
    return AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")


@pytest.fixture
def gpt2_tokenizer():
    """GPT-2 tokenizer without chat template (for fallback testing)."""
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def dummy_server():
    return DummyInferenceServer()


@pytest.fixture
def inference_ctx(llama3_tokenizer, dummy_server):
    return LevanterInferenceContext(
        LevanterInferenceContextConfig(
            inference_server_config=None,
            tokenizer=llama3_tokenizer,
            stop_tokens=None,
            max_tokens=100,
            mesh=None,
            axis_mapping={},
        )
    )


def create_choice_with_logprobs(tokenizer, response_text: str, logprobs_values: list[float] | None = None) -> Choice:
    """Create a Choice with proper BPE tokens and logprobs."""
    # Tokenize the response to get real token IDs
    token_ids = tokenizer.encode(response_text, add_special_tokens=False)

    if logprobs_values is None:
        logprobs_values = [-1.0] * len(token_ids)

    # Convert token IDs back to BPE tokens (preserves Ġ prefixes)
    logprobs_content = []
    for token_id, logprob in zip(token_ids, logprobs_values, strict=True):
        bpe_token = tokenizer.convert_ids_to_tokens(token_id)
        logprobs_content.append(
            ChatCompletionTokenLogprob(
                token=bpe_token,
                logprob=logprob,
                bytes=list(bpe_token.encode("utf-8")),
                top_logprobs=[],
            )
        )

    return Choice(
        finish_reason="stop",
        index=0,
        message=ChatCompletionMessage(role="assistant", content=response_text),
        logprobs=ChoiceLogprobs(content=logprobs_content),
    )


def test_apply_chat_template(llama3_tokenizer):
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Bigger: 87 or 3? Just the number:"),
    ]
    dict_messages = [msg.model_dump(exclude_none=True) for msg in messages]
    tokens = llama3_tokenizer.apply_chat_template(dict_messages, tokenize=True, add_generation_prompt=True)
    decoded = llama3_tokenizer.decode(tokens)

    assert "helpful assistant" in decoded
    assert "Bigger: 87 or 3?" in decoded
    assert "<|start_header_id|>assistant<|end_header_id|>" in decoded  # Llama 3's generation prompt marker


def test_bpe_round_trip_various_texts(llama3_tokenizer):
    """Validate BPE round-trip for diverse text patterns."""
    for text in ["!!}", "Hello world", "  spaces  ", "123", "\n\n"]:
        for token_id in llama3_tokenizer.encode(text, add_special_tokens=False):
            token_str = llama3_tokenizer.convert_ids_to_tokens(token_id)
            assert llama3_tokenizer.convert_tokens_to_ids(token_str) == token_id


def test_tokenize_prompt_adds_special_tokens(inference_ctx, llama3_tokenizer):
    """Test that tokenize_prompt uses chat template and adds special tokens."""
    prompt = "What is 2+2?"

    # InferenceContext uses chat template
    ctx_tokens = inference_ctx.tokenize_prompt(prompt)

    # Direct encode without template
    plain_tokens = llama3_tokenizer.encode(prompt, add_special_tokens=False)

    # Chat template should add tokens (system prompt markers, instruction markers, etc.)
    assert len(ctx_tokens) > len(plain_tokens)

    # Verify it's using the chat template by checking decoded output
    decoded = llama3_tokenizer.decode(ctx_tokens)
    assert "<|start_header_id|>user<|end_header_id|>" in decoded  # Llama 3 instruction markers
    assert prompt in decoded


def test_tokenize_prompt_fallback_no_template(gpt2_tokenizer, dummy_server):
    """Test fallback when tokenizer has no chat template."""
    ctx = LevanterInferenceContext(
        LevanterInferenceContextConfig(
            inference_server_config=None,
            tokenizer=gpt2_tokenizer,
            stop_tokens=None,
            max_tokens=100,
            mesh=None,
            axis_mapping={},
        )
    )

    prompt = "Test prompt"
    tokens = ctx.tokenize_prompt(prompt)

    # Should fallback to "user: Test prompt" format
    decoded = gpt2_tokenizer.decode(tokens)
    assert "user:" in decoded
    assert prompt in decoded


def test_response_tokens_from_choice(inference_ctx, llama3_tokenizer):
    """Test extracting token IDs from Choice using BPE round-trip."""
    response_text = "The answer is 42"
    choice = create_choice_with_logprobs(llama3_tokenizer, response_text)

    tokens = inference_ctx.response_tokens_from_choice(choice)

    # Should match tokenizer's encoding
    expected_tokens = llama3_tokenizer.encode(response_text, add_special_tokens=False)
    np.testing.assert_array_equal(tokens, expected_tokens)


def test_logprobs_from_choice(inference_ctx, llama3_tokenizer):
    """Test extracting logprobs array from Choice."""
    response_text = "The answer"
    choice = create_choice_with_logprobs(llama3_tokenizer, response_text)

    logprobs = inference_ctx.logprobs_from_choice(choice)

    # Should have same length as tokenized response
    expected_length = len(llama3_tokenizer.encode(response_text, add_special_tokens=False))
    assert logprobs.dtype == np.float32
    assert len(logprobs) == expected_length


def test_missing_logprobs_raises(inference_ctx):
    """Test that missing logprobs raises ValueError."""
    choice = Choice(
        finish_reason="stop",
        index=0,
        message=ChatCompletionMessage(role="assistant", content="test"),
        logprobs=None,
    )

    with pytest.raises(ValueError, match="missing logprobs"):
        inference_ctx.response_tokens_from_choice(choice)

    with pytest.raises(ValueError, match="missing logprobs"):
        inference_ctx.logprobs_from_choice(choice)


def test_create_rollout_from_choice_end_to_end(inference_ctx, llama3_tokenizer):
    """Test full rollout construction from prompt and choice."""
    prompt = "What is 2+2?"
    response_text = "The answer is 4"
    logprobs_values = [-0.5, -1.0, -0.8, -0.3, -1.2]
    reward = 1.0

    choice = create_choice_with_logprobs(llama3_tokenizer, response_text, logprobs_values)

    rollout = inference_ctx.create_rollout_from_choice(
        prompt=prompt, choice=choice, env_name="math_env", env_example_id="ex_001", reward=reward
    )

    # Verify metadata
    assert rollout.env_name == "math_env"
    assert rollout.env_example_id == "ex_001"
    assert rollout.episode_reward == reward

    # Verify prompt tokens use chat template (longer than plain encoding)
    plain_prompt_tokens = llama3_tokenizer.encode(prompt, add_special_tokens=False)
    assert len(rollout.prompt_tokens) > len(plain_prompt_tokens)

    # Verify response tokens match expected encoding
    expected_response_tokens = llama3_tokenizer.encode(response_text, add_special_tokens=False)
    np.testing.assert_array_equal(rollout.response_tokens, expected_response_tokens)

    # Verify logprobs
    np.testing.assert_array_almost_equal(rollout.response_logprobs, logprobs_values)

    # Verify token rewards
    assert len(rollout.token_rewards) == len(expected_response_tokens)
    np.testing.assert_array_equal(rollout.token_rewards, np.full(len(expected_response_tokens), reward))
