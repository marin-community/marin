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

from unittest.mock import Mock

import numpy as np
import pytest

from marin.rl.environments.mock_env import (
    AdditionTask,
    MoarCatsTask,
    NumberComparisonTask,
    OppositesTask,
    compute_soft_reward,
)
from marin.rl.types import InferenceChoice, InferenceResponse


@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    tokenizer.encode = Mock(side_effect=lambda text: [ord(c) for c in text])
    return tokenizer


@pytest.fixture
def mock_inference_ctx(mock_tokenizer):
    ctx = Mock()
    ctx.tokenizer = mock_tokenizer

    def mock_generate(prompts, temperature, n_generations):
        responses = []
        for prompt in prompts:
            choices = []
            for i in range(n_generations):
                response_text = f"mock_response_{i}"
                response_tokens = np.array([ord(c) for c in response_text], dtype=np.int32)
                logprobs = np.full(len(response_tokens), -1.0, dtype=np.float32)
                choices.append(InferenceChoice(response_text, response_tokens, logprobs))

            prompt_tokens = np.array([ord(c) for c in prompt], dtype=np.int32)
            responses.append(InferenceResponse(prompt, prompt_tokens, choices))
        return responses

    ctx.generate = mock_generate
    return ctx


def test_compute_soft_reward_format_loss():
    assert compute_soft_reward("42", "42") > compute_soft_reward("42", "42 extra words")
    assert compute_soft_reward("42", "42") > compute_soft_reward("42", "wrong")

    assert compute_soft_reward("42", "42") == pytest.approx(0.3)
    assert compute_soft_reward("42", "wrong") < 0

    short_format_score = compute_soft_reward("42", "43")
    long_format_score = compute_soft_reward("42", "43 with lots of extra words")
    assert short_format_score > long_format_score
    assert short_format_score < 0
    assert long_format_score < 0


def test_addition_task_reward():
    task = AdditionTask()
    examples = task.generate_training_examples(10, np.random.default_rng(42))
    assert len(examples) == 10
    assert all("+" in ex["prompt"] for ex in examples)

    assert task.compute_reward("42", "42") == pytest.approx(0.3)
    assert task.compute_reward("42", "43") < 0
    assert task.compute_reward("42", "-") < 0
    assert task.compute_reward("42", "-2") < 0


def test_opposites_task_reward():
    task = OppositesTask()
    examples = task.generate_training_examples(10, np.random.default_rng(42))
    assert len(examples) == 10

    assert task.compute_reward("cold", "cold") == pytest.approx(0.3)
    assert task.compute_reward("cold", "warm") < 0


def test_number_comparison_task_format_bonus():
    task = NumberComparisonTask()

    digit_reward = task.compute_reward("42", "42")
    non_digit_reward = task.compute_reward("42", "forty-two")

    assert digit_reward > non_digit_reward
    assert digit_reward > 0
    assert non_digit_reward < 0


def test_cats_task_reward():
    task = MoarCatsTask()

    assert task.compute_reward("cats", "cats cats cats") > task.compute_reward("cats", "cats")
    assert task.compute_reward("cats", "i love cats") > task.compute_reward("cats", "i like cats")

    assert task.compute_reward("cats", "cat") > 0
    assert task.compute_reward("cats", "dog") == 0
