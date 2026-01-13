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

"""Tests for CodeR1 environment."""

import jax.random
import numpy as np
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.completion_usage import CompletionUsage
from transformers import AutoTokenizer

from marin.rl.environments.code_r1_env import (
    CodeR1Env,
    extract_python_code,
    execute_code_with_tests,
)
from marin.rl.environments.inference_ctx import LevanterInferenceContext


class TestExtractPythonCode:
    """Tests for Python code extraction from model responses."""

    def test_extract_from_python_block(self):
        """Extract code from ```python ... ``` block."""
        response = """I'll solve this step by step.

```python
def add(a, b):
    return a + b
```

This function adds two numbers."""
        code = extract_python_code(response)
        assert code is not None
        assert "def add(a, b):" in code
        assert "return a + b" in code

    def test_extract_from_generic_block(self):
        """Extract code from generic ``` ... ``` block."""
        response = """Here's the solution:

```
def multiply(x, y):
    return x * y
```
"""
        code = extract_python_code(response)
        assert code is not None
        assert "def multiply(x, y):" in code

    def test_no_code_returns_none(self):
        """Return None when no code is found."""
        response = "I'm not sure how to solve this problem."
        code = extract_python_code(response)
        assert code is None

    def test_fallback_to_def_statement(self):
        """Fallback to finding def statement in response."""
        # Functionality not currently supported in strict Code-R1 mode
        pass


class TestExecuteCodeWithTests:
    """Tests for code execution with test cases."""

    def test_passing_code(self):
        """Test that passing code returns success."""
        code = "def add(a, b): return a + b"
        test_cases = "assert add(1, 2) == 3\nassert add(0, 0) == 0"
        passed, error = execute_code_with_tests(code, test_cases)
        assert passed is True
        assert error == ""

    def test_failing_code(self):
        """Test that failing code returns failure."""
        code = "def add(a, b): return a - b  # Wrong implementation"
        test_cases = "assert add(1, 2) == 3"
        passed, error = execute_code_with_tests(code, test_cases)
        assert passed is False
        assert "AssertionError" in error or "exit code" in error.lower()

    def test_syntax_error(self):
        """Test that syntax errors are handled."""
        code = "def broken(: return"  # Invalid syntax
        test_cases = "assert broken() == 1"
        passed, _error = execute_code_with_tests(code, test_cases)
        assert passed is False

    def test_empty_code(self):
        """Test that empty code returns failure."""
        passed, error = execute_code_with_tests("", "assert True")
        assert passed is False
        assert "No code" in error

    def test_timeout(self):
        """Test that infinite loops are handled."""
        code = "def infinite(): \n    while True: pass"
        test_cases = "infinite()"
        passed, error = execute_code_with_tests(code, test_cases, timeout=1)
        assert passed is False
        assert "timeout" in error.lower()

    def test_default_timeout_alignment(self):
        """Test that default timeout is at least 30s (aligned with Code-R1)."""
        # This code sleeps for 6 seconds.
        # It fails with the old 5s timeout, but passes with the new 30s timeout.
        code = "import time\ndef slow():\n    time.sleep(6)"
        test_cases = "slow()"

        # Should pass with default timeout (now 30s)
        passed, error = execute_code_with_tests(code, test_cases)
        assert passed is True, f"Code failed with error: {error}"


def create_mock_code_completion(tokenizer, code_response: str) -> ChatCompletion:
    """Create a mock ChatCompletion with code response."""
    tokens = tokenizer.encode(code_response, add_special_tokens=False)
    logprobs_content = [
        ChatCompletionTokenLogprob(
            token=tokenizer.convert_ids_to_tokens([tok])[0], logprob=-0.5, bytes=[], top_logprobs=[]
        )
        for tok in tokens
    ]

    return ChatCompletion(
        id="chatcmpl-test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=code_response),
                logprobs={"content": logprobs_content},
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=len(tokens), prompt_tokens=10, total_tokens=10 + len(tokens)),
    )


class DummyInferenceContext(LevanterInferenceContext):
    """Mock inference context for testing."""

    def __init__(self, tokenizer, code_response: str):
        self.tokenizer = tokenizer
        self._stop_tokens = None
        self.max_tokens = 512
        self.code_response = code_response

    def batch_completions(
        self,
        prompts,
        temperature,
        n,
        max_tokens=None,
        top_k=None,
        stop=None,
        system_prompt=None,
        prefill=None,
    ):
        """Return mock completions for each prompt."""
        return [create_mock_code_completion(self.tokenizer, self.code_response) for _ in prompts]


class TestCodeR1Env:
    """Tests for CodeR1Env class."""

    def test_env_initialization_with_mock_data(self):
        """Test environment initialization with mock dataset."""
        train_data = [
            {
                "content": "Write a function that adds two numbers.",
                "test": "assert add(1, 2) == 3\nassert add(-1, 1) == 0",
                "python": "def add(a, b): return a + b",
                "difficulty": "easy",
            }
        ]

        env = CodeR1Env(train_dataset=train_data, max_train_examples=1)

        assert len(env.train_examples) == 1
        assert env.system_prompt is None
        assert "Please provide a self-contained Python script" in env.train_examples[0].processed_prompt

    def test_sample_with_passing_code(self):
        """Test sampling with code that passes tests."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Response with correct code (no tags needed anymore)
        code_response = """def add(a, b):
    return a + b
```"""

        inference_ctx = DummyInferenceContext(tokenizer, code_response)

        train_data = [
            {
                "content": "Write a function that adds two numbers.",
                "test": "assert add(1, 2) == 3\nassert add(-1, 1) == 0",
                "python": "def add(a, b): return a + b",
            }
        ]

        env = CodeR1Env(train_dataset=train_data, max_train_examples=1)
        prng_key = jax.random.PRNGKey(42)

        rollout_groups, metrics = env.sample(
            inference_ctx=inference_ctx,
            n_examples=1,
            n_generations=1,
            temperature=0.7,
            prng_key=prng_key,
            mode="train",
        )

        assert len(rollout_groups) == 1
        rollout = rollout_groups[0].rollouts[0]

        assert rollout.env_name == "code_r1"
        assert rollout.prompt_tokens.dtype == np.int32
        assert rollout.response_tokens.dtype == np.int32

        # Code should pass, reward should be 1.0 (correct)
        assert rollout.episode_reward == pytest.approx(1.1)  # FORMAT_REWARD + ANSWER_REWARD
        assert metrics["code_r1.train_pass_rate"] == pytest.approx(1.0)

    def test_sample_with_failing_code(self):
        """Test sampling with code that fails tests."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Response with incorrect code
        code_response = """def add(a, b):
    return a - b  # Wrong!
```"""

        inference_ctx = DummyInferenceContext(tokenizer, code_response)

        train_data = [
            {
                "content": "Write a function that adds two numbers.",
                "test": "assert add(1, 2) == 3",
                "python": "def add(a, b): return a + b",
            }
        ]

        env = CodeR1Env(train_dataset=train_data, max_train_examples=1)
        prng_key = jax.random.PRNGKey(42)

        rollout_groups, metrics = env.sample(
            inference_ctx=inference_ctx,
            n_examples=1,
            n_generations=1,
            temperature=0.7,
            prng_key=prng_key,
            mode="train",
        )

        rollout = rollout_groups[0].rollouts[0]
        # Code failed, reward should be FORMAT_REWARD
        assert rollout.episode_reward == pytest.approx(0.1)

        # But code was extracted
        assert metrics["code_r1.train_code_extracted_rate"] == pytest.approx(1.0)

    def test_prompt_format(self):
        """Test that prompts follow Code-R1 format."""
        train_data = [
            {
                "content": "Test problem.",
                "test": "assert True",
                "python": "pass",
            }
        ]

        env = CodeR1Env(train_dataset=train_data)

        assert len(env.train_examples) == 1
        prompt = env.train_examples[0].processed_prompt

        # Should contain the user template
        assert "Please provide a self-contained Python script" in prompt
        assert "Test problem." in prompt

    def test_metrics_include_execution_time(self):
        """Test that execution time metrics are returned."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        code_response = """def add(a, b): return a + b
```"""

        inference_ctx = DummyInferenceContext(tokenizer, code_response)
        train_data = [{"content": "Prob", "test": "assert add(1,1)==2", "python": "sol"}]
        env = CodeR1Env(train_dataset=train_data, max_train_examples=1)
        prng_key = jax.random.PRNGKey(42)

        _, metrics = env.sample(
            inference_ctx=inference_ctx,
            n_examples=1,
            n_generations=1,
            temperature=0.7,
            prng_key=prng_key,
            mode="train",
        )

        assert "code_r1.train_mean_execution_time" in metrics
        assert "code_r1.train_max_execution_time" in metrics
        assert "code_r1.train_min_execution_time" in metrics
        assert "code_r1.train_total_execution_time" in metrics
        assert metrics["code_r1.train_mean_execution_time"] >= 0.0
        assert metrics["code_r1.train_max_execution_time"] >= 0.0
        assert metrics["code_r1.train_min_execution_time"] >= 0.0
        assert metrics["code_r1.train_total_execution_time"] >= 0.0
