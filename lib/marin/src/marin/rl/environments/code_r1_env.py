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

"""CodeR1 RL environment using LeetCode dataset for code generation training.

Based on the Code-R1 approach: https://github.com/ganler/code-r1
Uses newfacade/LeetCodeDataset for training and HumanEvalPlus for evaluation.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import datasets
import jax
import numpy as np

from marin.rl.environments.inference_ctx.base import BaseInferenceContext
from marin.rl.types import Rollout, RolloutGroup
from .base import MarinEnv, extract_seed

logger = logging.getLogger(__name__)

LEETCODE_DATASET = "newfacade/LeetCodeDataset"

HUMANEVAL_DATASET = "evalplus/humanevalplus"

# Code-R1 system prompt for training
CODE_R1_SYSTEM_PROMPT = (
    "You are a helpful programming assistant. The user will ask you a question and you as "
    "the assistant solve it. The assistant first thinks how to solve the task through reasoning "
    "and then provides the user with the final answer. The reasoning process and answer are "
    "enclosed within <think>...</think> and <answer>...</answer> tags, respectively."
)

# Code-R1 user prompt template
CODE_R1_USER_TEMPLATE = "Solve the programming task below in a Python markdown code block.\n{problem}"

# Default timeout for code execution (seconds)
CODE_EXECUTION_TIMEOUT = 5


@dataclass(slots=True)
class CodeExample:
    """Container for a single code problem."""

    problem: str
    """The problem description."""

    test_cases: str
    """Test cases as executable Python code."""

    solution: str | None
    """Reference solution (if available)."""

    processed_prompt: str
    """The formatted prompt for the model."""

    example_id: str
    """Unique identifier for this example."""

    difficulty: str | None = None
    """Problem difficulty (easy/medium/hard)."""

    metadata: dict[str, Any] = field(default_factory=dict)


LoadDatasetFn = Callable[..., Any]


def extract_python_code(response: str) -> str | None:
    """Extract Python code from a response that may contain markdown code blocks.

    Handles both ```python ... ``` and ``` ... ``` formats, as well as
    code inside <answer>...</answer> tags.
    """
    # First try to extract from <answer> tags
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer_match:
        response = answer_match.group(1)

    # Try to find Python code block
    python_block = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if python_block:
        return python_block.group(1).strip()

    # Try generic code block
    generic_block = re.search(r"```\s*(.*?)```", response, re.DOTALL)
    if generic_block:
        return generic_block.group(1).strip()

    # If no code blocks, try to find code-like content (fallback)
    # Look for def/class statements
    if "def " in response or "class " in response:
        return response.strip()

    return None


def execute_code_with_tests(code: str, test_cases: str, timeout: int = CODE_EXECUTION_TIMEOUT) -> tuple[bool, str]:
    """Execute Python code with test cases in a sandboxed subprocess.

    Args:
        code: The Python code to execute.
        test_cases: Test case code that should run after the main code.
        timeout: Maximum execution time in seconds.

    Returns:
        Tuple of (success, error_message).
    """
    if not code:
        return False, "No code extracted"

    # Combine code and tests
    full_code = f"{code}\n\n{test_cases}"

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            temp_path = f.name

        result = subprocess.run(
            ["python", temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr[:500] if result.stderr else "Non-zero exit code"

    except subprocess.TimeoutExpired:
        return False, f"Execution timeout ({timeout}s)"
    except Exception as e:
        return False, str(e)[:500]


class CodeR1Env(MarinEnv):
    """CodeR1 environment for code generation RL training.

    Uses LeetCode dataset for training and supports HumanEvalPlus for evaluation.
    """

    def __init__(
        self,
        train_source: str = LEETCODE_DATASET,
        eval_source: str | None = HUMANEVAL_DATASET,
        eval_split: str | None = None,
        *,
        max_train_examples: int | None = None,
        max_eval_examples: int | None = None,
        seed: int | None = None,
        trust_remote_code: bool = True,
        datasets_loader: LoadDatasetFn | None = None,
        train_dataset: Iterable[dict[str, Any]] | None = None,
        eval_dataset: Iterable[dict[str, Any]] | None = None,
        execution_timeout: int = CODE_EXECUTION_TIMEOUT,
    ) -> None:
        """Initialize CodeR1 environment.

        Args:
            train_source: HuggingFace dataset ID for training data.
            eval_source: HuggingFace dataset ID for eval data (if None, uses train split).
            max_train_examples: Maximum number of training examples to load.
            max_eval_examples: Maximum number of eval examples to load.
            seed: Random seed for sampling.
            trust_remote_code: Whether to trust remote code when loading datasets.
            datasets_loader: Custom dataset loader function.
            train_dataset: Pre-loaded training dataset (overrides train_source).
            eval_dataset: Pre-loaded eval dataset (overrides eval_source).
            execution_timeout: Timeout for code execution in seconds.
        """
        self.train_source = train_source
        self.eval_source = eval_source
        self.eval_split = eval_split
        self.max_train_examples = max_train_examples
        self.max_eval_examples = max_eval_examples
        self._trust_remote_code = trust_remote_code
        self._datasets_loader = datasets_loader or datasets.load_dataset
        self._rng = np.random.default_rng(seed)
        self.execution_timeout = execution_timeout
        self.system_prompt = CODE_R1_SYSTEM_PROMPT

        self.train_examples = self._prepare_split(
            split_name="train",
            hf_split="train",
            examples_iter=train_dataset,
            source=train_source,
            limit=max_train_examples,
        )

        # For eval, use provided dataset or skip if no eval_source
        if eval_dataset is not None:
            # User provided explicit dataset iterator
            self.eval_examples = self._prepare_split(
                split_name="eval",
                hf_split="train",  # Dummy split name when using iterator
                examples_iter=eval_dataset,
                source=eval_source or train_source,
                limit=max_eval_examples,
            )
        elif self.eval_source is not None:
            # Load from HuggingFace source
            effective_eval_split = self.eval_split
            if effective_eval_split is None:
                effective_eval_split = "test" if self.eval_source == HUMANEVAL_DATASET else "train"

            self.eval_examples = self._prepare_split(
                split_name="eval",
                hf_split=effective_eval_split,
                examples_iter=None,
                source=self.eval_source,
                limit=max_eval_examples,
            )
        else:
            # No eval dataset
            self.eval_examples = []

        logger.info(
            "Initialized CodeR1Env with %d train examples and %d eval examples.",
            len(self.train_examples),
            len(self.eval_examples),
        )

    def _prepare_split(
        self,
        *,
        split_name: str,
        hf_split: str,
        examples_iter: Iterable[dict[str, Any]] | None,
        source: str,
        limit: int | None,
    ) -> list[CodeExample]:
        """Load and clean a dataset split."""

        if examples_iter is None:
            dataset = self._datasets_loader(
                source,
                split=hf_split,
                trust_remote_code=self._trust_remote_code,
            )
        else:
            dataset = examples_iter

        cleaned_examples: list[CodeExample] = []
        for idx, item in enumerate(dataset):
            example = self._clean_example(item, f"{split_name}_{idx}")
            if example is not None:
                cleaned_examples.append(example)
                if limit is not None and len(cleaned_examples) >= limit:
                    break

        return cleaned_examples

    def _clean_example(self, item: dict[str, Any], example_id: str) -> CodeExample | None:
        """Convert a raw dataset item into a CodeExample."""

        # Detect dataset schema
        if "canonical_solution" in item:
            # HumanEvalPlus format
            problem = item["prompt"]
            test_cases = item["test"]
            solution = item["canonical_solution"]
            difficulty = None
            # Use task_id as example_id if available
            example_id = item.get("task_id", example_id)
        elif "problem_description" in item:
            # LeetCodeDataset format (new schema)
            problem = item["problem_description"]
            test_cases = item["test"]
            solution = item.get("completion")
            difficulty = item.get("difficulty")
            example_id = item.get("task_id", example_id)
        elif "content" in item:
            # LeetCodeDataset format (old schema/fallback)
            problem = item.get("content", "")
            test_cases = item.get("test", "")
            solution = item.get("python", None)
            difficulty = item.get("difficulty", None)
        else:
            # Unknown format
            logger.warning(f"Unknown dataset format for example {example_id}. Keys: {item.keys()}")
            return None

        if not problem or not test_cases:
            return None

        # Format prompt using Code-R1 template
        processed_prompt = CODE_R1_USER_TEMPLATE.format(problem=problem)

        return CodeExample(
            problem=problem,
            test_cases=test_cases,
            solution=solution,
            processed_prompt=processed_prompt,
            example_id=example_id,
            difficulty=difficulty,
            metadata={"source": self.train_source},
        )

    def sample(
        self,
        inference_ctx: BaseInferenceContext,
        n_examples: int,
        n_generations: int,
        temperature: float,
        prng_key,
        mode: str = "train",
        max_tokens: int | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> tuple[list[RolloutGroup], dict[str, float]]:
        """Sample prompts, generate code, execute, and create rollouts."""

        if mode not in ("train", "eval"):
            raise ValueError(f"Unsupported mode: {mode}")

        available_examples = self.train_examples if mode == "train" else self.eval_examples
        if not available_examples:
            raise ValueError(f"No examples available for mode '{mode}'")

        n_to_sample = min(n_examples, len(available_examples))
        seed = extract_seed(prng_key)
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(available_examples), size=n_to_sample, replace=False)
        sampled_examples = [available_examples[int(idx)] for idx in indices]

        # Use Code-R1 system prompt if not overridden
        effective_system_prompt = system_prompt or self.system_prompt

        prompts = [example.processed_prompt for example in sampled_examples]
        completions = inference_ctx.batch_completions(
            prompts=prompts,
            temperature=temperature,
            n=n_generations,
            max_tokens=max_tokens,
            top_k=top_k,
            stop=stop,
            system_prompt=effective_system_prompt,
        )

        rollout_groups: list[RolloutGroup] = []
        total_choices = 0
        reward_sum = 0.0
        pass_sum = 0.0
        code_extracted_sum = 0.0
        response_token_count = 0
        truncated_count = 0

        for example, completion in zip(sampled_examples, completions, strict=True):
            group_rollouts: list[Rollout] = []

            for choice in completion.choices:
                reward, passed, code_extracted = self._score_choice(
                    example=example,
                    response_text=choice.message.content,
                )

                rollout = inference_ctx.create_rollout_from_choice(
                    prompt=example.processed_prompt,
                    choice=choice,
                    env_name="code_r1",
                    env_example_id=example.example_id,
                    reward=reward,
                    correctness_reward=passed,
                    temperature=temperature,
                    top_k=top_k,
                    system_prompt=effective_system_prompt,
                )

                group_rollouts.append(rollout)
                total_choices += 1
                reward_sum += reward
                pass_sum += passed
                code_extracted_sum += code_extracted
                response_token_count += rollout.response_tokens.size

                if choice.finish_reason == "length":
                    truncated_count += 1

            if group_rollouts:
                rollout_groups.append(RolloutGroup(rollouts=group_rollouts))

        if total_choices == 0:
            raise RuntimeError("Inference context returned no choices; cannot compute metrics")

        prefix = f"code_r1.{mode}"
        metrics = {
            f"{prefix}_mean_reward": reward_sum / total_choices,
            f"{prefix}_pass_rate": pass_sum / total_choices,
            f"{prefix}_code_extracted_rate": code_extracted_sum / total_choices,
            f"{prefix}_mean_response_tokens": response_token_count / total_choices,
            f"{prefix}_total_responses": float(total_choices),
            f"{prefix}_sampled_examples": float(len(sampled_examples)),
            f"{prefix}_truncated_percentage": float(truncated_count) / total_choices,
        }

        return rollout_groups, metrics

    def _score_choice(self, example: CodeExample, response_text: str) -> tuple[float, float, float]:
        """Score a generated code response by executing it against test cases.

        Returns:
            Tuple of (reward, passed, code_extracted).
            - reward: 1.0 if all tests pass, 0.0 otherwise
            - passed: 1.0 if tests passed, 0.0 otherwise
            - code_extracted: 1.0 if code was successfully extracted, 0.0 otherwise
        """
        code = extract_python_code(response_text)

        if code is None:
            return 0.0, 0.0, 0.0

        passed, error_msg = execute_code_with_tests(code, example.test_cases, self.execution_timeout)

        if passed:
            return 1.0, 1.0, 1.0
        else:
            logger.debug(f"Code execution failed for {example.example_id}: {error_msg}")
            return 0.0, 0.0, 1.0

    def training_data(self) -> Iterator[CodeExample]:
        """Stream training examples for debugging."""
        yield from self.train_examples

    def eval_data(self) -> Iterator[CodeExample]:
        """Stream evaluation examples for debugging."""
        yield from self.eval_examples

    def get_eval_examples(self, n_examples: int) -> list[dict[str, Any]]:
        """Sample evaluation examples deterministically."""

        if not self.eval_examples:
            return []

        eval_key = jax.random.PRNGKey(42)
        n_to_sample = min(n_examples, len(self.eval_examples))
        indices = jax.random.choice(eval_key, len(self.eval_examples), shape=(n_to_sample,), replace=False)
        return [
            {
                "prompt": self.eval_examples[int(idx)].processed_prompt,
                "test_cases": self.eval_examples[int(idx)].test_cases,
                "example_id": self.eval_examples[int(idx)].example_id,
            }
            for idx in indices
        ]
