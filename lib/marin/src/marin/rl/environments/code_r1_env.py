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
import concurrent.futures
import tempfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, NamedTuple
import time

import datasets
import jax
import numpy as np

from marin.rl.environments.inference_ctx.base import BaseInferenceContext
from marin.rl.types import Rollout, RolloutGroup
from .base import MarinEnv, extract_seed

logger = logging.getLogger(__name__)

LEETCODE_DATASET = "newfacade/LeetCodeDataset"

HUMANEVAL_DATASET = "evalplus/humanevalplus"

# EvalPlus prompt format (Standard EvalPlus behavior)
EVALPLUS_PROMPT_TEMPLATE = (
    "Please provide a self-contained Python script that solves the following problem in a markdown code block:\n"
    "```\n"
    "{problem}\n"
    "```\n"
)

EVALPLUS_RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that" " solves the problem and passes corresponding tests:"
)

# Default timeout for code execution (seconds)
CODE_EXECUTION_TIMEOUT = 30

# Code-R1 standard imports for LeetCode environment
# Source: code-r1/examples/data_preprocess/coder1.py
LEETCODE_IMPORTS = (
    "import heapq\n"
    "from math import floor, gcd\n"
    "import random\n"
    "import sys\n"
    "from typing import *\n"
    "from functools import *\n"
    "import collections\n"
    "from collections import *\n"
    "from itertools import *\n"
    "from heapq import *\n"
    "from bisect import *\n"
    "from string import *\n"
    "import math\n"
    "import datetime\n"
    "inf = float('inf')\n"
)


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

    entry_point: str | None = None
    """Function name to call for testing (HumanEval)."""


LoadDatasetFn = Callable[..., Any]


FORMAT_REWARD = 0.1
ANSWER_REWARD = 1.0


class ScoreResult(NamedTuple):
    """Result of scoring a generated code response."""

    reward: float
    is_correct: float
    code_extracted: float
    execution_time: float


def extract_python_code(response: str) -> str | None:
    """Extract Python code from a response within markdown code blocks."""
    # Extract code blocks
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)\n```", response, re.DOTALL)
    if code_blocks:
        return "\n".join(code_blocks).strip()

    return None


def execute_code_with_tests(
    code: str,
    test_cases: str,
    timeout: int = CODE_EXECUTION_TIMEOUT,
    entry_point: str | None = None,
) -> tuple[bool, str]:
    """Execute Python code with test cases in a sandboxed subprocess.

    Args:
        code: The Python code to execute.
        test_cases: Test case code that should run after the main code.
        timeout: Maximum execution time in seconds.
        entry_point: Optional function name to verify using check().

    Returns:
        Tuple of (success, error_message).
    """
    if not code:
        return False, "No code extracted"

    # Combine code and tests
    if entry_point:
        # HumanEval style: append check(entry_point)
        full_code = f"{code}\n\n{test_cases}\n\ncheck({entry_point})"
    else:
        # LeetCode style: tests are self-contained calls
        # Prepend standard Code-R1 imports (heapq, bisect, etc)
        full_code = f"{LEETCODE_IMPORTS}\n{code}\n\n{test_cases}"

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
        max_workers: int = 32,
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
            eval_dataset: Pre-loaded eval dataset (overrides eval_source).
            execution_timeout: Timeout for code execution in seconds.
            max_workers: Number of threads to use for parallel code execution.
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
        self.max_workers = max_workers
        self.system_prompt = None

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
            example = self._clean_example(item, f"{split_name}_{idx}", source)
            if example is not None:
                cleaned_examples.append(example)
                if limit is not None and len(cleaned_examples) >= limit:
                    break

        return cleaned_examples

    def _clean_example(self, item: dict[str, Any], example_id: str, source: str) -> CodeExample | None:
        """Convert a raw dataset item into a CodeExample."""

        # Detect dataset schema
        if "canonical_solution" in item:
            # HumanEvalPlus format
            problem = item["prompt"]
            test_cases = item["test"]
            solution = item["canonical_solution"]
            difficulty = None
            entry_point = item.get("entry_point")
            # Use task_id as example_id if available
            example_id = item.get("task_id", example_id)
        elif "problem_description" in item:
            # LeetCodeDataset format (new schema)
            problem = item["problem_description"]
            test_cases = item["test"]
            solution = item.get("completion")
            difficulty = item.get("difficulty")
            entry_point = item.get("entry_point")
            example_id = item.get("task_id", example_id)
        elif "content" in item:
            # LeetCodeDataset format (old schema/fallback)
            problem = item.get("content", "")
            test_cases = item.get("test", "")
            solution = item.get("python", None)
            difficulty = item.get("difficulty", None)
            entry_point = None
        else:
            # Unknown format
            logger.warning(f"Unknown dataset format for example {example_id}. Keys: {item.keys()}")
            return None

        if not problem or not test_cases:
            return None

        # Use EvalPlus standard instruction for all examples
        processed_prompt = EVALPLUS_PROMPT_TEMPLATE.format(problem=problem.strip())

        return CodeExample(
            problem=problem,
            test_cases=test_cases,
            solution=solution,
            processed_prompt=processed_prompt,
            example_id=example_id,
            difficulty=difficulty,
            entry_point=entry_point,
            metadata={"source": source},
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

        # Use EvalPlus assistant prefill for all examples
        effective_system_prompt = None
        stop_string = "\n```"
        prefill = f"{EVALPLUS_RESPONSE_PREFIX}\n```python\n"

        if stop is None:
            stop = [stop_string]
        elif stop_string not in stop:
            stop = [*list(stop), stop_string]

        prompts = [example.processed_prompt for example in sampled_examples]

        completions = inference_ctx.batch_completions(
            prompts=prompts,
            temperature=temperature,
            n=n_generations,
            max_tokens=max_tokens,
            top_k=top_k,
            stop=stop,
            system_prompt=effective_system_prompt,
            prefill=prefill,
        )

        rollout_groups: list[RolloutGroup] = []
        total_choices = 0
        reward_sum = 0.0
        pass_sum = 0.0
        code_extracted_sum = 0.0
        execution_time_sum = 0.0
        execution_times = []
        response_token_count = 0
        truncated_count = 0

        # Prepare tasks for parallel execution
        # We need to map back to the correct rollout group structure
        # Structure: tasks[group_idx][choice_idx] = assignment_future

        # Helper to preserve order and context
        @dataclass
        class ScoringTask:
            example: CodeExample
            full_response: str
            choice: Any
            group_idx: int
            choice_idx: int

        scoring_tasks: list[ScoringTask] = []

        # Intermediate structure to hold rollouts
        # group_rollouts_map[group_idx] = [None] * n_choices
        group_rollouts_map: list[list[Rollout | None]] = []

        for group_idx, (example, completion) in enumerate(zip(sampled_examples, completions, strict=True)):
            group_rollouts_map.append([None] * len(completion.choices))

            for choice_idx, choice in enumerate(completion.choices):
                # Prepend the prefill to the response content so the evaluator can parse it correctly
                # (e.g. including the opening ```python block)
                full_response = (prefill or "") + choice.message.content

                scoring_tasks.append(
                    ScoringTask(
                        example=example,
                        full_response=full_response,
                        choice=choice,
                        group_idx=group_idx,
                        choice_idx=choice_idx,
                    )
                )

        # Execute scoring in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # We map the _score_choice function over the tasks
            # We wrap it to handle the unpacking
            def score_task(task: ScoringTask) -> tuple[ScoringTask, ScoreResult]:
                result = self._score_choice(
                    example=task.example,
                    response_text=task.full_response,
                )
                return task, result

            results = executor.map(score_task, scoring_tasks)

        # Process results and build rollouts
        for task, score_result in results:
            rollout = inference_ctx.create_rollout_from_choice(
                prompt=task.example.processed_prompt,
                choice=task.choice,
                env_name="code_r1",
                env_example_id=task.example.example_id,
                reward=score_result.reward,
                correctness_reward=score_result.is_correct,
                temperature=temperature,
                top_k=top_k,
                system_prompt=effective_system_prompt,
            )

            # Place rollout in the correct position
            group_rollouts_map[task.group_idx][task.choice_idx] = rollout

            total_choices += 1
            reward_sum += score_result.reward
            pass_sum += score_result.is_correct
            code_extracted_sum += score_result.code_extracted
            execution_time_sum += score_result.execution_time
            execution_times.append(score_result.execution_time)
            response_token_count += rollout.response_tokens.size

            if task.choice.finish_reason == "length":
                truncated_count += 1

        # Construct final rollout groups
        for group_rollouts in group_rollouts_map:  # type: ignore
            if group_rollouts:
                # Filter out any Nones if something went wrong, though shouldn't happen
                valid_rollouts = [r for r in group_rollouts if r is not None]
                if valid_rollouts:
                    rollout_groups.append(RolloutGroup(rollouts=valid_rollouts))

        if total_choices == 0:
            raise RuntimeError("Inference context returned no choices; cannot compute metrics")

        prefix = f"code_r1.{mode}"
        metrics = {
            f"{prefix}_mean_reward": reward_sum / total_choices,
            f"{prefix}_pass_rate": pass_sum / total_choices,
            f"{prefix}_code_extracted_rate": code_extracted_sum / total_choices,
            f"{prefix}_mean_response_tokens": response_token_count / total_choices,
            f"{prefix}_mean_execution_time": execution_time_sum / total_choices,
            f"{prefix}_max_execution_time": max(execution_times) if execution_times else 0.0,
            f"{prefix}_min_execution_time": min(execution_times) if execution_times else 0.0,
            f"{prefix}_total_execution_time": execution_time_sum,
            f"{prefix}_total_responses": float(total_choices),
            f"{prefix}_sampled_examples": float(len(sampled_examples)),
            f"{prefix}_truncated_percentage": float(truncated_count) / total_choices,
        }

        return rollout_groups, metrics

    def _score_choice(self, example: CodeExample, response_text: str) -> ScoreResult:
        """Score a generated code response by executing it against test cases.

        Returns:
            ScoreResult containing reward, correctness, code extraction status, and execution time.
        """
        # Use unified extraction for all examples
        code = extract_python_code(response_text)
        if not code:
            return ScoreResult(
                reward=0.0,
                is_correct=0.0,
                code_extracted=0.0,
                execution_time=0.0,
            )

        start_time = time.perf_counter()
        passed, error_msg = execute_code_with_tests(
            code,
            example.test_cases,
            self.execution_timeout,
            entry_point=example.entry_point,
        )
        execution_time = time.perf_counter() - start_time

        if passed:
            return ScoreResult(
                reward=FORMAT_REWARD + ANSWER_REWARD,
                is_correct=1.0,
                code_extracted=1.0,
                execution_time=execution_time,
            )
        else:
            logger.debug(f"Code execution failed for {example.example_id}: {error_msg}")
            return ScoreResult(
                reward=FORMAT_REWARD,
                is_correct=0.0,
                code_extracted=1.0,
                execution_time=execution_time,
            )

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
