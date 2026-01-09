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

"""HumanEvalPlus evaluation environment for code generation.

This module provides evaluation using the EvalPlus framework,
which extends HumanEval with 80x more tests for robust code evaluation.

Reference: https://github.com/evalplus/evalplus
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np

from marin.rl.environments.inference_ctx.base import BaseInferenceContext
from marin.rl.types import Rollout, RolloutGroup
from .base import MarinEnv, extract_seed

logger = logging.getLogger(__name__)

# HumanEval+ uses a simpler prompt format than Code-R1
# Just the function signature and docstring
HUMANEVAL_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Complete the given function based on the docstring. "
    "Only output the completed function, no explanations."
)


@dataclass(slots=True)
class HumanEvalExample:
    """Container for a single HumanEval problem."""

    task_id: str
    """HumanEval task ID (e.g., 'HumanEval/0')."""

    prompt: str
    """The function signature and docstring."""

    canonical_solution: str
    """The reference solution."""

    test: str
    """Test cases for the problem."""

    entry_point: str
    """The function name to call."""

    processed_prompt: str
    """Formatted prompt for the model."""

    metadata: dict[str, Any] = field(default_factory=dict)


def extract_function_code(response: str, entry_point: str) -> str | None:
    """Extract function code from a response.

    Args:
        response: The model's response.
        entry_point: The expected function name.

    Returns:
        The extracted function code or None.
    """
    # Try to find code in markdown blocks first
    python_block = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if python_block:
        return python_block.group(1).strip()

    generic_block = re.search(r"```\s*(.*?)```", response, re.DOTALL)
    if generic_block:
        return generic_block.group(1).strip()

    # Try to find the function definition
    func_pattern = rf"(def\s+{entry_point}\s*\(.*?(?=\ndef\s|\Z))"
    func_match = re.search(func_pattern, response, re.DOTALL)
    if func_match:
        return func_match.group(1).strip()

    # Fallback: return the whole response if it contains the function
    if f"def {entry_point}" in response:
        return response.strip()

    return None


def run_humaneval_tests(code: str, test: str, entry_point: str, timeout: int = 10) -> tuple[bool, str]:
    """Run HumanEval test cases against generated code.

    Args:
        code: The generated function code.
        test: The test code from HumanEval.
        entry_point: The function name.
        timeout: Execution timeout in seconds.

    Returns:
        Tuple of (passed, error_message).
    """
    if not code:
        return False, "No code provided"

    # Combine code and tests
    # HumanEval tests typically call check(entry_point)
    full_code = f"{code}\n\n{test}\n\ncheck({entry_point})"

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


class HumanEvalPlusEnv(MarinEnv):
    """HumanEvalPlus evaluation environment.

    Uses the EvalPlus framework for rigorous code evaluation.
    This environment is primarily for evaluation, not training.
    """

    def __init__(
        self,
        *,
        max_eval_examples: int | None = None,
        seed: int | None = None,
        execution_timeout: int = 10,
        use_evalplus: bool = True,
    ) -> None:
        """Initialize HumanEvalPlus environment.

        Args:
            max_eval_examples: Maximum number of eval examples to load.
            seed: Random seed for sampling.
            execution_timeout: Timeout for code execution in seconds.
            use_evalplus: If True, use evalplus library; otherwise use basic HumanEval.
        """
        self.max_eval_examples = max_eval_examples
        self._rng = np.random.default_rng(seed)
        self.execution_timeout = execution_timeout
        self.use_evalplus = use_evalplus
        self.system_prompt = HUMANEVAL_SYSTEM_PROMPT

        self.eval_examples = self._load_humaneval()

        logger.info(
            "Initialized HumanEvalPlusEnv with %d eval examples.",
            len(self.eval_examples),
        )

    def _load_humaneval(self) -> list[HumanEvalExample]:
        """Load HumanEval dataset."""
        examples = []

        try:
            if self.use_evalplus:
                # Try to use evalplus for extended test coverage
                from evalplus.data import get_human_eval_plus

                problems = get_human_eval_plus()
            else:
                # Fallback to basic HumanEval from datasets
                import datasets

                ds = datasets.load_dataset("openai/openai_humaneval", split="test")
                problems = {item["task_id"]: item for item in ds}
        except ImportError:
            logger.warning("evalplus not available, falling back to openai/openai_humaneval")
            import datasets

            ds = datasets.load_dataset("openai/openai_humaneval", split="test")
            problems = {item["task_id"]: item for item in ds}

        for task_id, problem in problems.items():
            example = HumanEvalExample(
                task_id=task_id,
                prompt=problem["prompt"],
                canonical_solution=problem.get("canonical_solution", ""),
                test=problem.get("test", ""),
                entry_point=problem.get("entry_point", ""),
                processed_prompt=problem["prompt"],  # HumanEval prompts are already formatted
                metadata={"task_id": task_id},
            )
            examples.append(example)

            if self.max_eval_examples and len(examples) >= self.max_eval_examples:
                break

        return examples

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
        """Sample prompts, generate code, and evaluate."""

        # HumanEvalPlus is eval-only
        if mode == "train":
            logger.warning("HumanEvalPlusEnv is intended for evaluation only, using eval mode")
            mode = "eval"

        available_examples = self.eval_examples
        if not available_examples:
            raise ValueError("No examples available")

        n_to_sample = min(n_examples, len(available_examples))
        seed = extract_seed(prng_key)
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(available_examples), size=n_to_sample, replace=False)
        sampled_examples = [available_examples[int(idx)] for idx in indices]

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
        pass_sum = 0.0
        code_extracted_sum = 0.0
        response_token_count = 0

        for example, completion in zip(sampled_examples, completions, strict=True):
            group_rollouts: list[Rollout] = []

            for choice in completion.choices:
                passed, code_extracted = self._score_choice(
                    example=example,
                    response_text=choice.message.content,
                )

                rollout = inference_ctx.create_rollout_from_choice(
                    prompt=example.processed_prompt,
                    choice=choice,
                    env_name="humaneval_plus",
                    env_example_id=example.task_id,
                    reward=passed,
                    correctness_reward=passed,
                    temperature=temperature,
                    top_k=top_k,
                    system_prompt=effective_system_prompt,
                )

                group_rollouts.append(rollout)
                total_choices += 1
                pass_sum += passed
                code_extracted_sum += code_extracted
                response_token_count += rollout.response_tokens.size

            if group_rollouts:
                rollout_groups.append(RolloutGroup(rollouts=group_rollouts))

        if total_choices == 0:
            raise RuntimeError("Inference context returned no choices")

        metrics = {
            "humaneval_plus.pass@1": pass_sum / total_choices,
            "humaneval_plus.code_extracted_rate": code_extracted_sum / total_choices,
            "humaneval_plus.mean_response_tokens": response_token_count / total_choices,
            "humaneval_plus.total_responses": float(total_choices),
            "humaneval_plus.sampled_examples": float(len(sampled_examples)),
        }

        return rollout_groups, metrics

    def _score_choice(self, example: HumanEvalExample, response_text: str) -> tuple[float, float]:
        """Score a generated response.

        Returns:
            Tuple of (passed, code_extracted).
        """
        code = extract_function_code(response_text, example.entry_point)

        if code is None:
            return 0.0, 0.0

        # Combine prompt (which has function signature) with generated code body
        # The model should complete the function, not redefine it
        full_code = example.prompt + code if not code.startswith("def ") else code

        passed, error_msg = run_humaneval_tests(full_code, example.test, example.entry_point, self.execution_timeout)

        if passed:
            return 1.0, 1.0
        else:
            logger.debug(f"Test failed for {example.task_id}: {error_msg}")
            return 0.0, 1.0

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
                "task_id": self.eval_examples[int(idx)].task_id,
                "entry_point": self.eval_examples[int(idx)].entry_point,
            }
            for idx in indices
        ]
