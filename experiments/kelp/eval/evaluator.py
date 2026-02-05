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

"""Evaluator for Kelp tree diffusion models."""

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import fsspec

from experiments.kelp.eval.config import KelpEvalTaskConfig, KelpEvaluationConfig
from experiments.kelp.eval.metrics import (
    PassAtKResult,
    ValidityResult,
    compute_pass_at_k,
    compute_validity_rate,
    execute_python_with_tests,
)

if TYPE_CHECKING:
    from experiments.kelp.model.model import TreeDiffusionModel

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for all evaluation results."""

    model_path: str
    config: KelpEvaluationConfig
    validity: ValidityResult | None = None
    mbpp_pass_at_1: PassAtKResult | None = None
    mbpp_pass_at_10: PassAtKResult | None = None
    humaneval_pass_at_1: PassAtKResult | None = None
    humaneval_pass_at_10: PassAtKResult | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "model_path": self.model_path,
        }
        if self.validity:
            result["validity"] = {
                "total": self.validity.total,
                "valid": self.validity.valid,
                "rate": self.validity.validity_rate,
            }
        for name, metric in [
            ("mbpp_pass_at_1", self.mbpp_pass_at_1),
            ("mbpp_pass_at_10", self.mbpp_pass_at_10),
            ("humaneval_pass_at_1", self.humaneval_pass_at_1),
            ("humaneval_pass_at_10", self.humaneval_pass_at_10),
        ]:
            if metric:
                result[name] = {
                    "k": metric.k,
                    "pass_rate": metric.pass_rate,
                    "total_problems": metric.total_problems,
                }
        return result


class TreeDiffusionEvaluator:
    """Evaluator for tree diffusion program synthesis models."""

    def __init__(
        self,
        model: "TreeDiffusionModel",
        config: KelpEvaluationConfig,
    ):
        self.model = model
        self.config = config

    def generate(
        self,
        prompts: list[str],
        num_samples: int = 1,
        max_iterations: int | None = None,
        temperature: float = 1.0,
    ) -> list[list[str]]:
        """Generate code samples for each prompt.

        Args:
            prompts: List of prompts (docstrings/signatures).
            num_samples: Number of samples to generate per prompt.
            max_iterations: Maximum diffusion iterations (uses model default if None).
            temperature: Sampling temperature.

        Returns:
            List of lists of generated code strings.
            results[i][j] = j-th sample for i-th prompt.
        """
        all_samples = []
        for prompt in prompts:
            samples = []
            for _ in range(num_samples):
                code = self.model.sample(
                    prompt,
                    max_iterations=max_iterations or self.model.config.num_diffusion_steps,
                    temperature=temperature,
                    use_grammar_constraints=self.config.use_grammar_constraints,
                )
                samples.append(code)
            all_samples.append(samples)
        return all_samples

    def evaluate_syntactic_validity(
        self,
        prompts: list[str],
        num_iterations: int | None = None,
    ) -> ValidityResult:
        """Evaluate syntactic validity rate.

        Generates one sample per prompt and checks if it parses as valid Python.

        Args:
            prompts: List of prompts to generate from.
            num_iterations: Maximum diffusion iterations.

        Returns:
            ValidityResult with validity rate and details.
        """
        logger.info(f"Evaluating syntactic validity on {len(prompts)} prompts")
        samples = self.generate(prompts, num_samples=1, max_iterations=num_iterations)
        generated_codes = [s[0] for s in samples]
        return compute_validity_rate(generated_codes, return_details=True)  # type: ignore

    def evaluate_pass_at_k(
        self,
        dataset: list[dict],
        k: int,
        n_samples: int,
        task_config: KelpEvalTaskConfig,
    ) -> PassAtKResult:
        """Evaluate pass@k on a code generation dataset.

        Args:
            dataset: List of problems with 'prompt' and 'test_cases' keys.
            k: k value for pass@k.
            n_samples: Number of samples to generate per problem.
            task_config: Task configuration.

        Returns:
            PassAtKResult with pass@k rate.
        """
        logger.info(f"Evaluating pass@{k} on {len(dataset)} problems (n={n_samples} samples each)")

        all_results = []
        for problem in dataset:
            prompt = problem["prompt"]
            test_cases = problem["test_cases"]

            samples = self.generate(
                [prompt],
                num_samples=n_samples,
                max_iterations=task_config.max_iterations,
                temperature=task_config.temperature,
            )[0]

            test_results = []
            for code in samples:
                passed = execute_python_with_tests(code, test_cases)
                test_results.append(all(passed))

            all_results.append(test_results)

        return compute_pass_at_k(all_results, k)

    def run_all_evals(self) -> EvaluationResults:
        """Run all configured evaluations.

        Returns:
            EvaluationResults with all computed metrics.
        """
        results = EvaluationResults(
            model_path=self.config.model_path,
            config=self.config,
        )

        for task in self.config.evals:
            if task.name == "validity":
                prompts = self._get_validity_prompts()
                if self.config.max_eval_instances:
                    prompts = prompts[: self.config.max_eval_instances]
                results.validity = self.evaluate_syntactic_validity(prompts, task.max_iterations)
                logger.info(f"Validity rate: {results.validity.validity_rate:.2%}")

            elif task.name == "mbpp":
                dataset = self._load_mbpp()
                if self.config.max_eval_instances:
                    dataset = dataset[: self.config.max_eval_instances]
                results.mbpp_pass_at_1 = self.evaluate_pass_at_k(dataset, k=1, n_samples=task.num_samples, task_config=task)
                results.mbpp_pass_at_10 = self.evaluate_pass_at_k(dataset, k=10, n_samples=task.num_samples, task_config=task)
                logger.info(f"MBPP pass@1: {results.mbpp_pass_at_1.pass_rate:.2%}")
                logger.info(f"MBPP pass@10: {results.mbpp_pass_at_10.pass_rate:.2%}")

            elif task.name == "humaneval":
                dataset = self._load_humaneval()
                if self.config.max_eval_instances:
                    dataset = dataset[: self.config.max_eval_instances]
                results.humaneval_pass_at_1 = self.evaluate_pass_at_k(dataset, k=1, n_samples=task.num_samples, task_config=task)
                results.humaneval_pass_at_10 = self.evaluate_pass_at_k(dataset, k=10, n_samples=task.num_samples, task_config=task)
                logger.info(f"HumanEval pass@1: {results.humaneval_pass_at_1.pass_rate:.2%}")
                logger.info(f"HumanEval pass@10: {results.humaneval_pass_at_10.pass_rate:.2%}")

        return results

    def _get_validity_prompts(self) -> list[str]:
        """Get prompts for validity evaluation.

        Returns a mix of simple and complex prompts.
        """
        return [
            'def add(a: int, b: int) -> int:\n    """Return the sum of two numbers."""',
            'def factorial(n: int) -> int:\n    """Compute factorial of n."""',
            'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""',
            'def is_prime(n: int) -> bool:\n    """Check if n is a prime number."""',
            'def reverse_string(s: str) -> str:\n    """Reverse a string."""',
            'def find_max(lst: list[int]) -> int:\n    """Find the maximum element in a list."""',
            'def binary_search(arr: list[int], target: int) -> int:\n    """Binary search for target in sorted array."""',
            'def merge_sort(arr: list[int]) -> list[int]:\n    """Sort array using merge sort."""',
            'def count_vowels(s: str) -> int:\n    """Count the number of vowels in a string."""',
            'def is_palindrome(s: str) -> bool:\n    """Check if string is a palindrome."""',
        ]

    def _load_mbpp(self) -> list[dict]:
        """Load MBPP dataset.

        Returns:
            List of problems with 'prompt' and 'test_cases' keys.
        """
        try:
            from datasets import load_dataset

            ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
            problems = []
            for item in ds:
                problems.append({
                    "prompt": item["text"],
                    "test_cases": item["test_list"],
                    "task_id": item["task_id"],
                })
            return problems
        except Exception as e:
            logger.warning(f"Could not load MBPP dataset: {e}")
            return []

    def _load_humaneval(self) -> list[dict]:
        """Load HumanEval dataset.

        Returns:
            List of problems with 'prompt' and 'test_cases' keys.
        """
        try:
            from datasets import load_dataset

            ds = load_dataset("openai/openai_humaneval", split="test")
            problems = []
            for item in ds:
                test_fn = item["test"]
                entry_point = item["entry_point"]
                test_cases = [f"{test_fn}\ncheck({entry_point})"]
                problems.append({
                    "prompt": item["prompt"],
                    "test_cases": test_cases,
                    "task_id": item["task_id"],
                })
            return problems
        except Exception as e:
            logger.warning(f"Could not load HumanEval dataset: {e}")
            return []


def save_results(results: EvaluationResults, output_path: str) -> None:
    """Save evaluation results to JSON file.

    Args:
        results: Evaluation results to save.
        output_path: Path to output file (local or GCS).
    """
    results_dict = results.to_dict()
    results_json = json.dumps(results_dict, indent=2)

    fs = fsspec.filesystem(output_path.split(":")[0] if "://" in output_path else "file")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        fs.makedirs(output_dir, exist_ok=True)

    with fs.open(output_path, "w") as f:
        f.write(results_json)

    logger.info(f"Saved evaluation results to {output_path}")
