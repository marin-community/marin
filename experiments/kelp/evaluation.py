#!/usr/bin/env python3
# Copyright 2026 The Marin Authors
from __future__ import annotations
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

"""
Evaluation module for Kelp tree diffusion models.

This module provides evaluation utilities that integrate with Marin's
ExecutorStep framework for running evaluations on HumanEval and other
code generation benchmarks.

Unlike standard LM evaluations that use lm-eval-harness via vLLM,
Kelp requires custom generation logic for tree diffusion.

Supported benchmarks:
- HumanEval: Function completion from signature + docstring
- MBPP: Similar format with more problems

Usage:
    from experiments.kelp.evaluation import evaluate_kelp_humaneval

    eval_step = evaluate_kelp_humaneval(
        model_path="outputs/kelp/checkpoint_final.pkl",
        output_path="outputs/kelp/eval_results",
    )
"""

import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Any

import jax.random as jrandom

from experiments.kelp.conditioning import CONDITION_VOCAB_SIZE
from experiments.kelp.generate import GenerationConfig, generate_function_body
from experiments.kelp.python_grammar import PythonNodeVocab, PythonValueVocab
from experiments.kelp.tree_diffusion import TreeDiffusionConfig, TreeDiffusionModel

logger = logging.getLogger(__name__)


@dataclass
class KelpEvalConfig:
    """Configuration for Kelp evaluation."""

    # Model
    model_path: str
    """Path to model checkpoint."""

    # Output
    output_path: str = "outputs/kelp/eval"
    """Path to write evaluation results."""

    # Evaluation settings
    num_samples: int = 1
    """Number of samples per problem (for pass@k)."""

    max_problems: int | None = None
    """Maximum problems to evaluate (None for all)."""

    # Generation settings
    max_generation_steps: int = 20
    """Maximum tree diffusion steps."""

    temperature: float = 0.8
    """Sampling temperature."""

    top_k: int = 50
    """Top-k sampling."""

    top_p: float = 0.95
    """Top-p (nucleus) sampling."""

    # Execution
    execution_timeout: float = 5.0
    """Timeout for code execution in seconds."""

    # Misc
    seed: int = 42
    """Random seed."""

    wandb_project: str | None = "kelp-eval"
    """W&B project for logging."""

    wandb_tags: list[str] = field(default_factory=list)
    """Tags for W&B run."""


@dataclass
class HumanEvalProblem:
    """A single HumanEval problem."""

    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str


def load_humaneval() -> list[HumanEvalProblem]:
    """Load HumanEval dataset from HuggingFace."""
    try:
        from datasets import load_dataset

        dataset = load_dataset("openai_humaneval", split="test")
        return [
            HumanEvalProblem(
                task_id=item["task_id"],
                prompt=item["prompt"],
                canonical_solution=item["canonical_solution"],
                test=item["test"],
                entry_point=item["entry_point"],
            )
            for item in dataset
        ]
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        return []
    except Exception as e:
        logger.error(f"Failed to load HumanEval: {e}")
        return []


def load_kelp_model(checkpoint_path: str) -> tuple[TreeDiffusionModel, TreeDiffusionConfig] | None:
    """Load Kelp model from checkpoint."""
    try:
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        config = checkpoint["config"]
        model_state = checkpoint["model"]

        # Initialize model structure
        key = jrandom.PRNGKey(0)
        model = TreeDiffusionModel.init(config, key=key)

        # Load weights (simplified - full implementation would use eqx.tree_deserialise)
        import equinox as eqx

        model = eqx.tree_at(
            lambda m: eqx.filter(m, eqx.is_array),
            model,
            model_state,
        )

        return model, config
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def execute_with_tests(
    code: str,
    test_code: str,
    entry_point: str,
    timeout: float = 5.0,
) -> tuple[bool, str]:
    """Execute code and check against tests."""
    import multiprocessing
    import signal

    full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})"

    def run():
        try:
            exec(full_code, {"__builtins__": __builtins__})
            return True, ""
        except AssertionError as e:
            return False, f"AssertionError: {e}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    try:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(1) as pool:
            result = pool.apply_async(run)
            try:
                return result.get(timeout=timeout)
            except multiprocessing.TimeoutError:
                return False, "Timeout"
    except Exception as e:
        return False, f"Execution error: {e}"


def compute_pass_at_k(results: list[list[bool]], k: int) -> float:
    """Compute pass@k metric."""
    from math import comb

    def single(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - comb(n - c, k) / comb(n, k)

    scores = []
    for r in results:
        n, c = len(r), sum(r)
        if n >= k:
            scores.append(single(n, c, k))

    return sum(scores) / len(scores) if scores else 0.0


def run_humaneval_evaluation(config: KelpEvalConfig) -> dict[str, Any]:
    """Run HumanEval evaluation.

    Args:
        config: Evaluation configuration

    Returns:
        Dictionary with evaluation results
    """
    # Load model
    result = load_kelp_model(config.model_path)
    if result is None:
        return {"error": "Failed to load model"}

    model, model_config = result

    # Load dataset
    problems = load_humaneval()
    if not problems:
        return {"error": "Failed to load HumanEval"}

    if config.max_problems:
        problems = problems[: config.max_problems]

    # Create vocabularies
    node_vocab = PythonNodeVocab()
    value_vocab = PythonValueVocab()

    # Generation config
    gen_config = GenerationConfig(
        max_steps=config.max_generation_steps,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
    )

    # Run evaluation
    key = jrandom.PRNGKey(config.seed)
    all_results = []
    detailed = []

    logger.info(f"Evaluating {len(problems)} problems with {config.num_samples} samples each")

    for i, problem in enumerate(problems):
        logger.info(f"[{i+1}/{len(problems)}] {problem.task_id}")

        problem_results = []

        for sample_idx in range(config.num_samples):
            key, sample_key = jrandom.split(key)

            # Generate
            body = generate_function_body(
                model,
                problem.prompt,
                node_vocab,
                value_vocab,
                gen_config,
                sample_key,
            )

            if body is None:
                problem_results.append(False)
                detailed.append({
                    "task_id": problem.task_id,
                    "sample": sample_idx,
                    "passed": False,
                    "error": "Generation failed",
                })
                continue

            # Combine with prompt
            full_code = f"{problem.prompt.rstrip()}\n{body}"

            # Execute
            passed, error = execute_with_tests(
                full_code,
                problem.test,
                problem.entry_point,
                config.execution_timeout,
            )

            problem_results.append(passed)
            detailed.append({
                "task_id": problem.task_id,
                "sample": sample_idx,
                "passed": passed,
                "error": error if not passed else None,
                "generated": full_code,
            })

        all_results.append(problem_results)

        if any(problem_results):
            logger.info(f"  ✓ Passed ({sum(problem_results)}/{len(problem_results)})")
        else:
            logger.info(f"  ✗ Failed")

    # Compute metrics
    pass_1 = compute_pass_at_k(all_results, 1)
    pass_10 = compute_pass_at_k(all_results, min(10, config.num_samples)) if config.num_samples >= 10 else None
    pass_100 = compute_pass_at_k(all_results, min(100, config.num_samples)) if config.num_samples >= 100 else None

    results = {
        "benchmark": "humaneval",
        "model_path": config.model_path,
        "num_problems": len(problems),
        "num_samples": config.num_samples,
        "pass@1": pass_1,
        "pass@10": pass_10,
        "pass@100": pass_100,
        "total_passed": sum(any(r) for r in all_results),
        "detailed_results": detailed,
    }

    return results


def evaluate_kelp_humaneval(
    model_path: str,
    output_path: str = "outputs/kelp/eval",
    num_samples: int = 1,
    max_problems: int | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Evaluate Kelp model on HumanEval.

    This is the main entry point for evaluation, designed to integrate
    with Marin's workflow.

    Args:
        model_path: Path to model checkpoint
        output_path: Path to write results
        num_samples: Samples per problem
        max_problems: Max problems to evaluate
        **kwargs: Additional config options

    Returns:
        Evaluation results dictionary
    """
    config = KelpEvalConfig(
        model_path=model_path,
        output_path=output_path,
        num_samples=num_samples,
        max_problems=max_problems,
        **kwargs,
    )

    logger.info("=" * 60)
    logger.info("Kelp HumanEval Evaluation")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_path}")
    logger.info(f"Samples: {config.num_samples}")

    results = run_humaneval_evaluation(config)

    # Save results
    os.makedirs(config.output_path, exist_ok=True)
    results_path = os.path.join(config.output_path, "humaneval_results.json")

    # Save without detailed results (too large)
    save_results = {k: v for k, v in results.items() if k != "detailed_results"}
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2)

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"pass@1: {results.get('pass@1', 0):.1%}")
    if results.get("pass@10"):
        logger.info(f"pass@10: {results['pass@10']:.1%}")
    logger.info(f"Results saved to: {results_path}")

    return results


# Integration with Marin's ExecutorStep (for future use)
def create_kelp_eval_step(
    model_path: str,
    output_path: str,
    num_samples: int = 1,
    max_problems: int | None = None,
):
    """Create an ExecutorStep for Kelp evaluation.

    This can be integrated with Marin's execution framework.

    Example:
        eval_step = create_kelp_eval_step(
            model_path="outputs/kelp/checkpoint_final.pkl",
            output_path="outputs/kelp/eval",
        )
        executor_main(steps=[eval_step])
    """
    from marin.execution.executor import ExecutorStep

    def run_eval():
        return evaluate_kelp_humaneval(
            model_path=model_path,
            output_path=output_path,
            num_samples=num_samples,
            max_problems=max_problems,
        )

    return ExecutorStep(
        name=f"kelp_eval/humaneval",
        fn=run_eval,
        config=None,
    )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", default="outputs/kelp/eval")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_problems", type=int, default=None)
    args = parser.parse_args()

    evaluate_kelp_humaneval(
        model_path=args.model_path,
        output_path=args.output_path,
        num_samples=args.num_samples,
        max_problems=args.max_problems,
    )
