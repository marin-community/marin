#!/usr/bin/env python3
# Copyright 2026 The Marin Authors
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
HumanEval evaluation for Kelp tree diffusion model.

This script evaluates a trained Kelp model on the HumanEval benchmark.
Unlike standard AR models that use lm-eval-harness, tree diffusion requires
a custom generation and evaluation pipeline.

HumanEval format:
- task_id: Unique identifier (e.g., "HumanEval/0")
- prompt: Function signature + docstring
- canonical_solution: Reference implementation
- test: Test code to verify correctness
- entry_point: Function name to call

Evaluation flow:
1. Parse prompt as conditioning (signature + docstring)
2. Generate function body using tree diffusion
3. Combine with signature to get complete function
4. Execute against test cases
5. Compute pass@k metrics

Usage:
    # Evaluate a checkpoint:
    uv run python experiments/kelp/eval_humaneval.py --checkpoint outputs/kelp/checkpoint_final.pkl

    # Quick test with fewer samples:
    uv run python experiments/kelp/eval_humaneval.py --num_samples=1 --max_problems=10
"""

import argparse
import ast
import json
import logging
import os
import pickle
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import haliax as hax

from experiments.kelp.conditioning import (
    CONDITION_VOCAB_SIZE,
    create_condition_mask,
    tokenize_condition,
)
from experiments.kelp.python_grammar import PythonNodeVocab, PythonValueVocab
from experiments.kelp.tree_diffusion import TreeDiffusionConfig, TreeDiffusionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class HumanEvalProblem:
    """A single HumanEval problem."""

    task_id: str
    prompt: str  # Function signature + docstring
    canonical_solution: str
    test: str
    entry_point: str


def load_humaneval_dataset() -> list[HumanEvalProblem]:
    """Load HumanEval dataset.

    First tries to load from HuggingFace datasets, then falls back to
    downloading directly.
    """
    try:
        from datasets import load_dataset

        logger.info("Loading HumanEval from HuggingFace datasets...")
        dataset = load_dataset("openai_humaneval", split="test")

        problems = []
        for item in dataset:
            problems.append(
                HumanEvalProblem(
                    task_id=item["task_id"],
                    prompt=item["prompt"],
                    canonical_solution=item["canonical_solution"],
                    test=item["test"],
                    entry_point=item["entry_point"],
                )
            )
        logger.info(f"Loaded {len(problems)} HumanEval problems")
        return problems

    except Exception as e:
        logger.warning(f"Could not load from HuggingFace: {e}")
        logger.info("Please install datasets: pip install datasets")
        return []


def extract_signature_and_docstring(prompt: str) -> tuple[str, str]:
    """Extract function signature and docstring from HumanEval prompt.

    HumanEval prompts have format:
        def function_name(args):
            '''docstring'''

    Returns:
        (signature, docstring) tuple
    """
    lines = prompt.strip().split("\n")

    # First line should be the signature
    signature = lines[0].strip()

    # Rest is the docstring (may be multi-line)
    docstring_lines = []
    in_docstring = False
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if in_docstring:
                # End of docstring
                if stripped.endswith('"""') or stripped.endswith("'''"):
                    docstring_lines.append(stripped[:-3])
                break
            else:
                # Start of docstring
                in_docstring = True
                content = stripped[3:]
                if content.endswith('"""') or content.endswith("'''"):
                    # Single-line docstring
                    docstring_lines.append(content[:-3])
                    break
                docstring_lines.append(content)
        elif in_docstring:
            docstring_lines.append(stripped)

    docstring = "\n".join(docstring_lines).strip()

    return signature, docstring


def generate_with_tree_diffusion(
    model: TreeDiffusionModel,
    condition_text: str,
    node_vocab: PythonNodeVocab,
    value_vocab: PythonValueVocab,
    num_steps: int = 10,
    key: jrandom.PRNGKey = None,
) -> str | None:
    """Generate code using tree diffusion.

    This is a simplified generation that starts with a minimal tree
    and applies denoising steps.

    For now, this is a placeholder that demonstrates the interface.
    Full implementation would require:
    1. Initialize with an empty/minimal AST matching the signature
    2. Run multiple denoising steps
    3. Convert final AST to code

    Args:
        model: Trained TreeDiffusionModel
        condition_text: Conditioning (signature + docstring)
        node_vocab: Node vocabulary
        value_vocab: Value vocabulary
        num_steps: Number of denoising steps
        key: Random key

    Returns:
        Generated code string, or None if generation failed
    """
    if key is None:
        key = jrandom.PRNGKey(42)

    config = model.config

    # For now, return a simple placeholder
    # TODO: Implement full tree diffusion generation
    # This would involve:
    # 1. Start with minimal tree (e.g., just "pass" or "return None")
    # 2. Run denoising: for each step, predict edit and apply it
    # 3. Stop when model predicts no more edits needed

    # Placeholder: return a simple body
    return "    pass"


def execute_code_with_tests(
    function_code: str,
    test_code: str,
    entry_point: str,
    timeout: float = 5.0,
) -> tuple[bool, str]:
    """Execute generated code against test cases.

    Args:
        function_code: Complete function code
        test_code: Test code from HumanEval
        entry_point: Function name to test
        timeout: Execution timeout in seconds

    Returns:
        (passed, error_message) tuple
    """
    import multiprocessing
    import signal

    # Combine function and tests
    full_code = f"{function_code}\n\n{test_code}\n\ncheck({entry_point})"

    def run_code():
        try:
            exec(full_code, {"__builtins__": __builtins__})
            return True, ""
        except AssertionError as e:
            return False, f"AssertionError: {e}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    # Run in subprocess with timeout
    try:
        # Use multiprocessing for timeout
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(1) as pool:
            result = pool.apply_async(run_code)
            try:
                passed, error = result.get(timeout=timeout)
                return passed, error
            except multiprocessing.TimeoutError:
                return False, "Timeout"
    except Exception as e:
        return False, f"Execution error: {e}"


def compute_pass_at_k(results: list[list[bool]], k: int) -> float:
    """Compute pass@k metric.

    pass@k = E[1 - C(n-c, k) / C(n, k)]

    where n = number of samples, c = number of correct samples

    Args:
        results: List of [list of pass/fail for each sample] per problem
        k: k value for pass@k

    Returns:
        pass@k score (0.0 to 1.0)
    """
    from math import comb

    def pass_at_k_single(n: int, c: int, k: int) -> float:
        """Compute pass@k for a single problem."""
        if n - c < k:
            return 1.0
        return 1.0 - comb(n - c, k) / comb(n, k)

    scores = []
    for problem_results in results:
        n = len(problem_results)
        c = sum(problem_results)
        if n >= k:
            scores.append(pass_at_k_single(n, c, k))

    return sum(scores) / len(scores) if scores else 0.0


def evaluate_humaneval(
    model: TreeDiffusionModel,
    problems: list[HumanEvalProblem],
    node_vocab: PythonNodeVocab,
    value_vocab: PythonValueVocab,
    num_samples: int = 1,
    max_problems: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate model on HumanEval.

    Args:
        model: Trained TreeDiffusionModel
        problems: HumanEval problems
        node_vocab: Node vocabulary
        value_vocab: Value vocabulary
        num_samples: Number of samples per problem (for pass@k)
        max_problems: Max problems to evaluate (None for all)
        seed: Random seed

    Returns:
        Dictionary with evaluation results
    """
    key = jrandom.PRNGKey(seed)

    if max_problems is not None:
        problems = problems[:max_problems]

    logger.info(f"Evaluating on {len(problems)} problems with {num_samples} samples each")

    all_results = []
    detailed_results = []

    for i, problem in enumerate(problems):
        logger.info(f"Problem {i + 1}/{len(problems)}: {problem.task_id}")

        # Extract conditioning
        signature, docstring = extract_signature_and_docstring(problem.prompt)
        condition_text = f"{signature}\n\"\"\"{docstring}\"\"\""

        problem_results = []

        for sample_idx in range(num_samples):
            key, sample_key = jrandom.split(key)

            # Generate code
            generated_body = generate_with_tree_diffusion(
                model,
                condition_text,
                node_vocab,
                value_vocab,
                key=sample_key,
            )

            if generated_body is None:
                problem_results.append(False)
                detailed_results.append({
                    "task_id": problem.task_id,
                    "sample": sample_idx,
                    "passed": False,
                    "error": "Generation failed",
                    "generated": None,
                })
                continue

            # Combine signature + generated body
            full_function = f"{problem.prompt.rstrip()}\n{generated_body}"

            # Execute tests
            passed, error = execute_code_with_tests(
                full_function,
                problem.test,
                problem.entry_point,
            )

            problem_results.append(passed)
            detailed_results.append({
                "task_id": problem.task_id,
                "sample": sample_idx,
                "passed": passed,
                "error": error if not passed else None,
                "generated": full_function,
            })

        all_results.append(problem_results)

        # Log progress
        if any(problem_results):
            logger.info(f"  ✓ Passed ({sum(problem_results)}/{len(problem_results)} samples)")
        else:
            logger.info(f"  ✗ Failed all samples")

    # Compute metrics
    pass_at_1 = compute_pass_at_k(all_results, k=1)
    pass_at_10 = compute_pass_at_k(all_results, k=min(10, num_samples)) if num_samples >= 10 else None
    pass_at_100 = compute_pass_at_k(all_results, k=min(100, num_samples)) if num_samples >= 100 else None

    total_passed = sum(any(r) for r in all_results)

    results = {
        "num_problems": len(problems),
        "num_samples": num_samples,
        "total_passed": total_passed,
        "pass_rate": total_passed / len(problems),
        "pass@1": pass_at_1,
        "pass@10": pass_at_10,
        "pass@100": pass_at_100,
        "detailed_results": detailed_results,
    }

    return results


def load_checkpoint(checkpoint_path: str) -> tuple[TreeDiffusionModel, TreeDiffusionConfig]:
    """Load model from checkpoint."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    config = checkpoint["config"]
    model_arrays = checkpoint["model"]

    # Reconstruct model
    key = jrandom.PRNGKey(0)
    model = TreeDiffusionModel.init(config, key=key)

    # This is a simplified approach - in practice you'd use eqx.tree_deserialize
    import equinox as eqx

    model = eqx.tree_at(
        lambda m: eqx.filter(m, eqx.is_array),
        model,
        model_arrays,
    )

    return model, config


def main():
    parser = argparse.ArgumentParser(description="Evaluate Kelp on HumanEval")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=1, help="Samples per problem")
    parser.add_argument("--max_problems", type=int, default=None, help="Max problems to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load HumanEval
    problems = load_humaneval_dataset()
    if not problems:
        logger.error("Could not load HumanEval dataset")
        return 1

    # Create vocabularies
    node_vocab = PythonNodeVocab()
    value_vocab = PythonValueVocab()

    # Load or create model
    if args.checkpoint and os.path.exists(args.checkpoint):
        model, config = load_checkpoint(args.checkpoint)
    else:
        logger.warning("No checkpoint provided, using random model (for testing)")
        config = TreeDiffusionConfig(
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            mlp_dim=512,
            max_nodes=128,
            node_vocab_size=node_vocab.vocab_size,
            value_vocab_size=value_vocab.vocab_size,
            use_conditioning=True,
            condition_vocab_size=CONDITION_VOCAB_SIZE,
            max_condition_len=128,
        )
        key = jrandom.PRNGKey(args.seed)
        model = TreeDiffusionModel.init(config, key=key)

    # Run evaluation
    logger.info("=" * 60)
    logger.info("Kelp HumanEval Evaluation")
    logger.info("=" * 60)

    results = evaluate_humaneval(
        model,
        problems,
        node_vocab,
        value_vocab,
        num_samples=args.num_samples,
        max_problems=args.max_problems,
        seed=args.seed,
    )

    # Print results
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Problems evaluated: {results['num_problems']}")
    logger.info(f"Samples per problem: {results['num_samples']}")
    logger.info(f"Total passed (any sample): {results['total_passed']}")
    logger.info(f"Pass rate: {results['pass_rate']:.1%}")
    logger.info(f"pass@1: {results['pass@1']:.1%}")
    if results['pass@10'] is not None:
        logger.info(f"pass@10: {results['pass@10']:.1%}")
    if results['pass@100'] is not None:
        logger.info(f"pass@100: {results['pass@100']:.1%}")

    # Save results
    if args.output:
        # Remove detailed results for JSON (too large)
        save_results = {k: v for k, v in results.items() if k != "detailed_results"}
        with open(args.output, "w") as f:
            json.dump(save_results, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
