# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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

"""Evaluation script for Kelp tree diffusion edit models.

Loads a trained checkpoint and evaluates it on held-out program repair tasks:
1. Start from a corrupted program (produced by AST subtree replacement)
2. Use beam search / best-of-N to generate candidate repairs
3. Score candidates by execution against test cases
4. Report metrics: syntactic validity, edit precision, test pass rate

Usage:
    uv run python experiments/kelp/evaluate.py --checkpoint-dir checkpoints/kelp-edit
    uv run python experiments/kelp/evaluate.py --checkpoint-dir checkpoints/kelp-edit --best-checkpoint
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import jax

from experiments.kelp.checkpointing import find_best_checkpoint, load_checkpoint
from experiments.kelp.corpus import is_valid_python, load_corpus
from experiments.kelp.model.config import TreeDiffusionConfig
from experiments.kelp.tree.beam_search import best_of_n
from experiments.kelp.tree.edit_model import EditModelParams
from experiments.kelp.tree.mutation import corrupt_program
from experiments.kelp.tree.subtree_bank import SubtreeBank
from experiments.kelp.tree.tokenizer import TreeDiffusionTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Evaluation tasks: (clean_program, test_cases)
# Each test case is (input_expression, expected_output).
EVAL_TASKS = [
    {
        "name": "add",
        "clean": "def add(a, b):\n    return a + b\n",
        "tests": [
            ("add(1, 2)", "3"),
            ("add(0, 0)", "0"),
            ("add(-1, 1)", "0"),
            ("add(10, 20)", "30"),
        ],
    },
    {
        "name": "sub",
        "clean": "def sub(a, b):\n    return a - b\n",
        "tests": [
            ("sub(5, 3)", "2"),
            ("sub(0, 0)", "0"),
            ("sub(1, 5)", "-4"),
        ],
    },
    {
        "name": "mul",
        "clean": "def mul(a, b):\n    return a * b\n",
        "tests": [
            ("mul(3, 4)", "12"),
            ("mul(0, 5)", "0"),
            ("mul(-2, 3)", "-6"),
        ],
    },
    {
        "name": "neg",
        "clean": "def neg(x):\n    return -x\n",
        "tests": [
            ("neg(5)", "-5"),
            ("neg(-3)", "3"),
            ("neg(0)", "0"),
        ],
    },
    {
        "name": "abs_val",
        "clean": "def abs_val(x):\n    if x < 0:\n        return -x\n    return x\n",
        "tests": [
            ("abs_val(5)", "5"),
            ("abs_val(-3)", "3"),
            ("abs_val(0)", "0"),
        ],
    },
    {
        "name": "max_val",
        "clean": "def max_val(a, b):\n    if a > b:\n        return a\n    return b\n",
        "tests": [
            ("max_val(3, 5)", "5"),
            ("max_val(5, 3)", "5"),
            ("max_val(4, 4)", "4"),
        ],
    },
    {
        "name": "min_val",
        "clean": "def min_val(a, b):\n    if a < b:\n        return a\n    return b\n",
        "tests": [
            ("min_val(3, 5)", "3"),
            ("min_val(5, 3)", "3"),
            ("min_val(4, 4)", "4"),
        ],
    },
    {
        "name": "clamp",
        "clean": (
            "def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x\n"
        ),
        "tests": [
            ("clamp(5, 1, 10)", "5"),
            ("clamp(-1, 0, 10)", "0"),
            ("clamp(15, 0, 10)", "10"),
        ],
    },
    {
        "name": "double",
        "clean": "def double(x):\n    return x + x\n",
        "tests": [
            ("double(3)", "6"),
            ("double(0)", "0"),
            ("double(-2)", "-4"),
        ],
    },
    {
        "name": "square",
        "clean": "def square(x):\n    return x * x\n",
        "tests": [
            ("square(3)", "9"),
            ("square(0)", "0"),
            ("square(-2)", "4"),
        ],
    },
]


def run_test(program: str, call_expr: str, expected: str) -> bool:
    """Execute a program and test case, return whether it passes."""
    try:
        namespace: dict = {}
        exec(program, namespace)
        result = eval(call_expr, namespace)
        return str(result) == expected
    except Exception:
        return False


def evaluate_task(
    task: dict,
    params: EditModelParams,
    config: TreeDiffusionConfig,
    tokenizer: TreeDiffusionTokenizer,
    bank: SubtreeBank,
    key: jax.Array,
    num_corruptions: int = 5,
    corruption_steps: int = 3,
    beam_size: int = 8,
    max_depth: int = 10,
    n_best_of: int = 16,
) -> dict:
    """Evaluate a single task with multiple corruption/repair trials.

    Returns a dict of metrics:
    - valid_rate: fraction of generated programs that are syntactically valid
    - exact_match_rate: fraction that exactly match the clean source
    - test_pass_rate: fraction of tests passed by the best candidate
    - best_candidate: best repair found
    - num_trials: number of corruption/repair trials
    """
    clean = task["clean"]
    tests = task["tests"]
    rng = random.Random(42)

    total_valid = 0
    total_exact_match = 0
    total_test_pass_rate = 0.0
    total_candidates = 0
    total_trials = 0
    best_overall_pass_rate = 0.0
    best_overall_candidate = clean

    for _trial in range(num_corruptions):
        key, _corrupt_key, search_key = jax.random.split(key, 3)

        # Corrupt the clean program.
        corrupted, _mutations = corrupt_program(
            clean,
            num_steps=corruption_steps,
            bank=bank,
            rng=rng,
        )

        if corrupted == clean:
            continue

        total_trials += 1

        # Run best-of-N.
        candidates = best_of_n(
            params=params,
            source=corrupted,
            cfg=config,
            tokenizer=tokenizer,
            key=search_key,
            n=n_best_of,
            max_depth=max_depth,
            temperature=0.8,
        )

        for c in candidates:
            total_candidates += 1
            if is_valid_python(c.source):
                total_valid += 1
            if c.source.strip() == clean.strip():
                total_exact_match += 1

        # Test all candidates and pick the one that passes the most tests.
        # This is execution-guided reranking: the model generates diverse
        # candidates and we select by functional correctness.
        if candidates:
            best_trial_pass_rate = 0.0
            best_trial_candidate = candidates[0].source
            for c in candidates:
                c_passed = sum(1 for call, exp in tests if run_test(c.source, call, exp))
                c_rate = c_passed / len(tests) if tests else 0.0
                if c_rate > best_trial_pass_rate or (c_rate == best_trial_pass_rate and c.score > candidates[0].score):
                    best_trial_pass_rate = c_rate
                    best_trial_candidate = c.source
            total_test_pass_rate += best_trial_pass_rate

            if best_trial_pass_rate > best_overall_pass_rate:
                best_overall_pass_rate = best_trial_pass_rate
                best_overall_candidate = best_trial_candidate

    return {
        "name": task["name"],
        "num_trials": total_trials,
        "total_candidates": total_candidates,
        "valid_rate": total_valid / max(total_candidates, 1),
        "exact_match_rate": total_exact_match / max(total_candidates, 1),
        "avg_test_pass_rate": total_test_pass_rate / max(total_trials, 1),
        "best_test_pass_rate": best_overall_pass_rate,
        "best_candidate": best_overall_candidate.strip(),
        "clean": clean.strip(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Kelp tree diffusion checkpoint")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/kelp-edit",
        help="Directory containing step-XXXXXX subdirectories",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint subdirectory (e.g., step-012000). Uses latest if not set.",
    )
    parser.add_argument("--num-corruptions", type=int, default=5, help="Corruption trials per task")
    parser.add_argument("--corruption-steps", type=int, default=3, help="AST mutations per corruption")
    parser.add_argument("--beam-size", type=int, default=8, help="Beam size for search")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum edit depth")
    parser.add_argument("--n-best-of", type=int, default=16, help="Number of independent rollouts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")
    parser.add_argument(
        "--corpus-file",
        type=str,
        default=None,
        help="Training corpus file for building subtree bank (produces more realistic corruptions)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_base = Path(args.checkpoint_dir)

    if args.checkpoint:
        ckpt_dir = checkpoint_base / args.checkpoint
    else:
        ckpt_dir = find_best_checkpoint(checkpoint_base)
        if ckpt_dir is None:
            logger.error(f"No checkpoints found in {checkpoint_base}")
            return 1

    logger.info(f"Evaluating checkpoint: {ckpt_dir}")

    params, config = load_checkpoint(ckpt_dir)
    tokenizer = TreeDiffusionTokenizer(max_seq_len=config.max_seq_len)

    # Build subtree bank: prefer training corpus for realistic corruption.
    if args.corpus_file:
        corpus = load_corpus(args.corpus_file)
        logger.info(f"Building subtree bank from training corpus: {len(corpus)} programs")
        bank = SubtreeBank.from_corpus(corpus)
    else:
        eval_programs = [t["clean"] for t in EVAL_TASKS]
        logger.info("Building subtree bank from eval programs only (pass --corpus-file for better corruption)")
        bank = SubtreeBank.from_corpus(eval_programs)
    logger.info(f"Subtree bank: {bank.total_entries} entries across {len(bank.entries)} node types")

    key = jax.random.PRNGKey(args.seed)

    logger.info(f"Running evaluation: {len(EVAL_TASKS)} tasks, {args.num_corruptions} corruptions each")
    logger.info(f"Inference: best-of-{args.n_best_of}, max_depth={args.max_depth}")
    logger.info("")

    all_results = []
    start_time = time.time()

    for i, task in enumerate(EVAL_TASKS):
        key, task_key = jax.random.split(key)
        logger.info(f"[{i + 1}/{len(EVAL_TASKS)}] Evaluating: {task['name']}")

        result = evaluate_task(
            task=task,
            params=params,
            config=config,
            tokenizer=tokenizer,
            bank=bank,
            key=task_key,
            num_corruptions=args.num_corruptions,
            corruption_steps=args.corruption_steps,
            beam_size=args.beam_size,
            max_depth=args.max_depth,
            n_best_of=args.n_best_of,
        )
        all_results.append(result)

        logger.info(
            f"  valid={result['valid_rate']:.1%} exact={result['exact_match_rate']:.1%} "
            f"test_pass={result['avg_test_pass_rate']:.1%} best_pass={result['best_test_pass_rate']:.1%}"
        )

    elapsed = time.time() - start_time

    # Aggregate metrics.
    avg_valid = sum(r["valid_rate"] for r in all_results) / len(all_results)
    avg_exact = sum(r["exact_match_rate"] for r in all_results) / len(all_results)
    avg_test_pass = sum(r["avg_test_pass_rate"] for r in all_results) / len(all_results)
    avg_best_pass = sum(r["best_test_pass_rate"] for r in all_results) / len(all_results)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Evaluation Summary")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {ckpt_dir}")
    logger.info(f"Tasks: {len(EVAL_TASKS)}, Corruptions/task: {args.num_corruptions}")
    logger.info(f"Inference: best-of-{args.n_best_of}, max_depth={args.max_depth}")
    logger.info(f"Time: {elapsed:.1f}s")
    logger.info("")
    logger.info(f"{'Metric':<30} {'Value':>10}")
    logger.info("-" * 42)
    logger.info(f"{'Syntactic validity rate':<30} {avg_valid:>10.1%}")
    logger.info(f"{'Exact match rate':<30} {avg_exact:>10.1%}")
    logger.info(f"{'Avg test pass rate':<30} {avg_test_pass:>10.1%}")
    logger.info(f"{'Best test pass rate':<30} {avg_best_pass:>10.1%}")
    logger.info("")

    # Per-task breakdown.
    logger.info(f"{'Task':<12} {'Valid':>7} {'Exact':>7} {'AvgPass':>8} {'BestPass':>9}")
    logger.info("-" * 46)
    for r in all_results:
        logger.info(
            f"{r['name']:<12} {r['valid_rate']:>7.1%} {r['exact_match_rate']:>7.1%} "
            f"{r['avg_test_pass_rate']:>8.1%} {r['best_test_pass_rate']:>9.1%}"
        )
    logger.info("=" * 70)

    # Save results if requested.
    output_path = args.output or str(ckpt_dir / "eval_results.json")
    results_data = {
        "checkpoint": str(ckpt_dir),
        "config": {
            "num_corruptions": args.num_corruptions,
            "corruption_steps": args.corruption_steps,
            "n_best_of": args.n_best_of,
            "max_depth": args.max_depth,
        },
        "aggregate": {
            "syntactic_validity": avg_valid,
            "exact_match": avg_exact,
            "avg_test_pass_rate": avg_test_pass,
            "best_test_pass_rate": avg_best_pass,
        },
        "per_task": all_results,
        "elapsed_seconds": elapsed,
    }

    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
