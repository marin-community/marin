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

"""MBPP-based evaluation for Kelp tree diffusion edit models.

Uses held-out MBPP programs (with real test cases) to evaluate program repair.
This provides a more representative and contamination-free eval compared to the
hand-crafted EVAL_TASKS in evaluate.py.

Pipeline:
1. Load MBPP programs and their assert-based test cases
2. Build subtree bank from training corpus (not eval programs)
3. For each program: corrupt via AST mutation, repair with best-of-N, test
4. Report per-program and aggregate metrics

Usage:
    uv run python experiments/kelp/evaluate_mbpp.py \\
        --checkpoint-dir checkpoints/kelp-edit-v3 \\
        --corpus-file experiments/kelp/corpus.txt
"""

import argparse
import ast
import json
import logging
import random
import sys
import time
from pathlib import Path

import jax

from experiments.kelp.checkpointing import find_best_checkpoint, load_checkpoint
from experiments.kelp.corpus import load_corpus
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


def load_mbpp_eval_tasks(max_length: int = 512, max_tasks: int = 0) -> list[dict]:
    """Load MBPP programs as eval tasks with executable test cases.

    Returns list of dicts with keys: task_id, text, clean, tests, setup_code.
    """
    from datasets import load_dataset

    tasks = []
    for split in ["train", "validation", "test", "prompt"]:
        try:
            ds = load_dataset("google-research-datasets/mbpp", "full", split=split, trust_remote_code=True)
        except Exception:
            continue

        for item in ds:
            code = item.get("code", "")
            test_list = item.get("test_list", [])
            setup_code = item.get("test_setup_code", "")

            if not code or not test_list:
                continue
            if len(code) > max_length or len(code) < 20:
                continue
            try:
                ast.parse(code)
            except SyntaxError:
                continue

            tasks.append(
                {
                    "task_id": item.get("task_id", len(tasks)),
                    "text": item.get("text", ""),
                    "clean": code,
                    "tests": test_list,
                    "setup_code": setup_code,
                }
            )

    if max_tasks > 0:
        tasks = tasks[:max_tasks]

    logger.info(f"Loaded {len(tasks)} MBPP eval tasks")
    return tasks


def run_mbpp_test(program: str, test_assert: str, setup_code: str = "") -> bool:
    """Execute an MBPP assert-based test case against a program."""
    try:
        namespace: dict = {}
        if setup_code:
            exec(setup_code, namespace)
        exec(program, namespace)
        exec(test_assert, namespace)
        return True
    except Exception:
        return False


def evaluate_mbpp_task(
    task: dict,
    params: EditModelParams,
    config: TreeDiffusionConfig,
    tokenizer: TreeDiffusionTokenizer,
    bank: SubtreeBank,
    key: jax.Array,
    num_corruptions: int = 5,
    corruption_steps: int = 3,
    n_best_of: int = 16,
    max_depth: int = 10,
) -> dict:
    """Evaluate a single MBPP task across multiple corruption/repair trials."""
    clean = task["clean"]
    tests = task["tests"]
    setup_code = task.get("setup_code", "")
    prompt = task.get("text") if tokenizer.prompt_tokens else None
    rng = random.Random(task["task_id"])

    total_valid = 0
    total_exact_match = 0
    total_test_pass_rate = 0.0
    total_candidates = 0
    total_trials = 0
    best_overall_pass_rate = 0.0
    best_overall_candidate = clean

    for _trial in range(num_corruptions):
        key, _corrupt_key, search_key = jax.random.split(key, 3)

        corrupted, _mutations = corrupt_program(
            clean,
            num_steps=corruption_steps,
            bank=bank,
            rng=rng,
        )

        if corrupted == clean:
            continue

        total_trials += 1

        candidates = best_of_n(
            params=params,
            source=corrupted,
            cfg=config,
            tokenizer=tokenizer,
            key=search_key,
            n=n_best_of,
            max_depth=max_depth,
            temperature=0.8,
            prompt=prompt,
        )

        for c in candidates:
            total_candidates += 1
            is_valid = True
            try:
                ast.parse(c.source)
            except SyntaxError:
                is_valid = False
            if is_valid:
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
                c_passed = sum(1 for t in tests if run_mbpp_test(c.source, t, setup_code))
                c_rate = c_passed / len(tests) if tests else 0.0
                if c_rate > best_trial_pass_rate:
                    best_trial_pass_rate = c_rate
                    best_trial_candidate = c.source
            total_test_pass_rate += best_trial_pass_rate

            if best_trial_pass_rate > best_overall_pass_rate:
                best_overall_pass_rate = best_trial_pass_rate
                best_overall_candidate = best_trial_candidate

    return {
        "task_id": task["task_id"],
        "text": task["text"][:100],
        "num_trials": total_trials,
        "total_candidates": total_candidates,
        "valid_rate": total_valid / max(total_candidates, 1),
        "exact_match_rate": total_exact_match / max(total_candidates, 1),
        "avg_test_pass_rate": total_test_pass_rate / max(total_trials, 1),
        "best_test_pass_rate": best_overall_pass_rate,
        "best_candidate": best_overall_candidate.strip()[:200],
        "clean": clean.strip()[:200],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Kelp checkpoint on held-out MBPP programs")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/kelp-edit-v3",
        help="Directory containing step-XXXXXX subdirectories",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint subdirectory (uses latest if not set)",
    )
    parser.add_argument(
        "--corpus-file",
        type=str,
        default=None,
        help="Training corpus file for building subtree bank (recommended for realistic corruption)",
    )
    parser.add_argument("--num-corruptions", type=int, default=5, help="Corruption trials per task")
    parser.add_argument("--corruption-steps", type=int, default=3, help="AST mutations per corruption")
    parser.add_argument("--n-best-of", type=int, default=16, help="Number of independent rollouts")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum edit depth")
    parser.add_argument("--max-tasks", type=int, default=50, help="Max MBPP tasks to evaluate (0=all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")
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
    tokenizer = TreeDiffusionTokenizer(max_seq_len=config.max_seq_len, prompt_tokens=config.prompt_tokens)

    # Load MBPP eval tasks.
    eval_tasks = load_mbpp_eval_tasks(max_tasks=args.max_tasks)
    if not eval_tasks:
        logger.error("No MBPP tasks loaded")
        return 1

    # Build subtree bank from training corpus if provided (issue #52),
    # otherwise fall back to building from eval programs.
    if args.corpus_file:
        corpus = load_corpus(args.corpus_file)
        logger.info(f"Building subtree bank from training corpus: {len(corpus)} programs")
        bank = SubtreeBank.from_corpus(corpus)
    else:
        eval_programs = [t["clean"] for t in eval_tasks]
        logger.info(f"Building subtree bank from {len(eval_programs)} eval programs (no --corpus-file)")
        bank = SubtreeBank.from_corpus(eval_programs)
    logger.info(f"Subtree bank: {bank.total_entries} entries across {len(bank.entries)} node types")

    key = jax.random.PRNGKey(args.seed)

    logger.info(f"Running MBPP evaluation: {len(eval_tasks)} tasks, {args.num_corruptions} corruptions each")
    logger.info(f"Inference: best-of-{args.n_best_of}, max_depth={args.max_depth}")
    logger.info("")

    all_results = []
    start_time = time.time()

    for i, task in enumerate(eval_tasks):
        key, task_key = jax.random.split(key)
        logger.info(f"[{i + 1}/{len(eval_tasks)}] task_id={task['task_id']}: {task['text'][:60]}")

        result = evaluate_mbpp_task(
            task=task,
            params=params,
            config=config,
            tokenizer=tokenizer,
            bank=bank,
            key=task_key,
            num_corruptions=args.num_corruptions,
            corruption_steps=args.corruption_steps,
            n_best_of=args.n_best_of,
            max_depth=args.max_depth,
        )
        all_results.append(result)

        logger.info(
            f"  valid={result['valid_rate']:.1%} exact={result['exact_match_rate']:.1%} "
            f"test_pass={result['avg_test_pass_rate']:.1%} best_pass={result['best_test_pass_rate']:.1%}"
        )

    elapsed = time.time() - start_time

    # Aggregate metrics.
    tasks_with_trials = [r for r in all_results if r["num_trials"] > 0]
    if not tasks_with_trials:
        logger.warning("No tasks had successful corruptions — cannot compute metrics")
        return 1

    avg_valid = sum(r["valid_rate"] for r in tasks_with_trials) / len(tasks_with_trials)
    avg_exact = sum(r["exact_match_rate"] for r in tasks_with_trials) / len(tasks_with_trials)
    avg_test_pass = sum(r["avg_test_pass_rate"] for r in tasks_with_trials) / len(tasks_with_trials)
    avg_best_pass = sum(r["best_test_pass_rate"] for r in tasks_with_trials) / len(tasks_with_trials)

    logger.info("")
    logger.info("=" * 70)
    logger.info("MBPP Evaluation Summary")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {ckpt_dir}")
    logger.info(f"Tasks evaluated: {len(tasks_with_trials)} (of {len(eval_tasks)} loaded)")
    logger.info(f"Corruptions/task: {args.num_corruptions}")
    logger.info(f"Inference: best-of-{args.n_best_of}, max_depth={args.max_depth}")
    logger.info(f"Subtree bank: {'training corpus' if args.corpus_file else 'eval programs only'}")
    logger.info(f"Time: {elapsed:.1f}s")
    logger.info("")
    logger.info(f"{'Metric':<30} {'Value':>10}")
    logger.info("-" * 42)
    logger.info(f"{'Syntactic validity rate':<30} {avg_valid:>10.1%}")
    logger.info(f"{'Exact match rate':<30} {avg_exact:>10.1%}")
    logger.info(f"{'Avg test pass rate':<30} {avg_test_pass:>10.1%}")
    logger.info(f"{'Best test pass rate':<30} {avg_best_pass:>10.1%}")
    logger.info("=" * 70)

    # Save results.
    output_path = args.output or str(ckpt_dir / "mbpp_eval_results.json")
    results_data = {
        "checkpoint": str(ckpt_dir),
        "config": {
            "num_corruptions": args.num_corruptions,
            "corruption_steps": args.corruption_steps,
            "n_best_of": args.n_best_of,
            "max_depth": args.max_depth,
            "max_tasks": args.max_tasks,
            "corpus_file": args.corpus_file,
        },
        "aggregate": {
            "tasks_evaluated": len(tasks_with_trials),
            "tasks_loaded": len(eval_tasks),
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
