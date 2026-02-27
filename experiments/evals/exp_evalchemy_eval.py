# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Example experiment for running Evalchemy reasoning benchmarks.

Evalchemy (https://github.com/mlfoundations/evalchemy) provides specialized
reasoning tasks including AIME24/25, MATH500, HumanEval+, MBPP+, and more.
"""
import argparse
import sys

from experiments.evals.evals import run_evalchemy_experiment
from experiments.evals.evalchemy_task_configs import (
    AIME24,
    AIME25,
    AMC23,
    CODEELO,
    CODEFORCES,
    GPQA_DIAMOND,
    HMMT,
    HUMANEVAL_PLUS,
    HUMANITYS_LAST_EXAM,
    JEEBENCH,
    LIVECODEBENCH,
    MATH500,
    MBPP_PLUS,
)
from fray.cluster import ResourceConfig

# =============================================================================
# Model Configuration
# =============================================================================
# Dict mapping a base eval run name to a list of checkpoint paths.
# Each checkpoint is evaluated on all tasks below.
# The base eval run name is used for output paths and wandb run names:
#   Per-seed:  evalchemy-{base_eval_run_name}[-step{N}]-{task}-seed{S}
#   Aggregate: evalchemy-{base_eval_run_name}[-step{N}]-{task}-avg{X}seeds
# Step suffix is auto-extracted from each checkpoint path if it contains step-NNNN.
#
# These defaults are overridden when --exp_name and --checkpoint are provided via CLI.
DEFAULT_CHECKPOINTS: dict[str, list[str]] = {
    "exp2262pt2-qwen2.5-7b-instruct-finetuned-ot4-30k-math-qwq-32b-32768tokens": [
        "gs://marin-us-east5/checkpoints/exp2262pt2_sft_qwen2pt5_ot4_30k_math_qwq_32b_32768tokens-aaa2fa/hf/step-234/",
    ],
}

# Whether to auto-discover the latest checkpoint in a training run directory.
DISCOVER_LATEST_CHECKPOINT = False

# =============================================================================
# Evaluation Configuration
# =============================================================================

# Pool of seeds (first N are used per group).
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

# Each entry: (list of tasks, list of seeds).
# Tasks in the same group share the same set of seeds.
# Compile steps are automatically created for groups with >1 seed.

MATH_TASK_SEED_GROUPS: list[tuple[list, list[int]]] = [
    ([AIME24, AIME25, AMC23, HMMT], SEEDS[:5]),  # 5 seeds
    ([MATH500], SEEDS[:1]),  # 1 seed
]

SCIENCE_TASK_SEED_GROUPS: list[tuple[list, list[int]]] = [
    ([GPQA_DIAMOND], SEEDS[:3]),  # 3 seeds
    ([JEEBENCH, HUMANITYS_LAST_EXAM], SEEDS[:1]),  # 1 seed
]

CODE_TASK_SEED_GROUPS: list[tuple[list, list[int]]] = [
    ([HUMANEVAL_PLUS], SEEDS[:3]),  # 3 seeds
    ([MBPP_PLUS, CODEELO, CODEFORCES, LIVECODEBENCH], SEEDS[:1]),  # 1 seed
]

SUITE_TO_TASK_SEED_GROUPS: dict[str, list[tuple[list, list[int]]]] = {
    "math": MATH_TASK_SEED_GROUPS,
    "science": SCIENCE_TASK_SEED_GROUPS,
    "code": CODE_TASK_SEED_GROUPS,
}

# =============================================================================
# Generation Parameters
# =============================================================================
# - temperature: Sampling temperature (0 for greedy, >0 for sampling)
# - max_gen_toks: Maximum generation length
# - seed: Engine-level seed for reproducibility with temperature > 0
#
# Note: TPU/JAX doesn't support per-request seeds. To enable non-zero temperature
# with reproducibility, we use engine-level seed (passed to vLLM at initialization)
# rather than per-request seeds.
BASE_GENERATION_PARAMS = {
    "temperature": 0.7,
    "top_p": 1.0,
    "max_gen_toks": 32768,
}

# =============================================================================
# Engine Configuration
# =============================================================================
# tensor_parallel_size: Number of TPU chips to use for tensor parallelism
# max_num_seqs: Batch size for parallel generation
# RESOURCE = ResourceConfig.with_tpu("v5p-8")  # v5p-8 has 4 chips
RESOURCE = ResourceConfig.with_tpu("v4-8")  # v4-8 has 4 chips
BATCH_SIZE = 256
ENGINE_KWARGS = {
    "tensor_parallel_size": 4,
    "max_num_seqs": BATCH_SIZE,  # For vLLM: Enable batched generation for better throughput
    "batch_size": BATCH_SIZE,  # For lm-eval: Submit all requests at once for batched inference
}

# =============================================================================
# Parallel Job Limit
# =============================================================================
# Maximum number of eval jobs to run in parallel. Eval steps are split into
# batches of this size, with each batch submitted as a separate executor_main
# call. Set to None to run all eval steps in a single executor_main call.
# WARNING: Setting this value higher will cause many jobs to be launched in parallel.
#          Please be mindful of other users sharing the cluster.
MAX_PARALLEL_JOBS = 3

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evalchemy reasoning benchmarks")
    parser.add_argument("--experiment", type=str, default=None, help="Base eval run name for output paths and wandb")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (GCS path or HF model name)")
    parser.add_argument(
        "--suite",
        type=str,
        default="math",
        choices=["math", "science", "code", "all"],
        help="Eval suite to run (default: math)",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    if args.checkpoint:
        checkpoints = {args.experiment: [args.checkpoint]}
    else:
        parser.error("--checkpoint must be provided.")

    if args.suite == "all":
        suites = ["math", "science", "code"]
    else:
        suites = [args.suite]

    task_seed_groups = []
    for suite in suites:
        task_seed_groups.extend(SUITE_TO_TASK_SEED_GROUPS[suite])

    run_evalchemy_experiment(
        checkpoints=checkpoints,
        task_seed_groups=task_seed_groups,
        base_generation_params=BASE_GENERATION_PARAMS,
        resource_config=RESOURCE,
        engine_kwargs=ENGINE_KWARGS,
        apply_chat_template=True,
        discover_latest_checkpoint=DISCOVER_LATEST_CHECKPOINT,
        max_parallel_jobs=MAX_PARALLEL_JOBS,
    )
