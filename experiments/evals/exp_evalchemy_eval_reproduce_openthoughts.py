# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Example experiment for running Evalchemy reasoning benchmarks.

Evalchemy (https://github.com/mlfoundations/evalchemy) provides specialized
reasoning tasks including AIME24/25, MATH500, HumanEval+, MBPP+, and more.
"""

from experiments.evals.evals import run_evalchemy_experiment
from experiments.evals.evalchemy_task_configs import (
    AIME24,
    AIME25,
    AMC23,
    CODEELO,
    CODEFORCES,
    GPQA_DIAMOND,
    HMMT,
    HUMANITYS_LAST_EXAM,
    JEEBENCH,
    LIVECODEBENCH,
    MATH500,
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
# Set the key to None to use the default auto-generated names.
CHECKPOINTS: dict[str | None, list[str]] = {
    None: [
        "open-thoughts/OpenThinker3-7B",
    ],
    # None: [
    #     "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    # ],
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
TASK_SEED_GROUPS: list[tuple[list, list[int]]] = [
    ([AIME24, AIME25, AMC23, HMMT], SEEDS[:10]),  # 10 seeds
    ([LIVECODEBENCH], SEEDS[:6]),  # 6 seeds
    ([CODEFORCES, CODEELO, GPQA_DIAMOND, JEEBENCH, HUMANITYS_LAST_EXAM], SEEDS[:3]),  # 3 seeds
    ([MATH500], SEEDS[:1]),  # 1 seed
]

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
# v6e-8 has 4 chips, so we use tensor_parallel_size=4 to utilize all chips
# max_num_seqs: Batch size for parallel generation
BATCH_SIZE = 256
ENGINE_KWARGS = {
    "tensor_parallel_size": (
        4
    ),  # 8 chips on v6e-8, but OpenThinker3-7B has 28 attention heads so 4 is largest possible given 2x4 TPU topology
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
MAX_PARALLEL_JOBS = 30

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    run_evalchemy_experiment(
        checkpoints=CHECKPOINTS,
        task_seed_groups=TASK_SEED_GROUPS,
        base_generation_params=BASE_GENERATION_PARAMS,
        resource_config=ResourceConfig.with_tpu("v6e-8"),
        engine_kwargs=ENGINE_KWARGS,
        apply_chat_template=True,
        discover_latest_checkpoint=DISCOVER_LATEST_CHECKPOINT,
        max_parallel_jobs=MAX_PARALLEL_JOBS,
    )
