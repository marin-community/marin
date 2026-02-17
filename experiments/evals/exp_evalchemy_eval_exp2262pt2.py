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
    HMMT,
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
CHECKPOINTS: dict[str, list[str]] = {
    # "exp2262pt2-qwen2.5-7b-instruct-finetuned-ot4-30k-math-qwq-32b-32768tokens": [
    #     "gs://marin-us-east5/checkpoints/exp2262pt2_sft_qwen2pt5_ot4_30k_math_qwq_32b_32768tokens-aaa2fa/hf/step-234/",  # Done: AIME24, AIME25, HMMT
    #     "gs://marin-us-east5/checkpoints/exp2262pt2_sft_qwen2pt5_ot4_30k_math_qwq_32b_32768tokens-aaa2fa/hf/step-468/",  # Done: AIME24, AIME25, HMMT
    #     "gs://marin-us-east5/checkpoints/exp2262pt2_sft_qwen2pt5_ot4_30k_math_qwq_32b_32768tokens-aaa2fa/hf/step-702/",  # Done: AIME24, AIME25, HMMT
    #     "gs://marin-us-east5/checkpoints/exp2262pt2_sft_qwen2pt5_ot4_30k_math_qwq_32b_32768tokens-aaa2fa/hf/step-936/",  # Done: AIME24, AIME25, HMMT
    #     "gs://marin-us-central1/checkpoints/exp2262pt2_sft_qwen2pt5_ot4_30k_math_qwq_32b_32768tokens-aaa2fa/hf/step-1170/",  # Done: AIME24, AIME25, HMMT
    #     "gs://marin-us-central1/checkpoints/exp2262pt2_sft_qwen2pt5_ot4_30k_math_qwq_32b_32768tokens-aaa2fa/hf/step-1404/",  # Done: AIME24, AIME25, HMMT
    #     "gs://marin-us-central1/checkpoints/exp2262pt2_sft_qwen2pt5_ot4_30k_math_qwq_32b_32768tokens-aaa2fa/hf/step-1638/",  # Done: AIME24, AIME25, HMMT
    #     "gs://marin-us-central1/checkpoints/exp2262pt2_sft_qwen2pt5_ot4_30k_math_qwq_32b_32768tokens-aaa2fa/hf/step-1872/",  # Done: AIME24, AIME25, HMMT
    # ],
    # "exp2262pt2a-qwen2.5-7b-instruct-finetuned-ot4-30k-math-qwen3-32b-32768tokens": [
    #     "gs://marin-us-central1/checkpoints/exp2262pt2a_sft_qwen2pt5_ot4_30k_math_qwen3_32b_32768tokens-56d459/hf/step-234/",  # Done: AIME24, AIME25, HMMT
    #     "gs://marin-us-central1/checkpoints/exp2262pt2a_sft_qwen2pt5_ot4_30k_math_qwen3_32b_32768tokens-56d459/hf/step-468/",  # Done: AIME24, AIME25, HMMT
    #     "gs://marin-us-central1/checkpoints/exp2262pt2a_sft_qwen2pt5_ot4_30k_math_qwen3_32b_32768tokens-56d459/hf/step-702/",  # Done: AIME24, AIME25, HMMT
    #     "gs://marin-us-central1/checkpoints/exp2262pt2a_sft_qwen2pt5_ot4_30k_math_qwen3_32b_32768tokens-56d459/hf/step-936/",  # Done: AIME24, AIME25, HMMT
    #     "gs://marin-us-central1/checkpoints/exp2262pt2a_sft_qwen2pt5_ot4_30k_math_qwen3_32b_32768tokens-56d459/hf/step-1170/",  # Done: AIME24, AIME25, HMMT
    #     "gs://marin-us-central1/checkpoints/exp2262pt2a_sft_qwen2pt5_ot4_30k_math_qwen3_32b_32768tokens-56d459/hf/step-1404/",  # Done: AIME24, AIME25, HMMT
    #     "gs://marin-us-central1/checkpoints/exp2262pt2a_sft_qwen2pt5_ot4_30k_math_qwen3_32b_32768tokens-56d459/hf/step-1638/",  # Done: AIME24, AIME25, HMMT
    #     "gs://marin-us-central1/checkpoints/exp2262pt2a_sft_qwen2pt5_ot4_30k_math_qwen3_32b_32768tokens-56d459/hf/step-1872/",  # Done: AIME24, AIME25, HMMT
    # ],
    "exp2262pt2c-qwen2.5-7b-instruct-finetuned-ot4-240k-math-qwen3-32b-32768tokens": [
        # "gs://marin-us-east5/checkpoints/exp2262pt2c_sft_qwen2pt5_ot4_240k_math_qwen3_32b_32768tokens-bd96f2/hf/step-468/",  # Done: AIME24, AIME25, HMMT
        # "gs://marin-us-central1/checkpoints/exp2262pt2c_sft_qwen2pt5_ot4_240k_math_qwen3_32b_32768tokens-bd96f2/hf/step-936/",  # Done: AIME24, AIME25, HMMT
        # "gs://marin-us-east5/checkpoints/exp2262pt2c_sft_qwen2pt5_ot4_240k_math_qwen3_32b_32768tokens-bd96f2/hf/step-1404/",  # Done: AIME24, AIME25, HMMT
        # "gs://marin-us-east5/checkpoints/exp2262pt2c_sft_qwen2pt5_ot4_240k_math_qwen3_32b_32768tokens-bd96f2/hf/step-1872/",  # Done: AIME24, AIME25, HMMT
        # "gs://marin-us-central1/checkpoints/exp2262pt2c_sft_qwen2pt5_ot4_240k_math_qwen3_32b_32768tokens-bd96f2/hf/step-3744/",  # Done: AIME24, AIME25, HMMT
        # "gs://marin-us-east5/checkpoints/exp2262pt2c_sft_qwen2pt5_ot4_240k_math_qwen3_32b_32768tokens-bd96f2/hf/step-5616/",  # Done: AIME24, AIME25, HMMT
        # "gs://marin-us-east5/checkpoints/exp2262pt2c_sft_qwen2pt5_ot4_240k_math_qwen3_32b_32768tokens-bd96f2/hf/step-7488/",  # Done: AIME24, AIME25, HMMT
        # "gs://marin-us-east5/checkpoints/exp2262pt2c_sft_qwen2pt5_ot4_240k_math_qwen3_32b_32768tokens-bd96f2/hf/step-9360/",  # Done: AIME24, AIME25, HMMT
        # "gs://marin-us-east5/checkpoints/exp2262pt2c_sft_qwen2pt5_ot4_240k_math_qwen3_32b_32768tokens-bd96f2/hf/step-11232/",  # Done: AIME24, AIME25, HMMT
        # "gs://marin-us-east5/checkpoints/exp2262pt2c_sft_qwen2pt5_ot4_240k_math_qwen3_32b_32768tokens-bd96f2/hf/step-13104/",  # Done: AIME24, AIME25, HMMT
        "gs://marin-us-east5/checkpoints/exp2262pt2c_sft_qwen2pt5_ot4_240k_math_qwen3_32b_32768tokens-bd96f2/hf/step-14976/",  # WIP
    ],
    # "exp2262pt2e-qwen2.5-7b-instruct-finetuned-ot4-154k-math-qwen3-32b-selfcons-32768tokens": [
    #     "gs://marin-us-central1/checkpoints/exp2262pt2e_sft_qwen2pt5_ot4_154k_math_qwen3_32b_selfcons_32768t-9bbe2e/hf/step-450/",  # Done: AIME24, AIME25, HMMT
    #     "gs://marin-us-central1/checkpoints/exp2262pt2e_sft_qwen2pt5_ot4_154k_math_qwen3_32b_selfcons_32768t-9bbe2e/hf/step-900/",  # Done: AIME24, AIME25, HMMT
    #     "gs://marin-us-central1/checkpoints/exp2262pt2e_sft_qwen2pt5_ot4_154k_math_qwen3_32b_selfcons_32768t-9bbe2e/hf/step-1350/",  # Done: AIME24, AIME25, HMMT
    #     "gs://marin-us-central1/checkpoints/exp2262pt2e_sft_qwen2pt5_ot4_154k_math_qwen3_32b_selfcons_32768t-9bbe2e/hf/step-1800/",  # Done: AIME24, AIME25, HMMT
    #     "gs://marin-us-central1/checkpoints/exp2262pt2e_sft_qwen2pt5_ot4_154k_math_qwen3_32b_selfcons_32768t-9bbe2e/hf/step-3750/",  # Done: AIME24, AIME25, HMMT
    #     "gs://marin-us-central1/checkpoints/exp2262pt2e_sft_qwen2pt5_ot4_154k_math_qwen3_32b_selfcons_32768t-9bbe2e/hf/step-5550/",  # Done: AIME24, AIME25, HMMT
    #     "gs://marin-us-east5/checkpoints/exp2262pt2e_sft_qwen2pt5_ot4_154k_math_qwen3_32b_selfcons_32768t-9bbe2e/hf/step-7500/",  # Done: AIME24, AIME25, HMMT
    #     "gs://marin-us-east5/checkpoints/exp2262pt2e_sft_qwen2pt5_ot4_154k_math_qwen3_32b_selfcons_32768t-9bbe2e/hf/step-9650/",  # Done: AIME24, AIME25, HMMT
    # ],
    "exp2262pt2_2-qwen3-8b-base-finetuned-ot4-30k-math-qwq-32b-32768tokens": [
        "gs://marin-us-central1/checkpoints/exp2262pt2_2_qwen3_base_ot4_30k_math_qwq_32b_32768tokens-fbbb10/hf/step-234/",  # WIP
        "gs://marin-us-central1/checkpoints/exp2262pt2_2_qwen3_base_ot4_30k_math_qwq_32b_32768tokens-fbbb10/hf/step-468/",  # WIP
        "gs://marin-us-central1/checkpoints/exp2262pt2_2_qwen3_base_ot4_30k_math_qwq_32b_32768tokens-fbbb10/hf/step-702/",  # WIP
        "gs://marin-us-central1/checkpoints/exp2262pt2_2_qwen3_base_ot4_30k_math_qwq_32b_32768tokens-fbbb10/hf/step-936/",  # WIP
        "gs://marin-us-central1/checkpoints/exp2262pt2_2_qwen3_base_ot4_30k_math_qwq_32b_32768tokens-fbbb10/hf/step-1170/",  # WIP
    ],
    "exp2262pt2a_2-qwen3-8b-base-finetuned-ot4-30k-math-qwen3-32b-32768tokens": [
        "gs://marin-us-central1/checkpoints/exp2262pt2a_2_qwen3_base_ot4_30k_math_qwen3_32b_32768tokens-9a60c1/hf/step-234/",  # WIP
        "gs://marin-us-central1/checkpoints/exp2262pt2a_2_qwen3_base_ot4_30k_math_qwen3_32b_32768tokens-9a60c1/hf/step-468/",  # WIP
        "gs://marin-us-central1/checkpoints/exp2262pt2a_2_qwen3_base_ot4_30k_math_qwen3_32b_32768tokens-9a60c1/hf/step-702/",  # WIP
        "gs://marin-us-central2/checkpoints/exp2262pt2a_2_qwen3_base_ot4_30k_math_qwen3_32b_32768tokens-9a60c1/hf/step-936/",  # WIP
        "gs://marin-us-central2/checkpoints/exp2262pt2a_2_qwen3_base_ot4_30k_math_qwen3_32b_32768tokens-9a60c1/hf/step-1170/",  # WIP
    ],
    "exp2262pt2c_2-qwen3-8b-base-finetuned-ot4-240k-math-qwen3-32b-32768tokens": [
        "gs://marin-us-central2/checkpoints/exp2262pt2c_2_sft_qwen3_base_ot4_240k_math_qwen3_32b_32768tokens-04c4d6/hf/step-936/",  # WIP
        "gs://marin-us-central2/checkpoints/exp2262pt2c_2_sft_qwen3_base_ot4_240k_math_qwen3_32b_32768tokens-04c4d6/hf/step-1872/",  # WIP
        "gs://marin-us-central2/checkpoints/exp2262pt2c_2_sft_qwen3_base_ot4_240k_math_qwen3_32b_32768tokens-04c4d6/hf/step-2808/",  # WIP
        "gs://marin-us-central2/checkpoints/exp2262pt2c_2_sft_qwen3_base_ot4_240k_math_qwen3_32b_32768tokens-04c4d6/hf/step-3744/",  # WIP
        "gs://marin-us-central2/checkpoints/exp2262pt2c_2_sft_qwen3_base_ot4_240k_math_qwen3_32b_32768tokens-04c4d6/hf/step-4681/",  # WIP
    ],
    "exp2262pt2e_2-qwen3-8b-base-finetuned-ot4-154k-math-qwen3-32b-selfcons-32768tokens": [
        "gs://marin-us-central2/checkpoints/exp2262pt2e_2_qwen3_base_ot4_154k_math_qwen3_32b_selfcons_32768t-b4ec2a/hf/step-602/",  # WIP
        "gs://marin-us-central2/checkpoints/exp2262pt2e_2_qwen3_base_ot4_154k_math_qwen3_32b_selfcons_32768t-b4ec2a/hf/step-1204/",  # WIP
        "gs://marin-us-central2/checkpoints/exp2262pt2e_2_qwen3_base_ot4_154k_math_qwen3_32b_selfcons_32768t-b4ec2a/hf/step-1806/",  # WIP
        "gs://marin-us-central2/checkpoints/exp2262pt2e_2_qwen3_base_ot4_154k_math_qwen3_32b_selfcons_32768t-b4ec2a/hf/step-2408/",  # WIP
        "gs://marin-us-central2/checkpoints/exp2262pt2e_2_qwen3_base_ot4_154k_math_qwen3_32b_selfcons_32768t-b4ec2a/hf/step-3015/",  # WIP
    ],
    "exp2262pt2_3-llama3.1-8b-instruct-finetuned-ot4-30k-math-qwq-32b-32768tokens": [
        "gs://marin-us-central1/checkpoints/exp2262pt2_3_llama3pt1_ot4_30k_math_qwq_32b_32768tokens-2fe4e8/hf/step-234/",  # WIP
        "gs://marin-us-central1/checkpoints/exp2262pt2_3_llama3pt1_ot4_30k_math_qwq_32b_32768tokens-2fe4e8/hf/step-468/",  # WIP
        "gs://marin-us-central1/checkpoints/exp2262pt2_3_llama3pt1_ot4_30k_math_qwq_32b_32768tokens-2fe4e8/hf/step-702/",  # WIP
        "gs://marin-us-central1/checkpoints/exp2262pt2_3_llama3pt1_ot4_30k_math_qwq_32b_32768tokens-2fe4e8/hf/step-936/",  # WIP
        "gs://marin-us-central1/checkpoints/exp2262pt2_3_llama3pt1_ot4_30k_math_qwq_32b_32768tokens-2fe4e8/hf/step-1170/",  # WIP
    ],
    "exp2262pt2a_3-llama3.1-8b-instruct-finetuned-ot4-30k-math-qwen3-32b-32768tokens": [
        "gs://marin-us-central1/checkpoints/exp2262pt2a_3_llama3pt1_ot4_30k_math_qwen3_32b_32768tokens-4e1ca5/hf/step-468/",  # WIP
        "gs://marin-us-central1/checkpoints/exp2262pt2a_3_llama3pt1_ot4_30k_math_qwen3_32b_32768tokens-4e1ca5/hf/step-702/",  # WIP
        "gs://marin-us-central1/checkpoints/exp2262pt2a_3_llama3pt1_ot4_30k_math_qwen3_32b_32768tokens-4e1ca5/hf/step-936/",  # WIP
        "gs://marin-us-central1/checkpoints/exp2262pt2a_3_llama3pt1_ot4_30k_math_qwen3_32b_32768tokens-4e1ca5/hf/step-1170/",  # WIP
    ],
    "exp2262pt2c_3-llama3.1-8b-instruct-finetuned-ot4-240k-math-qwen3-32b-32768tokens": [
        "gs://marin-us-central1/checkpoints/exp2262pt2c_3_llama3pt1_ot4_240k_math_qwen3_32b_32768tokens-9ac0db/hf/step-936/",  # WIP
        "gs://marin-us-central2/checkpoints/exp2262pt2c_3_llama3pt1_ot4_240k_math_qwen3_32b_32768tokens-9ac0db/hf/step-1872/",  # WIP
        # "gs://marin-us-central2/checkpoints/exp2262pt2c_3_llama3pt1_ot4_240k_math_qwen3_32b_32768tokens-9ac0db/hf/step-2808/",  # TODO
        # "gs://marin-us-central2/checkpoints/exp2262pt2c_3_llama3pt1_ot4_240k_math_qwen3_32b_32768tokens-9ac0db/hf/step-3744/",  # TODO
        # "gs://marin-us-central2/checkpoints/exp2262pt2c_3_llama3pt1_ot4_240k_math_qwen3_32b_32768tokens-9ac0db/hf/step-4681/",  # TODO
    ],
    "exp2262pt2e_3-llama3.1-8b-instruct-finetuned-ot4-154k-math-qwen3-32b-selfcons-32768tokens": [
        "gs://marin-us-central1/checkpoints/exp2262pt2e_3_llama3pt1_ot4_154k_math_qwen3_32b_selfcons_32768to-5f03e7/hf/step-603/",  # WIP
        "gs://marin-us-central1/checkpoints/exp2262pt2e_3_llama3pt1_ot4_154k_math_qwen3_32b_selfcons_32768to-5f03e7/hf/step-1206/",  # WIP
        # "gs://marin-us-central2/checkpoints/exp2262pt2e_3_llama3pt1_ot4_154k_math_qwen3_32b_selfcons_32768to-5f03e7/hf/step-1809/",  # TODO
        # "gs://marin-us-central2/checkpoints/exp2262pt2e_3_llama3pt1_ot4_154k_math_qwen3_32b_selfcons_32768to-5f03e7/hf/step-2412/",  # TODO
        # "gs://marin-us-central2/checkpoints/exp2262pt2e_3_llama3pt1_ot4_154k_math_qwen3_32b_selfcons_32768to-5f03e7/hf/step-3015/",  # TODO
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
TASK_SEED_GROUPS: list[tuple[list, list[int]]] = [
    ([AIME24, AIME25, HMMT], SEEDS[:5]),  # Hard math benchmarks: 5 seeds
    # ([AMC23], SEEDS[:5]),  # Easy math benchmarks: 5 seeds
    # ([MATH500], SEEDS[:1]),  # Easy math benchmarks: 1 seed
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
# v5p-8 has 4 chips, so we use tensor_parallel_size=4 to utilize all chips
# max_num_seqs: Batch size for parallel generation
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
MAX_PARALLEL_JOBS = 30

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    run_evalchemy_experiment(
        checkpoints=CHECKPOINTS,
        task_seed_groups=TASK_SEED_GROUPS,
        base_generation_params=BASE_GENERATION_PARAMS,
        resource_config=ResourceConfig.with_tpu("v5p-8"),
        engine_kwargs=ENGINE_KWARGS,
        apply_chat_template=True,
        discover_latest_checkpoint=DISCOVER_LATEST_CHECKPOINT,
        max_parallel_jobs=MAX_PARALLEL_JOBS,
    )
