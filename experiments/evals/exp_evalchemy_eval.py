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

"""
Example experiment for running Evalchemy reasoning benchmarks.

Evalchemy (https://github.com/mlfoundations/evalchemy) provides specialized
reasoning tasks including AIME24/25, MATH500, HumanEval+, MBPP+, and more.

Usage:
    # Single run (1 seed):
    python experiments/evals/exp_evalchemy_eval.py

    # Multiple seeds with aggregation:
    # Set SEEDS to [42, 43, 44, ...] for multiple iterations
"""

from experiments.evals.evals import default_evalchemy_eval, compile_evalchemy_results
# Import all task configs for easy access when modifying EVAL_TASKS
from experiments.evals.evalchemy_task_configs import (  # noqa: F401
    # Math tasks
    AIME24,
    AIME25,
    AMC23,
    MATH500,
    HMMT,
    # Code tasks
    HUMANEVAL_PLUS,
    MBPP_PLUS,
    LIVECODEBENCH,
    BIGCODEBENCH,
    CODEFORCES,
    CODEELO,
    # Science tasks
    GPQA_DIAMOND,
    JEEBENCH,
    # Reasoning tasks
    ALICE_IN_WONDERLAND,
    HUMANITYS_LAST_EXAM,
    # Task groups
    EVALCHEMY_MATH_TASKS,
    EVALCHEMY_CODE_TASKS,
    EVALCHEMY_SCIENCE_TASKS,
    EVALCHEMY_REASONING_TASKS,
    EVALCHEMY_CORE_TASKS,
)
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

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
    #     "gs://marin-us-east5/checkpoints/exp2262pt2_sft_qwen2pt5_ot4_30k_math_qwq_32b_32768tokens-aaa2fa/hf/step-234/",
    #     "gs://marin-us-east5/checkpoints/exp2262pt2_sft_qwen2pt5_ot4_30k_math_qwq_32b_32768tokens-aaa2fa/hf/step-468/",
    #     "gs://marin-us-east5/checkpoints/exp2262pt2_sft_qwen2pt5_ot4_30k_math_qwq_32b_32768tokens-aaa2fa/hf/step-702/",
    #     "gs://marin-us-east5/checkpoints/exp2262pt2_sft_qwen2pt5_ot4_30k_math_qwq_32b_32768tokens-aaa2fa/hf/step-936/",
    #     "gs://marin-us-central1/checkpoints/exp2262pt2_sft_qwen2pt5_ot4_30k_math_qwq_32b_32768tokens-aaa2fa/hf/step-1170/",
    #     "gs://marin-us-central1/checkpoints/exp2262pt2_sft_qwen2pt5_ot4_30k_math_qwq_32b_32768tokens-aaa2fa/hf/step-1404/",
    #     "gs://marin-us-central1/checkpoints/exp2262pt2_sft_qwen2pt5_ot4_30k_math_qwq_32b_32768tokens-aaa2fa/hf/step-1638/",
    #     "gs://marin-us-central1/checkpoints/exp2262pt2_sft_qwen2pt5_ot4_30k_math_qwq_32b_32768tokens-aaa2fa/hf/step-1872/",
    # ],
    "exp2262pt2a-qwen2.5-7b-instruct-finetuned-ot4-30k-math-qwen3-32b-32768tokens": [
        "gs://marin-us-central1/checkpoints/exp2262pt2a_sft_qwen2pt5_ot4_30k_math_qwen3_32b_32768tokens-56d459/hf/step-234/",
        "gs://marin-us-central1/checkpoints/exp2262pt2a_sft_qwen2pt5_ot4_30k_math_qwen3_32b_32768tokens-56d459/hf/step-468/",
        "gs://marin-us-central1/checkpoints/exp2262pt2a_sft_qwen2pt5_ot4_30k_math_qwen3_32b_32768tokens-56d459/hf/step-702/",
        "gs://marin-us-central1/checkpoints/exp2262pt2a_sft_qwen2pt5_ot4_30k_math_qwen3_32b_32768tokens-56d459/hf/step-936/",
        "gs://marin-us-central1/checkpoints/exp2262pt2a_sft_qwen2pt5_ot4_30k_math_qwen3_32b_32768tokens-56d459/hf/step-1170/",
        "gs://marin-us-central1/checkpoints/exp2262pt2a_sft_qwen2pt5_ot4_30k_math_qwen3_32b_32768tokens-56d459/hf/step-1404/",
        "gs://marin-us-central1/checkpoints/exp2262pt2a_sft_qwen2pt5_ot4_30k_math_qwen3_32b_32768tokens-56d459/hf/step-1638/",
        "gs://marin-us-central1/checkpoints/exp2262pt2a_sft_qwen2pt5_ot4_30k_math_qwen3_32b_32768tokens-56d459/hf/step-1872/",
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
    ([AIME24, AIME25, AMC23, HMMT], SEEDS[:5]),                                        # 5 seeds
    ([MATH500], SEEDS[:1]),                                                              # 1 seed
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
# max_num_seqs: Batch size for parallel generation (default is very low!)
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
    eval_steps = []
    compile_steps = []

    # Create one evaluation step per (checkpoint, task, seed) combination.
    # Each combination runs as an independent parallel step.
    for base_eval_run_name, checkpoints in CHECKPOINTS.items():
        for checkpoint in checkpoints:
            task_seed_pairs = []
            for tasks, seeds in TASK_SEED_GROUPS:
                task_seed_pairs += [(t, seeds) for t in tasks]
            for task, seeds in task_seed_pairs:
                task_steps = []
                for seed in seeds:
                    generation_params = {**BASE_GENERATION_PARAMS, "seed": seed}
                    step = default_evalchemy_eval(
                        step=checkpoint,
                        resource_config=ResourceConfig.with_tpu("v5p-8"),
                        evals=[task],
                        engine_kwargs=ENGINE_KWARGS,
                        generation_params=generation_params,
                        apply_chat_template=True,
                        discover_latest_checkpoint=DISCOVER_LATEST_CHECKPOINT,
                        base_eval_run_name=base_eval_run_name,
                    )
                    task_steps.append(step)
                    eval_steps.append(step)

                # Add compile step to aggregate results across seeds for this checkpoint+task
                if len(seeds) > 1:
                    compile_step = compile_evalchemy_results(
                        task_steps, seeds=seeds,
                        base_eval_run_name=base_eval_run_name, model_path=checkpoint,
                        task_name=task.name,
                    )
                    compile_steps.append(compile_step)

    # Run eval steps in batches to limit parallelism.
    # Each executor_main call runs up to MAX_PARALLEL_JOBS eval steps concurrently.
    # Already-completed steps are automatically skipped via status files on disk.
    if MAX_PARALLEL_JOBS is not None:
        for i in range(0, len(eval_steps), MAX_PARALLEL_JOBS):
            batch = eval_steps[i : i + MAX_PARALLEL_JOBS]
            executor_main(steps=batch)
    else:
        executor_main(steps=eval_steps)

    # Run compile steps separately. Their eval-step dependencies have already
    # succeeded, so the executor skips them and only runs the compile steps.
    if compile_steps:
        executor_main(steps=compile_steps)
