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
MODEL = "Qwen/Qwen3-8B"

# =============================================================================
# Evaluation Configuration
# =============================================================================

# Seeds for multiple evaluation runs to compute averaged results
# Set to [42] for single run, or expand for multiple iterations
SEEDS = [42]  # Single seed for quick testing
# SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # 10 seeds for full evaluation

# -----------------------------------------------------------------------------
# Task Selection - Uncomment the tasks you want to evaluate
# -----------------------------------------------------------------------------

# === Math Benchmarks ===
EVAL_TASKS = [AIME25]
# EVAL_TASKS = [AIME24]  # AIME 2024 (30 problems)
# EVAL_TASKS = [AIME25]  # AIME 2025 (30 problems)
# EVAL_TASKS = [AMC23]  # AMC 2023
# EVAL_TASKS = [MATH500]  # MATH dataset (500 problems)
# EVAL_TASKS = [HMMT]  # Harvard-MIT Math Tournament

# === Code Benchmarks ===
# EVAL_TASKS = [HUMANEVAL_PLUS]  # HumanEval+ (improved HumanEval)
# EVAL_TASKS = [MBPP_PLUS]  # MBPP+ (improved MBPP)
# EVAL_TASKS = [LIVECODEBENCH]  # LiveCodeBench (competitive programming)
# EVAL_TASKS = [BIGCODEBENCH]  # BigCodeBench
# EVAL_TASKS = [CODEFORCES]  # Codeforces problems
# EVAL_TASKS = [CODEELO]  # CodeElo benchmark

# === Science Benchmarks ===
# EVAL_TASKS = [GPQA_DIAMOND]  # GPQA Diamond (graduate-level science)
# EVAL_TASKS = [JEEBENCH]  # IIT JEE entrance exam problems

# === Reasoning Benchmarks ===
# EVAL_TASKS = [ALICE_IN_WONDERLAND]  # Alice in Wonderland reasoning
# EVAL_TASKS = [HUMANITYS_LAST_EXAM]  # Humanity's Last Exam

# === Task Groups (multiple benchmarks) ===
# EVAL_TASKS = list(EVALCHEMY_MATH_TASKS)  # All math tasks
# EVAL_TASKS = list(EVALCHEMY_CODE_TASKS)  # All code tasks
# EVAL_TASKS = list(EVALCHEMY_SCIENCE_TASKS)  # All science tasks
# EVAL_TASKS = list(EVALCHEMY_REASONING_TASKS)  # All reasoning tasks
# EVAL_TASKS = list(EVALCHEMY_CORE_TASKS)  # All tasks combined

# === Custom combinations ===
# EVAL_TASKS = [AIME24, AIME25, MATH500]  # Math-focused evaluation
# EVAL_TASKS = [HUMANEVAL_PLUS, LIVECODEBENCH]  # Code-focused evaluation
# EVAL_TASKS = [GPQA_DIAMOND, JEEBENCH]  # Science-focused evaluation

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
ENGINE_KWARGS = {
    "tensor_parallel_size": 4,
    "max_num_seqs": 30,  # Enable batched generation for better throughput
}

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    all_steps = []

    # Create one evaluation step per seed
    for seed in SEEDS:
        generation_params = {**BASE_GENERATION_PARAMS, "seed": seed}

        step = default_evalchemy_eval(
            step=MODEL,
            resource_config=ResourceConfig.with_tpu("v5p-8"),
            evals=EVAL_TASKS,
            engine_kwargs=ENGINE_KWARGS,
            generation_params=generation_params,
            apply_chat_template=True,
            discover_latest_checkpoint=False,
        )
        all_steps.append(step)

    # Add compile step to aggregate results across seeds and log to wandb
    # Only add compile step if we have multiple seeds (aggregation needs >1 run)
    if len(SEEDS) > 1:
        compile_step = compile_evalchemy_results(all_steps, seeds=SEEDS)
        all_steps.append(compile_step)

    executor_main(steps=all_steps)
