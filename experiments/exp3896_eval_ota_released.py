# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate OT-Agent's released SFT weights on TB-Lite and TB2.

Evaluates:
- laion/exp_tas_optimal_combined_traces (32K model) - target: 23.8% TB-Lite, 12.6% TB2
- laion/GLM-4_7-r2egym_sandboxes-maxeps-131k-lc (131K model) - to be confirmed

Usage (32K model):
    uv run lib/marin/src/marin/run/ray_run.py \
        --env_vars WANDB_ENTITY marin-community \
        --env_vars WANDB_PROJECT marin \
        --extra harbor,vllm \
        --cluster us-central1 \
        --no_wait \
        -- python experiments/exp3896_eval_ota_released.py

Set MODEL=131k to evaluate the 131K model instead:
    uv run lib/marin/src/marin/run/ray_run.py \
        --env_vars WANDB_ENTITY marin-community \
        --env_vars WANDB_PROJECT marin \
        --env_vars MODEL 131k \
        --extra harbor,vllm \
        --cluster us-central1 \
        --no_wait \
        -- python experiments/exp3896_eval_ota_released.py
"""

import os

from experiments.evals.evals import evaluate_harbor
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

MODEL_VARIANT = os.environ.get("MODEL", "32k")

if MODEL_VARIANT == "131k":
    MODEL_NAME = "ota-131k-released"
    MODEL_PATH = "laion/GLM-4_7-r2egym_sandboxes-maxeps-131k-lc"
    ISSUE = 3897
else:
    MODEL_NAME = "ota-32k-released"
    MODEL_PATH = "laion/exp_tas_optimal_combined_traces"
    ISSUE = 3896

# Harbor eval on TB2 (terminal-bench@2.0, 89 tasks)
tb2_eval = evaluate_harbor(
    model_name=MODEL_NAME,
    model_path=MODEL_PATH,
    dataset="terminal-bench",
    version="2.0",
    agent="terminus-2",
    n_concurrent=25,
    env="daytona",
    resource_config=ResourceConfig.with_tpu("v5p-8"),
)

# Harbor eval on TB-Lite (openthoughts-tblite@2.0)
tb_lite_eval = evaluate_harbor(
    model_name=MODEL_NAME,
    model_path=MODEL_PATH,
    dataset="openthoughts-tblite",
    version="2.0",
    agent="terminus-2",
    n_concurrent=25,
    env="daytona",
    resource_config=ResourceConfig.with_tpu("v5p-8"),
)

if __name__ == "__main__":
    print(f"=== Evaluating OT-Agent {MODEL_VARIANT} released weights ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Tracked in: https://github.com/marin-community/marin/issues/{ISSUE}")
    executor_main(steps=[tb2_eval, tb_lite_eval])
