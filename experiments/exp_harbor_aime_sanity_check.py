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
Harbor + AIME sanity check with Qwen3-8B served via vLLM

Evaluates Qwen3-8B on AIME@1.0 from Harbor registry (60 competition math problems)
using a vLLM server started automatically by the evaluator.

Usage (Ray Cluster with Daytona):
    uv run lib/marin/src/marin/run/ray_run.py \
        --env_vars MARIN_PREFIX gs://marin-us-central1 \
        --env_vars DAYTONA_API_KEY ${DAYTONA_API_KEY} \
        --env_vars WANDB_API_KEY ${WANDB_API_KEY} \
        --env_vars WANDB_ENTITY marin-community \
        --env_vars WANDB_PROJECT harbor \
        --env_vars HF_TOKEN ${HF_TOKEN} \
        --cluster us-central1 \
        --extra harbor \
        --no_wait \
        -- python experiments/exp_harbor_aime_sanity_check.py

Results are saved to:
    - GCS: gs://marin-us-central1/evaluation/harbor/{dataset}/{model_name_escaped}/
      - samples_TIMESTAMP.jsonl: Per-task samples (1 line per task)
      - results_TIMESTAMP.json: Aggregated metrics and pointer to samples file
      - trajectories/{task_id}.jsonl: Full agent interaction traces (best-effort)
    - W&B: Metrics and trajectory lengths logged to wandb.ai/marin-community/harbor
"""

import logging
import os

from experiments.evals.evals import evaluate_harbor
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def _escape_model_name_for_path(model_name: str) -> str:
    return model_name.replace("://", "__").replace("/", "__")


# Qwen3-8B served via vLLM (the evaluator starts the vLLM server automatically)
MODEL = {
    "name": "qwen3-8b",
    "path": "gs://marin-us-central1/models/Qwen--Qwen3-8B--main/",
}

# TPU resource config for vLLM serving
resource_config = ResourceConfig.with_tpu("v5p-8")

# Get environment type from ENV_TYPE environment variable (default: daytona)
ENV_TYPE = os.environ.get("ENV_TYPE", "daytona")

# Harbor agent configuration
HARBOR_AGENT = os.environ.get("HARBOR_AGENT", "terminus-2")


def _optional_int_from_env(var_name: str, *, default: int | None) -> int | None:
    value = os.environ.get(var_name)
    if value is None:
        return default

    if value.strip().lower() in {"none", "null", "all"}:
        return None

    return int(value)


HARBOR_MAX_INSTANCES = _optional_int_from_env("HARBOR_MAX_INSTANCES", default=5)
HARBOR_N_CONCURRENT = _optional_int_from_env("HARBOR_N_CONCURRENT", default=5)

# Warn if using Daytona without API key
if ENV_TYPE == "daytona" and not os.environ.get("DAYTONA_API_KEY"):
    logger.warning(
        "DAYTONA_API_KEY not set but ENV_TYPE=daytona! "
        "The evaluation will likely fail. Set DAYTONA_API_KEY or use ENV_TYPE=local for Docker."
    )

# AIME configuration from Harbor registry
DATASET = "aime"
VERSION = "1.0"

MODEL_NAME_ESCAPED = _escape_model_name_for_path(MODEL["name"])
OUTPUT_DIR = os.path.join("evaluation", "harbor", DATASET, MODEL_NAME_ESCAPED)

logger.info("=" * 80)
logger.info("Harbor + AIME Sanity Check (Qwen3-8B via vLLM)")
logger.info("=" * 80)
logger.info(f"Dataset: {DATASET}@{VERSION}")
logger.info(f"Model: {MODEL['name']}")
logger.info(f"Max instances: {HARBOR_MAX_INSTANCES}")
logger.info(f"Agent: {HARBOR_AGENT}")
logger.info(f"Concurrent trials: {HARBOR_N_CONCURRENT}")
logger.info(f"Harbor env: {ENV_TYPE} - {'Docker containers' if ENV_TYPE == 'local' else 'Daytona cloud workspaces'}")
logger.info(f"Output dir (under MARIN_PREFIX): {OUTPUT_DIR}")
logger.info("=" * 80)

# Create evaluation step
step = evaluate_harbor(
    model_name=MODEL["name"],
    model_path=MODEL["path"],
    dataset=DATASET,
    version=VERSION,
    max_eval_instances=HARBOR_MAX_INSTANCES,
    resource_config=resource_config,
    wandb_tags=["harbor", DATASET, "sanity-check", HARBOR_AGENT, ENV_TYPE, "qwen3-8b", "vllm"],
    agent=HARBOR_AGENT,
    n_concurrent=HARBOR_N_CONCURRENT or 4,
    env=ENV_TYPE,
)
step = step.with_output_path(OUTPUT_DIR)

if __name__ == "__main__":
    logger.info("Starting Harbor evaluation...")
    env_desc = "local Docker containers" if ENV_TYPE == "local" else f"{ENV_TYPE} cloud workspaces"
    task_count_desc = "all tasks" if HARBOR_MAX_INSTANCES is None else f"the first {HARBOR_MAX_INSTANCES} task(s)"
    logger.info(
        f"This will:\n"
        f"1. Start vLLM server serving {MODEL['name']}\n"
        f"2. Load AIME@1.0 dataset from Harbor registry (60 tasks total)\n"
        f"3. Run {task_count_desc} using agent '{HARBOR_AGENT}'\n"
        f"4. Execute in {env_desc}\n"
        f"5. Save trajectories and results to GCS\n"
        f"6. Log metrics to W&B"
    )

    executor_main(steps=[step])

    logger.info("=" * 80)
    logger.info("Harbor sanity check complete!")
    logger.info("=" * 80)
