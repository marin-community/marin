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
Harbor + AIME sanity check

This demonstrates the generic Harbor integration working with AIME@1.0.
AIME@1.0 from Harbor registry contains 60 competition math problems:
- 30 from AIME 2024 (aime_60 through aime_89)
- 15 from AIME 2025-I (aime_i-1 through aime_i-15)
- 15 from AIME 2025-II (aime_ii-1 through aime_ii-15)

No custom adapters needed! Harbor loads AIME directly from its registry.

This same pattern works for ANY Harbor dataset:
- terminal-bench@2.0 (89 tasks)
- swebench-verified@1.0 (500 tasks)
- And 40+ other benchmarks!

Usage (Local):
    MARIN_PREFIX=gs://marin-us-central1 ANTHROPIC_API_KEY=<key> python experiments/exp_harbor_aime_sanity_check.py

Usage (Ray Cluster):
    uv run lib/marin/src/marin/run/ray_run.py \
        --env_vars MARIN_PREFIX gs://marin-us-central1 \
        --env_vars ANTHROPIC_API_KEY ${ANTHROPIC_API_KEY} \
        --env_vars WANDB_API_KEY ${WANDB_API_KEY} \
        --env_vars WANDB_ENTITY marin-community \
        --env_vars WANDB_PROJECT harbor \
        --cluster us-central1 \
        --extra harbor,cpu \
        --no_wait \
        -- python experiments/exp_harbor_aime_sanity_check.py

Results are saved to:
    - GCS: gs://marin-us-central1/evaluation/harbor/{dataset}/{model_name_escaped}/
      - samples_TIMESTAMP.jsonl: Per-task samples (1 line per task)
      - results_TIMESTAMP.json: Aggregated metrics and pointer to samples file
      - trajectories/{task_id}.jsonl: Full agent interaction traces (best-effort)
    - W&B: Metrics and trajectory lengths logged to wandb.ai/marin-community/harbor

Note: Requires ANTHROPIC_API_KEY environment variable for Claude agent.
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


# Resource configuration (Harbor runs via API + containers; no accelerator required)
resource_config = ResourceConfig.with_cpu()

# Model configuration
# For the sanity check, we'll use Claude Haiku 4.5 for faster, cost-effective evaluation
MODEL = {
    "name": "anthropic/claude-haiku-4-5-20251001",
    "path": None,  # API model, no path needed
}

# Check for API key
if not os.environ.get("ANTHROPIC_API_KEY"):
    logger.warning(
        "ANTHROPIC_API_KEY not set! The evaluation will fail without it. " "Set it in your environment or .env file."
    )

# AIME configuration from Harbor registry
# See https://harborframework.com/registry/aime/1.0
DATASET = "aime"
VERSION = "1.0"
MAX_INSTANCES = 5  # Start with just 5 tasks for sanity check

MODEL_NAME_ESCAPED = _escape_model_name_for_path(MODEL["name"])
OUTPUT_DIR = os.path.join("evaluation", "harbor", DATASET, MODEL_NAME_ESCAPED)

logger.info("=" * 80)
logger.info("Harbor + AIME Sanity Check")
logger.info("=" * 80)
logger.info(f"Dataset: {DATASET}@{VERSION}")
logger.info(f"Model: {MODEL['name']}")
logger.info(f"Max instances: {MAX_INSTANCES}")
logger.info("Agent: claude-code (Harbor's built-in Claude agent)")
logger.info("Harbor env: local (Docker) - run via Ray for cluster execution")
logger.info(f"Output dir (under MARIN_PREFIX): {OUTPUT_DIR}")
logger.info("=" * 80)

# Create evaluation step
step = evaluate_harbor(
    model_name=MODEL["name"],
    model_path=MODEL["path"],
    dataset=DATASET,
    version=VERSION,
    max_eval_instances=MAX_INSTANCES,
    resource_config=resource_config,
    wandb_tags=["harbor", "aime", "sanity-check", "claude"],
    agent="claude-code",  # Use Harbor's built-in Claude Code agent
    n_concurrent=5,  # Run all 5 tasks in parallel
    env="local",  # Use local Docker containers
)
step = step.with_output_path(OUTPUT_DIR)

if __name__ == "__main__":
    logger.info("Starting Harbor evaluation...")
    logger.info(
        "This will:\n"
        "1. Load AIME@1.0 dataset from Harbor registry (60 tasks total)\n"
        "2. Run first 5 tasks using Claude Code agent\n"
        "3. Execute in local Docker containers (or Ray cluster if using ray_run.py)\n"
        "4. Save trajectories and results to GCS\n"
        "5. Log metrics to W&B"
    )

    executor_main(steps=[step])

    logger.info("=" * 80)
    logger.info("Sanity check complete!")
    logger.info("Next steps:")
    logger.info("1. Check W&B for results: https://wandb.ai/marin-community/harbor")
    logger.info(
        "2. Check GCS for results: gs://marin-us-central1/"
        f"{OUTPUT_DIR}/samples_TIMESTAMP.jsonl and gs://marin-us-central1/{OUTPUT_DIR}/results_TIMESTAMP.json"
    )
    logger.info("3. For full evaluation, run on Ray cluster with MAX_INSTANCES=None (or set max_eval_instances=None)")
    logger.info("4. Try other datasets: terminal-bench@2.0, swebench-verified@1.0")
    logger.info("=" * 80)
