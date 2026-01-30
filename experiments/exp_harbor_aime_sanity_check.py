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

Usage:
    python experiments/exp_harbor_aime_sanity_check.py

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

# Resource configuration
resource_config = ResourceConfig.with_tpu("v5p-8")

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

logger.info("=" * 80)
logger.info("Harbor + AIME Sanity Check")
logger.info("=" * 80)
logger.info(f"Dataset: {DATASET}@{VERSION}")
logger.info(f"Model: {MODEL['name']}")
logger.info(f"Max instances: {MAX_INSTANCES}")
logger.info("Agent: claude-code (Harbor's built-in Claude agent)")
logger.info("Environment: local (Docker)")
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
    n_concurrent=2,  # Run 2 tasks in parallel for sanity check
    env="local",  # Use local Docker containers
)

if __name__ == "__main__":
    logger.info("Starting Harbor evaluation...")
    logger.info(
        "This will:\n"
        "1. Load AIME@1.0 dataset from Harbor registry (60 tasks total)\n"
        "2. Run first 5 tasks using Claude Code agent\n"
        "3. Execute in local Docker containers\n"
        "4. Save results to GCS and log to W&B"
    )

    executor_main(steps=[step])

    logger.info("=" * 80)
    logger.info("Sanity check complete!")
    logger.info("Next steps:")
    logger.info("1. Check W&B for results")
    logger.info("2. If successful, try full 60 tasks: set MAX_INSTANCES=None")
    logger.info("3. Try other datasets: terminal-bench@2.0, swebench-verified@1.0")
    logger.info("=" * 80)
