#!/usr/bin/env python3
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
Standalone rollout worker script for testing.

This script extracts the rollout worker configuration from an experiment
and runs it standalone on a single TPU worker for testing purposes.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import ray

# Add experiments to path so we can import them
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from exp1247_rl_async import rl_train
from marin.rl.rl_job import RLJob
from marin.rl.rollout_storage import RolloutStorageConfig, StorageType
from marin.rl.rollout_worker import RolloutWorker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run rollout worker standalone for testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max-rollouts",
        type=int,
        default=10,
        help="Maximum number of rollout batches to generate before stopping",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/rollout_test",
        help="Directory to store rollout outputs",
    )
    parser.add_argument(
        "--storage-type",
        choices=["file", "memory"],
        default="file",
        help="Type of rollout storage to use",
    )
    args = parser.parse_args()

    # Ray may already be running on TPU, try to connect or initialize
    # Use a unique temp directory to avoid permission conflicts with existing Ray instances
    if not ray.is_initialized():
        logger.info("Initializing Ray locally with isolated temp directory...")
        ray_temp_dir = f"/tmp/ray_rollout_worker/{os.getpid()}"
        try:
            ray.init(
                address="local",
                namespace="rollout_worker",
                _temp_dir=ray_temp_dir,
                ignore_reinit_error=True,
            )
            logger.info(f"Ray initialized with temp dir: {ray_temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to initialize Ray: {e}. Continuing anyway...")

    logger.info("Loading experiment configuration from exp1247_rl_async.py...")
    # Get the experiment step configuration
    step = rl_train(name="rollout-worker-test")
    job_config = step.config

    # Create RLJob and extract worker configs
    job = RLJob(job_config)
    _, rollout_config = job.to_worker_configs()

    logger.info("Configuring rollout worker for standalone execution...")
    # Modify for standalone testing
    rollout_config.max_rollouts = args.max_rollouts

    # Configure rollout storage
    storage_type = StorageType.FILE if args.storage_type == "file" else StorageType.IN_MEMORY
    rollout_config.rollout_storage = RolloutStorageConfig(
        storage_type=storage_type,
        path=f"{args.output_dir}/rollouts" if storage_type == StorageType.FILE else None,
        queue_name="standalone_test" if storage_type == StorageType.IN_MEMORY else None,
    )

    logger.info("Rollout configuration:")
    logger.info(f"  - Max rollouts: {rollout_config.max_rollouts}")
    logger.info(f"  - Storage type: {storage_type}")
    logger.info(f"  - Output dir: {args.output_dir}")
    logger.info(f"  - Model: {rollout_config.model}")
    logger.info(f"  - Curriculum lessons: {list(rollout_config.curriculum_config.lessons.keys())}")

    logger.info("Starting rollout worker...")
    # logger.info(f"Number of JAX devices: {len(jax.devices())}")
    # logger.info(f"JAX devices: {jax.devices()}")

    worker = RolloutWorker(config=rollout_config)
    worker.run()

    logger.info("Rollout worker completed successfully!")
    if storage_type == StorageType.FILE:
        logger.info(f"Rollouts saved to: {args.output_dir}/rollouts")


if __name__ == "__main__":
    main()
