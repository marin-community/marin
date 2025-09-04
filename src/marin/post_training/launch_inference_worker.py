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
Launch script for inference worker.

This script initializes and runs an inference worker that generates rollouts
from a single environment and writes them to a specified output location.
"""

import logging

import tyro

from .inference_worker import InferenceWorker
from .training_config import InferenceWorkerConfig, TrainingConfig

logger = logging.getLogger(__name__)


def main(
    training_config: TrainingConfig,
    inference_config: InferenceWorkerConfig,
    log_level: str = "INFO",
):
    """Launch inference worker.

    Args:
        training_config: Training configuration with model/generation settings.
        inference_config: Inference worker specific configuration.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting inference worker...")
    logger.info(f"Environment: {inference_config.environment_spec}")
    logger.info(f"Checkpoint source: {inference_config.checkpoint_source_path}")
    logger.info(f"Output path: {inference_config.rollout_output_path}")

    try:
        # Create and run inference worker
        worker = InferenceWorker(training_config, inference_config)
        worker.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping inference worker")
    except Exception as e:
        logger.error(f"Inference worker failed: {e}")
        raise


if __name__ == "__main__":
    tyro.cli(main)