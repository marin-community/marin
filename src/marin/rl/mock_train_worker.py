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
Mock training worker for benchmarking inference throughput.

This worker allocates TPU resources, loads the initial model on device, and
serves it via weight transfer without performing actual training. Useful for
isolating and measuring inference performance.
"""

import logging
import time

import haliax as hax
import jax.random as jrandom
import levanter

from marin.rl import weight_transfer
from marin.rl.model_utils import load_model_from_checkpoint
from marin.rl.train_worker import TrainWorkerConfig

logger = logging.getLogger(__name__)


class MockTrainWorker:
    """Mock training worker that serves initial weights without training."""

    config: TrainWorkerConfig
    transfer_server: weight_transfer.WeightTransferServer

    def __init__(self, config: TrainWorkerConfig):
        """Initialize mock training worker.

        Args:
            config: Training worker configuration (same as real TrainWorker).
        """
        logger.info("Initializing MockTrainWorker (no training, only serving initial weights)")
        logger.info(f"Run id: {config.run_id}")

        config.trainer.id = f"{config.run_id}-mock-train"
        levanter.initialize(config.trainer)
        
        self.config = config
        self._should_stop = False

        # Use the trainer's device mesh and axis mapping like real TrainWorker
        self.transfer_server = weight_transfer.create_weight_transfer_server(
            config.weight_transfer,
            mesh=self.config.trainer.device_mesh,
            axis_mapping=self.config.trainer.compute_axis_mapping,
        )

        # Load initial model
        self._load_initial_model()

    def _load_initial_model(self):
        """Load the initial model to serve to rollout workers."""
        config = self.config
        model_key = jrandom.PRNGKey(config.seed)
        Vocab = hax.Axis("vocab", config.tokenizer.vocab_size)

        if config.initial_checkpoint is not None:
            logger.info(f"Loading initial model from checkpoint: {config.initial_checkpoint}")
        else:
            logger.info("Building new model from scratch")

        self.model = load_model_from_checkpoint(
            checkpoint=config.initial_checkpoint,
            model_config=config.model,
            trainer_config=config.trainer,
            vocab_axis=Vocab,
            tokenizer=config.tokenizer,
            mesh=config.trainer.device_mesh,
            axis_mapping=config.trainer.parameter_axis_mapping,
            key=model_key,
        )

        logger.info("Initial model loaded successfully")

    def train(self):
        """Main method that serves initial weights without training.

        This method serves the initial model weights once and then keeps
        running to allow rollout workers to continuously fetch weights.
        """
        logger.info("Starting MockTrainWorker - serving initial weights only")

        # Serve initial weights with step 0
        step = 0
        logger.info(f"Serving initial weights with step {step}")
        self.transfer_server.serve_weights(step, self.model)
        logger.info("Initial weights served successfully")

        # Keep running until stopped
        logger.info("MockTrainWorker running - weights will remain unchanged")
        try:
            while not self._should_stop:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("MockTrainWorker interrupted")

        self.cleanup()

    def stop(self):
        """Stop the mock training worker."""
        self._should_stop = True

    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up MockTrainWorker")
        self.transfer_server.cleanup()
