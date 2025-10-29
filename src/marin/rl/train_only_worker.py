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
Training-only worker for benchmarking training throughput in isolation.

This worker generates simple random tokens for training and measures
training throughput without the overhead of rollouts, replay buffer,
weight transfer, or curriculum.
"""

import logging
from dataclasses import dataclass

import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import levanter
from levanter.models.lm_model import LmConfig
from levanter.optim import OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from transformers import PreTrainedTokenizer

from marin.rl.model_utils import load_model_from_checkpoint
from marin.rl.rl_losses import RLLossModule
from marin.rl.types import TrainingBatch

logger = logging.getLogger(__name__)


@dataclass
class TrainOnlyWorkerConfig:
    """Configuration for training-only worker (no RL components)."""

    model: LmConfig
    trainer: TrainerConfig
    optimizer: OptimizerConfig
    loss: RLLossModule
    tokenizer: PreTrainedTokenizer
    run_id: str

    initial_checkpoint: str | None = None
    """Initial checkpoint for the model (auto-detects HF repo vs local path)."""

    seed: int = 0
    """Random seed for data generation and model construction."""

    sequence_length: int = 1024
    """Fixed sequence length for random token generation."""


class RandomTokenDataLoader:
    """Simple data loader that generates random tokens for training."""

    def __init__(
        self,
        config: TrainOnlyWorkerConfig,
    ):
        """Initialize the random token data loader.

        Args:
            config: Train-only worker config with tokenizer and sequence length
        """
        self.config = config
        self.vocab_size = config.tokenizer.vocab_size
        self.sequence_length = config.sequence_length
        self.batch_size = config.trainer.train_batch_size
        self.seed = config.seed

        self.pad_token_id = config.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = config.tokenizer.eos_token_id

        # Create key for random generation
        self.key = jrandom.PRNGKey(self.seed)

    def __iter__(self):
        """Yield batches of random tokens continuously."""
        while True:
            self.key, batch_key = jrandom.split(self.key)

            # Generate random token IDs
            token_ids = jrandom.randint(
                batch_key,
                shape=(self.batch_size, self.sequence_length),
                minval=0,
                maxval=self.vocab_size,
            )

            # Create position IDs
            position_ids = jnp.arange(self.sequence_length, dtype=jnp.int32)[None, :]
            position_ids = jnp.broadcast_to(position_ids, (self.batch_size, self.sequence_length))

            # Create loss masks (all ones for random tokens - we treat all tokens as response)
            loss_masks = jnp.ones((self.batch_size, self.sequence_length), dtype=jnp.float32)

            # Create loss weights (random advantages for benchmark)
            self.key, reward_key = jrandom.split(self.key)
            advantages = jrandom.uniform(reward_key, shape=(self.batch_size, 1))
            loss_weights = jnp.broadcast_to(advantages, (self.batch_size, self.sequence_length))

            # Create policy logprobs (zeros for initial policy)
            policy_logprobs = jnp.zeros((self.batch_size, self.sequence_length), dtype=jnp.float32)

            # Create TrainingBatch with named arrays
            batch = TrainingBatch(
                input_ids=hax.named(token_ids, ["batch", "position"]),
                position_ids=hax.named(position_ids, ["batch", "position"]),
                loss_weights=hax.named(loss_weights, ["batch", "position"]),
                loss_masks=hax.named(loss_masks, ["batch", "position"]),
                policy_logprobs=hax.named(policy_logprobs, ["batch", "position"]),
            )

            # Shard onto the device mesh
            with hax.set_mesh(self.config.trainer.device_mesh):
                sharded_batch = hax.shard(batch, self.config.trainer.compute_axis_mapping)

            yield sharded_batch


class TrainOnlyWorker:
    """Training-only worker for benchmarking training throughput."""

    config: TrainOnlyWorkerConfig

    def __init__(
        self,
        config: TrainOnlyWorkerConfig,
    ):
        """Initialize training-only worker.

        Args:
            config: Training-only worker configuration.
        """

        print("Run id: ", config.run_id)

        config.trainer.id = f"{config.run_id}-train"
        levanter.initialize(config.trainer)
        self.config = config
        self.tokenizer = config.tokenizer
        self.loss_module = config.loss

        # Timing metrics for benchmarking
        self._total_tokens_trained = 0

        self.data_loader = RandomTokenDataLoader(config)

        self._build_models()

    def _build_models(self):
        """Build initial model."""
        config = self.config
        model_key = jrandom.PRNGKey(config.seed)
        Vocab = hax.Axis("vocab", self.tokenizer.vocab_size)

        if config.initial_checkpoint is not None:
            logger.info(f"Loading initial model from checkpoint: {config.initial_checkpoint}")
        else:
            logger.info("Building new model from scratch")

        self.model = load_model_from_checkpoint(
            checkpoint=config.initial_checkpoint,
            model_config=config.model,
            trainer_config=config.trainer,
            vocab_axis=Vocab,
            tokenizer=self.tokenizer,
            mesh=config.trainer.device_mesh,
            axis_mapping=self.config.trainer.parameter_axis_mapping,
            key=model_key,
        )

    def train(self):
        """Main training method using Levanter's standard infrastructure."""
        logger.info("Starting training-only benchmark...")

        config = self.config
        optimizer = config.optimizer.build(config.trainer.num_train_steps)

        # Create a simplified loss function (no reference model needed for benchmarking)
        loss_fn = self.loss_module.create_loss_fn(self.model, None)

        @jax.jit
        def _loss_function(model, batch, key):
            return loss_fn(model, batch, key)

        with (
            config.trainer.device_mesh,
            hax.axis_mapping(config.trainer.compute_axis_mapping),
            Trainer(config=config.trainer, optimizer=optimizer, loss_fn=_loss_function) as trainer,
        ):
            seed = config.trainer.seed
            _, training_key = jrandom.split(jrandom.PRNGKey(seed), 2)

            state = trainer.initial_state(training_key, model=self.model)

            self._configure_training_hooks(trainer)

            trainer.train(state, self.data_loader)

    def _configure_training_hooks(self, trainer):
        """Configure hooks for logging training metrics."""

        def _log_throughput(info: levanter.callbacks.StepInfo):
            step_duration = info.step_duration
            metrics = {
                "train.step_duration_sec": step_duration,
            }

            # Calculate tokens processed in this step
            ntokens = self.config.sequence_length * self.config.trainer.train_batch_size
            self._total_tokens_trained += ntokens
            
            if step_duration > 0:
                tokens_per_sec = ntokens / step_duration
                metrics["train.tokens_per_second"] = tokens_per_sec
            
            metrics["train.total_tokens_trained"] = self._total_tokens_trained

            trainer.tracker.log(metrics, step=info.step)
            logger.info(f"Step {info.step}: {metrics.get('train.tokens_per_second', 0):.2f} tokens/sec")

        trainer.add_hook(_log_throughput, every=10)
