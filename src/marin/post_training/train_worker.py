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
Training worker for RL/post-training tasks.

This worker reads rollout information from a queue which is populated by the
rollout workers, and periodically dumps new checkpoints to disk. These
checkpoints are read by the rollout workers to update their models.
"""

import logging
from dataclasses import dataclass

import haliax as hax
import jax
import jax.random as jrandom
import levanter
from levanter.models.lm_model import LmConfig
from levanter.optim import OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from transformers import AutoTokenizer

from marin.post_training import weight_transfer
from marin.post_training.weight_transfer import WeightTransferConfig

from .model_utils import load_model_from_checkpoint
from .replay_buffer import ReplayBuffer, ReplayDataLoader
from .rl_losses import rloo_loss_with_importance_sampling
from .rollout_storage import RolloutStorageConfig
from .train_batch import create_training_batch_from_rollouts

logger = logging.getLogger(__name__)


class StreamingRolloutLoader:
    """Direct loader for streaming rollout data.

    Rollouts are a continous stream of data, not really well modeled by the
    default Levanter indexing API. Instead of implemented a Dataset, we
    implement the expected data loader interface directly.
    """

    def __init__(
        self,
        data_loader: ReplayDataLoader,
        config: TrainerConfig,
        max_input_length: int,
        max_output_length: int,
        pad_token_id: int,
    ):
        """Initialize the streaming rollout loader.

        Args:
            data_loader: The replay data loader to get rollouts from
            config: Trainer config with mesh and axis mapping information
            max_input_length: Maximum input sequence length for padding
            max_output_length: Maximum output sequence length for padding
            pad_token_id: Token ID to use for padding
        """
        self.data_loader = data_loader
        self.config = config
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.pad_token_id = pad_token_id
        self.timeout = 60.0

    def __iter__(self):
        """Yield batches continuously from the replay buffer."""
        while True:
            rollouts = self.data_loader.get_rollouts(timeout=self.timeout)
            if not rollouts:
                logger.warning("No rollouts received from data loader within timeout, retrying...")
                continue

            # Convert rollouts to training batch
            batch = create_training_batch_from_rollouts(
                rollouts, self.max_input_length, self.max_output_length, self.pad_token_id
            )

            # shard onto the device mesh
            with self.config.device_mesh:
                sharded_batch = hax.shard(batch, self.config.compute_axis_mapping)

            yield sharded_batch


class StopTrainerException(Exception):
    """Exception to signal stopping the trainer."""

    pass


@dataclass
class ReplayBufferConfig:
    """Configuration for the replay buffer."""

    capacity: int = 10000
    """Maximum number of examples per environment in the buffer."""

    alpha: float = 3.0
    """Recency bias for sampling, higher values favor newer examples."""

    max_samples: int = 4
    """Maximum number of times to use an example before retiring."""


@dataclass
class TrainWorkerConfig:
    """Configuration for Levanter-based RL training worker."""

    rollout_storage: RolloutStorageConfig
    model: LmConfig
    trainer: TrainerConfig
    optimizer: OptimizerConfig
    replay_buffer: ReplayBufferConfig

    weight_transfer: WeightTransferConfig

    # Unique run ID for checkpointing and logging
    # (Not sure why this isn't part of TrainerConfig)
    run_id: str

    # Initial checkpoint for the reference model (auto-detects HF repo vs local path)
    initial_checkpoint: str | None = None

    # Optimization parameters
    kl_coef: float = 0.1


class TrainWorker:
    """Training worker that reads rollout data from a queue and trains the model using Levanter."""

    def __init__(
        self,
        config: TrainWorkerConfig,
    ):
        """Initialize training worker.

        Args:
            config: Training worker configuration with Levanter components.
        """

        print("Run id: ", config.run_id)

        config.trainer.id = f"{config.run_id}-train"
        levanter.initialize(config.trainer)
        self.config = config
        self._should_stop = False
        self.weight_id = 0
        if isinstance(config.model.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer)
        else:
            self.tokenizer = config.model.tokenizer

        self.rollout_reader = config.rollout_storage.create_reader()

        self.replay_buffer = ReplayBuffer(
            process_id=jax.process_index(),
            total_processes=jax.process_count(),
            capacity=config.replay_buffer.capacity,
            local_batch_size=config.trainer.train_batch_size,
            recency_alpha=config.replay_buffer.alpha,
            max_samples=config.replay_buffer.max_samples,
            max_input_length=getattr(config.model, "seq_len", 512),
            max_output_length=getattr(config.model, "seq_len", 512),
            pad_token_id=getattr(self.tokenizer, "pad_token_id", 0) or 0,
        )
        self.data_loader = ReplayDataLoader(
            rollout_reader=self.rollout_reader,
            replay_buffer=self.replay_buffer,
            rollout_fetch_interval=1.0,
        )

        self.transfer_server = weight_transfer.create_weight_transfer_server(
            config.weight_transfer,
            mesh=self.config.trainer.device_mesh,
            axis_mapping=self.config.trainer.compute_axis_mapping,
        )

        self._build_models()

    def _build_models(self):
        """Build reference and initial policy models."""
        config = self.config
        seed = config.trainer.seed
        model_key = jrandom.PRNGKey(seed)
        Vocab = hax.Axis("vocab", self.tokenizer.vocab_size)

        if config.initial_checkpoint is not None:
            logger.info(f"Loading initial model from checkpoint: {config.initial_checkpoint}")
        else:
            logger.info("Building new model from scratch")

        def _load_model():
            return load_model_from_checkpoint(
                checkpoint=config.initial_checkpoint,
                model_config=config.model,
                trainer_config=config.trainer,
                vocab_axis=Vocab,
                tokenizer=self.tokenizer,
                mesh=config.trainer.device_mesh,
                axis_mapping=self.config.trainer.parameter_axis_mapping,
                key=model_key,
            )

        self.reference_model = _load_model()

    def train(self):
        """Main training method using Levanter's standard train_lm infrastructure."""
        logger.info("Starting RLOO training with Levanter...")

        config = self.config
        optimizer = config.optimizer.build(config.trainer.num_train_steps)

        @jax.jit
        def _loss_function(model, batch, key):
            return rloo_loss_with_importance_sampling(
                model, self.reference_model, batch, key=key, kl_coef=config.kl_coef, clip_epsilon=5
            )
            # return ppo_loss(model, batch, key=key, kl_coef=config.kl_coef, clip_epsilon=0.5)

        with (
            config.trainer.device_mesh,
            hax.axis_mapping(config.trainer.compute_axis_mapping),
            Trainer(config.trainer, optimizer, _loss_function) as trainer,
            self.data_loader,
        ):
            seed = config.trainer.seed
            _, training_key = jrandom.split(jrandom.PRNGKey(seed), 2)

            state = trainer.initial_state(training_key, model=self.reference_model)

            self._configure_training_hooks(trainer)
            train_loader = StreamingRolloutLoader(
                self.data_loader,
                config.trainer,
                self.replay_buffer.max_input_length,
                self.replay_buffer.max_output_length,
                self.replay_buffer.pad_token_id,
            )

            try:
                trainer.train(state, train_loader)
            except StopTrainerException:
                pass

    def _configure_training_hooks(self, trainer):
        trainer.add_hook(
            self.create_weight_transfer_hook(),
            every=self.config.weight_transfer.sync_interval_steps,
        )

        def _stop_on_signal(info: levanter.callbacks.StepInfo):
            if self._should_stop:
                raise StopTrainerException()

        trainer.add_hook(_stop_on_signal, every=1)

    def create_weight_transfer_hook(self):
        def weight_transfer_hook(info: levanter.callbacks.StepInfo):
            step = info.step
            state = info.state

            if step % self.config.weight_transfer.sync_interval_steps != 0:
                return

            self.weight_id += 1
            logger.info(
                "Transferring weights at step %d, weight_id %d, loss=%s",
                step,
                self.weight_id,
                info.loss,
            )

            model_params = state.model
            self.transfer_server.serve_weights(self.weight_id, model_params)
            logger.info(f"Successfully transferred weights with ID {self.weight_id}")

        return weight_transfer_hook

    def stop(self):
        """Stop the training worker."""
        self._should_stop = True
