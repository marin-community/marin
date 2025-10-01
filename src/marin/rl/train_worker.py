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

import dataclasses
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

from marin.rl import weight_transfer
from marin.rl.model_utils import load_model_from_checkpoint
from marin.rl.weight_transfer import WeightTransferConfig

from .replay_buffer import ReplayBuffer, ReplayBufferConfig, ReplayDataLoader
from .rl_losses import rloo_loss_with_importance_sampling
from .rollout_storage import RolloutStorageConfig
from .train_batch import create_training_batch_from_rollouts

logger = logging.getLogger(__name__)


@dataclass
class TrainWorkerConfig:
    """Configuration for Levanter-based RL training worker."""

    rollout_storage: RolloutStorageConfig
    model: LmConfig
    trainer: TrainerConfig
    optimizer: OptimizerConfig
    replay_buffer: ReplayBufferConfig

    weight_transfer: WeightTransferConfig

    max_input_length: int
    max_output_length: int
    pad_token_id: int

    # Unique run ID for checkpointing and logging
    # (Not sure why this isn't part of TrainerConfig)
    run_id: str

    # Initial checkpoint for the reference model (auto-detects HF repo vs local path)
    initial_checkpoint: str | None = None

    # Optimization parameters
    kl_coef: float = 0.1


class StreamingRolloutLoader:
    """Direct loader for streaming rollout data.

    Rollouts are a continous stream of data, not really well modeled by the
    default Levanter indexing API. Instead of implemented a Dataset, we
    implement the expected data loader interface directly.
    """

    config: TrainWorkerConfig

    def __init__(
        self,
        data_loader: ReplayDataLoader,
        config: TrainWorkerConfig,
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
                rollouts, self.config.max_input_length, self.config.max_output_length, self.config.pad_token_id
            )
            # shard onto the device mesh
            with self.config.trainer.device_mesh:
                sharded_batch = hax.shard(batch, self.config.trainer.compute_axis_mapping)

            yield sharded_batch


class StopTrainerException(Exception):
    """Exception to signal stopping the trainer."""

    pass


class TrainWorker:
    """Training worker that reads rollout data from a queue and trains the model using Levanter."""

    config: TrainWorkerConfig

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
        if isinstance(config.model.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer)
        else:
            self.tokenizer = config.model.tokenizer

        self.rollout_reader = config.rollout_storage.create_reader()

        self.replay_buffer = ReplayBuffer(
            config=config.replay_buffer,
            local_batch_size=config.trainer.train_batch_size,
            process_id=jax.process_index(),
            total_processes=jax.process_count(),
        )

        self.replay_loader = ReplayDataLoader(
            rollout_reader=self.rollout_reader,
            replay_buffer=self.replay_buffer,
            rollout_fetch_interval=1.0,
        )

        self.data_loader = StreamingRolloutLoader(
            self.replay_loader,
            config,
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
                model, self.reference_model, batch, key=key, kl_coef=config.kl_coef, clip_epsilon=0.2
            )
            # return ppo_loss(model, batch, key=key, kl_coef=config.kl_coef, clip_epsilon=0.5)

        with (
            config.trainer.device_mesh,
            hax.axis_mapping(config.trainer.compute_axis_mapping),
            Trainer(config.trainer, optimizer, _loss_function) as trainer,
            self.replay_loader,
        ):
            seed = config.trainer.seed
            _, training_key = jrandom.split(jrandom.PRNGKey(seed), 2)

            state = trainer.initial_state(training_key, model=self.reference_model)

            self._configure_training_hooks(trainer)

            try:
                trainer.train(state, self.data_loader)
            except StopTrainerException:
                pass

    def _configure_training_hooks(self, trainer):
        def _weight_transfer_hook(info: levanter.callbacks.StepInfo):
            self.weight_transfer_hook(trainer, info)

        trainer.add_hook(
            _weight_transfer_hook,
            every=self.config.weight_transfer.sync_interval_steps,
        )

        def _update_current_step(info: levanter.callbacks.StepInfo):
            self.replay_buffer.set_current_step(info.step)

        trainer.add_hook(_update_current_step, every=1)

        def _stop_on_signal(info: levanter.callbacks.StepInfo):
            if self._should_stop:
                raise StopTrainerException()

        trainer.add_hook(_stop_on_signal, every=1)

    def weight_transfer_hook(self, trainer: Trainer, info: levanter.callbacks.StepInfo):
        step = info.step
        state = info.state

        logger.info(
            "Transferring weights at step %d, loss=%s",
            step,
            info.loss,
        )

        model_params = state.model
        self.transfer_server.serve_weights(step, model_params)
        metrics = {
            f"train.weight_transfer.{k}": v for k, v in dataclasses.asdict(self.transfer_server.get_metrics()).items()
        }
        trainer.tracker.log(metrics, step=step)
        logger.info(f"Successfully transferred weights with ID {step}")

    def stop(self):
        """Stop the training worker."""
        self._should_stop = True
        self.transfer_server.cleanup()
