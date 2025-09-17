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

For now we use GCS as a storage backend.
"""

import asyncio
import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrandom
import levanter
from levanter.data import AsyncDataset
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.trainer import Trainer, TrainerConfig
from optax import softmax_cross_entropy_with_integer_labels

from marin.post_training.training_config import OptimizerConfig

from .replay_buffer import ReplayBuffer, ReplayDataLoader
from .rollout_storage import RolloutBatch, RolloutReader

logger = logging.getLogger(__name__)


def compute_rloo_loss(
    model: LmHeadModel, batch: RolloutBatch, *, key: jax.Array | None = None, kl_coef: float = 0.1
) -> jax.Array:
    """Compute RLOO (Reward Leave-One-Out) loss with importance sampling.

    Args:
        model: The language model
        batch: Batch containing rollout data with RLOO advantages
        key: JAX random key for dropout
        kl_coef: Coefficient for KL regularization

    Returns:
        Tuple of (loss, aux_metrics)
    """
    # Get logits from current policy
    model_output = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        position_ids=batch.position_ids,
        key=key,
    )
    # Handle both tuple and object outputs
    logits = model_output.logits if hasattr(model_output, "logits") else model_output[0]

    logits = logits.astype(jnp.float32)
    token_loss = softmax_cross_entropy_with_integer_labels(logits, batch.target_ids)

    # Current policy log probabilities (π_θ at training time)
    current_logprobs = -token_loss

    # RLOO importance sampling ratio: π_θ(a|s) / π_old(a|s)
    # batch.policy_logprobs contains log probs from the policy that generated samples
    log_ratio = jnp.subtract(current_logprobs, batch.policy_logprobs)
    ratio = jnp.exp(log_ratio)

    # RLOO loss with importance sampling
    # batch.loss_weights contains RLOO advantages: r_i - mean(r_j for j≠i)
    # Multiply by ratio for off-policy correction
    weighted_advantages = ratio * batch.loss_weights
    reinforce_loss = -jnp.sum(weighted_advantages * batch.loss_masks) / jnp.sum(batch.loss_masks)

    # KL regularization against reference policy (prevents drift)
    # This is standard KL(π_θ || π_ref) to keep policy close to reference
    kl_penalty = jnp.subtract(current_logprobs, batch.reference_logprobs)
    kl_loss = -jnp.sum(kl_penalty * batch.loss_masks) / jnp.sum(batch.loss_masks)

    loss = reinforce_loss + kl_coef * kl_loss

    # Log auxiliary metrics to tracker (Levanter handles this automatically)
    import levanter.tracker

    with levanter.tracker.defer_tracker_for_jit() as tracker:
        tracker.log(
            {
                "rloo/reinforce_loss": reinforce_loss,
                "rloo/kl_loss": kl_loss,
                "rloo/importance_ratio_mean": jnp.mean(ratio),
                "rloo/importance_ratio_max": jnp.max(ratio),
            }
        )

    return loss


class RolloutDataset(AsyncDataset[RolloutBatch]):
    """AsyncDataset wrapper for rollout data from replay buffer."""

    def __init__(self, data_loader: ReplayDataLoader, timeout: float = 60.0):
        super().__init__()
        self.data_loader = data_loader
        self.timeout = timeout

    async def async_len(self) -> int:
        """Return a large number for infinite streaming dataset."""
        return 1_000_000  # Large number to indicate streaming

    async def final_length_is_known(self) -> bool:
        """Streaming dataset never has a final known length."""
        return False

    def is_finite(self) -> bool:
        """Rollout dataset is infinite/streaming."""
        return False

    async def current_len(self) -> int | None:
        """Return current size of the replay buffer."""
        return len(self.data_loader.replay_buffer)

    async def get_batch(self, indices: list[int]) -> list[RolloutBatch]:
        """Get a batch of rollout data from the replay buffer."""
        # For rollout data, we ignore indices and get whatever is available
        batch = await asyncio.to_thread(self.data_loader.get_training_batch, timeout=self.timeout)
        if batch is None:
            return []
        return [self._convert_to_named_batch(batch)]

    def _convert_to_named_batch(self, batch: RolloutBatch) -> RolloutBatch:
        """Convert numpy arrays to JAX arrays with proper named axes."""
        import haliax as hax
        import jax.numpy as jnp
        from haliax import Axis

        # Create batch and sequence axes
        batch_size, seq_len = batch.input_ids.shape
        Batch = Axis("batch", batch_size)
        Pos = Axis("position", seq_len)

        # Convert to JAX NamedArrays
        return RolloutBatch(
            input_ids=hax.named(jnp.array(batch.input_ids), (Batch, Pos)),
            attention_mask=hax.named(jnp.array(batch.attention_mask), (Batch, Pos)),
            position_ids=hax.named(jnp.array(batch.position_ids), (Batch, Pos)),
            target_ids=hax.named(jnp.array(batch.target_ids), (Batch, Pos)),
            loss_weights=hax.named(jnp.array(batch.loss_weights), (Batch, Pos)),
            loss_masks=hax.named(jnp.array(batch.loss_masks), (Batch, Pos)),
            reference_logprobs=hax.named(jnp.array(batch.reference_logprobs), (Batch, Pos)),
            policy_logprobs=hax.named(jnp.array(batch.policy_logprobs), (Batch, Pos)),
        )


logger = logging.getLogger(__name__)


class StopTrainerException(Exception):
    """Exception to signal stopping the trainer."""

    pass


class RolloutDataConfig:
    """Data config that returns our rollout dataset."""

    def __init__(self, rollout_dataset):
        self.rollout_dataset = rollout_dataset

    def train_set(self, pos_axis, batch_schedule, key=None, epochs=0):
        """Return our rollout dataset."""
        return self.rollout_dataset

    def tagged_eval_sets(self, pos_axis):
        """Return empty eval sets since we don't have eval for RLOO."""
        return {}

    @property
    def the_tokenizer(self):
        """Return a dummy tokenizer - we don't use it for RLOO."""
        return None


@dataclass
class TrainingWorkerConfig:
    """Configuration for Levanter-based RL training worker."""

    rollout_reader: RolloutReader
    model: LmConfig
    trainer: TrainerConfig
    optimizer: OptimizerConfig

    # RLOO-specific parameters
    kl_coef: float = 0.1
    reference_logprobs_bsize: int = 32

    # Weight transfer settings
    weight_transfer_sync_interval: int = 100


class TrainingWorker:
    """Training worker that reads rollout data from a queue and trains the model using Levanter."""

    def __init__(
        self,
        config: TrainingWorkerConfig,
        coordinator=None,
    ):
        """Initialize training worker.

        Args:
            config: Training worker configuration with Levanter components.
            coordinator: Coordinator for weight transfer.
        """
        self.config = config
        self.coordinator = coordinator
        self._should_stop = False
        self.weight_id = 0

        self.rollout_reader = config.rollout_reader

        self.replay_buffer = ReplayBuffer(
            capacity=32000,
            local_batch_size=config.trainer.train_batch_size,
            recency_alpha=3.0,
            process_id=jax.process_index(),
            total_processes=jax.process_count(),
        )
        self.data_loader = ReplayDataLoader(
            rollout_reader=self.rollout_reader,
            replay_buffer=self.replay_buffer,
            rollout_fetch_interval=1.0,
        )

        # Create rollout dataset for Levanter
        self.rollout_dataset = RolloutDataset(self.data_loader)

    def train(self):
        """Main training method using Levanter's standard train_lm infrastructure."""
        logger.info("Starting RLOO training with Levanter...")
        config = self.config
        levanter.initialize(self.config.trainer)
        optimizer = config.optimizer.build(config.trainer.num_train_steps)

        def _rloo_loss_function(model, batch, key):
            return compute_rloo_loss(model, batch, key=key, kl_coef=config.kl_coef)

        with Trainer(config.trainer, optimizer, _rloo_loss_function) as trainer:
            seed = config.trainer.seed
            data_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 3)

            data_config = RolloutDataConfig(self.rollout_dataset)
            train_dataset = data_config.train_set(config.model.Pos, config.trainer.batch_schedule, key=data_key)

            state = trainer.initial_state(
                training_key,
                model_init=lambda: config.model.build(
                    # Need to handle vocab axis properly
                    config.model.Vocab if hasattr(config.model, "Vocab") else None,
                    key=model_key,
                ),
            )

            trainer.add_hook(
                self.create_weight_transfer_hook(),
                every=self.config.weight_transfer_sync_interval,
            )
            checkpointer = trainer.config.checkpointer.create("train-id-0")
            trainer.add_hook(checkpointer)

            def _stop_on_signal(info: levanter.callbacks.StepInfo):
                if self._should_stop:
                    raise StopTrainerException()

            trainer.add_hook(_stop_on_signal, every=1)

            train_loader = trainer.data_loader(train_dataset)
            try:
                trainer.train(state, train_loader)
            except StopTrainerException:
                pass

    def create_weight_transfer_hook(self):
        def weight_transfer_hook(info: levanter.callbacks.StepInfo):
            step = info.step
            state = info.state

            self.weight_id += 1
            logger.info("Transferring weights at step %d, weight_id %d...", step, self.weight_id)

            model_params = state.model
            self.coordinator.serve_weights(self.weight_id, model_params)
            logger.info(f"Successfully transferred weights with ID {self.weight_id}")

        return weight_transfer_hook

    def stop(self):
        """Stop the training worker."""
        self._should_stop = True
