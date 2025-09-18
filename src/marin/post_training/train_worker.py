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

import logging
from dataclasses import dataclass

import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import levanter
import levanter.tracker
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.optim import OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from optax import softmax_cross_entropy_with_integer_labels
from transformers import PreTrainedTokenizer

from marin.post_training import weight_transfer_manager
from marin.post_training.training_config import WeightTransferConfig

from .replay_buffer import ReplayBuffer, ReplayDataLoader
from .rollout_storage import JaxRolloutBatch, RolloutBatch, RolloutReader

logger = logging.getLogger(__name__)


def compute_rloo_loss(
    model: LmHeadModel,
    batch: JaxRolloutBatch,
    *,
    key: jax.Array | None = None,
    kl_coef: float = 0.1,
) -> jax.Array:
    """Compute RLOO (Reward Leave-One-Out) loss with importance sampling.

    Args:
        model: The language model
        batch: JaxRolloutBatch containing rollout data with RLOO advantages
        key: JAX random key for dropout
        kl_coef: Coefficient for KL regularization

    Returns:
        Tuple of (loss, aux_metrics)
    """
    # Get logits from current policy
    model_output = model(
        input_ids=batch.input_ids,
        attn_mask=batch.attention_mask,
        pos_ids=batch.position_ids,
        key=key,
    )
    # LlamaLMHeadModel returns a NamedArray
    logits = model_output

    # Extract raw JAX arrays from NamedArrays for optax/jax functions
    logits_array = logits.array
    target_ids_array = batch.target_ids.array
    policy_logprobs_array = batch.policy_logprobs.array
    loss_weights_array = batch.loss_weights.array
    loss_masks_array = batch.loss_masks.array
    reference_logprobs_array = batch.reference_logprobs.array

    logits_array = logits_array.astype(jnp.float32)
    token_loss = softmax_cross_entropy_with_integer_labels(logits_array, target_ids_array)

    # Current policy log probabilities (π_θ at training time)
    current_logprobs = -token_loss

    # RLOO importance sampling ratio: π_θ(a|s) / π_old(a|s)
    log_ratio = jnp.subtract(current_logprobs, policy_logprobs_array)
    ratio = jnp.exp(log_ratio)

    # RLOO loss with importance sampling
    # batch.loss_weights contains RLOO advantages: r_i - mean(r_j for j≠i)
    weighted_advantages = ratio * loss_weights_array
    reinforce_loss = -jnp.sum(weighted_advantages * loss_masks_array) / jnp.sum(loss_masks_array)

    # KL regularization against reference policy (prevents drift)
    kl_penalty = jnp.subtract(current_logprobs, reference_logprobs_array)
    kl_loss = -jnp.sum(kl_penalty * loss_masks_array) / jnp.sum(loss_masks_array)

    loss = reinforce_loss + kl_coef * kl_loss

    # with levanter.tracker.defer_tracker_for_jit() as metrics:
    #     metrics.update(
    #         {
    #             "rloo/token_loss": jnp.mean(token_loss),
    #             "rloo/log_ratio": jnp.mean(log_ratio),
    #             "rloo/reinforce_loss": reinforce_loss,
    #             "rloo/loss": loss,
    #             "rloo/kl_loss": kl_loss,
    #             "rloo/importance_ratio_mean": jnp.mean(ratio),
    #             "rloo/importance_ratio_max": jnp.max(ratio),
    #         }
    #     )

    return loss


class StreamingRolloutLoader:
    """Direct loader for streaming rollout data that bypasses Levanter's DataLoader."""

    def __init__(self, data_loader: ReplayDataLoader, config):
        """Initialize the streaming rollout loader.

        Args:
            data_loader: The replay data loader to get batches from
            config: Trainer config with mesh and axis mapping information
        """
        self.data_loader = data_loader
        self.config = config
        self.timeout = 60.0

    def __iter__(self):
        """Yield batches continuously from the replay buffer."""
        while True:
            batch = self.data_loader.get_training_batch(timeout=self.timeout)
            if not batch:
                logger.warning("No batch received from data loader within timeout, retrying...")
                continue

            named_batch = self._convert_to_named_batch(batch)
            with self.config.device_mesh:
                named_batch = hax.shard(named_batch, self.config.compute_axis_mapping)

            yield named_batch

    def _convert_to_named_batch(self, batch: RolloutBatch):
        """Convert numpy arrays to JAX arrays with proper named axes."""
        from marin.post_training.rollout_storage import JaxRolloutBatch

        # Convert to JAX arrays first
        jax_batch = batch.to_jax()

        # Create batch and sequence axes
        batch_size, seq_len = jax_batch.input_ids.shape
        Batch = hax.Axis("batch", batch_size)
        Pos = hax.Axis("position", seq_len)

        # Add named axes to all fields
        return JaxRolloutBatch(
            input_ids=hax.named(jax_batch.input_ids, (Batch, Pos)),
            attention_mask=hax.named(jax_batch.attention_mask, (Batch, Pos)),
            position_ids=hax.named(jax_batch.position_ids, (Batch, Pos)),
            target_ids=hax.named(jax_batch.target_ids, (Batch, Pos)),
            loss_weights=hax.named(jax_batch.loss_weights, (Batch, Pos)),
            loss_masks=hax.named(jax_batch.loss_masks, (Batch, Pos)),
            reference_logprobs=hax.named(jax_batch.reference_logprobs, (Batch, Pos)),
            policy_logprobs=hax.named(jax_batch.policy_logprobs, (Batch, Pos)),
        )


logger = logging.getLogger(__name__)


class StopTrainerException(Exception):
    """Exception to signal stopping the trainer."""

    pass


@dataclass
class TrainWorkerConfig:
    """Configuration for Levanter-based RL training worker."""

    rollout_reader: RolloutReader
    model: LmConfig
    trainer: TrainerConfig
    optimizer: OptimizerConfig

    tokenizer: str | PreTrainedTokenizer

    # RLOO-specific parameters
    kl_coef: float = 0.1
    reference_logprobs_bsize: int = 32

    weight_transfer: WeightTransferConfig


class TrainWorker:
    """Training worker that reads rollout data from a queue and trains the model using Levanter."""

    def __init__(
        self,
        config: TrainWorkerConfig,
    ):
        """Initialize training worker.

        Args:
            config: Training worker configuration with Levanter components.
            coordinator: Coordinator for weight transfer.
        """
        levanter.initialize(config.trainer)
        self.config = config
        self._should_stop = False
        self.weight_id = 0
        if isinstance(config.tokenizer, str):
            self.tokenizer = PreTrainedTokenizer.from_pretrained(config.tokenizer)
        else:
            self.tokenizer = config.tokenizer

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

        self.transfer_server = weight_transfer_manager.create_weight_transfer_server(
            config.weight_transfer,
            mesh=self.config.trainer.device_mesh,
            axis_mapping=self.config.trainer.compute_axis_mapping,
        )

    def train(self):
        """Main training method using Levanter's standard train_lm infrastructure."""
        logger.info("Starting RLOO training with Levanter...")

        config = self.config
        levanter.initialize(self.config.trainer)
        optimizer = config.optimizer.build(config.trainer.num_train_steps)

        def _rloo_loss_function(model, batch, key):
            return compute_rloo_loss(model, batch, key=key, kl_coef=config.kl_coef)

        with Trainer(config.trainer, optimizer, _rloo_loss_function) as trainer, self.data_loader:
            seed = config.trainer.seed
            model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 2)

            Vocab = hax.Axis("vocab", self.tokenizer.vocab_size)

            state = trainer.initial_state(
                training_key,
                model_init=lambda: config.model.build(Vocab, key=model_key),
            )

            self._configure_training_hooks(trainer)

            # Use our custom streaming loader instead of Levanter's DataLoader
            train_loader = StreamingRolloutLoader(self.data_loader, config.trainer)

            try:
                trainer.train(state, train_loader)
            except StopTrainerException:
                pass

    def _configure_training_hooks(self, trainer):
        """Configure training hooks. Override in tests for additional hooks."""
        trainer.add_hook(
            self.create_weight_transfer_hook(),
            every=self.config.weight_transfer_sync_interval,
        )
        checkpointer = trainer.config.checkpointer.create("train-id-0")

        def _checkpoint_step(info: levanter.callbacks.StepInfo):
            logger.info("Checking for checkpoint at step %d", info.step)
            checkpointer.on_step(info, force=True)

        trainer.add_hook(_checkpoint_step, every=1)

        def _stop_on_signal(info: levanter.callbacks.StepInfo):
            if self._should_stop:
                raise StopTrainerException()

        trainer.add_hook(_stop_on_signal, every=1)

    def create_weight_transfer_hook(self):
        def weight_transfer_hook(info: levanter.callbacks.StepInfo):
            step = info.step
            state = info.state

            self.weight_id += 1
            logger.info("Transferring weights at step %d, weight_id %d...", step, self.weight_id)

            model_params = state.model
            self.transfer_server.serve_weights(self.weight_id, model_params)
            logger.info(f"Successfully transferred weights with ID {self.weight_id}")

        return weight_transfer_hook

    def stop(self):
        """Stop the training worker."""
        self._should_stop = True
