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
import jax.numpy as jnp
import jax.random as jrandom
import levanter
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.optim import OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from optax import softmax_cross_entropy_with_integer_labels
from transformers import AutoTokenizer

from marin.post_training import weight_transfer
from marin.post_training.weight_transfer import WeightTransferConfig

from .model_utils import load_model_from_checkpoint
from .replay_buffer import ReplayBuffer, ReplayDataLoader
from .rollout_storage import JaxRolloutBatch, RolloutStorageConfig

logger = logging.getLogger(__name__)


def ppo_loss(
    model: LmHeadModel,
    batch: JaxRolloutBatch,
    *,
    key: jax.Array | None,
    kl_coef: float,
    clip_epsilon: float,
) -> jax.Array:
    """Compute PPO-style loss with RLOO advantages."""
    model_output = model(
        input_ids=batch.input_ids,
        attn_mask=batch.attention_mask,
        pos_ids=batch.position_ids,
        key=key,
    )

    logits = model_output.array.astype(jnp.float32)

    token_ce_loss = softmax_cross_entropy_with_integer_labels(logits, batch.target_ids.array)
    current_logprobs = -token_ce_loss

    # Get the old policy's log probs (from the worker policy that collected the data)
    old_logprobs = batch.policy_logprobs.array

    # Compute importance sampling ratio exp(log π_current - log π_old)
    log_ratio = current_logprobs - old_logprobs
    ratio = jnp.exp(log_ratio)

    # RLOO advantages (returned from the worker, and smeared across tokens)
    advantages = batch.loss_weights.array

    # Get the mask for valid tokens (e.g., excluding padding)
    mask = batch.loss_masks.array

    # PPO objective with clipping
    # We want to maximize advantage-weighted log probs, so we minimize the negative

    # Unclipped surrogate objective: ratio * advantage
    surrogate_1 = ratio * advantages

    # Clipped surrogate objective
    clipped_ratio = jnp.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    surrogate_2 = clipped_ratio * advantages

    # PPO takes the minimum of the two (pessimistic bound)
    # We're minimizing negative rewards, so we take minimum of surrogate objectives
    # then negate to convert maximization to minimization
    ppo_loss_per_token = -jnp.minimum(surrogate_1, surrogate_2)

    # Apply mask and average
    ppo_loss = jnp.sum(ppo_loss_per_token * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    # KL penalty from reference policy (optional regularization)
    # KL(π_current || π_ref) ≈ π_current * (log π_current - log π_ref)
    reference_logprobs = batch.reference_logprobs.array
    kl_div = jnp.exp(current_logprobs) * (current_logprobs - reference_logprobs)
    kl_loss = jnp.sum(kl_div * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    # Total loss
    total_loss = ppo_loss + kl_coef * kl_loss
    return total_loss


def rloo_loss_with_importance_sampling(
    model: LmHeadModel,
    batch: JaxRolloutBatch,
    *,
    key: jax.Array | None,
    kl_coef: float,
) -> jax.Array:
    """Compute RLOO (Reward Leave-One-Out) loss with importance sampling for off-policy data.

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

    logits = model_output

    logits_array = logits.array
    target_ids_array = batch.target_ids.array
    policy_logprobs_array = batch.policy_logprobs.array
    loss_weights_array = batch.loss_weights.array
    loss_masks_array = batch.loss_masks.array
    reference_logprobs_array = batch.reference_logprobs.array

    logits_array = logits_array.astype(jnp.float32)
    token_loss = softmax_cross_entropy_with_integer_labels(logits_array, target_ids_array)

    current_logprobs = -token_loss

    # importance sampling since we're using off-policy data
    # ratio = π_current(a|s) / π_old(a|s) = log(π_current) - log(π_old)
    log_ratio = jnp.subtract(current_logprobs, policy_logprobs_array)
    ratio = jnp.exp(log_ratio)

    # N.B. This should be enabled, but we seem to be training far enough
    # off of policy that we're not learning anything when we clip.
    # ratio = jnp.clip(ratio, min=0.8, max=1.2)

    # RLOO loss with importance sampling
    # batch.loss_weights contains RLOO advantages: r_i - mean(r_j for j≠i)
    weighted_loss = -ratio * loss_weights_array * loss_masks_array
    reinforce_loss = jnp.sum(weighted_loss) / jnp.sum(loss_masks_array)

    # KL regularization
    kl_penalty = jnp.exp(current_logprobs) * (current_logprobs - reference_logprobs_array)
    kl_loss = kl_coef * jnp.sum(kl_penalty * loss_masks_array) / jnp.sum(loss_masks_array)

    loss = reinforce_loss + kl_loss
    return loss


class StreamingRolloutLoader:
    """Direct loader for streaming rollout data.

    Rollouts are a continous stream of data, not really well modeled by the
    default Levanter indexing API. Instead of implemented a Dataset, we
    implement the expected data loader interface directly.
    """

    def __init__(self, data_loader: ReplayDataLoader, config: TrainerConfig):
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

    max_samples: int = 1
    """Maximum number of times an example can be sampled before removal."""


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
            capacity=config.replay_buffer.capacity,
            local_batch_size=config.trainer.train_batch_size,
            recency_alpha=config.replay_buffer.alpha,
            max_samples=config.replay_buffer.max_samples,
            process_id=jax.process_index(),
            total_processes=jax.process_count(),
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

        initial_model = load_model_from_checkpoint(
            checkpoint=config.initial_checkpoint,
            model_config=config.model,
            trainer_config=config.trainer,
            vocab_axis=Vocab,
            tokenizer=self.tokenizer,
            mesh=config.trainer.device_mesh,
            axis_mapping=self.config.trainer.parameter_axis_mapping,
            key=model_key,
        )

        # Reference model is the frozen initial model (for KL regularization)
        self.reference_model = initial_model

    def train(self):
        """Main training method using Levanter's standard train_lm infrastructure."""
        logger.info("Starting RLOO training with Levanter...")

        config = self.config
        optimizer = config.optimizer.build(config.trainer.num_train_steps)

        def _loss_function(model, batch, key):
            return rloo_loss_with_importance_sampling(model, batch, key=key, kl_coef=config.kl_coef)

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
            train_loader = StreamingRolloutLoader(self.data_loader, config.trainer)

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
