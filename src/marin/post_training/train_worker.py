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

import contextlib
import dataclasses
import logging
import os
import time
from collections import deque
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from jax.sharding import PartitionSpec as PS
from optax import softmax_cross_entropy_with_integer_labels
from pydantic import BaseModel
from scalax.sharding import MeshShardingHelper, TreePathShardingRule

from .model_helpers import (
    build_training_model,
    llama_config_from_model_config,
    load_tokenizer,
)
from .optimizer import load_adamw_optimizer
from .replay_buffer import ReplayBuffer, ReplayDataLoader
from .rollout_storage import RolloutBatch, RolloutReader
from .training_config import TrainingConfig
from .utils import (
    WandbLogger,
    checkpointer,
    delete_with_bucket,
    get_weight_decay_mask,
    global_norm,
    jax_distributed_barrier,
    jax_distributed_initalize,
    load_checkpoint,
)
from .weight_transfer_manager import create_weight_transfer_server

logger = logging.getLogger(__name__)


class StepMetrics(BaseModel):
    step: int = 0
    loss: float = 0
    train_step_time: float = 0
    batch_fetch_time: float = 0

    @contextlib.contextmanager
    def timer(self):
        start = time.time()
        yield lambda: time.time() - start


class TrainingWorker:
    """Training worker that reads rollout data from a queue and trains the model."""

    def __init__(
        self,
        training_config: TrainingConfig,
        rollout_reader: RolloutReader,
        coordinator=None,
    ):
        """Initialize training worker.

        Args:
            training_config: Training configuration.
            rollout_reader: Reader to read rollout data from.
            coordinator: Coordinator for weight transfer (required for RAY_REMOTING and JAX_TRANSFER_SERVER modes).
        """
        self.training_config = training_config
        self.coordinator = coordinator

        jax_distributed_initalize(**self.training_config.distributed.jax_distributed_initialize_config)
        jax_distributed_barrier()

        self.mesh = MeshShardingHelper(
            self.training_config.distributed.train_sharding,
            ["replica", "fsdp", "sequence", "tensor"],
            mesh_axis_splitting=self.training_config.distributed.physical_axis_splitting,
        )

        self.rollout_reader = rollout_reader

        self._should_stop = False
        self.weight_id = 0

        # Initialize components within mesh context
        with self.mesh.get_context():
            llama_config = llama_config_from_model_config(
                self.training_config.model.model_paths,
                self.training_config.model.model_config_override,
            )
            self.model = build_training_model(llama_config, self.training_config)
            jax_distributed_barrier()

            config = self.model.config
            self.train_params_sharding_rules = TreePathShardingRule(
                *config.get_partition_rules(
                    model_all_gather_axis=("fsdp", "sequence"),
                )
            )
            self.train_intermediate_sharding_rules = config.get_intermediate_sharding_rules(
                data_axis=("replica", "fsdp"),
                sequence_axis="sequence",
            )
            self.tokenizer = load_tokenizer(
                self.training_config.model.model_paths,
                self.training_config.model.tokenizer_override,
            )

            # Extract frequently used config values
            self.max_input_length = self.training_config.hyperparameters.max_input_length
            self.max_output_length = self.training_config.hyperparameters.max_output_length
            self.train_bsize = self.training_config.hyperparameters.train_bsize
            self.pad_token_id = self.training_config.hyperparameters.pad_token_id
            self.kl_coef = self.training_config.hyperparameters.kl_coef

            self._setup_optimizer()
            self._compile_functions()
            self._setup_logger()
            self.train_state_shape = jax.eval_shape(lambda: self.init_fn(jax.random.PRNGKey(0)))
            self.train_state_shard_fns, self.train_state_gather_fns = self.mesh.make_shard_and_gather_fns(
                self.train_state_shape, self.train_params_sharding_rules
            )
            self._setup_weight_transfer()

            # Initialize replay buffer and data loader after all setup
            self.replay_buffer = ReplayBuffer(
                capacity=32000,
                local_batch_size=self.train_bsize,
                recency_alpha=3.0,
                process_id=jax.process_index(),
                total_processes=jax.process_count(),
            )
            self.data_loader = ReplayDataLoader(
                rollout_reader=self.rollout_reader,
                replay_buffer=self.replay_buffer,
                rollout_fetch_interval=1.0,
            )

    def _setup_optimizer(self):
        """Setup optimizer configuration."""
        optim_config = self.training_config.hyperparameters.optim_config
        weight_decay_mask = get_weight_decay_mask(optim_config.weight_decay_exclusions)
        self.grad_accum_steps = optim_config.grad_accum_steps
        optimizer, self.optimizer_info = load_adamw_optimizer(config=optim_config, weight_decay_mask=weight_decay_mask)

        if self.grad_accum_steps > 1:
            optimizer = optax.MultiSteps(optimizer, self.grad_accum_steps)

        self.optimizer = optimizer

    def _setup_logger(self):
        """Setup logger for training metrics."""
        logging_config = self.training_config.logging
        if logging_config.enable is None:
            logging_config.enable = jax.process_index() == 0
        if logging_config.config_to_log is None:
            logging_config.config_to_log = dataclasses.asdict(self.training_config)

        logger.info(f"Initializing logger on worker {jax.process_index()}, enabled: {logging_config.enable}")

        self.logger = WandbLogger(
            self.training_config.logging.wandb_project,
            output_dir=self.training_config.output_dir,
            online=logging_config.online,
            prefix=logging_config.prefix,
            prefix_to_id=logging_config.prefix_to_id,
            experiment_id=logging_config.experiment_id,
            enable=logging_config.enable,
            config_to_log=logging_config.config_to_log,
        )

    def _compile_functions(self):
        """Compile JAX functions for training."""

        @partial(
            self.mesh.sjit,
            in_shardings=(self.train_params_sharding_rules,),
            out_shardings=self.train_params_sharding_rules,
            annotation_shardings=self.train_intermediate_sharding_rules,
        )
        def create_train_state_from_params(params):
            return TrainState.create(params=params, tx=self.optimizer, apply_fn=None)

        @partial(
            self.mesh.sjit,
            in_shardings=(PS(),),
            out_shardings=self.train_params_sharding_rules,
            annotation_shardings=self.train_intermediate_sharding_rules,
        )
        def init_fn(rng):
            params = self.model.init_weights(rng, (self.train_bsize, self.max_input_length + self.max_output_length - 1))
            return create_train_state_from_params(params)

        @partial(
            self.mesh.sjit,
            in_shardings=(self.train_params_sharding_rules, PS(), PS()),
            out_shardings=(self.train_params_sharding_rules, PS()),
            args_sharding_constraint=(
                self.train_params_sharding_rules,
                PS(),
                PS(("replica", "fsdp")),
            ),
            donate_argnums=(0,),
            annotation_shardings=self.train_intermediate_sharding_rules,
        )
        def train_step(train_state, rng, batch: RolloutBatch):
            def loss(params):
                logits = self.model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    position_ids=batch.position_ids,
                    params=params,
                    dropout_rng=rng,
                    train=True,
                ).logits
                logits = logits.astype(jnp.float32)
                token_loss = softmax_cross_entropy_with_integer_labels(logits, batch.target_ids)

                # Current policy log probabilities (π_θ at training time)
                current_logprobs = -token_loss

                # RLOO importance sampling ratio: π_θ(a|s) / π_old(a|s)
                # batch.policy_logprobs contains log probs from the policy that generated samples
                log_ratio = current_logprobs - batch.policy_logprobs
                ratio = jnp.exp(log_ratio)

                # RLOO loss with importance sampling
                # batch.loss_weights contains RLOO advantages: r_i - mean(r_j for j≠i)
                # Multiply by ratio for off-policy correction
                weighted_advantages = ratio * batch.loss_weights
                reinforce_loss = -jnp.sum(weighted_advantages * batch.loss_masks) / jnp.sum(batch.loss_masks)

                # KL regularization against reference policy (prevents drift)
                # This is standard KL(π_θ || π_ref) to keep policy close to reference
                kl_penalty = current_logprobs - batch.reference_logprobs
                kl_loss = -jnp.sum(kl_penalty * batch.loss_masks) / jnp.sum(batch.loss_masks)

                loss = reinforce_loss + self.kl_coef * kl_loss
                return loss, {
                    "reinforce_loss": reinforce_loss,
                    "kl_loss": kl_loss,
                }

            grad_fn = jax.value_and_grad(loss, has_aux=True)
            (loss, aux), grads = grad_fn(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            metrics = dict(
                loss=loss,
                reinforce_loss=aux["reinforce_loss"],
                kl_loss=aux["kl_loss"],
                learning_rate=self.optimizer_info["learning_rate_schedule"](train_state.step),
                gradient_norm=global_norm(grads),
                param_norm=global_norm(train_state.params),
            )
            return train_state, metrics

        # Store compiled functions
        self.create_train_state_from_params = create_train_state_from_params
        self.init_fn = init_fn
        self.train_step = train_step

    def _initialize_training_state(self):
        """Initialize training state with proper checkpoint loading."""
        jax_distributed_barrier()

        checkpoint_path = self._find_checkpoint_path()

        if checkpoint_path:
            train_state = self.create_train_state_from_params(
                load_checkpoint(
                    checkpoint_path,
                    shard_fns=self.train_state_shard_fns.params,
                    remove_dict_prefix=self.training_config.model.model_paths.remove_dict_prefix,
                    convert_to_dtypes=jax.tree_util.tree_map(
                        lambda x: self.training_config.model.training_param_dtype,
                        self.train_state_shape.params,
                    ),
                )
            )
        else:
            logger.warning("No params path provided, initializing with random weights...")
            train_state = self.init_fn(jax.random.PRNGKey(0))

        jax_distributed_barrier()
        self.checkpoint_queue = deque()
        return train_state

    def _find_checkpoint_path(self) -> str | None:
        """Find the best checkpoint path to load from."""
        # Check for existing checkpoint to resume from
        checkpoint_dir = os.path.join(self.training_config.output_dir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            checkpoint_steps = [int(d.split("_")[1]) for d in os.listdir(checkpoint_dir) if d.startswith("step_")]
            if checkpoint_steps:
                latest_checkpoint_step = max(checkpoint_steps)
                logger.info(f"Resuming training from checkpoint at step {latest_checkpoint_step}...")
                return str(
                    Path(self.training_config.output_dir)
                    / "checkpoints"
                    / f"step_{latest_checkpoint_step}"
                    / "params.msgpack"
                )

        # Check for provided params path
        if self.training_config.model.model_paths.params:
            logger.info("Restoring model from provided params path...")
            return self.training_config.model.model_paths.params

        # Check for provided train state path
        if self.training_config.model.model_paths.train_state:
            logger.info("Restoring model from provided train state path...")
            return self.training_config.model.model_paths.train_state

        return None

    def _setup_weight_transfer(self):
        """Initialize weight transfer manager."""
        # Setup checkpoint directory for weight transfer config
        if not self.training_config.weight_transfer.checkpoint_dir:
            self.training_config.weight_transfer.checkpoint_dir = os.path.join(
                self.training_config.output_dir, "checkpoints"
            )

        self.weight_transfer_manager = create_weight_transfer_server(
            config=self.training_config.weight_transfer,
            mesh=self.mesh,
            params_sharding_rules=self.train_params_sharding_rules,
            gather_fns=self.train_state_gather_fns,
            model_config=self.model.config,
            coordinator=self.coordinator,
        )

    def _transfer_weights(self, train_state, step):
        """Transfer weights using the weight transfer manager."""
        self.weight_id += 1
        logger.info("Transferring weights at step %d, weight_id %d...", step, self.weight_id)
        self.weight_transfer_manager.serve_weights(self.weight_id, train_state.params)

    def stop(self):
        """Stop the training worker."""
        self._should_stop = True

    def save_checkpoint(self, train_state, step):
        """Save model checkpoint."""
        jax_distributed_barrier()
        if (self.training_config.logging.max_checkpoints is not None) and (
            len(self.checkpoint_queue) >= self.training_config.logging.max_checkpoints
        ):
            old_step = self.checkpoint_queue.popleft()
            # TODO(power): this is an ugly way to check for the coordinator
            if self.logger.can_save():
                old_path = os.path.join(self.training_config.output_dir, "checkpoints", f"step_{old_step}")
                delete_with_bucket(old_path, recursive=True)

        logger.info(f"Saving checkpoint at step {step}...")

        metadata = dict(
            step=step,
            args_dict=dataclasses.asdict(self.training_config),
        )

        checkpoint_config = dataclasses.asdict(self.training_config.checkpoint)
        checkpoint_config.pop("save_model_freq", None)
        checkpoint_config.pop("save_optimizer_state", None)

        if self.training_config.checkpoint.save_optimizer_state is False:
            params = train_state.params
        else:
            params = train_state

        checkpointer(
            path=os.path.join(self.training_config.output_dir, "checkpoints", f"step_{step}"),
            params=params,
            config=self.model.config.to_dict(),
            gather_fns=self.train_state_gather_fns,
            metadata=metadata,
            active=jax.process_index() == 0,
            **checkpoint_config,
        )

        self.checkpoint_queue.append(step)
        logger.info("Checkpoint saved.")
        jax_distributed_barrier()

    def train(self):
        """Main training loop reading from rollout queue."""
        logger.info("Starting training worker...")
        jax_distributed_barrier()

        step = 0
        rng = jax.random.PRNGKey(0)

        logger.info(
            "Beginning training loop, target steps: %s",
            self.training_config.hyperparameters.num_train_steps,
        )
        train_state = self._initialize_training_state()

        if self.training_config.logging.save_initial_checkpoint:
            self.save_checkpoint(train_state, 0)

        # Start data loader
        with self.data_loader:
            while step < self.training_config.hyperparameters.num_train_steps:
                if self._should_stop:
                    logger.info("Stop signal received, stopping training worker...")
                    break

                logger.info(f"Starting training step {step}...")
                step_metrics = StepMetrics(step=step)

                # Get training data from replay buffer
                with step_metrics.timer() as batch_fetch_timer:
                    batch = self.data_loader.get_training_batch(timeout=60.0)
                    if batch is None:
                        logger.info("No training batch available, waiting...")
                        continue
                    step_metrics.batch_fetch_time = batch_fetch_timer()

                step += 1

                # Truncate batch to max sequence length just in case we are reading
                # an older inference file.
                max_seq_len = self.training_config.model.model_config_override.max_sequence_length
                batch = batch.truncate_sequence(max_seq_len)
                batch = batch.to_jax()

                logger.info("Training on batch with shape: %s", batch.input_ids.shape)

                with step_metrics.timer() as train_step_timer:
                    rng, subrng = jax.random.split(rng)
                    train_state, metrics = self.train_step(train_state, subrng, batch)
                    step_metrics.train_step_time = train_step_timer()

                with step_metrics.timer() as sync_timer:
                    jax_distributed_barrier()

                step_metrics.train_step_time += sync_timer()
                if step % 10 == 0:
                    logger.info("Finished training step", step_metrics.model_dump())

                if self.training_config.logging.log_freq > 0 and (step % self.training_config.logging.log_freq == 0):
                    log_metrics = {}
                    log_metrics.update(jax.device_get(metrics))
                    log_metrics.update(self.weight_transfer_manager.get_metrics())
                    log_metrics.update(step_metrics.model_dump())
                    self.logger.log(log_metrics)
                    logger.info(f"Logging metrics at step {step}... {log_metrics}")

                if (step % self.training_config.weight_transfer.sync_interval_steps) == 0:
                    self._transfer_weights(train_state, step)

                if (
                    self.training_config.checkpoint.save_model_freq > 0
                    and step % self.training_config.checkpoint.save_model_freq == 0
                ):
                    logger.info("Saving checkpoint at step %d...", step)
                    self.save_checkpoint(train_state, step)

        # Final checkpoint and weight transfer
        if self.training_config.checkpoint.save_model_freq > 0:
            self._transfer_weights(train_state, step)
            self.save_checkpoint(train_state, step)  # Always save final checkpoint

        # Cleanup
        self.weight_transfer_manager.cleanup()
        jax_distributed_barrier()
        self.logger.finish()
        jax_distributed_barrier()

        logger.info(f"Training completed after {step} steps")
