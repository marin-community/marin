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

import dataclasses
import logging
import os
from collections import deque
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from jax.sharding import PartitionSpec as PS
from optax import softmax_cross_entropy_with_integer_labels
from scalax.sharding import MeshShardingHelper, TreePathShardingRule

from .model_helpers import (
    build_training_model,
    llama_config_from_model_config,
    load_tokenizer,
)
from .optimizer import load_adamw_optimizer
from .rollout_storage import RolloutBatch, RolloutReader
from .training_config import TrainingConfig, TrainWorkerConfig
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

logger = logging.getLogger(__name__)


class TrainingWorker:
    """Training worker that reads rollout data from a queue and trains the model."""

    def __init__(
        self,
        training_config: TrainingConfig,
        worker_config: TrainWorkerConfig,
        rollout_reader: RolloutReader | None = None,
    ):
        """Initialize training worker.

        Args:
            training_config: Training configuration.
            worker_config: Worker-specific configuration.
            rollout_reader: Reader to read rollout data from. If None, creates FileRolloutReader.
        """
        self.training_config = training_config
        self.worker_config = worker_config

        # Initialize JAX distributed
        jax_distributed_initalize(**training_config.distributed.jax_distributed_initalize_config)
        jax_distributed_barrier()

        self.mesh = MeshShardingHelper(
            self.training_config.distributed.sharding,
            ["replica", "fsdp", "sequence", "tensor"],
            mesh_axis_splitting=self.training_config.distributed.physical_axis_splitting,
        )

        self.rollout_reader = rollout_reader
        self._should_stop = False

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

        logger.info(f"Initializing logger. Worker id: {jax.process_index}, enabled: {logging_config.enable}")

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
        def train_step(train_state, rng, batch):
            def loss(params):
                logits = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    position_ids=batch["position_ids"],
                    params=params,
                    dropout_rng=rng,
                    train=True,
                ).logits
                logits = logits.astype(jnp.float32)
                token_loss = softmax_cross_entropy_with_integer_labels(logits, batch["target_ids"])
                log_ratio = jnp.exp((-token_loss) - jax.lax.stop_gradient(-token_loss))
                weighted_log_ratio = log_ratio * batch["loss_weights"]
                reinforce_loss = jnp.mean(-weighted_log_ratio, where=batch["loss_masks"] > 0.0)
                ref_log_ratio = batch["reference_logprobs"] + token_loss
                kl_loss = jnp.exp(ref_log_ratio) - 1 - ref_log_ratio
                kl_loss = jnp.mean(kl_loss, where=batch["loss_masks"] > 0.0)
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
        train_state_shape = jax.eval_shape(lambda: self.init_fn(jax.random.PRNGKey(0)))
        self.train_state_shard_fns, self.train_state_gather_fns = self.mesh.make_shard_and_gather_fns(
            train_state_shape, self.train_params_sharding_rules
        )

        # Initialize training state from checkpoint or params
        latest_checkpoint_step = -1
        if self.logger.can_save():
            checkpoint_dir = os.path.join(self.training_config.output_dir, "checkpoints")
            if os.path.exists(checkpoint_dir):
                checkpoint_steps = [int(d.split("_")[1]) for d in os.listdir(checkpoint_dir) if d.startswith("step_")]
                if checkpoint_steps:
                    latest_checkpoint_step = max(checkpoint_steps)

        if latest_checkpoint_step > 0:
            logger.info(f"Resuming training from checkpoint at step {latest_checkpoint_step}...")
            checkpoint_path = os.path.join(
                self.training_config.output_dir,
                "checkpoints",
                f"step_{latest_checkpoint_step}",
                "params.msgpack",
            )
            train_state = self.create_train_state_from_params(
                load_checkpoint(
                    checkpoint_path,
                    shard_fns=self.train_state_shard_fns.params,
                    remove_dict_prefix=self.training_config.model.model_paths.remove_dict_prefix,
                    convert_to_dtypes=jax.tree_util.tree_map(
                        lambda x: self.training_config.model.training_param_dtype,
                        train_state_shape.params,
                    ),
                )
            )
        elif self.training_config.model.model_paths.params:
            train_state = self.create_train_state_from_params(
                load_checkpoint(
                    self.training_config.model.model_paths.params,
                    shard_fns=self.train_state_shard_fns.params,
                    remove_dict_prefix=self.training_config.model.model_paths.remove_dict_prefix,
                    convert_to_dtypes=jax.tree_util.tree_map(
                        lambda x: self.training_config.model.training_param_dtype,
                        train_state_shape.params,
                    ),
                )
            )
        elif self.training_config.model.model_paths.train_state:
            train_state = load_checkpoint(
                self.training_config.model.model_paths.train_state,
                shard_fns=self.train_state_shard_fns,
                remove_dict_prefix=self.training_config.model.model_paths.remove_dict_prefix,
                convert_to_dtypes=jax.tree_util.tree_map(
                    lambda x: self.training_config.model.training_param_dtype, train_state_shape
                ),
            )
        else:
            logger.warning("No params path provided, initializing with random weights...")
            train_state = self.init_fn(jax.random.PRNGKey(0))

        jax_distributed_barrier()
        self.checkpoint_queue = deque()
        return train_state

    def stop(self):
        """Stop the training worker."""
        self._should_stop = True

    def _slice_batch(self, batch: RolloutBatch, start_idx: int, end_idx: int) -> RolloutBatch:
        """Slice a RolloutBatch to get a subset of samples."""
        return RolloutBatch(
            input_ids=batch.input_ids[start_idx:end_idx],
            attention_mask=batch.attention_mask[start_idx:end_idx],
            position_ids=batch.position_ids[start_idx:end_idx],
            target_ids=batch.target_ids[start_idx:end_idx],
            loss_weights=batch.loss_weights[start_idx:end_idx],
            loss_masks=batch.loss_masks[start_idx:end_idx],
            reference_logprobs=batch.reference_logprobs[start_idx:end_idx],
            metadata=batch.metadata,
        )

    def _convert_batch_to_jax(self, batch: RolloutBatch) -> dict[str, jnp.ndarray]:
        """Convert RolloutBatch to JAX arrays for training."""
        return {
            "input_ids": jnp.array(batch.input_ids),
            "attention_mask": jnp.array(batch.attention_mask),
            "position_ids": jnp.array(batch.position_ids),
            "target_ids": jnp.array(batch.target_ids),
            "loss_weights": jnp.array(batch.loss_weights),
            "loss_masks": jnp.array(batch.loss_masks),
            "reference_logprobs": jnp.array(batch.reference_logprobs),
        }

    def save_checkpoint(self, train_state, step):
        """Save model checkpoint."""
        jax_distributed_barrier()
        if (self.training_config.logging.max_checkpoints is not None) and (
            len(self.checkpoint_queue) >= self.training_config.logging.max_checkpoints
        ):
            old_step = self.checkpoint_queue.popleft()
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

        checkpointer(
            path=os.path.join(self.training_config.output_dir, "checkpoints", f"step_{step}"),
            train_state=train_state,
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

        # Initialize training state
        train_state = self._initialize_training_state()

        step = 0
        rng = jax.random.PRNGKey(0)

        logger.info(
            "Beginning training loop, target steps: %s",
            self.training_config.hyperparameters.num_train_steps,
        )

        if self.training_config.logging.save_initial_checkpoint:
            self.save_checkpoint(train_state, 0)

        while step < self.training_config.hyperparameters.num_train_steps and not self._should_stop:
            logger.info(f"Training step {step}")
            # TODO(power) -- ensure we read disjoint batches from the rollouts.
            batch = self.rollout_reader.read_batch(timeout=5)

            if batch is None:
                logger.info("No batch available, waiting for new data...")
                continue

            # Get batch size and slice if necessary
            batch_size = len(batch.input_ids)

            # Process batch in chunks of train_bsize
            for i in range(0, batch_size, self.train_bsize):
                if step >= self.training_config.hyperparameters.num_train_steps or self._should_stop:
                    break

                jax_distributed_barrier()
                end_idx = min(i + self.train_bsize, batch_size)
                batch_slice = self._slice_batch(batch, i, end_idx)
                logger.info(f"Training on batch of shape: {batch_slice.attention_mask.shape}")
                jax_batch = self._convert_batch_to_jax(batch_slice)

                # Perform training step
                rng, subrng = jax.random.split(rng)
                train_state, metrics = self.train_step(train_state, subrng, jax_batch)
                jax_distributed_barrier()

                step += 1

                if self.training_config.logging.log_freq > 0 and step % self.training_config.logging.log_freq == 0:
                    log_metrics = {"step": step}
                    log_metrics.update(jax.device_get(metrics))
                    log_metrics.update(batch_slice.metadata)
                    self.logger.log(log_metrics)
                    logger.info(f"Step {step}: {log_metrics}")

                # Save checkpoint
                if (
                    self.training_config.checkpoint.save_model_freq > 0
                    and step % self.worker_config.checkpoint_sync_interval == 0
                ):
                    self.save_checkpoint(train_state, step)

        # Final checkpoint
        if self.training_config.checkpoint.save_model_freq > 0:
            self.save_checkpoint(train_state, step)

        # Cleanup
        jax_distributed_barrier()
        self.logger.finish()
        jax_distributed_barrier()

        logger.info(f"Training completed after {step} steps")
