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
Inference worker for RL/post-training rollout generation.

This worker loads model checkpoints, generates rollouts from a single environment,
and writes the rollout data to files for training workers to consume.
"""

import logging
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as PS
from optax import softmax_cross_entropy_with_integer_labels
from scalax.sharding import MeshShardingHelper, TreePathShardingRule

from .inference import build_sampler
from .load_environments import load_environment_from_spec
from .model_helpers import (
    build_generate_model,
    build_prefill_model,
    llama_config_from_model_config,
    load_tokenizer,
)
from .rl_dataset import create_dataset_from_environment
from .rollout_storage import FileRolloutWriter, RolloutBatch, RolloutWriter
from .training_config import InferenceWorkerConfig, TrainingConfig
from .utils import (
    float_to_dtype,
    get_float_dtype_by_name,
    jax_distributed_barrier,
    jax_distributed_initalize,
    load_checkpoint,
)

logger = logging.getLogger(__name__)


class InferenceWorker:
    """Inference worker that generates rollouts from a single environment."""

    _running: bool = True
    training_config: TrainingConfig
    inference_config: InferenceWorkerConfig
    rollout_writer: RolloutWriter

    def __init__(
        self,
        training_config: TrainingConfig,
        inference_config: InferenceWorkerConfig,
        rollout_writer: RolloutWriter | None = None,
    ):
        """Initialize inference worker.

        Args:
            training_config: Training configuration (for model/generation settings).
            inference_config: Inference worker specific configuration.
            rollout_writer: Optional rollout writer. If None, creates FileRolloutWriter.
        """
        self.training_config = training_config
        self.inference_config = inference_config

        # Initialize JAX distributed
        jax_distributed_initalize(**training_config.distributed.jax_distributed_initalize_config)
        jax_distributed_barrier()

        self.mesh = MeshShardingHelper(
            self.training_config.distributed.sharding,
            ["replica", "fsdp", "sequence", "tensor"],
            mesh_axis_splitting=self.training_config.distributed.physical_axis_splitting,
        )

        # Initialize components within mesh context
        with self.mesh.get_context():
            self._setup_components()

        # Initialize storage
        self.rollout_writer = rollout_writer or FileRolloutWriter(inference_config.rollout_output_path)

        # Track state
        self.current_step = 0
        self.latest_checkpoint_path = None
        self.current_params = None

    def stop(self):
        """Stop the inference worker loop."""
        self._running = False

    def _setup_components(self):
        """Setup models, tokenizer, and environment."""
        model_config = self.training_config.model

        # Setup models
        llama_config = llama_config_from_model_config(model_config.model_paths, model_config.model_config_override)
        self.prefill_model = build_prefill_model(llama_config, self.training_config)
        self.generate_model = build_generate_model(llama_config, self.training_config)

        # Setup tokenizer
        self.tokenizer = load_tokenizer(model_config.model_paths, model_config.tokenizer_override)

        # Load environment
        self.environment_name = self.inference_config.environment_spec
        self.environment = load_environment_from_spec(self.inference_config.environment_spec, self.tokenizer)

        # Extract frequently used config values
        self.max_input_length = self.training_config.hyperparameters.max_input_length
        self.max_output_length = self.training_config.hyperparameters.max_output_length
        self.reference_logprobs_bsize = self.training_config.hyperparameters.reference_logprobs_bsize
        self.pad_token_id = self.training_config.hyperparameters.pad_token_id

        self._setup_samplers()
        self._compile_functions()

    def _setup_samplers(self):
        """Setup sampling configurations."""
        generation_config = self.training_config.generation_config

        config = self.prefill_model.config
        self.inference_params_sharding_rules = TreePathShardingRule(
            *config.get_partition_rules(
                model_all_gather_axis=("fsdp", "sequence"),
            )
        )
        self.inference_intermediate_sharding_rules = config.get_intermediate_sharding_rules(
            data_axis=("replica", "fsdp"),
            sequence_axis=None,
        )

        sampler_kwargs = {
            "prefill_model": self.prefill_model,
            "generate_model": self.generate_model,
            "tokenizer": self.tokenizer,
            "bsize": self.training_config.hyperparameters.decode_bsize,
            "prefill_bsize": self.training_config.hyperparameters.prefill_bsize,
            "max_input_length": self.max_input_length,
            "params_sharding_rules": self.inference_params_sharding_rules,
            "intermediate_sharding_rules": self.inference_intermediate_sharding_rules,
            "replica_axis_name": ("replica", "fsdp"),
            "tp_axis_name": "tensor",
            "mesh": self.mesh,
            "pad_token_id": self.pad_token_id,
        }

        self.sampler = build_sampler(generation_config=generation_config, **sampler_kwargs)

    def _compile_functions(self):
        """Compile JAX functions for inference."""

        @partial(
            self.mesh.sjit,
            in_shardings=(PS(),),
            out_shardings=self.inference_params_sharding_rules,
            annotation_shardings=self.inference_intermediate_sharding_rules,
        )
        def init_params(rng):
            # Initialize with same pattern as Trainer - use training model for init
            # then convert to inference format
            params = self.prefill_model.init_weights(
                rng,
                (
                    self.training_config.hyperparameters.decode_bsize,
                    self.max_input_length + self.max_output_length - 1,
                ),
            )
            # Convert to inference dtype
            params = float_to_dtype(params, self.training_config.model.inference_param_dtype)
            return params

        @partial(
            self.mesh.sjit,
            in_shardings=(
                self.inference_params_sharding_rules,
                PS(),
                PS(),
                PS(),
                PS(),
            ),
            out_shardings=PS(),
            args_sharding_constraint=(
                self.inference_params_sharding_rules,
                PS(("replica", "fsdp")),
                PS(("replica", "fsdp")),
                PS(("replica", "fsdp")),
                PS(("replica", "fsdp")),
            ),
        )
        def get_logprobs(
            params,
            input_tokens,
            input_attention_mask,
            target_tokens,
            target_attention_mask,
        ):
            full_tokens = jnp.concatenate([input_tokens, target_tokens], axis=1)
            full_attention_mask = jnp.concatenate([input_attention_mask, target_attention_mask], axis=1)
            full_position_ids = jnp.maximum(jnp.cumsum(full_attention_mask, axis=1) - 1, 0)

            logits = self.prefill_model(
                full_tokens[:, :-1],
                full_attention_mask[:, :-1],
                full_position_ids[:, :-1],
                params=params,
                train=False,
            ).logits

            logits = logits[:, input_tokens.shape[1] - 1 :]
            logprobs = -softmax_cross_entropy_with_integer_labels(
                logits.astype(jnp.float32), target_tokens.astype(jnp.int32)
            )
            return logprobs

        self.init_params = init_params
        self.get_logprobs = get_logprobs

    def _find_latest_checkpoint(self) -> str | None:
        """Find the latest checkpoint in the source directory."""
        source_path = Path(self.inference_config.checkpoint_source_path)

        if not source_path.exists():
            return None

        # Look for checkpoint directories
        checkpoint_dirs = []
        for item in source_path.iterdir():
            if item.is_dir() and item.name.startswith("step_"):
                try:
                    step_num = int(item.name.split("_")[1])
                    checkpoint_dirs.append((step_num, item))
                except (ValueError, IndexError):
                    continue

        if not checkpoint_dirs:
            return None

        # Return path to the latest checkpoint's params.msgpack
        latest_step, latest_dir = max(checkpoint_dirs)
        params_path = latest_dir / "params.msgpack"

        if params_path.exists():
            return str(params_path)

        return None

    def _load_checkpoint(self, checkpoint_path: str):
        """Load model parameters from checkpoint."""
        with self.mesh.get_context():
            logger.info(f"Loading checkpoint from {checkpoint_path}")

            # Get parameter shapes for sharding
            params_shape = jax.eval_shape(
                lambda: self.prefill_model.init_weights(
                    jax.random.PRNGKey(0),
                    (
                        self.training_config.hyperparameters.decode_bsize,
                        self.max_input_length + self.max_output_length - 1,
                    ),
                )
            )

            # Create sharding functions
            shard_fns, _ = self.mesh.make_shard_and_gather_fns(params_shape, self.inference_params_sharding_rules)

            # Load and convert parameters
            params = load_checkpoint(
                checkpoint_path,
                shard_fns=shard_fns,
                remove_dict_prefix=self.training_config.model.model_paths.remove_dict_prefix,
                convert_to_dtypes=jax.tree_util.tree_map(
                    lambda x: get_float_dtype_by_name(self.training_config.model.inference_param_dtype),
                    params_shape,
                ),
            )

            # Convert to inference dtype
            params = float_to_dtype(params, self.training_config.model.inference_param_dtype)

            self.current_params = params
            self.latest_checkpoint_path = checkpoint_path

            logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
            jax_distributed_barrier()

    def _check_for_new_checkpoint(self):
        """Check if a new checkpoint is available and load it."""
        latest_checkpoint = self._find_latest_checkpoint()

        if latest_checkpoint is not None and latest_checkpoint != self.latest_checkpoint_path:
            self._load_checkpoint(latest_checkpoint)

    def _load_from_config(self):
        """Initialize the model from a training checkpoint, initial checkpoint, or randomly."""
        jax_distributed_barrier()
        latest_checkpoint = self._find_latest_checkpoint()

        if latest_checkpoint != self.latest_checkpoint_path:
            self._load_checkpoint(latest_checkpoint)

        model_paths = self.training_config.model.model_paths
        if model_paths.params:
            logger.info(f"Found checkpoint at {model_paths.params}")
            self._load_checkpoint(model_paths.params)
        elif model_paths.train_state:
            logger.info(f"Found checkpoint at {model_paths.train_state}")
            self._load_checkpoint(model_paths.train_state)
        else:
            logger.warning("No checkpoints found, initializing with random weights...")
            self.current_params = self.init_params(jax.random.PRNGKey(0))
            self.latest_checkpoint_path = "random_init"

        logger.info("Inference model initialized.")
        jax_distributed_barrier()

    def _generate_rollout_batch(self) -> tuple[dict, dict]:
        """Generate a single rollout batch from the environment."""
        if self.current_params is None:
            raise RuntimeError("No model parameters loaded")

        rng = jax.random.PRNGKey(int(time.time() * 1000) % (2**32))

        # Create RL dataset from environment
        rl_dataset, dataset_metrics = create_dataset_from_environment(
            environment=self.environment,
            sampler=self.sampler,
            params=self.current_params,
            reference_params=self.current_params,  # Use same params as reference for now
            get_logprobs_fn=self.get_logprobs,
            n_examples=self.training_config.hyperparameters.n_prompts_per_step,
            n_generations=self.training_config.generation_config.n_generations,
            prng_key=rng,
            reference_logprobs_bsize=self.reference_logprobs_bsize,
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
            pad_token_id=self.pad_token_id,
            tokenizer=self.tokenizer,
            mode="train",
        )

        for batch in rl_dataset.iterate_batches(
            batch_size=self.inference_config.rollout_batch_size, shuffle=True, loop=False
        ):
            return batch, dataset_metrics

    def run(self):
        """Main inference worker loop."""
        logger.info("Starting inference worker...")

        # Write initial metadata
        self.rollout_writer.write_metadata(
            {
                "environment_spec": self.inference_config.environment_spec,
                "environment_name": self.environment_name,
                "inference_config": self.inference_config.__dict__,
                "start_time": time.time(),
                "rollouts_generated": 0,
            }
        )

        rollouts_generated = 0
        last_checkpoint_check = 0

        self._load_from_config()

        while self._running:
            jax_distributed_barrier()
            # Check for new checkpoints periodically
            current_time = time.time()
            if current_time - last_checkpoint_check >= self.inference_config.checkpoint_poll_interval:
                logger.info("Checking for new checkpoints.")
                self._check_for_new_checkpoint()
                jax_distributed_barrier()
                last_checkpoint_check = current_time

            if (
                self.inference_config.max_rollouts is not None
                and rollouts_generated >= self.inference_config.max_rollouts
            ):
                logger.info(f"Reached max rollouts ({self.inference_config.max_rollouts}), stopping")
                break

            logger.info(f"Generating rollout batch {rollouts_generated}")
            jax_distributed_barrier()
            batch_data, metrics = self._generate_rollout_batch()

            # Create RolloutBatch
            rollout_batch = RolloutBatch(
                input_ids=batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                position_ids=batch_data["position_ids"],
                target_ids=batch_data["target_ids"],
                loss_weights=batch_data["loss_weights"],
                loss_masks=batch_data["loss_masks"],
                reference_logprobs=batch_data["reference_logprobs"],
                metadata=metrics,
            )

            # Write rollout batch
            self.rollout_writer.write_batch(rollout_batch)

            rollouts_generated += 1
            logger.info(f"Generated rollout {rollouts_generated}")

            # Update metadata
            self.rollout_writer.write_metadata(
                {
                    "environment_spec": self.inference_config.environment_spec,
                    "environment_name": self.environment_name,
                    "inference_config": self.inference_config.__dict__,
                    "start_time": time.time(),
                    "rollouts_generated": rollouts_generated,
                    "latest_checkpoint": self.latest_checkpoint_path,
                }
            )

        logger.info(f"Inference worker completed after generating {rollouts_generated} rollouts")
        jax_distributed_barrier()
