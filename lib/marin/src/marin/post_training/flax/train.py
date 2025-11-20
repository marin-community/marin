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

import dataclasses
from collections import deque
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tyro
from flax.training.train_state import TrainState
from jax.sharding import PartitionSpec as PS
from optax import softmax_cross_entropy_with_integer_labels
from scalax.sharding import MeshShardingHelper, TreePathShardingRule
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from ..environments.load_environments import load_environments_from_config
from ..environments.marin_env import MarinEnv
from ..rl_dataset import create_dataset_from_environment
from .inference import FlaxInferenceContext, build_sampler
from .llama3 import FlaxLLaMAForCausalLM
from .model_helpers import (
    build_generate_model,
    build_prefill_model,
    build_training_model,
    llama_config_from_model_config,
    load_tokenizer,
)
from .optimizer import load_adamw_optimizer
from .training_config import TrainingConfig
from .utils import (
    WandbLogger,
    checkpointer,
    delete_with_bucket,
    float_to_dtype,
    get_weight_decay_mask,
    global_norm,
    jax_distributed_barrier,
    jax_distributed_initalize,
    load_attention_kernel_config,
    load_checkpoint,
)


class Trainer:
    """RL trainer"""

    config: TrainingConfig
    mesh: MeshShardingHelper
    models: dict[str, FlaxLLaMAForCausalLM]
    tokenizer: AutoTokenizer
    train_environments: list[tuple[str, MarinEnv]]
    test_environments: list[tuple[str, MarinEnv]]
    logger: WandbLogger

    def __init__(
        self,
        config: TrainingConfig,
        mesh: MeshShardingHelper,
        models: dict[str, FlaxLLaMAForCausalLM],
        tokenizer: AutoTokenizer,
        train_environments: list[tuple[str, MarinEnv]],
        test_environments: list[tuple[str, MarinEnv]],
        logger: WandbLogger,
    ):
        self.config = config
        self.mesh = mesh
        self.models = models
        self.tokenizer = tokenizer
        self.train_environments = train_environments
        self.test_environments = test_environments
        self.logger = logger

        # Extract frequently used config values
        self.max_input_length = config.hyperparameters.max_input_length
        self.max_output_length = config.hyperparameters.max_output_length
        self.train_bsize = config.hyperparameters.train_bsize
        self.reference_logprobs_bsize = config.hyperparameters.reference_logprobs_bsize
        self.pad_token_id = config.hyperparameters.pad_token_id
        self.kl_coef = config.hyperparameters.kl_coef

        # Setup training components
        self._setup_optimizer()
        self._setup_models()
        self._setup_samplers()
        self._compile_functions()

    def _setup_optimizer(self):
        """Setup optimizer configuration."""
        optim_config = self.config.hyperparameters.optim_config
        weight_decay_mask = get_weight_decay_mask(optim_config.weight_decay_exclusions)
        self.grad_accum_steps = optim_config.grad_accum_steps
        optimizer, self.optimizer_info = load_adamw_optimizer(config=optim_config, weight_decay_mask=weight_decay_mask)


        self.optimizer = optimizer

    def _setup_models(self):
        """Setup model configurations and sharding."""
        self.train_model = self.models["train"]
        self.prefill_model = self.models["prefill"]
        self.generate_model = self.models["generate"]

        # Setup sharding rules
        config = self.train_model.config
        self.train_params_sharding_rules = TreePathShardingRule(
            *config.get_partition_rules(
                model_all_gather_axis=("fsdp", "sequence"),
            )
        )
        self.inference_params_sharding_rules = TreePathShardingRule(
            *config.get_partition_rules(
                model_all_gather_axis=None,
            )
        )
        self.train_intermediate_sharding_rules = config.get_intermediate_sharding_rules(
            data_axis=("replica", "fsdp"),
            sequence_axis="sequence",
        )
        self.inference_intermediate_sharding_rules = config.get_intermediate_sharding_rules(
            data_axis=("replica", "fsdp"),
            sequence_axis=None,
        )

    def _setup_samplers(self):
        """Setup sampling configurations."""
        generation_config = self.config.generation_config
        test_generation_config = self.config.test_generation_config

        sampler_kwargs = {
            "prefill_model": self.prefill_model,
            "generate_model": self.generate_model,
            "tokenizer": self.tokenizer,
            "bsize": self.config.hyperparameters.decode_bsize,
            "prefill_bsize": self.config.hyperparameters.prefill_bsize,
            "max_input_length": self.max_input_length,
            "params_sharding_rules": self.inference_params_sharding_rules,
            "intermediate_sharding_rules": self.inference_intermediate_sharding_rules,
            "replica_axis_name": ("replica", "fsdp"),
            "tp_axis_name": "tensor",
            "mesh": self.mesh,
            "pad_token_id": self.pad_token_id,
        }

        self.sampler = build_sampler(generation_config=generation_config, **sampler_kwargs)
        self.test_sampler = build_sampler(generation_config=test_generation_config, **sampler_kwargs)

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
            params = self.train_model.init_weights(
                rng, (self.train_bsize, self.max_input_length + self.max_output_length - 1)
            )
            return create_train_state_from_params(params)

        @partial(
            self.mesh.sjit,
            in_shardings=(self.train_params_sharding_rules,),
            out_shardings=self.inference_params_sharding_rules,
            args_sharding_constraint=(self.train_params_sharding_rules,),
        )
        def reshard_params(params):
            params = float_to_dtype(params, self.config.model.inference_param_dtype)
            return params

        @partial(
            self.mesh.sjit,
            in_shardings=(
                self.train_params_sharding_rules,
                PS(),
                PS(),
                PS(),
                PS(),
            ),
            out_shardings=PS(),
            args_sharding_constraint=(
                self.train_params_sharding_rules,
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
            logits = self.train_model(
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
            logprobs = MeshShardingHelper.with_sharding_constraint(logprobs, PS(("replica", "fsdp"), None))
            return logprobs

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
            def loss_fn(params, input_ids, attention_mask, position_ids, target_ids,
                       policy_logprobs, reference_logprobs, loss_weights, loss_masks, key):
                # Forward pass
                logits = self.train_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    params=params,
                    dropout_rng=key,
                    train=True,
                ).logits
                logits = logits.astype(jnp.float32)
                token_loss = softmax_cross_entropy_with_integer_labels(logits, target_ids)
                
                # Compute probability ratio: exp(target_logprobs - sampling_logprobs)
                target_logprobs = -token_loss
                prob_ratio = jnp.exp(target_logprobs - policy_logprobs)
                
                # Compute importance-weighted REINFORCE loss
                reinforce_loss = -jnp.mean(prob_ratio * loss_weights, where=loss_masks > 0.0)
                ref_log_ratio = reference_logprobs + token_loss
                kl_loss = jnp.exp(ref_log_ratio) - 1 - ref_log_ratio
                kl_loss = jnp.mean(kl_loss, where=loss_masks > 0.0)
                loss = reinforce_loss + self.kl_coef * kl_loss
                
                # Compute KL divergence metrics between sampling and training logprobs
                training_logprobs = target_logprobs
                action_mask = loss_masks > 0.0
                
                logprob_diff = policy_logprobs - training_logprobs
                kl_sample_train_v1 = jnp.mean(logprob_diff, where=action_mask)
                kl_sample_train_v2 = 0.5 * jnp.mean(logprob_diff**2, where=action_mask)
                entropy_sample = -jnp.mean(policy_logprobs, where=action_mask)
                
                return loss, {
                    "reinforce_loss": reinforce_loss,
                    "kl_loss": kl_loss,
                    "kl_sample_train_v1": kl_sample_train_v1,
                    "kl_sample_train_v2": kl_sample_train_v2,
                    "entropy": entropy_sample,
                }
            
            if self.grad_accum_steps > 1:
                # Use scan for gradient accumulation to save memory
                batch_size = batch["input_ids"].shape[0]
                microbatch_size = batch_size // self.grad_accum_steps
                
                # Split RNG for each microbatch
                rng_keys = jax.random.split(rng, self.grad_accum_steps)
                
                # Reshape batch data from (B, ...) to (AccumSteps, MicroB, ...)
                reshaped_batch = jax.tree.map(
                    lambda x: x.reshape((self.grad_accum_steps, microbatch_size) + x.shape[1:]),
                    batch
                )
                
                def scan_step(carry, inputs):
                    accumulated_grads, accumulated_loss, accumulated_aux = carry
                    micro_batch_data, key = inputs
                    
                    # Compute gradients for this microbatch
                    grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
                    (micro_loss, micro_aux), micro_grads = grad_fn(
                        train_state.params,
                        micro_batch_data["input_ids"],
                        micro_batch_data["attention_mask"],
                        micro_batch_data["position_ids"],
                        micro_batch_data["target_ids"],
                        micro_batch_data["policy_logprobs"],
                        micro_batch_data["reference_logprobs"],
                        micro_batch_data["loss_weights"],
                        micro_batch_data["loss_masks"],
                        key=key,
                    )
                    
                    # Accumulate gradients
                    new_grads = jax.tree.map(lambda a, b: a + b, accumulated_grads, micro_grads)
                    new_loss = accumulated_loss + micro_loss
                    new_aux = jax.tree.map(lambda a, b: a + b, accumulated_aux, micro_aux)
                    
                    return (new_grads, new_loss, new_aux), None
                
                # Initialize accumulators
                zero_grads = jax.tree.map(jnp.zeros_like, train_state.params)
                zero_aux = jax.tree.map(lambda x: jnp.zeros_like(x), loss_fn(
                    train_state.params,
                    batch["input_ids"][:microbatch_size],
                    batch["attention_mask"][:microbatch_size],
                    batch["position_ids"][:microbatch_size],
                    batch["target_ids"][:microbatch_size],
                    batch["policy_logprobs"][:microbatch_size],
                    batch["reference_logprobs"][:microbatch_size],
                    batch["loss_weights"][:microbatch_size],
                    batch["loss_masks"][:microbatch_size],
                    key=rng,
                )[1])
                
                # Run scan over microbatches
                (grads, loss, aux), _ = jax.lax.scan(
                    scan_step,
                    (zero_grads, 0.0, zero_aux),
                    (reshaped_batch, rng_keys)
                )
                
                # Average gradients and loss
                grads = jax.tree.map(lambda g: g / self.grad_accum_steps, grads)
                loss = loss / self.grad_accum_steps
                aux = jax.tree.map(lambda a: a / self.grad_accum_steps, aux)
            else:
                # No gradient accumulation - compute directly
                grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
                (loss, aux), grads = grad_fn(
                    train_state.params,
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["position_ids"],
                    batch["target_ids"],
                    batch["policy_logprobs"],
                    batch["reference_logprobs"],
                    batch["loss_weights"],
                    batch["loss_masks"],
                    key=rng,
                )
            
            train_state = train_state.apply_gradients(grads=grads)
            metrics = dict(
                loss=loss,
                reinforce_loss=aux["reinforce_loss"],
                kl_loss=aux["kl_loss"],
                learning_rate=self.optimizer_info["learning_rate_schedule"](train_state.step),
                gradient_norm=global_norm(grads),
                param_norm=global_norm(train_state.params),
            )
            metrics["optim/kl_sample_train_v1"] = aux["kl_sample_train_v1"]
            metrics["optim/kl_sample_train_v2"] = aux["kl_sample_train_v2"]
            metrics["optim/entropy"] = aux["entropy"]
            return train_state, metrics

        # Store compiled functions
        self.create_train_state_from_params = create_train_state_from_params
        self.init_fn = init_fn
        self.reshard_params = reshard_params
        self.get_logprobs = get_logprobs
        self.train_step = train_step

    def evaluate_data_from_environment(self, params, prng_key):
        """Evaluate model using environment."""
        inference_params = self.reshard_params(params)

        # Create evaluation inference context
        eval_ctx = FlaxInferenceContext(
            params=inference_params,
            sampler=self.test_sampler,
            prng_key=prng_key,
            tokenizer=self.tokenizer,
            get_logprobs_fn=self.get_logprobs,
            reference_logprobs_bsize=self.reference_logprobs_bsize,
        )

        eval_metrics = {}
        for env_name, environment in self.test_environments:
            # Get evaluation examples from environment
            eval_examples = environment.get_eval_examples(self.config.logging.num_eval_examples)
            # Generate samples for evaluation
            samples = eval_ctx.generate(
                [example["prompt"] for example in eval_examples],
                temperature=self.config.test_generation_config.temperature,
                n_generations=self.config.test_generation_config.n_generations,
            )
            del inference_params

            # Compute rewards using environment's reward computation
            _, metrics = environment._compute_rewards(eval_examples, samples, self.tokenizer)

            for k, v in metrics.items():
                eval_metrics[k.replace("train/", f"test/{env_name}/")] = v
        return eval_metrics

    def save_checkpoint(self, train_state, step):
        """Save model checkpoint."""
        if (self.config.logging.max_checkpoints is not None) and (
            len(self.checkpoint_queue) >= self.config.logging.max_checkpoints
        ):
            old_step = self.checkpoint_queue.popleft()
            if self.logger.can_save():
                old_path = str(Path(self.config.output_dir) / "checkpoints" / f"step_{old_step}")
                delete_with_bucket(old_path, recursive=True)

        if self.logger.can_save():
            print(f"saving checkpoint at step {step} ...")

            metadata = dict(
                step=step,
                args_dict=self.config,
            )

            if self.config.checkpoint.save_optimizer_state:
                params = train_state
            else:
                params = train_state.params

            checkpointer(
                path=str(Path(self.config.output_dir) / "checkpoints" / f"step_{step}"),
                params=params,
                config=self.train_model.config.to_dict(),
                gather_fns=self.train_state_gather_fns,
                metadata=metadata,
                active=self.logger.can_save(),
                save_float_dtype=self.config.checkpoint.save_float_dtype,
            )

            self.checkpoint_queue.append(step)
            print("saved.")

    def train(self):
        """Main training loop."""
        # Initialize training state shapes and sharding functions
        train_state_shape = jax.eval_shape(lambda: self.init_fn(jax.random.PRNGKey(0)))
        inference_params_shape = jax.eval_shape(
            lambda: self.prefill_model.init_weights(
                jax.random.PRNGKey(0),
                (
                    self.config.hyperparameters.decode_bsize,
                    self.max_input_length + self.max_output_length - 1,
                ),
            )
        )
        self.train_state_shard_fns, self.train_state_gather_fns = self.mesh.make_shard_and_gather_fns(
            train_state_shape, self.train_params_sharding_rules
        )
        # Validate inference params sharding (side effect: validates sharding rules)
        self.mesh.make_shard_and_gather_fns(inference_params_shape, self.inference_params_sharding_rules)

        # Initialize training state
        latest_checkpoint_step = -1
        if self.logger.can_save():
            checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
            if checkpoint_dir.exists():
                checkpoint_steps = [
                    int(d.name.split("_")[1]) for d in checkpoint_dir.iterdir() if d.name.startswith("step_")
                ]
                if checkpoint_steps:
                    latest_checkpoint_step = max(checkpoint_steps)

        if latest_checkpoint_step > 0:
            print(f"Resuming training from checkpoint at step {latest_checkpoint_step}...")
            checkpoint_path = (
                Path(self.config.output_dir) / "checkpoints" / f"step_{latest_checkpoint_step}" / "params.msgpack"
            )
            train_state = self.create_train_state_from_params(
                load_checkpoint(
                    checkpoint_path,
                    shard_fns=self.train_state_shard_fns.params,
                    remove_dict_prefix=self.config.model.model_paths.remove_dict_prefix,
                    convert_to_dtypes=jax.tree.map(
                        lambda x: self.config.model.training_param_dtype, train_state_shape.params
                    ),
                )
            )
        elif self.config.model.model_paths.params:
            train_state = self.create_train_state_from_params(
                load_checkpoint(
                    self.config.model.model_paths.params,
                    shard_fns=self.train_state_shard_fns.params,
                    remove_dict_prefix=self.config.model.model_paths.remove_dict_prefix,
                    convert_to_dtypes=jax.tree.map(
                        lambda x: self.config.model.training_param_dtype, train_state_shape.params
                    ),
                )
            )
        elif self.config.model.model_paths.train_state:
            train_state: TrainState = load_checkpoint(
                self.config.model.model_paths.params,
                shard_fns=self.train_state_shard_fns,
                remove_dict_prefix=self.config.model.model_paths.remove_dict_prefix,
                convert_to_dtypes=jax.tree.map(
                    lambda x: self.config.model.training_param_dtype, train_state_shape
                ),
            )
        else:
            print("WARNING: no params path provided, initializing with random weights...")
            train_state: TrainState = self.init_fn(jax.random.PRNGKey(0))

        self.reference_params = float_to_dtype(train_state.params, self.config.model.inference_param_dtype)
        self.checkpoint_queue = deque()

        if self.config.logging.save_initial_checkpoint and latest_checkpoint_step < 0:
            self.save_checkpoint(train_state, 0)

        # Training loop
        rng = jax.random.PRNGKey(0)

        # Evaluate before first training iteration if requested
        # if self.config.logging.log_initial_step and latest_checkpoint_step < 0:
        #     if self.config.logging.num_eval_examples > 0:
        #         print("Evaluating before first training step...")
        #         rng, subrng = jax.random.split(rng)
        #         eval_metrics = self.evaluate_data_from_environment(train_state.params, subrng)
        #         log_metrics = {"step": -1}
        #         log_metrics.update(eval_metrics)
        #         log_metrics = jax.device_get(log_metrics)
        #         self.logger.log(log_metrics)
        #         print(log_metrics)

        for step in tqdm(
            range(max(0, latest_checkpoint_step), self.config.hyperparameters.num_train_steps),
            total=self.config.hyperparameters.num_train_steps,
        ):
            rng, subrng = jax.random.split(rng)

            idx = jax.random.randint(subrng, shape=(), minval=0, maxval=len(self.train_environments))
            _env_name, environment = self.train_environments[idx]

            rng, subrng = jax.random.split(subrng)
            inference_params = self.reshard_params(train_state.params)

            # Create inference contexts
            policy_ctx = FlaxInferenceContext(
                params=inference_params,
                sampler=self.sampler,
                prng_key=subrng,
                tokenizer=self.tokenizer,
                get_logprobs_fn=self.get_logprobs,
                reference_logprobs_bsize=self.reference_logprobs_bsize,
            )

            reference_ctx = FlaxInferenceContext(
                params=self.reference_params,
                sampler=self.sampler,
                prng_key=subrng,
                tokenizer=self.tokenizer,
                get_logprobs_fn=self.get_logprobs,
                reference_logprobs_bsize=self.reference_logprobs_bsize,
            )

            rl_dataset, dataset_metrics = create_dataset_from_environment(
                environment=environment,
                policy_ctx=policy_ctx,
                reference_ctx=reference_ctx,
                n_examples=self.config.hyperparameters.n_prompts_per_step,
                prng_key=subrng,
                n_generations=self.config.generation_config.n_generations,
                max_input_length=self.max_input_length,
                max_output_length=self.max_output_length,
                pad_token_id=self.pad_token_id,
                mode="train",
                temperature=self.config.generation_config.temperature,
            )
            del inference_params

            num_batches = 0
            for batch in tqdm(rl_dataset.iterate_batches(batch_size=self.train_bsize, shuffle=True, loop=False)):
                train_state, metrics = self.train_step(train_state, subrng, batch)
                num_batches += 1
                if num_batches > 1:
                    raise ValueError("Only one batch should be generated per environment")

            if self.config.logging.log_freq > 0 and (
                (step + 1) % self.config.logging.log_freq == 0 or (self.config.logging.log_initial_step and step == 0)
            ):
                if self.config.logging.num_eval_examples > 0:
                    rng, subrng = jax.random.split(rng)
                    metrics.update(self.evaluate_data_from_environment(train_state.params, subrng))

                log_metrics = {"step": step + 1}
                log_metrics.update(metrics)
                log_metrics.update(dataset_metrics)
                log_metrics = jax.device_get(log_metrics)
                self.logger.log(log_metrics)
                print(log_metrics)

            if self.config.checkpoint.save_model_freq > 0 and (step + 1) % self.config.checkpoint.save_model_freq == 0:
                self.save_checkpoint(train_state, step + 1)

        if self.config.checkpoint.save_model_freq > 0 and (
            self.config.hyperparameters.num_train_steps not in self.checkpoint_queue
        ):
            self.save_checkpoint(train_state, self.config.hyperparameters.num_train_steps)


def main(config: TrainingConfig):
    """Main training script with environment."""
    logging_config = config.logging

    # Synchronous trainer requires same sharding for training and inference
    assert (
        config.distributed.train_sharding == config.distributed.inference_sharding
    ), "Synchronous trainer requires train_sharding == inference_sharding"

    # Initialize JAX distributed
    jax_distributed_initalize(**config.distributed.jax_distributed_initialize_config)
    jax_distributed_barrier()

    # Validate attention kernel configurations (will raise ValueError if malformed)
    load_attention_kernel_config(config.model.prefill_attention_kernel_config, ["splash", "default"])
    load_attention_kernel_config(config.model.generate_attention_kernel_config, ["paged", "default"])
    load_attention_kernel_config(config.model.train_attention_kernel_config, ["splash", "default", "ring", "ring_jax"])

    mesh = MeshShardingHelper(
        config.distributed.train_sharding,
        ["replica", "fsdp", "sequence", "tensor"],
        mesh_axis_splitting=config.distributed.physical_axis_splitting,
    )

    with mesh.get_context():
        model_paths = config.model.model_paths
        llama_config = llama_config_from_model_config(model_paths, config.model.model_config_override)
        training_model = build_training_model(llama_config, config)
        inference_model = build_generate_model(llama_config, config)
        prefill_model = build_prefill_model(llama_config, config)

        tokenizer = load_tokenizer(model_paths, config.model.tokenizer_override)
        # TODO (Kevin): Update end_of_message_token if using non-Llama models
        end_of_message_token = tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0]

        # Initialize environment with tokenization parameters
        train_environments = load_environments_from_config(
            Path(__file__).resolve().parent / config.environment.train_environments_path, tokenizer, end_of_message_token
        )
        test_environments = load_environments_from_config(
            Path(__file__).resolve().parent / config.environment.test_environments_path, tokenizer, end_of_message_token
        )

        # Initialize logger
        if logging_config.enable is None:
            logging_config.enable = jax.process_index() == 0
        if logging_config.config_to_log is None:
            logging_config.config_to_log = {}
        logging_config.config_to_log.update(dataclasses.asdict(config))
        logger = WandbLogger(
            config.logging.wandb_project,
            output_dir=config.output_dir,
            online=logging_config.online,
            prefix=logging_config.prefix,
            prefix_to_id=logging_config.prefix_to_id,
            experiment_id=logging_config.experiment_id,
            enable=logging_config.enable,
            config_to_log=logging_config.config_to_log,
        )

        # Initialize and run trainer
        trainer = Trainer(
            config,
            mesh,
            {
                "train": training_model,
                "prefill": prefill_model,
                "generate": inference_model,
            },
            tokenizer,
            train_environments,
            test_environments,
            logger,
        )
        trainer.train()

        jax_distributed_barrier()
        logger.finish()
        jax_distributed_barrier()


if __name__ == "__main__":
    tyro.cli(main)
