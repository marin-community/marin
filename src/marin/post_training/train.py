import copy
import json
import os
import tempfile
from collections import deque
from functools import partial
from typing import Any

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
from pathlib import Path

from .environments.marin_env import MarinEnv
from .inference import GenerationConfig, batch_inference, build_sampler
from .llama3 import (
    LLAMA_STANDARD_CONFIGS,
    FlaxLLaMAForCausalLM,
    LLaMAConfig,
)
from .load_environments import load_environments_from_config
from .optimizer import load_adamw_optimizer
from .rl_dataset import create_dataset_from_environment
from .utils import (
    WandbLogger,
    checkpointer,
    delete_with_bucket,
    float_to_dtype,
    get_float_dtype_by_name,
    get_weight_decay_mask,
    global_norm,
    jax_distributed_barrier,
    jax_distributed_initalize,
    load_attention_kernel_config,
    load_checkpoint,
    open_with_bucket,
)


class Trainer:
    """RL trainer"""

    def __init__(
        self,
        config: dict[str, Any],
        mesh: MeshShardingHelper,
        models: dict[str, FlaxLLaMAForCausalLM],
        tokenizer: AutoTokenizer,
        environment: MarinEnv,
        logger: WandbLogger,
    ):
        self.config = config
        self.mesh = mesh
        self.models = models
        self.tokenizer = tokenizer
        self.environment = environment
        self.logger = logger

        # Extract frequently used config values
        self.max_input_length = config["max_input_length"]
        self.max_output_length = config["max_output_length"]
        self.train_bsize = config["train_bsize"]
        self.reference_logprobs_bsize = config["reference_logprobs_bsize"]
        self.pad_token_id = config["pad_token_id"]
        self.kl_coef = config["kl_coef"]

        # Setup training components
        self._setup_optimizer()
        self._setup_models()
        self._setup_samplers()
        self._compile_functions()

    def _setup_optimizer(self):
        """Setup optimizer configuration."""
        optim_config = self.config["optim_config"]
        if optim_config.startswith("adamw:"):
            optim_config = json.loads(optim_config[len("adamw:") :])
            optim_config["weight_decay_mask"] = get_weight_decay_mask(
                optim_config.pop("weight_decay_exclusions", tuple())
            )
            self.grad_accum_steps = optim_config.pop("grad_accum_steps", 1)
            optimizer, self.optimizer_info = load_adamw_optimizer(**optim_config)
        else:
            raise ValueError(f"Unknown optimizer config: {optim_config}")

        if self.grad_accum_steps > 1:
            optimizer = optax.MultiSteps(optimizer, self.grad_accum_steps)

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
        generation_config = GenerationConfig(**self.config["generation_config"])
        test_generation_config = GenerationConfig(**self.config["test_generation_config"])

        sampler_kwargs = {
            "prefill_model": self.prefill_model,
            "generate_model": self.generate_model,
            "tokenizer": self.tokenizer,
            "bsize": self.config["decode_bsize"],
            "prefill_bsize": self.config["prefill_bsize"],
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
            params = float_to_dtype(params, self.config["inference_param_dtype"])
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
            args_sharding_constraint=(self.train_params_sharding_rules, PS(), PS(("replica", "fsdp"))),
            donate_argnums=(0,),
            annotation_shardings=self.train_intermediate_sharding_rules,
        )
        def train_step(train_state, rng, batch):
            def loss(params):
                logits = self.train_model(
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
        self.reshard_params = reshard_params
        self.get_logprobs = get_logprobs
        self.train_step = train_step

    def evaluate_data_from_environment(self, params, prng_key):
        """Evaluate model using environment."""
        inference_params = self.reshard_params(params)

        # Get evaluation examples from environment
        eval_examples = self.environment.get_eval_examples(self.config["num_eval_examples"])

        # Generate samples for evaluation
        samples = batch_inference(
            self.test_sampler,
            inference_params,
            [example["prompt"] for example in eval_examples],
            prng_key,
            self.config["test_generation_config"]["n_generations"],
            verbose=True,
        )
        del inference_params

        # Compute rewards using environment's reward computation
        _, metrics = self.environment._compute_rewards(eval_examples, samples)

        # Rename metrics for evaluation
        eval_metrics = {}
        for k, v in metrics.items():
            eval_metrics[k.replace("train_", "test_")] = v
        return eval_metrics

    def save_checkpoint(self, train_state, step):
        """Save model checkpoint."""
        if (self.config["max_checkpoints"] is not None) and (
            len(self.checkpoint_queue) >= self.config["max_checkpoints"]
        ):
            old_step = self.checkpoint_queue.popleft()
            if self.logger.can_save():
                old_path = os.path.join(self.logger.output_dir, "checkpoints", f"step_{old_step}")
                delete_with_bucket(old_path, recursive=True)

        if self.logger.can_save():
            print(f"saving checkpoint at step {step} ...")

            metadata = dict(
                step=step,
                args_dict=self.config,
            )

            checkpointer(
                path=os.path.join(self.logger.output_dir, "checkpoints", f"step_{step}"),
                train_state=train_state,
                config=self.train_model.config.to_dict(),
                gather_fns=self.train_state_gather_fns,
                metadata=metadata,
                active=self.logger.can_save(),
                **self.config.get("checkpointer_config", {}),
            )

            self.checkpoint_queue.append(step)
            print("saved.")

    def train(self):
        """Main training loop."""
        # Initialize training state shapes and sharding functions
        train_state_shape = jax.eval_shape(lambda: self.init_fn(jax.random.PRNGKey(0)))
        inference_params_shape = jax.eval_shape(
            lambda: self.prefill_model.init_weights(
                jax.random.PRNGKey(0), (self.config["decode_bsize"], self.max_input_length + self.max_output_length - 1)
            )
        )
        self.train_state_shard_fns, self.train_state_gather_fns = self.mesh.make_shard_and_gather_fns(
            train_state_shape, self.train_params_sharding_rules
        )
        inference_param_shard_fns, inference_param_gather_fns = self.mesh.make_shard_and_gather_fns(
            inference_params_shape, self.inference_params_sharding_rules
        )

        # Initialize training state
        latest_checkpoint_step = -1
        if self.logger.can_save():
            checkpoint_dir = os.path.join(self.logger.output_dir, "checkpoints")
            if os.path.exists(checkpoint_dir):
                checkpoint_steps = [
                    int(d.split("_")[1]) for d in os.listdir(checkpoint_dir) if d.startswith("step_")
                ]
                if checkpoint_steps:
                    latest_checkpoint_step = max(checkpoint_steps)

        if latest_checkpoint_step > 0:
            print(f"Resuming training from checkpoint at step {latest_checkpoint_step}...")
            checkpoint_path = os.path.join(
                self.logger.output_dir, "checkpoints", f"step_{latest_checkpoint_step}", "params.msgpack"
            )
            train_state = self.create_train_state_from_params(
                load_checkpoint(
                    checkpoint_path,
                    shard_fns=self.train_state_shard_fns.params,
                    remove_dict_prefix=self.config["model_paths"]["remove_dict_prefix"],
                    convert_to_dtypes=jax.tree_util.tree_map(
                        lambda x: self.config["training_param_dtype"], train_state_shape.params
                    ),
                )
            )
        elif "params" in self.config["model_paths"]:
            train_state = self.create_train_state_from_params(
                load_checkpoint(
                    self.config["model_paths"]["params"],
                    shard_fns=self.train_state_shard_fns.params,
                    remove_dict_prefix=self.config["model_paths"]["remove_dict_prefix"],
                    convert_to_dtypes=jax.tree_util.tree_map(
                        lambda x: self.config["training_param_dtype"], train_state_shape.params
                    ),
                )
            )
        elif "train_state" in self.config["model_paths"]:
            train_state = load_checkpoint(
                self.config["model_paths"]["train_state"],
                shard_fns=self.train_state_shard_fns,
                remove_dict_prefix=self.config["model_paths"]["remove_dict_prefix"],
                convert_to_dtypes=jax.tree_util.tree_map(
                    lambda x: self.config["training_param_dtype"], train_state_shape
                ),
            )
        else:
            print("WARNING: no params path provided, initializing with random weights...")
            train_state = self.init_fn(jax.random.PRNGKey(0))

        self.reference_params = float_to_dtype(train_state.params, self.config["inference_param_dtype"])
        self.checkpoint_queue = deque()

        if self.config.get("save_initial_checkpoint", False):
            self.save_checkpoint(train_state, 0)

        # Training loop
        rng = jax.random.PRNGKey(0)

        for step in tqdm(range(max(0, latest_checkpoint_step),
                               self.config["num_train_steps"]),
                         total=self.config["num_train_steps"]):
            rng, subrng = jax.random.split(rng)

            inference_params = self.reshard_params(train_state.params)
            rl_dataset, dataset_metrics = create_dataset_from_environment(
                environment=self.environment,
                sampler=self.sampler,
                params=inference_params,
                reference_params=self.reference_params,
                get_logprobs_fn=self.get_logprobs,
                n_examples=self.config["n_prompts_per_step"],
                prng_key=subrng,
                reference_logprobs_bsize=self.reference_logprobs_bsize,
                max_input_length=self.max_input_length,
                max_output_length=self.max_output_length,
                pad_token_id=self.pad_token_id,
                tokenizer=self.tokenizer,
                generation_config=self.config["generation_config"],
                mode="train",
            )
            del inference_params

            for batch in tqdm(rl_dataset.iterate_batches(batch_size=self.train_bsize, shuffle=True, loop=False)):
                train_state, metrics = self.train_step(train_state, subrng, batch)

            if self.config["log_freq"] > 0 and (
                (step + 1) % self.config["log_freq"] == 0 or (self.config.get("log_initial_step", True) and step == 0)
            ):
                if self.config["num_eval_examples"] > 0:
                    rng, subrng = jax.random.split(rng)
                    metrics.update(self.evaluate_data_from_environment(train_state.params, subrng))

                log_metrics = {"step": step + 1}
                log_metrics.update(metrics)
                log_metrics.update(dataset_metrics)
                log_metrics = jax.device_get(log_metrics)
                self.logger.log(log_metrics)
                print(log_metrics)

            if self.config["save_model_freq"] > 0 and (step + 1) % self.config["save_model_freq"] == 0:
                self.save_checkpoint(train_state, step + 1)

        if self.config["save_model_freq"] > 0 and (self.config["num_train_steps"] not in self.checkpoint_queue):
            self.save_checkpoint(train_state, self.config["num_train_steps"])


def main(
    load_model: str,
    output_dir: str | None,
    sharding: str,
    num_train_steps: int,
    max_input_length: int,
    max_output_length: int,
    train_bsize: int,
    decode_bsize: int,
    prefill_bsize: int,
    reference_logprobs_bsize: int,
    n_prompts_per_step: int,
    log_freq: int,
    num_eval_examples: int,
    save_model_freq: int,
    wandb_project: str,
    environments_path: str = "environments.json",
    inference_param_dtype: str = "bf16",
    inference_activation_dtype: str = "bf16",
    training_param_dtype: str = "fp32",
    training_activation_dtype: str = "fp32",
    optim_config: str = "adamw:{}",
    logger_config: str = "{}",
    checkpointer_config: str = "{}",
    generation_config: str = "{}",
    test_generation_config: str = "{}",
    model_config_override: str = "{}",
    tokenizer_override: str = "{}",
    train_attention_kernel_config: str = "splash:{}",
    prefill_attention_kernel_config: str = "splash:{}",
    generate_attention_kernel_config: str = "paged:{}",
    jax_distributed_initalize_config: str = "{}",
    save_initial_checkpoint: bool = False,
    log_initial_step: bool = True,
    max_checkpoints: int | None = None,
    physical_axis_splitting: bool = False,
    pad_token_id: int = 128002,
    kl_coef: float = 0.0,
):
    """Main training script with environment."""
    # Parse configurations
    args_dict = dict(locals())
    print(args_dict)
    sharding: list[int] = list(map(lambda x: int(x.strip()), sharding.split(",")))

    # Parse dtype configurations
    inference_param_dtype = get_float_dtype_by_name(inference_param_dtype)
    inference_activation_dtype = get_float_dtype_by_name(inference_activation_dtype)
    training_param_dtype = get_float_dtype_by_name(training_param_dtype)
    training_activation_dtype = get_float_dtype_by_name(training_activation_dtype)

    # Parse JSON configurations
    logger_config: dict[str, Any] = json.loads(logger_config)
    checkpointer_config: dict[str, Any] = json.loads(checkpointer_config)
    generation_config: dict[str, Any] = json.loads(generation_config)
    test_generation_config: dict[str, Any] = json.loads(test_generation_config)
    model_config_override: dict[str, Any] = json.loads(model_config_override)
    tokenizer_override: dict[str, Any] = json.loads(tokenizer_override)
    jax_distributed_initalize_config: dict[str, Any] = json.loads(jax_distributed_initalize_config)

    # Initialize JAX distributed
    jax_distributed_initalize(**jax_distributed_initalize_config)
    jax_distributed_barrier()

    # Load attention kernel configurations
    prefill_attention_kernel, prefill_attention_kernel_config = load_attention_kernel_config(
        prefill_attention_kernel_config, ["splash", "default"]
    )
    generate_attention_kernel, generate_attention_kernel_config = load_attention_kernel_config(
        generate_attention_kernel_config, ["paged", "default"]
    )
    train_attention_kernel, train_attention_kernel_config = load_attention_kernel_config(
        train_attention_kernel_config, ["splash", "default", "ring", "ring_jax"]
    )

    # Setup mesh
    mesh = MeshShardingHelper(
        sharding, ["replica", "fsdp", "sequence", "tensor"], mesh_axis_splitting=physical_axis_splitting
    )

    with mesh.get_context():
        # Load model configuration and paths
        if load_model.startswith("paths:"):
            model_paths = json.loads(load_model[len("paths:") :])
            if "remove_dict_prefix" not in model_paths:
                model_paths["remove_dict_prefix"] = None
        else:
            raise ValueError(f"Unknown model info type: {load_model}")

        # Load model config
        config_is_temp = False
        if "config" in model_paths and model_paths["config"].startswith("gs://"):
            temp_file = tempfile.NamedTemporaryFile("wb", delete=False)
            with open_with_bucket(model_paths["config"], "rb") as f:
                temp_file.write(f.read())
            temp_file.close()
            model_paths["config"] = temp_file.name
            config_is_temp = True

        if "config" in model_paths:
            config = LLaMAConfig.from_pretrained(model_paths["config"], **model_config_override)
        elif "default_config_name" in model_paths:
            config = LLaMAConfig(**LLAMA_STANDARD_CONFIGS[model_paths["default_config_name"]], **model_config_override)
        else:
            config = LLaMAConfig(**model_config_override)

        # Create model configurations
        prefill_config = copy.deepcopy(config)
        prefill_config.attention_kernel = prefill_attention_kernel
        prefill_config.attention_kernel_settings = prefill_attention_kernel_config

        generate_config = copy.deepcopy(config)
        train_config = copy.deepcopy(config)

        if config_is_temp:
            os.remove(model_paths["config"])

        # Initialize models
        prefill_model = FlaxLLaMAForCausalLM(
            prefill_config,
            dtype=inference_activation_dtype,
            _do_init=False,
            param_dtype=inference_param_dtype,
            input_shape=(prefill_bsize, max_input_length),
        )
        generate_model = FlaxLLaMAForCausalLM(
            generate_config,
            dtype=inference_activation_dtype,
            _do_init=False,
            param_dtype=inference_param_dtype,
            input_shape=(decode_bsize, max_input_length + max_output_length - 1),
        )
        train_model = FlaxLLaMAForCausalLM(
            train_config,
            dtype=training_activation_dtype,
            _do_init=False,
            param_dtype=training_param_dtype,
            input_shape=(train_bsize, max_input_length + max_output_length - 1),
        )
        generate_model.config.attention_kernel = generate_attention_kernel
        generate_model.config.attention_kernel_settings = generate_attention_kernel_config
        train_model.config.attention_kernel = train_attention_kernel
        train_model.config.attention_kernel_settings = train_attention_kernel_config

        models = {
            "prefill": prefill_model,
            "generate": generate_model,
            "train": train_model,
        }

        # Load tokenizer
        tokenizer_is_temp = False
        if model_paths["tokenizer"].startswith("gs://"):
            temp_file = tempfile.NamedTemporaryFile("wb", delete=False)
            with open_with_bucket(model_paths["tokenizer"], "rb") as f:
                temp_file.write(f.read())
            temp_file.close()
            model_paths["tokenizer"] = temp_file.name
            tokenizer_is_temp = True

        tokenizer_kwargs = dict(
            truncation_side="right",
            padding_side="right",
            pad_token="<|reserved_special_token_0|>",
        )
        tokenizer_kwargs.update(tokenizer_override)
        tokenizer = AutoTokenizer.from_pretrained(model_paths["tokenizer"], **tokenizer_kwargs)

        if tokenizer_is_temp:
            os.remove(model_paths["tokenizer"])

        # Initialize environment with tokenization parameters
        environment = load_environments_from_config(Path(__file__).resolve().parent / environments_path, tokenizer)

        # Initialize logger
        if "enable" not in logger_config:
            logger_config["enable"] = jax.process_index() == 0
        if "config_to_log" in logger_config:
            logger_config["config_to_log"].update(args_dict)
        else:
            logger_config["config_to_log"] = args_dict
        logger = WandbLogger(wandb_project, output_dir=output_dir, **logger_config)

        # Create trainer configuration
        trainer_config = {
            **args_dict,
            "model_paths": model_paths,
            "optim_config": optim_config,
            "generation_config": generation_config,
            "test_generation_config": test_generation_config,
            "checkpointer_config": checkpointer_config,
            "inference_param_dtype": inference_param_dtype,
            "inference_activation_dtype": inference_activation_dtype,
            "training_param_dtype": training_param_dtype,
            "training_activation_dtype": training_activation_dtype,
        }

        # Initialize and run trainer
        trainer = Trainer(trainer_config, mesh, models, tokenizer, environment, logger)
        trainer.train()

        jax_distributed_barrier()
        logger.finish()
        jax_distributed_barrier()


if __name__ == "__main__":
    tyro.cli(main)
