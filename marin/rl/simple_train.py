import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import equinox as eqx
import haliax as hax
import haliax.haxtyping as ht
import jax
import jax.numpy as jnp
import jax.random as jrandom
import levanter
from haliax import NamedArray
from haliax.jax_utils import maybe_rng_split
from levanter.compat.hf_checkpoints import HFCompatConfig, load_tokenizer
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig
from levanter.models.loss import next_token_loss
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count

from submodules.levanter.src.levanter.checkpoint import load_checkpoint_or_initialize

logger = logging.getLogger(__name__)


@dataclass
class TrainRlConfig:
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)

    # model config
    model: LmConfig = field(default_factory=LlamaConfig)
    initialize_from_hf: bool | str = False
    use_hf_model_config: bool = False
    initialize_from_checkpoint_path: str | None = None

    tokenizer: str | None = None

    environments_path: str = "environments.json"

    # training params
    max_input_length: int = 512
    max_output_length: int = 512

    train_bsize: int = 32
    decode_bsize: int = 32
    prefill_bsize: int = 32
    reference_logprobs_bsize: int = 32
    n_prompts_per_step: int = 128

    num_eval_examples: int = 0

    kl_coef: float = 0.0

    pad_token_id: int = 128002

    # dtypes
    inference_param_dtype: str = "bf16"
    training_param_dtype: str = "bf16"

    inference_activation_dtype: str = "bf16"
    training_activation_dtype: str = "bf16"

    # attention kernels
    train_attention_kernel_config: str = "splash: {}"
    prefill_attention_kernel_config: str = "splash: {}"
    generate_attention_kernel_config: str = "paged: {}"

    # configs that were json strings
    generation_config: dict[str, Any] = field(default_factory=dict)
    test_generation_config: dict[str, Any] = field(default_factory=dict)
    model_config_override: dict[str, Any] = field(default_factory=dict)
    tokenizer_override: dict[str, Any] = field(default_factory=dict)


class RlExample(eqx.Module):
    input_ids: ht.i32[NamedArray, "batch position"]
    loss_mask: ht.bool_[NamedArray, "batch position"]  # indicates prompt vs not prompt
    segment_ids: ht.i32[NamedArray, "batch position"]  # mostly 1/0 for padding
    returns: ht.Float[NamedArray, "batch"]  # RLOO advantages or similar
    policy_logprobs: ht.Float[NamedArray, "batch position"]
    # recompute reference logprobs on the fly?
    # reference_logprobs: ht.f32[NamedArray, "batch position"]


def main(config: TrainRlConfig):
    # Levanter initialization
    levanter.initialize(config)
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    tokenizer = config.tokenizer

    # this is some unpleasant code to allow us to initialize from a hf checkpoint. If this is your first read through,
    # I recommend skipping it for now
    if config.initialize_from_hf:
        if config.trainer.initialize_from is not None:
            raise ValueError("Cannot specify both initialize_from_hf and initialize_from")

        assert isinstance(config.model, HFCompatConfig)
        converter = config.model.hf_checkpoint_converter()

        if tokenizer is None:
            tokenizer = converter.tokenizer

        if isinstance(config.initialize_from_hf, str):
            converter = converter.replaced(reference_checkpoint=config.initialize_from_hf, tokenizer=tokenizer)
        else:
            converter = converter.replaced(tokenizer=tokenizer)

        if config.use_hf_model_config:
            # TODO: log diff of old and new config
            # NB: gross mutability
            config.model = converter.config_from_hf_config(converter.default_hf_config)
    elif isinstance(config.model, HFCompatConfig):
        converter = config.model.hf_checkpoint_converter()
        tokenizer = converter.tokenizer
    else:
        converter = None
        if config.tokenizer is None:
            raise ValueError("Tokenizer is required")
        tokenizer = config.tokenizer

    tokenizer = load_tokenizer(tokenizer)
    Vocab = hax.Axis("vocab", len(tokenizer))
    data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(config.trainer.seed), 4)

    # Load Environment
    environment = TODO

    # Loss function
    def loss_fn(model, batch: RlExample, key=None):
        dropout_key, key = maybe_rng_split(key)

        example = batch.to_lm_example()

        logits = model(
            input_ids=example.tokens,
            attn_mask=example.attn_mask,
            key=dropout_key,
        )
        logits = logits.astype(jnp.float32)
        token_loss = next_token_loss(
            "position", "vocab", logits, example.tokens, loss_mask=example.loss_mask, reduction=None
        )
        log_ratio = hax.exp((-token_loss) - jax.lax.stop_gradient(-token_loss))
        weighted_log_ratio = log_ratio * batch.loss_weights
        reinforce_loss = hax.mean(hax.negative(weighted_log_ratio), where=batch.loss_mask)
        ref_log_ratio = batch.reference_logprobs + token_loss
        kl_loss = hax.exp(ref_log_ratio) - 1 - ref_log_ratio
        kl_loss = hax.mean(kl_loss, where=batch.loss_mask)
        loss = reinforce_loss + config.kl_coef * kl_loss

        levanter.tracker.jit_log(
            {
                "train/loss": loss,
                "train/kl_loss": kl_loss,
                "train/reinforce_loss": reinforce_loss,
            }
        )

        return loss

    with Trainer(config.trainer, optimizer, loss_fn) as trainer:
        seed = config.trainer.seed

        # Model Config Loading

        if config.initialize_from_checkpoint_path is not None:
            reference_model = load_checkpoint_or_initialize(
                lambda: config.model.build(Vocab, key=model_key),
                config.initialize_from_checkpoint_path,
                subpath="model",
            )()
        elif config.initialize_from_hf:
            assert converter is not None
            # initialize from an hf pretrained model
            reference_model = converter.load_pretrained(
                config.model.model_type,
                config=config.model if not config.use_hf_model_config else None,
                axis_mapping=trainer.parameter_axis_mapping,
                dtype=trainer.mp.param_dtype,
            )
            reference_model = hax.named_jit(trainer.mp.cast_to_param, trainer.parameter_axis_mapping)(reference_model)

        state = trainer.initial_state(training_key, model=reference_model)
        reference_model = hax.named_jit(trainer.mp.cast_to_compute, trainer.parameter_axis_mapping)(reference_model)

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.model)})

        # TODO: sampler

        def get_logprobs(model, batch: RlExample, key=None):
            example = batch.to_lm_example()
            logits = model(
                input_ids=example.tokens,
                attn_mask=example.attn_mask,
                key=key,
            )
            logits = logits.astype(jnp.float32)
            token_loss = next_token_loss(
                "position", "vocab", logits, example.tokens, loss_mask=example.loss_mask, reduction=None
            )
            return -token_loss

        # Evaluation Hook
        # def eval_hook(step_info: StepInfo):
        #     if config.num_eval_examples > 0:
        #         eval_examples = environment.get_eval_examples(config.num_eval_examples)
        #         samples = batch_inference(
        #             test_sampler,
        #             step_info.model.params,
        #             [example["prompt"] for example in eval_examples],
        #             jax.random.PRNGKey(0), # Should be handled better
        #             config.test_generation_config.get("n_generations", 1),
        #         )
        #         _, metrics = environment.compute_rewards(eval_examples, samples)
        #         renamed_metrics = {f"eval/{k.replace('train_', 'test_')}": v for k, v in metrics.items()}
        #         levanter.tracker.log_metrics(renamed_metrics, step=step_info.step)

        # trainer.add_hook(eval_hook, every=trainer.config.steps_per_eval)

        # RL Data Loader
        class RlDatasetIterator(Iterator[dict[str, Any]]):
            def __init__(self, key):
                self.key = key

            def __iter__(self):
                return self

            def __next__(self):
                self.key, data_key = jax.random.split(self.key)
                rl_dataset, _ = create_dataset_from_environment(
                    environment=environment,
                    sampler=sampler,
                    params=trainer.state.model.params,
                    reference_params=reference_params,
                    get_logprobs_fn=get_logprobs,
                    n_examples=config.n_prompts_per_step,
                    prng_key=data_key,
                    reference_logprobs_bsize=config.reference_logprobs_bsize,
                    max_input_length=config.max_input_length,
                    max_output_length=config.max_output_length,
                    pad_token_id=config.pad_token_id,
                    tokenizer=tokenizer,
                    generation_config=config.generation_config,
                    mode="train",
                )
                # This returns a single batch for one training step.
                # Levanter's trainer will call this for each step.
                return next(rl_dataset.iterate_batches(batch_size=config.train_bsize, shuffle=True, loop=False))

        train_loader = RlDatasetIterator(key)
        trainer.train(state, train_loader)


if __name__ == "__main__":
    levanter.config.main(main)()
