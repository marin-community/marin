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
from levanter.layers import AttentionMask
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig, LmExample
from levanter.models.loss import next_token_loss
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count

from levanter.checkpoint import load_checkpoint_or_initialize

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
    initialize_from_scratch: bool = False

    tokenizer: str | None = None

    environments_path: str = "environments.json"

    # training params
    max_input_length: int = 512
    max_output_length: int = 512

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
    input_ids: ht.i32[NamedArray, "batch position"]  # type: ignore
    loss_mask: ht.bool_[NamedArray, "batch position"]  # type: ignore
    """indicates prompt vs not prompt"""
    segment_ids: ht.i32[NamedArray, "batch position"]  # type: ignore
    """mostly 1/0 for padding"""
    loss_weights: ht.f32[NamedArray, "batch position"]  # type: ignore
    """RLOO advantages or similar"""
    policy_logprobs: ht.Float[NamedArray, "batch position"]
    """policy logprobs"""
    reference_logprobs: ht.Float[NamedArray, "batch position"]

    def to_lm_example(self) -> LmExample:
        return hax.vmap(LmExample.causal, "batch")(
            tokens=self.input_ids,
            loss_mask=self.loss_mask,
            # segment_ids=self.segment_ids,
        )


def create_dataset_from_environment(
    n_examples, prng_key, max_input_length, train_bsize, tokenizer, **kwargs,
):
    # This is a dummy implementation that generates random data.
    # Replace this with your actual data loading logic.
    del kwargs
    Batch = hax.Axis("batch", train_bsize)
    Pos = hax.Axis("position", max_input_length)
    vocab_size = len(tokenizer)

    @hax.named_jit
    def example_generator(key):
        # while True:
        key, subkey = jax.random.split(key)
        input_ids = hax.random.randint(subkey, (Batch, Pos), 0, vocab_size, dtype=jnp.int32)
        loss_mask = hax.random.bernoulli(subkey, p=0.5, shape=(Batch, Pos))
        segment_ids = hax.ones((Batch, Pos), dtype=jnp.int32)
        loss_weights = hax.random.normal(subkey, (Batch, Pos))
        policy_logprobs = hax.random.normal(subkey, (Batch, Pos))
        reference_logprobs = hax.random.normal(subkey, (Batch, Pos))

        return RlExample(
            input_ids=input_ids,
            loss_mask=loss_mask,
            segment_ids=segment_ids,
            loss_weights=loss_weights,
            policy_logprobs=policy_logprobs,
            reference_logprobs=reference_logprobs,
        )

    class StubDataset:
        def __init__(self, key):
            self.key = key

        def __iter__(self):
            while True:
                self.key, subkey = jax.random.split(self.key)
                yield example_generator(subkey)

    return StubDataset(prng_key), None


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
    environment = None  # TODO: implement this

    # Loss function
    def loss_fn(model, batch: RlExample, reduction=hax.mean, reduction_axis=None, **kwargs):
        dropout_key, key = maybe_rng_split(training_key)

        example = batch.to_lm_example()

        logits = model(
            input_ids=example.tokens,
            attn_mask=example.attn_mask,
            key=dropout_key,
        )
        logits = logits.astype(jnp.float32)
        token_loss = next_token_loss(
            "position", Vocab, logits, example.tokens, loss_mask=example.loss_mask, reduction=None
        )
        log_ratio = hax.exp((-token_loss)) #- jax.lax.stop_gradient(-token_loss)))
        weighted_log_ratio = log_ratio * batch.loss_weights
        reinforce_loss = hax.mean(hax.negative(weighted_log_ratio), where=batch.loss_mask)
        ref_log_ratio = batch.reference_logprobs + token_loss
        kl_loss = hax.exp(ref_log_ratio) - 1 - ref_log_ratio
        kl_loss = hax.mean(kl_loss, where=batch.loss_mask)
        loss = reinforce_loss + config.kl_coef * kl_loss
        #
        levanter.tracker.jit_log(
            {
                # "train/loss": logits.mean().scalar(),
                # "train/loss": loss,
                # "train/kl_loss": kl_loss,
                # "train/reinforce_loss": reinforce_loss,
            }
        )

        return loss

    with Trainer(config.trainer, optimizer, loss_fn) as trainer:
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
        elif config.initialize_from_scratch:
            def init(key):
                return trainer.mp.cast_to_param(config.model.build(Vocab, key=key))
            reference_model = hax.named_jit(init, trainer.parameter_axis_mapping)(model_key)
        else:
            raise ValueError("Must specify initialize_from_checkpoint_path or initialize_from_hf or initialize_from_scratch")

        # create a copy of reference model
        init_model = reference_model  # full precision
        reference_model = hax.named_jit(trainer.mp.cast_to_compute, trainer.parameter_axis_mapping)(reference_model)
        state = trainer.initial_state(training_key, model=init_model)

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.model)})

        # TODO: sampler
        sampler = None

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
        class RlDatasetIterator(Iterator[RlExample]):
            def __init__(self, key):
                self.key = key
                self.dataset, _ = create_dataset_from_environment(
                    environment=environment,
                    # sampler=sampler,
                    # params=trainer.state.model.params,
                    # reference_params=reference_model,
                    get_logprobs_fn=get_logprobs,
                    n_examples=config.n_prompts_per_step,
                    prng_key=self.key,
                    reference_logprobs_bsize=config.reference_logprobs_bsize,
                    max_input_length=config.max_input_length,
                    max_output_length=config.max_output_length,
                    pad_token_id=config.pad_token_id,
                    tokenizer=tokenizer,
                    generation_config=config.generation_config,
                    train_bsize=config.trainer.TrainBatch.size,
                    mode="train",
                )
                self.iterator = iter(self.dataset)

            def __iter__(self):
                return self

            def __next__(self):
                return next(self.iterator)

        # TODO: fix resume
        train_loader = RlDatasetIterator(loader_key)
        trainer.train(state, train_loader)


if __name__ == "__main__":
    levanter.config.main(main)()
