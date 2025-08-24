import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import equinox as eqx
import haliax as hax
import haliax.haxtyping as ht
import jax.numpy as jnp
import jax.random as jrandom
import levanter
from haliax import NamedArray
from haliax.jax_utils import maybe_rng_split
from haliax.nn.loss import maybe_reduce_loss
from levanter.compat.hf_checkpoints import HFCompatConfig, load_tokenizer
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig, LmExample
from levanter.models.loss import next_token_loss
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count

from levanter.checkpoint import load_checkpoint_or_initialize
import ray
from marin.rl.datatypes import InferenceEndpoint
from marin.rl.legacy_adapter import NewStyleEnvWrapper
from marin.rl.envs.hello import HelloEnvConfig
from marin.post_training.rl_dataset import (
    create_dataset_from_environment as pt_create_dataset_from_environment,
)

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
    num_eval_examples: int = 0
    kl_coef: float = 0.0

    pad_token_id: int = 128002
    steps_per_weight_transfer: int = 8
    steps_per_replay_buffer_flush: int = 8


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
            segment_ids=self.segment_ids,
        )


def create_dataset_from_environment(*args, **kwargs):
    return pt_create_dataset_from_environment(*args, **kwargs)


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

    # Initialize Ray and load Environment via wrapper (push-based RL env -> MarinEnv API)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    environment = NewStyleEnvWrapper(
        env_cfg=HelloEnvConfig(),
        inference=InferenceEndpoint("http://unused"),
        batch_size=1,
        replica_id=0,
        tokenizer=tokenizer,
    )

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

        log_ratio = hax.exp(-token_loss + batch.policy_logprobs)
        weighted_log_ratio = log_ratio * batch.loss_weights
        reinforce_loss = maybe_reduce_loss(weighted_log_ratio, reduction, reduction_axis, batch.loss_mask)

        ref_log_ratio = batch.reference_logprobs + token_loss
        kl_loss = hax.exp(ref_log_ratio) - 1 - ref_log_ratio
        kl_loss = hax.mean(kl_loss, where=batch.loss_mask)
        kl_loss = maybe_reduce_loss(kl_loss, reduction, reduction_axis, batch.loss_mask)

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
            raise ValueError(
                "Must specify initialize_from_checkpoint_path or initialize_from_hf or initialize_from_scratch"
            )

        # create a copy of reference model
        init_model = reference_model  # full precision
        reference_model = hax.named_jit(trainer.mp.cast_to_compute, trainer.parameter_axis_mapping)(reference_model)
        state = trainer.initial_state(training_key, model=init_model)

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.model)})

        # Sampler is unused by NewStyleEnvWrapper; pass None
        sampler = None

        def get_reference_logprobs(
            model,
            batch: RlExample,
        ) -> ht.Float[NamedArray, "batch position"]:
            return next_token_loss(
                "position",
                Vocab,
                model(input_ids=batch.input_ids, attn_mask=batch.loss_mask),
                batch.input_ids,
                reduction=None,
            )

        while int(state.step) < config.trainer.num_train_steps:
            batch = next(train_loader)
            step_info = trainer.train_step(state, batch)
            state = step_info.state

            # see if it's time to do a weight transfer
            # TODO: implement weight transfer
            # if int(state.step) % config.steps_per_weight_transfer == 0:
            # weight_transfer_coordinator.schedule_weight_transfer()

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
                    sampler=sampler,
                    params=state.model,
                    reference_params=reference_model,
                    get_logprobs_fn=get_reference_logprobs,
                    n_examples=config.n_prompts_per_step,
                    prng_key=self.key,
                    reference_logprobs_bsize=config.reference_logprobs_bsize,
                    max_input_length=config.max_input_length,
                    max_output_length=config.max_output_length,
                    pad_token_id=config.pad_token_id,
                    tokenizer=tokenizer,
                    generation_config=config.generation_config,
                    mode="train",
                )
                # Use dataset's batch iterator instead of iterating the dataset directly
                self.iterator = self.dataset.iterate_batches(
                    batch_size=config.reference_logprobs_bsize, shuffle=True, loop=True
                )

            def __iter__(self):
                return self

            def __next__(self):
                batch = next(self.iterator)
                # Convert dict -> RlExample with Haliax named arrays
                batch_size = batch["input_ids"].shape[0]
                seq_len = batch["input_ids"].shape[1]
                Batch = hax.Axis("batch", batch_size)
                Pos = hax.Axis("position", seq_len)

                input_ids = hax.named(jnp.asarray(batch["input_ids"]).astype(jnp.int32), (Batch, Pos))
                loss_mask = hax.named(jnp.asarray(batch["loss_masks"]).astype(jnp.bool_), (Batch, Pos))
                segment_ids = hax.named(jnp.zeros_like(input_ids.array, dtype=jnp.int32), (Batch, Pos))
                loss_weights = hax.named(jnp.asarray(batch["loss_weights"]).astype(jnp.float32), (Batch, Pos))
                reference_logprobs = hax.named(
                    jnp.asarray(batch["reference_logprobs"]).astype(jnp.float32), (Batch, Pos)
                )
                # Policy logprobs are optional; use zeros if not provided
                if "policy_logprobs" in batch:
                    policy_logprobs = hax.named(jnp.asarray(batch["policy_logprobs"]).astype(jnp.float32), (Batch, Pos))
                else:
                    policy_logprobs = hax.named(jnp.zeros_like(reference_logprobs.array), (Batch, Pos))

                return RlExample(
                    input_ids=input_ids,
                    loss_mask=loss_mask,
                    segment_ids=segment_ids,
                    loss_weights=loss_weights,
                    policy_logprobs=policy_logprobs,
                    reference_logprobs=reference_logprobs,
                )

        # TODO: fix resume
        train_loader = RlDatasetIterator(loader_key)
        trainer.train(state, train_loader)


if __name__ == "__main__":
    levanter.config.main(main)()
