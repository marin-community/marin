# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning

import levanter
import levanter.callbacks
import levanter.eval
import levanter.eval_harness
from levanter import callbacks
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCompatConfig, save_hf_checkpoint_callback
from levanter.data.text import LMMixtureDatasetConfig, SingleDatasetLMConfig, UrlSingleDatasetLMConfig
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel, compute_next_token_loss
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count


logger = logging.getLogger(__name__)


@dataclass
class TrainLmConfig:
    data: Union[SingleDatasetLMConfig, LMMixtureDatasetConfig] = field(default_factory=UrlSingleDatasetLMConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)

    # config related to continued pretraining
    initialize_from_hf: Union[bool, str] = False
    """if provided, this will override the model config in the config. if true, use the default hf checkpoint for this model class"""
    use_hf_model_config: bool = False  # if true, replace the model config with the hf config from the checkpoint

    # TODO: atm we don't support loading from a checkpoint that has a different tokenizer. this is a bit annoying
    # TODO: atm you have to at least specify a levanter model config with the same type as the hf checkpoint

    z_loss_weight: float = 0.0

    hf_save_path: Optional[str] = None
    hf_upload: Optional[str] = None
    hf_save_steps: int = 10000
    hf_save_dtype: Optional[str] = None

    data_seed: Optional[int] = None  # if provided, will override the data seed from the trainer
    initialize_from_checkpoint_path: Optional[str] = None
    """
    If provided, will initialize from this checkpoint, used for llama style ablation. This resets the data loader.
    Note that this differs from --trainer.initialize_from, which does not reset the data loader.
    """
    epoch: int = 0
    eval_harness: Optional[LmEvalHarnessConfig] = None
    eval_harness_steps: int = 10000

    # TODO: really need to add callback framework
    log_entropy: bool = False


def main(config: TrainLmConfig):
    tokenizer = config.data.the_tokenizer

    # this is some unpleasant code to allow us to initialize from a hf checkpoint. If this is your first read through,
    # I recommend skipping it for now
    if config.initialize_from_hf:
        if config.trainer.initialize_from is not None:
            raise ValueError("Cannot specify both initialize_from_hf and initialize_from")

        assert isinstance(config.model, HFCompatConfig)
        converter = config.model.hf_checkpoint_converter()
        if hasattr(tokenizer, "vocab") and tokenizer.vocab != converter.tokenizer.vocab:
            logger.warning("The tokenizers appear to be different. You may want to check this.")

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
        converter = converter.replaced(tokenizer=tokenizer)
    else:
        converter = None

    levanter.initialize(config)
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    loss_function = functools.partial(compute_next_token_loss, logsumexp_weight=config.z_loss_weight)

    # Using the trainer as a context manager does 3 things:
    # 1. Sets the device mesh
    # 2. Sets the axis mapping (for fsdp)
    # 3. Sets the global metrics tracker
    with Trainer(config.trainer, optimizer, loss_function) as trainer:
        # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
        # this makes deterministic training pretty easy
        seed = config.trainer.seed
        data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)

        if config.data_seed is not None:
            logger.info(f"Overriding data seed with {config.data_seed}")
            data_key = jrandom.PRNGKey(config.data_seed)

        # We have two axis_mappings: one for storing the model and optimizer states, and one for compute
        # This allows Zero-3-style parameter sharding, where we shard the parameters and optimizer state across the mesh
        compute_axis_mapping = trainer.compute_axis_mapping
        parameter_axis_mapping = trainer.parameter_axis_mapping

        # some axes we need
        EvalBatch = config.trainer.EvalBatch
        Pos = config.model.Pos

        # to do partitioning, our dimensions have to be divisible by the size of the physical axes they're mapped to
        # For most things, we just insist you specify the config right, but tokenizers often have strange numbers of
        # tokens: gpt-2 has 50257, for example. So we round up.
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        # Get the training dataset
        train_dataset = config.data.train_set(
            Pos,
            config.trainer.batch_schedule,
            key=data_key,
            epochs=config.epoch,
        )

        # Get the tagged evaluation datasets
        tagged_eval_datasets = config.data.tagged_eval_sets(Pos)

        state = trainer.initial_state(training_key, model_init=lambda: config.model.build(Vocab, key=model_key))

        if int(state.step) == 0 and config.initialize_from_checkpoint_path is not None:
            state = load_checkpoint(state, config.initialize_from_checkpoint_path)

        if int(state.step) == 0:
            # TODO: I don't love that we init the model twice, but it's not a big deal i think?
            if config.initialize_from_hf:
                # initialize from an hf pretrained model
                logger.info(
                    "No training checkpoint found. Initializing model from HF checkpoint"
                    f" '{converter.reference_checkpoint}'"
                )
                # this is a bit gross, but we want to free up the memory from the model we just built
                state = dataclasses.replace(state, model=None)
                gc.collect()
                model = converter.load_pretrained(
                    config.model.model_type,
                    config=config.model if not config.use_hf_model_config else None,
                    axis_mapping=parameter_axis_mapping,
                    dtype=trainer.mp.compute_dtype,
                )
                model = named_jit(trainer.mp.cast_to_param, parameter_axis_mapping)(model)
                state = dataclasses.replace(state, model=model)
            else:
                logger.info("No checkpoint found. Starting from scratch.")

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.model)})

        max_eval_examples_per_ds = config.trainer.max_eval_batches
        if max_eval_examples_per_ds is not None:
            max_eval_examples_per_ds *= config.trainer.eval_batch_size

        if len(tagged_eval_datasets) == 0:
            logger.warning("No evaluation datasets provided.")
        else:
            cb = levanter.eval.cb_tagged_lm_evaluate(
                EvalBatch,
                tagged_eval_datasets,
                tokenizer,
                trainer.device_mesh,
                compute_axis_mapping,
                max_eval_examples_per_ds,
                mp=config.trainer.mp,
            )
            trainer.add_hook(cb, every=config.trainer.steps_per_eval)

        flops_per_token = config.model.flops_per_token(vocab_size)
        flops_per_example = 3 * flops_per_token * Pos.size if flops_per_token is not None else None
        trainer.add_hook(
            callbacks.log_performance_stats(Pos.size, trainer.config.batch_schedule, flops_per_example), every=1
        )
        # trainer.add_hook(callbacks.GradWatchCallback(include_histograms=True), every=5)

        if config.hf_save_path is not None:
            # bit gross to reach this far into the config, but it's fine
            if config.trainer.checkpointer.append_run_id_to_base_path:
                full_save_path = os.path.join(config.hf_save_path, trainer.run_id)
            else:
                full_save_path = config.hf_save_path

            save_dtype: Optional[jnp.dtype] = None
            if config.hf_save_dtype is not None:
                try:
                    save_dtype = jnp.dtype(config.hf_save_dtype)
                except TypeError:
                    logger.warning(f"Invalid hf_save_dtype: {config.hf_save_dtype}. Defaulting to None.")

            trainer.add_hook(
                save_hf_checkpoint_callback(
                    full_save_path, converter, upload_to_hf=config.hf_upload or False, save_dtype=save_dtype
                ),
                every=config.hf_save_steps,
            )

        if config.eval_harness is not None:
            eval_harness = config.eval_harness
            trainer.add_hook(
                levanter.eval_harness.lm_eval_harness(
                    eval_harness, tokenizer, EvalBatch, compute_axis_mapping, trainer.mp
                ),
                every=config.eval_harness_steps,
            )

        @named_jit(
            in_axis_resources=parameter_axis_mapping,
            axis_resources=compute_axis_mapping,
            out_axis_resources=compute_axis_mapping,
        )
        def compute_logits(model: LmHeadModel, example: LmExample):
            model = trainer.mp.cast_to_compute(model)
            activations = model.activations(example.tokens, key=None, attn_mask=example.attn_mask)
            head = model.get_lm_head()
            logits = hax.dot(activations, head, axis=model.Embed)
            return logits

        if config.log_entropy:
            for name, dataset in config.data.validation_sets(Pos).items():
                trainer.add_hook(
                    levanter.analysis.cb_compute_entropies(
                        compute_logits,
                        Vocab,
                        dataset,
                        prefix=os.path.join("analysis", name) if name else "analysis",
                        batch_size=EvalBatch.size,
                        mapping=compute_axis_mapping,
                    ),
                    every=config.trainer.steps_per_eval,
                )

        train_loader = trainer.data_loader(train_dataset)
        if state.step > 0:
            logger.info(f"Resuming training from step {state.step}")
            train_loader = train_loader.iter_from_step(state.step)
        else:
            train_loader = train_loader.iter_from_step(0)

        ## OK, actually run training!
        last_info = trainer.train(state, train_loader)

        # If running EpochDataset save latest checkpoint by default
        if trainer.config.checkpointer is not None and config.epoch > 0:
            trainer.run_hooks(last_info, force=True)
            checkpointer = trainer.config.checkpointer.create(trainer.run_id)
            checkpointer.wait_until_finished()

    # This isn't necessary except when Levanter is run in a subprocess (as happens w/ ray)
    trainer.tracker.finish()


if __name__ == "__main__":
    levanter.config.main(main)()
