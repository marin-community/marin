# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass, field
from typing import Optional

import equinox as eqx
import jax
import jmp

import haliax
import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

import levanter
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.data import DataLoader
from levanter.data.text import LMMixtureDatasetConfig, SingleDatasetLMConfigBase
from levanter.eval import TaggedEvaluator, eval_model
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel, compute_next_token_loss
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.tree_utils import inference_mode


logger = logging.getLogger(__name__)


@dataclass
class EvalLmConfig:

    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[RepoRef] = None
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: SingleDatasetLMConfigBase | LMMixtureDatasetConfig = field(default_factory=SingleDatasetLMConfigBase)
    model: LmConfig = field(default_factory=LlamaConfig)

    eval_on_train: bool = False

    log_entropy: bool = False
    log_top2_gap: bool = False
    log_param_stats: bool = False


def main(config: EvalLmConfig):
    levanter.initialize(config)
    tokenizer = config.data.the_tokenizer

    Batch = config.trainer.EvalBatch
    Pos = config.model.Pos

    if config.eval_on_train:
        datasets_dict = config.data.train_sets(Pos, key=jax.random.PRNGKey(0))
        # need tagged eval sets for the evaluator
        datasets = [(ds, [name]) for name, ds in datasets_dict.items()]
    else:
        datasets = config.data.tagged_eval_sets(Pos)

    if not datasets:
        raise ValueError("no dataset found!")

    if config.trainer.max_eval_batches is not None:
        max_examples = config.trainer.max_eval_batches * config.trainer.eval_batch_size
        datasets = [(ds.take(max_examples), tags) for ds, tags in datasets]
    else:
        max_examples = None

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    if config.checkpoint_path is None and config.hf_checkpoint is None:
        raise ValueError("Must specify either checkpoint_path or hf_checkpoint")
    if config.checkpoint_path is not None and config.hf_checkpoint is not None:
        raise ValueError("Must specify either checkpoint_path or hf_checkpoint, not both")

    with config.trainer.device_mesh, hax.axis_mapping(parameter_axis_mapping):
        evaluator = TaggedEvaluator(
            Batch, datasets, tokenizer, max_examples_per_dataset=max_examples, axis_mapping=compute_axis_mapping
        )

        key = jax.random.PRNGKey(0)

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        mp: jmp.Policy = config.trainer.mp

        @hax.named_jit
        def compute_loss(model: LmHeadModel, example: LmExample):
            with hax.axis_mapping(compute_axis_mapping):
                model = inference_mode(model, True)
                model = mp.cast_to_compute(model)
                return compute_next_token_loss(model, example, key=None)

        def compute_logits(model: LmHeadModel, example: LmExample):
            model = mp.cast_to_compute(model)
            with hax.axis_mapping(compute_axis_mapping):
                activations = model.activations(example.tokens, key=None, attn_mask=example.attn_mask)
                head = model.get_lm_head()
                logits = hax.dot(activations, head, axis=model.Embed)
                return logits

        # initialize the model
        if config.checkpoint_path is not None:
            # initialize the model
            with use_cpu_device():
                model = eqx.filter_eval_shape(config.model.build, Vocab, key=key)
                # TODO: can't load the EMA model with current setup here. Not a big deal for now.
                # TODO: don't load the entire checkpoint into CPU memory when we only need our share of the model
                model = load_checkpoint(model, config.checkpoint_path, subpath="model")

            model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)
        elif config.hf_checkpoint is not None:
            # load the huggingface model
            model_config = config.model
            if not hasattr(model_config, "hf_checkpoint_converter"):
                raise ValueError("Model config does not have an HF checkpoint converter. Can't load HF checkpoint.")
            converter: HFCheckpointConverter = model_config.hf_checkpoint_converter()
            converter = converter.replaced(reference_checkpoint=config.hf_checkpoint, tokenizer=tokenizer)
            model = converter.load_pretrained(
                model_config.model_type, ref=config.hf_checkpoint, dtype=mp.compute_dtype
            )
        else:
            assert False, "Should not get here"

        log_dict = eval_model(evaluator, model, prefix="eval")

        levanter.tracker.log(log_dict, step=0)

        print("Loss:", log_dict["eval/loss"])

        if config.log_entropy:
            logger.info("Computing entropy...")
            for name, dataset in config.data.validation_sets(Pos).items():
                if config.trainer.max_eval_batches is not None:
                    dataset = dataset.take(config.trainer.max_eval_batches * config.trainer.eval_batch_size)
                loader = DataLoader(dataset, batch_size=config.trainer.eval_batch_size)
                entropy_hist = levanter.analysis.compute_entropy_histogram(
                    model,
                    Vocab,
                    compute_logits,
                    loader,
                )

                levanter.tracker.log(
                    {
                        f"analysis/{name}/entropy": entropy_hist,
                    },
                    step=0,
                )

        if config.log_top2_gap:
            logger.info("Computing top2_gap...")
            for name, dataset in config.data.validation_sets(Pos).items():
                if config.trainer.max_eval_batches is not None:
                    dataset = dataset.take(config.trainer.max_eval_batches * config.trainer.eval_batch_size)
                    loader = DataLoader(dataset, batch_size=config.trainer.eval_batch_size)
                    top2_gap_hist = levanter.analysis.compute_top2_gap_histogram(
                        model,
                        Vocab,
                        compute_logits,
                        loader,
                    )

                    levanter.tracker.log(
                        {
                            f"analysis/{name}/top2_gap": top2_gap_hist,
                        },
                        step=0,
                    )

        if config.log_param_stats:
            logger.info("Computing param stats...")
            log_dict = haliax.named_jit(levanter.analysis.summary_statistics_for_tree)(
                "params", model, split_scan_layers=True, include_histogram=True
            )

            levanter.tracker.log(log_dict, step=0)

    # ray tasks don't reliably wait for the subprocesses to finish, so we need to manually finish the tracker
    levanter.tracker.current_tracker().finish()


if __name__ == "__main__":
    levanter.config.main(main)()
