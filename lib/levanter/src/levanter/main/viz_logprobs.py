# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import typing
from dataclasses import dataclass, field

import jax
import jmp

import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

import levanter
import levanter.config
from levanter.data.loader import DataLoader
from levanter.data.text.datasets import LmDataConfig
from levanter.model_loading import load_hf_checkpoint, load_levanter_checkpoint
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel, split_activations
from levanter.models.loss import next_token_loss
from levanter.trainer import TrainerConfig
from levanter.utils.tree_utils import inference_mode
from levanter.analysis.visualization import compute_and_diff_log_probs, compute_and_visualize_log_probs


logger = logging.getLogger(__name__)


@dataclass
class VizLmConfig:
    checkpoint_path: str
    path: str = "logprobs.html"
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: LmDataConfig = field(default_factory=LmDataConfig)
    model: LmConfig = field(default_factory=LlamaConfig)

    max_eval_length: int = 4096

    num_docs: int = 32

    checkpoint_is_hf: bool = False

    data_seed: int | None = 0

    comparison_model_path: str | None = None
    comparison_is_hf: bool = False


def main(config: VizLmConfig):
    levanter.trainer.initialize(config)
    tokenizer = config.data.the_tokenizer

    # some axes we use outside the model proper
    EvalBatch = config.trainer.EvalBatch
    Pos = config.model.max_Pos.resize(config.max_eval_length)

    validation_sets = config.data.validation_sets(Pos)

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    with config.trainer.use_device_mesh():
        key = jax.random.PRNGKey(0)

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        mp: jmp.Policy = config.trainer.mp

        # don't want to compute the mask w.r.t. the final token

        @hax.named_jit(axis_resources=compute_axis_mapping)
        def compute_log_probs(model: LmHeadModel, example: LmExample):
            model = inference_mode(model, True)
            model = mp.cast_to_compute(model)

            activations, _ = split_activations(model.activations(example.tokens, example.attn_mask, key=key))
            logits = hax.dot(activations, model.get_lm_head(), axis=model.Embed)

            loss = next_token_loss(
                model.Pos,
                model.Vocab,
                logits=logits,
                true_ids=example.tokens,
                loss_weight=example.loss_weight,
                reduction=None,
            )
            logprobs = -loss
            # roll forward to get the loss for each predicted token
            logprobs = hax.roll(logprobs, 1, Pos)
            logits = hax.roll(logits, 1, Pos)
            argmaxes = hax.argmax(logits, axis=Vocab)
            return logprobs.rearrange((EvalBatch, Pos)).array, argmaxes.rearrange((EvalBatch, Pos)).array

        model: LmHeadModel
        if config.checkpoint_is_hf:
            model = load_hf_checkpoint(
                config.model,
                config.checkpoint_path,
                axis_mapping=parameter_axis_mapping,
                tokenizer=tokenizer,
                compute_dtype=mp.compute_dtype,
            )
        else:
            model = load_levanter_checkpoint(
                config.model,
                config.checkpoint_path,
                Vocab=Vocab,
                axis_mapping=parameter_axis_mapping,
                key=key,
            )
        model = typing.cast(LmHeadModel, inference_mode(model, True))

        comparison_model: LmHeadModel | None = None
        if config.comparison_model_path is not None:
            if config.comparison_is_hf:
                comparison_model = load_hf_checkpoint(
                    config.model,
                    config.comparison_model_path,
                    axis_mapping=parameter_axis_mapping,
                    tokenizer=tokenizer,
                    compute_dtype=mp.compute_dtype,
                )
            else:
                comparison_model = load_levanter_checkpoint(
                    config.model,
                    config.comparison_model_path,
                    Vocab=Vocab,
                    axis_mapping=parameter_axis_mapping,
                    key=key,
                )
            comparison_model = typing.cast(LmHeadModel, inference_mode(comparison_model, True))

        for name, dataset in validation_sets.items():

            if config.data_seed is not None:
                dataset = dataset.shuffle(jax.random.PRNGKey(config.data_seed))

            dataset = dataset.slice_dataset(0, config.num_docs)

            loader = DataLoader(
                dataset,
                config.trainer.eval_batch_size,
                mesh=config.trainer.device_mesh,
                axis_resources=config.trainer.compute_axis_mapping,
            )

            if name:
                path = os.path.join(config.path, f"{name}.html")
            else:
                path = config.path
                if not path.endswith(".html"):
                    path = f"{path}.html"

            compute_and_visualize_log_probs(
                path=path,
                model=model,
                tokenizer=tokenizer,
                log_prob_fn=compute_log_probs,
                test_data=loader,
                max_docs=config.num_docs,
            )

            if comparison_model is not None:
                diff_path = path.replace(".html", "_diff.html")
                compute_and_diff_log_probs(
                    path=diff_path,
                    model=model,
                    comparison_model=comparison_model,
                    tokenizer=tokenizer,
                    log_prob_fn=compute_log_probs,
                    test_data=loader,
                    max_docs=config.num_docs,
                )


if __name__ == "__main__":
    levanter.config.main(main)()
