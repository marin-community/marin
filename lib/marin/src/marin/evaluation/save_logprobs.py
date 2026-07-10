# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Save per-token log probabilities for a language model on a dataset.

This module computes per-token logprobs using Levanter on TPU and saves them
to gzipped JSONL files. Optionally saves top-k logprobs at each position.
"""

import json
import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass, field, replace

import haliax as hax
import jax
import jmp
import levanter
import levanter.tracker
import numpy as np
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, TpuConfig, create_environment
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from jax.experimental import multihost_utils
from levanter.data.loader import DataLoader
from levanter.data.text.datasets import LmDataConfig
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.models.loss import next_token_loss
from levanter.trainer import TrainerConfig
from levanter.utils.tree_utils import inference_mode
from rigging.filesystem import open_url

from marin.evaluation.model_loading import load_eval_model
from marin.processing.tokenize.data_configs import with_pack

logger = logging.getLogger(__name__)


@dataclass
class SaveLogprobsConfig:
    """Configuration for saving per-token logprobs. Also serves as the Levanter init config."""

    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(mp=jmp.get_policy("c=bf16")))
    data: LmDataConfig = field(default_factory=LmDataConfig)
    model: LmConfig = field(default_factory=LlamaConfig)
    checkpoint_path: str | None = None
    checkpoint_is_hf: bool = False
    max_eval_length: int = 4096
    output_path: str = ""
    top_k: int | None = None


@dataclass(frozen=True)
class SaveLogprobsOnPodConfig:
    """Wrapper config for running save_logprobs on a TPU pod via fray."""

    save_logprobs_config: SaveLogprobsConfig
    resources: ResourceConfig


def _force_pack_data(data: LmDataConfig) -> LmDataConfig:
    return replace(with_pack(data, True), block_cross_document_attention=True)


def save_logprobs(config: SaveLogprobsConfig) -> None:
    """Compute and save per-token logprobs."""
    levanter.trainer.initialize(config)
    tokenizer = config.data.the_tokenizer

    if config.checkpoint_path is None:
        raise ValueError("save_logprobs requires checkpoint_path")

    EvalBatch = config.trainer.EvalBatch
    Pos = config.model.max_Pos.resize(config.max_eval_length)

    packed_data = _force_pack_data(config.data)
    validation_sets = packed_data.validation_sets(Pos)

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    with config.trainer.use_device_mesh(), hax.axis_mapping(parameter_axis_mapping):
        key = jax.random.PRNGKey(0)

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        mp: jmp.Policy = config.trainer.mp

        @hax.named_jit
        def compute_forward(model: LmHeadModel, example: LmExample):
            """Shared forward pass: returns per-token logprobs and logits."""
            model = inference_mode(model, True)
            model = mp.cast_to_compute(model)
            activations = model.activations(example.tokens, example.attn_mask, key=key)
            logits = hax.dot(activations, model.get_lm_head(), axis=model.Embed)
            loss = next_token_loss(
                model.Pos,
                model.Vocab,
                logits=logits,
                true_ids=example.tokens,
                loss_weight=example.loss_weight,
                reduction=None,
            )
            logprobs = hax.nn.log_softmax(logits, axis=model.Vocab)

            return loss.rearrange((EvalBatch, Pos)), logprobs.rearrange((EvalBatch, Pos, model.Vocab))

        @hax.named_jit
        def compute_top(logprobs: hax.NamedArray, k: int):
            top_k_values, top_k_indices = hax.top_k(logprobs, model.Vocab, k=k, new_axis="top_k")
            TopK = top_k_values.resolve_axis("top_k")
            return top_k_values.rearrange((EvalBatch, Pos, TopK)), top_k_indices.rearrange((EvalBatch, Pos, TopK))

        model = load_eval_model(
            config.model,
            config.checkpoint_path,
            checkpoint_is_hf=config.checkpoint_is_hf,
            Vocab=Vocab,
            axis_mapping=parameter_axis_mapping,
            tokenizer=tokenizer,
            mp=mp,
            key=key,
        )

        for name, dataset in validation_sets.items():
            loader = DataLoader(
                dataset,
                config.trainer.eval_batch_size,
                mesh=config.trainer.device_mesh,
                axis_resources=compute_axis_mapping,
            )

            output_file = os.path.join(config.output_path, name, "outputs.jsonl.gz")
            cm = open_url(output_file, "wt", compression="gzip") if jax.process_index() == 0 else nullcontext()
            with cm as f:
                for batch in loader:
                    with hax.axis_mapping(compute_axis_mapping):
                        out = compute_forward(model, batch)
                        b_loss, b_logprobs = out

                        if config.top_k is not None:
                            b_topk_vals, b_topk_ids = compute_top(b_logprobs, config.top_k)
                            b_topk_vals, b_topk_ids = multihost_utils.process_allgather(
                                (b_topk_vals, b_topk_ids), tiled=True
                            )

                        b_tokens, b_seg_ids = (
                            batch.tokens.rearrange((EvalBatch, Pos)),
                            batch.attn_mask.segment_ids[0].rearrange((EvalBatch, Pos)),
                        )
                        b_loss, b_tokens, b_seg_ids = multihost_utils.process_allgather(
                            (b_loss, b_tokens, b_seg_ids), tiled=True
                        )

                    if jax.process_index() == 0:
                        b_loss = np.array(b_loss.array)
                        b_tokens = np.array(b_tokens.array)
                        b_seg_ids = np.array(b_seg_ids.array)

                        if config.top_k is not None:
                            b_topk_ids = np.array(b_topk_ids.array)
                            b_topk_vals = np.array(b_topk_vals.array)

                        for i in range(len(b_tokens)):
                            if np.all(b_tokens[i] == 0):
                                continue

                            unique_ids = np.unique(b_seg_ids[i])
                            unique_ids = unique_ids[unique_ids >= 0]  # exclude padding (-1)

                            for seg_id in unique_ids:
                                mask = b_seg_ids[i] == seg_id
                                record = {
                                    "token_ids": b_tokens[i][mask].tolist(),
                                    "losses": b_loss[i][mask].tolist(),
                                }
                                if config.top_k is not None:
                                    record["top_k_token_ids"] = b_topk_ids[i][mask].tolist()
                                    record["top_k_logprobs"] = b_topk_vals[i][mask].tolist()
                                f.write(json.dumps(record) + "\n")

            if jax.process_index() == 0:
                logger.info(f"Saved logprobs to {output_file}")

    levanter.tracker.current_tracker().finish()


def run_save_logprobs_on_pod(config: SaveLogprobsOnPodConfig) -> None:
    """Submit save_logprobs as a fray job on a TPU pod and wait for completion."""
    client = current_client()

    extras = []
    if isinstance(config.resources.device, TpuConfig):
        extras.append("tpu")

    job_request = JobRequest(
        name="save_logprobs",
        entrypoint=Entrypoint.from_callable(save_logprobs, args=[config.save_logprobs_config]),
        resources=config.resources,
        environment=create_environment(extras=extras),
    )
    job = client.submit(job_request)
    job.wait(raise_on_failure=True)
