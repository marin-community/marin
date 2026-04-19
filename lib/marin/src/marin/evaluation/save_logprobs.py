# Copyright The Marin Authors
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

import equinox as eqx
import fsspec
import jax

import jmp
import numpy as np
from jax.experimental import multihost_utils

import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

import levanter
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.data import DataLoader
from levanter.data.text import DatasetComponent, LmDataConfig, LMMixtureDatasetConfig
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.models.loss import next_token_loss
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.tree_utils import inference_mode

from fray.v2 import current_client
from fray.v2.types import Entrypoint, JobRequest, ResourceConfig, TpuConfig, create_environment

from marin.execution.executor import ExecutorStep, InputName, this_output_path
from marin.utils import fsspec_exists
from marin.utilities.executor_utils import ckpt_path_to_step_name

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
    packed_components = {
        name: replace(component, pack=True) if isinstance(component, DatasetComponent) else component
        for name, component in data.components.items()
    }
    packed_data = replace(data, components=packed_components, block_cross_document_attention=True)
    return packed_data


def save_logprobs(config: SaveLogprobsConfig) -> None:
    """Compute and save per-token logprobs."""
    levanter.initialize(config)
    tokenizer = config.data.the_tokenizer

    hf_checkpoint = RepoRef.from_string(config.checkpoint_path) if config.checkpoint_is_hf else None

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

        # Load model
        if config.checkpoint_path is not None and not config.checkpoint_is_hf:
            with use_cpu_device():
                model = eqx.filter_eval_shape(config.model.build, Vocab, key=key)
                model = load_checkpoint(model, config.checkpoint_path, subpath="model")
            model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)
        elif hf_checkpoint is not None:
            model_config = config.model
            if not hasattr(model_config, "hf_checkpoint_converter"):
                raise ValueError("Model config does not have an HF checkpoint converter. Can't load HF checkpoint.")
            converter: HFCheckpointConverter = model_config.hf_checkpoint_converter()
            converter = converter.replaced(reference_checkpoint=hf_checkpoint, tokenizer=tokenizer)
            model = converter.load_pretrained(model_config.model_type, ref=hf_checkpoint, dtype=mp.compute_dtype)
        else:
            raise AssertionError("Should not get here")

        for name, dataset in validation_sets.items():
            output_file = os.path.join(config.output_path, name, "outputs.jsonl.gz")
            success_file = output_file + ".SUCCESS"
            if fsspec_exists(success_file):
                if jax.process_index() == 0:
                    logger.info(f"Skipping {name}: already completed")
                continue

            loader = DataLoader(
                dataset,
                config.trainer.eval_batch_size,
                mesh=config.trainer.device_mesh,
                axis_resources=compute_axis_mapping,
            )

            cm = fsspec.open(output_file, "wt", compression="gzip") if jax.process_index() == 0 else nullcontext()
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

                        b_tokens, b_seg_ids = batch.tokens.rearrange((EvalBatch, Pos)), batch.attn_mask.segment_ids[
                            0
                        ].rearrange((EvalBatch, Pos))
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
                with fsspec.open(success_file, "w") as marker:
                    marker.write("")

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


def default_save_logprobs(
    checkpoint: str | InputName,
    model: LmConfig,
    data: LMMixtureDatasetConfig,
    resource_config: ResourceConfig,
    checkpoint_is_hf: bool,
    per_device_batch_size: int = 4,
    top_k: int | None = None,
    max_eval_length: int = 4096,
    name: str | None = None,
) -> ExecutorStep:
    """Creates an ExecutorStep that saves per-token logprobs to disk."""
    if not name:
        name = ckpt_path_to_step_name(checkpoint)

    return ExecutorStep(
        name=f"analysis/logprobs/{name}",
        fn=run_save_logprobs_on_pod,
        config=SaveLogprobsOnPodConfig(
            save_logprobs_config=SaveLogprobsConfig(
                checkpoint_path=checkpoint,  # type: ignore
                checkpoint_is_hf=checkpoint_is_hf,
                model=model,
                data=data,
                max_eval_length=max_eval_length,
                trainer=TrainerConfig(
                    tracker=NoopConfig(),
                    per_device_eval_parallelism=per_device_batch_size,
                    mp=jmp.get_policy("c=bf16"),
                ),
                output_path=this_output_path(),
                top_k=top_k,
            ),
            resources=resource_config,
        ),
    )
