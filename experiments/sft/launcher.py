# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""General chat-SFT launcher on marin's ``ArtifactStep`` flow.

``sft_step(spec, resources)`` expresses a full chat-SFT run as a lazy
``ArtifactStep[LevanterCheckpoint]``:

    dataset transform (ShareGPT/OpenAI -> canonical messages)
        -> chat tokenize/pack (a pluggable chat template + completions-only masking)
        -> Levanter SFT (``initialize_from_hf`` + ``use_hf_model_config``)
        -> HF export

The chat template, dataset mixture, model, sequence length, and packing are all fields of
:class:`SFTSpec`, so nothing is hardcoded to a model family; ``configs/delphi_1e22.py`` is one
worked example. The accelerator is *not* part of the spec — it is chosen at launch time (see
:func:`resources_from_accelerator` and :func:`run_sft_cli`) and threaded to the training job as a
runtime arg, so the same recipe fingerprints identically whether it runs on TPU or GPU.

Why a custom step rather than ``marin.experiment.train.train_lm``: that helper inits from a
checkpoint handle (``initialize_from_checkpoint_path``), not ``initialize_from_hf`` +
``use_hf_model_config`` (the SFT-of-an-HF-checkpoint path), and ``marin.experiment.data.tokenized``
cannot emit a chat cache (template + completions-only masking). The dataset side uses the native
``transform_dataset_step`` + ``multi_turn_adapter`` (``experiments/datasets/instruction.py``) to
canonicalize each source into an OpenAI-messages cache the chat tokenizer reads.

Identity vs execution: the ``ArtifactStep`` graph is cluster-agnostic; ``remote()`` dispatches the
training job onto whatever ``resources`` name (Fray -> Iris on TPU/CoreWeave). On a single-driver
``remote()`` dispatch the per-step ``StepRunner`` lock is a no-op.

Launch (a CPU coordinator job submits the training sub-job)::

    uv run iris --cluster=marin job run --job-name sft-coord --region us-east5 \\
      --cpu 1 --memory 2G --extra cpu --priority interactive --no-wait \\
      -e MARIN_PREFIX gs://marin-us-east5 -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.sft.configs.delphi_1e22 --accelerator v4-64

The CoreWeave H100 path only changes the launch flags: ``--accelerator 8xH100`` and an
``s3://`` ``MARIN_PREFIX``; no target cluster is required (Iris places H100 work on any cluster
that has them).
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta

import click
import jmp
from fray.types import ANY_REGION, GpuConfig, ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import ChatLmDatasetFormat
from levanter.main.train_lm import TrainLmConfig
from levanter.models.lm_model import LmConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import DEFAULT_JAX_CONFIG, TrainerConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, lower
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig, run_levanter_train_lm
from rigging.filesystem import prefix_join

from experiments.datasets.instruction import (
    InstructionDatasetConfig,
    multi_turn_adapter,
    transform_dataset_step,
)

# Runtime-arg key for the accelerator the job is dispatched onto (excluded from the fingerprint).
_TRAIN_RESOURCES = "train_resources"
# Compute in bf16, keep master params and optimizer state in f32 — the standard marin policy.
_MARIN_PRECISION = "p=f32,c=bfloat16"
_MIXTURE_BLOCK_SIZE = 2048


@dataclass(frozen=True)
class DatasetSpec:
    """One instruction source in the SFT mixture.

    ``adapter_kwargs`` are forwarded to ``multi_turn_adapter(**adapter_kwargs)`` so a source's
    ShareGPT/OpenAI schema (column plus role/content conventions) is declared per source.
    ``weight`` is the un-normalized mixture weight; the data config normalizes across the mix.
    """

    slug: str  # short mixture key, e.g. "magpie"
    hf_dataset_id: str
    revision: str  # 7-char commit pin, for fingerprint stability
    adapter_kwargs: Mapping[str, object]
    weight: float


@dataclass(frozen=True)
class HFModel:
    """A base model + tokenizer used verbatim: an HF hub id or a staged directory.

    Both are passed straight to Levanter (``initialize_from_hf`` and the data tokenizer), which
    resolves hub ids and fsspec paths alike. Use this when no preparation is needed.
    ``tokenizer_path`` defaults to ``model_ref`` (the common case).
    """

    model_ref: str
    tokenizer_path: str | None = None


@dataclass(frozen=True)
class PreparedModel:
    """A base model produced by a preparation :class:`ArtifactStep` (e.g. the Delphi reserved-slot
    rename + embedding reinit). ``sft_step`` adds ``step`` as a dependency and resolves both
    ``initialize_from_hf`` and the tokenizer to its output directory.
    """

    step: ArtifactStep[Artifact]


# How the base checkpoint + tokenizer are sourced: verbatim, or built by a preparation step.
ModelSource = HFModel | PreparedModel


def _model_deps(model: ModelSource) -> tuple[ArtifactStep, ...]:
    """The preparation step, if the model is built by one, else no dependencies."""
    return (model.step,) if isinstance(model, PreparedModel) else ()


def _model_refs(model: ModelSource, ctx: StepContext) -> tuple[str, str]:
    """Resolve ``(initialize_from_hf, tokenizer)`` for the model source.

    A :class:`PreparedModel` resolves both to its step's output path (a pulled value, so the
    accelerator/prefix never bears on identity); an :class:`HFModel` uses its literal ids.
    """
    if isinstance(model, PreparedModel):
        path = ctx.artifact_path(model.step)
        return path, path
    return model.model_ref, model.tokenizer_path or model.model_ref


@dataclass(frozen=True)
class SFTSpec:
    """A full chat-SFT run. The chat template is a parameter, so Delphi is just one instance."""

    name: str  # artifact name, e.g. "checkpoints/delphi-1e22-magpie-warmup-levanter-sft"
    version: str  # calver "2026.07.15"; a "-dev" suffix opts out of the cache (always rebuild)
    model: ModelSource  # HFModel (verbatim) or PreparedModel (built by a preparation step)
    chat_template: str  # any jinja carrying a {% generation %} block (completions-only mask)
    datasets: Sequence[DatasetSpec]  # the instruction mixture
    # Levanter model registry key. Selects the HF-checkpoint converter class; use_hf_model_config
    # then re-derives the arch from the checkpoint, so this only has to match its architecture.
    # Delphi and the Qwen3 smoke are both Qwen3ForCausalLM; set it per config for another arch.
    model_type: str = "qwen3"
    seq_len: int = 4096
    pack: bool = True  # chat packs by default; num_train_steps must count packed examples
    lr: float = 1e-5
    batch_size: int = 16
    num_train_steps: int = 5307  # packed 1-epoch: total_tokens/seq_len / weight / batch
    beta2: float = 0.98
    warmup_ratio: float = 0.1
    eos_token_ids: Sequence[int] = (128001, 128009)  # Delphi: <|end_of_text|> + <|eot_id|>
    wandb_project: str = "marin-sft-launcher"


# Accelerator strings: "<count>x<gpu>" (e.g. "8xH100") for GPU, else a TPU slice variant ("v4-64").
_GPU_ACCELERATOR = re.compile(r"(?P<count>\d+)x(?P<variant>[A-Za-z][\w-]*)")


def resources_from_accelerator(accelerator: str) -> ResourceConfig:
    """Parse a launch-time accelerator string into a ``ResourceConfig``.

    GPU: ``"<count>x<type>"`` (e.g. ``"8xH100"``) runs on any region that has them — Iris places
    the work, so no target cluster is required. TPU: a slice variant (e.g. ``"v4-64"``,
    ``"v6e-4"``) inherits the coordinator job's region, since TPU pools are region-scoped.
    """
    gpu = _GPU_ACCELERATOR.fullmatch(accelerator)
    if gpu is not None:
        return ResourceConfig.with_gpu(gpu["variant"], count=int(gpu["count"]), regions=[ANY_REGION])
    return ResourceConfig.with_tpu(accelerator)


def _chat_format(spec: SFTSpec) -> ChatLmDatasetFormat:
    """Chat format carrying the spec's template plus completions-only (assistant-span) masking.

    ``pack=None`` selects Levanter's default (packs multiple conversations per sequence); ``False``
    is one conversation per sequence, padded — slower but parity-neutral.
    """
    return ChatLmDatasetFormat(
        messages_field="messages",
        chat_template=spec.chat_template,
        mask_user_turns=True,
        pack=None if spec.pack else False,
    )


def _data_config(spec: SFTSpec, dep_paths: Sequence[str], tokenizer: str) -> LmDataConfig:
    """One cache-backed chat component per dataset, weighted by ``spec.datasets``.

    ``dep_paths`` are the resolved ``transform_dataset_step`` outputs, aligned with
    ``spec.datasets``; Levanter builds the chat caches from them at train time. ``tokenizer`` is
    the resolved model tokenizer (a hub id/dir, or the prepared checkpoint's output path).
    """
    fmt = _chat_format(spec)
    components: dict[str, DatasetComponent] = {}
    weights: dict[str, float] = {}
    for dataset, cache_dir in zip(spec.datasets, dep_paths, strict=True):
        components[dataset.slug] = DatasetComponent(
            source=UrlDatasetSourceConfig(train_urls=[prefix_join(cache_dir, "**/*.jsonl.gz")]),
            cache_dir=cache_dir,
            format=fmt,
            split="train",
        )
        weights[dataset.slug] = dataset.weight
    return LmDataConfig(
        tokenizer=tokenizer,
        chat_template=spec.chat_template,  # data-level default; the component format overrides it
        enforce_eos=True,
        auto_build_caches=True,
        components=components,
        train_weights=weights,
        mixture_block_size=_MIXTURE_BLOCK_SIZE,
    )


def _optimizer(spec: SFTSpec) -> AdamConfig:
    return AdamConfig(
        learning_rate=spec.lr,
        beta1=0.9,
        beta2=spec.beta2,
        epsilon=1e-8,
        max_grad_norm=1.0,
        weight_decay=0.0,
        lr_schedule="cosine",
        warmup=spec.warmup_ratio,
        min_lr_ratio=0.0,
    )


def _trainer(spec: SFTSpec, *, gpu_allocator: bool) -> TrainerConfig:
    """Trainer config. ``gpu_allocator`` adds the GPU-only cuda_async PJRT allocator."""
    jax_config = dict(DEFAULT_JAX_CONFIG)
    if gpu_allocator:
        # The cuda_async allocator is the resume-OOM defrag fix (marin #7115), GPU-only: passing
        # allocator:... to PJRT_Client_Create aborts the TPU backend and JAX falls back to CPU.
        jax_config["jax_pjrt_client_create_options"] = "allocator:cuda_async"
    return TrainerConfig(
        train_batch_size=spec.batch_size,
        num_train_steps=spec.num_train_steps,
        steps_per_eval=500,
        jax_config=jax_config,
        mp=jmp.get_policy(_MARIN_PRECISION),
        per_device_parallelism=-1,  # auto microbatch; raise on OOM (math-equivalent)
        tracker=WandbConfig(
            project=spec.wandb_project,
            tags=["sft", "levanter", spec.name.split("/")[-1]],
        ),
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/",
            save_interval=timedelta(minutes=30),
            keep=[{"every": 5000}],
            append_run_id_to_base_path=True,
        ),
    )


def build_sft_train_config(
    spec: SFTSpec, dep_paths: Sequence[str], *, init_ref: str, tokenizer: str, gpu_allocator: bool
) -> TrainLmConfig:
    """Assemble the identity-bearing ``TrainLmConfig`` for ``spec`` (full-FT chat, init-from-HF).

    ``init_ref`` and ``tokenizer`` are the resolved model source (a hub id/dir, or the prepared
    checkpoint's output path); see :func:`_model_refs`.
    """
    # use_hf_model_config re-derives the arch from the checkpoint, so only the model config *class*
    # matters here — it selects the HF converter (LevConfigClass). Its fields would be discarded.
    model = LmConfig.get_choice_class(spec.model_type)()
    return TrainLmConfig(
        data=_data_config(spec, dep_paths, tokenizer),
        model=model,
        optimizer=_optimizer(spec),
        trainer=_trainer(spec, gpu_allocator=gpu_allocator),
        train_seq_len=spec.seq_len,
        initialize_from_hf=init_ref,
        use_hf_model_config=True,
        # Qwen (and others) pad the embedding vocab past the tokenizer's for TPU efficiency
        # (Qwen3: model 151936 vs tokenizer 151669). Without this the Vocab axis is built from
        # len(tokenizer) while the checkpoint embedding is larger -> a pytree Vocab-size mismatch
        # at train_step trace. No-op when they already match (e.g. the Delphi prepared tokenizer).
        pad_tokenizer_to_match_model=True,
        hf_save_steps=spec.num_train_steps,  # one HF export at the end
        hf_generation_eos_token_ids=list(spec.eos_token_ids),
        z_loss_weight=0.0,
    )


def _dataset_deps(spec: SFTSpec) -> tuple[ArtifactStep, ...]:
    """One native ShareGPT/OpenAI -> canonical transform per source (schema from adapter_kwargs)."""
    return tuple(
        transform_dataset_step(
            InstructionDatasetConfig(
                hf_dataset_id=dataset.hf_dataset_id,
                revision=dataset.revision,
                adapter=multi_turn_adapter(**dict(dataset.adapter_kwargs)),
                metadata_columns=[],
                name=dataset.slug,
            )
        )
        for dataset in spec.datasets
    )


def _train_job(pod_config: TrainLmOnPodConfig) -> None:
    """The step's ``run``: dispatch the config as its own Fray training job."""
    remote(run_levanter_train_lm, resources=pod_config.resources)(pod_config)


def sft_step(spec: SFTSpec, resources: ResourceConfig) -> ArtifactStep[LevanterCheckpoint]:
    """The chat-SFT run as a lazy ``ArtifactStep[LevanterCheckpoint]``.

    ``resources`` is where/how the run executes, not what it computes: it is a runtime arg, so
    changing the accelerator never forks the checkpoint's identity. The GPU-only cuda_async
    allocator is likewise resolved from the run-time accelerator, keeping the fingerprint
    device-agnostic.
    """
    dataset_deps = _dataset_deps(spec)
    deps = (*_model_deps(spec.model), *dataset_deps)

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        run_resources = ctx.runtime_arg(_TRAIN_RESOURCES)
        gpu_allocator = not ctx.is_fingerprint and isinstance(run_resources.device, GpuConfig)
        dep_paths = [ctx.artifact_path(dep) for dep in dataset_deps]
        init_ref, tokenizer = _model_refs(spec.model, ctx)
        return TrainLmOnPodConfig(
            train_config=build_sft_train_config(
                spec, dep_paths, init_ref=init_ref, tokenizer=tokenizer, gpu_allocator=gpu_allocator
            ),
            resources=run_resources,
            output_path=ctx.output_path,
            auto_build_caches=True,
        )

    return ArtifactStep(
        name=spec.name,
        version=spec.version,
        artifact_type=LevanterCheckpoint,
        run=_train_job,
        build_config=build_config,
        deps=deps,
        runtime_args={_TRAIN_RESOURCES: resources},
    )


def run_sft_cli(spec: SFTSpec) -> None:
    """Click entry point: launch ``spec`` on the accelerator named by ``--accelerator``.

    A config module ends with ``if __name__ == "__main__": run_sft_cli(SPEC)``; the user then runs
    ``python -m experiments.sft.configs.<name> --accelerator <accel>``.
    """

    @click.command()
    @click.option(
        "--accelerator",
        required=True,
        help="Accelerator: '<count>x<gpu>' (e.g. '8xH100') or a TPU slice variant (e.g. 'v4-64').",
    )
    def _cli(accelerator: str) -> None:
        resources = resources_from_accelerator(accelerator)
        StepRunner().run([lower(sft_step(spec, resources))])

    _cli()
