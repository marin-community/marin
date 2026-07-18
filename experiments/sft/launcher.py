# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""General chat-SFT launcher on marin's ``ArtifactStep`` flow.

``sft_step(spec, resources)`` expresses a full chat-SFT run as a lazy
``ArtifactStep[LevanterCheckpoint]``:

    dataset transform (ShareGPT/OpenAI -> canonical messages)
        -> chat tokenize/pack (a pluggable chat template + completions-only masking)
        -> weights init from a :class:`ModelSource`
        -> SFT training (Levanter ``train_lm`` or a vendored backend)

The chat template, dataset mixture, sequence length, and packing are fields of :class:`SFTSpec`;
the model — architecture, tokenizer, where the initial weights come from, and which training
backend runs it — is a separate :class:`ModelSource`, so an experiment composes model and data
as independent inputs. ``configs/delphi_1e22.py`` is one worked example (an :class:`HFModel` source
plus a magpie/warmup mixture).

Init sources (the :class:`ModelSource` implementations):

- :class:`HFModel` — SFT of an HF checkpoint used verbatim (``initialize_from_hf`` +
  ``use_hf_model_config``); Levanter re-derives the arch from the checkpoint.
- :class:`PreparedModel` — like :class:`HFModel`, but the base checkpoint + tokenizer are produced
  by an upstream preparation :class:`ArtifactStep` (e.g. the Delphi reserved-slot rename), resolved
  to that step's output directory. The step becomes a dependency.
- :class:`ConvertedCheckpointModel` — SFT from a materialized HF->Levanter conversion
  (:func:`~marin.experiment.checkpoints.hf_to_levanter`), weights-only with a fresh optimizer
  (``initialize_model_from_checkpoint_path``, the same semantics as ``initialize_from_hf``). The
  conversion carries the arch and emits the padded tokenizer, so nothing else is required.
- :class:`LevanterCheckpointModel` — SFT from an existing native Levanter checkpoint (a static dir or a
  prior ``sft_step`` step, for stage chaining), weights-only with a fresh optimizer. The checkpoint
  carries no arch/tokenizer, so ``model`` and ``tokenizer_path`` are supplied explicitly.

A vendored model family that is not a Levanter-registry ``LmHeadModel`` (its own train loop and
train state) supplies its own :class:`ModelSource` implementing the same protocol — see
``experiments/june_tpu_67b_a2b/moe/sft_launch.py``'s ``GrugModel``. The dependency runs vendored
experiment -> this launcher, never the reverse, so this module stays model-family-agnostic.

The accelerator is not part of the spec — it is chosen at launch time (see
:func:`resources_from_accelerator` and :func:`run_sft_cli`) and threaded to the training job as a
runtime arg, so the same recipe fingerprints identically whether it runs on TPU or GPU.

Why a custom step rather than ``marin.experiment.train.train_lm``: that helper cannot emit a chat
cache (``marin.experiment.data.tokenized`` has no template + completions-only masking) and only
inits from a checkpoint handle. The dataset side uses the native ``transform_dataset_step`` +
``multi_turn_adapter`` (``experiments/datasets/instruction.py``) to canonicalize each source into an
OpenAI-messages cache the chat tokenizer reads.

Identity vs execution: the ``ArtifactStep`` graph is cluster-agnostic; the training backend
dispatches the job onto whatever ``resources`` name (Fray -> Iris on TPU/CoreWeave). On a
single-driver dispatch the per-step ``StepRunner`` lock is a no-op.

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

import hashlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol, runtime_checkable

import click
import jmp
from fray.types import ANY_REGION, GpuConfig, ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import ChatLmDatasetFormat
from levanter.main.train_lm import TrainLmConfig
from levanter.models.lm_model import LmConfig
from levanter.optim.config import OptimizerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import DEFAULT_JAX_CONFIG, TrainerConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, lower
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.experiment.checkpoints import HfToLevanterCheckpoint
from marin.processing.tokenize.tokenize import TokenizeConfig, TokenizedCache
from marin.processing.tokenize.tokenize import tokenize as run_tokenize
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
# Bump to rebuild every chat cache; the (tokenizer, template, pack) hash in the name already forks a
# new cache when any of those change, so this is only for reprocessing the same recipe.
_CHAT_CACHE_VERSION = "2026.07.17"


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


@runtime_checkable
class ModelSource(Protocol):
    """A model to fine-tune: its tokenizer, its training backend, and how it is initialised.

    An implementation owns the backend-specific training config and the backend dispatch callable
    (``run``). ``sft_step`` builds the shared chat-data config and hands it to
    :meth:`build_train_config`; the returned object is what ``run`` is called with.

    The tokenizer is exposed two ways because it may not be known until the graph runs (a
    :class:`PreparedModel`'s tokenizer is an upstream step's output directory):
    :meth:`tokenizer_cache_key` is a construction-time-stable id used only to fork chat caches, and
    :meth:`resolve_tokenizer` is the build-time path handed to Levanter / the tokenizer.
    """

    def tokenizer_cache_key(self) -> str:
        """A stable, construction-time id for the tokenizer, used only to fork chat caches."""
        ...

    def resolve_tokenizer(self, ctx: StepContext) -> str:
        """The build-time tokenizer path (a hub id/dir, or a prepared step's output directory)."""
        ...

    @property
    def run(self) -> Callable[..., None]:
        """The Fray-dispatched backend entry point invoked with :meth:`build_train_config`'s output."""
        ...

    def init_deps(self) -> tuple[ArtifactStep, ...]:
        """Extra dependency steps this source contributes (e.g. a preparation or init step)."""
        ...

    def build_train_config(
        self,
        ctx: StepContext,
        spec: SFTSpec,
        data_config: LmDataConfig,
        resources: ResourceConfig,
        num_train_steps: int,
    ) -> object:
        """Build the backend training config from the shared spec + chat-data config + resolved steps."""
        ...


def _levanter_train_config(
    spec: SFTSpec,
    *,
    model_config: LmConfig,
    data_config: LmDataConfig,
    num_train_steps: int,
    initialize_from_hf: bool | str,
    initialize_model_from_checkpoint_path: str | None,
    use_hf_model_config: bool,
    eos_token_ids: Sequence[int],
    gpu_allocator: bool,
) -> TrainLmConfig:
    """Assemble the identity-bearing ``TrainLmConfig`` for a Levanter-backend SFT run.

    ``model_config`` is the concrete ``LmConfig`` the pytree is built from. With
    ``use_hf_model_config`` (the HF/prepared paths) its fields are re-derived from the checkpoint, so
    only its class matters; native init (``initialize_model_from_checkpoint_path``) does not re-derive
    it, so it must be the architecture the checkpoint was saved with.
    """
    return TrainLmConfig(
        data=data_config,
        model=model_config,
        optimizer=spec.optimizer,
        trainer=_trainer(spec, num_train_steps=num_train_steps, gpu_allocator=gpu_allocator),
        train_seq_len=spec.seq_len,
        initialize_from_hf=initialize_from_hf,
        initialize_model_from_checkpoint_path=initialize_model_from_checkpoint_path,
        use_hf_model_config=use_hf_model_config,
        # Qwen (and others) pad the embedding vocab past the tokenizer's for TPU efficiency
        # (Qwen3: model 151936 vs tokenizer 151669). Without this the Vocab axis is built from
        # len(tokenizer) while the checkpoint embedding is larger -> a pytree Vocab-size mismatch
        # at train_step trace. No-op when they already match (e.g. the Delphi prepared tokenizer).
        pad_tokenizer_to_match_model=True,
        hf_save_steps=num_train_steps,  # one HF export at the end
        hf_generation_eos_token_ids=list(eos_token_ids),
        z_loss_weight=0.0,
    )


def _levanter_pod_config(
    ctx: StepContext,
    spec: SFTSpec,
    data_config: LmDataConfig,
    resources: ResourceConfig,
    num_train_steps: int,
    *,
    model_config: LmConfig,
    initialize_from_hf: bool | str,
    initialize_model_from_checkpoint_path: str | None,
    use_hf_model_config: bool,
    eos_token_ids: Sequence[int],
) -> TrainLmOnPodConfig:
    """Wrap a Levanter ``TrainLmConfig`` in the on-pod config the ``train_lm`` backend dispatches.

    The pod's ``auto_build_caches`` follows the data config: the step-count path passes raw urls and
    builds the chat cache on the pod; the epoch path passes a pre-built chat cache (built by a
    ``chat_tokenize`` dep) and disables on-the-fly building.
    """
    gpu_allocator = not ctx.is_fingerprint and isinstance(resources.device, GpuConfig)
    return TrainLmOnPodConfig(
        train_config=_levanter_train_config(
            spec,
            model_config=model_config,
            data_config=data_config,
            num_train_steps=num_train_steps,
            initialize_from_hf=initialize_from_hf,
            initialize_model_from_checkpoint_path=initialize_model_from_checkpoint_path,
            use_hf_model_config=use_hf_model_config,
            eos_token_ids=eos_token_ids,
            gpu_allocator=gpu_allocator,
        ),
        resources=resources,
        output_path=ctx.output_path,
        auto_build_caches=data_config.auto_build_caches,
    )


def _levanter_train_job(pod_config: TrainLmOnPodConfig) -> None:
    """Dispatch a Levanter ``train_lm`` config as its own Fray training job."""
    remote(run_levanter_train_lm, resources=pod_config.resources)(pod_config)


@dataclass(frozen=True)
class HFModel:
    """Init from an HF checkpoint used verbatim: ``initialize_from_hf`` + ``use_hf_model_config``.

    ``model_ref`` is an HF hub id or a staged directory; ``tokenizer_path`` defaults to it (the
    common case). ``model_type`` is the Levanter model-registry key; ``use_hf_model_config``
    re-derives the arch from the checkpoint, so it only has to match the architecture (Delphi and
    the Qwen3 smoke are both ``qwen3``).
    """

    model_ref: str  # HF id or staged prepared-checkpoint dir
    tokenizer_path: str | None = None  # defaults to model_ref
    model_type: str = "qwen3"
    eos_token_ids: Sequence[int] = (128001, 128009)  # Delphi: <|end_of_text|> + <|eot_id|>

    def tokenizer_cache_key(self) -> str:
        return self.tokenizer_path or self.model_ref

    def resolve_tokenizer(self, ctx: StepContext) -> str:
        return self.tokenizer_path or self.model_ref

    @property
    def run(self) -> Callable[..., None]:
        return _levanter_train_job

    def init_deps(self) -> tuple[ArtifactStep, ...]:
        return ()

    def build_train_config(
        self,
        ctx: StepContext,
        spec: SFTSpec,
        data_config: LmDataConfig,
        resources: ResourceConfig,
        num_train_steps: int,
    ) -> TrainLmOnPodConfig:
        return _levanter_pod_config(
            ctx,
            spec,
            data_config,
            resources,
            num_train_steps,
            # use_hf_model_config re-derives the arch from the checkpoint, so only the class matters.
            model_config=LmConfig.get_choice_class(self.model_type)(),
            initialize_from_hf=self.model_ref,
            initialize_model_from_checkpoint_path=None,
            use_hf_model_config=True,
            eos_token_ids=self.eos_token_ids,
        )


@dataclass(frozen=True)
class PreparedModel:
    """Init from a base checkpoint produced by a preparation :class:`ArtifactStep` (Levanter backend).

    ``step`` builds the base (e.g. the Delphi reserved-slot rename + embedding reinit); ``sft_step``
    adds it as a dependency and resolves both ``initialize_from_hf`` and the tokenizer to its output
    directory. Otherwise identical to :class:`HFModel` (verbatim ``initialize_from_hf`` +
    ``use_hf_model_config``).
    """

    step: ArtifactStep[Artifact]
    model_type: str = "qwen3"
    eos_token_ids: Sequence[int] = (128001, 128009)

    def tokenizer_cache_key(self) -> str:
        # The step's name is stable at graph-construction time; the output path is not yet known.
        return self.step.name

    def resolve_tokenizer(self, ctx: StepContext) -> str:
        return ctx.artifact_path(self.step)

    @property
    def run(self) -> Callable[..., None]:
        return _levanter_train_job

    def init_deps(self) -> tuple[ArtifactStep, ...]:
        return (self.step,)

    def build_train_config(
        self,
        ctx: StepContext,
        spec: SFTSpec,
        data_config: LmDataConfig,
        resources: ResourceConfig,
        num_train_steps: int,
    ) -> TrainLmOnPodConfig:
        prepared_path = ctx.artifact_path(self.step)
        return _levanter_pod_config(
            ctx,
            spec,
            data_config,
            resources,
            num_train_steps,
            # use_hf_model_config re-derives the arch from the prepared checkpoint, so only the class matters.
            model_config=LmConfig.get_choice_class(self.model_type)(),
            initialize_from_hf=prepared_path,
            initialize_model_from_checkpoint_path=None,
            use_hf_model_config=True,
            eos_token_ids=self.eos_token_ids,
        )


def _native_init_pod_config(
    ctx: StepContext,
    spec: SFTSpec,
    data_config: LmDataConfig,
    resources: ResourceConfig,
    num_train_steps: int,
    *,
    checkpoint_path: str,
    model_config: LmConfig,
    eos_token_ids: Sequence[int],
) -> TrainLmOnPodConfig:
    """Pod config for weights-only init from a native Levanter checkpoint (fresh optimizer, step 0).

    Loads only the ``model`` subtree via ``initialize_model_from_checkpoint_path``, strictly (every
    model leaf must be present) — the same init as :class:`HFModel`'s ``initialize_from_hf`` but from a
    native checkpoint. The arch is not re-derived, so ``model_config`` must match the checkpoint's.
    """
    return _levanter_pod_config(
        ctx,
        spec,
        data_config,
        resources,
        num_train_steps,
        model_config=model_config,
        initialize_from_hf=False,
        initialize_model_from_checkpoint_path=checkpoint_path,
        use_hf_model_config=False,
        eos_token_ids=eos_token_ids,
    )


@dataclass(frozen=True)
class ConvertedCheckpointModel:
    """Init a run's weights from a materialized HF->Levanter conversion (fresh optimizer, step 0).

    The conversion (:func:`~marin.experiment.checkpoints.hf_to_levanter`) carries the architecture and
    emits a tokenizer padded to the model vocab at its output root, so this source needs nothing else.
    The conversion step becomes a dependency.
    """

    conversion: HfToLevanterCheckpoint
    eos_token_ids: Sequence[int] = (128001, 128009)

    def tokenizer_cache_key(self) -> str:
        # The conversion emits the tokenizer; its step name is a stable construction-time id.
        return self.conversion.step.name

    def resolve_tokenizer(self, ctx: StepContext) -> str:
        # The conversion emits a tokenizer padded to the model vocab at its output root.
        return ctx.artifact_path(self.conversion.step)

    @property
    def run(self) -> Callable[..., None]:
        return _levanter_train_job

    def init_deps(self) -> tuple[ArtifactStep, ...]:
        return (self.conversion.step,)

    def build_train_config(
        self,
        ctx: StepContext,
        spec: SFTSpec,
        data_config: LmDataConfig,
        resources: ResourceConfig,
        num_train_steps: int,
    ) -> TrainLmOnPodConfig:
        # The conversion saves the model as a `model` subtree at its output root; native init reads it
        # with load_checkpoint(..., subpath="model").
        return _native_init_pod_config(
            ctx,
            spec,
            data_config,
            resources,
            num_train_steps,
            checkpoint_path=ctx.artifact_path(self.conversion.step),
            model_config=self.conversion.model,
            eos_token_ids=self.eos_token_ids,
        )


@dataclass(frozen=True)
class LevanterCheckpointModel:
    """Init a run's weights from an existing native Levanter checkpoint (fresh optimizer, step 0).

    ``init_from`` is a static checkpoint directory, or an :class:`ArtifactStep` producing one (e.g. a
    prior ``sft_step`` output, for stage chaining — which becomes a dependency). The checkpoint carries
    no architecture or tokenizer, so both ``model`` (the checkpoint's architecture) and ``tokenizer_path``
    are required. For a materialized HF->Levanter conversion, which supplies both, use
    :class:`ConvertedCheckpointModel` instead.
    """

    init_from: str | ArtifactStep
    model: LmConfig
    tokenizer_path: str
    eos_token_ids: Sequence[int] = (128001, 128009)

    def tokenizer_cache_key(self) -> str:
        return self.tokenizer_path

    def resolve_tokenizer(self, ctx: StepContext) -> str:
        return self.tokenizer_path

    @property
    def run(self) -> Callable[..., None]:
        return _levanter_train_job

    def init_deps(self) -> tuple[ArtifactStep, ...]:
        return (self.init_from,) if isinstance(self.init_from, ArtifactStep) else ()

    def _init_path(self, ctx: StepContext) -> str:
        # A prior sft_step (a TrainerState whose model field serializes to `model/`) and a static native
        # checkpoint both expose the weights as a `model` subtree; native init reads them with subpath="model".
        if isinstance(self.init_from, ArtifactStep):
            return ctx.artifact_path(self.init_from)
        return self.init_from

    def build_train_config(
        self,
        ctx: StepContext,
        spec: SFTSpec,
        data_config: LmDataConfig,
        resources: ResourceConfig,
        num_train_steps: int,
    ) -> TrainLmOnPodConfig:
        return _native_init_pod_config(
            ctx,
            spec,
            data_config,
            resources,
            num_train_steps,
            checkpoint_path=self._init_path(ctx),
            model_config=self.model,
            eos_token_ids=self.eos_token_ids,
        )


@dataclass(frozen=True)
class SFTSpec:
    """A full chat-SFT run: the data and the training hyperparameters.

    The model (arch, tokenizer, init source, backend) is a separate :class:`ModelSource` so the
    two compose independently. The chat template is a parameter, so Delphi is just one instance.
    """

    name: str  # artifact name, e.g. "checkpoints/delphi-1e22-magpie-warmup-levanter-sft"
    version: str  # calver "2026.07.15"; a "-dev" suffix opts out of the cache (always rebuild)
    model: ModelSource  # arch + tokenizer + where the initial weights come from + training backend
    chat_template: str  # any jinja carrying a {% generation %} block (completions-only mask)
    datasets: Sequence[DatasetSpec]  # the instruction mixture
    optimizer: OptimizerConfig  # e.g. AdamConfig for the Levanter backend
    seq_len: int = 4096
    pack: bool = True  # chat packs by default; a step count must count packed examples
    batch_size: int = 16
    # Training length is exactly one of these. ``num_train_epochs`` (single dataset only) resolves the
    # step count at run time from the chat cache's token total -- ``ceil(epochs * tokens / (seq_len *
    # batch))``, the packed-sequence count -- so it is not hand-calibrated; it routes the run through a
    # ``chat_tokenize`` dep (see :func:`sft_step`). ``num_train_steps`` is the explicit count (required
    # for a mixture, where epoch semantics are undefined) and keeps the ``auto_build_caches`` path.
    num_train_steps: int | None = None
    num_train_epochs: int | None = None
    wandb_project: str = "marin-sft-launcher"

    def __post_init__(self) -> None:
        if (self.num_train_steps is None) == (self.num_train_epochs is None):
            raise ValueError("Set exactly one of num_train_steps or num_train_epochs.")
        if self.num_train_epochs is not None and len(self.datasets) != 1:
            raise ValueError(
                f"num_train_epochs is only defined for a single dataset (got {len(self.datasets)}); "
                "use num_train_steps for a mixture."
            )


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


def _chat_mixture_data_config(
    spec: SFTSpec,
    cache_dirs: Sequence[str],
    tokenizer: str,
    *,
    build_component: Callable[[str, ChatLmDatasetFormat], DatasetComponent],
    auto_build_caches: bool,
) -> LmDataConfig:
    """The weighted chat mixture ``LmDataConfig`` shared by the auto-build and pre-built cache paths.

    One component per dataset (``build_component`` turns a cache dir + the chat format into the
    ``DatasetComponent`` — the two paths differ only in that source shape and in
    ``auto_build_caches``). ``tokenizer`` is the resolved model tokenizer, so data and model stay
    consistent.
    """
    fmt = _chat_format(spec)
    components: dict[str, DatasetComponent] = {}
    weights: dict[str, float] = {}
    for dataset, cache_dir in zip(spec.datasets, cache_dirs, strict=True):
        components[dataset.slug] = build_component(cache_dir, fmt)
        weights[dataset.slug] = dataset.weight
    return LmDataConfig(
        tokenizer=tokenizer,
        chat_template=spec.chat_template,  # data-level default; the component format overrides it
        enforce_eos=True,
        auto_build_caches=auto_build_caches,
        components=components,
        train_weights=weights,
        mixture_block_size=_MIXTURE_BLOCK_SIZE,
    )


def build_chat_data_config(spec: SFTSpec, dep_paths: Sequence[str], tokenizer: str) -> LmDataConfig:
    """Chat caches built on the training pod from the ``transform_dataset_step`` outputs.

    ``dep_paths`` are the resolved transform outputs, aligned with ``spec.datasets``; each component
    reads the transformed ``jsonl.gz`` and Levanter builds (``auto_build_caches``) the chat cache at
    train time.
    """

    def build_component(cache_dir: str, fmt: ChatLmDatasetFormat) -> DatasetComponent:
        return DatasetComponent(
            source=UrlDatasetSourceConfig(train_urls=[prefix_join(cache_dir, "**/*.jsonl.gz")]),
            cache_dir=cache_dir,
            format=fmt,
            split="train",
        )

    return _chat_mixture_data_config(spec, dep_paths, tokenizer, build_component=build_component, auto_build_caches=True)


def _prebuilt_chat_data_config(spec: SFTSpec, cache_paths: Sequence[str], tokenizer: str) -> LmDataConfig:
    """A chat ``LmDataConfig`` over pre-built chat caches (the ``chat_tokenize`` outputs).

    ``auto_build_caches=False`` and no raw urls: the caches already exist, so Levanter reads them
    directly instead of rebuilding on the training pod.
    """

    def build_component(cache_dir: str, fmt: ChatLmDatasetFormat) -> DatasetComponent:
        return DatasetComponent(
            source=UrlDatasetSourceConfig(train_urls=[], cache_dir=cache_dir, format=fmt),
            cache_dir=cache_dir,
            format=fmt,
            split="train",
        )

    return _chat_mixture_data_config(
        spec, cache_paths, tokenizer, build_component=build_component, auto_build_caches=False
    )


def _trainer(spec: SFTSpec, *, num_train_steps: int, gpu_allocator: bool) -> TrainerConfig:
    """Trainer config. ``gpu_allocator`` adds the GPU-only cuda_async PJRT allocator."""
    jax_config = dict(DEFAULT_JAX_CONFIG)
    if gpu_allocator:
        # The cuda_async allocator is the resume-OOM defrag fix (marin #7115), GPU-only: passing
        # allocator:... to PJRT_Client_Create aborts the TPU backend and JAX falls back to CPU.
        jax_config["jax_pjrt_client_create_options"] = "allocator:cuda_async"
    return TrainerConfig(
        train_batch_size=spec.batch_size,
        num_train_steps=num_train_steps,
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


def chat_tokenize(spec: SFTSpec, dataset: DatasetSpec, transform_dep: ArtifactStep) -> ArtifactStep[TokenizedCache]:
    """A chat-format ``TokenizedCache`` step: tokenize the canonical messages with the chat template +
    completions-only mask + packing, off the training pod.

    Unlike ``auto_build_caches`` (which builds the same cache lazily on the training pod), this
    materializes the cache as an artifact, so its ``.stats.json`` token total is available to resolve
    ``num_train_epochs`` -> ``num_train_steps`` before the run starts (marin #7244). The cache's
    identity forks on ``(tokenizer, chat_template, pack)`` via the name suffix, so two recipes never
    collide on the ``StepRunner``'s name@version key.
    """
    key = hashlib.md5(f"{spec.model.tokenizer_cache_key()}|{spec.chat_template}|{spec.pack}".encode()).hexdigest()[:6]
    name = f"tokenized/{dataset.slug}-chat-{key}"

    def build_config(ctx: StepContext) -> TokenizeConfig:
        return TokenizeConfig(
            train_paths=[prefix_join(ctx.artifact_path(transform_dep), "**/*.jsonl.gz")],
            validation_paths=[],
            cache_path=ctx.output_path,
            tokenizer=spec.model.resolve_tokenizer(ctx),
            format=_chat_format(spec),
            tags=[dataset.slug],
        )

    return ArtifactStep(
        name=name,
        version=_CHAT_CACHE_VERSION,
        artifact_type=TokenizedCache,
        run=run_tokenize,
        build_config=build_config,
        # The model's init step also provides the tokenizer (a prepared checkpoint or an HF->Levanter
        # conversion emits it), and build_config resolves the tokenizer through it, so declare it here.
        deps=(transform_dep, *spec.model.init_deps()),
    )


def _resolve_epoch_steps(ctx: StepContext, spec: SFTSpec, chat_cache: ArtifactStep[TokenizedCache]) -> int:
    """Steps for ``num_train_epochs`` full passes over the packed chat cache.

    ``ceil(epochs * total_tokens / (seq_len * batch))`` counts packed sequences, not raw documents.
    At fingerprint time the cache is not built, so ``num_train_epochs`` stands in as the identity
    placeholder (it keeps the epoch count in the fingerprint); the real total is read at run time.
    """
    assert spec.num_train_epochs is not None
    if ctx.is_fingerprint:
        return spec.num_train_epochs
    total_tokens = ctx.resolved(chat_cache).num_train_tokens
    return math.ceil(spec.num_train_epochs * total_tokens / (spec.seq_len * spec.batch_size))


def sft_step(spec: SFTSpec, resources: ResourceConfig) -> ArtifactStep[LevanterCheckpoint]:
    """The chat-SFT run as a lazy ``ArtifactStep[LevanterCheckpoint]``.

    ``spec.model`` supplies the backend-specific training config and dispatch; ``resources`` is a
    runtime arg, so changing the accelerator never forks the checkpoint's identity. The data flow
    depends on how the training length is set:

    - ``num_train_steps`` (a mixture, or an explicit count): the transforms are the deps and Levanter
      builds the chat cache on the training pod (``auto_build_caches``).
    - ``num_train_epochs`` (a single dataset): a ``chat_tokenize`` dep materializes the chat cache so
      the step count resolves from its token total, and training reads the pre-built cache.
    """
    model = spec.model
    transform_deps = _dataset_deps(spec)

    if spec.num_train_epochs is not None:
        chat_caches = tuple(
            chat_tokenize(spec, dataset, dep) for dataset, dep in zip(spec.datasets, transform_deps, strict=True)
        )
        deps: tuple[ArtifactStep, ...] = (*chat_caches, *model.init_deps())

        def build_config(ctx: StepContext) -> object:
            run_resources = ctx.runtime_arg(_TRAIN_RESOURCES)
            tokenizer = model.resolve_tokenizer(ctx)
            data_config = _prebuilt_chat_data_config(spec, [ctx.artifact_path(c) for c in chat_caches], tokenizer)
            num_train_steps = _resolve_epoch_steps(ctx, spec, chat_caches[0])
            return model.build_train_config(ctx, spec, data_config, run_resources, num_train_steps)

    else:
        assert spec.num_train_steps is not None
        deps = (*transform_deps, *model.init_deps())

        def build_config(ctx: StepContext) -> object:
            run_resources = ctx.runtime_arg(_TRAIN_RESOURCES)
            tokenizer = model.resolve_tokenizer(ctx)
            data_config = build_chat_data_config(spec, [ctx.artifact_path(dep) for dep in transform_deps], tokenizer)
            return model.build_train_config(ctx, spec, data_config, run_resources, spec.num_train_steps)

    return ArtifactStep(
        name=spec.name,
        version=spec.version,
        artifact_type=LevanterCheckpoint,
        run=model.run,
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
