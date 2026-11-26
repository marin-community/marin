# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""General chat-SFT launcher on marin's ``ArtifactStep`` flow.

``sft_step(spec, resources)`` expresses a full chat-SFT run as a lazy
``ArtifactStep[LevanterCheckpoint]``:

    dataset transform or prepared messages artifact
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
  ``use_hf_model_config``); Levanter re-derives the arch from the checkpoint. A pinned Hub ref is
  first copied to the regional prefix by a CPU download step, so the accelerator reads its weights
  (and, by default, its tokenizer) from the bucket rather than the Hub.
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

The recipe or command line selects the resources for the training job. Resources are runtime
arguments, so the same recipe fingerprints identically on compatible TPU and GPU allocations.

Why a custom step rather than ``marin.experiment.train.train_lm``: that helper cannot emit a chat
cache (``marin.experiment.data.tokenized`` has no template + completions-only masking) and only
inits from a checkpoint handle. Hugging Face sources use ``transform_dataset_step`` +
``multi_turn_adapter`` (``experiments/datasets/instruction.py``). An upstream
``ArtifactDatasetSpec`` can supply canonical OpenAI messages directly.

Identity vs execution: the ``ArtifactStep`` graph is cluster-agnostic; the training backend
dispatches the job onto whatever ``resources`` name (Fray -> Iris on TPU/CoreWeave). On a
single-driver dispatch the per-step ``StepRunner`` lock is a no-op.

Launch (a CPU coordinator job submits the training sub-job)::

    uv run iris --cluster=marin job run --job-name sft-coord --region us-east5 \\
      --cpu 1 --memory 2G --extra cpu --priority interactive --no-wait \\
      -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.sft.configs.delphi_1e22 --accelerator v4-64

CoreWeave H100 runs use an ``s3://`` ``MARIN_PREFIX``. Use the recipe's H100 resources or pass
``--accelerator 8xH100`` for a recipe that takes its allocation from the command line.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import timedelta
from functools import partial
from typing import Protocol, runtime_checkable

import click
import draccus
import jax.numpy as jnp
import jmp
from fray.types import ANY_REGION, GpuConfig, ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.compat.hf_checkpoints import is_hf_model_id
from levanter.data.text.datasets import (
    ConcatDatasetComponent,
    DatasetComponent,
    DatasetComponentBase,
    LmDataConfig,
    UrlDatasetSourceConfig,
)
from levanter.data.text.formats import ChatLmDatasetFormat, LmDatasetFormatBase
from levanter.eval_audit import InitialEvalMatchConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.lm_model import LmConfig
from levanter.models.qwen import Qwen3Config
from levanter.models.snowball_sft import SnowballSftConfig
from levanter.optim.config import OptimizerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import DEFAULT_JAX_CONFIG, TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.artifact import Artifact, is_mutable_version
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import ArtifactStep, StepContext, lower
from marin.execution.step_runner import StepRunner
from marin.experiment.checkpoints import HfToLevanterCheckpoint
from marin.experiment.train import train_job
from marin.processing.tokenize.tokenize import TokenizeConfig, TokenizedCache
from marin.processing.tokenize.tokenize import tokenize as run_tokenize
from marin.rl.skyrl import ArtifactHfModel
from marin.training.training import DEFAULT_HF_CHECKPOINTS_PATH, LevanterCheckpoint, TrainLmOnPodConfig
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.timing import Duration

from experiments.datasets.instruction import (
    InstructionDatasetConfig,
    multi_turn_adapter,
    transform_dataset_step,
)
from experiments.models import ModelConfig, download_model

# Runtime-arg key for the accelerator the job is dispatched onto (excluded from the fingerprint).
_TRAIN_RESOURCES = "train_resources"
# Compute in bf16, keep master params and optimizer state in f32 — the standard marin policy.
_MARIN_PRECISION = "p=f32,c=bfloat16"
_MIXTURE_BLOCK_SIZE = 2048
# Bump to rebuild every chat cache; the (tokenizer, template, pack) hash in the name already forks a
# new cache when any of those change, so this is only for reprocessing the same recipe.
_CHAT_CACHE_VERSION = "2026.07.17"
_GPU_HOST_CPU_PER_DEVICE = 8
_GPU_HOST_RAM_GB_PER_DEVICE = 96
_GPU_HOST_DISK_GB_PER_DEVICE = 48
LLAMA3_CHAT_EOS_TOKEN_IDS = (128001, 128009)  # <|end_of_text|> and <|eot_id|>
_HF_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


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
class ArtifactDatasetSpec:
    """An artifact containing rows with canonical OpenAI ``messages``.

    ``train_glob`` selects records within the artifact; ``slug`` names the mixture component.
    """

    slug: str
    artifact: ArtifactStep[Artifact]
    train_glob: str
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
    ``use_hf_model_config``, the checkpoint supplies the architecture and the given config supplies
    supported runtime settings. Native init needs the architecture the checkpoint was saved with.
    """
    return TrainLmConfig(
        data=data_config,
        model=model_config,
        optimizer=spec.optimizer,
        trainer=_trainer(
            spec,
            num_train_steps=num_train_steps,
            gpu_allocator=gpu_allocator,
            use_explicit_mesh_axes=model_config.requires_explicit_mesh_axes,
        ),
        train_seq_len=spec.seq_len,
        initialize_from_hf=initialize_from_hf,
        initialize_model_from_checkpoint_path=initialize_model_from_checkpoint_path,
        use_hf_model_config=use_hf_model_config,
        # A converted checkpoint emits a tokenizer already padded to its model vocab. Native init
        # has no HF reference from which the generic padding helper could re-derive that vocab.
        pad_tokenizer_to_match_model=initialize_model_from_checkpoint_path is None,
        # One HF export: the trainer's forced end-of-run hook pass. A period equal to the run
        # length would also fire on the last step and export the same step twice.
        hf_save_steps=num_train_steps + 1,
        hf_save_dtype=jnp.dtype(spec.hf_save_dtype) if spec.hf_save_dtype is not None else None,
        hf_save_chat_template=spec.chat_template,
        hf_generation_eos_token_ids=list(eos_token_ids),
        z_loss_weight=spec.z_loss_weight,
        global_gradient_clip=spec.global_gradient_clip,
        optimizer_num_train_steps=spec.optimizer_num_train_steps,
        snowball_sft=spec.snowball_sft,
        token_weighted_accumulation=spec.token_weighted_accumulation,
        data_seed=spec.data_seed,
        tensorstore_cache_bytes=spec.tensorstore_cache_bytes,
        evaluation_data=spec.evaluation_data,
        initial_eval_match=spec.initial_eval_match,
    )


def _validate_gpu_mesh(spec: SFTSpec, resources: ResourceConfig) -> None:
    if not isinstance(resources.device, GpuConfig):
        return
    mesh = spec.mesh or MeshConfig()
    try:
        ici, dcn = mesh.axis_shapes(resources.device.count * resources.replicas, resources.replicas)
    except ValueError as error:
        raise ValueError(f"mesh does not fit SFT resources: {error}") from error
    batch_axes = mesh.resolved_compute_mapping["batch"]
    batch_axes = (batch_axes,) if isinstance(batch_axes, str) else batch_axes
    sizes = ici | dcn
    batch_shards = math.prod(sizes.get(axis, 1) for axis in batch_axes)
    microbatch = batch_shards * max(spec.per_device_parallelism, 1)
    if spec.batch_size % microbatch:
        raise ValueError(f"SFT batch_size={spec.batch_size} must be divisible by the mesh microbatch ({microbatch})")


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
    if gpu_allocator:
        _validate_gpu_mesh(spec, resources)
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
        env_vars=({**spec.env_vars, **({"XLA_FLAGS": " ".join(spec.xla_flags)} if spec.xla_flags else {})} or None),
        auto_build_caches=data_config.auto_build_caches,
    )


def _hf_mirror_step(model_ref: str) -> ArtifactStep[LevanterCheckpoint] | None:
    """The CPU step that copies a pinned Hub repo to the regional prefix, or None for a path ref.

    Hub-vs-path uses Levanter's own rule (:func:`is_hf_model_id`), so the launcher and the loader agree. An
    ``hf://org/model@<sha>`` ref is the same Hub ref and is mirrored too.
    """
    hub_ref = model_ref.removeprefix("hf://")
    if not is_hf_model_id(hub_ref):
        return None
    repo, _, revision = hub_ref.partition("@")
    if not _HF_COMMIT_SHA.fullmatch(revision):
        raise ValueError(f"HFModel needs a Hub ref pinned to a full commit sha (org/model@<sha>), got {model_ref!r}")
    return download_model(ModelConfig(hf_repo_id=repo, hf_revision=revision))


@runtime_checkable
class HFExportSource(Protocol):
    """A model source that publishes a Hugging Face export relative to its training output."""

    def hf_export_path(self, num_train_steps: int | None) -> str | None:
        """A known relative path, or None when the final export is discovered at runtime."""
        ...


class _LevanterModelSource:
    @property
    def run(self) -> Callable[..., None]:
        return train_job

    def hf_export_path(self, num_train_steps: int | None) -> str | None:
        if num_train_steps is None:
            return None
        return f"{DEFAULT_HF_CHECKPOINTS_PATH}/step-{num_train_steps - 1}"


@dataclass(frozen=True)
class HFModel(_LevanterModelSource):
    """Init from an HF checkpoint used verbatim: ``initialize_from_hf`` + ``use_hf_model_config``.

    ``model_ref`` is a Hub repo pinned to a commit (``org/model@<40-hex sha>``) or an object-store or
    local directory holding an HF snapshot. A Hub ref is copied to the regional prefix by a CPU
    download step (:func:`experiments.models.download_model`) that becomes a dependency, so the
    accelerator task loads weights from the bucket rather than the Hub. A path is loaded in place.
    ``tokenizer_path`` defaults to the loaded checkpoint. The checkpoint supplies architecture
    fields and ``model`` supplies runtime settings.
    """

    model_ref: str  # pinned Hub ref, or a bucket/local HF snapshot dir
    tokenizer_path: str | None = None  # defaults to the loaded checkpoint
    model: LmConfig = field(default_factory=Qwen3Config)
    eos_token_ids: Sequence[int] = LLAMA3_CHAT_EOS_TOKEN_IDS
    task_timeout_seconds: int | None = None
    priority: int = 0
    max_retries_failure: int = 0
    max_task_failures: int = 0
    # Built once: ArtifactStep compares by identity, so init_deps and build_train_config must share it.
    _mirror: ArtifactStep[LevanterCheckpoint] | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_mirror", _hf_mirror_step(self.model_ref))

    def tokenizer_cache_key(self) -> str:
        return self.tokenizer_path or self.model_ref

    def resolve_tokenizer(self, ctx: StepContext) -> str:
        return self.tokenizer_path or self._checkpoint_path(ctx)

    def _checkpoint_path(self, ctx: StepContext) -> str:
        return self.model_ref if self._mirror is None else ctx.artifact_path(self._mirror)

    @property
    def run(self) -> Callable[..., None]:
        timeout = None if self.task_timeout_seconds is None else Duration.from_seconds(self.task_timeout_seconds)
        return partial(
            train_job,
            timeout=timeout,
            priority=self.priority,
            max_retries_failure=self.max_retries_failure,
            max_task_failures=self.max_task_failures,
        )

    def init_deps(self) -> tuple[ArtifactStep, ...]:
        return () if self._mirror is None else (self._mirror,)

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
            model_config=self.model,
            initialize_from_hf=self._checkpoint_path(ctx),
            initialize_model_from_checkpoint_path=None,
            use_hf_model_config=True,
            eos_token_ids=self.eos_token_ids,
        )


@dataclass(frozen=True)
class PreparedModel(_LevanterModelSource):
    """Init from a base checkpoint produced by a preparation :class:`ArtifactStep` (Levanter backend).

    ``step`` builds the base (e.g. the Delphi reserved-slot rename + embedding reinit); ``sft_step``
    adds it as a dependency and resolves both ``initialize_from_hf`` and the tokenizer to its output
    directory. Otherwise identical to :class:`HFModel` (verbatim ``initialize_from_hf`` +
    ``use_hf_model_config``).
    """

    step: ArtifactStep[Artifact]
    model: LmConfig = field(default_factory=Qwen3Config)
    eos_token_ids: Sequence[int] = LLAMA3_CHAT_EOS_TOKEN_IDS

    def tokenizer_cache_key(self) -> str:
        # The step's name is stable at graph-construction time; the output path is not yet known.
        return self.step.name

    def resolve_tokenizer(self, ctx: StepContext) -> str:
        return ctx.artifact_path(self.step)

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
            model_config=self.model,
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
class ConvertedCheckpointModel(_LevanterModelSource):
    """Init a run's weights from a materialized HF->Levanter conversion (fresh optimizer, step 0).

    The conversion (:func:`~marin.experiment.checkpoints.hf_to_levanter`) carries the architecture and
    emits a tokenizer padded to the model vocab at its output root, so this source needs nothing else.
    The conversion step becomes a dependency.
    """

    conversion: HfToLevanterCheckpoint
    eos_token_ids: Sequence[int] = LLAMA3_CHAT_EOS_TOKEN_IDS

    def tokenizer_cache_key(self) -> str:
        # The conversion emits the tokenizer; its step name is a stable construction-time id.
        return self.conversion.step.name

    def resolve_tokenizer(self, ctx: StepContext) -> str:
        # The conversion emits a tokenizer padded to the model vocab at its output root.
        return ctx.artifact_path(self.conversion.step)

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
class LevanterCheckpointModel(_LevanterModelSource):
    """Initialize weights from a native checkpoint with a fresh optimizer at step zero.

    ``init_from`` accepts a checkpoint-series directory or a Levanter checkpoint
    artifact. Native checkpoints do not identify their architecture or tokenizer,
    so ``model`` and ``tokenizer_path`` are required. Use
    :class:`ConvertedCheckpointModel` when an HF conversion supplies both.
    """

    init_from: str | ArtifactStep
    model: LmConfig
    tokenizer_path: str
    eos_token_ids: Sequence[int] = LLAMA3_CHAT_EOS_TOKEN_IDS

    def tokenizer_cache_key(self) -> str:
        return self.tokenizer_path

    def resolve_tokenizer(self, ctx: StepContext) -> str:
        return self.tokenizer_path

    def init_deps(self) -> tuple[ArtifactStep, ...]:
        return (self.init_from,) if isinstance(self.init_from, ArtifactStep) else ()

    def _init_path(self, ctx: StepContext) -> str:
        if isinstance(self.init_from, ArtifactStep):
            artifact = LevanterCheckpoint(path=ctx.artifact_path(self.init_from))
            return artifact.checkpoint_dir
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
class PreparedData:
    """A published LM data manifest and its tokenizer identity."""

    step: ArtifactStep
    file: str
    tokenizer: str

    def mix_fingerprint(self) -> str:
        """Identify a versioned mix without reading its unbuilt manifest."""
        if is_mutable_version(self.step.version):
            raise ValueError("SFT data mixes require an immutable artifact version")
        return fingerprint_hash(
            json.dumps((self.step.name, self.step.version, self.step.fingerprint_payload(), self.file, self.tokenizer))
        )

    def load(self, ctx: StepContext) -> LmDataConfig:
        """Read the completed manifest, checking its declared tokenizer."""
        path = prefix_join(ctx.artifact_path(self.step), self.file)
        with StoragePath(path).open("r") as stream:
            data = draccus.decode(LmDataConfig, json.load(stream))
        if data.tokenizer != self.tokenizer:
            raise ValueError("prepared_data file tokenizer does not match prepared_data.tokenizer")
        return data


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
    datasets: Sequence[DatasetSpec | ArtifactDatasetSpec]  # the instruction mixture
    optimizer: OptimizerConfig  # e.g. AdamConfig for the Levanter backend
    trainer_config: TrainerConfig | None = None
    z_loss_weight: float = 0.0
    global_gradient_clip: float | None = None
    initial_eval_match: InitialEvalMatchConfig | None = None
    optimizer_num_train_steps: int | None = None
    snowball_sft: SnowballSftConfig | None = None
    token_weighted_accumulation: bool = False
    data_seed: int | None = None
    tensorstore_cache_bytes: int | None = None
    evaluation_data: LmDataConfig | None = None
    mesh: MeshConfig | None = None
    seq_len: int = 4096
    pack: bool | int = True  # chat packs by default; an integer caps the number of conversations per example
    batch_size: int = 16
    # Training length is exactly one of these. ``num_train_epochs`` (single dataset only) resolves the
    # step count at run time from the chat cache's token total -- ``ceil(epochs * tokens / (seq_len *
    # batch))``, the packed-sequence count -- so it is not hand-calibrated; it routes the run through a
    # ``chat_tokenize`` dep (see :func:`sft_step`). ``num_train_steps`` is the explicit count (required
    # for a mixture, where epoch semantics are undefined) and keeps the ``auto_build_caches`` path.
    num_train_steps: int | None = None
    num_train_epochs: int | None = None
    hf_save_dtype: str | None = None
    wandb_project: str = "marin-sft-launcher"
    prepared_data: PreparedData | None = None
    prepared_data_format: LmDatasetFormatBase | None = None
    pack_lookahead_window: int | None = None
    per_device_parallelism: int = -1
    env_vars: Mapping[str, str] = field(default_factory=dict)
    xla_flags: tuple[str, ...] = ()
    checkpointer: CheckpointerConfig = field(
        default_factory=lambda: CheckpointerConfig(
            save_interval=timedelta(minutes=30), keep=None, keep_last_temporary_checkpoints=1
        )
    )
    """Rolling resume checkpoints every 30 minutes; only the final checkpoint is permanent.

    Set ``keep=[{"every": N}]`` to retain intermediate permanent checkpoints explicitly.
    Marin resolves permanent and temporary paths from the run's output directory at launch.
    """

    timeout: Duration | None = None
    """Time limit for the training task when the backend supports it."""

    def __post_init__(self) -> None:
        if self.per_device_parallelism != -1 and self.per_device_parallelism <= 0:
            raise ValueError("per_device_parallelism must be -1 or positive")
        if (self.num_train_steps is None) == (self.num_train_epochs is None):
            raise ValueError("Set exactly one of num_train_steps or num_train_epochs.")
        if self.prepared_data is not None and self.num_train_epochs is not None:
            raise ValueError("prepared_data requires num_train_steps")
        if self.num_train_epochs is not None and len(self.datasets) != 1:
            raise ValueError(
                f"num_train_epochs is only defined for a single dataset (got {len(self.datasets)}); "
                "use num_train_steps for a mixture."
            )
        if self.prepared_data is not None:
            if self.datasets:
                raise ValueError("prepared_data and datasets are mutually exclusive")
            if self.prepared_data.tokenizer != self.model.tokenizer_cache_key():
                raise ValueError("prepared_data.tokenizer must match the SFT model's tokenizer")

    def with_data_mix(self, data: PreparedData) -> SFTSpec:
        """Replace only the data input and give the changed run its own identity."""
        return replace(self, name=f"{self.name}-mix-{data.mix_fingerprint()}", datasets=(), prepared_data=data)

    def with_steps(self, steps: int, *, version: str) -> SFTSpec:
        """Set the update budget and run without a total task deadline."""
        model = replace(self.model, task_timeout_seconds=None) if isinstance(self.model, HFModel) else self.model
        return replace(self, num_train_steps=steps, num_train_epochs=None, version=version, timeout=None, model=model)

    def trained(
        self, resources: ResourceConfig, *, coordinator_cluster: str | None = None, coordinator_prefix: str | None = None
    ) -> ArtifactHfModel:
        """Build the training step and resolve its final Hugging Face export."""
        if not isinstance(self.model, HFExportSource):
            raise ValueError(f"Build a Hugging Face export for the {type(self.model).__name__} trainer")
        resources = sft_child_resources(resources, coordinator_cluster)
        _validate_gpu_mesh(self, resources)
        training = sft_step(self, resources)
        if coordinator_prefix is not None:
            training = replace(training, output_path_prefix=coordinator_prefix)
        return ArtifactHfModel(
            step=training,
            relative_path=self.model.hf_export_path(self.num_train_steps),
        )


class SFTRecipe(Protocol):
    """Produce a Hugging Face export with tokenizer files for RL and serving."""

    def trained(
        self, resources: ResourceConfig, *, coordinator_cluster: str | None = None, coordinator_prefix: str | None = None
    ) -> ArtifactHfModel: ...

    def with_data_mix(self, data: PreparedData) -> SFTRecipe: ...

    def with_steps(self, steps: int, *, version: str) -> SFTRecipe: ...


@dataclass(frozen=True)
class SFTPreset[RecipeT: SFTRecipe]:
    """Bind a named SFT recipe to its training resources."""

    name: str
    recipe: RecipeT
    resources: ResourceConfig

    def model(self, *, coordinator_cluster: str | None = None, coordinator_prefix: str | None = None) -> ArtifactHfModel:
        """Build the model export using the preset's fixed allocation."""
        return self.recipe.trained(
            self.resources, coordinator_cluster=coordinator_cluster, coordinator_prefix=coordinator_prefix
        )


def sft_child_resources(resources: ResourceConfig, coordinator_cluster: str | None) -> ResourceConfig:
    """Route same-cluster SFT jobs through the coordinator's local Iris controller."""
    if coordinator_cluster is None or resources.target_cluster is None:
        return resources
    if resources.target_cluster != coordinator_cluster:
        raise ValueError(f"SFT targets {resources.target_cluster}, but the RL coordinator targets {coordinator_cluster}")
    # A CoreWeave peer already places the coordinator and its children in one region.
    regions = () if coordinator_cluster.startswith("cw-") else resources.regions
    return replace(resources, target_cluster=None, regions=regions)


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
        count = int(gpu["count"])
        return ResourceConfig.with_gpu(
            gpu["variant"],
            count=count,
            cpu=_GPU_HOST_CPU_PER_DEVICE * count,
            ram=f"{_GPU_HOST_RAM_GB_PER_DEVICE * count}g",
            disk=f"{_GPU_HOST_DISK_GB_PER_DEVICE * count}g",
            regions=[ANY_REGION],
        )
    return ResourceConfig.with_tpu(accelerator)


def _chat_format(spec: SFTSpec) -> ChatLmDatasetFormat:
    """Chat format carrying the spec's template plus completions-only (assistant-span) masking.

    ``True`` uses Levanter's default packing limit. An integer sets the limit explicitly.
    ``False`` pads each conversation and rejects conversations that exceed the context window.
    """
    return ChatLmDatasetFormat(
        messages_field="messages",
        chat_template=spec.chat_template,
        mask_user_turns=True,
        pack=None if spec.pack is True else spec.pack,
    )


def _chat_mixture_data_config(
    spec: SFTSpec,
    components: Mapping[str, DatasetComponent],
    tokenizer: str,
    *,
    auto_build_caches: bool,
) -> LmDataConfig:
    """Apply mixture weights to either raw-message or prebuilt-cache components."""
    return LmDataConfig(
        tokenizer=tokenizer,
        chat_template=spec.chat_template,  # data-level default; the component format overrides it
        enforce_eos=True,
        auto_build_caches=auto_build_caches,
        components=dict(components),
        train_weights={dataset.slug: dataset.weight for dataset in spec.datasets},
        mixture_block_size=_MIXTURE_BLOCK_SIZE,
    )


def build_chat_data_config(spec: SFTSpec, dep_paths: Sequence[str], tokenizer: str) -> LmDataConfig:
    """Chat caches built on the training pod from canonical messages artifacts.

    ``dep_paths`` are the resolved source outputs, aligned with ``spec.datasets``.
    Levanter builds the chat cache at train time.
    """

    fmt = _chat_format(spec)
    components: dict[str, DatasetComponent] = {}
    for dataset, cache_dir in zip(spec.datasets, dep_paths, strict=True):
        train_glob = dataset.train_glob if isinstance(dataset, ArtifactDatasetSpec) else "**/*.jsonl.gz"
        components[dataset.slug] = DatasetComponent(
            source=UrlDatasetSourceConfig(train_urls=[prefix_join(cache_dir, train_glob)]),
            cache_dir=cache_dir,
            format=fmt,
            split="train",
        )

    return _chat_mixture_data_config(spec, components, tokenizer, auto_build_caches=True)


def _prebuilt_chat_data_config(spec: SFTSpec, cache_paths: Sequence[str], tokenizer: str) -> LmDataConfig:
    """A chat ``LmDataConfig`` over pre-built chat caches (the ``chat_tokenize`` outputs).

    ``auto_build_caches=False`` and no raw urls: the caches already exist, so Levanter reads them
    directly instead of rebuilding on the training pod.
    """

    fmt = _chat_format(spec)
    components = {
        dataset.slug: DatasetComponent(
            source=UrlDatasetSourceConfig(train_urls=[], cache_dir=cache_dir, format=fmt),
            cache_dir=cache_dir,
            format=fmt,
            split="train",
        )
        for dataset, cache_dir in zip(spec.datasets, cache_paths, strict=True)
    }

    return _chat_mixture_data_config(spec, components, tokenizer, auto_build_caches=False)


def sft_run_id(spec: SFTSpec) -> str:
    """Include the artifact name so a shared version does not merge tracker histories."""
    identity = hashlib.sha256(f"{spec.name}@{spec.version}".encode()).hexdigest()[:12]
    return f"{spec.name.split('/')[-1]}-{spec.version}-{identity}"


def _trainer(
    spec: SFTSpec, *, num_train_steps: int, gpu_allocator: bool, use_explicit_mesh_axes: bool = False
) -> TrainerConfig:
    """Trainer config. ``gpu_allocator`` adds the GPU-only cuda_async PJRT allocator."""
    if spec.trainer_config is not None:
        jax_config = dict(spec.trainer_config.jax_config)
        if gpu_allocator:
            jax_config["jax_pjrt_client_create_options"] = "allocator:cuda_async"
        return replace(
            spec.trainer_config,
            id=sft_run_id(spec),
            train_batch_size=spec.batch_size,
            num_train_steps=num_train_steps,
            per_device_parallelism=spec.per_device_parallelism,
            use_explicit_mesh_axes=use_explicit_mesh_axes,
            mesh=spec.mesh if spec.mesh is not None else spec.trainer_config.mesh,
            checkpointer=spec.checkpointer,
            jax_config=jax_config,
        )
    jax_config = dict(DEFAULT_JAX_CONFIG)
    if gpu_allocator:
        # The cuda_async allocator is the resume-OOM defrag fix (marin #7115), GPU-only: passing
        # allocator:... to PJRT_Client_Create aborts the TPU backend and JAX falls back to CPU.
        jax_config["jax_pjrt_client_create_options"] = "allocator:cuda_async"
    return TrainerConfig(
        id=sft_run_id(spec),
        train_batch_size=spec.batch_size,
        num_train_steps=num_train_steps,
        steps_per_eval=500,
        jax_config=jax_config,
        mp=jmp.get_policy(_MARIN_PRECISION),
        per_device_parallelism=spec.per_device_parallelism,
        use_explicit_mesh_axes=use_explicit_mesh_axes,
        mesh=spec.mesh if spec.mesh is not None else MeshConfig(),
        tracker=WandbConfig(
            project=spec.wandb_project,
            tags=["sft", "levanter", spec.name.split("/")[-1]],
        ),
        checkpointer=spec.checkpointer,
    )


def _dataset_deps(spec: SFTSpec) -> tuple[ArtifactStep, ...]:
    """Resolve each source to canonical messages, reusing already prepared artifacts."""
    return tuple(
        (
            dataset.artifact
            if isinstance(dataset, ArtifactDatasetSpec)
            else transform_dataset_step(
                InstructionDatasetConfig(
                    hf_dataset_id=dataset.hf_dataset_id,
                    revision=dataset.revision,
                    adapter=multi_turn_adapter(**dict(dataset.adapter_kwargs)),
                    metadata_columns=[],
                    name=dataset.slug,
                )
            )
        )
        for dataset in spec.datasets
    )


def chat_tokenize(
    spec: SFTSpec, dataset: DatasetSpec | ArtifactDatasetSpec, transform_dep: ArtifactStep
) -> ArtifactStep[TokenizedCache]:
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
        train_glob = dataset.train_glob if isinstance(dataset, ArtifactDatasetSpec) else "**/*.jsonl.gz"
        return TokenizeConfig(
            train_paths=[prefix_join(ctx.artifact_path(transform_dep), train_glob)],
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


def _prepared_component_format(component: DatasetComponentBase, spec: SFTSpec) -> DatasetComponentBase:
    if isinstance(component, ConcatDatasetComponent):
        return replace(
            component,
            children={name: _prepared_component_format(child, spec) for name, child in component.children.items()},
        )
    if not isinstance(component, DatasetComponent):
        raise ValueError("Prepared data format overrides require cached dataset components")
    assert spec.prepared_data_format is not None
    return replace(
        component, format=spec.prepared_data_format, pack=spec.pack, pack_lookahead_window=spec.pack_lookahead_window
    )


def sft_step(spec: SFTSpec, resources: ResourceConfig) -> ArtifactStep[LevanterCheckpoint]:
    """The chat-SFT run as a lazy ``ArtifactStep[LevanterCheckpoint]``.

    ``spec.model`` supplies the backend-specific training config and dispatch; ``resources`` is a
    runtime arg, so changing the accelerator never forks the checkpoint's identity. The data flow
    depends on how the training length is set:

    - ``num_train_steps`` (a mixture, or an explicit count): the message sources are deps and
      Levanter builds the chat cache on the training pod (``auto_build_caches``).
    - ``num_train_epochs`` (a single dataset): a ``chat_tokenize`` dep materializes the chat cache so
      the step count resolves from its token total, and training reads the pre-built cache.
    """
    model = spec.model
    transform_deps = _dataset_deps(spec)

    def train_config(ctx: StepContext, data: LmDataConfig, steps: int) -> object:
        return model.build_train_config(ctx, spec, data, ctx.runtime_arg(_TRAIN_RESOURCES), steps)

    if (prepared := spec.prepared_data) is not None:
        deps = (prepared.step, *model.init_deps())

        def build_config(ctx: StepContext) -> object:
            path = prefix_join(ctx.artifact_path(prepared.step), prepared.file)
            assert spec.num_train_steps is not None
            if ctx.is_fingerprint:
                data_config = LmDataConfig(tokenizer=prepared.tokenizer, auto_build_caches=False)
                config = {
                    "prepared_data_manifest": path,
                    "training": train_config(ctx, data_config, spec.num_train_steps),
                }
                if spec.prepared_data_format is not None:
                    config["prepared_data_format"] = spec.prepared_data_format
                    config["pack_lookahead_window"] = spec.pack_lookahead_window
                    config["pack"] = spec.pack
                return config
            data_config = prepared.load(ctx)
            if data_config.tokenizer != model.resolve_tokenizer(ctx):
                raise ValueError("prepared_data file tokenizer must match the resolved SFT model tokenizer")
            if spec.prepared_data_format is not None:
                data_config = replace(
                    data_config,
                    components={
                        name: _prepared_component_format(component, spec)
                        for name, component in data_config.components.items()
                    },
                )
            return train_config(ctx, data_config, spec.num_train_steps)

    elif spec.num_train_epochs is not None:
        chat_caches = tuple(
            chat_tokenize(spec, dataset, dep) for dataset, dep in zip(spec.datasets, transform_deps, strict=True)
        )
        deps: tuple[ArtifactStep, ...] = (*chat_caches, *model.init_deps())

        def build_config(ctx: StepContext) -> object:
            tokenizer = model.resolve_tokenizer(ctx)
            data_config = _prebuilt_chat_data_config(spec, [ctx.artifact_path(c) for c in chat_caches], tokenizer)
            num_train_steps = _resolve_epoch_steps(ctx, spec, chat_caches[0])
            return train_config(ctx, data_config, num_train_steps)

    else:
        assert spec.num_train_steps is not None
        deps = (*transform_deps, *model.init_deps())

        def build_config(ctx: StepContext) -> object:
            tokenizer = model.resolve_tokenizer(ctx)
            data_config = build_chat_data_config(spec, [ctx.artifact_path(dep) for dep in transform_deps], tokenizer)
            return train_config(ctx, data_config, spec.num_train_steps)

    return ArtifactStep(
        name=spec.name,
        version=spec.version,
        artifact_type=LevanterCheckpoint,
        run=partial(model.run, timeout=spec.timeout) if spec.timeout is not None else model.run,
        build_config=build_config,
        deps=deps,
        runtime_args={_TRAIN_RESOURCES: resources},
    )


def main(presets: Sequence[SFTPreset]) -> None:
    """Print or run a named SFT preset."""
    by_name = {preset.name: preset for preset in presets}

    @click.command()
    @click.option("--preset", type=click.Choice(tuple(by_name)), required=True)
    @click.option("--run", is_flag=True, help="Launch training and export the model.")
    def _cli(preset: str, run: bool) -> None:
        try:
            model = by_name[preset].model()
        except ValueError as error:
            raise click.UsageError(str(error)) from error
        StepRunner().run([lower(model.step)], dry_run=not run)

    _cli()


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
