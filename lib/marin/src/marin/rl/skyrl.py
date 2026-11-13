# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin ArtifactStep adapter for the vendored SkyRL executable."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import tomllib
import uuid
from collections import deque
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal, cast

import fsspec
import yaml
from finestore.rl.run_records import (
    AttemptRecord,
    LaunchReceipt,
    LaunchResult,
    LaunchState,
)
from iris.cli.connect import connect_controller, rpc_client
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import JobName, is_job_finished
from iris.rpc import controller_pb2, job_pb2
from rigging.config_discovery import find_project_root
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import StoragePath, prefix_join

from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.execution.remote import sanitize_job_name
from marin.rl.skyrl_records import (
    EXPORTS_SUBDIRECTORY,
    RESOLVED_LAUNCH_FILENAME,
    TERMINAL_RECEIPT_FILENAME,
    read_skyrl_launch_config,
    read_stored_skyrl_receipt,
)
from marin.rollouts.catalog import RolloutRunKind, record_rollout_run, rollout_run_record
from marin.training.training import LevanterCheckpoint

IRIS_HUB_CLUSTER_CONFIG = "lib/iris/config/marin.yaml"

_EXECUTION = "skyrl_execution"
_MARINSKYRL_STAGING_ROOT = PurePosixPath("/tmp/marinskyrl")
_TEMPORARY_OUTPUT_PREFIX = "skyrl"
_TRACE_JOBS_SUBDIR = "trace_jobs"
_TRAJECTORIES_SUBDIR = "trajectories"
_LAUNCHER_DIAGNOSTIC_LINES = 20
SKYRL_TEMPORARY_STORAGE_TTL_DAYS = 14
BEST_CHECKPOINT_CALLBACK = "best_checkpoint"
SKYRL_PROJECT = Path("vendor/skyrl")
# The SkyRL runtime is packaged separately, so its matching limit is checked there too.
_COLOCATED_EXPERT_BLOCK_GPU_COUNT = 4


def skyrl_temporary_run_path(output_path: str, *, ttl_days: int) -> str:
    """Return the lifecycle-managed storage path for a SkyRL run."""
    temporary_root = marin_temp_bucket(ttl_days=ttl_days, source_prefix=output_path)
    return str(StoragePath(temporary_root) / _TEMPORARY_OUTPUT_PREFIX / StoragePath(output_path).key)


@dataclass(frozen=True)
class MegatronGeometry:
    """Megatron parallelism for a policy or reference model."""

    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    context_parallel_size: int
    expert_model_parallel_size: int
    expert_tensor_parallel_size: int


def validate_megatron_geometry(geometry: MegatronGeometry, *, role: str, num_gpus: int) -> None:
    """Require a role's GPUs to form complete Megatron model and expert groups."""
    tp = geometry.tensor_model_parallel_size
    pp = geometry.pipeline_model_parallel_size
    cp = geometry.context_parallel_size
    ep = geometry.expert_model_parallel_size
    etp = geometry.expert_tensor_parallel_size
    sizes = f"TP={tp}, PP={pp}, CP={cp}, EP={ep}, ETP={etp}"
    if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in (tp, pp, cp, ep, etp)):
        raise ValueError(f"Megatron {role} has {num_gpus} GPUs with {sizes}; parallel sizes must be positive integers")
    # Megatron builds separate dense (TP, CP, DP, PP) and expert
    # (ETP, EP, expert-DP, PP) grids. CP ranks also participate in the expert grid.
    if num_gpus % (tp * pp * cp) or num_gpus % (etp * ep * pp):
        raise ValueError(
            f"Megatron {role} has {num_gpus} GPUs but requires {sizes}; "
            "GPUs must be divisible by TP*PP*CP and ETP*EP*PP"
        )


def _training_entrypoint() -> str:
    from skyrlrecipe import RLEntrypoint  # noqa: PLC0415 - optional authoring dependency

    return RLEntrypoint.STANDARD.value


@dataclass(frozen=True)
class SkyRLRolePlan:
    """Training role geometry and registered training entrypoint.

    The reference role reuses the policy's GPUs-per-node width: the vendored
    launcher requires a single GPUs-per-node value across all Iris bundles, so a
    distinct reference width is not representable downstream.
    """

    colocate_all: bool
    policy_num_nodes: int
    policy_num_gpus_per_node: int
    num_inference_engines: int
    inference_engine_tensor_parallel_size: int
    inference_engine_pipeline_parallel_size: int
    inference_engine_data_parallel_size: int
    inference_engine_expert_parallel_size: int
    train_batch_size: int
    policy_mini_batch_size: int
    micro_train_batch_size_per_gpu: int
    n_samples_per_prompt: int
    reference_num_nodes: int | None = None
    entrypoint: str = field(default_factory=_training_entrypoint)

    def __post_init__(self) -> None:
        from skyrlrecipe import RLEntrypoint  # noqa: PLC0415 - optional authoring dependency

        selected = RLEntrypoint(self.entrypoint)
        if selected not in (RLEntrypoint.STANDARD, RLEntrypoint.TASKCOMPENDIUM):
            raise ValueError("SkyRL training plans support standard or taskcompendium entrypoints")
        object.__setattr__(self, "entrypoint", selected.value)

    @property
    def colocate_policy_ref(self) -> bool:
        return self.reference_num_nodes is None

    @property
    def num_nodes(self) -> int:
        """Count the nodes used by training and inference roles."""
        if self.colocate_all:
            return self.policy_num_nodes
        return self.policy_num_nodes + (self.reference_num_nodes or 0) + _inference_nodes(self)

    def config_values(self) -> dict[str, object]:
        """Canonical native placement, batch and inference settings."""
        return {
            "trainer.placement.colocate_all": self.colocate_all,
            "trainer.placement.colocate_policy_ref": self.colocate_policy_ref,
            "trainer.placement.policy_num_nodes": self.policy_num_nodes,
            "trainer.placement.policy_num_gpus_per_node": self.policy_num_gpus_per_node,
            "trainer.placement.ref_num_nodes": self.reference_num_nodes or self.policy_num_nodes,
            "trainer.placement.ref_num_gpus_per_node": self.policy_num_gpus_per_node,
            "trainer.train_batch_size": self.train_batch_size,
            "trainer.policy_mini_batch_size": self.policy_mini_batch_size,
            "trainer.micro_train_batch_size_per_gpu": self.micro_train_batch_size_per_gpu,
            "generator.num_inference_engines": self.num_inference_engines,
            "generator.inference_engine_tensor_parallel_size": self.inference_engine_tensor_parallel_size,
            "generator.inference_engine_pipeline_parallel_size": self.inference_engine_pipeline_parallel_size,
            "generator.inference_engine_data_parallel_size": self.inference_engine_data_parallel_size,
            "generator.inference_engine_expert_parallel_size": self.inference_engine_expert_parallel_size,
            "generator.n_samples_per_prompt": self.n_samples_per_prompt,
        }

    def validate(self, topology: SkyRLTopology) -> None:
        """Validate training geometry against the allocated topology."""
        plan = self
        _validate_inference_geometry(plan, topology)
        tensor_pipeline_gpus = self.inference_engine_tensor_parallel_size * self.inference_engine_pipeline_parallel_size
        rollout_gpus = self.num_inference_engines * tensor_pipeline_gpus * self.inference_engine_data_parallel_size
        for field_name in (
            "policy_num_nodes",
            "policy_num_gpus_per_node",
            "train_batch_size",
            "policy_mini_batch_size",
            "micro_train_batch_size_per_gpu",
            "n_samples_per_prompt",
        ):
            if getattr(plan, field_name) <= 0:
                raise ValueError(f"SkyRL role plan {field_name} must be positive")
        if plan.num_inference_engines == 0 and plan.colocate_all:
            raise ValueError("a SkyRL role plan without inference engines cannot colocate policy and rollout roles")
        if plan.policy_num_nodes > topology.num_nodes:
            raise ValueError("SkyRL policy_num_nodes exceeds the allocated topology")
        if plan.policy_num_gpus_per_node > topology.gpus_per_node:
            raise ValueError("SkyRL policy_num_gpus_per_node exceeds the GPUs on one allocated node")
        if plan.colocate_all and not plan.colocate_policy_ref:
            raise ValueError("colocate_all requires a shared policy/reference footprint")
        if plan.reference_num_nodes is not None and plan.reference_num_nodes <= 0:
            raise ValueError("SkyRL reference_num_nodes must be positive")

        if plan.colocate_all and topology.gpus_per_node % tensor_pipeline_gpus:
            raise ValueError(
                "each colocated SkyRL inference engine TP*PP slice must divide gpus_per_node; "
                f"got TP*PP={tensor_pipeline_gpus} and gpus_per_node={topology.gpus_per_node}"
            )
        policy_gpus = plan.policy_num_nodes * plan.policy_num_gpus_per_node
        if plan.colocate_all and policy_gpus != rollout_gpus:
            raise ValueError(
                "colocated SkyRL roles must use the same GPUs: "
                f"policy={policy_gpus}, rollout={rollout_gpus} "
                f"({plan.num_inference_engines} engines x TP{plan.inference_engine_tensor_parallel_size} "
                f"x PP{plan.inference_engine_pipeline_parallel_size} x DP{plan.inference_engine_data_parallel_size})"
            )
        planned_gpus = policy_gpus if plan.colocate_all else policy_gpus + rollout_gpus
        if plan.reference_num_nodes is not None:
            planned_gpus += plan.reference_num_nodes * plan.policy_num_gpus_per_node
        allocated_gpus = topology.num_nodes * topology.gpus_per_node
        # SkyRL derives optional critic, teacher, and draft-trainer claims from the recipe
        # and validates that the complete role plan exactly consumes this allocation.
        if planned_gpus > allocated_gpus:
            placement = "colocated policy/rollout" if plan.colocate_all else "policy + rollout"
            raise ValueError(
                "SkyRL core role plan exceeds the allocated topology: "
                f"{placement}={planned_gpus} GPUs, topology={allocated_gpus} GPUs "
                f"({topology.num_nodes} nodes x {topology.gpus_per_node})"
            )

        if plan.train_batch_size % plan.policy_mini_batch_size:
            raise ValueError("SkyRL train_batch_size must be divisible by policy_mini_batch_size")
        if plan.policy_mini_batch_size % plan.micro_train_batch_size_per_gpu:
            raise ValueError("SkyRL policy_mini_batch_size must be divisible by micro_train_batch_size_per_gpu")


@dataclass(frozen=True)
class SkyRLGenerationPlan:
    """Inference allocation with the canonical native role geometry."""

    geometry: SkyRLRolePlan

    @property
    def entrypoint(self) -> str:
        from skyrlrecipe import RLEntrypoint  # noqa: PLC0415 - optional authoring dependency

        return RLEntrypoint.GENERATE.value

    @property
    def num_nodes(self) -> int:
        return _inference_nodes(self.geometry)

    def config_values(self) -> dict[str, object]:
        return self.geometry.config_values()

    def validate(self, topology: SkyRLTopology) -> None:
        plan = self.geometry
        _validate_inference_geometry(plan, topology)
        if plan.num_inference_engines <= 0:
            raise ValueError("SkyRL role plan num_inference_engines must be positive")
        if topology.num_nodes != self.num_nodes:
            raise ValueError("SkyRL generation topology must allocate exactly the inference engine nodes")
        if plan.policy_num_gpus_per_node != topology.gpus_per_node:
            raise ValueError("SkyRL inference node width must match the whole-node topology width")


def _inference_nodes(plan: SkyRLRolePlan) -> int:
    engine_gpus = (
        plan.inference_engine_tensor_parallel_size
        * plan.inference_engine_pipeline_parallel_size
        * plan.inference_engine_data_parallel_size
    )
    engines_per_node = plan.policy_num_gpus_per_node // engine_gpus
    if not engines_per_node:
        raise ValueError("each SkyRL inference engine must fit on one node")
    return (plan.num_inference_engines + engines_per_node - 1) // engines_per_node


def _validate_inference_geometry(plan: SkyRLRolePlan, topology: SkyRLTopology) -> None:
    for name in (
        "inference_engine_tensor_parallel_size",
        "inference_engine_pipeline_parallel_size",
        "inference_engine_data_parallel_size",
        "inference_engine_expert_parallel_size",
    ):
        if getattr(plan, name) <= 0:
            raise ValueError(f"SkyRL role plan {name} must be positive")
    if plan.num_inference_engines < 0:
        raise ValueError("SkyRL role plan num_inference_engines cannot be negative")
    engine_gpus = (
        plan.inference_engine_tensor_parallel_size
        * plan.inference_engine_pipeline_parallel_size
        * plan.inference_engine_data_parallel_size
    )
    if (
        isinstance(topology.role_plan, SkyRLGenerationPlan) or not plan.colocate_all
    ) and engine_gpus > topology.gpus_per_node:
        raise ValueError(
            "each SkyRL inference engine must fit on one node, but "
            f"TP*PP*DP={engine_gpus} exceeds gpus_per_node={topology.gpus_per_node}"
        )
    expert_group_gpus = plan.inference_engine_tensor_parallel_size * plan.inference_engine_data_parallel_size
    if expert_group_gpus % plan.inference_engine_expert_parallel_size:
        raise ValueError(
            "SkyRL inference engine TP*DP must be divisible by expert parallel size; "
            f"got {expert_group_gpus} and EP={plan.inference_engine_expert_parallel_size}"
        )


@dataclass(frozen=True)
class SkyRLTopology:
    """Logical allocation validated by the selected training or generation plan."""

    num_nodes: int
    gpus_per_node: int
    gpu_variant: str
    role_plan: SkyRLRolePlan | SkyRLGenerationPlan

    def __post_init__(self) -> None:
        if self.num_nodes <= 0 or self.gpus_per_node <= 0:
            raise ValueError("SkyRL topology node and GPU counts must be positive")
        self.role_plan.validate(self)


@dataclass(frozen=True)
class SkyRLRetentionPolicy:
    """Temporary storage lifetime and rolling resume depth for one SkyRL run.

    When HF export is requested, a successful run produces a durable canonical
    export from its terminal checkpoint.
    """

    resume_checkpoint_count: int = 2
    temporary_storage_ttl_days: int = SKYRL_TEMPORARY_STORAGE_TTL_DAYS

    def __post_init__(self) -> None:
        if not 1 <= self.resume_checkpoint_count <= 10:
            raise ValueError("SkyRL resume_checkpoint_count must be between one and ten")
        if self.temporary_storage_ttl_days <= 0:
            raise ValueError("SkyRL temporary_storage_ttl_days must be positive")


@dataclass(frozen=True)
class ResolvedModelLocator:
    uri: str
    identity: str
    local_path: str
    tokenizer_uri: str
    tokenizer_revision: str


@dataclass(frozen=True)
class ResolvedDirectoryDataSource:
    uri: str
    identity: str
    local_path: str
    relative_path: str
    kind: Literal["directory"] = "directory"


class TaskTroveTagMatch(StrEnum):
    ALL = "all"
    ANY = "any"


def _normalized_selection_values(name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    if any(not value.strip() for value in values):
        raise ValueError(f"TaskTrove {name} cannot contain blank values")
    if len(set(values)) != len(values):
        raise ValueError(f"TaskTrove {name} cannot contain duplicate values")
    return tuple(sorted(values))


@dataclass(frozen=True)
class TaskTroveSelection:
    """An exact metadata predicate over one TaskTrove Clean release."""

    sources: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    modes: tuple[str, ...] = ()
    tag_match: TaskTroveTagMatch = TaskTroveTagMatch.ALL
    limit: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", _normalized_selection_values("sources", self.sources))
        object.__setattr__(self, "tags", _normalized_selection_values("tags", self.tags))
        object.__setattr__(self, "modes", _normalized_selection_values("modes", self.modes))
        if not self.sources and not self.tags and not self.modes:
            raise ValueError("TaskTrove selection requires at least one source, tag, or mode")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("TaskTrove selection limit must be positive")


@dataclass(frozen=True)
class ResolvedTaskTroveDataSource:
    uri: str
    identity: str
    local_path: str
    relative_path: str
    verifier_ref: str
    selection: TaskTroveSelection
    kind: Literal["tasktrove_parquet"] = "tasktrove_parquet"


type ResolvedDataSource = ResolvedDirectoryDataSource | ResolvedTaskTroveDataSource


def _artifact_local_path(category: str, step: ArtifactStep) -> str:
    return str(_MARINSKYRL_STAGING_ROOT / category / PurePosixPath(step.name).name)


def _validate_relative_file_path(name: str, value: str) -> None:
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"TaskTrove {name} must identify a file below the release root: {value!r}")


@dataclass(frozen=True)
class ArtifactHfModel:
    """An exact HF export; an empty relative path selects the artifact root."""

    step: ArtifactStep[LevanterCheckpoint]
    tokenizer_uri: str
    tokenizer_revision: str
    relative_path: str | None = None

    def __post_init__(self) -> None:
        if self.relative_path is not None and any(part in (".", "..") for part in self.relative_path.split("/")):
            raise ValueError("HF model relative_path must not contain dot segments; use '' for the artifact root")

    def deps(self) -> tuple[ArtifactStep, ...]:
        return (self.step,)

    def resolve(self, ctx: StepContext) -> ResolvedModelLocator:
        artifact_path = ctx.artifact_path(self.step)
        if self.relative_path is not None:
            uri = prefix_join(artifact_path, self.relative_path)
        elif ctx.is_fingerprint:
            uri = prefix_join(artifact_path, "<terminal-hf-export>")
        else:
            checkpoints = discover_hf_checkpoints(artifact_path)
            if not checkpoints:
                raise ValueError(f"SFT artifact has no HF export: {artifact_path}")
            uri = checkpoints[-1]
        return ResolvedModelLocator(
            uri=uri,
            identity=artifact_identity(self.step),
            local_path=_artifact_local_path("models", self.step),
            tokenizer_uri=self.tokenizer_uri,
            tokenizer_revision=self.tokenizer_revision,
        )


def resolve_skyrl_policy(ctx: StepContext, source: ArtifactStep[SkyRLRun]) -> ResolvedModelLocator:
    """Resolve a terminal policy and tokenizer, using stable fingerprint placeholders."""
    identity = artifact_identity(source)
    if ctx.is_fingerprint:
        uri, tokenizer, revision = f"{identity}/policy", "<skyrl-tokenizer>", "<skyrl-tokenizer-revision>"
    else:
        result = ctx.resolved(source)
        if result.hf_model_uri is None:
            raise ValueError("A completed SkyRL policy requires export_hf=True")
        uri, tokenizer, revision = result.hf_model_uri, result.tokenizer_uri, result.tokenizer_revision
    return ResolvedModelLocator(
        uri=uri,
        identity=identity,
        local_path=_artifact_local_path("models", source),
        tokenizer_uri=tokenizer,
        tokenizer_revision=revision,
    )


@dataclass(frozen=True)
class SkyRLPolicyModel:
    """The terminal HF policy export and tokenizer of a completed SkyRL run."""

    step: ArtifactStep[SkyRLRun]

    def deps(self) -> tuple[ArtifactStep, ...]:
        return (self.step,)

    def resolve(self, ctx: StepContext) -> ResolvedModelLocator:
        return resolve_skyrl_policy(ctx, self.step)


type SkyRLModelSource = ArtifactHfModel | SkyRLPolicyModel


@dataclass(frozen=True)
class ArtifactDataSource:
    """An immutable data directory produced by another Marin artifact step."""

    step: ArtifactStep[Artifact]
    relative_path: str = ""

    def deps(self) -> tuple[ArtifactStep, ...]:
        return (self.step,)

    def resolve(self, ctx: StepContext) -> ResolvedDirectoryDataSource:
        artifact_path = ctx.artifact_path(self.step)
        return ResolvedDirectoryDataSource(
            uri=artifact_path,
            identity=artifact_identity(self.step),
            local_path=_artifact_local_path("data", self.step),
            relative_path=self.relative_path,
        )


@dataclass(frozen=True)
class TaskTroveDataSource:
    """A metadata-selected cohort from an explicitly named packed Harbor file."""

    step: ArtifactStep[Artifact]
    selection: TaskTroveSelection
    relative_path: str
    manifest_path: str = "manifest.json"

    def __post_init__(self) -> None:
        _validate_relative_file_path("relative_path", self.relative_path)
        _validate_relative_file_path("manifest_path", self.manifest_path)

    def deps(self) -> tuple[ArtifactStep, ...]:
        return (self.step,)

    def resolve(self, ctx: StepContext) -> ResolvedTaskTroveDataSource:
        artifact_path = ctx.artifact_path(self.step)
        verifier_ref = "<tasktrove-verifier-ref>"
        if not ctx.is_fingerprint:
            with fsspec.open(prefix_join(artifact_path, self.manifest_path), "r") as manifest_file:
                manifest = json.load(manifest_file)
            verifier_ref = manifest.get("verify_tool_ref")
            if not isinstance(verifier_ref, str) or not verifier_ref:
                raise ValueError("TaskTrove manifest verify_tool_ref must be a non-empty string")
        return ResolvedTaskTroveDataSource(
            uri=prefix_join(artifact_path, self.relative_path),
            identity=f"{artifact_identity(self.step)}/{self.relative_path}",
            local_path=_artifact_local_path("data", self.step),
            relative_path=PurePosixPath(self.relative_path).name,
            verifier_ref=verifier_ref,
            selection=self.selection,
        )


type SkyRLDataSource = ArtifactDataSource | TaskTroveDataSource


@dataclass(frozen=True)
class SkyRLSpec:
    """Backend-neutral, identity-bearing SkyRL experiment definition."""

    name: str
    version: str
    config_yaml: str
    model: SkyRLModelSource
    train_data: tuple[SkyRLDataSource, ...]
    validation_data: tuple[SkyRLDataSource, ...]
    topology: SkyRLTopology
    retention: SkyRLRetentionPolicy
    seed: int
    evaluation_data: tuple[tuple[str, tuple[SkyRLDataSource, ...]], ...] = ()

    def __post_init__(self) -> None:
        _validate_skyrl_recipe(self.config_yaml, self.topology)
        if not self.evaluation_data:
            return
        names = [name for name, _ in self.evaluation_data]
        if len(names) != len(set(names)):
            raise ValueError("duplicate evaluation data names")


@dataclass(frozen=True)
class IrisSkyRLExecution:
    """Runtime-only Iris placement and retry policy."""

    cluster: str
    cluster_config: str
    cpu: float
    memory: str
    disk: str
    priority: str
    max_retries: int
    target_cluster: str | None
    parent_cluster_config: str | None
    coordinator_timeout_hours: int
    wandb_entity: str | None = None
    job_timeout_seconds: int = 0
    required_env_vars: tuple[str, ...] = ()
    container_profile: job_pb2.ContainerProfile | None = None

    def __post_init__(self) -> None:
        if self.coordinator_timeout_hours <= 0:
            raise ValueError("SkyRL coordinator_timeout_hours must be positive")
        if self.job_timeout_seconds < 0:
            raise ValueError("SkyRL job_timeout_seconds cannot be negative")
        if (self.target_cluster is None) != (self.parent_cluster_config is None):
            raise ValueError("SkyRL target_cluster and parent_cluster_config must be set together")
        if self.target_cluster is not None and self.target_cluster != self.cluster:
            raise ValueError("SkyRL target_cluster must match the execution cluster")


_FINGERPRINT_EXECUTION = IrisSkyRLExecution(
    cluster="<runtime>",
    cluster_config="<runtime>",
    cpu=0.0,
    memory="<runtime>",
    disk="<runtime>",
    priority="<runtime>",
    max_retries=0,
    target_cluster=None,
    parent_cluster_config=None,
    coordinator_timeout_hours=1,
)


@dataclass(frozen=True)
class SkyRLOutputPaths:
    checkpoint_root: str
    export_root: str
    attempts_root: str
    resolved_config_uri: str
    terminal_manifest_uri: str


MISSING_CONFIG_VALUE = object()


def _parsed_config(config_yaml: str) -> dict[str, object]:
    try:
        config = yaml.safe_load(config_yaml)
    except yaml.YAMLError as exc:
        raise ValueError(f"SkyRL config_yaml is not valid YAML: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("SkyRL config_yaml must contain a mapping at the document root")
    return config


def declared_config_value(config: dict[str, object], dotted_key: str) -> object:
    value: object = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return MISSING_CONFIG_VALUE
        value = value[part]
    return value


def _validate_role_plan_config(config: dict[str, object], role_plan: SkyRLRolePlan | SkyRLGenerationPlan) -> None:
    """Reject recipe values that disagree with Marin's canonical role plan."""
    for dotted_key, expected in role_plan.config_values().items():
        actual = declared_config_value(config, dotted_key)
        if actual is MISSING_CONFIG_VALUE:
            continue
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"SkyRL config {dotted_key}={actual!r} disagrees with the role plan value {expected!r}")


def set_config_value(config: dict[str, object], dotted_key: str, value: object) -> None:
    node = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = node.get(part)
        if child is None:
            child = {}
            node[part] = child
        if not isinstance(child, dict):
            raise ValueError(f"SkyRL config {'.'.join(parts[:-1])} must be a mapping")
        node = child
    node[parts[-1]] = value


def _materialize_role_plan_config(config: dict[str, object], role_plan: SkyRLRolePlan | SkyRLGenerationPlan) -> None:
    """Render the typed Marin role plan into the SkyRL Hydra document."""
    for dotted_key, value in role_plan.config_values().items():
        set_config_value(config, dotted_key, value)


def _validate_skyrl_backend_constraints(
    config: dict[str, object],
    topology: SkyRLTopology,
) -> None:
    """Validate assumptions imposed by Marin's scalar role-plan interface."""
    run_engines_locally = declared_config_value(config, "generator.run_engines_locally")
    if run_engines_locally is MISSING_CONFIG_VALUE:
        raise ValueError("SkyRL config must explicitly set generator.run_engines_locally")
    if run_engines_locally is not True:
        raise ValueError("Marin SkyRL artifact topology requires generator.run_engines_locally=true")
    if not isinstance(declared_config_value(config, "generator.backend"), str):
        raise ValueError("SkyRL config must explicitly set a non-empty generator.backend")

    for key in ("use_conversation_multi_turn", "require_exact_chat_transport"):
        if declared_config_value(config, f"generator.{key}") is not MISSING_CONFIG_VALUE:
            raise ValueError(f"SkyRL task sessions do not support generator.{key}. Remove this setting.")

    use_kl_loss = declared_config_value(config, "trainer.algorithm.use_kl_loss")
    if use_kl_loss is MISSING_CONFIG_VALUE:
        raise ValueError("SkyRL config must explicitly set trainer.algorithm.use_kl_loss")


def _validate_skyrl_recipe(
    config_yaml: str,
    topology: SkyRLTopology,
) -> None:
    """Validate one effective recipe before building an artifact."""
    config = _parsed_config(config_yaml)
    if declared_config_value(config, "trainer.strategy") != "megatron":
        raise ValueError("SkyRL requires trainer.strategy=megatron")
    _validate_role_plan_config(config, topology.role_plan)
    entrypoint = declared_config_value(config, "entrypoint")
    plan = topology.role_plan
    if isinstance(plan, SkyRLRolePlan):
        training_plan: SkyRLRolePlan | None = plan
        geometry = plan
    else:
        training_plan = None
        geometry = plan.geometry
    if declared_config_value(config, "generator.weight_sync_transport") == "expert_block" and geometry.colocate_all:
        plan = geometry
        supported = (
            training_plan is not None
            and training_plan.entrypoint == "standard"
            and plan.policy_num_nodes == 1
            and plan.policy_num_gpus_per_node == _COLOCATED_EXPERT_BLOCK_GPU_COUNT
            and plan.num_inference_engines == 1
            and plan.inference_engine_tensor_parallel_size == 1
            and plan.inference_engine_pipeline_parallel_size == 1
            and plan.inference_engine_data_parallel_size == plan.policy_num_gpus_per_node
            and plan.inference_engine_expert_parallel_size == plan.policy_num_gpus_per_node
            and declared_config_value(config, "trainer.policy.megatron_config.tensor_model_parallel_size") == 1
            and declared_config_value(config, "trainer.policy.megatron_config.pipeline_model_parallel_size") == 1
            and declared_config_value(config, "trainer.policy.megatron_config.expert_tensor_parallel_size") in (1, None)
            and declared_config_value(config, "trainer.policy.megatron_config.expert_model_parallel_size")
            == plan.policy_num_gpus_per_node
            and declared_config_value(config, "trainer.policy.megatron_config.nccl_max_channels")
            in (None, 1, MISSING_CONFIG_VALUE)
            and declared_config_value(config, "generator.backend") == "vllm"
            and declared_config_value(config, "generator.run_engines_locally") is True
            and declared_config_value(config, "generator.weight_sync_backend") == "nccl"
            and declared_config_value(config, "generator.expert_block_sync.verify") is True
            and declared_config_value(config, "generator.engine_init_kwargs.kernel_config.moe_backend") == "triton"
            and declared_config_value(config, "trainer.policy.model.lora.rank") in (0, MISSING_CONFIG_VALUE)
            and declared_config_value(config, "generator.enforce_eager") in (False, MISSING_CONFIG_VALUE)
        )
        if not supported:
            raise ValueError(
                "SkyRL colocated expert_block requires entrypoint=standard, one four-GPU node, policy TP=PP=ETP=1 EP=4, "
                "one local vLLM TP=PP=1 DP=EP=4 TRITON engine, NCCL, verification, and no LoRA or eager mode"
            )
    if entrypoint is not MISSING_CONFIG_VALUE:
        raise ValueError("SkyRL entrypoint is derived from the selected plan")
    _validate_skyrl_backend_constraints(config, topology)
    if training_plan is None:
        return
    plan = training_plan
    # The reference role inherits the policy width, so this one check covers both:
    # the launcher requires a single GPUs-per-node value across all bundles.
    if plan.policy_num_gpus_per_node != topology.gpus_per_node:
        raise ValueError(
            "SkyRL policy_num_gpus_per_node must match the whole-node topology width; "
            f"got {plan.policy_num_gpus_per_node} and {topology.gpus_per_node}"
        )

    critic_path = declared_config_value(config, "trainer.critic.model.path")
    if critic_path is not MISSING_CONFIG_VALUE and critic_path:
        raise ValueError("Marin SkyRL artifact topology does not yet describe a separate critic role")
    forward_batch = declared_config_value(config, "trainer.micro_forward_batch_size_per_gpu")
    if forward_batch is not MISSING_CONFIG_VALUE and forward_batch != plan.micro_train_batch_size_per_gpu:
        raise ValueError("Megatron requires equal micro_forward_batch_size_per_gpu and micro_train_batch_size_per_gpu")
    for role, nodes in (("policy", plan.policy_num_nodes), ("ref", plan.reference_num_nodes or plan.policy_num_nodes)):
        dimensions: dict[str, int] = {}
        for dimension in fields(MegatronGeometry):
            value = declared_config_value(config, f"trainer.{role}.megatron_config.{dimension.name}")
            dimensions[dimension.name] = 1 if value is MISSING_CONFIG_VALUE else cast(int, value)
        etp = declared_config_value(config, f"trainer.{role}.megatron_config.expert_tensor_parallel_size")
        if etp is MISSING_CONFIG_VALUE or etp is None:
            dimensions["expert_tensor_parallel_size"] = dimensions["tensor_model_parallel_size"]
        validate_megatron_geometry(
            MegatronGeometry(**dimensions), role=role, num_gpus=nodes * plan.policy_num_gpus_per_node
        )


@dataclass(frozen=True)
class SkyRLRunConfig:
    launch_config_yaml: str
    run_id: str
    attempt_id: str
    model: ResolvedModelLocator
    output: SkyRLOutputPaths
    export_hf: bool
    draft_checkpoint_root: str | None


class SkyRLRun(Artifact):
    """Terminal result from a SkyRL run."""

    hf_model_uri: str | None
    global_step: int | None
    tokenizer_uri: str
    tokenizer_revision: str
    checkpoint_root: str
    draft_checkpoint_root: str | None
    terminal_manifest_uri: str
    iris_job_id: str


def skyrl_cpu_command(*arguments: str) -> list[str]:
    """Return argv for SkyRL's frozen CPU environment using its declared Python requirement."""
    workspace_root = find_project_root(Path(__file__))
    if workspace_root is None:
        raise RuntimeError("The MarinSkyRL launcher requires a Marin checkout")
    project_root = workspace_root / SKYRL_PROJECT
    uv_version = (project_root / ".uv-version").read_text().strip()
    python_requirement = tomllib.loads((project_root / "pyproject.toml").read_text())["project"]["requires-python"]
    return [
        "uv",
        "tool",
        "run",
        "--from",
        f"uv=={uv_version}",
        "--python",
        python_requirement,
        "--no-python-downloads",
        "uv",
        "run",
        "--project",
        str(project_root),
        "--python",
        python_requirement,
        "--no-python-downloads",
        "--frozen",
        "--extra",
        "cpu",
        "--no-group",
        "dev",
        *arguments,
    ]


def _launcher_command(config_path: str) -> list[str]:
    return skyrl_cpu_command("marinskyrl", "iris", "launch", "--config", config_path)


def _run_launcher(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the launcher, forwarding its live logs and keeping a tail to explain a failure."""
    tail: deque[str] = deque(maxlen=_LAUNCHER_DIAGNOSTIC_LINES)
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        environment.pop(key, None)
    with (
        tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as response,
        subprocess.Popen(
            command, stdout=response, stderr=subprocess.PIPE, text=True, errors="replace", env=environment
        ) as process,
    ):
        try:
            assert process.stderr is not None
            for line in process.stderr:
                sys.stderr.write(line)
                tail.append(line)
            returncode = process.wait()
        except BaseException:
            # Popen.__exit__ waits but never kills, so without this an interrupt orphans the launcher.
            process.kill()
            raise
        response.seek(0)
        return subprocess.CompletedProcess(command, returncode, response.read(), "".join(tail))


def run_skyrl(config: SkyRLRunConfig) -> SkyRLRun:
    """Run the nested SkyRL executable and return its validated result."""
    response: LaunchResult | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8") as launch_file:
            launch_file.write(config.launch_config_yaml)
            launch_file.flush()
            completed = _run_launcher(_launcher_command(launch_file.name))
        if not completed.stdout.strip():
            raise RuntimeError(
                f"MarinSkyRL launcher exited {completed.returncode} without a terminal response:\n"
                f"{completed.stderr.strip() or '(the launcher wrote nothing to stderr)'}"
            )
        response = LaunchResult.model_validate_json(completed.stdout)
        if completed.returncode != 0 or response.state != "succeeded":
            failure = response.failure or f"launcher exited {completed.returncode}"
            raise RuntimeError(f"MarinSkyRL attempt {config.attempt_id} failed: {failure}\n{completed.stderr.strip()}")
        if response.iris_job_id is None:
            raise ValueError("successful MarinSkyRL response requires iris_job_id")
        model = response.model
        if config.export_hf and model is None:
            raise ValueError("successful MarinSkyRL response requires a model when export_hf is enabled")
        result = SkyRLRun(
            path=config.output.terminal_manifest_uri,
            hf_model_uri=model.policy_export_uri if model is not None else None,
            global_step=model.global_step if model is not None else None,
            tokenizer_uri=config.model.tokenizer_uri,
            tokenizer_revision=config.model.tokenizer_revision,
            checkpoint_root=config.output.checkpoint_root,
            draft_checkpoint_root=config.draft_checkpoint_root,
            terminal_manifest_uri=config.output.terminal_manifest_uri,
            iris_job_id=response.iris_job_id,
        )
    except Exception:
        _record_skyrl_run(config, "failed", response)
        raise
    _record_skyrl_run(config, "succeeded", response)
    return result


def _record_skyrl_run(config: SkyRLRunConfig, status: str, response: LaunchResult | None) -> None:
    output = config.output
    record_rollout_run(
        rollout_run_record(
            run_id=config.run_id,
            attempt_id=config.attempt_id,
            run_kind=RolloutRunKind.REINFORCEMENT_LEARNING,
            producer="skyrl",
            status=status,
            rollout_uri=prefix_join(output.attempts_root, _TRAJECTORIES_SUBDIR),
            storage_format="skyrl_trajectory",
            artifact_uri=output.terminal_manifest_uri,
            model=config.model.identity,
            job_id=response.iris_job_id if response is not None else None,
            attributes={
                "checkpoint_root": output.checkpoint_root,
                "trace_jobs_uri": prefix_join(output.attempts_root, _TRACE_JOBS_SUBDIR),
            },
        )
    )


def _launch_data_source(source: ResolvedDataSource) -> dict:
    value = asdict(source)
    if isinstance(source, ResolvedTaskTroveDataSource):
        value["selection"]["tag_match"] = source.selection.tag_match.value
    return value


def _launch_config_yaml(
    spec: SkyRLSpec,
    execution: IrisSkyRLExecution,
    *,
    export_hf: bool,
    run_id: str,
    attempt_id: str,
    model: ResolvedModelLocator,
    train_data: tuple[ResolvedDataSource, ...],
    validation_data: tuple[ResolvedDataSource, ...],
    evaluation_data: tuple[tuple[str, tuple[ResolvedDataSource, ...]], ...],
    output: SkyRLOutputPaths,
    coordinator_iris_job_id: str | None,
) -> str:
    recipe = _parsed_config(spec.config_yaml)
    recipe["entrypoint"] = spec.topology.role_plan.entrypoint
    _materialize_role_plan_config(recipe, spec.topology.role_plan)
    for callback in recipe.get("trainer", {}).get("callbacks") or []:
        if callback["type"] == BEST_CHECKPOINT_CALLBACK:
            callback["initial_model_uri"] = model.uri
    task_env = recipe.get("extra_env", {})
    if not isinstance(task_env, dict):
        raise ValueError("SkyRL extra_env must be a mapping")
    launch = {
        "schema_version": 1,
        "run": {
            "id": run_id,
            "attempt_id": attempt_id,
            "seed": spec.seed,
            "mode": "train",
            "submission": "wait",
            "export_hf": export_hf,
            **({"coordinator_iris_job_id": coordinator_iris_job_id} if coordinator_iris_job_id is not None else {}),
        },
        "runtime": {
            "profile": declared_config_value(recipe, "trainer.strategy"),
            "entrypoint": "",
            "experiments_dir": "/app/experiments",
            "task_env": task_env,
        },
        "iris": {
            "cluster": execution.cluster,
            "cluster_config": execution.cluster_config,
            "job_name": sanitize_job_name(f"{run_id}-{attempt_id}"),
            "wandb_entity": execution.wandb_entity,
            "allocation": {
                "num_nodes": spec.topology.num_nodes,
                "gpus_per_node": spec.topology.gpus_per_node,
                "gpu_variant": spec.topology.gpu_variant,
                "cpu": execution.cpu,
                "memory": execution.memory,
                "disk": execution.disk,
            },
            "priority": execution.priority,
            "max_retries": execution.max_retries,
            "timeout": execution.job_timeout_seconds,
            "target_cluster": execution.target_cluster,
            "parent_cluster_config": execution.parent_cluster_config,
        },
        "ray": {
            "spill_backend": "local",
            "spill_dir": "/tmp/skyrl-ray-spill",
            "rendezvous_dir": prefix_join(output.attempts_root, "rendezvous"),
            "log_dir": prefix_join(output.attempts_root, "ray-logs"),
            "rendezvous_timeout": 1800,
            "cluster_join_timeout": 1800,
            "driver_liveness_timeout": 9000,
        },
        "artifacts": {
            **asdict(output),
            "resume_checkpoint_count": spec.retention.resume_checkpoint_count,
        },
        "inputs": {
            "model": {**asdict(model), "chat_template": None},
            "data_kind": (
                declared_config_value(recipe, "data.kind")
                if declared_config_value(recipe, "data.kind") is not MISSING_CONFIG_VALUE
                else "tasks"
            ),
            "train_data": [_launch_data_source(source) for source in train_data],
            "validation_data": [_launch_data_source(source) for source in validation_data],
        },
        "skyrl": recipe,
    }
    if evaluation_data:
        launch["inputs"]["evaluation_data"] = {
            name: [_launch_data_source(source) for source in sources] for name, sources in evaluation_data
        }
    if execution.container_profile is not None:
        launch["iris"]["container_profile"] = job_pb2.ContainerProfile.Name(execution.container_profile)
    return yaml.safe_dump(launch, sort_keys=False)


def skyrl_step(
    spec: SkyRLSpec,
    execution: IrisSkyRLExecution,
    *,
    export_hf: bool = False,
) -> ArtifactStep[SkyRLRun]:
    """Build a versioned SkyRL training artifact."""
    step_name = spec.name
    deps = tuple(
        dict.fromkeys(
            (
                *spec.model.deps(),
                *(dep for source in spec.train_data for dep in source.deps()),
                *(dep for source in spec.validation_data for dep in source.deps()),
                *(dep for _, sources in spec.evaluation_data for source in sources for dep in source.deps()),
            )
        )
    )

    def build_config(ctx: StepContext) -> SkyRLRunConfig:
        attempt_id = "<attempt_id>" if ctx.is_fingerprint else uuid.uuid4().hex[:12]
        if ctx.is_fingerprint:
            temporary_root = "<temporary_output_path>"
        else:
            temporary_root = skyrl_temporary_run_path(
                ctx.output_path,
                ttl_days=spec.retention.temporary_storage_ttl_days,
            )
        attempts_root = prefix_join(temporary_root, "attempts")
        output = SkyRLOutputPaths(
            checkpoint_root=prefix_join(temporary_root, "checkpoints"),
            export_root=prefix_join(ctx.output_path, EXPORTS_SUBDIRECTORY),
            attempts_root=attempts_root,
            resolved_config_uri=prefix_join(ctx.output_path, RESOLVED_LAUNCH_FILENAME),
            terminal_manifest_uri=prefix_join(ctx.output_path, TERMINAL_RECEIPT_FILENAME),
        )
        draft_training = declared_config_value(
            _parsed_config(spec.config_yaml), "generator.speculative_decoding.training"
        )
        draft_checkpoint_root = (
            prefix_join(output.checkpoint_root, "drafts")
            if draft_training is not MISSING_CONFIG_VALUE and draft_training is not None
            else None
        )
        run_id = f"{step_name}-{spec.version}"
        model = spec.model.resolve(ctx)
        train_data = tuple(source.resolve(ctx) for source in spec.train_data)
        validation_data = tuple(source.resolve(ctx) for source in spec.validation_data)
        evaluation_data = tuple(
            (name, tuple(source.resolve(ctx) for source in sources)) for name, sources in spec.evaluation_data
        )
        execution = (
            _FINGERPRINT_EXECUTION if ctx.is_fingerprint else cast(IrisSkyRLExecution, ctx.runtime_arg(_EXECUTION))
        )
        coordinator = None if ctx.is_fingerprint else get_job_info()
        return SkyRLRunConfig(
            launch_config_yaml=_launch_config_yaml(
                spec,
                execution,
                export_hf=export_hf,
                run_id=run_id,
                attempt_id=attempt_id,
                model=model,
                train_data=train_data,
                validation_data=validation_data,
                evaluation_data=evaluation_data,
                output=output,
                coordinator_iris_job_id=str(coordinator.job_id) if coordinator is not None else None,
            ),
            run_id=run_id,
            attempt_id=attempt_id,
            model=model,
            output=output,
            export_hf=export_hf,
            draft_checkpoint_root=draft_checkpoint_root,
        )

    return ArtifactStep(
        name=step_name,
        version=spec.version,
        artifact_type=SkyRLRun,
        run=run_skyrl,
        build_config=build_config,
        deps=deps,
        runtime_args={_EXECUTION: execution},
    )


class SkyRLReceiptLookupError(ValueError):
    """The recorded Iris job cannot be identified uniquely."""


def read_skyrl_launch_receipt(run_output: str) -> AttemptRecord | None:
    """Read a launch's terminal or attempt receipt, or build one from its Iris job's state.

    Returns None when the output has no resolved launch. Without a receipt file, the job state comes from the Iris
    controller and may be running. Raises SkyRLReceiptLookupError when the job name matches no job or several.
    """
    record = read_stored_skyrl_receipt(run_output)
    if record is not None and record.native.result.state in {LaunchState.SUCCEEDED, LaunchState.FAILED}:
        return record
    if record is not None:
        config = record.native.config
    else:
        try:
            config = read_skyrl_launch_config(run_output)
        except FileNotFoundError:
            return None
    name = config["iris"]["job_name"]
    info = get_job_info()
    with ExitStack() as stack:
        if info is not None and info.controller_address:
            address, credentials = info.controller_address, info.credentials
        else:
            root = find_project_root(Path(__file__))
            if root is None:
                raise RuntimeError("Iris receipt discovery requires a Marin checkout")
            route = config["iris"]["parent_cluster_config"] or config["iris"]["cluster_config"]
            endpoint = stack.enter_context(connect_controller(config_file=root / route))
            address, credentials = endpoint.url, endpoint.credentials
        client = stack.enter_context(rpc_client(address, credentials))
        response = client.list_jobs(
            controller_pb2.Controller.ListJobsRequest(
                query=controller_pb2.Controller.JobQuery(name_filter=name, limit=500)
            )
        )
        jobs = response.jobs
    matches = [job for job in jobs if str(JobName.from_wire(job.job_id)).rsplit("/", 1)[-1] == name]
    if len(matches) != 1:
        raise SkyRLReceiptLookupError(f"cannot identify the Iris job {name}; its state is unavailable")
    job = matches[0]
    state = LaunchState.SUBMITTED
    if is_job_finished(job.state):
        state = LaunchState.SUCCEEDED if job.state == job_pb2.JOB_STATE_SUCCEEDED else LaunchState.FAILED
    receipt = LaunchReceipt.model_validate(
        {
            "config": config,
            "result": {
                "state": state,
                "iris_job_id": str(JobName.from_wire(job.job_id)),
                "model": None,
            },
        }
    )
    if record is not None:
        return record.model_copy(update={"config": config, "result": receipt.result})
    return AttemptRecord(
        run_id=config["run"]["id"], attempt_id=config["run"]["attempt_id"], config=config, result=receipt.result
    )
