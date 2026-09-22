# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin ArtifactStep adapter for the external MarinSkyRL trainer."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal, cast

import fsspec
import yaml
from iris.cluster.client.job_info import get_job_info
from pydantic import BaseModel
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import StoragePath, prefix_join

from marin.evaluation.model_config import ModelConfig
from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import sanitize_job_name
from marin.external_dependencies import MARIN_SKYRL
from marin.rollouts.catalog import RolloutRunKind, record_rollout_run, rollout_run_record
from marin.training.training import LevanterCheckpoint

_EXECUTION = "skyrl_execution"
_LAUNCHER_PYTHON = "3.12"
_MARINSKYRL_STAGING_ROOT = PurePosixPath("/tmp/marinskyrl")
_TEMPORARY_OUTPUT_PREFIX = "skyrl"
_TRACE_JOBS_SUBDIR = "trace_jobs"
_TRAJECTORIES_SUBDIR = "trajectories"
_LAUNCHER_DIAGNOSTIC_LINES = 20
SKYRL_POLICY_LOCATION = "<skyrl-policy>"
SKYRL_TEMPORARY_STORAGE_TTL_DAYS = 14
IRIS_HUB_CLUSTER_CONFIG = "lib/iris/config/marin.yaml"


def skyrl_temporary_run_path(output_path: str, *, ttl_days: int) -> str:
    """Return the lifecycle-managed storage path for a SkyRL run."""
    temporary_root = marin_temp_bucket(ttl_days=ttl_days, source_prefix=output_path)
    return str(StoragePath(temporary_root) / _TEMPORARY_OUTPUT_PREFIX / StoragePath(output_path).key)


class SkyRLRuntimeProfile(StrEnum):
    """Frozen upstream dependency set for a SkyRL training strategy."""

    FSDP = "fsdp"
    MEGATRON = "megatron"


@dataclass(frozen=True)
class SkyRLRuntime:
    """Identity-bearing SkyRL revision and locked dependency profile."""

    profile: SkyRLRuntimeProfile
    commit: str = field(init=False, default=MARIN_SKYRL.commit)


@dataclass(frozen=True)
class SkyRLRolePlan:
    """Explicit policy and rollout settings that bear experiment identity."""

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


@dataclass(frozen=True)
class SkyRLTopology:
    """Logical resource plan that may change training semantics."""

    num_nodes: int
    gpus_per_node: int
    gpu_variant: str
    role_plan: SkyRLRolePlan

    def __post_init__(self) -> None:
        """Reject role geometry that does not exactly consume the requested GPUs."""
        if self.num_nodes <= 0 or self.gpus_per_node <= 0:
            raise ValueError("SkyRL topology node and GPU counts must be positive")

        plan = self.role_plan
        positive_fields = (
            "policy_num_nodes",
            "policy_num_gpus_per_node",
            "num_inference_engines",
            "inference_engine_tensor_parallel_size",
            "inference_engine_pipeline_parallel_size",
            "inference_engine_data_parallel_size",
            "inference_engine_expert_parallel_size",
            "train_batch_size",
            "policy_mini_batch_size",
            "micro_train_batch_size_per_gpu",
            "n_samples_per_prompt",
        )
        for field_name in positive_fields:
            if getattr(plan, field_name) <= 0:
                raise ValueError(f"SkyRL role plan {field_name} must be positive")

        if plan.policy_num_nodes > self.num_nodes:
            raise ValueError("SkyRL policy_num_nodes exceeds the allocated topology")
        if plan.policy_num_gpus_per_node > self.gpus_per_node:
            raise ValueError("SkyRL policy_num_gpus_per_node exceeds the GPUs on one allocated node")

        tensor_pipeline_gpus = plan.inference_engine_tensor_parallel_size * plan.inference_engine_pipeline_parallel_size
        engine_gpus = tensor_pipeline_gpus * plan.inference_engine_data_parallel_size
        if plan.colocate_all and self.gpus_per_node % tensor_pipeline_gpus:
            raise ValueError(
                "each colocated SkyRL inference engine TP*PP slice must divide gpus_per_node; "
                f"got TP*PP={tensor_pipeline_gpus} and gpus_per_node={self.gpus_per_node}"
            )
        if not plan.colocate_all and engine_gpus > self.gpus_per_node:
            raise ValueError(
                "each SkyRL inference engine must fit on one node, but "
                f"TP*PP*DP={engine_gpus} exceeds gpus_per_node={self.gpus_per_node}"
            )
        expert_group_gpus = plan.inference_engine_tensor_parallel_size * plan.inference_engine_data_parallel_size
        if expert_group_gpus % plan.inference_engine_expert_parallel_size:
            raise ValueError(
                "SkyRL inference engine TP*DP must be divisible by expert parallel size; "
                f"got {expert_group_gpus} and EP={plan.inference_engine_expert_parallel_size}"
            )

        policy_gpus = plan.policy_num_nodes * plan.policy_num_gpus_per_node
        rollout_gpus = plan.num_inference_engines * tensor_pipeline_gpus * plan.inference_engine_data_parallel_size
        if plan.colocate_all and policy_gpus != rollout_gpus:
            raise ValueError(
                "colocated SkyRL roles must use the same GPUs: "
                f"policy={policy_gpus}, rollout={rollout_gpus} "
                f"({plan.num_inference_engines} engines x TP{plan.inference_engine_tensor_parallel_size} "
                f"x PP{plan.inference_engine_pipeline_parallel_size} x DP{plan.inference_engine_data_parallel_size})"
            )
        planned_gpus = policy_gpus if plan.colocate_all else policy_gpus + rollout_gpus
        allocated_gpus = self.num_nodes * self.gpus_per_node
        if planned_gpus != allocated_gpus:
            placement = "colocated policy/rollout" if plan.colocate_all else "policy + rollout"
            raise ValueError(
                "SkyRL role plan does not consume the allocated topology: "
                f"{placement}={planned_gpus} GPUs, topology={allocated_gpus} GPUs "
                f"({self.num_nodes} nodes x {self.gpus_per_node}); rollout uses "
                f"{plan.num_inference_engines} engines x TP{plan.inference_engine_tensor_parallel_size} "
                f"x PP{plan.inference_engine_pipeline_parallel_size} x DP{plan.inference_engine_data_parallel_size}"
            )

        if plan.train_batch_size % plan.policy_mini_batch_size:
            raise ValueError("SkyRL train_batch_size must be divisible by policy_mini_batch_size")
        if plan.policy_mini_batch_size % plan.micro_train_batch_size_per_gpu:
            raise ValueError("SkyRL policy_mini_batch_size must be divisible by micro_train_batch_size_per_gpu")


@dataclass(frozen=True)
class _SkyRLModelRoleClaim:
    role_id: str
    kind: str
    execution: str
    backend: str
    colocation_group: str
    num_nodes: int
    gpus_per_node: int
    replicas: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    expert_parallel_size: int


@dataclass(frozen=True)
class _SkyRLRoleBundle:
    name: str
    role_ids: tuple[str, ...]
    num_nodes: int
    gpus_per_node: int


@dataclass(frozen=True)
class _MarinSkyRLRolePlan:
    claims: tuple[_SkyRLModelRoleClaim, ...]
    bundles: tuple[_SkyRLRoleBundle, ...]
    train_batch_size: int
    policy_mini_batch_size: int
    micro_train_batch_size_per_gpu: int
    n_samples_per_prompt: int


@dataclass(frozen=True)
class _MarinSkyRLProtocolConfig:
    strategy: str
    rollout_backend: str
    use_reference: bool


@dataclass(frozen=True)
class _MarinSkyRLTopology:
    num_nodes: int
    gpus_per_node: int
    gpu_variant: str
    role_plan: _MarinSkyRLRolePlan


@dataclass(frozen=True)
class _MarinSkyRLLaunchRequest:
    run_id: str
    attempt_id: str
    config_yaml: str
    runtime: SkyRLRuntime
    model: ResolvedModelLocator
    train_data: tuple[ResolvedDataSource, ...]
    validation_data: tuple[ResolvedDataSource, ...]
    topology: _MarinSkyRLTopology
    output: SkyRLOutputPaths
    export_hf: bool
    seed: int
    overrides: tuple[str, ...]


@dataclass(frozen=True)
class _MarinSkyRLIrisExecution:
    cluster: str
    cluster_config: str
    cpu: float
    memory: str
    disk: str
    target_cluster: str | None
    parent_cluster_config: str | None
    priority: str
    max_retries: int
    job_name: str
    wandb_entity: str | None


@dataclass(frozen=True)
class _MarinSkyRLJobSpec:
    request: _MarinSkyRLLaunchRequest
    execution: _MarinSkyRLIrisExecution


@dataclass(frozen=True)
class SkyRLRetentionPolicy:
    """Temporary storage lifetime and rolling resume depth for one SkyRL run.

    Every successful run produces one durable canonical export from its terminal
    checkpoint.
    """

    resume_checkpoint_count: int = 2
    temporary_storage_ttl_days: int = SKYRL_TEMPORARY_STORAGE_TTL_DAYS

    def __post_init__(self) -> None:
        if not 1 <= self.resume_checkpoint_count <= 5:
            raise ValueError("SkyRL resume_checkpoint_count must be between one and five")
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


def _artifact_identity(step: ArtifactStep) -> str:
    return f"{step.name}@{step.version}:{step.fingerprint()}"


def _artifact_local_path(category: str, step: ArtifactStep) -> str:
    return str(_MARINSKYRL_STAGING_ROOT / category / PurePosixPath(step.name).name)


def _validate_relative_file_path(name: str, value: str) -> None:
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"TaskTrove {name} must identify a file below the release root: {value!r}")


@dataclass(frozen=True)
class ArtifactHfModel:
    """An exact HF export produced by another Marin artifact step."""

    step: ArtifactStep[LevanterCheckpoint]
    tokenizer_uri: str
    tokenizer_revision: str
    relative_path: str | None = None

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
            identity=_artifact_identity(self.step),
            local_path=_artifact_local_path("models", self.step),
            tokenizer_uri=self.tokenizer_uri,
            tokenizer_revision=self.tokenizer_revision,
        )


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
            identity=_artifact_identity(self.step),
            local_path=_artifact_local_path("data", self.step),
            relative_path=self.relative_path,
        )


@dataclass(frozen=True)
class TaskTroveDataSource:
    """A metadata-selected cohort from the compatibility RL view of a TaskTrove release."""

    step: ArtifactStep[Artifact]
    selection: TaskTroveSelection
    relative_path: str = "tasks/part-00000.parquet"
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
            identity=f"{_artifact_identity(self.step)}/{self.relative_path}",
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
    runtime: SkyRLRuntime
    model: ArtifactHfModel
    train_data: tuple[SkyRLDataSource, ...]
    validation_data: tuple[SkyRLDataSource, ...]
    topology: SkyRLTopology
    retention: SkyRLRetentionPolicy
    seed: int
    overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the complete launch recipe before an artifact can be submitted."""
        _validate_skyrl_recipe(self.config_yaml, self.overrides, self.runtime, self.topology)


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

    def __post_init__(self) -> None:
        if self.coordinator_timeout_hours <= 0:
            raise ValueError("SkyRL coordinator_timeout_hours must be positive")
        if (self.target_cluster is None) != (self.parent_cluster_config is None):
            raise ValueError("SkyRL target_cluster and parent_cluster_config must be set together")
        if self.target_cluster is not None and self.target_cluster != self.cluster:
            raise ValueError("SkyRL target_cluster must match the execution cluster")


@dataclass(frozen=True)
class SkyRLOutputPaths:
    checkpoint_root: str
    export_root: str
    attempts_root: str
    resolved_config_uri: str
    terminal_manifest_uri: str


# The trainer strategy each runtime profile installs the closure for. A profile decides which
# dependencies reach the pod; `trainer.strategy` decides which backend the trainer then asks for.
# Nothing downstream reconciles them, so a mismatch installs one backend and runs another.
_STRATEGY_FOR_PROFILE = {
    SkyRLRuntimeProfile.FSDP: "fsdp2",
    SkyRLRuntimeProfile.MEGATRON: "megatron",
}

_ROLE_PLAN_CONFIG_FIELDS = {
    "trainer.placement.colocate_all": "colocate_all",
    "trainer.train_batch_size": "train_batch_size",
    "trainer.policy_mini_batch_size": "policy_mini_batch_size",
    "trainer.micro_train_batch_size_per_gpu": "micro_train_batch_size_per_gpu",
    "generator.num_inference_engines": "num_inference_engines",
    "generator.inference_engine_tensor_parallel_size": "inference_engine_tensor_parallel_size",
    "generator.inference_engine_pipeline_parallel_size": "inference_engine_pipeline_parallel_size",
    "generator.inference_engine_data_parallel_size": "inference_engine_data_parallel_size",
    "generator.inference_engine_expert_parallel_size": "inference_engine_expert_parallel_size",
    "generator.n_samples_per_prompt": "n_samples_per_prompt",
}
_MISSING_CONFIG_VALUE = object()


def _parsed_config(config_yaml: str) -> dict[str, object]:
    try:
        config = yaml.safe_load(config_yaml)
    except yaml.YAMLError as exc:
        raise ValueError(f"SkyRL config_yaml is not valid YAML: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("SkyRL config_yaml must contain a mapping at the document root")
    return config


def _declared_config_value(config: dict[str, object], dotted_key: str) -> object:
    value: object = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING_CONFIG_VALUE
        value = value[part]
    return value


def _effective_config_value(config: dict[str, object], overrides: tuple[str, ...], dotted_key: str) -> object:
    for override in reversed(overrides):
        key, separator, value = override.lstrip("+").partition("=")
        if separator and key == dotted_key:
            try:
                return yaml.safe_load(value)
            except yaml.YAMLError as exc:
                raise ValueError(f"SkyRL override for {dotted_key} is invalid YAML: {value!r}") from exc
    return _declared_config_value(config, dotted_key)


def _validate_role_plan_config(config: dict[str, object], overrides: tuple[str, ...], role_plan: SkyRLRolePlan) -> None:
    """Ensure the trainer config cannot silently disagree with the identity-bearing role plan."""
    for dotted_key, field_name in _ROLE_PLAN_CONFIG_FIELDS.items():
        expected = getattr(role_plan, field_name)
        actual = _effective_config_value(config, overrides, dotted_key)
        if actual is _MISSING_CONFIG_VALUE:
            raise ValueError(f"SkyRL config must explicitly set {dotted_key} from role_plan.{field_name}")
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"SkyRL config {dotted_key}={actual!r} disagrees with role_plan.{field_name}={expected!r}")


def _validate_entrypoint_config(config: dict[str, object], overrides: tuple[str, ...], role_plan: SkyRLRolePlan) -> None:
    """Reject entrypoint-specific constraints that MarinSkyRL would otherwise discover at startup."""
    entrypoint = _effective_config_value(config, overrides, "entrypoint")
    if entrypoint == "fully_async" and role_plan.train_batch_size != role_plan.policy_mini_batch_size:
        raise ValueError(
            "SkyRL fully_async entrypoint requires train_batch_size == policy_mini_batch_size; "
            f"got {role_plan.train_batch_size} and {role_plan.policy_mini_batch_size}"
        )


def _effective_strategy(config: dict[str, object], overrides: tuple[str, ...]) -> str | None:
    """Return the trainer strategy the launched run will use, or None when nothing names one.

    MarinSkyRL applies overrides as Hydra arguments after the config, so the last override naming
    `trainer.strategy` wins. `trainer:` with nothing under it names no strategy.
    """
    for override in reversed(overrides):
        key, separator, value = override.lstrip("+").partition("=")
        if separator and key == "trainer.strategy":
            return value.strip("'\"")
    trainer = config.get("trainer")
    return trainer.get("strategy") if isinstance(trainer, dict) else None


def _validate_runtime_strategy(config: dict[str, object], overrides: tuple[str, ...], runtime: SkyRLRuntime) -> None:
    strategy = _effective_strategy(config, overrides)
    expected = _STRATEGY_FOR_PROFILE.get(runtime.profile)
    if strategy is not None and expected is not None and strategy != expected:
        raise ValueError(
            f"runtime profile {runtime.profile.value!r} installs the {expected!r} backend, "
            f"but config_yaml asks for trainer.strategy={strategy!r}"
        )


def _marinskyrl_protocol_config(
    config: dict[str, object],
    overrides: tuple[str, ...],
    runtime: SkyRLRuntime,
    topology: SkyRLTopology,
) -> _MarinSkyRLProtocolConfig:
    """Validate assumptions imposed by Marin's scalar role-plan interface."""
    plan = topology.role_plan
    if plan.policy_num_gpus_per_node != topology.gpus_per_node:
        raise ValueError(
            "SkyRL policy_num_gpus_per_node must match the whole-node topology width; "
            f"got {plan.policy_num_gpus_per_node} and {topology.gpus_per_node}"
        )

    run_engines_locally = _effective_config_value(config, overrides, "generator.run_engines_locally")
    if run_engines_locally is _MISSING_CONFIG_VALUE:
        raise ValueError("SkyRL config must explicitly set generator.run_engines_locally")
    if run_engines_locally is not True:
        raise ValueError("Marin SkyRL artifact topology requires generator.run_engines_locally=true")
    rollout_backend = _effective_config_value(config, overrides, "generator.backend")
    if not isinstance(rollout_backend, str) or not rollout_backend:
        raise ValueError("SkyRL config must explicitly set a non-empty generator.backend")

    use_kl_loss = _effective_config_value(config, overrides, "trainer.algorithm.use_kl_loss")
    if use_kl_loss is _MISSING_CONFIG_VALUE:
        raise ValueError("SkyRL config must explicitly set trainer.algorithm.use_kl_loss")
    use_kl_in_reward = _effective_config_value(config, overrides, "trainer.algorithm.use_kl_in_reward")
    use_reference = bool(use_kl_loss) or (use_kl_in_reward is not _MISSING_CONFIG_VALUE and bool(use_kl_in_reward))
    critic_path = _effective_config_value(config, overrides, "trainer.critic.model.path")
    if critic_path is not _MISSING_CONFIG_VALUE and critic_path:
        raise ValueError("Marin SkyRL artifact topology does not yet describe a separate critic role")

    if use_reference:
        colocate_policy_ref = _effective_config_value(config, overrides, "trainer.placement.colocate_policy_ref")
        if colocate_policy_ref is not _MISSING_CONFIG_VALUE and colocate_policy_ref is not True:
            raise ValueError("Marin SkyRL artifact topology requires policy and reference roles to be colocated")
        ref_num_nodes = _effective_config_value(config, overrides, "trainer.placement.ref_num_nodes")
        ref_num_gpus = _effective_config_value(config, overrides, "trainer.placement.ref_num_gpus_per_node")
        ref_num_nodes = plan.policy_num_nodes if ref_num_nodes in (_MISSING_CONFIG_VALUE, None) else ref_num_nodes
        ref_num_gpus = plan.policy_num_gpus_per_node if ref_num_gpus in (_MISSING_CONFIG_VALUE, None) else ref_num_gpus
        if (ref_num_nodes, ref_num_gpus) != (plan.policy_num_nodes, plan.policy_num_gpus_per_node):
            raise ValueError("Marin SkyRL artifact topology requires policy and reference roles to share one footprint")

    return _MarinSkyRLProtocolConfig(
        strategy=_effective_strategy(config, overrides) or _STRATEGY_FOR_PROFILE[runtime.profile],
        rollout_backend=rollout_backend,
        use_reference=use_reference,
    )


def _marinskyrl_role_plan(
    config: dict[str, object],
    overrides: tuple[str, ...],
    runtime: SkyRLRuntime,
    topology: SkyRLTopology,
) -> _MarinSkyRLRolePlan:
    """Compile Marin's validated scalar plan into MarinSkyRL's role protocol."""
    plan = topology.role_plan
    protocol = _marinskyrl_protocol_config(config, overrides, runtime, topology)
    policy_group = "all" if plan.colocate_all else "policy"
    policy_replicas = plan.policy_num_nodes * plan.policy_num_gpus_per_node
    claims = [
        _SkyRLModelRoleClaim(
            role_id="policy",
            kind="policy",
            execution="local",
            backend=protocol.strategy,
            colocation_group=policy_group,
            num_nodes=plan.policy_num_nodes,
            gpus_per_node=topology.gpus_per_node,
            replicas=policy_replicas,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=policy_replicas,
            expert_parallel_size=1,
        )
    ]
    policy_role_ids = ["policy"]
    if protocol.use_reference:
        claims.append(replace(claims[0], role_id="reference", kind="reference"))
        policy_role_ids.append("reference")

    rollout_group = "all" if plan.colocate_all else "rollout"
    rollout_gpus = (
        plan.num_inference_engines
        * plan.inference_engine_tensor_parallel_size
        * plan.inference_engine_pipeline_parallel_size
        * plan.inference_engine_data_parallel_size
    )
    rollout_nodes = plan.policy_num_nodes if plan.colocate_all else rollout_gpus // topology.gpus_per_node
    claims.append(
        _SkyRLModelRoleClaim(
            role_id="rollout",
            kind="rollout",
            execution="local",
            backend=protocol.rollout_backend,
            colocation_group=rollout_group,
            num_nodes=rollout_nodes,
            gpus_per_node=topology.gpus_per_node,
            replicas=plan.num_inference_engines,
            tensor_parallel_size=plan.inference_engine_tensor_parallel_size,
            pipeline_parallel_size=plan.inference_engine_pipeline_parallel_size,
            data_parallel_size=plan.inference_engine_data_parallel_size,
            expert_parallel_size=plan.inference_engine_expert_parallel_size,
        )
    )

    if plan.colocate_all:
        bundles = (
            _SkyRLRoleBundle(
                name="all",
                role_ids=(*policy_role_ids, "rollout"),
                num_nodes=plan.policy_num_nodes,
                gpus_per_node=topology.gpus_per_node,
            ),
        )
    else:
        bundles = (
            _SkyRLRoleBundle(
                name="policy",
                role_ids=tuple(policy_role_ids),
                num_nodes=plan.policy_num_nodes,
                gpus_per_node=topology.gpus_per_node,
            ),
            _SkyRLRoleBundle(
                name="rollout",
                role_ids=("rollout",),
                num_nodes=rollout_nodes,
                gpus_per_node=topology.gpus_per_node,
            ),
        )
    return _MarinSkyRLRolePlan(
        claims=tuple(claims),
        bundles=bundles,
        train_batch_size=plan.train_batch_size,
        policy_mini_batch_size=plan.policy_mini_batch_size,
        micro_train_batch_size_per_gpu=plan.micro_train_batch_size_per_gpu,
        n_samples_per_prompt=plan.n_samples_per_prompt,
    )


def _validate_skyrl_recipe(
    config_yaml: str,
    overrides: tuple[str, ...],
    runtime: SkyRLRuntime,
    topology: SkyRLTopology,
) -> _MarinSkyRLRolePlan:
    """Validate one effective recipe and return its launcher role protocol."""
    config = _parsed_config(config_yaml)
    _validate_runtime_strategy(config, overrides, runtime)
    _validate_role_plan_config(config, overrides, topology.role_plan)
    _validate_entrypoint_config(config, overrides, topology.role_plan)
    return _marinskyrl_role_plan(config, overrides, runtime, topology)


@dataclass(frozen=True)
class SkyRLLaunchRequest:
    run_id: str
    attempt_id: str
    config_yaml: str
    runtime: SkyRLRuntime
    model: ResolvedModelLocator
    train_data: tuple[ResolvedDataSource, ...]
    validation_data: tuple[ResolvedDataSource, ...]
    topology: SkyRLTopology
    output: SkyRLOutputPaths
    export_hf: bool
    seed: int
    overrides: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the complete effective recipe before invoking the external launcher."""
        _validate_skyrl_recipe(self.config_yaml, self.overrides, self.runtime, self.topology)


@dataclass(frozen=True)
class SkyRLRunConfig:
    request: SkyRLLaunchRequest
    execution: IrisSkyRLExecution
    launcher_requirement: str


class SkyRLModel(Artifact):
    """Validated terminal HF policy export from a MarinSkyRL run."""

    policy_export_uri: str
    global_step: int
    tokenizer_uri: str
    tokenizer_revision: str
    checkpoint_root: str
    terminal_manifest_uri: str
    iris_job_id: str


class _SkyRLTerminalModel(BaseModel):
    policy_export_uri: str
    global_step: int
    tokenizer_uri: str
    tokenizer_revision: str
    checkpoint_root: str
    terminal_manifest_uri: str


class _SkyRLLaunchResponse(BaseModel):
    state: str
    iris_job_id: str | None = None
    failure: str | None = None
    model: _SkyRLTerminalModel | None = None


@dataclass(frozen=True)
class SkyRLEvaluationModel:
    """A terminal SkyRL policy adapted to the shared evaluation model contract."""

    step: ArtifactStep[SkyRLModel]
    model: ModelConfig

    def __post_init__(self) -> None:
        if self.model.location != SKYRL_POLICY_LOCATION:
            raise ValueError(f"SkyRL evaluation model location must be {SKYRL_POLICY_LOCATION!r}")
        if self.model.tokenizer is None:
            raise ValueError("SkyRL evaluation models require an explicit Hugging Face tokenizer")

    def deps(self) -> tuple[ArtifactStep, ...]:
        return (self.step,)

    def resolve(self, ctx: StepContext) -> ModelConfig:
        if ctx.is_fingerprint:
            location = f"{_artifact_identity(self.step)}/policy"
            tokenizer = self.model.tokenizer
        else:
            terminal = ctx.resolved(self.step)
            location = terminal.policy_export_uri
            tokenizer = terminal.tokenizer_uri
        return replace(self.model, location=location, tokenizer=tokenizer)


def _launcher_command(requirement: str, request_path: str) -> list[str]:
    return [
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--prerelease=allow",
        "--python",
        _LAUNCHER_PYTHON,
        "--with",
        requirement,
        "marinskyrl",
        "iris",
        "launch",
        "--request",
        request_path,
    ]


def _run_launcher(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the launcher, forwarding its live logs and keeping a tail to explain a failure."""
    tail: deque[str] = deque(maxlen=_LAUNCHER_DIAGNOSTIC_LINES)
    with (
        tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as response,
        subprocess.Popen(command, stdout=response, stderr=subprocess.PIPE, text=True, errors="replace") as process,
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


def _marinskyrl_job_spec(config: SkyRLRunConfig) -> _MarinSkyRLJobSpec:
    request = config.request
    role_plan = _validate_skyrl_recipe(request.config_yaml, request.overrides, request.runtime, request.topology)
    launcher_request = _MarinSkyRLLaunchRequest(
        run_id=request.run_id,
        attempt_id=request.attempt_id,
        config_yaml=request.config_yaml,
        runtime=request.runtime,
        model=request.model,
        train_data=request.train_data,
        validation_data=request.validation_data,
        topology=_MarinSkyRLTopology(
            num_nodes=request.topology.num_nodes,
            gpus_per_node=request.topology.gpus_per_node,
            gpu_variant=request.topology.gpu_variant,
            role_plan=role_plan,
        ),
        output=request.output,
        export_hf=request.export_hf,
        seed=request.seed,
        overrides=request.overrides,
    )

    execution = config.execution
    submit_through_ambient_controller = get_job_info() is not None and execution.target_cluster is not None
    launcher_execution = _MarinSkyRLIrisExecution(
        cluster=execution.cluster,
        cluster_config=execution.cluster_config,
        cpu=execution.cpu,
        memory=execution.memory,
        disk=execution.disk,
        target_cluster=None if submit_through_ambient_controller else execution.target_cluster,
        parent_cluster_config=None if submit_through_ambient_controller else execution.parent_cluster_config,
        priority=execution.priority,
        max_retries=execution.max_retries,
        job_name=sanitize_job_name(f"{request.run_id}-{request.attempt_id}"),
        wandb_entity=execution.wandb_entity,
    )
    return _MarinSkyRLJobSpec(request=launcher_request, execution=launcher_execution)


def run_skyrl(config: SkyRLRunConfig) -> SkyRLModel:
    """Run the pinned external launcher and return its validated model value."""
    envelope = asdict(_marinskyrl_job_spec(config))
    response: _SkyRLLaunchResponse | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as request_file:
            json.dump(envelope, request_file, sort_keys=True)
            request_file.flush()
            completed = _run_launcher(_launcher_command(config.launcher_requirement, request_file.name))
        if not completed.stdout.strip():
            raise RuntimeError(
                f"MarinSkyRL launcher exited {completed.returncode} without a terminal response:\n"
                f"{completed.stderr.strip() or '(the launcher wrote nothing to stderr)'}"
            )
        response = _SkyRLLaunchResponse.model_validate_json(completed.stdout)
        if completed.returncode != 0 or response.state != "succeeded":
            failure = response.failure or f"launcher exited {completed.returncode}"
            raise RuntimeError(
                f"MarinSkyRL attempt {config.request.attempt_id} failed: {failure}\n{completed.stderr.strip()}"
            )
        model = response.model
        if model is None or response.iris_job_id is None:
            raise ValueError("successful MarinSkyRL response requires model and iris_job_id")
        result = SkyRLModel(
            path=config.request.output.terminal_manifest_uri,
            policy_export_uri=model.policy_export_uri,
            global_step=model.global_step,
            tokenizer_uri=model.tokenizer_uri,
            tokenizer_revision=model.tokenizer_revision,
            checkpoint_root=model.checkpoint_root,
            terminal_manifest_uri=model.terminal_manifest_uri,
            iris_job_id=response.iris_job_id,
        )
    except Exception:
        _record_skyrl_run(config, "failed", response)
        raise
    _record_skyrl_run(config, "succeeded", response)
    return result


def _record_skyrl_run(config: SkyRLRunConfig, status: str, response: _SkyRLLaunchResponse | None) -> None:
    output = config.request.output
    record_rollout_run(
        rollout_run_record(
            run_id=config.request.run_id,
            attempt_id=config.request.attempt_id,
            run_kind=RolloutRunKind.REINFORCEMENT_LEARNING,
            producer="skyrl",
            status=status,
            rollout_uri=prefix_join(output.attempts_root, _TRAJECTORIES_SUBDIR),
            storage_format="skyrl_trajectory",
            artifact_uri=output.terminal_manifest_uri,
            model=config.request.model.identity,
            job_id=response.iris_job_id if response is not None else None,
            attributes={
                "checkpoint_root": output.checkpoint_root,
                "trace_jobs_uri": prefix_join(output.attempts_root, _TRACE_JOBS_SUBDIR),
            },
        )
    )


def skyrl_step(spec: SkyRLSpec, execution: IrisSkyRLExecution) -> ArtifactStep[SkyRLModel]:
    """Build a versioned MarinSkyRL training artifact."""
    step_name = spec.name
    deps = tuple(
        dict.fromkeys(
            (
                *spec.model.deps(),
                *(dep for source in spec.train_data for dep in source.deps()),
                *(dep for source in spec.validation_data for dep in source.deps()),
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
            export_root=prefix_join(ctx.output_path, "exports"),
            attempts_root=attempts_root,
            resolved_config_uri=prefix_join(ctx.output_path, "resolved-skyrl.json"),
            terminal_manifest_uri=prefix_join(ctx.output_path, "terminal.json"),
        )
        retention_overrides = (
            f"++trainer.max_ckpts_to_keep={spec.retention.resume_checkpoint_count}",
            f"++terminal_bench_config.trials_dir='{prefix_join(attempts_root, _TRACE_JOBS_SUBDIR)}'",
            f"++generator.trajectory_retention.output_path='{prefix_join(attempts_root, _TRAJECTORIES_SUBDIR)}'",
        )
        request = SkyRLLaunchRequest(
            run_id=f"{step_name}-{spec.version}",
            attempt_id=attempt_id,
            config_yaml=spec.config_yaml,
            runtime=spec.runtime,
            model=spec.model.resolve(ctx),
            train_data=tuple(source.resolve(ctx) for source in spec.train_data),
            validation_data=tuple(source.resolve(ctx) for source in spec.validation_data),
            topology=spec.topology,
            output=output,
            export_hf=True,
            seed=spec.seed,
            overrides=(*spec.overrides, *retention_overrides),
        )
        return SkyRLRunConfig(
            request=request,
            execution=cast(IrisSkyRLExecution, ctx.runtime_arg(_EXECUTION)),
            launcher_requirement=MARIN_SKYRL.requirement(),
        )

    return ArtifactStep(
        name=step_name,
        version=spec.version,
        artifact_type=SkyRLModel,
        run=run_skyrl,
        build_config=build_config,
        deps=deps,
        runtime_args={_EXECUTION: execution},
    )
