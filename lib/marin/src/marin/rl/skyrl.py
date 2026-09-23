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
    target_cluster: str | None = None
    parent_cluster_config: str | None = None
    wandb_entity: str | None = None


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


def _effective_strategy(config_yaml: str, overrides: tuple[str, ...]) -> str | None:
    """Return the trainer strategy the launched run will use, or None when nothing names one.

    MarinSkyRL applies overrides as Hydra arguments after the config, so the last override naming
    `trainer.strategy` wins. `trainer:` with nothing under it names no strategy.
    """
    for override in reversed(overrides):
        key, separator, value = override.lstrip("+").partition("=")
        if separator and key == "trainer.strategy":
            return value.strip("'\"")
    declared = yaml.safe_load(config_yaml)
    if not isinstance(declared, dict):
        return None
    trainer = declared.get("trainer")
    return trainer.get("strategy") if isinstance(trainer, dict) else None


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
    seed: int
    overrides: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject a runtime profile that does not install the strategy the config asks for."""
        strategy = _effective_strategy(self.config_yaml, self.overrides)
        expected = _STRATEGY_FOR_PROFILE.get(self.runtime.profile)
        if strategy is not None and expected is not None and strategy != expected:
            raise ValueError(
                f"runtime profile {self.runtime.profile.value!r} installs the {expected!r} backend, "
                f"but config_yaml asks for trainer.strategy={strategy!r}"
            )


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


def run_skyrl(config: SkyRLRunConfig) -> SkyRLModel:
    """Run the pinned external launcher and return its validated model value."""
    envelope = {
        "request": asdict(config.request),
        "execution": {
            **asdict(config.execution),
            "job_name": sanitize_job_name(f"{config.request.run_id}-{config.request.attempt_id}"),
        },
    }
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
