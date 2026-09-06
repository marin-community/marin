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
from typing import cast

from marin.evaluation.model_config import ModelConfig
from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import sanitize_job_name
from marin.external_dependencies import MARIN_SKYRL
from marin.training.training import LevanterCheckpoint, temporary_storage_base_path
from rigging.filesystem.storage_path import prefix_join

_EXECUTION = "skyrl_execution"
_LAUNCHER_PYTHON = "3.12"
_MARINSKYRL_STAGING_ROOT = PurePosixPath("/tmp/marinskyrl")
_TEMPORARY_OUTPUT_PREFIX = "skyrl"
_LAUNCHER_DIAGNOSTIC_LINES = 20
_PROTOCOL_SCHEMA_VERSION = 2
SKYRL_POLICY_LOCATION = "<skyrl-policy>"


class SkyRLRuntimeProfile(StrEnum):
    """Frozen upstream dependency set for a SkyRL training strategy."""

    FSDP = "fsdp"
    MEGATRON = "megatron"


class SkyRLCompletionMode(StrEnum):
    """Artifact retained after a successful training process."""

    METRICS = "metrics"
    CHECKPOINT = "checkpoint"


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

    Metrics completion retains no checkpoint. Checkpoint completion retains native
    state under this lifetime; a separate export step produces a durable HF model.
    """

    resume_checkpoint_count: int = 2
    temporary_storage_ttl_days: int = 14

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
class ResolvedDataLocator:
    uri: str
    identity: str
    local_path: str
    relative_path: str


def _artifact_identity(step: ArtifactStep) -> str:
    return f"{step.name}@{step.version}:{step.fingerprint()}"


def _artifact_local_path(category: str, step: ArtifactStep) -> str:
    return str(_MARINSKYRL_STAGING_ROOT / category / PurePosixPath(step.name).name)


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

    def resolve(self, ctx: StepContext) -> ResolvedDataLocator:
        artifact_path = ctx.artifact_path(self.step)
        return ResolvedDataLocator(
            uri=artifact_path,
            identity=_artifact_identity(self.step),
            local_path=_artifact_local_path("data", self.step),
            relative_path=self.relative_path,
        )


@dataclass(frozen=True)
class SkyRLSpec:
    """Backend-neutral, identity-bearing SkyRL experiment definition."""

    name: str
    version: str
    config_yaml: str
    runtime: SkyRLRuntime
    model: ArtifactHfModel
    train_data: tuple[ArtifactDataSource, ...]
    validation_data: tuple[ArtifactDataSource, ...]
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
    timeout_seconds: int = 0

    def __post_init__(self) -> None:
        if self.timeout_seconds < 0:
            raise ValueError("SkyRL timeout_seconds must be nonnegative (0 disables the deadline)")


@dataclass(frozen=True)
class SkyRLOutputPaths:
    checkpoint_root: str
    export_root: str
    attempts_root: str
    resolved_config_uri: str
    terminal_manifest_uri: str


@dataclass(frozen=True)
class SkyRLLaunchRequest:
    run_id: str
    attempt_id: str
    config_yaml: str
    runtime: SkyRLRuntime
    model: ResolvedModelLocator
    train_data: tuple[ResolvedDataLocator, ...]
    validation_data: tuple[ResolvedDataLocator, ...]
    topology: SkyRLTopology
    output: SkyRLOutputPaths
    seed: int
    overrides: tuple[str, ...]
    completion_mode: SkyRLCompletionMode
    checkpoint_retention_days: int | None


@dataclass(frozen=True)
class SkyRLRunConfig:
    request: SkyRLLaunchRequest
    execution: IrisSkyRLExecution
    launcher_requirement: str


@dataclass(frozen=True)
class NativeCheckpointFile:
    path: str
    size: int


class SkyRLTrainingResult(Artifact):
    """Durable proof that SkyRL training completed, without a model claim."""

    global_step: int
    receipt_uri: str
    resolved_config_uri: str
    terminal_manifest_uri: str
    iris_job_id: str


class SkyRLCheckpoint(SkyRLTrainingResult):
    """A native, resumable checkpoint retained under an explicit TTL policy."""

    checkpoint_path: str
    trainer_state_sha256: str
    files: tuple[NativeCheckpointFile, ...]
    checkpoint_retention_days: int
    runtime_commit: str
    runtime_profile: SkyRLRuntimeProfile
    tokenizer_uri: str
    tokenizer_revision: str


@dataclass(frozen=True)
class SkyRLExportOutputPaths:
    export_root: str
    attempts_root: str
    terminal_manifest_uri: str


@dataclass(frozen=True)
class SkyRLExportRequest:
    training_manifest_uri: str
    attempt_id: str
    output: SkyRLExportOutputPaths


@dataclass(frozen=True)
class SkyRLExportConfig:
    request: SkyRLExportRequest
    execution: IrisSkyRLExecution
    source_runtime_commit: str
    source_runtime_profile: SkyRLRuntimeProfile | str


class SkyRLModel(Artifact):
    """Validated terminal HF policy export from a MarinSkyRL run."""

    policy_export_uri: str
    global_step: int
    tokenizer_uri: str
    tokenizer_revision: str
    checkpoint_root: str
    terminal_manifest_uri: str
    iris_job_id: str


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


def _launcher_command(requirement: str, request_path: str, action: str = "launch") -> list[str]:
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
        action,
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


def run_skyrl_training(config: SkyRLRunConfig) -> SkyRLTrainingResult | SkyRLCheckpoint:
    """Run training and return its validated metrics or checkpoint result."""
    envelope = {
        "schema_version": _PROTOCOL_SCHEMA_VERSION,
        "request": asdict(config.request),
        "execution": {
            **asdict(config.execution),
            "job_name": sanitize_job_name(f"{config.request.run_id}-{config.request.attempt_id}"),
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as request_file:
        json.dump(envelope, request_file, sort_keys=True)
        request_file.flush()
        completed = _run_launcher(_launcher_command(config.launcher_requirement, request_file.name))
    if not completed.stdout.strip():
        raise RuntimeError(
            f"MarinSkyRL launcher exited {completed.returncode} without a terminal response:\n"
            f"{completed.stderr.strip() or '(the launcher wrote nothing to stderr)'}"
        )
    response = json.loads(completed.stdout)
    if completed.returncode != 0 or response["state"] != "succeeded":
        failure = response.get("failure") or f"launcher exited {completed.returncode}"
        raise RuntimeError(
            f"MarinSkyRL attempt {config.request.attempt_id} failed: {failure}\n{completed.stderr.strip()}"
        )
    training = response["training"]
    common = dict(
        path=config.request.output.terminal_manifest_uri,
        global_step=training["global_step"],
        receipt_uri=training["receipt_uri"],
        resolved_config_uri=training["resolved_config_uri"],
        terminal_manifest_uri=config.request.output.terminal_manifest_uri,
        iris_job_id=response["iris_job_id"],
    )
    checkpoint = training["checkpoint"]
    if checkpoint is None:
        return SkyRLTrainingResult(**common)
    return SkyRLCheckpoint(
        **common,
        checkpoint_path=checkpoint["checkpoint_path"],
        trainer_state_sha256=checkpoint["trainer_state_sha256"],
        files=tuple(NativeCheckpointFile(**item) for item in checkpoint["files"]),
        checkpoint_retention_days=cast(int, config.request.checkpoint_retention_days),
        runtime_commit=config.request.runtime.commit,
        runtime_profile=config.request.runtime.profile,
        tokenizer_uri=config.request.model.tokenizer_uri,
        tokenizer_revision=config.request.model.tokenizer_revision,
    )


def _training_step(
    spec: SkyRLSpec,
    execution: IrisSkyRLExecution,
    *,
    completion_mode: SkyRLCompletionMode,
    step_name: str,
) -> ArtifactStep:
    """Build the common training node used by metrics and checkpoint APIs."""
    forbidden_positive_intervals = {"trainer.hf_save_interval"}
    if completion_mode is SkyRLCompletionMode.METRICS:
        forbidden_positive_intervals.add("trainer.ckpt_interval")
    for override in spec.overrides:
        key, separator, raw_value = override.lstrip("+").partition("=")
        if separator and key in forbidden_positive_intervals:
            try:
                enabled = int(raw_value) > 0
            except ValueError:
                enabled = True
            if enabled:
                raise ValueError(f"{completion_mode.value} completion forbids positive {key}: {override}")
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
            temporary_root = temporary_storage_base_path(
                ctx.output_path,
                ttl_days=spec.retention.temporary_storage_ttl_days,
                category=_TEMPORARY_OUTPUT_PREFIX,
            )
        attempts_root = prefix_join(temporary_root, "attempts")
        output = SkyRLOutputPaths(
            checkpoint_root=prefix_join(temporary_root, "checkpoints"),
            export_root=prefix_join(ctx.output_path, "exports"),
            attempts_root=attempts_root,
            resolved_config_uri=prefix_join(ctx.output_path, "resolved-skyrl.json"),
            terminal_manifest_uri=prefix_join(ctx.output_path, "terminal.json"),
        )
        completion_overrides = (
            ("++trainer.ckpt_interval=-1", "++trainer.hf_save_interval=-1")
            if completion_mode is SkyRLCompletionMode.METRICS
            else ("++trainer.hf_save_interval=-1",)
        )
        retention_overrides = (
            f"++trainer.max_ckpts_to_keep={spec.retention.resume_checkpoint_count}",
            f"++terminal_bench_config.trials_dir='{prefix_join(attempts_root, 'trace_jobs')}'",
            f"++generator.trajectory_retention.output_path='{prefix_join(attempts_root, 'trajectories')}'",
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
            overrides=(*spec.overrides, *retention_overrides, *completion_overrides),
            completion_mode=completion_mode,
            checkpoint_retention_days=(
                spec.retention.temporary_storage_ttl_days if completion_mode is SkyRLCompletionMode.CHECKPOINT else None
            ),
        )
        return SkyRLRunConfig(
            request=request,
            execution=cast(IrisSkyRLExecution, ctx.runtime_arg(_EXECUTION)),
            launcher_requirement=MARIN_SKYRL.requirement(),
        )

    artifact_type = SkyRLCheckpoint if completion_mode is SkyRLCompletionMode.CHECKPOINT else SkyRLTrainingResult
    return ArtifactStep(
        name=step_name,
        version=spec.version,
        artifact_type=artifact_type,
        run=run_skyrl_training,
        build_config=build_config,
        deps=deps,
        runtime_args={_EXECUTION: execution},
    )


def skyrl_metrics_step(spec: SkyRLSpec, execution: IrisSkyRLExecution) -> ArtifactStep[SkyRLTrainingResult]:
    """Train without creating a native checkpoint or portable model."""
    return _training_step(
        spec,
        execution,
        completion_mode=SkyRLCompletionMode.METRICS,
        step_name=f"{spec.name}-metrics",
    )


def skyrl_checkpoint_step(spec: SkyRLSpec, execution: IrisSkyRLExecution) -> ArtifactStep[SkyRLCheckpoint]:
    """Train once and retain an exact native checkpoint for resume or later export."""
    return _training_step(
        spec,
        execution,
        completion_mode=SkyRLCompletionMode.CHECKPOINT,
        step_name=f"{spec.name}-training",
    )


def _run_export(config: SkyRLExportConfig) -> SkyRLModel:
    source_dependency = replace(MARIN_SKYRL, commit=config.source_runtime_commit)
    envelope = {
        "schema_version": _PROTOCOL_SCHEMA_VERSION,
        "request": asdict(config.request),
        "execution": {
            **asdict(config.execution),
            "job_name": sanitize_job_name(f"{config.request.attempt_id}-export"),
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as request_file:
        json.dump(envelope, request_file, sort_keys=True)
        request_file.flush()
        completed = _run_launcher(_launcher_command(source_dependency.requirement(), request_file.name, action="export"))
    if not completed.stdout.strip():
        raise RuntimeError(
            f"MarinSkyRL export exited {completed.returncode} without a terminal response:\n"
            f"{completed.stderr.strip() or '(the launcher wrote nothing to stderr)'}"
        )
    response = json.loads(completed.stdout)
    if completed.returncode != 0 or response["state"] != "succeeded":
        failure = response.get("failure") or f"launcher exited {completed.returncode}"
        raise RuntimeError(
            f"MarinSkyRL export {config.request.attempt_id} failed: {failure}\n{completed.stderr.strip()}"
        )
    model = response["model"]
    return SkyRLModel(
        path=config.request.output.terminal_manifest_uri,
        policy_export_uri=model["policy_export_uri"],
        global_step=model["global_step"],
        tokenizer_uri=model["tokenizer_uri"],
        tokenizer_revision=model["tokenizer_revision"],
        checkpoint_root=model["checkpoint_root"],
        terminal_manifest_uri=model["terminal_manifest_uri"],
        iris_job_id=response["training_iris_job_id"],
    )


def skyrl_export_step(
    checkpoint_step: ArtifactStep[SkyRLCheckpoint],
    execution: IrisSkyRLExecution,
) -> ArtifactStep[SkyRLModel]:
    """Export an immutable native checkpoint without rerunning training."""
    final_name = checkpoint_step.name.removesuffix("-training")

    def build_config(ctx: StepContext) -> SkyRLExportConfig:
        attempt_id = "<attempt_id>" if ctx.is_fingerprint else uuid.uuid4().hex[:12]
        if ctx.is_fingerprint:
            training_manifest_uri = f"{_artifact_identity(checkpoint_step)}/terminal.json"
            source_runtime_commit = "<from-checkpoint-artifact>"
            source_runtime_profile: SkyRLRuntimeProfile | str = "<from-checkpoint-artifact>"
        else:
            checkpoint = ctx.resolved(checkpoint_step)
            training_manifest_uri = checkpoint.terminal_manifest_uri
            source_runtime_commit = checkpoint.runtime_commit
            source_runtime_profile = checkpoint.runtime_profile
        return SkyRLExportConfig(
            request=SkyRLExportRequest(
                training_manifest_uri=training_manifest_uri,
                attempt_id=attempt_id,
                output=SkyRLExportOutputPaths(
                    export_root=prefix_join(ctx.output_path, "exports"),
                    attempts_root=prefix_join(ctx.output_path, "attempts"),
                    terminal_manifest_uri=prefix_join(ctx.output_path, "terminal.json"),
                ),
            ),
            execution=cast(IrisSkyRLExecution, ctx.runtime_arg(_EXECUTION)),
            source_runtime_commit=source_runtime_commit,
            source_runtime_profile=source_runtime_profile,
        )

    return ArtifactStep(
        name=final_name,
        version=checkpoint_step.version,
        artifact_type=SkyRLModel,
        run=_run_export,
        build_config=build_config,
        deps=(checkpoint_step,),
        runtime_args={_EXECUTION: execution},
    )


def skyrl_step(spec: SkyRLSpec, execution: IrisSkyRLExecution) -> ArtifactStep[SkyRLModel]:
    """Compatibility facade: train to a native checkpoint, then export a portable model."""
    return skyrl_export_step(skyrl_checkpoint_step(spec, execution), execution)
