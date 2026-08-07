# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin ArtifactStep adapter for the external MarinSkyRL trainer."""

from __future__ import annotations

import json
import subprocess
import tempfile
import uuid
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
from marin.training.training import LevanterCheckpoint
from rigging.filesystem import prefix_join

_EXECUTION = "skyrl_execution"
_LAUNCHER_PYTHON = "3.12"
_MARINSKYRL_STAGING_ROOT = PurePosixPath("/tmp/marinskyrl")
SKYRL_POLICY_LOCATION = "<skyrl-policy>"


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


def run_skyrl(config: SkyRLRunConfig) -> SkyRLModel:
    """Run the pinned external launcher and return its validated model value."""
    envelope = {
        "request": asdict(config.request),
        "execution": {
            **asdict(config.execution),
            "job_name": sanitize_job_name(f"{config.request.run_id}-{config.request.attempt_id}"),
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as request_file:
        json.dump(envelope, request_file, sort_keys=True)
        request_file.flush()
        completed = subprocess.run(
            _launcher_command(config.launcher_requirement, request_file.name),
            check=False,
            stdout=subprocess.PIPE,
            text=True,
        )
    if not completed.stdout.strip():
        raise RuntimeError(f"MarinSkyRL launcher exited {completed.returncode} without a terminal response")
    response = json.loads(completed.stdout)
    if completed.returncode != 0 or response["state"] != "succeeded":
        failure = response.get("failure") or f"launcher exited {completed.returncode}"
        raise RuntimeError(f"MarinSkyRL attempt {config.request.attempt_id} failed: {failure}")
    model = response["model"]
    return SkyRLModel(
        path=config.request.output.terminal_manifest_uri,
        policy_export_uri=model["policy_export_uri"],
        global_step=model["global_step"],
        tokenizer_uri=model["tokenizer_uri"],
        tokenizer_revision=model["tokenizer_revision"],
        checkpoint_root=model["checkpoint_root"],
        terminal_manifest_uri=model["terminal_manifest_uri"],
        iris_job_id=response["iris_job_id"],
    )


def skyrl_step(spec: SkyRLSpec, execution: IrisSkyRLExecution) -> ArtifactStep[SkyRLModel]:
    """Build a versioned MarinSkyRL training artifact."""
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
        output = SkyRLOutputPaths(
            checkpoint_root=prefix_join(ctx.output_path, "checkpoints"),
            export_root=prefix_join(ctx.output_path, "exports"),
            attempts_root=prefix_join(ctx.output_path, "attempts"),
            resolved_config_uri=prefix_join(ctx.output_path, "resolved-skyrl.json"),
            terminal_manifest_uri=prefix_join(ctx.output_path, "terminal.json"),
        )
        request = SkyRLLaunchRequest(
            run_id=f"{spec.name}-{spec.version}",
            attempt_id=attempt_id,
            config_yaml=spec.config_yaml,
            runtime=spec.runtime,
            model=spec.model.resolve(ctx),
            train_data=tuple(source.resolve(ctx) for source in spec.train_data),
            validation_data=tuple(source.resolve(ctx) for source in spec.validation_data),
            topology=spec.topology,
            output=output,
            seed=spec.seed,
            overrides=spec.overrides,
        )
        return SkyRLRunConfig(
            request=request,
            execution=cast(IrisSkyRLExecution, ctx.runtime_arg(_EXECUTION)),
            launcher_requirement=MARIN_SKYRL.requirement(),
        )

    return ArtifactStep(
        name=spec.name,
        version=spec.version,
        artifact_type=SkyRLModel,
        run=run_skyrl,
        build_config=build_config,
        deps=deps,
        runtime_args={_EXECUTION: execution},
    )
