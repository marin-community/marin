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
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal, cast

import fsspec
import yaml
from iris.cluster.client.job_info import get_job_info
from pydantic import TypeAdapter
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import StoragePath, prefix_join

from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.execution.remote import sanitize_job_name
from marin.external_dependencies import MARIN_SKYRL
from marin.rollouts.catalog import RolloutRunKind, record_rollout_run, rollout_run_record
from marin.skyrl_recipe import LaunchResult, LaunchState, SkyRLRecipe
from marin.training.training import LevanterCheckpoint

_EXECUTION = "skyrl_execution"
_LAUNCHER_PYTHON = "3.12"
_MARINSKYRL_STAGING_ROOT = PurePosixPath("/tmp/marinskyrl")
_TEMPORARY_OUTPUT_PREFIX = "skyrl"
_TRACE_JOBS_SUBDIR = "trace_jobs"
_TRAJECTORIES_SUBDIR = "trajectories"
_LAUNCHER_DIAGNOSTIC_LINES = 20
SKYRL_TEMPORARY_STORAGE_TTL_DAYS = 14
IRIS_HUB_CLUSTER_CONFIG = "lib/iris/config/marin.yaml"


def skyrl_temporary_run_path(output_path: str, *, ttl_days: int) -> str:
    """Return the lifecycle-managed storage path for a SkyRL run."""
    temporary_root = marin_temp_bucket(ttl_days=ttl_days, source_prefix=output_path)
    return str(StoragePath(temporary_root) / _TEMPORARY_OUTPUT_PREFIX / StoragePath(output_path).key)


@dataclass(frozen=True)
class SkyRLHardware:
    """GPU model and width of each allocated node."""

    gpu_variant: str
    gpus_per_node: int

    def __post_init__(self) -> None:
        if self.gpus_per_node <= 0:
            raise ValueError("SkyRL gpus_per_node must be positive")


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
            identity=artifact_identity(self.step),
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
            identity=artifact_identity(self.step),
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
    recipe: SkyRLRecipe
    model: ArtifactHfModel
    train_data: tuple[SkyRLDataSource, ...]
    validation_data: tuple[SkyRLDataSource, ...]
    hardware: SkyRLHardware
    retention: SkyRLRetentionPolicy
    seed: int

    def __post_init__(self) -> None:
        recipe = self.recipe
        trainer, generator = recipe.trainer, recipe.generator
        placement = trainer.placement
        for section, path, keys in (
            (
                trainer,
                "trainer",
                ("strategy", "train_batch_size", "policy_mini_batch_size", "micro_train_batch_size_per_gpu"),
            ),
            (
                placement,
                "trainer.placement",
                (
                    "colocate_all",
                    "colocate_policy_ref",
                    "policy_num_nodes",
                    "policy_num_gpus_per_node",
                    "ref_num_nodes",
                    "ref_num_gpus_per_node",
                ),
            ),
            (
                generator,
                "generator",
                (
                    "backend",
                    "run_engines_locally",
                    "num_inference_engines",
                    "inference_engine_tensor_parallel_size",
                    "inference_engine_pipeline_parallel_size",
                    "inference_engine_data_parallel_size",
                    "inference_engine_expert_parallel_size",
                    "n_samples_per_prompt",
                ),
            ),
            (trainer.algorithm, "trainer.algorithm", ("use_kl_loss",)),
        ):
            for key in keys:
                if key not in section.model_fields_set:
                    raise ValueError(f"SkyRL recipe must explicitly set {path}.{key}")
        if trainer.strategy != "megatron":
            raise ValueError("Marin SkyRL artifacts require trainer.strategy=megatron")
        if generator.run_engines_locally is not True:
            raise ValueError("Marin SkyRL artifacts require generator.run_engines_locally=true")
        if placement.policy_num_gpus_per_node != self.hardware.gpus_per_node:
            raise ValueError("SkyRL policy_num_gpus_per_node must match the whole-node hardware width")
        if trainer.critic.model.path:
            raise ValueError("Marin SkyRL artifacts do not describe a separate critic role")
        use_reference = trainer.algorithm.use_kl_loss or trainer.algorithm.use_kl_in_reward
        if not use_reference and not placement.colocate_policy_ref:
            raise ValueError("SkyRL separate reference nodes require a reference model")


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


@dataclass(frozen=True)
class SkyRLRunConfig:
    launch_config_yaml: str
    run_id: str
    attempt_id: str
    model: ResolvedModelLocator
    output: SkyRLOutputPaths
    export_hf: bool
    draft_checkpoint_root: str | None
    launcher_requirement: str


class SkyRLRun(Artifact):
    """Terminal result from a MarinSkyRL run."""

    hf_model_uri: str | None
    global_step: int | None
    tokenizer_uri: str
    tokenizer_revision: str
    checkpoint_root: str
    draft_checkpoint_root: str | None
    terminal_manifest_uri: str
    iris_job_id: str


_LAUNCH_RESULT = TypeAdapter(LaunchResult)


def _launcher_command(requirement: str, config_path: str) -> list[str]:
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
        "--config",
        config_path,
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


def run_skyrl(config: SkyRLRunConfig) -> SkyRLRun:
    """Run the pinned external launcher and return its validated result."""
    response: LaunchResult | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8") as launch_file:
            launch_file.write(config.launch_config_yaml)
            launch_file.flush()
            completed = _run_launcher(_launcher_command(config.launcher_requirement, launch_file.name))
        if not completed.stdout.strip():
            raise RuntimeError(
                f"MarinSkyRL launcher exited {completed.returncode} without a terminal response:\n"
                f"{completed.stderr.strip() or '(the launcher wrote nothing to stderr)'}"
            )
        response = _LAUNCH_RESULT.validate_json(completed.stdout)
        if completed.returncode != 0 or response.state != LaunchState.SUCCEEDED:
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
    output: SkyRLOutputPaths,
) -> str:
    recipe = spec.recipe.to_skyrl()
    task_env = recipe.get("extra_env", {})
    terminal_bench = spec.recipe.terminal_bench or {}
    harbor = terminal_bench.get("harbor", {})
    controller_ingress = harbor.get("name") == "opencode"
    submit_through_ambient_controller = get_job_info() is not None and execution.target_cluster is not None
    target_cluster = None if submit_through_ambient_controller else execution.target_cluster
    parent_cluster_config = None if submit_through_ambient_controller else execution.parent_cluster_config
    launch = {
        "schema_version": 1,
        "run": {
            "id": run_id,
            "attempt_id": attempt_id,
            "seed": spec.seed,
            "mode": "train",
            "submission": "wait",
            "export_hf": export_hf,
        },
        "runtime": {
            "launcher_commit": MARIN_SKYRL.commit,
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
                "gpus_per_node": spec.hardware.gpus_per_node,
                "gpu_variant": spec.hardware.gpu_variant,
                "cpu": execution.cpu,
                "memory": execution.memory,
                "disk": execution.disk,
            },
            "priority": execution.priority,
            "max_retries": execution.max_retries,
            "timeout": execution.job_timeout_seconds,
            "target_cluster": target_cluster,
            "parent_cluster_config": parent_cluster_config,
        },
        "ingress": {
            "host": "iris.oa.dev" if controller_ingress and execution.cluster.startswith("cw-") else "",
            "record_literal": controller_ingress,
            "vllm_http_port": 8000,
        },
        "ray": {
            "port": 6379,
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
            "train_data": [_launch_data_source(source) for source in train_data],
            "validation_data": [_launch_data_source(source) for source in validation_data],
        },
        "skyrl": recipe,
    }
    return yaml.safe_dump(launch, sort_keys=False)


def skyrl_step(
    spec: SkyRLSpec,
    execution: IrisSkyRLExecution,
    *,
    export_hf: bool = False,
) -> ArtifactStep[SkyRLRun]:
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
            resolved_config_uri=prefix_join(ctx.output_path, "resolved-launch.yaml"),
            terminal_manifest_uri=prefix_join(ctx.output_path, "terminal.json"),
        )
        speculative = spec.recipe.generator.speculative_decoding
        draft_training = speculative.training if speculative is not None else None
        draft_checkpoint_root = prefix_join(output.checkpoint_root, "drafts") if draft_training is not None else None
        run_id = f"{step_name}-{spec.version}"
        model = spec.model.resolve(ctx)
        train_data = tuple(source.resolve(ctx) for source in spec.train_data)
        validation_data = tuple(source.resolve(ctx) for source in spec.validation_data)
        execution = (
            _FINGERPRINT_EXECUTION if ctx.is_fingerprint else cast(IrisSkyRLExecution, ctx.runtime_arg(_EXECUTION))
        )
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
                output=output,
            ),
            run_id=run_id,
            attempt_id=attempt_id,
            model=model,
            output=output,
            export_hf=export_hf,
            draft_checkpoint_root=draft_checkpoint_root,
            launcher_requirement=MARIN_SKYRL.requirement(),
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
