"""Validate, submit, and record one resolved SkyRL launch config."""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from iris.client.client import JobFailedError
from iris.cluster.client.job_info import get_job_info
from iris.resources.state import JobState
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from cloud.iris.artifacts import terminal_checkpoint_step, write_json
from cloud.iris.launch_config import RunMode, SubmissionMode, hugging_face_models, load_launch_config
from cloud.iris.hf_model_cache import UnsupportedHuggingFaceSnapshot, ensure_hugging_face_model_cache
from marinskyrl.distillation import TeacherSource, compile_distillation_plan_from_config
from marinskyrl.checkpoint_paths import policy_export_path, selected_export_step
from marinskyrl.hf_model import HuggingFaceModelCache, validate_portable_hf_model_files
from marinskyrl.packed_tasks import select_task_references
from marinskyrl.speculative_decoding import SpeculatorModelConfig, SpeculatorModelSourceKind
from marinskyrl.resource_locator import join_resource_path
from marinskyrl.task_sources import TaskTroveParquetSource, TaskTroveSelectionSnapshot, data_source, data_source_dict
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance, launch_provenance


class LaunchState(StrEnum):
    PREPARED = "prepared"
    SUBMITTED = "submitted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class LaunchOutcome:
    """Result of one launcher backend submission."""

    job_id: str
    job_state: str
    exit_code: int


@dataclass(frozen=True)
class ExportedPolicy:
    policy_export_uri: str
    global_step: int
    tokenizer_uri: str
    tokenizer_revision: str
    checkpoint_root: str
    terminal_manifest_uri: str


@dataclass(frozen=True)
class LaunchResult:
    run_id: str
    attempt_id: str
    state: LaunchState
    iris_job_id: str | None
    iris_job_state: str | None
    provenance: Provenance
    runtime_profile: str
    model: ExportedPolicy | None
    failure: str | None


class LaunchBackend(Protocol):
    """I/O boundary for one validated launch config."""

    def validate(self, config_path: Path) -> None: ...

    def launch(self, config_path: Path) -> LaunchOutcome: ...

    def export_terminal_policy(self, config_path: Path) -> None: ...


def _prepared_sources(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for value in values:
        source = data_source(value)
        if isinstance(source, TaskTroveParquetSource):
            summary = select_task_references(source)
            source = replace(
                source,
                snapshot=TaskTroveSelectionSnapshot(
                    count=len(summary.references),
                    digest=summary.digest,
                    distinct_environment_count=summary.distinct_environment_count,
                ),
            )
        prepared.append(data_source_dict(source))
    return prepared


def _prepare_config(config: DictConfig) -> DictConfig:
    prepared = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    OmegaConf.set_struct(prepared, False)
    prepared.inputs.train_data = _prepared_sources(list(config.inputs.train_data))
    prepared.inputs.validation_data = _prepared_sources(list(config.inputs.validation_data))
    OmegaConf.set_struct(prepared, True)
    return prepared


def _prepare_model_caches(config: DictConfig) -> None:
    models = hugging_face_models(config)
    speculative = config.skyrl.get("generator", {}).get("speculative_decoding") or {}
    draft_mapping = speculative.get("model") if config.run.mode != RunMode.CHECKPOINT_EXPORT else None
    draft = (
        SpeculatorModelConfig.from_mapping(
            OmegaConf.to_container(draft_mapping, resolve=True), context="generator.speculative_decoding.model"
        )
        if draft_mapping
        else None
    )
    hub_draft = draft is not None and draft.source_kind is SpeculatorModelSourceKind.HUGGING_FACE
    if not models and not hub_draft:
        return
    if get_job_info() is None:
        raise RuntimeError(
            "Hugging Face model pre-caching requires an Iris CPU coordinator; run your rl_build_options experiment with --run"
        )
    plan = compile_distillation_plan_from_config(config.skyrl)
    teacher_models = (
        {teacher.model for teacher in plan.teachers if teacher.source is TeacherSource.LOCAL_INFERENCE}
        if plan is not None
        else set()
    )
    caches = []
    for model in models:
        tokenizer_uri = str(config.inputs.model.tokenizer_uri or "")
        separate_tokenizer = bool(tokenizer_uri and tokenizer_uri != model.path)
        policy_only = model.path == str(config.inputs.model.uri) and model not in teacher_models
        try:
            uri, manifest = ensure_hugging_face_model_cache(
                model.path,
                model.revision,
                ttl_days=int(config.artifacts.draft_model_cache_ttl_days),
                source_prefix=str(config.artifacts.checkpoint_root),
                tokenizer_mode="policy" if policy_only and separate_tokenizer else "embedded",
            )
        except UnsupportedHuggingFaceSnapshot as error:
            logger.warning(
                "Cannot mirror {}@{}; retaining node-side Hub fallback: {}", model.path, model.revision, error
            )
            continue
        assert manifest.revision is not None
        caches.append(HuggingFaceModelCache(model.path, model.revision, uri, manifest.identity, manifest.revision))
    if hub_draft:
        assert draft is not None and draft.hugging_face_repo_id is not None
        cached = next(
            (
                model
                for model in caches
                if model.model_id == draft.hugging_face_repo_id and model.revision == draft.source_identity
            ),
            None,
        )
        if cached is None:
            uri, manifest = ensure_hugging_face_model_cache(
                draft.hugging_face_repo_id,
                draft.source_identity,
                ttl_days=int(config.artifacts.draft_model_cache_ttl_days),
                source_prefix=str(config.artifacts.checkpoint_root),
                tokenizer_mode="policy",
            )
            assert manifest.revision is not None
            cached = HuggingFaceModelCache(
                draft.hugging_face_repo_id, draft.source_identity, uri, manifest.identity, manifest.revision
            )
            caches.append(cached)
        config.skyrl.generator.speculative_decoding.model = asdict(SpeculatorModelConfig(cached.uri, cached.identity))
    config.inputs.hugging_face_models = caches


def _exported_policy(config: DictConfig) -> ExportedPolicy:
    checkpoint_root = str(config.artifacts.checkpoint_root)
    global_step = selected_export_step(
        checkpoint_root, str(config.skyrl.trainer.hf_export_checkpoint), terminal_checkpoint_step(checkpoint_root)
    )
    policy_uri = policy_export_path(str(config.artifacts.export_root), global_step)
    root = StoragePath(policy_uri)
    names = {(directory / name).relative_to(root) for directory, _, files in root.walk() for name in files}
    validate_portable_hf_model_files(names, policy_uri)
    return ExportedPolicy(
        policy_export_uri=policy_uri,
        global_step=global_step,
        tokenizer_uri=str(config.inputs.model.tokenizer_uri),
        tokenizer_revision=str(config.inputs.model.tokenizer_revision),
        checkpoint_root=checkpoint_root,
        terminal_manifest_uri=str(config.artifacts.terminal_manifest_uri),
    )


def _result(
    config: DictConfig,
    state: LaunchState,
    *,
    outcome: LaunchOutcome | None = None,
    model: ExportedPolicy | None = None,
    failure: str | None = None,
) -> LaunchResult:
    return LaunchResult(
        run_id=str(config.run.id),
        attempt_id=str(config.run.attempt_id),
        state=state,
        iris_job_id=outcome.job_id if outcome else None,
        iris_job_state=outcome.job_state if outcome else None,
        provenance=launch_provenance(),
        runtime_profile=str(config.runtime.profile),
        model=model,
        failure=failure,
    )


def _manifest(config: DictConfig, result: LaunchResult) -> dict[str, Any]:
    return {
        "config": OmegaConf.to_container(config, resolve=True),
        "result": asdict(result),
    }


def _record_failure(
    config: DictConfig,
    outcome: LaunchOutcome,
    failure: str,
) -> LaunchResult:
    result = _result(config, LaunchState.FAILED, outcome=outcome, failure=failure)
    attempt_uri = join_resource_path(str(config.artifacts.attempts_root), f"{config.run.attempt_id}.json")
    write_json(attempt_uri, _manifest(config, result))
    return result


def execute_launch(config_path: Path, *, backend: LaunchBackend | None = None) -> LaunchResult:
    """Execute the lifecycle declared by one resolved launch config."""
    config = _prepare_config(load_launch_config(config_path))
    terminal_manifest_uri = str(config.artifacts.terminal_manifest_uri)
    if StoragePath(terminal_manifest_uri).exists():
        raise ValueError(f"Terminal manifest is immutable and already exists: {terminal_manifest_uri}")

    if backend is None:
        # The concrete Iris backend imports the training/export stack. Keep it
        # outside the CPU-only config validation and launcher import boundary.
        from cloud.iris.iris_backend import IrisBackend  # noqa: PLC0415

        active_backend: LaunchBackend = IrisBackend()
    else:
        active_backend = backend
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8") as resolved_file:
        OmegaConf.save(config, resolved_file.name, resolve=False)
        resolved_path = Path(resolved_file.name)
        if config.run.submission == SubmissionMode.PREPARE:
            active_backend.validate(resolved_path)
            return _result(config, LaunchState.PREPARED)
        _prepare_model_caches(config)
        OmegaConf.save(config, resolved_file.name, resolve=False)
        try:
            outcome = active_backend.launch(resolved_path)
        except JobFailedError as error:
            outcome = LaunchOutcome(job_id=str(error.job_id), job_state=error.status.state.value, exit_code=1)

        if outcome.exit_code != 0:
            return _record_failure(config, outcome, f"Iris job reached {outcome.job_state}")
        if config.run.submission == SubmissionMode.DETACH:
            return _result(config, LaunchState.SUBMITTED, outcome=outcome)
        if outcome.job_state != JobState.SUCCEEDED.value:
            return _record_failure(config, outcome, f"Iris job reached {outcome.job_state}")
        if config.run.export_hf:
            try:
                active_backend.export_terminal_policy(resolved_path)
            except (OSError, subprocess.CalledProcessError, ValueError) as error:
                return _record_failure(config, outcome, f"Terminal policy export failed: {error}")

    model = None
    if config.run.export_hf:
        try:
            model = _exported_policy(config)
        except ValueError as error:
            return _record_failure(config, outcome, str(error))
    if not StoragePath(str(config.artifacts.resolved_config_uri)).exists():
        return _record_failure(
            config,
            outcome,
            f"Successful Iris job did not persist resolved config: {config.artifacts.resolved_config_uri}",
        )
    result = _result(config, LaunchState.SUCCEEDED, outcome=outcome, model=model)
    payload = _manifest(config, result)
    attempt_uri = join_resource_path(str(config.artifacts.attempts_root), f"{config.run.attempt_id}.json")
    write_json(attempt_uri, payload)
    write_json(terminal_manifest_uri, payload)
    return result


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch SkyRL from one resolved Hydra config")
    commands = parser.add_subparsers(dest="component", required=True)
    iris = commands.add_parser("iris")
    iris_commands = iris.add_subparsers(dest="action", required=True)
    launch = iris_commands.add_parser("launch")
    launch.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    with contextlib.redirect_stdout(sys.stderr):
        result = execute_launch(args.config)
    json.dump(asdict(result), sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if result.state in {LaunchState.PREPARED, LaunchState.SUBMITTED, LaunchState.SUCCEEDED} else 1


if __name__ == "__main__":
    sys.exit(main())
