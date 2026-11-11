"""Validate, submit, and record one resolved SkyRL launch config."""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from finestore.rl.run_records import (
    AttemptRecord,
    ExportedPolicy,
    LaunchReceipt,
    LaunchResult,
    LaunchState,
    read_attempt,
    register_run_tables,
    run_records_root,
    write_attempt,
)
from finestore.store import DataStore
from iris.client.client import JobFailedError
from iris.resources.state import JobState
from omegaconf import DictConfig, OmegaConf

from cloud.iris.artifacts import terminal_checkpoint_step, write_json
from cloud.iris.launch_config import RunMode, SubmissionMode, load_launch_config
from marinskyrl.checkpoint_paths import policy_export_path, selected_export_step
from marinskyrl.resource_locator import join_resource_path
from marinskyrl.hf_model import validate_portable_hf_model_files
from marinskyrl.packed_tasks import select_task_references
from marinskyrl.task_sources import TaskTroveParquetSource, TaskTroveSelectionSnapshot, data_source, data_source_dict
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import launch_provenance


@dataclass(frozen=True)
class LaunchOutcome:
    """Result of one launcher backend submission."""

    job_id: str
    job_state: str
    exit_code: int
    started_at: str | None = None
    finished_at: str | None = None


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
    if config.inputs.get("evaluation_data"):
        prepared.inputs.evaluation_data = {
            name: _prepared_sources(list(sources)) for name, sources in config.inputs.evaluation_data.items()
        }
    OmegaConf.set_struct(prepared, True)
    return prepared


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
        provenance=asdict(launch_provenance()),
        runtime_profile=str(config.runtime.profile),
        model=model,
        failure=failure,
        started_at=outcome.started_at if outcome else None,
        finished_at=outcome.finished_at if outcome else None,
    )


def _record_failure(
    config: DictConfig,
    outcome: LaunchOutcome,
    failure: str,
) -> LaunchResult:
    result = _result(config, LaunchState.FAILED, outcome=outcome, failure=failure)
    _write_attempt_receipt(config, result)
    return result


def _write_attempt_receipt(config: DictConfig, result: LaunchResult) -> LaunchReceipt:
    export_root = str(config.artifacts.export_root)
    record = read_attempt(export_root, str(config.run.attempt_id))
    receipt = LaunchReceipt(config=OmegaConf.to_container(config, resolve=True), result=result)
    if record is None:
        record = AttemptRecord(run_id=str(config.run.id), attempt_id=str(config.run.attempt_id), launch=receipt)
    else:
        record = record.model_copy(update={"launch": receipt})
    with DataStore.open(run_records_root(export_root)) as store:
        tables = register_run_tables(store)
        write_attempt(tables.attempts, record)
        store.flush()
    return receipt


def execute_launch(config_path: Path, *, backend: LaunchBackend | None = None) -> LaunchResult:
    """Execute the lifecycle declared by one resolved launch config."""
    config = _prepare_config(load_launch_config(config_path))
    terminal_receipt_uri = str(config.artifacts.terminal_manifest_uri)
    if StoragePath(terminal_receipt_uri).exists():
        raise ValueError(f"Terminal receipt is immutable and already exists: {terminal_receipt_uri}")

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
        try:
            outcome = active_backend.launch(resolved_path)
        except JobFailedError as error:
            outcome = LaunchOutcome(
                job_id=str(error.job_id),
                job_state=error.status.state.value,
                exit_code=1,
                started_at=error.status.started_at.as_formatted_date() if error.status.started_at else None,
                finished_at=error.status.finished_at.as_formatted_date() if error.status.finished_at else None,
            )

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
    receipt = _write_attempt_receipt(config, result)
    write_json(terminal_receipt_uri, receipt.model_dump(mode="json"))
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
    json.dump(result.model_dump(mode="json"), sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if result.state in {LaunchState.PREPARED, LaunchState.SUBMITTED, LaunchState.SUCCEEDED} else 1


if __name__ == "__main__":
    sys.exit(main())
