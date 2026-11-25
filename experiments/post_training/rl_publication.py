# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish native RL attempts from stored outputs on the launcher."""

from __future__ import annotations

import json
import logging
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

import click
from finestore.rl.rl_progress import (
    GRADER_REPORT_FILENAME,
    GRADER_REPORTS_DIR,
    AttemptSummary,
    Comparison,
    ComparisonRole,
    ControlReplicate,
    GateResult,
    GraderReport,
)
from finestore.rl.run_records import (
    AttemptRecord,
    attempt_lineage,
    read_attempt,
    read_attempt_metrics,
    read_attempts,
    write_attempt,
)
from marin.execution.artifact import read_record
from marin.execution.fingerprint import canonical_json, fingerprint_hash
from marin.rl.skyrl_records import (
    EXPORTS_SUBDIRECTORY,
    RecordedLaunch,
    attempt_tracking,
    read_stored_skyrl_attempts,
    recorded_launch,
)
from pydantic import BaseModel, Field, ValidationError
from rigging.filesystem.conditional_object import ConditionalWriteError, VersionedBytes, conditional_object
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.redaction import REDACTED_VALUE, is_sensitive_key_name, redact_string, redact_value

from experiments.post_training import rl_gate as gate
from experiments.post_training import rl_presets as presets
from experiments.post_training.rl_command import attempt_grafana_url
from experiments.post_training.rl_measurements import (
    InvalidComparison,
    RunInputs,
    evaluation_protocol,
    evaluation_status,
    lineage_windows,
    read_run_inputs,
    training_invariants,
)

APPLET_ID = tomllib.loads(
    (Path(__file__).resolve().parents[2] / "infra/marina/apps/rl_progress/applet.toml").read_text()
)["applet_id"]
ISSUE_FILENAME = "issue.json"
logger = logging.getLogger(__name__)


def publish_result(summary: AttemptSummary, service_url: str | None = None) -> str:
    """Upsert one native attempt and return its verified dashboard URL."""
    from marina import client  # noqa: PLC0415

    service_url = service_url or client.DEFAULT_MARINA_URL
    details = client.marina_request(service_url, "GET", f"/api/marina/applets/{APPLET_ID}")
    origin = client.applet_origin(service_url, details["authenticated_url"])
    served = client.marina_request(origin, "POST", f"/a/{APPLET_ID}/api/runs", json_body=summary.model_dump(mode="json"))
    if not isinstance(served, dict) or (served.get("run_id"), served.get("attempt_id")) != (
        summary.run_id,
        summary.attempt_id,
    ):
        raise RuntimeError(f"Published attempt {summary.run_id}/{summary.attempt_id} has a different returned identity")
    return details["authenticated_url"] + "#" + urlencode({"run": summary.run_id, "attempt": summary.attempt_id})


class PublicationOutcome(BaseModel):
    served_records: list[tuple[str, str]] = Field(default_factory=list)
    unserved_records: list[tuple[str, str] | str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    urls: dict[str, str] = Field(default_factory=dict)
    retry: str | None = None

    def to_dict(self) -> dict:
        return self.model_dump(mode="json", exclude={"urls"}, exclude_none=True)


def _visible_setting(text: str) -> str:
    key, _, value = text.partition("=")
    if is_sensitive_key_name(key):
        return f"{key}={REDACTED_VALUE}"
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return f"{key}={redact_string(value)}"
    redacted = redact_value(parsed)
    return f"{key}={json.dumps(redacted)}" if redacted != parsed else text


def run_label(inputs: RunInputs) -> str:
    parts = [inputs.preset_name, *map(_visible_setting, inputs.settings)]
    if inputs.tasks:
        parts.append(f"tasks={inputs.tasks}")
    if inputs.init:
        parts.append(f"init={inputs.init.artifact_name}@{inputs.init.artifact_version}")
    return " ".join(parts)


@dataclass(frozen=True)
class _TerminalMeasurements:
    failed_facts: list[str] = field(default_factory=list)
    evaluations: list[Comparison] = field(default_factory=list)
    optimizer_steps: int | None = None
    elapsed_seconds: float | None = None
    elapsed_excluding_eval_seconds: float | None = None
    evaluation_task_set: str | None = None


def _terminal_measurements(
    run_output: str, preset: presets.Preset, attempt: AttemptRecord, launch: RecordedLaunch, evidence: list[str]
) -> _TerminalMeasurements:
    attempt_id = launch.attempt_id
    result = attempt.native.result
    history = attempt_lineage(launch.export_root, attempt_id)
    complete_history = (
        bool(history)
        and all(entry.record.metrics_started_at is not None for entry in history)
        and history[-1].record.resumed_step in (None, 0)
    )
    metrics = read_attempt_metrics(launch.export_root, attempt_id)
    invariants = training_invariants(preset, metrics) if attempt.metrics_started_at is not None else ()
    failed = [fact.name for fact in invariants if not fact.holds]
    evaluations = []
    evaluation_task_set = None
    has_evaluation = bool(launch.validation_data or launch.evaluation_data) or any(
        callback.type == "evaluation" for callback in launch.callbacks or ()
    )
    if has_evaluation and (result.state == "succeeded" or attempt.metrics_started_at is not None):
        status = evaluation_status(run_output, preset, attempt, launch)
        evaluations = list(status.evaluations)
        evaluation_task_set = status.evaluation_task_set
        failed.extend(fact.name for fact in status.evaluation_invariants if not fact.holds)
        if status.evaluation_error is not None:
            evidence.append(f"Evaluation evidence unavailable: {status.evaluation_error}")
    if result.state != "succeeded":
        failed.insert(0, "native_attempt")
    optimizer_steps = (
        next((int(fact.value) for fact in invariants if fact.name == "planned_updates"), None)
        if complete_history
        else None
    )
    if isinstance(preset.recipe.layout.workload, presets.Generation) and optimizer_steps is not None:
        optimizer_steps = 0
    windows = lineage_windows(history)
    elapsed = sum(window.seconds for window in windows) if windows is not None else None
    if windows is None:
        evidence.append(
            "Checkpoint lineage timing is unknown: a shared attempt record or native time window is missing."
        )
    trainer_durations = [entry.record.elapsed_excluding_eval_seconds for entry in history]
    elapsed_excluding_eval = (
        float(sum(duration for duration in trainer_durations if duration is not None))
        if windows is not None and all(duration is not None for duration in trainer_durations)
        else None
    )
    if elapsed_excluding_eval is None and isinstance(preset.recipe.layout.workload, presets.Training):
        evidence.append("Training time excluding evaluation is unknown for this checkpoint lineage.")
    return _TerminalMeasurements(
        failed, evaluations, optimizer_steps, elapsed, elapsed_excluding_eval, evaluation_task_set
    )


def _read_grader_report(launch: RecordedLaunch, attempt_id: str, evidence: list[str]) -> GraderReport | None:
    report_path = StoragePath(launch.export_root) / GRADER_REPORTS_DIR / attempt_id / GRADER_REPORT_FILENAME
    try:
        return GraderReport.model_validate_json(report_path.read_text())
    except FileNotFoundError:
        pass
    except ValidationError:
        evidence.append("Stored grader report failed validation.")
    return None


def _gate_result(report: gate.GateReport | None) -> GateResult | None:
    if report is None:
        return None
    decisions = report.rows(ComparisonRole.OBJECTIVE, ComparisonRole.GUARD, ComparisonRole.REQUIREMENT)
    decision_metrics = {row.metric for row in decisions}
    history_scores = [
        row
        for row in report.rows(ComparisonRole.SLICE)
        if row.metric not in decision_metrics
        and (
            row.metric.partition("/")[2] in {"mean", "family_mean"}
            or ("/family/" in row.metric and row.metric.endswith("/mean"))
        )
    ]
    return GateResult(
        verdict=report.verdict,
        effect=report.effect,
        reason=report.reason,
        invariants=list(report.invariants),
        comparisons=[*decisions, *history_scores],
        attempt_ids=list(report.run.attempt_ids),
        replicates=report.run.replicates,
        control_commit=report.control.commit,
        control_version=report.control.version,
    )


def _evaluation_identities(launch: RecordedLaunch, evidence: list[str]) -> tuple[str | None, str | None]:
    """Return the stored evaluation sources and the evaluation protocol digest."""
    sources = [*launch.validation_data, *(source for group in launch.evaluation_data.values() for source in group)]
    if not sources:
        return None, None
    data = "; ".join(sorted({f"{source.identity}/{source.relative_path}" for source in sources}))
    try:
        protocols = {name: evaluation_protocol(launch, name) for name in ("", *launch.evaluation_data)}
    except InvalidComparison:
        evidence.append("The stored evaluation protocol is unavailable.")
        return data, None
    return data, fingerprint_hash(canonical_json(protocols))


def build_record(
    run_output: str,
    preset: presets.Preset,
    *,
    table: presets.ModelPresets,
    attempt: AttemptRecord,
    report: gate.GateReport | None = None,
    gate_name: str | None = None,
    gate_failure: bool = False,
    control: ControlReplicate | None = None,
) -> AttemptSummary:
    """Build a progress row from stored native outputs without live lookups."""
    launch, result = recorded_launch(attempt), attempt.native.result
    evidence = []
    try:
        inputs = read_run_inputs(run_output)
    except FileNotFoundError:
        inputs = RunInputs(preset_name=preset.name)
        evidence.append("Stored RL inputs are unavailable; using the supplied preset.")
    preset = table.by_name[inputs.preset_name]
    if launch.max_steps is None:
        raise ValueError("Stored native max_steps is unavailable")
    preset = preset.with_updates(launch.max_steps)
    attempt_id = launch.attempt_id
    provenance = result.provenance or {}
    tracking = attempt_tracking(attempt, launch)
    started, finished = result.started_at, result.finished_at
    index = attempt.index
    terminal = result.state.is_terminal
    interrupted = attempt.interruption is not None and not terminal
    measured = (
        _terminal_measurements(run_output, preset, attempt, launch, evidence) if terminal else _TerminalMeasurements()
    )
    failed = list(measured.failed_facts)
    if interrupted:
        failed.append("coordinator_interrupted")
    if gate_failure:
        failed.append("required_gate_report")
    if not provenance and index is None and terminal:
        evidence.append("Recorded source provenance is unavailable.")
    if index is not None and index.base_preset_version is None:
        evidence.append("The base preset version is unavailable from the branch's main merge base.")
    grader_report = _read_grader_report(launch, attempt_id, evidence) if terminal else None
    evaluation_data, evaluation_protocol_id = _evaluation_identities(launch, evidence)
    config = redact_value(attempt.native.config)
    assert isinstance(config, dict)
    return AttemptSummary(
        run_id=attempt.index_run_id,
        native_run_id=attempt.run_id,
        attempt_id=attempt_id,
        label=run_label(inputs),
        series=preset.name,
        model=index.model if index is not None else launch.model_name or table.model,
        preset_version=index.preset_version if index is not None else gate.config_version(attempt.native.config, launch),
        is_control=bool(control or inputs.control or (index and index.is_control)),
        base_preset_version=index.base_preset_version if index is not None else None,
        control=control or inputs.control,
        control_preset=preset.control_preset if preset.gated else None,
        evaluation_data_identity=evaluation_data,
        evaluation_task_set=measured.evaluation_task_set,
        evaluation_protocol_identity=evaluation_protocol_id,
        user=index.user if index is not None else None,
        branch=index.branch if index is not None else provenance.get("branch"),
        commit=index.source_revision if index is not None else provenance.get("base_commit"),
        settings=list(map(_visible_setting, inputs.settings)),
        launch_config=config,
        dirty=index.dirty if index is not None else provenance.get("dirty"),
        pr_number=index.pr_number if index is not None else None,
        cluster=launch.cluster,
        issue=lineage_issue(launch.export_root),
        run_date=started or tracking.get("started_at") or (index.submitted_at if index is not None else None),
        started_at=started,
        completed_at=finished if terminal else None,
        planned_updates=preset.recipe.layout.workload.update_count(preset.updates),
        measured_updates=list(preset.measured),
        iris_job_id=result.iris_job_id or launch.coordinator_iris_job_id,
        wandb_url=tracking.get("wandb_url"),
        grafana_url=attempt_grafana_url(attempt, launch),
        timing_scope="checkpoint_lineage_native_jobs",
        source_revision=(
            f"base={index.source_revision} tree={index.tree_hash or 'Unknown'}"
            if index is not None
            else f"base={provenance.get('base_commit')} tree={provenance.get('tree_hash')}" if provenance else "Unknown"
        ),
        data_identity="; ".join(source.identity for source in launch.train_data) or None,
        initializer_identity=launch.model_identity,
        artifact_uri=run_output,
        evidence=evidence,
        status="failed" if failed else "succeeded" if terminal else "running",
        failed_facts=failed,
        gate=gate_name,
        gate_result=_gate_result(report),
        elapsed_seconds=measured.elapsed_seconds,
        elapsed_excluding_eval_seconds=measured.elapsed_excluding_eval_seconds,
        optimizer_steps=measured.optimizer_steps,
        evaluations=measured.evaluations,
        grader_report=grader_report,
        engine=tracking.get("engine"),
    )


class IssueConflict(ValueError):
    """The native output already has a different assigned issue."""


def _check_issue(export_root: str, requested: int | None, reservation: VersionedBytes | None) -> int | None:
    issues = (
        {int(reservation.data)}
        if reservation is not None
        else {attempt.issue for attempt in read_attempts(export_root).values() if attempt.issue is not None}
    )
    if requested is not None:
        issues.add(requested)
    if len(issues) > 1:
        raise IssueConflict(f"Run has conflicting lineage issues {sorted(issues)} at {export_root}")
    return next(iter(issues), None)


def lineage_issue(export_root: str, requested: int | None = None) -> int | None:
    """Read the run's issue and verify a proposed value against stored assignments."""
    assignment = conditional_object(str(StoragePath(export_root) / ISSUE_FILENAME))
    return _check_issue(export_root, requested, assignment.read())


def reserve_issue(export_root: str, requested: int | None) -> int | None:
    """Reserve the lineage issue once for all writers to the native output."""
    assignment = conditional_object(str(StoragePath(export_root) / ISSUE_FILENAME))
    reservation = assignment.read()
    assigned = _check_issue(export_root, requested, reservation)
    if assigned is not None and reservation is None:
        try:
            assignment.write(str(assigned).encode(), expected_version=None)
        except ConditionalWriteError:
            return _check_issue(export_root, assigned, assignment.read())
    return assigned


def _coordinator_attempts(path: str, coordinator: str) -> dict[str, AttemptRecord]:
    export_root = str(StoragePath(path) / EXPORTS_SUBDIRECTORY)
    return {
        key: attempt
        for key, attempt in read_attempts(export_root).items()
        if attempt.index is not None
        and attempt.config is not None
        and attempt.config["run"].get("coordinator_iris_job_id") == coordinator
    }


def publish_running(
    path: str,
    preset: presets.Preset,
    table: presets.ModelPresets,
    coordinator: str,
    service_url: str | None,
    published: set[tuple[str, str]],
) -> None:
    """Upsert the prepared attempt for this coordinator while native work runs."""
    try:
        for attempt in _coordinator_attempts(path, coordinator).values():
            key = (attempt.index_run_id, attempt.attempt_id)
            if key in published or attempt.native.result.state.is_terminal or attempt.interruption is not None:
                continue
            url = publish_result(build_record(path, preset, table=table, attempt=attempt), service_url)
            published.add(key)
            click.echo(f"RL progress {url}")
    except Exception as error:
        logger.warning("Running row publication will retry for %s (%s)", path, type(error).__name__)


@dataclass(frozen=True)
class _OutputAttempts:
    current: str | None
    records: dict[str, AttemptRecord]


def _prepare_publication_attempts(path: str, failed_coordinator: str | None) -> _OutputAttempts:
    current, attempts = read_stored_skyrl_attempts(path)
    if failed_coordinator is not None:
        for key, record in _coordinator_attempts(path, failed_coordinator).items():
            record = attempts.get(key, record)
            if not record.native.result.state.is_terminal and record.interruption is None:
                export_root = recorded_launch(record).export_root
                write_attempt(
                    export_root,
                    AttemptRecord(run_id=record.run_id, attempt_id=key, interruption="coordinator_terminated"),
                )
                record = read_attempt(export_root, key) or record.model_copy(
                    update={"interruption": "coordinator_terminated"}
                )
            attempts[key] = record
            if current is None:
                current = key
    return _OutputAttempts(current, attempts)


def _read_publication_gate(
    gate_path: str | None, current_attempts: tuple[str | None, ...], replicates: int
) -> gate.GateReport | None:
    report = gate.read_gate_report(gate_path)
    if report is not None and (report.run.replicates != replicates or report.run.attempt_ids != current_attempts):
        raise ValueError("Gate report attempts differ from the current output group; repeat with --wait")
    return report


def publish_runs(
    outputs: Sequence[str],
    preset: presets.Preset,
    *,
    table: presets.ModelPresets,
    gate_path: str | None = None,
    service_url: str | None = None,
    attempt_id: str | None = None,
    issue: int | None = None,
    failed_coordinator: str | None = None,
    controls: Mapping[str, ControlReplicate] | None = None,
) -> PublicationOutcome:
    """Rebuild and upsert terminal or interrupted attempts, isolating publication failures."""
    if issue is not None and (len(outputs) != 1 or attempt_id is None):
        raise ValueError("An issue requires exactly one output and its explicit attempt ID")
    result = PublicationOutcome()
    snapshots = {}
    for path in outputs:
        try:
            snapshots[path] = _prepare_publication_attempts(path, failed_coordinator)
        except Exception as error:
            result.unserved_records.append(path)
            result.errors.append(str(error))
    report = None
    report_error = None
    try:
        current_ids = tuple(snapshot.current for snapshot in snapshots.values() if snapshot.current in snapshot.records)
        report = _read_publication_gate(gate_path, current_ids, len(outputs))
    except Exception as error:
        report = None
        report_error = str(error)
    for path, snapshot in snapshots.items():
        current_id, attempts = snapshot.current, snapshot.records
        if attempt_id is None and current_id not in attempts:
            result.unserved_records.append(path)
            result.errors.append(f"No stored native receipt at {path} for current attempt {current_id}")
        selected = (
            attempts if attempt_id is None else {key: value for key, value in attempts.items() if key == attempt_id}
        )
        if attempt_id is not None and not selected:
            result.unserved_records.append((path, attempt_id))
            result.errors.append(f"No stored native receipt at {path} for {attempt_id}")
        for selected_id, receipt in selected.items():
            if attempt_id is None and not receipt.native.result.state.is_terminal and receipt.interruption is None:
                continue
            destination = (receipt.index_run_id, selected_id)
            try:
                if not receipt.native.result.state.is_terminal and receipt.interruption is None:
                    raise ValueError("Publication requires a terminal native attempt")
                current_report = report if selected_id == current_id else None
                missing_gate = (
                    gate_path is not None
                    and selected_id == current_id
                    and receipt.native.result.state == "succeeded"
                    and current_report is None
                )
                if missing_gate and failed_coordinator is None:
                    raise ValueError(
                        report_error or "Gate report missing for this attempt; repeat the original command with --wait"
                    )
                record = build_record(
                    path,
                    preset,
                    table=table,
                    attempt=receipt,
                    report=current_report,
                    gate_name=read_record(gate_path).name if current_report is not None else None,
                    gate_failure=missing_gate,
                    control=controls.get(path) if controls is not None else None,
                )
                if issue is not None:
                    record = record.model_copy(
                        update={"issue": reserve_issue(recorded_launch(receipt).export_root, issue)}
                    )
                url = publish_result(record, service_url)
                result.served_records.append(destination)
                if selected_id == current_id:
                    result.urls[path] = url
            except Exception as error:
                result.unserved_records.append(destination)
                result.errors.append(str(error))
    if result.unserved_records:
        result.retry = "Repeat the original command with --wait."
    return result


def publish_controls(
    table: presets.ModelPresets, preset_name: str, *, service_url: str | None = None
) -> PublicationOutcome:
    """Publish the exact pinned controls from their stored native outputs."""
    preset = table.by_name.get(preset_name)
    if preset is None or not preset.gated:
        raise click.BadParameter("Choose a dashboard preset with controls", param_hint="--preset")
    try:
        pin = gate.control_pin(preset, table=table)
    except InvalidComparison as error:
        raise click.UsageError(str(error)) from error
    configure_coreweave_s3()
    controls = {
        gate.control_path(preset, pin, index, table=table): ControlReplicate(
            commit=pin.commit, version=pin.version, replicate=index
        )
        for index in range(pin.replicates)
    }
    result = publish_runs(
        tuple(controls),
        table.by_name[pin.preset],
        table=table,
        service_url=service_url,
        controls=controls,
    )
    if result.unserved_records:
        result.retry = "Repeat the control publication command."
    return result


def command(table: presets.ModelPresets) -> click.Command:
    """Build stored-attempt and pinned-control publication commands for a model table."""

    @click.command()
    @click.option("--preset", "preset_name", type=click.Choice(tuple(table.by_name)), required=True)
    @click.option("--marina-url", "url", help="Marina origin (defaults to the Marina service).")
    @click.option("--run-output", "run_output", help="Stored native run output; requires --attempt.")
    @click.option("--attempt", help="Publish this stored attempt from --run-output, without native execution.")
    @click.option("--issue", type=click.IntRange(min=1), help="Associate an existing native attempt with an issue.")
    def command(
        preset_name: str,
        url: str | None,
        run_output: str | None,
        attempt: str | None,
        issue: int | None,
    ) -> None:
        """Publish pinned controls or one stored native attempt."""
        if (run_output is None) != (attempt is None):
            raise click.UsageError("--run-output and --attempt are required together")
        if issue is not None and attempt is None:
            raise click.UsageError("--issue requires --run-output and --attempt")
        if run_output is not None:
            configure_coreweave_s3()
            result = publish_runs(
                [run_output],
                table.by_name[preset_name],
                table=table,
                service_url=url,
                attempt_id=attempt,
                issue=issue,
            )
            if result.unserved_records:
                result.retry = "Repeat this publication command with --run-output and --attempt."
        else:
            result = publish_controls(table, preset_name, service_url=url)
        click.echo(json.dumps(result.to_dict()))
        if result.unserved_records:
            raise click.exceptions.Exit(1)

    return command
