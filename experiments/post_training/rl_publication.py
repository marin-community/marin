# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish native RL attempts from stored outputs on the launcher."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import click
from finestore.commit import CommitConflict
from finestore.reader import ReadView
from finestore.rl.rl_progress import (
    GRADER_REPORT_FILENAME,
    GRADER_REPORTS_DIR,
    AttemptSummary,
    Comparison,
    ComparisonRole,
    GateResult,
    GraderReport,
)
from finestore.rl.run_records import (
    AttemptRecord,
    read_attempt,
    read_attempt_metrics,
    read_attempts,
    run_records_root,
    update_attempt,
    write_attempt,
)
from finestore.store import DataStore
from marin.execution.artifact import read_record
from marin.rl.skyrl_records import (
    RecordedLaunch,
    attempt_tracking,
    read_stored_skyrl_attempts,
    recorded_launch,
)
from pydantic import BaseModel, Field, ValidationError
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.redaction import redact_value

from experiments.post_training import rl_gate as gate
from experiments.post_training import rl_presets as presets
from experiments.post_training.rl_command import attempt_grafana_url, visible_setting
from experiments.post_training.rl_index import publish_result
from experiments.post_training.rl_measurements import (
    InvalidComparison,
    RunInputs,
    evaluation_protocol,
    evaluation_status,
    lineage_windows,
    measured_training,
    read_run_inputs,
    recorded_lineage,
)

logger = logging.getLogger(__name__)


class PublicationOutcome(BaseModel):
    served_records: list[tuple[str, str]] = Field(default_factory=list)
    unserved_records: list[tuple[str, str] | str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    urls: dict[str, str] = Field(default_factory=dict)
    attempts: dict[str, AttemptRecord] = Field(default_factory=dict, exclude=True)
    gate_path: str | None = Field(default=None, exclude=True)
    retry: str | None = None

    def to_dict(self) -> dict:
        return self.model_dump(mode="json", exclude={"urls"}, exclude_none=True)


def run_label(inputs: RunInputs) -> str:
    parts = [inputs.preset_name, *map(visible_setting, inputs.settings)]
    if inputs.tasks:
        parts.append(f"tasks={inputs.tasks}")
    if inputs.init:
        parts.append(f"init={inputs.init.artifact_name}@{inputs.init.artifact_version}")
    return " ".join(parts)


@dataclass(frozen=True)
class _TerminalMeasurements:
    measured_updates: range
    failed_facts: list[str] = field(default_factory=list)
    evaluations: list[Comparison] = field(default_factory=list)
    optimizer_steps: int | None = None
    elapsed_seconds: float | None = None
    elapsed_excluding_eval_seconds: float | None = None


def _terminal_measurements(
    run_output: str,
    preset: presets.Preset,
    attempt: AttemptRecord,
    launch: RecordedLaunch,
    evidence: list[str],
    inputs: RunInputs,
) -> _TerminalMeasurements:
    attempt_id = launch.attempt_id
    result = attempt.native.result
    history = recorded_lineage(attempt, launch)
    complete_history = (
        bool(history)
        and all(entry.record.metrics_started_at is not None for entry in history)
        and history[-1].record.resumed_step in (None, 0)
    )
    metrics = read_attempt_metrics(launch.export_root, attempt_id)
    measured_preset, invariants, spend = measured_training(preset, metrics, launch.rollout_token_budget)
    if attempt.metrics_started_at is None:
        invariants = ()
    elif spend is not None:
        evidence.append(f"Completed checkpoint-lineage updates observed {spend.observed:,} response tokens.")
        evidence.append(
            "The response-token upper bound is unavailable for this checkpoint lineage."
            if spend.upper_bound is None
            else f"The completed-update upper bound including unobserved requests is {spend.upper_bound:,} tokens."
        )
    failed = [fact.name for fact in invariants if not fact.holds]
    evaluations = []
    has_evaluation = bool(launch.validation_data or launch.evaluation_data) or any(
        callback.type == "evaluation" for callback in launch.callbacks or ()
    )
    if attempt.index is not None and attempt.index.is_control and result.state == "succeeded":
        try:
            control = gate.read_control_measurements(run_output, preset, attempt, inputs)
        except (InvalidComparison, FileNotFoundError) as error:
            failed.append("control_measurements")
            evidence.append(str(error))
        else:
            evaluations = list(control.evaluations)
            failed = [fact.name for fact in control.invariants if not fact.holds]
    elif has_evaluation and (result.state == "succeeded" or attempt.metrics_started_at is not None):
        status = evaluation_status(run_output, preset, attempt, launch, inputs=inputs)
        evaluations = list(status.evaluations)
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
        measured_updates=measured_preset.measured,
        failed_facts=failed,
        evaluations=evaluations,
        optimizer_steps=optimizer_steps,
        elapsed_seconds=elapsed,
        elapsed_excluding_eval_seconds=elapsed_excluding_eval,
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
        attempt_ids=[attempt for attempt in report.run.attempt_ids if attempt is not None],
        replicates=report.run.replicates,
    )


def _evaluation_data_identity(launch: RecordedLaunch, evidence: list[str]) -> str | None:
    sources = [*launch.validation_data, *(source for group in launch.evaluation_data.values() for source in group)]
    if not sources:
        return None
    try:
        for name in launch.evaluation_data:
            evaluation_protocol(launch, name)
    except InvalidComparison:
        evidence.append("The stored evaluation protocol is unavailable.")
    return "; ".join(sorted({f"{source.identity}/{source.relative_path}" for source in sources}))


def build_record(
    run_output: str,
    preset: presets.Preset,
    *,
    table: presets.ModelPresets,
    attempt: AttemptRecord,
    report: gate.GateReport | None = None,
    gate_name: str | None = None,
    gate_failure: bool = False,
) -> AttemptSummary:
    """Build a progress row from stored native outputs without live lookups."""
    launch, result = recorded_launch(attempt), attempt.native.result
    index = attempt.index
    evidence = []
    try:
        inputs = read_run_inputs(run_output)
    except FileNotFoundError:
        inputs = RunInputs(preset_name=index.preset if index is not None else preset.name)
        evidence.append("Stored RL inputs are unavailable; using the recorded preset.")
    if index is not None and (index.model != table.model or index.preset != inputs.preset_name):
        raise ValueError("Stored preset inputs disagree with the native run index identity")
    preset = table.by_name[inputs.preset_name]
    if launch.max_steps is None:
        raise ValueError("Stored native max_steps is unavailable")
    preset = preset.with_updates(launch.max_steps)
    attempt_id = launch.attempt_id
    provenance = result.provenance or {}
    tracking = attempt_tracking(attempt, launch)
    started, finished = result.started_at, result.finished_at
    terminal = result.state.is_terminal
    interrupted = attempt.interruption is not None and not terminal
    measured = (
        _terminal_measurements(run_output, preset, attempt, launch, evidence, inputs)
        if terminal
        else _TerminalMeasurements(preset.measured)
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
    evaluation_data = _evaluation_data_identity(launch, evidence)
    return AttemptSummary(
        run_id=attempt.index_run_id,
        native_run_id=attempt.run_id,
        attempt_id=attempt_id,
        label=run_label(inputs),
        series=preset.name,
        model=index.model if index is not None else launch.model_name or table.model,
        preset_version=index.preset_version if index is not None else gate.config_version(attempt.native.config, launch),
        is_control=index.is_control if index is not None else None,
        base_commit=index.base_commit if index is not None else None,
        base_preset_version=index.base_preset_version if index is not None else None,
        evaluation_data_identity=evaluation_data,
        user=index.user if index is not None else None,
        branch=index.branch if index is not None else provenance.get("branch"),
        commit=index.source_revision if index is not None else provenance.get("base_commit"),
        settings=list(map(visible_setting, inputs.settings)),
        launch_config=cast(dict, redact_value(attempt.native.config)),
        dirty=index.dirty if index is not None else provenance.get("dirty"),
        pr_number=index.pr_number if index is not None else None,
        cluster=launch.cluster,
        issue=lineage_issue(launch.export_root),
        run_date=started or tracking.get("started_at") or (index.submitted_at if index is not None else None),
        started_at=started,
        completed_at=finished if terminal else None,
        planned_updates=preset.recipe.layout.workload.update_count(preset.updates),
        measured_updates=list(measured.measured_updates),
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


def lineage_issue(export_root: str, requested: int | None = None, *, view: ReadView | None = None) -> int | None:
    """Read the run's issue and verify a proposed value against stored assignments."""
    issues = {attempt.issue for attempt in read_attempts(export_root, view=view).values() if attempt.issue is not None}
    if requested is not None:
        issues.add(requested)
    if len(issues) > 1:
        raise IssueConflict(f"Run has conflicting lineage issues {sorted(issues)} at {export_root}")
    return next(iter(issues), None)


def assign_issue(export_root: str, run_id: str, attempt_id: str, requested: int | None) -> int | None:
    with DataStore.open(run_records_root(export_root)) as store:
        while True:
            view = store.read_view()
            issue = lineage_issue(export_root, requested, view=view)
            if issue is None:
                return None
            try:
                update_attempt(store, AttemptRecord(run_id=run_id, attempt_id=attempt_id, issue=issue), base=view)
            except CommitConflict:
                continue
            return issue


def _coordinator_attempts(attempts: Mapping[str, AttemptRecord], coordinator: str) -> dict[str, AttemptRecord]:
    return {
        key: attempt
        for key, attempt in attempts.items()
        if attempt.config is not None and attempt.config["run"].get("coordinator_iris_job_id") == coordinator
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
        _, stored = read_stored_skyrl_attempts(path)
        for attempt in _coordinator_attempts(stored, coordinator).values():
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


def _prepare_publication_attempts(
    path: str,
    coordinator: str | None,
    interrupted: bool,
    cached_attempt: str | None,
) -> _OutputAttempts:
    current, stored = read_stored_skyrl_attempts(path)
    if coordinator is None:
        return _OutputAttempts(current, stored)
    attempts = _coordinator_attempts(stored, coordinator)
    if current not in attempts:
        if attempts:
            succeeded = [key for key, row in attempts.items() if row.native.result.state == "succeeded"]
            candidates = succeeded or list(attempts)
            current = candidates[0] if len(candidates) == 1 else None
        elif (
            cached_attempt is not None
            and cached_attempt in stored
            and stored[cached_attempt].native.result.state == "succeeded"
        ):
            current = cached_attempt
            attempts[current] = stored[current]
        else:
            current = None
    if interrupted:
        for key, record in attempts.items():
            if record.native.result.state.is_terminal or record.interruption is not None:
                continue
            export_root = recorded_launch(record).export_root
            write_attempt(
                export_root,
                AttemptRecord(run_id=record.run_id, attempt_id=key, interruption="coordinator_terminated"),
            )
            attempts[key] = read_attempt(export_root, key) or record.model_copy(
                update={"interruption": "coordinator_terminated"}
            )
    return _OutputAttempts(current, attempts)


def publish_runs(
    outputs: Sequence[str],
    preset: presets.Preset,
    *,
    table: presets.ModelPresets,
    gate_path: str | None = None,
    service_url: str | None = None,
    attempt_id: str | None = None,
    issue: int | None = None,
    coordinator: str | None = None,
    cached_attempts: Mapping[str, str] | None = None,
    interrupted: bool = False,
    submitted_gate: gate.GateAddress | None = None,
) -> PublicationOutcome:
    """Rebuild and upsert terminal or interrupted attempts, isolating publication failures."""
    if coordinator is None and issue is not None and (len(outputs) != 1 or attempt_id is None):
        raise ValueError("An issue requires exactly one output and its explicit attempt ID")
    result = PublicationOutcome()
    cache = cached_attempts or {}
    snapshots = {}
    for path in outputs:
        try:
            snapshots[path] = _prepare_publication_attempts(path, coordinator, interrupted, cache.get(path))
        except Exception as error:
            result.unserved_records.append(path)
            result.errors.append(str(error))
    report = None
    report_error = None
    try:
        candidates = tuple(
            snapshot.records.get(snapshot.current) if (snapshot := snapshots.get(path)) is not None else None
            for path in outputs
        )
        if submitted_gate is not None and (
            not interrupted
            or any(
                row is not None and row.attempt_id != cache.get(path)
                for path, row in zip(outputs, candidates, strict=True)
            )
        ):
            gate_path = submitted_gate.completed_path(candidates)
        report = gate.read_gate_report(gate_path)
        if report is not None:
            current_ids = tuple(row.attempt_id for row in candidates if row is not None)
            if len(current_ids) == len(outputs) and report.run.attempt_ids != current_ids:
                raise ValueError("Gate report attempts differ from the current output group; repeat with --wait")
            result.gate_path = gate_path
    except Exception as error:
        report = None
        report_error = str(error)
    for path, snapshot in snapshots.items():
        current_id, attempts = snapshot.current, snapshot.records
        if current_id in attempts:
            result.attempts[path] = attempts[current_id]
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
            if (
                interrupted
                and selected_id == cache.get(path)
                and receipt.native.config["run"].get("coordinator_iris_job_id") != coordinator
            ):
                continue
            if attempt_id is None and not receipt.native.result.state.is_terminal and receipt.interruption is None:
                continue
            destination = (receipt.index_run_id, selected_id)
            try:
                if not receipt.native.result.state.is_terminal and receipt.interruption is None:
                    raise ValueError("Publication requires a terminal native attempt")
                current_report = report if selected_id == current_id else None
                missing_gate = (
                    (submitted_gate is not None or gate_path is not None)
                    and selected_id == current_id
                    and receipt.native.result.state == "succeeded"
                    and current_report is None
                )
                if missing_gate and not interrupted:
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
                )
                if issue is not None:
                    assigned = assign_issue(recorded_launch(receipt).export_root, receipt.run_id, selected_id, issue)
                    record = record.model_copy(update={"issue": assigned})
                url = publish_result(record, service_url)
                result.served_records.append(destination)
                if (
                    (selected_id == current_id or attempt_id is not None)
                    and record.is_control
                    and record.status != "succeeded"
                ):
                    result.errors.append(
                        f"Control {record.run_id}/{record.attempt_id} is ineligible: {', '.join(record.failed_facts)}"
                    )
                if selected_id == current_id:
                    result.urls[path] = url
            except Exception as error:
                result.unserved_records.append(destination)
                result.errors.append(str(error))
    if result.unserved_records:
        result.retry = "Repeat the original command with --wait."
    return result


def command(table: presets.ModelPresets) -> click.Command:
    """Build the stored-attempt publication command for a model table."""

    @click.command()
    @click.option("--preset", "preset_name", type=click.Choice(tuple(table.by_name)), required=True)
    @click.option("--marina-url", "url", help="Marina origin (defaults to the Marina service).")
    @click.option("--run-output", required=True, help="Stored native run output.")
    @click.option("--attempt", required=True, help="Attempt ID within the stored native run.")
    @click.option("--issue", type=click.IntRange(min=1), help="Associate an existing native attempt with an issue.")
    def command(
        preset_name: str,
        url: str | None,
        run_output: str,
        attempt: str,
        issue: int | None,
    ) -> None:
        """Publish one stored native attempt."""
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
        click.echo(json.dumps(result.to_dict()))
        if result.unserved_records:
            raise click.exceptions.Exit(1)

    return command
