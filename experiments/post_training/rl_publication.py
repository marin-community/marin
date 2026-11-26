# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish native RL attempts from stored outputs on the launcher."""

from __future__ import annotations

import json
import math
import tomllib
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlencode

import click
from evalrecords.records import EvalRef, ModelRef, RLAttemptRef
from finestore.rl.rl_progress import (
    GRADER_REPORT_FILENAME,
    GRADER_REPORTS_DIR,
    AttemptSummary,
    ComparisonRole,
    GateResult,
)
from finestore.rl.run_records import (
    METRIC_ATTEMPT_IDS,
    AttemptRecord,
    EvaluationTaskRow,
    attempt_lineage,
    read_attempt_metrics,
    read_evaluations,
)
from marin.execution.artifact import read_record
from marin.rl import skyrl_records
from marin.rl.skyrl_records import attempt_tracking, read_stored_skyrl_attempts, recorded_launch
from pydantic import BaseModel, Field
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import rl_compose as recipe
from experiments.post_training import rl_gate as gate
from experiments.post_training import rl_presets as presets
from experiments.post_training.rl_command import attempt_grafana_url
from experiments.post_training.rl_evaluation_records import publish_skyrl_evaluation
from experiments.post_training.rl_measurements import (
    InvalidComparison,
    RunInputs,
    evaluation_status,
    native_prefix,
    read_run_inputs,
    training_invariants,
)

APPLET_ID = tomllib.loads(
    (Path(__file__).resolve().parents[2] / "infra/marina/apps/rl_progress/applet.toml").read_text()
)["applet_id"]


def publish_result(summary: AttemptSummary, service_url: str | None = None) -> str:
    """Upsert one native attempt and return its verified dashboard URL."""
    from marina import client  # noqa: PLC0415

    service_url = service_url or client.DEFAULT_MARINA_URL
    details = client.marina_request(service_url, "GET", f"/api/marina/applets/{APPLET_ID}")
    origin = client.applet_origin(service_url, details["authenticated_url"])
    served = AttemptSummary.model_validate_json(
        json.dumps(
            client.marina_request(
                origin, "POST", f"/a/{APPLET_ID}/api/runs", json_body=summary.model_dump(mode="json", exclude_unset=True)
            ),
            allow_nan=False,
        ),
        strict=True,
    )
    if (served.run_id, served.attempt_id) != (summary.run_id, summary.attempt_id):
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


def run_label(inputs: RunInputs) -> str:
    parts = [inputs.preset_name, *inputs.settings]
    if inputs.tasks:
        parts.append(f"tasks={inputs.tasks}")
    if inputs.init:
        parts.append(f"init={inputs.init.artifact_name}@{inputs.init.artifact_version}")
    return " ".join(parts)


def build_record(
    run_output: str,
    preset: presets.Preset,
    *,
    table: presets.ModelPresets,
    attempt: AttemptRecord,
    report: gate.GateReport | None = None,
    gate_name: str | None = None,
    evaldash_run_ids: tuple[str, ...] | None = None,
) -> AttemptSummary:
    """Build a terminal progress record from stored native outputs without live lookups."""
    launch, result = recorded_launch(attempt), attempt.native.result
    if result.state not in {"succeeded", "failed"}:
        raise ValueError("A progress record requires a terminal native attempt")
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
    history = attempt_lineage(launch.export_root, attempt_id)
    complete_history = (
        bool(history)
        and all(entry.record.metrics_started_at is not None for entry in history)
        and history[-1].record.resumed_step in (None, 0)
    )
    metrics = skyrl_records.read_skyrl_metrics(run_output, attempt_id=attempt_id)
    invariants = training_invariants(preset, metrics) if attempt.metrics_started_at is not None else ()
    failed = [fact.name for fact in invariants if not fact.holds]
    status = evaluation_status(run_output, preset, attempt, launch)
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
    windows = [
        (
            (entry.record.native.result.started_at, entry.record.native.result.finished_at)
            if entry.record.config is not None
            else (None, None)
        )
        for entry in history
    ]
    elapsed = None
    if complete_history and all(start is not None and finish is not None for start, finish in windows):
        durations = [recipe.utc_millis(finish) - recipe.utc_millis(start) for start, finish in windows]
        if any(value < 0 for value in durations):
            raise ValueError("Native completion precedes its recorded start")
        elapsed = sum(durations) / 1000
    else:
        evidence.append(
            "Checkpoint lineage timing is unknown: a shared attempt record or native time window is missing."
        )
    if not provenance:
        evidence.append("Recorded source provenance is unavailable.")
    report_path = StoragePath(launch.export_root) / GRADER_REPORTS_DIR / attempt_id / GRADER_REPORT_FILENAME
    try:
        grader_report = json.loads(report_path.read_text())
    except FileNotFoundError:
        grader_report = None
    return AttemptSummary(
        run_id=attempt.run_id,
        attempt_id=attempt_id,
        label=run_label(inputs),
        series=preset.name,
        model=launch.model_name,
        issue=attempt.issue,
        evaldash_run_ids=list(dict.fromkeys(evaldash_run_ids)) if evaldash_run_ids is not None else None,
        gate=gate_name,
        gate_result=(
            GateResult(
                verdict=report.verdict,
                effect=report.effect,
                reason=report.reason,
                invariants=list(report.invariants),
                comparisons=list(
                    report.rows(ComparisonRole.OBJECTIVE, ComparisonRole.GUARD, ComparisonRole.REQUIREMENT)
                ),
                attempt_ids=list(report.run.attempt_ids),
                replicates=report.run.replicates,
            )
            if report is not None
            else None
        ),
        run_date=started or tracking.get("started_at"),
        started_at=started,
        completed_at=finished,
        elapsed_seconds=elapsed,
        timing_scope="checkpoint_lineage_native_jobs",
        status="failed" if failed else "succeeded",
        failed_facts=failed,
        optimizer_steps=optimizer_steps,
        planned_updates=preset.recipe.layout.workload.update_count(preset.updates),
        measured_updates=list(preset.measured),
        iris_job_id=result.iris_job_id,
        wandb_url=tracking.get("wandb_url"),
        grafana_url=attempt_grafana_url(attempt, launch),
        evaluations=evaluations,
        grader_report=grader_report,
        engine=tracking.get("engine"),
        source_revision=(
            f"base={provenance.get('base_commit')} tree={provenance.get('tree_hash')}" if provenance else "Unknown"
        ),
        data_identity="; ".join(source.identity for source in launch.train_data),
        initializer_identity=launch.model_identity,
        artifact_uri=run_output,
        evidence=evidence,
    )


def _publish_training_evaluations(preset: presets.Preset, attempt: AttemptRecord) -> tuple[str, ...]:
    if attempt.evaluation_storage_version != 1:
        return ()
    launch = recorded_launch(attempt)
    if launch.run_mode != "train":
        return ()
    result = attempt.native.result
    names = dict.fromkeys((*(evaluation.name for evaluation in preset.evaluations[:1]), *launch.evaluation_data))
    attempt_ref = RLAttemptRef(run_id=attempt.run_id, attempt_id=attempt.attempt_id)
    exported_model = result.model
    metrics = {
        step: {key: values[key] for key, owner in values[METRIC_ATTEMPT_IDS].items() if owner == attempt.attempt_id}
        for step, values in read_attempt_metrics(launch.export_root, attempt.attempt_id).items()
    }
    ids = []
    for name in names:
        prefix = native_prefix(launch, name)
        sessions: dict[int, list[EvaluationTaskRow]] = defaultdict(list)
        for row in read_evaluations(launch.export_root, attempt_id=attempt.attempt_id, evaluation=prefix):
            sessions[row.step].append(row)
        attempted_key = f"{prefix}all/num_attempted"
        complete_steps = []
        for step, rows in sessions.items():
            attempted = metrics.get(step, {}).get(attempted_key)
            if attempted == sum(len(row.answers) for row in rows) or (attempted is None and result.state == "succeeded"):
                complete_steps.append(step)
        if not complete_steps:
            continue
        for step in dict.fromkeys((min(complete_steps), max(complete_steps))):
            session_metrics = metrics.get(step, {})
            score = session_metrics.get(f"{prefix}all/avg_verifier_score")
            record = publish_skyrl_evaluation(
                attempt=attempt,
                evaluation=EvalRef(name=name, mechanism="skyrl"),
                model=ModelRef(
                    name=launch.model_name or preset.name,
                    location=(
                        exported_model.policy_export_uri
                        if exported_model is not None and exported_model.global_step == step
                        else (
                            launch.model_uri
                            if step <= 0 and not attempt.resumed_step
                            else f"skyrl:{attempt.run_id}/{attempt.attempt_id}@{step}"
                        )
                    ),
                    backend=launch.engine.backend,
                ),
                rows=sessions[step],
                rl_attempt=attempt_ref,
                score=score if score is not None and math.isfinite(score) else None,
            )
            ids.append(record.run_id)
    return tuple(ids)


def publish_runs(
    outputs: Sequence[str],
    preset: presets.Preset,
    *,
    table: presets.ModelPresets,
    gate_path: str | None = None,
    service_url: str | None = None,
    attempt_id: str | None = None,
    evaldash_run_ids: tuple[str, ...] | None = None,
) -> PublicationOutcome:
    """Rebuild and upsert each stored attempt, isolating publication failures."""
    if evaldash_run_ids is not None and (len(outputs) != 1 or attempt_id is None):
        raise ValueError("EvalDash run IDs require exactly one output and its explicit attempt ID")
    result = PublicationOutcome()
    snapshots = {}
    for path in outputs:
        try:
            snapshots[path] = read_stored_skyrl_attempts(path)
        except Exception as error:
            result.unserved_records.append(path)
            result.errors.append(str(error))
    report = None
    report_error = None
    try:
        report = gate.read_gate_report(gate_path)
        if report is not None:
            current_ids = tuple(
                snapshots[path][0] for path in outputs if path in snapshots and snapshots[path][0] in snapshots[path][1]
            )
            if report.run.replicates != len(outputs) or report.run.attempt_ids != current_ids:
                raise ValueError("Gate report attempts differ from the current output group; repeat with --wait")
    except Exception as error:
        report = None
        report_error = str(error)
    for path, (current_id, attempts) in snapshots.items():
        if current_id not in attempts:
            result.unserved_records.append(path)
            result.errors.append(f"No stored native receipt at {path} for current attempt {current_id}")
        selected = (
            attempts if attempt_id is None else {key: value for key, value in attempts.items() if key == attempt_id}
        )
        if attempt_id is not None and not selected:
            result.unserved_records.append((path, attempt_id))
            result.errors.append(f"No stored native receipt at {path} for {attempt_id}")
        for selected_id, receipt in selected.items():
            destination = (receipt.run_id, selected_id)
            try:
                if gate_path is not None and selected_id == current_id and receipt.native.result.state == "succeeded":
                    if report_error:
                        raise ValueError(report_error)
                    if report is None:
                        raise ValueError("Gate report missing for this attempt; repeat the original command with --wait")
                current_report = report if selected_id == current_id else None
                record = build_record(
                    path,
                    preset,
                    table=table,
                    attempt=receipt,
                    report=current_report,
                    gate_name=read_record(gate_path).name if current_report is not None else None,
                    evaldash_run_ids=evaldash_run_ids,
                )
                try:
                    evaluation_ids = _publish_training_evaluations(table.by_name[record.series], receipt)
                except Exception as error:
                    result.errors.append(f"EvalDash publication for {receipt.run_id}/{selected_id}: {error}")
                    evaluation_ids = ()
                if evaluation_ids:
                    record = record.model_copy(
                        update={"evaldash_run_ids": list(dict.fromkeys((*evaluation_ids, *(evaldash_run_ids or ()))))}
                    )
                url = publish_result(record, service_url)
                result.served_records.append(destination)
                if selected_id == current_id:
                    result.urls[path] = url
            except Exception as error:
                result.unserved_records.append(destination)
                result.errors.append(str(error))
    if result.errors:
        result.retry = "Repeat the original command with --wait."
    return result


def publish_controls(
    table: presets.ModelPresets, preset_name: str, *, service_url: str | None = None
) -> PublicationOutcome:
    """Publish the exact pinned controls from their stored native outputs."""
    preset = table.by_name.get(preset_name)
    if preset is None or not (preset.publishes and preset.gated):
        raise click.BadParameter("Choose a dashboard preset with controls", param_hint="--preset")
    try:
        pin = gate.control_pin(preset, table=table)
    except InvalidComparison as error:
        raise click.UsageError(str(error)) from error
    configure_coreweave_s3()
    result = publish_runs(
        [gate.control_path(preset, pin, index, table=table) for index in range(pin.replicates)],
        table.by_name[pin.preset],
        table=table,
        service_url=service_url,
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
    @click.option("--evaldash-run-id", multiple=True, help="Link an EvalDash run to this attempt. Repeatable.")
    def command(
        preset_name: str, url: str | None, run_output: str | None, attempt: str | None, evaldash_run_id: tuple[str, ...]
    ) -> None:
        """Publish pinned controls or one stored native attempt."""
        if (run_output is None) != (attempt is None):
            raise click.UsageError("--run-output and --attempt are required together")
        if evaldash_run_id and attempt is None:
            raise click.UsageError("--evaldash-run-id requires --run-output and --attempt")
        if run_output is not None:
            configure_coreweave_s3()
            result = publish_runs(
                [run_output],
                table.by_name[preset_name],
                table=table,
                service_url=url,
                attempt_id=attempt,
                evaldash_run_ids=evaldash_run_id or None,
            )
            if result.errors:
                result.retry = "Repeat this publication command with --run-output and --attempt."
        else:
            result = publish_controls(table, preset_name, service_url=url)
        click.echo(json.dumps(result.to_dict()))
        if result.errors:
            raise click.exceptions.Exit(1)

    return command
