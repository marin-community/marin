# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish native Snowball attempts from stored outputs on the launcher."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import click
from finestore.rl.run_records import AttemptRecord, attempt_lineage
from marin.execution.artifact import read_record
from marin.rl import skyrl_records
from marin.rl.rl_progress import GateResult, RunSummary
from marin.rl.skyrl_records import attempt_tracking, read_stored_skyrl_attempts, recorded_launch
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training import publish_rl_progress
from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets


@dataclass
class PublicationOutcome:
    served_records: list[tuple[str, str]] = field(default_factory=list)
    unserved_records: list[tuple[str, str] | str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    urls: dict[str, str] = field(default_factory=dict)
    retry: str | None = None

    def to_dict(self) -> dict:
        return {
            "served_records": [list(key) for key in self.served_records],
            "unserved_records": [list(key) if isinstance(key, tuple) else key for key in self.unserved_records],
            "errors": self.errors,
        } | ({"retry": self.retry} if self.retry is not None else {})


def label(inputs: gate.GateInputs) -> str:
    parts = [inputs.preset_name, *inputs.settings]
    if inputs.tasks:
        parts.append(f"tasks={inputs.tasks}")
    if inputs.init:
        parts.append(f"init={inputs.init}")
    return " ".join(parts)


def build_record(
    run_output: str,
    preset: presets.Preset,
    *,
    attempt_id: str,
    attempts: tuple[str | None, dict[str, AttemptRecord]],
    report: gate.GateReport | None = None,
    gate_name: str | None = None,
    issue: int | None = None,
) -> RunSummary:
    """Build a terminal progress record from stored native outputs without live lookups."""
    _, receipts = attempts
    receipt = receipts.get(attempt_id)
    if receipt is None:
        raise FileNotFoundError(f"No stored native receipt at {run_output} for {attempt_id}")
    launch, result = recorded_launch(receipt), receipt.native.result
    if result.state not in {"succeeded", "failed"}:
        raise ValueError("A progress record requires a terminal native attempt")
    evidence = []
    try:
        inputs = gate.read_gate_inputs(run_output)
    except FileNotFoundError:
        inputs = gate.GateInputs(preset_name=preset.name)
        evidence.append("Stored Snowball inputs are unavailable; using the supplied preset.")
    preset = presets.PRESETS[inputs.preset_name]
    if launch.max_steps is None:
        raise ValueError("Stored native max_steps is unavailable")
    preset = replace(
        preset, updates=launch.max_steps, measured=preset.recipe.layout.workload.planned_steps(launch.max_steps)
    )
    attempt_id = launch.attempt_id
    provenance = result.provenance or {}
    tracking = attempt_tracking(receipt, launch)
    started, finished = result.started_at, result.finished_at
    history = attempt_lineage(launch.export_root, attempt_id)
    complete_history = (
        bool(history)
        and all(entry.record.metrics_started_at is not None for entry in history)
        and history[-1].record.resumed_step in (None, 0)
    )
    metrics = skyrl_records.read_skyrl_metrics(run_output, attempt_id=attempt_id)
    invariants = gate.training_invariants(preset, metrics) if receipt.metrics_started_at is not None else ()
    failed = [fact.name for fact in invariants if not fact.holds]
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
        durations = [presets.utc_millis(finish) - presets.utc_millis(start) for start, finish in windows]
        if any(value < 0 for value in durations):
            raise ValueError("Native completion precedes its recorded start")
        elapsed = sum(durations) / 1000
    else:
        evidence.append(
            "Checkpoint lineage timing is unknown: a shared attempt record or native time window is missing."
        )
    if not provenance:
        evidence.append("Recorded source provenance is unavailable.")
    if report is not None and len(report.run.attempt_ids) == report.run.replicates:
        replicate = report.run.attempt_ids.index(attempt_id)
        failed = list(
            dict.fromkeys(
                [*failed, *(fact.name for fact in report.invariants if not fact.holds and fact.replicate == replicate)]
            )
        )
    return RunSummary(
        run_id=receipt.run_id,
        attempt_id=attempt_id,
        label=label(inputs),
        series=preset.name,
        issue=issue,
        gate=gate_name,
        gate_result=(
            GateResult(
                verdict=report.verdict,
                label=report.label,
                reason=report.reason,
                invariants=list(report.invariants),
                comparisons=list(
                    report.rows(
                        gate.ComparisonRole.OBJECTIVE, gate.ComparisonRole.CONSTRAINT, gate.ComparisonRole.REQUIREMENT
                    )
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
        iris_job_id=result.iris_job_id,
        wandb_url=tracking.get("wandb_url"),
        engine=tracking.get("engine"),
        source_revision=(
            f"base={provenance.get('base_commit')} tree={provenance.get('tree_hash')}" if provenance else "Unknown"
        ),
        data_identity="; ".join(source.identity for source in launch.train_data),
        initializer_identity=launch.model_identity,
        artifact_uri=run_output,
        evidence=evidence,
    )


def publish_runs(
    outputs: Sequence[str],
    preset: presets.Preset,
    *,
    gate_path: str | None = None,
    issue: int | None = None,
    service_url: str | None = None,
) -> PublicationOutcome:
    """Rebuild and upsert each stored attempt, isolating publication failures."""
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
        for attempt_id, receipt in attempts.items():
            destination = (receipt.run_id, attempt_id)
            try:
                if gate_path is not None and attempt_id == current_id and receipt.native.result.state == "succeeded":
                    if report_error:
                        raise ValueError(report_error)
                    if report is None:
                        raise ValueError("Gate report missing for this attempt; repeat the original command with --wait")
                current_report = report if attempt_id == current_id else None
                record = build_record(
                    path,
                    preset,
                    attempt_id=attempt_id,
                    attempts=(current_id, attempts),
                    report=current_report,
                    gate_name=read_record(gate_path).name if current_report is not None else None,
                    issue=issue if attempt_id == current_id else None,
                )
                url = publish_rl_progress.publish_result(record, service_url)
                result.served_records.append(destination)
                if attempt_id == current_id:
                    result.urls[path] = url
            except Exception as error:
                result.unserved_records.append(destination)
                result.errors.append(str(error))
    if result.unserved_records:
        result.retry = "Repeat the original command with --wait."
    return result


def publish_controls(preset_name: str, *, service_url: str | None = None) -> PublicationOutcome:
    """Publish the exact pinned controls from their stored native outputs."""
    preset = presets.PRESETS[preset_name]
    result = PublicationOutcome()
    try:
        pins = gate.control_pins(preset)
    except gate.InvalidComparison as error:
        raise click.UsageError(str(error)) from error
    for pin in pins:
        outcome = publish_runs(
            [gate.control_path(preset, pin, index) for index in range(pin.runs)],
            presets.PRESETS[pin.preset],
            service_url=service_url,
        )
        result.served_records.extend(outcome.served_records)
        result.unserved_records.extend(outcome.unserved_records)
        result.errors.extend(outcome.errors)
        result.urls.update(outcome.urls)
    if result.unserved_records:
        result.retry = f"Repeat python -m experiments.post_training.snowball_rl_publication --preset {preset_name}."
    return result


@click.command()
@click.option(
    "--preset",
    "preset_name",
    type=click.Choice([name for name, preset in presets.PRESETS.items() if preset.publishes]),
    required=True,
)
@click.option("--url", help="Marina origin (defaults to the Marina service).")
def main(preset_name: str, url: str | None) -> None:
    """Publish a preset's pinned controls from their stored native outputs."""
    configure_coreweave_s3()
    result = publish_controls(preset_name, service_url=url)
    click.echo(json.dumps(result.to_dict()))
    if result.unserved_records:
        raise click.exceptions.Exit(1)


if __name__ == "__main__":
    main()
