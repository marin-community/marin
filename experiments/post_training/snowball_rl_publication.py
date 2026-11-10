# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish native Snowball attempts from stored outputs on the launcher."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import click
from marin.execution.artifact import RECORD_FILENAME, read_record
from marin.rl import skyrl_records
from marin.rl.rl_progress import RunSummary
from marin.rl.skyrl_records import read_stored_skyrl_attempts
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import publish_rl_progress
from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets


@dataclass
class PublicationOutcome:
    served_records: list[str] = field(default_factory=list)
    unserved_records: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    urls: dict[str, str] = field(default_factory=dict)
    retry: str | None = None

    def to_dict(self) -> dict:
        return {
            "served_records": self.served_records,
            "unserved_records": self.unserved_records,
            "errors": self.errors,
        } | ({"retry": self.retry} if self.retry is not None else {})


def publishes(preset: presets.Preset) -> bool:
    return not preset.recipe.layout.role_plan.generate_only and preset.focus in (
        presets.Focus.LEARNING,
        presets.Focus.DATA,
        presets.Focus.TRAIN,
    )


CONTROL_PRESETS = tuple(
    sorted(
        name
        for name, preset in presets.PRESETS.items()
        if publishes(preset)
        and any(
            pin.revision == presets.PRESET_REVISION
            for pin in gate.CONTROLS.get(gate.SHARED_CONTROLS.get(name, name), ())
        )
    )
)


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
    attempt_id: str | None = None,
    attempts: tuple[str | None, dict[str, skyrl_records.SkyRLLaunchReceipt]] | None = None,
) -> dict:
    """Build a terminal progress record from stored native outputs without live lookups."""
    current_id, receipts = attempts if attempts is not None else read_stored_skyrl_attempts(run_output)
    selected_id = attempt_id or current_id
    receipt = receipts.get(selected_id) if selected_id is not None else None
    if receipt is None:
        raise FileNotFoundError(f"No stored native receipt at {run_output} for {attempt_id}")
    launch, result = receipt.config, receipt.result
    if result.state not in {"succeeded", "failed"}:
        raise ValueError("A progress record requires a terminal native attempt")
    evidence = []
    try:
        inputs = gate.read_gate_inputs(run_output)
    except FileNotFoundError:
        inputs = gate.GateInputs(preset_name=preset.name)
        evidence.append("Stored Snowball inputs are unavailable; using the supplied preset.")
    preset = replace(presets.PRESETS[inputs.preset_name], updates=launch["skyrl"]["trainer"]["max_steps"])
    if not publishes(preset):
        raise ValueError("Progress publication requires a trained-panel preset")
    native = launch["run"]
    attempt_id = native["attempt_id"]
    provenance = result.provenance or {}
    tracking = skyrl_records.read_skyrl_attempt_tracking(run_output, attempt_id)
    started, finished = result.started_at, result.finished_at
    metrics = skyrl_records.read_skyrl_metrics(run_output)
    if attempt_id != current_id:
        selected_steps = [step for step, row in metrics.items() if row.get("attempt_id") == attempt_id]
        last_step = max(selected_steps, default=-1)
        metrics = {step: row for step, row in metrics.items() if step <= last_step}
    attributable = attempt_id == current_id or bool(metrics)
    invariants = gate.training_invariants(preset, metrics) if attributable else ()
    failed = [fact.name for fact in invariants if not fact.holds]
    if result.state != "succeeded":
        failed.insert(0, "native_attempt")
    optimizer_steps = (
        next(int(fact.value) for fact in invariants if fact.name == "planned_updates") if attributable else None
    )
    lineage_ids = {attempt_id} | {row["attempt_id"] for row in metrics.values() if row.get("attempt_id")}
    lineage_known = attributable and (
        len(receipts) == 1 or (bool(metrics) and all(row.get("attempt_id") for row in metrics.values()))
    )
    windows = [
        (receipts[key].result.started_at, receipts[key].result.finished_at) if key in receipts else (None, None)
        for key in lineage_ids
    ]
    elapsed = None
    if lineage_known and all(start is not None and finish is not None for start, finish in windows):
        durations = [presets.utc_millis(finish) - presets.utc_millis(start) for start, finish in windows]
        if any(value < 0 for value in durations):
            raise ValueError("Native completion precedes its recorded start")
        elapsed = sum(durations) / 1000
    else:
        evidence.append("Checkpoint lineage timing is unknown: attempt attribution or a native time window is missing.")
    if not attributable:
        evidence.append("Historical checkpoint updates are unknown without stored per-step attempt attribution.")
    if not provenance:
        evidence.append("Recorded source provenance is unavailable.")
    record = {
        "run_id": native["id"],
        "attempt_id": attempt_id,
        "label": label(inputs),
        "series": preset.name,
        "issue": None,
        "verdict": None,
        "gate": None,
        "gate_result": None,
        "min_scored_fraction": presets.MINIMUM_PANEL_COVERAGE,
        "date_basis": "launch_started",
        "run_date": started or tracking.get("started_at"),
        "started_at": started,
        "completed_at": finished,
        "elapsed_seconds": elapsed,
        "timing_scope": "checkpoint_lineage_native_jobs",
        "status": "failed" if failed else "succeeded",
        "failed_facts": failed,
        "optimizer_steps": optimizer_steps,
        "iris_job_id": result.iris_job_id,
        "wandb_url": tracking.get("wandb_url"),
        "engine": tracking.get("engine"),
        "source_revision": (
            f"base={provenance.get('base_commit')} tree={provenance.get('tree_hash')}" if provenance else "Unknown"
        ),
        "data_identity": "; ".join(row["identity"] for row in launch["inputs"]["train_data"]),
        "initializer_identity": launch["inputs"]["model"]["identity"],
        "evaluation": None,
        "baseline": None,
        "final": None,
        "artifact_uri": run_output,
        "evidence": evidence,
    }
    RunSummary.model_validate(record)
    return record


def gate_complete(gate_path: str | None) -> bool:
    """A completed gate has its report and the executor's artifact record."""
    return gate_path is not None and all(
        (StoragePath(gate_path) / filename).exists() for filename in (gate.GATE_REPORT_FILENAME, RECORD_FILENAME)
    )


def publish_runs(
    outputs: Sequence[str],
    preset: presets.Preset,
    *,
    gate_path: str | None = None,
    gate_required: bool = False,
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
        if gate_complete(gate_path):
            report = gate.GateReport.model_validate_json(
                (StoragePath(gate_path) / gate.GATE_REPORT_FILENAME).read_text()
            )
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
            destination = str(StoragePath(path) / "exports/attempts" / attempt_id)
            try:
                if gate_required and attempt_id == current_id and receipt.result.state == "succeeded":
                    if report_error:
                        raise ValueError(report_error)
                    if report is None or attempt_id not in report.run.attempt_ids:
                        raise ValueError("Gate report missing for this attempt; repeat the original command with --wait")
                record = build_record(path, preset, attempt_id=attempt_id, attempts=(current_id, attempts))
                if attempt_id == current_id:
                    record["issue"] = issue
                    if report is not None and attempt_id in report.run.attempt_ids:
                        record.update(
                            gate=read_record(gate_path).name,
                            verdict=report.verdict.value,
                            gate_result={
                                **report.model_dump(mode="json", include={"verdict", "label", "reason", "invariants"}),
                                "comparisons": [
                                    row.model_dump(mode="json")
                                    for row in report.comparisons
                                    if row.role
                                    in (
                                        gate.ComparisonRole.OBJECTIVE,
                                        gate.ComparisonRole.CONSTRAINT,
                                        gate.ComparisonRole.REQUIREMENT,
                                    )
                                ],
                                "attempt_ids": list(report.run.attempt_ids),
                                "replicates": report.run.replicates,
                            },
                        )
                        if len(report.run.attempt_ids) == report.run.replicates:
                            replicate = report.run.attempt_ids.index(attempt_id)
                            record["failed_facts"] = list(
                                dict.fromkeys(
                                    [
                                        *record["failed_facts"],
                                        *(
                                            fact.name
                                            for fact in report.invariants
                                            if not fact.holds and fact.replicate == replicate
                                        ),
                                    ]
                                )
                            )
                            record["status"] = "failed" if record["failed_facts"] else "succeeded"
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
    if preset_name not in CONTROL_PRESETS:
        raise click.UsageError("Progress publication requires a trained-panel preset with current control pins")
    preset = presets.PRESETS[preset_name]
    result = PublicationOutcome()
    for pin in gate.control_pins(preset_name):
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
    type=click.Choice(CONTROL_PRESETS),
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
