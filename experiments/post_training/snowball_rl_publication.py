# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain native Snowball attempts and publish them from the launcher."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from typing import NamedTuple

import click
import pyarrow as pa
import pyarrow.compute as pc
from marin.execution.artifact import RECORD_FILENAME, Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.rl import skyrl
from marin.rl.rl_progress import RunSummary, membership_sha256, protocol_sha256
from marin.rl.skyrl import read_skyrl_launch_config, read_stored_skyrl_receipt
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import publish_rl_progress
from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_scoring as scoring

RESULT_FILENAME = "progress-record.json"
PUBLICATION_FILENAME = "publication.json"


def publishes(preset: presets.Preset) -> bool:
    return not preset.recipe.layout.role_plan.generate_only and preset.focus in (
        presets.Focus.LEARNING,
        presets.Focus.DATA,
        presets.Focus.TRAIN,
    )


def _phase_result(
    scores: pa.Table,
    evaluation: presets.Evaluation,
    step: int,
    metrics: dict[int, dict],
    prefix: str,
    task_ids: set[str],
    evidence: list[str],
) -> dict | None:
    table = scores.filter(pc.equal(scores["step"], step))
    rows = table.to_pylist()
    if not rows:
        return None
    counts = Counter(row["item"] for row in rows)
    if set(counts) != task_ids:
        evidence.append(f"Evaluation step {step} task IDs differ from its recorded input cohort.")
        return None
    if any(count != evaluation.answers for count in counts.values()):
        evidence.append(f"Evaluation step {step} has an unexpected number of answers per task.")
        return None
    observation = metrics.get(step, {})
    scored = sum(row["scored"] for row in rows)
    if (
        observation.get(prefix + "all/num_attempted") != len(rows)
        or observation.get(prefix + "all/num_scored") != scored
    ):
        evidence.append(f"Evaluation step {step} lacks matching lineage counters.")
        return None
    values = scoring.evaluation_statistics(table)
    reward = values.get(evaluation.name + "/family_mean")
    if reward is None:
        evidence.append(f"Evaluation step {step} has no scored reward.")
        return None
    scored_tasks = {row["item"] for row in rows if row["scored"]}
    return {"reward": reward, "attempted": len(counts), "scored": len(scored_tasks)}


def record_path(run_output: str, attempt_id: str) -> StoragePath:
    return StoragePath(run_output) / "exports/attempts" / attempt_id / RESULT_FILENAME


def publication_path(run_output: str, attempt_id: str) -> StoragePath:
    return StoragePath(run_output) / "publication" / attempt_id


def stored_attempts(run_output: str) -> dict[str, skyrl.SkyRLLaunchReceipt]:
    """Discover native attempts from their stored receipts, including attempts with no progress record."""
    current = read_stored_skyrl_receipt(run_output)
    try:
        launch = current.config if current is not None else read_skyrl_launch_config(run_output)
    except FileNotFoundError:
        return {}
    attempts = {current.config["run"]["attempt_id"]: current} if current is not None else {}
    for path in scoring.paths(StoragePath(launch["artifacts"]["attempts_root"]) / "*.json"):
        attempt_id = path.name.removesuffix(".json")
        receipt = read_stored_skyrl_receipt(run_output, attempt_id)
        if receipt is not None:
            attempts[attempt_id] = receipt
    return attempts


def label(preset: presets.Preset, inputs: gate.GateInputs) -> str:
    parts = [preset.name]
    for text in inputs.settings:
        key, value = text.split("=", 1)
        parsed = json.loads(value)
        parts.append(f"{key.rsplit('.', 1)[-1]}={parsed if isinstance(parsed, str) else json.dumps(parsed)}")
    return " ".join(parts)


def build_record(
    run_output: str,
    preset: presets.Preset,
    *,
    attempt_id: str | None = None,
) -> dict:
    """Build a terminal progress record from stored native outputs without launching or live lookups."""
    if not publishes(preset):
        raise ValueError("Progress publication requires a trained-panel preset")
    receipt = read_stored_skyrl_receipt(run_output, attempt_id)
    if receipt is None:
        raise FileNotFoundError(f"No stored native receipt at {run_output} for {attempt_id}")
    launch, result = receipt.config, receipt.result
    if result.state not in {"succeeded", "failed"}:
        raise ValueError("A progress record requires a terminal native attempt")
    preset = replace(preset, updates=launch["skyrl"]["trainer"]["max_steps"])
    native = launch["run"]
    attempt_id = native["attempt_id"]
    destination = record_path(run_output, attempt_id)
    evidence = []
    provenance = result.provenance or {}
    options = gate.recorded_run_options(provenance.get("command_line", []))
    tracking = skyrl.read_skyrl_attempt_tracking(run_output, attempt_id)
    attempts = stored_attempts(run_output)
    current = read_stored_skyrl_receipt(run_output)
    latest = current is not None and current.config["run"]["attempt_id"] == attempt_id
    started, finished = result.started_at, result.finished_at
    metrics = gate.read_metrics(StoragePath(run_output))
    if not latest:
        selected_steps = [step for step, row in metrics.items() if row.get("attempt_id") == attempt_id]
        last_step = max(selected_steps, default=-1)
        metrics = {step: row for step, row in metrics.items() if step <= last_step}
    checked_metrics = (
        metrics
        if options.resume_from is None
        else {step: row for step, row in metrics.items() if "policy/policy_update_steps" in row}
    )
    invariants = gate.training_invariants(preset, checked_metrics)
    failed = [
        fact.name
        for fact in invariants
        if not fact.holds and (fact.name != "planned_updates" or options.resume_from is None)
    ]
    if options.resume_from:
        evidence.append("The resume step is not stored; baseline and planned-update validation remain unknown.")
    if result.state != "succeeded":
        failed.insert(0, "native_attempt")
    optimizer_steps = sum(
        int(row.get("policy/policy_update_steps", 0)) for step, row in metrics.items() if 0 < step <= preset.updates
    )
    lineage_ids = {attempt_id} | {row["attempt_id"] for row in metrics.values() if row.get("attempt_id")}
    windows = [
        (attempts[key].result.started_at, attempts[key].result.finished_at) if key in attempts else (None, None)
        for key in lineage_ids
    ]
    elapsed = None
    if all(start is not None and finish is not None for start, finish in windows):
        durations = [presets.utc_millis(finish) - presets.utc_millis(start) for start, finish in windows]
        if any(value < 0 for value in durations):
            raise ValueError("Native completion precedes its recorded start")
        elapsed = sum(durations) / 1000
    else:
        evidence.append("A lineage attempt's native time window is unavailable; elapsed time remains unknown.")
    gain = next(
        (metric.key.split("/", 1)[0] for metric in preset.objectives if "/gain" in metric.key),
        preset.evaluations[0].name,
    )
    evaluation = next(item for item in preset.evaluations if item.name == gain)
    sources = scoring.recorded_sources(launch, evaluation.name)
    panel = scoring.read_recorded_panel(launch, evaluation.name)
    task_ids = list(panel.task_ids)
    protocol = scoring.evaluation_protocol(launch, evaluation.name)
    evaluation = replace(evaluation, answers=protocol["n_samples_per_prompt"])
    phases = [None, None]
    try:
        scores = scoring.panel_scores(evaluation, run_output) if options.resume_from is None else None
        phases = (
            [
                _phase_result(
                    scores,
                    evaluation,
                    step,
                    metrics,
                    scoring.native_prefix(launch, evaluation.name),
                    set(task_ids),
                    evidence,
                )
                for step in (0, preset.updates)
            ]
            if scores is not None
            else phases
        )
    except scoring.InvalidComparison as error:
        evidence.append(f"Evaluation measurements unavailable: {error}")
    if phases[0] is None and phases[1] is not None:
        evidence.append("Final gain lacks an attributable baseline measurement.")
        phases[1] = None
    if not provenance:
        evidence.append("Recorded source provenance is unavailable.")
    record = {
        "run_id": native["id"],
        "attempt_id": attempt_id,
        "label": label(preset, options.inputs),
        "series": preset.name,
        "issue": None,
        "verdict": None,
        "gate": None,
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
        "resume_from": options.resume_from,
        "iris_job_id": result.iris_job_id,
        "wandb_url": tracking.get("wandb_url"),
        "engine": tracking.get("engine"),
        "source_revision": (
            f"base={provenance.get('base_commit')} tree={provenance.get('tree_hash')}" if provenance else "Unknown"
        ),
        "data_identity": "; ".join(row["identity"] for row in launch["inputs"]["train_data"]),
        "initializer_identity": launch["inputs"]["model"]["identity"],
        "evaluation": {
            "name": evaluation.name,
            "version": "; ".join(source["identity"] for source in sources),
            "task_count": len(task_ids),
            "task_set_sha256": panel.task_set_sha256,
            "membership_sha256": membership_sha256(task_ids),
            "protocol": protocol,
            "protocol_sha256": protocol_sha256(protocol),
        },
        "baseline": phases[0],
        "final": phases[1],
        "artifact_uri": str(destination),
        "evidence": evidence,
    }
    RunSummary.model_validate(record)
    return record


def retain_record(run_output: str, preset: presets.Preset, *, attempt_id: str | None = None) -> dict:
    """Build and retain one terminal record from its stored native outputs."""
    record = build_record(run_output, preset, attempt_id=attempt_id)
    destination = StoragePath(record["artifact_uri"])
    destination.parent.mkdirs()
    destination.write_text(json.dumps(record, allow_nan=False, indent=2))
    return record


def _publish_attempt(record: dict, service_url: str | None, output: str) -> Artifact:
    root = StoragePath(output)
    frozen = root / RESULT_FILENAME
    if not presets.exists(frozen):
        root.mkdirs()
        frozen.write_text(json.dumps(record, allow_nan=False, indent=2))
    receipt = publish_rl_progress.publish_result(json.loads(presets.read_text(frozen)), service_url)
    (root / PUBLICATION_FILENAME).write_text(json.dumps(receipt, indent=2))
    return Artifact(path=output)


def publication_step(
    run_output: str,
    name: str,
    version: str,
    record: dict,
    *,
    gate_path: str | None = None,
    issue: int | None = None,
    service_url: str | None = None,
) -> ArtifactStep[Artifact]:
    """Publish one native pair using its first retained publication payload and executor lock."""
    record = dict(record, issue=issue)
    gate_record = (
        presets.artifact_record(gate_path)
        if gate_path is not None and presets.exists(StoragePath(gate_path) / RECORD_FILENAME)
        else None
    )
    if gate_record is not None:
        record.update(gate=gate_record.name, verdict=gate_record.result["verdict"])
    attempt_id = record["attempt_id"]
    step = apply(
        f"{name}/publication/{attempt_id}",
        _publish_attempt,
        version=version,
        record=record,
        service_url=service_url,
        output=OUT,
    )
    return replace(step, override_path=str(publication_path(run_output, attempt_id)))


def publish_run(
    run_output: str,
    name: str,
    version: str,
    preset: presets.Preset,
    *,
    gate_path: str | None = None,
    gate_required: bool = False,
    issue: int | None = None,
    service_url: str | None = None,
) -> dict:
    """Build and publish terminal attempts on the caller, isolating each publication failure."""
    from marin.execution.lazy import run  # noqa: PLC0415
    from rigging.auth import IapCredentialsUnavailable, IapLoginRequired  # noqa: PLC0415

    result = {"served_records": [], "unserved_records": [], "errors": []}
    path = run_output
    try:
        attempts = stored_attempts(path)
        if not attempts:
            raise FileNotFoundError(f"No stored native receipt at {path}")
    except Exception as error:
        attempts = {}
        result["unserved_records"].append(path)
        result["errors"].append(str(error))
    current = read_stored_skyrl_receipt(path) if attempts else None
    current_id = current.config["run"]["attempt_id"] if current is not None else None
    for attempt_id in attempts:
        destination = record_path(path, attempt_id)
        try:
            if (
                gate_required
                and attempt_id == current_id
                and attempts[attempt_id].result.state == "succeeded"
                and (
                    gate_path is None
                    or not presets.exists(StoragePath(gate_path) / gate.GATE_REPORT_FILENAME)
                    or not presets.exists(StoragePath(gate_path) / RECORD_FILENAME)
                )
            ):
                raise ValueError("gate report missing; repeat the original command with --wait")
            record = (
                json.loads(presets.read_text(destination))
                if presets.exists(destination)
                else retain_record(path, preset, attempt_id=attempt_id)
            )
            step = publication_step(
                run_output,
                name,
                version,
                record,
                gate_path=gate_path if attempt_id == current_id else None,
                issue=issue if attempt_id == current_id else None,
                service_url=service_url,
            )
            run(step, max_concurrent=1)
            result["served_records"].append(str(destination))
        except Exception as error:
            causes = []
            cause = error
            while cause is not None:
                causes.append(str(cause))
                if isinstance(cause, (IapCredentialsUnavailable, IapLoginRequired)):
                    result["login"] = "iris login"
                cause = cause.__cause__
            result["unserved_records"].append(str(destination))
            result["errors"].append(": ".join(dict.fromkeys(causes)))
    if result["unserved_records"]:
        result["retry"] = "Repeat the original command with --wait."
    return result


class RunOutput(NamedTuple):
    path: str
    name: str
    version: str


def publish_runs(runs: Sequence[RunOutput], preset: presets.Preset, **kwargs) -> dict:
    """Publish stored runs and combine each attempt's publication outcome."""
    result = {"served_records": [], "unserved_records": [], "errors": []}
    for output in runs:
        published = publish_run(output.path, output.name, output.version, preset, **kwargs)
        for key, value in published.items():
            if isinstance(value, list):
                result[key].extend(value)
            else:
                result[key] = value
    return result


def publish_controls(preset_name: str, *, service_url: str | None = None) -> dict:
    """Publish the exact pinned control paths using the same frozen payloads as the launcher."""
    preset = presets.PRESETS[preset_name]
    if not publishes(preset):
        raise click.UsageError("Progress publication requires a trained-panel preset")
    outputs = [
        RunOutput(
            gate.control_path(preset, pin, index),
            presets.control_name(pin.preset, pin.main_commit, index),
            pin.revision,
        )
        for pin in gate.control_pins(preset_name)
        for index in range(pin.runs)
    ]
    result = publish_runs(outputs, preset, service_url=service_url)
    if result["unserved_records"]:
        result["retry"] = f"Repeat python -m experiments.post_training.snowball_rl_publication --preset {preset_name}."
    return result


@click.command()
@click.option(
    "--preset",
    "preset_name",
    type=click.Choice(sorted(name for name, preset in presets.PRESETS.items() if publishes(preset))),
    required=True,
)
@click.option("--url", help="Marina origin (defaults to the Marina service).")
def main(preset_name: str, url: str | None) -> None:
    """Publish a preset's pinned controls from their stored native outputs."""
    from rigging.filesystem.s3_compat import configure_coreweave_s3  # noqa: PLC0415

    configure_coreweave_s3()
    result = publish_controls(preset_name, service_url=url)
    click.echo(json.dumps(result))
    if result["unserved_records"]:
        raise click.exceptions.Exit(1)


if __name__ == "__main__":
    main()
