# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain native Snowball attempts and publish them from the launcher."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import pairwise

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from marin.execution.artifact import RECORD_FILENAME, Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.rl import skyrl
from marin.rl.rl_progress import RunSummary
from marin.rl.skyrl import SkyRLRun, SkyRLRunConfig, read_skyrl_launch_config, run_skyrl
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import publish_rl_progress
from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_scoring as scoring

RESULT_FILENAME = "loop-result.json"
PUBLICATION_FILENAME = "publication.json"


@dataclass(frozen=True)
class RecordedRun:
    skyrl: SkyRLRunConfig
    preset: presets.Preset
    label: str


def publishes(preset: presets.Preset) -> bool:
    return not preset.recipe.layout.role_plan.generate_only and preset.focus in (
        presets.Focus.LEARNING,
        presets.Focus.DATA,
        presets.Focus.TRAIN,
    )


def _json_observation(path: StoragePath, evidence: list[str]) -> dict | None:
    try:
        with path.open("rb") as source:
            value = source.read(2 * 1024 * 1024 + 1)
        if len(value) > 2 * 1024 * 1024:
            raise ValueError("Attempt observation exceeds its read limit")
        row = json.loads(value)
        if not isinstance(row, dict):
            raise ValueError("Attempt observation must be an object")
        return row
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        evidence.append(f"Could not read {path}: {type(error).__name__}.")
        return None


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
        evidence.append(f"Evaluation step {step} lacks matching current-attempt counters.")
        return None
    values = scoring.evaluation_statistics(table)
    reward = values.get(evaluation.name + "/family_mean")
    if reward is None:
        evidence.append(f"Evaluation step {step} has no scored reward.")
        return None
    scored_tasks = {row["item"] for row in rows if row["scored"]}
    return {"reward": reward, "attempted": len(counts), "scored": len(scored_tasks)}


def _stored_receipt(run_output: str, attempt_id: str | None) -> dict:
    root = StoragePath(run_output)
    terminal = root / "terminal.json"
    receipt = json.loads(presets.read_text(terminal)) if presets.exists(terminal) else None
    if receipt is None or (attempt_id is not None and receipt["config"]["run"]["attempt_id"] != attempt_id):
        launch = read_skyrl_launch_config(run_output)
        path = StoragePath(launch["artifacts"]["attempts_root"]) / f"{attempt_id or launch['run']['attempt_id']}.json"
        receipt = json.loads(presets.read_text(path))
    if attempt_id is not None and receipt["config"]["run"]["attempt_id"] != attempt_id:
        raise ValueError("Stored receipt belongs to another native attempt")
    if receipt["result"]["state"] not in {"succeeded", "failed", "cancelled", "killed", "error", "terminated"}:
        raise ValueError("A progress record requires a terminal native attempt")
    return receipt


def build_record(
    run_output: str,
    preset: presets.Preset,
    *,
    attempt_id: str | None = None,
    label: str | None = None,
) -> dict:
    """Build a terminal progress record from stored native outputs without launching or live lookups."""
    if not publishes(preset):
        raise ValueError("Progress publication requires a trained-panel preset")
    receipt = _stored_receipt(run_output, attempt_id)
    launch, result = receipt["config"], receipt["result"]
    native = launch["run"]
    attempt_id = native["attempt_id"]
    preset = replace(preset, updates=launch["skyrl"]["trainer"]["max_steps"])
    root = StoragePath(run_output)
    export = root / "exports"
    destination = export / "attempts" / attempt_id / RESULT_FILENAME
    evidence = []
    tracking = skyrl.read_skyrl_attempt_tracking(run_output, attempt_id)
    metrics = {}
    for path in scoring.paths(export / "metrics/step-*.json"):
        row = _json_observation(path, evidence)
        if row is not None and row.get("attempt_id") == attempt_id:
            metrics[row["step"]] = row
    invariants = gate.training_invariants(preset, metrics)
    failed = [fact.name for fact in invariants if not fact.holds]
    if result["state"] != "succeeded":
        failed.insert(0, "native_attempt")
    optimizer_steps = sum(
        int(row.get("policy/policy_update_steps", 0)) for step, row in metrics.items() if 0 < step <= preset.updates
    )
    started, finished = result.get("started_at"), result.get("finished_at")
    elapsed = None
    if started is not None and finished is not None:
        elapsed = (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()
        if elapsed < 0:
            raise ValueError("Native completion precedes its recorded start")
    else:
        evidence.append("Native start or finish time is unavailable; elapsed time remains unknown.")
    evaluation = preset.evaluations[0]
    sources = scoring.recorded_sources(launch, evaluation.name)
    task_ids = []
    for source in sources:
        with (StoragePath(source["uri"]) / source["relative_path"]).open("rb") as stream:
            task_ids.extend(
                json.loads(value)["id"]
                for value in pq.read_table(stream, columns=["task_spec"])["task_spec"].to_pylist()
            )
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("Recorded evaluation inputs must contain distinct task IDs")
    protocol = scoring.evaluation_protocol(launch, evaluation.name)
    evaluation = replace(evaluation, answers=protocol["n_samples_per_prompt"])
    protocol_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    phases = [None, None]
    try:
        scores = scoring.panel_scores(evaluation, run_output)
        phases = [
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
    except scoring.InvalidComparison as error:
        evidence.append(f"Evaluation measurements unavailable: {error}")
    if phases[0] is None and phases[1] is not None:
        evidence.append("Final gain lacks an attributable baseline measurement.")
        phases[1] = None
    provenance = result.get("provenance") or {}
    if not provenance:
        evidence.append("Recorded source provenance is unavailable.")
    if label is None:
        arguments = provenance.get("command_line") or ()
        overrides = [value for previous, value in pairwise(arguments) if previous == "--set"]
        overrides.extend(value.partition("=")[2] for value in arguments if value.startswith("--set="))
        label = " ".join(
            [
                preset.name,
                *(
                    value.split("=", 1)[0].rsplit(".", 1)[-1] + "=" + value.split("=", 1)[1]
                    for value in overrides
                    if "=" in value
                ),
            ]
        )
    record = {
        "run_id": native["id"],
        "attempt_id": attempt_id,
        "label": label,
        "campaign": None,
        "series": preset.name,
        "issue": None,
        "layout": preset.recipe.layout.name,
        "verdict": None,
        "gate": None,
        "min_scored_fraction": presets.MINIMUM_PANEL_COVERAGE,
        "date_basis": "launch_started",
        "run_date": started or tracking.get("started_at"),
        "started_at": started,
        "completed_at": finished,
        "elapsed_seconds": elapsed,
        "timing_scope": "native_job_start_to_finish",
        "status": "failed" if failed else "succeeded",
        "failed_facts": failed,
        "optimizer_steps": optimizer_steps,
        "iris_job_id": result.get("iris_job_id"),
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
            "task_set_sha256": scoring.task_set_sha256(launch, evaluation.name),
            "membership_sha256": (
                hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()
            ),
            "protocol": protocol,
            "protocol_sha256": protocol_hash,
        },
        "baseline": phases[0],
        "final": phases[1],
        "artifact_uri": str(destination),
        "evidence": evidence,
    }
    RunSummary.model_validate(record)
    return record


def _persist_attempt(config: RecordedRun) -> dict:
    record = build_record(
        str(StoragePath(config.skyrl.output.export_root).parent),
        config.preset,
        attempt_id=config.skyrl.attempt_id,
        label=config.label,
    )
    destination = StoragePath(record["artifact_uri"])
    destination.parent.mkdirs()
    destination.write_text(json.dumps(record, allow_nan=False, indent=2))
    return record


def run_loop(config: RecordedRun) -> SkyRLRun:
    """Run native training and retain the terminal attempt before returning or raising."""
    try:
        native = run_skyrl(config.skyrl)
    except Exception as training_error:
        try:
            _persist_attempt(config)
        except Exception as recording_error:
            raise ExceptionGroup(
                "Native execution and record persistence failed", [training_error, recording_error]
            ) from None
        raise
    _persist_attempt(config)
    return native


def retained_records(training_path: str) -> list[str]:
    """Find terminal records retained under every native attempt of one run."""
    return [str(path) for path in scoring.paths(StoragePath(training_path) / "exports/attempts/*" / RESULT_FILENAME)]


def _publish_attempt(record_path: str, gate_path: str | None, issue: int | None, output: str) -> Artifact:
    root = StoragePath(output)
    frozen = root / RESULT_FILENAME
    if not presets.exists(frozen):
        record = json.loads(presets.read_text(record_path))
        record["issue"] = issue
        gate_record = (
            presets.artifact_record(gate_path)
            if gate_path is not None and presets.exists(StoragePath(gate_path) / RECORD_FILENAME)
            else None
        )
        if gate_record is not None:
            record.update(gate=gate_record.name, verdict=gate_record.result["verdict"])
        RunSummary.model_validate(record)
        root.mkdirs()
        frozen.write_text(json.dumps(record, allow_nan=False, indent=2))
    receipt = publish_rl_progress.publish_result(json.loads(presets.read_text(frozen)))
    (root / PUBLICATION_FILENAME).write_text(json.dumps(receipt, indent=2))
    return Artifact(path=output)


def publication_step(
    training: ArtifactStep, record_path: str, *, gate_path: str | None = None, issue: int | None = None
) -> ArtifactStep[Artifact]:
    """Publish one native pair using its first retained publication payload."""
    record = RunSummary.model_validate_json(presets.read_text(record_path))
    step = apply(
        f"{training.name}/publication/{record.attempt_id}",
        _publish_attempt,
        version=training.version,
        record_path=record_path,
        gate_path=gate_path,
        issue=issue,
        output=OUT,
    )
    return replace(step, override_path=str(StoragePath(training.path()) / "publication" / record.attempt_id))
