# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.11"
# dependencies = ["wandb"]
# ///
"""Mirror OLMo-Eval file outputs to W&B training-run summaries.

This script intentionally treats OLMo-Eval's local ``metrics.json`` as the
source artifact and mirrors scalar task metrics to the original training run's
W&B summary. It also logs a W&B artifact from the writeback run containing the
full task table and provenance, so collaborators can reconstruct the same CSV
from W&B without depending on a local Fieldbook ledger.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import wandb

DEFAULT_ENTITY = "marin-community"
DEFAULT_PROJECT = "marin"
DEFAULT_KEY_PREFIX = "olmo_base_eval/easy_bpb"
ARTIFACT_TYPE = "eval-results"
WRITEBACK_JOB_TYPE = "olmo-base-eval-writeback"
METADATA_FIELDNAMES = (
    "run_name",
    "wandb_run_id",
    "checkpoint_root",
    "hf_repo",
    "fieldbook_run_id",
    "fieldbook_job_id",
    "slurm_job_id",
    "olmo_eval_git_sha",
)
TASK_FIELDNAMES = (
    "task",
    "metric_path",
    "value",
    "is_primary",
    "primary_metric",
    "num_instances",
    "wandb_summary_key",
)
SEGMENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class TargetMetadata:
    """Metadata linking an OLMo-Eval output back to a Marin training run."""

    run_name: str
    wandb_run_id: str
    checkpoint_root: str | None = None
    hf_repo: str | None = None
    fieldbook_run_id: str | None = None
    fieldbook_job_id: str | None = None
    slurm_job_id: str | None = None
    olmo_eval_git_sha: str | None = None


@dataclass(frozen=True)
class MetricRecord:
    """One scalar OLMo-Eval task metric ready for W&B mirroring."""

    task: str
    metric_path: str
    value: float
    is_primary: bool
    primary_metric: str | None
    num_instances: int | None
    wandb_summary_key: str


@dataclass(frozen=True)
class WritebackPayload:
    """Prepared W&B summary values and provenance files."""

    target: TargetMetadata
    experiment_name: str | None
    experiment_group: str | None
    source_metrics_json: str
    metric_records: tuple[MetricRecord, ...]
    summary_updates: dict[str, float | int | str]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--target-run-name")
    parser.add_argument("--target-wandb-run-id")
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--hf-repo")
    parser.add_argument("--fieldbook-run-id")
    parser.add_argument("--fieldbook-job-id")
    parser.add_argument("--slurm-job-id")
    parser.add_argument("--olmo-eval-git-sha")
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--key-prefix", default=DEFAULT_KEY_PREFIX)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--skip-artifact", action="store_true")
    return parser.parse_args()


def sanitize_segment(value: str) -> str:
    """Return a W&B-key-safe path segment while preserving readability."""
    cleaned = SEGMENT_RE.sub("_", value.strip()).strip("_.-")
    return cleaned or "unknown"


def numeric_value(value: Any) -> float | None:
    """Convert finite ints/floats to float and ignore non-scalar values."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def flatten_metrics(metrics: dict[str, Any], prefix: tuple[str, ...] = ()) -> list[tuple[str, float]]:
    """Flatten nested OLMo-Eval metrics into ``metric/path`` scalar pairs."""
    rows: list[tuple[str, float]] = []
    for key, value in sorted(metrics.items()):
        path = (*prefix, str(key))
        scalar = numeric_value(value)
        if scalar is not None:
            rows.append(("/".join(path), scalar))
        elif isinstance(value, dict):
            rows.extend(flatten_metrics(value, path))
    return rows


def task_metric_key(*, key_prefix: str, task: str, metric_path: str) -> str:
    """Build the stable W&B summary key for one task metric."""
    metric_slug = "/".join(sanitize_segment(segment) for segment in metric_path.split("/"))
    return f"{key_prefix}/{sanitize_segment(task)}/{metric_slug}"


def primary_metric_path(primary_metric: str | None) -> str | None:
    """Normalize OLMo-Eval's colon-separated primary metric path."""
    if not primary_metric:
        return None
    return "/".join(part for part in primary_metric.split(":") if part)


def load_target_metadata(
    *,
    metadata_csv: Path | None,
    target_run_name: str | None,
    target_wandb_run_id: str | None,
    checkpoint_root: str | None,
    hf_repo: str | None,
    fieldbook_run_id: str | None,
    fieldbook_job_id: str | None,
    slurm_job_id: str | None,
    olmo_eval_git_sha: str | None,
) -> TargetMetadata:
    """Resolve target training-run metadata from explicit args or a matrix CSV."""
    csv_row: dict[str, str] = {}
    if metadata_csv is not None:
        if not target_run_name:
            raise ValueError("--target-run-name is required when --metadata-csv is used")
        with metadata_csv.open(newline="") as f:
            matches = [row for row in csv.DictReader(f) if row.get("run_name") == target_run_name]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one metadata row for {target_run_name!r}, found {len(matches)}")
        csv_row = matches[0]

    resolved_run_name = target_run_name or csv_row.get("run_name")
    resolved_wandb_run_id = target_wandb_run_id or csv_row.get("wandb_run_id")
    if not resolved_run_name:
        raise ValueError("A target run name is required")
    if not resolved_wandb_run_id:
        raise ValueError("A target W&B run id is required")

    return TargetMetadata(
        run_name=resolved_run_name,
        wandb_run_id=resolved_wandb_run_id,
        checkpoint_root=checkpoint_root or csv_row.get("checkpoint_root") or None,
        hf_repo=hf_repo or csv_row.get("hf_repo") or None,
        fieldbook_run_id=fieldbook_run_id,
        fieldbook_job_id=fieldbook_job_id,
        slurm_job_id=slurm_job_id,
        olmo_eval_git_sha=olmo_eval_git_sha,
    )


def build_payload(metrics_json: Path, target: TargetMetadata, key_prefix: str) -> WritebackPayload:
    """Prepare flattened metrics and W&B summary values from OLMo-Eval output."""
    metrics_data = json.loads(metrics_json.read_text())
    records: list[MetricRecord] = []
    primary_values: list[float] = []

    for task_row in metrics_data.get("tasks", []):
        task = str(task_row["task"])
        primary_path = primary_metric_path(task_row.get("primary_metric"))
        num_instances = task_row.get("num_instances")
        if not isinstance(num_instances, int):
            num_instances = None
        for metric_path, value in flatten_metrics(task_row.get("metrics", {})):
            is_primary = metric_path == primary_path
            if is_primary:
                primary_values.append(value)
            records.append(
                MetricRecord(
                    task=task,
                    metric_path=metric_path,
                    value=value,
                    is_primary=is_primary,
                    primary_metric=task_row.get("primary_metric"),
                    num_instances=num_instances,
                    wandb_summary_key=task_metric_key(key_prefix=key_prefix, task=task, metric_path=metric_path),
                )
            )

    if not records:
        raise ValueError(f"No scalar task metrics found in {metrics_json}")

    summary_updates: dict[str, float | int | str] = {record.wandb_summary_key: record.value for record in records}
    summary_updates[f"{key_prefix}/_summary/task_count"] = len(metrics_data.get("tasks", []))
    summary_updates[f"{key_prefix}/_summary/scalar_metric_count"] = len(records)
    if primary_values:
        summary_updates[f"{key_prefix}/_summary/primary_metric_count"] = len(primary_values)
        summary_updates[f"{key_prefix}/_summary/primary_metric_mean"] = float(sum(primary_values) / len(primary_values))
    duration = numeric_value(metrics_data.get("experiment_duration_seconds"))
    if duration is not None:
        summary_updates[f"{key_prefix}/_summary/experiment_duration_seconds"] = duration
    if metrics_data.get("experiment_id"):
        summary_updates[f"{key_prefix}/_provenance/olmo_eval_experiment_id"] = str(metrics_data["experiment_id"])
    if target.fieldbook_run_id:
        summary_updates[f"{key_prefix}/_provenance/fieldbook_run_id"] = target.fieldbook_run_id
    if target.slurm_job_id:
        summary_updates[f"{key_prefix}/_provenance/slurm_job_id"] = target.slurm_job_id

    return WritebackPayload(
        target=target,
        experiment_name=metrics_data.get("experiment_name"),
        experiment_group=metrics_data.get("experiment_group"),
        source_metrics_json=str(metrics_json),
        metric_records=tuple(records),
        summary_updates=summary_updates,
    )


def write_payload_files(payload: WritebackPayload, output_dir: Path) -> tuple[Path, Path]:
    """Write machine-readable payload files used for W&B artifact provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "olmo_eval_metrics_long.csv"
    manifest_path = output_dir / "wandb_writeback_manifest.json"
    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TASK_FIELDNAMES)
        writer.writeheader()
        for record in payload.metric_records:
            writer.writerow(asdict(record))
    manifest = {
        "target": asdict(payload.target),
        "experiment_name": payload.experiment_name,
        "experiment_group": payload.experiment_group,
        "source_metrics_json": payload.source_metrics_json,
        "summary_update_count": len(payload.summary_updates),
        "metric_record_count": len(payload.metric_records),
        "metadata_fieldnames": METADATA_FIELDNAMES,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return table_path, manifest_path


def logged_artifact_uri(logged_artifact: Any) -> str:
    """Return a stable W&B artifact URI after logging."""
    name = str(logged_artifact.name)
    versioned_name = name if ":" in name else f"{name}:{logged_artifact.version}"
    return f"{logged_artifact.entity}/{logged_artifact.project}/{versioned_name}"


def apply_wandb_writeback(
    *,
    payload: WritebackPayload,
    entity: str,
    project: str,
    artifact_files: tuple[Path, Path] | None,
    key_prefix: str,
) -> str | None:
    """Update target run summary and optionally log a full provenance artifact."""
    api = wandb.Api(timeout=60)
    target_run = api.run(f"{entity}/{project}/{payload.target.wandb_run_id}")
    target_run.summary.update(payload.summary_updates, overwrite=True)

    if artifact_files is None:
        return None

    artifact_name = "-".join(
        sanitize_segment(part)
        for part in (
            "olmo-base-eval",
            payload.target.run_name,
            payload.experiment_name or "unknown-experiment",
        )
    )
    run = wandb.init(
        entity=entity,
        project=project,
        job_type=WRITEBACK_JOB_TYPE,
        name=f"{WRITEBACK_JOB_TYPE}-{payload.target.run_name}",
        config={
            "target_wandb_run_id": payload.target.wandb_run_id,
            "target_run_name": payload.target.run_name,
            "key_prefix": key_prefix,
            "source_metrics_json": payload.source_metrics_json,
        },
    )
    try:
        artifact = wandb.Artifact(
            name=artifact_name[:128],
            type=ARTIFACT_TYPE,
            metadata={
                "target": asdict(payload.target),
                "experiment_name": payload.experiment_name,
                "experiment_group": payload.experiment_group,
                "key_prefix": key_prefix,
            },
        )
        for path in artifact_files:
            artifact.add_file(str(path))
        logged = run.log_artifact(
            artifact,
            aliases=[
                "latest",
                sanitize_segment(payload.target.run_name),
                sanitize_segment(payload.target.wandb_run_id),
            ],
        )
        logged.wait()
        artifact_uri = logged_artifact_uri(logged)
    finally:
        run.finish()
    return artifact_uri


def main() -> None:
    """Run the CLI."""
    args = parse_args()
    target = load_target_metadata(
        metadata_csv=args.metadata_csv,
        target_run_name=args.target_run_name,
        target_wandb_run_id=args.target_wandb_run_id,
        checkpoint_root=args.checkpoint_root,
        hf_repo=args.hf_repo,
        fieldbook_run_id=args.fieldbook_run_id,
        fieldbook_job_id=args.fieldbook_job_id,
        slurm_job_id=args.slurm_job_id,
        olmo_eval_git_sha=args.olmo_eval_git_sha,
    )
    payload = build_payload(args.metrics_json, target, args.key_prefix)
    output_dir = args.output_dir
    temporary_dir: tempfile.TemporaryDirectory[str] | None = None
    if output_dir is None:
        temporary_dir = tempfile.TemporaryDirectory()
        output_dir = Path(temporary_dir.name)
    try:
        artifact_files = write_payload_files(payload, output_dir)
        artifact_uri = None
        if args.apply:
            artifact_uri = apply_wandb_writeback(
                payload=payload,
                entity=args.entity,
                project=args.project,
                artifact_files=None if args.skip_artifact else artifact_files,
                key_prefix=args.key_prefix,
            )
        print(
            json.dumps(
                {
                    "apply": args.apply,
                    "target_wandb_run_id": payload.target.wandb_run_id,
                    "summary_update_count": len(payload.summary_updates),
                    "metric_record_count": len(payload.metric_records),
                    "primary_metric_mean": payload.summary_updates.get(
                        f"{args.key_prefix}/_summary/primary_metric_mean"
                    ),
                    "table_path": str(artifact_files[0]),
                    "manifest_path": str(artifact_files[1]),
                    "artifact_uri": artifact_uri,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        if temporary_dir is not None:
            temporary_dir.cleanup()


if __name__ == "__main__":
    main()
