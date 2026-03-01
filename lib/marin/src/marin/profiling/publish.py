# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish profile summaries and reports as W&B artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wandb

from marin.profiling.schema import ProfileSummary, profile_summary_from_dict
from marin.utilities.wandb_utils import WANDB_ENTITY, WANDB_PROJECT

PROFILE_SUMMARY_ARTIFACT_TYPE = "profile_summary"


def load_profile_summary(path: Path) -> ProfileSummary:
    """Load a profile summary JSON file into a typed object."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in summary file '{path}'.")
    return profile_summary_from_dict(data)


def default_profile_summary_artifact_name(summary: ProfileSummary) -> str:
    """Construct a stable default artifact name from summary metadata."""
    run_id = summary.run_metadata.run_id
    if run_id:
        return f"profile-summary-{run_id}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"profile-summary-{timestamp}"


def build_profile_summary_artifact_metadata(summary: ProfileSummary) -> dict[str, Any]:
    """Build compact metadata for published profile-summary artifacts."""
    return {
        "schema_version": summary.schema_version,
        "source_format": summary.source_format,
        "source_path": summary.source_path,
        "run_path": summary.run_metadata.run_path,
        "run_id": summary.run_metadata.run_id,
        "source_artifact_ref": summary.run_metadata.artifact_ref,
        "hardware_type": summary.run_metadata.hardware_type,
        "mesh_or_topology": summary.run_metadata.mesh_or_topology,
        "git_sha": summary.run_metadata.git_sha,
        "config_hash": summary.run_metadata.config_hash,
        "step_median": summary.step_time.steady_state_steps.median,
        "step_p90": summary.step_time.steady_state_steps.p90,
        "compute_share": summary.time_breakdown.compute.share_of_total,
        "communication_share": summary.time_breakdown.communication.share_of_total,
        "host_share": summary.time_breakdown.host.share_of_total,
        "stall_share": summary.time_breakdown.stall.share_of_total,
        "top_pre_gap_op": summary.gap_before_ops[0].name if summary.gap_before_ops else None,
        "top_pre_gap_total": summary.gap_before_ops[0].total_gap_duration if summary.gap_before_ops else None,
        "num_hierarchical_regions": len(summary.hierarchical_regions),
        "num_gap_region_contexts": len(summary.gap_region_contexts),
        "generated_at_utc": summary.generated_at_utc,
    }


def publish_profile_summary_artifact(
    *,
    summary_path: Path,
    report_path: Path | None = None,
    entity: str = WANDB_ENTITY,
    project: str = WANDB_PROJECT,
    artifact_name: str | None = None,
    aliases: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Publish summary/report files as a W&B `profile_summary` artifact.

    Returns a dictionary with publication metadata.
    """
    summary = load_profile_summary(summary_path)
    final_artifact_name = artifact_name or default_profile_summary_artifact_name(summary)
    final_aliases = aliases or ["latest"]
    metadata = build_profile_summary_artifact_metadata(summary)

    if dry_run:
        return {
            "status": "dry_run",
            "entity": entity,
            "project": project,
            "artifact_name": final_artifact_name,
            "artifact_type": PROFILE_SUMMARY_ARTIFACT_TYPE,
            "aliases": final_aliases,
            "summary_path": str(summary_path),
            "report_path": str(report_path) if report_path is not None else None,
            "metadata": metadata,
        }

    run = wandb.init(
        entity=entity,
        project=project,
        job_type="profile-summary-publish",
        name=f"publish-{final_artifact_name}",
    )
    assert run is not None

    artifact = wandb.Artifact(
        name=final_artifact_name,
        type=PROFILE_SUMMARY_ARTIFACT_TYPE,
        metadata=metadata,
    )
    artifact.add_file(str(summary_path), name=summary_path.name)
    if report_path is not None:
        artifact.add_file(str(report_path), name=report_path.name)

    logged = run.log_artifact(artifact, aliases=final_aliases)
    run.finish()

    return {
        "status": "published",
        "entity": entity,
        "project": project,
        "artifact_name": final_artifact_name,
        "artifact_type": PROFILE_SUMMARY_ARTIFACT_TYPE,
        "aliases": final_aliases,
        "artifact_version": getattr(logged, "version", None),
        "summary_path": str(summary_path),
        "report_path": str(report_path) if report_path is not None else None,
    }
