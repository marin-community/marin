# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ingest JAX profile artifacts into a normalized profile summary."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import wandb
from google.protobuf.message import DecodeError
from levanter.utils.fsspec_utils import join_path, mirror_directory

from marin.profiling.schema import ProfileSummary, RunMetadata
from marin.profiling.trace_summary import (
    load_trace_payload,
    parse_complete_events,
    sha256_for_path,
    summarize_complete_events,
)
from marin.profiling.xplane import MultipleXPlaneFilesError, find_xplane_file, summarize_xplane

logger = logging.getLogger(__name__)

PROFILE_ARTIFACT_TYPE = "jax_profile"
PROFILE_DIR_NAME = "profiler"


class WandbRunLike(Protocol):
    config: Mapping[str, Any]
    path: Sequence[str]
    id: str
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class DownloadedProfileArtifact:
    """Downloaded W&B profile artifact and associated metadata."""

    artifact_ref: str
    artifact_name: str
    artifact_dir: Path
    run_metadata: RunMetadata


@dataclass(frozen=True)
class DownloadedProfileDir:
    """Downloaded profiler directory and associated metadata."""

    profile_dir: Path
    run_metadata: RunMetadata


def download_wandb_profile_artifact(
    artifact_ref: str,
    *,
    download_root: Path | None = None,
    artifact_type: str = PROFILE_ARTIFACT_TYPE,
) -> DownloadedProfileArtifact:
    """
    Download a W&B profile artifact and attach run metadata when available.

    Args:
        artifact_ref: Fully qualified artifact reference, for example
            `entity/project/name:v0`.
        download_root: Optional output directory for downloads.
        artifact_type: Artifact type, defaults to `jax_profile`.

    Returns:
        DownloadedProfileArtifact with download path and extracted run metadata.
    """
    api = wandb.Api()
    artifact = api.artifact(artifact_ref, type=artifact_type)
    return _download_artifact_with_metadata(
        artifact=artifact,
        artifact_ref=artifact_ref,
        run=None,
        download_root=download_root,
    )


def download_profile_dir_for_run(
    run_target: str,
    *,
    entity: str | None = None,
    project: str | None = None,
    download_root: Path | None = None,
) -> DownloadedProfileDir:
    """
    Download the profiler directory for a W&B run.

    Args:
        run_target: Bare run id, `entity/project/run_id`, or W&B run URL.
        entity: W&B entity when `run_target` is a bare run id.
        project: W&B project when `run_target` is a bare run id.
        download_root: Optional output directory where the profiler tree will be mirrored.
    """
    run_entity, run_project, run_id = normalize_run_target(run_target, entity=entity, project=project)
    run_path = f"{run_entity}/{run_project}/{run_id}"

    api = wandb.Api()
    run = api.run(run_path)
    profile_dir = resolve_profile_dir_from_run(run)

    return _download_profile_dir_with_metadata(
        profile_dir=profile_dir,
        run=run,
        download_root=download_root,
    )


def find_profile_trace(profile_dir: Path) -> Path:
    """
    Locate the preferred trace file inside a downloaded JAX profile artifact.

    Preference order:
    1) `perfetto_trace.json.gz`
    2) `*.trace.json.gz`
    3) `*.trace.json`
    """
    if not profile_dir.exists():
        raise FileNotFoundError(f"Profile directory does not exist: {profile_dir}")

    perfetto = sorted(profile_dir.rglob("perfetto_trace.json.gz"))
    if perfetto:
        return perfetto[0]

    trace_gz = sorted(profile_dir.rglob("*.trace.json.gz"))
    if trace_gz:
        return trace_gz[0]

    trace_json = sorted(profile_dir.rglob("*.trace.json"))
    if trace_json:
        return trace_json[0]

    raise FileNotFoundError(
        f"No profile trace JSON found under '{profile_dir}'. Expected perfetto_trace.json.gz or *.trace.json(.gz)."
    )


def resolve_profile_dir_from_run(run: WandbRunLike) -> str:
    """Resolve the profiler directory for a W&B run from trainer config."""
    run_id = resolve_profile_run_id_from_run(run)
    trainer_config = run.config.get("trainer")
    if not isinstance(trainer_config, dict):
        raise RuntimeError(f"Run {run.path} does not expose a trainer config.")

    log_dir = trainer_config.get("log_dir")
    if not isinstance(log_dir, str) or not log_dir:
        raise RuntimeError(f"Run {run.path} does not expose trainer.log_dir.")

    return join_path(join_path(log_dir, run_id), PROFILE_DIR_NAME)


def resolve_profile_run_id_from_run(run: WandbRunLike) -> str:
    trainer_config = run.config.get("trainer")
    if not isinstance(trainer_config, dict):
        raise RuntimeError(f"Run {run.path} does not expose a trainer config.")

    run_id = trainer_config.get("id")
    if isinstance(run_id, str) and run_id:
        return run_id

    return run.path[-1]


def summarize_profile_artifact(
    profile_dir: Path,
    *,
    run_metadata: RunMetadata | None = None,
    warmup_steps: int = 5,
    hot_op_limit: int = 25,
    breakdown_mode: str = "exclusive_per_track",
) -> ProfileSummary:
    """
    Summarize a downloaded profile artifact into the normalized schema.

    Args:
        profile_dir: Local path to a `jax_profile` artifact directory.
        run_metadata: Optional run metadata to attach.
        warmup_steps: Number of initial steps to exclude from steady-state stats.
        hot_op_limit: Maximum number of hot ops to include.
    """
    try:
        return summarize_xplane(
            find_xplane_file(profile_dir),
            run_metadata=run_metadata,
            warmup_steps=warmup_steps,
            hot_op_limit=hot_op_limit,
            breakdown_mode=breakdown_mode,
        )
    except DecodeError:
        trace_path = find_profile_trace(profile_dir)
        return summarize_trace(
            trace_path,
            run_metadata=run_metadata,
            warmup_steps=warmup_steps,
            hot_op_limit=hot_op_limit,
            breakdown_mode=breakdown_mode,
        )
    except FileNotFoundError as xplane_error:
        try:
            trace_path = find_profile_trace(profile_dir)
        except FileNotFoundError:
            raise xplane_error from None
        return summarize_trace(
            trace_path,
            run_metadata=run_metadata,
            warmup_steps=warmup_steps,
            hot_op_limit=hot_op_limit,
            breakdown_mode=breakdown_mode,
        )
    except MultipleXPlaneFilesError as xplane_error:
        try:
            trace_path = find_profile_trace(profile_dir)
        except FileNotFoundError:
            raise xplane_error from None
        return summarize_trace(
            trace_path,
            run_metadata=run_metadata,
            warmup_steps=warmup_steps,
            hot_op_limit=hot_op_limit,
            breakdown_mode=breakdown_mode,
        )


def summarize_trace(
    trace_path: Path,
    *,
    run_metadata: RunMetadata | None = None,
    warmup_steps: int = 5,
    hot_op_limit: int = 25,
    breakdown_mode: str = "exclusive_per_track",
) -> ProfileSummary:
    """
    Summarize a single trace file into the normalized profile schema.

    Args:
        trace_path: Path to `perfetto_trace.json.gz` or `*.trace.json(.gz)`.
        run_metadata: Optional run metadata to attach.
        warmup_steps: Number of initial steps to exclude from steady-state stats.
        hot_op_limit: Maximum number of hot ops to include.
    """
    payload = load_trace_payload(trace_path)
    display_time_unit = payload.get("displayTimeUnit")
    all_events = payload.get("traceEvents", [])
    if not isinstance(all_events, list):
        raise ValueError(f"Trace at '{trace_path}' does not contain a list under 'traceEvents'.")

    parsed_events, process_names, thread_names = parse_complete_events(all_events)
    return summarize_complete_events(
        parsed_events,
        source_format="perfetto_trace_json",
        source_path=trace_path,
        display_time_unit=display_time_unit if isinstance(display_time_unit, str) else None,
        num_events_total=len(all_events),
        process_names=process_names,
        thread_names=thread_names,
        trace_sha256=sha256_for_path(trace_path),
        run_metadata=run_metadata,
        warmup_steps=warmup_steps,
        hot_op_limit=hot_op_limit,
        breakdown_mode=breakdown_mode,
    )


def normalize_run_target(target: str, *, entity: str | None, project: str | None) -> tuple[str, str, str]:
    """
    Normalize run target into `(entity, project, run_id)`.

    Accepted target forms:
    - bare run id (`abc123`) with explicit `entity` and `project`
    - `entity/project/run_id`
    - W&B run URL (`https://wandb.ai/entity/project/runs/run_id`)
    """
    if target.startswith(("http://", "https://")):
        parts = [part for part in urlparse(target).path.split("/") if part]
        if len(parts) < 3:
            raise ValueError(f"Could not parse run information from URL: {target}")
    else:
        parts = [part for part in target.split("/") if part]
        if len(parts) == 1:
            if entity is None or project is None:
                raise ValueError("Bare run ids require --entity and --project.")
            return entity, project, parts[0]
        if len(parts) < 3:
            raise ValueError(f"Unrecognized run target: {target}")

    run_id = parts[3] if parts[2] == "runs" and len(parts) >= 4 else parts[2]
    return parts[0], parts[1], run_id


def _download_artifact_with_metadata(
    *,
    artifact: Any,
    artifact_ref: str,
    run: Any | None,
    download_root: Path | None,
) -> DownloadedProfileArtifact:
    artifact_dir = Path(artifact.download(root=str(download_root) if download_root is not None else None))
    metadata = RunMetadata(
        artifact_ref=artifact_ref,
        artifact_name=artifact.name,
    )

    linked_run = run
    if linked_run is None:
        try:
            linked_run = artifact.logged_by()
        except Exception:  # pragma: no cover - network/API-specific fallback
            logger.warning("Failed to load run metadata for artifact '%s'.", artifact_ref, exc_info=True)
            linked_run = None

    if linked_run is not None:
        metadata = _run_metadata_from_run(linked_run, artifact_ref=artifact_ref, artifact_name=artifact.name)

    return DownloadedProfileArtifact(
        artifact_ref=artifact_ref,
        artifact_name=artifact.name,
        artifact_dir=artifact_dir,
        run_metadata=metadata,
    )


def _download_profile_dir_with_metadata(
    *,
    profile_dir: str,
    run: WandbRunLike,
    download_root: Path | None,
) -> DownloadedProfileDir:
    run_id = resolve_profile_run_id_from_run(run)
    local_profile_dir = mirror_directory(
        profile_dir,
        download_root,
        run_id=run_id,
        leaf_dirname=PROFILE_DIR_NAME,
        temp_dir_prefix="marin-profile-",
    )
    metadata = _run_metadata_from_run(run, artifact_ref=profile_dir, artifact_name=PROFILE_DIR_NAME)
    return DownloadedProfileDir(profile_dir=local_profile_dir, run_metadata=metadata)


def _pick_first(mapping: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _infer_topology(config: dict[str, Any]) -> str | None:
    trainer = config.get("trainer")
    if not isinstance(trainer, dict):
        return None
    resources = trainer.get("resources")
    if not isinstance(resources, dict):
        return None
    for key in ("topology", "tpu_topology", "mesh_shape"):
        value = resources.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _run_metadata_from_run(run: WandbRunLike, *, artifact_ref: str, artifact_name: str) -> RunMetadata:
    summary = dict(run.summary)
    config = dict(run.config)
    return RunMetadata(
        run_path="/".join(run.path),
        run_id=run.id,
        artifact_ref=artifact_ref,
        artifact_name=artifact_name,
        hardware_type=_pick_first(summary, "throughput/device_kind", "device_kind"),
        mesh_or_topology=_infer_topology(config),
        git_sha=_pick_first(config, "git_commit", "git_sha"),
        config_hash=_config_hash(config),
        num_devices=_int_or_none(summary.get("num_devices") or config.get("num_devices")),
        num_hosts=_int_or_none(summary.get("num_hosts") or config.get("num_hosts")),
    )
