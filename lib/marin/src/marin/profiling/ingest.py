# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ingest JAX profile artifacts into a normalized profile summary."""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import wandb

from marin.profiling.schema import (
    BreakdownPart,
    CommunicationOp,
    DurationStats,
    GapBeforeOp,
    GapRegionContext,
    HotOp,
    OptimizationCandidate,
    ProfileSummary,
    RegionAggregate,
    RunMetadata,
    StepTimeSummary,
    TimeBreakdown,
    TraceOverview,
)

logger = logging.getLogger(__name__)

PROFILE_ARTIFACT_TYPE = "jax_profile"
DEFAULT_ARTIFACT_ALIAS = "latest"

_COMM_PATTERNS = (
    "all-reduce",
    "all_gather",
    "all-gather",
    "reduce-scatter",
    "all-to-all",
    "alltoall",
    "collective",
    "collective-permute",
    "permute",
    "psum",
    "send",
    "recv",
)

_STALL_PATTERN = re.compile(
    r"wait|barrier|dependency-wait|donation holds|semaphore|acquire|idle|blocked|sleep", re.IGNORECASE
)
_HIERARCHY_DELIMITERS = ("=>", "::")
_TF_OP_WRAPPERS = {"jit", "jvp", "transpose", "vmap", "pjit", "named_call", "remat", "checkpoint"}
_HIERARCHY_SEGMENT_BLACKLIST_EXACT = {
    "xla",
    "xla_ops",
    "xla_modules",
    "xla_traceme",
    "xla_trace_me",
    "pallas_call",
    "shard_map",
    "call",
    "execute",
    "launch",
    "tpu_launch",
}
_HIERARCHY_SEGMENT_BLACKLIST_PREFIX = {
    "pjrt",
    "xla_",
    "tpu_",
    "stream_executor",
}
_HIERARCHY_SEGMENT_BLACKLIST_CONTAINS = {
    "launch",
    "execute",
    "thunk",
    "runtime",
}


@dataclass(frozen=True)
class DownloadedProfileArtifact:
    """Downloaded W&B profile artifact and associated metadata."""

    artifact_ref: str
    artifact_name: str
    artifact_dir: Path
    run_metadata: RunMetadata


@dataclass(frozen=True)
class _CompleteTraceEvent:
    name: str
    pid: int
    tid: int
    ts: float
    dur: float
    tf_op: str | None
    source: str | None
    hlo_category: str | None
    process_name: str | None
    thread_name: str | None


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


def download_latest_profile_artifact_for_run(
    run_target: str,
    *,
    entity: str | None = None,
    project: str | None = None,
    alias: str = DEFAULT_ARTIFACT_ALIAS,
    download_root: Path | None = None,
) -> DownloadedProfileArtifact:
    """
    Download the latest (or alias-selected) `jax_profile` artifact for a W&B run.

    Args:
        run_target: Bare run id, `entity/project/run_id`, or W&B run URL.
        entity: W&B entity when `run_target` is a bare run id.
        project: W&B project when `run_target` is a bare run id.
        alias: Artifact alias preference (defaults to `latest`).
        download_root: Optional output directory for artifact download.
    """
    run_entity, run_project, run_id = normalize_run_target(run_target, entity=entity, project=project)
    run_path = f"{run_entity}/{run_project}/{run_id}"

    api = wandb.Api()
    run = api.run(run_path)
    artifact = select_profile_artifact(run, alias=alias)
    artifact_ref = f"{run_entity}/{run_project}/{artifact.name}"

    return _download_artifact_with_metadata(
        artifact=artifact,
        artifact_ref=artifact_ref,
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


def summarize_profile_artifact(
    profile_dir: Path,
    *,
    run_metadata: RunMetadata | None = None,
    warmup_steps: int = 5,
    hot_op_limit: int = 25,
) -> ProfileSummary:
    """
    Summarize a downloaded profile artifact into the normalized schema.

    Args:
        profile_dir: Local path to a `jax_profile` artifact directory.
        run_metadata: Optional run metadata to attach.
        warmup_steps: Number of initial steps to exclude from steady-state stats.
        hot_op_limit: Maximum number of hot ops to include.
    """
    trace_path = find_profile_trace(profile_dir)
    return summarize_trace(
        trace_path,
        run_metadata=run_metadata,
        warmup_steps=warmup_steps,
        hot_op_limit=hot_op_limit,
    )


def summarize_trace(
    trace_path: Path,
    *,
    run_metadata: RunMetadata | None = None,
    warmup_steps: int = 5,
    hot_op_limit: int = 25,
) -> ProfileSummary:
    """
    Summarize a single trace file into the normalized profile schema.

    Args:
        trace_path: Path to `perfetto_trace.json.gz` or `*.trace.json(.gz)`.
        run_metadata: Optional run metadata to attach.
        warmup_steps: Number of initial steps to exclude from steady-state stats.
        hot_op_limit: Maximum number of hot ops to include.
    """
    payload = _load_trace_payload(trace_path)
    display_time_unit = payload.get("displayTimeUnit")
    all_events = payload.get("traceEvents", [])
    if not isinstance(all_events, list):
        raise ValueError(f"Trace at '{trace_path}' does not contain a list under 'traceEvents'.")

    parsed_events, process_names, thread_names = _parse_complete_events(all_events)
    exclusive_durations = _compute_exclusive_durations(parsed_events)

    trace_overview = _make_trace_overview(
        display_time_unit=display_time_unit if isinstance(display_time_unit, str) else None,
        all_events=all_events,
        complete_events=parsed_events,
        process_names=process_names,
        thread_names=thread_names,
    )
    step_time = _summarize_step_times(parsed_events, warmup_steps=warmup_steps)
    time_breakdown = _summarize_breakdown(parsed_events, exclusive_durations)
    hot_ops = _summarize_hot_ops(parsed_events, exclusive_durations, limit=hot_op_limit)
    communication_ops = _summarize_communication(parsed_events, exclusive_durations)
    gap_before_ops = _summarize_pre_op_gaps(parsed_events, limit=max(hot_op_limit, 500))
    hierarchical_regions = _summarize_hierarchical_regions(
        parsed_events,
        exclusive_durations,
        limit=max(hot_op_limit, 500),
    )
    gap_region_contexts = _summarize_gap_region_contexts(
        parsed_events,
        limit=max(hot_op_limit, 500),
    )

    summary = ProfileSummary.create(
        source_format="perfetto_trace_json",
        source_path=str(trace_path),
        run_metadata=run_metadata or RunMetadata(),
        trace_overview=trace_overview,
        step_time=step_time,
        time_breakdown=time_breakdown,
        hot_ops=hot_ops,
        communication_ops=communication_ops,
        gap_before_ops=gap_before_ops,
        hierarchical_regions=hierarchical_regions,
        gap_region_contexts=gap_region_contexts,
        optimization_candidates=[],
    )

    candidates = _derive_optimization_candidates(summary)
    return ProfileSummary(
        schema_version=summary.schema_version,
        generated_at_utc=summary.generated_at_utc,
        source_format=summary.source_format,
        source_path=summary.source_path,
        run_metadata=summary.run_metadata,
        trace_overview=summary.trace_overview,
        step_time=summary.step_time,
        time_breakdown=summary.time_breakdown,
        hot_ops=summary.hot_ops,
        communication_ops=summary.communication_ops,
        gap_before_ops=summary.gap_before_ops,
        hierarchical_regions=summary.hierarchical_regions,
        gap_region_contexts=summary.gap_region_contexts,
        optimization_candidates=candidates,
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
        parsed = urlparse(target)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 3:
            raise ValueError(f"Could not parse run information from URL: {target}")
        run_entity = parts[0]
        run_project = parts[1]
        if parts[2] == "runs" and len(parts) >= 4:
            run_id = parts[3]
        else:
            run_id = parts[2]
        return run_entity, run_project, run_id

    parts = [part for part in target.split("/") if part]
    if len(parts) == 1:
        if entity is None or project is None:
            raise ValueError("Bare run ids require --entity and --project.")
        return entity, project, parts[0]

    if len(parts) >= 3:
        run_entity = parts[0]
        run_project = parts[1]
        if parts[2] == "runs" and len(parts) >= 4:
            run_id = parts[3]
        else:
            run_id = parts[2]
        return run_entity, run_project, run_id

    raise ValueError(f"Unrecognized run target: {target}")


def _load_trace_payload(trace_path: Path) -> dict[str, Any]:
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file does not exist: {trace_path}")

    if trace_path.suffix == ".gz":
        with gzip.open(trace_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        with trace_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in trace file '{trace_path}', found {type(payload)}.")
    return payload


def select_profile_artifact(run: Any, *, alias: str | None) -> Any:
    """
    Select a `jax_profile` artifact from a W&B run.

    If `alias` matches no artifact, falls back to the most recently updated one.
    """
    candidates = [
        artifact for artifact in run.logged_artifacts() if getattr(artifact, "type", None) == PROFILE_ARTIFACT_TYPE
    ]
    if not candidates:
        raise RuntimeError(f"No artifacts of type '{PROFILE_ARTIFACT_TYPE}' were found for run {run.path}.")

    if alias:
        for artifact in candidates:
            if alias in _alias_names(artifact):
                return artifact

    candidates.sort(
        key=lambda artifact: getattr(artifact, "updated_at", None) or getattr(artifact, "created_at", None),
        reverse=True,
    )
    return candidates[0]


def _alias_names(artifact: Any) -> set[str]:
    names: set[str] = set()
    for alias in getattr(artifact, "aliases", None) or []:
        name = getattr(alias, "name", alias)
        if name is not None:
            names.add(str(name))
    return names


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


def _parse_complete_events(
    events: list[Any],
) -> tuple[list[_CompleteTraceEvent], dict[int, str], dict[tuple[int, int], str]]:
    process_names: dict[int, str] = {}
    thread_names: dict[tuple[int, int], str] = {}

    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("ph") != "M":
            continue

        name = event.get("name")
        pid = event.get("pid")
        tid = event.get("tid")
        args = event.get("args", {})

        if not isinstance(args, dict) or not isinstance(pid, int):
            continue

        if name == "process_name":
            value = args.get("name")
            if isinstance(value, str):
                process_names[pid] = value
        elif name == "thread_name" and isinstance(tid, int):
            value = args.get("name")
            if isinstance(value, str):
                thread_names[(pid, tid)] = value

    complete_events: list[_CompleteTraceEvent] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("ph") != "X":
            continue

        pid = event.get("pid")
        tid = event.get("tid")
        ts = event.get("ts")
        dur = event.get("dur")
        name = event.get("name")

        if not isinstance(pid, int) or not isinstance(tid, int):
            continue
        if not isinstance(ts, (int, float)) or not isinstance(dur, (int, float)):
            continue
        if not isinstance(name, str):
            continue
        if dur <= 0:
            continue

        complete_events.append(
            _CompleteTraceEvent(
                name=name,
                pid=pid,
                tid=tid,
                ts=float(ts),
                dur=float(dur),
                tf_op=_string_arg(event.get("args"), "tf_op"),
                source=_string_arg(event.get("args"), "source"),
                hlo_category=_string_arg(event.get("args"), "hlo_category"),
                process_name=process_names.get(pid),
                thread_name=thread_names.get((pid, tid)),
            )
        )

    return complete_events, process_names, thread_names


def _compute_exclusive_durations(events: list[_CompleteTraceEvent]) -> list[float]:
    exclusive = [event.dur for event in events]
    by_track: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        by_track[(event.pid, event.tid)].append(index)

    for indices in by_track.values():
        sorted_indices = sorted(indices, key=lambda idx: (events[idx].ts, -(events[idx].ts + events[idx].dur)))
        stack: list[int] = []
        child_durations: dict[int, float] = {}

        for idx in sorted_indices:
            start = events[idx].ts
            end = start + events[idx].dur

            while stack and start >= events[stack[-1]].ts + events[stack[-1]].dur:
                _finalize_top(stack=stack, child_durations=child_durations, exclusive=exclusive, events=events)

            while stack and end > events[stack[-1]].ts + events[stack[-1]].dur:
                _finalize_top(stack=stack, child_durations=child_durations, exclusive=exclusive, events=events)

            stack.append(idx)
            child_durations[idx] = 0.0

        while stack:
            _finalize_top(stack=stack, child_durations=child_durations, exclusive=exclusive, events=events)

    return exclusive


def _make_trace_overview(
    *,
    display_time_unit: str | None,
    all_events: list[Any],
    complete_events: list[_CompleteTraceEvent],
    process_names: dict[int, str],
    thread_names: dict[tuple[int, int], str],
) -> TraceOverview:
    if complete_events:
        start = min(event.ts for event in complete_events)
        end = max(event.ts + event.dur for event in complete_events)
    else:
        start = None
        end = None

    return TraceOverview(
        display_time_unit=display_time_unit,
        num_events_total=len(all_events),
        num_complete_events=len(complete_events),
        num_processes=len(process_names),
        num_threads=len(thread_names),
        profile_start_ts=start,
        profile_end_ts=end,
        duration_basis="exclusive_duration_per_track",
    )


def _summarize_step_times(events: list[_CompleteTraceEvent], *, warmup_steps: int) -> StepTimeSummary:
    per_step: dict[int, list[float]] = defaultdict(list)
    for event in events:
        if not _is_device_event(event):
            continue
        if event.thread_name != "Steps":
            continue
        try:
            step = int(event.name)
        except ValueError:
            continue
        per_step[step].append(event.dur)

    averaged_steps: list[tuple[int, float]] = []
    for step, durations in per_step.items():
        if not durations:
            continue
        averaged_steps.append((step, sum(durations) / len(durations)))
    averaged_steps.sort(key=lambda item: item[0])

    all_values = [duration for _, duration in averaged_steps]
    steady_values = [duration for step, duration in averaged_steps if step >= warmup_steps]

    return StepTimeSummary(
        warmup_steps_ignored=warmup_steps,
        all_steps=DurationStats.from_values(all_values),
        steady_state_steps=DurationStats.from_values(steady_values),
    )


def _summarize_breakdown(events: list[_CompleteTraceEvent], exclusive: list[float]) -> TimeBreakdown:
    totals = {
        "compute": 0.0,
        "communication": 0.0,
        "host": 0.0,
        "stall": 0.0,
        "other": 0.0,
    }

    for event, duration in zip(events, exclusive, strict=True):
        # "Steps" is a wrapper timeline and heavily overlaps with lower-level device events.
        if event.thread_name == "Steps":
            continue
        category = _event_category(event)
        totals[category] += duration

    total_duration = sum(totals.values())

    return TimeBreakdown(
        duration_basis="exclusive_duration_per_track",
        total_duration=total_duration,
        compute=_breakdown_part(totals["compute"], total_duration),
        communication=_breakdown_part(totals["communication"], total_duration),
        host=_breakdown_part(totals["host"], total_duration),
        stall=_breakdown_part(totals["stall"], total_duration),
        other=_breakdown_part(totals["other"], total_duration),
    )


def _summarize_hot_ops(
    events: list[_CompleteTraceEvent],
    exclusive: list[float],
    *,
    limit: int,
) -> list[HotOp]:
    aggregate: dict[str, dict[str, float | int | str]] = {}

    for event, exclusive_duration in zip(events, exclusive, strict=True):
        if not _is_device_event(event):
            continue
        if event.thread_name not in {"XLA Ops", "Async XLA Ops"}:
            continue

        bucket = aggregate.setdefault(
            event.name,
            {
                "name": event.name,
                "category": _op_category(event.name),
                "count": 0,
                "total_duration": 0.0,
                "exclusive_duration": 0.0,
            },
        )
        bucket["count"] = int(bucket["count"]) + 1
        bucket["total_duration"] = float(bucket["total_duration"]) + event.dur
        bucket["exclusive_duration"] = float(bucket["exclusive_duration"]) + exclusive_duration

    ranked = sorted(
        aggregate.values(),
        key=lambda item: (
            -float(item["exclusive_duration"]),
            -float(item["total_duration"]),
            str(item["name"]),
        ),
    )

    result: list[HotOp] = []
    for item in ranked[:limit]:
        count = int(item["count"])
        total_duration = float(item["total_duration"])
        exclusive_duration = float(item["exclusive_duration"])
        result.append(
            HotOp(
                name=str(item["name"]),
                category=str(item["category"]),
                count=count,
                total_duration=total_duration,
                exclusive_duration=exclusive_duration,
                avg_duration=(total_duration / count) if count else 0.0,
            )
        )

    return result


def _summarize_communication(events: list[_CompleteTraceEvent], exclusive: list[float]) -> list[CommunicationOp]:
    aggregate: dict[str, tuple[int, float]] = {}

    for event, duration in zip(events, exclusive, strict=True):
        if not _is_device_event(event):
            continue
        if not _is_communication_name(event.name):
            continue
        if event.thread_name not in {"XLA Ops", "Async XLA Ops"}:
            continue

        collective = _collective_kind(event.name)
        count, total = aggregate.get(collective, (0, 0.0))
        aggregate[collective] = (count + 1, total + duration)

    sorted_items = sorted(aggregate.items(), key=lambda item: (-item[1][1], item[0]))
    return [
        CommunicationOp(
            collective=collective,
            count=count,
            total_duration=total_duration,
            avg_duration=(total_duration / count) if count else 0.0,
        )
        for collective, (count, total_duration) in sorted_items
    ]


def _summarize_pre_op_gaps(events: list[_CompleteTraceEvent], *, limit: int) -> list[GapBeforeOp]:
    aggregate: dict[str, dict[str, float | int]] = {}

    by_track: dict[tuple[int, int], list[_CompleteTraceEvent]] = defaultdict(list)
    for event in events:
        if not _is_device_event(event):
            continue
        if event.thread_name not in {"XLA Ops", "Async XLA Ops"}:
            continue
        by_track[(event.pid, event.tid)].append(event)

    for track_events in by_track.values():
        sorted_events = sorted(track_events, key=lambda event: (event.ts, event.ts + event.dur))
        previous_end: float | None = None
        for event in sorted_events:
            if previous_end is not None and event.ts > previous_end:
                gap = event.ts - previous_end
                bucket = aggregate.setdefault(
                    event.name,
                    {"count": 0, "total_gap_duration": 0.0, "max_gap_duration": 0.0},
                )
                bucket["count"] = int(bucket["count"]) + 1
                bucket["total_gap_duration"] = float(bucket["total_gap_duration"]) + gap
                bucket["max_gap_duration"] = max(float(bucket["max_gap_duration"]), gap)
            end = event.ts + event.dur
            previous_end = end if previous_end is None else max(previous_end, end)

    ranked = sorted(
        aggregate.items(),
        key=lambda item: (
            -float(item[1]["total_gap_duration"]),
            -float(item[1]["max_gap_duration"]),
            item[0],
        ),
    )

    result: list[GapBeforeOp] = []
    for name, stats in ranked[:limit]:
        count = int(stats["count"])
        total_gap_duration = float(stats["total_gap_duration"])
        max_gap_duration = float(stats["max_gap_duration"])
        result.append(
            GapBeforeOp(
                name=name,
                count=count,
                total_gap_duration=total_gap_duration,
                max_gap_duration=max_gap_duration,
                avg_gap_duration=(total_gap_duration / count) if count else 0.0,
            )
        )
    return result


def _summarize_hierarchical_regions(
    events: list[_CompleteTraceEvent],
    exclusive: list[float],
    *,
    limit: int,
) -> list[RegionAggregate]:
    aggregate: dict[str, dict[str, float | int]] = {}

    for event, exclusive_duration in zip(events, exclusive, strict=True):
        if not _is_device_event(event):
            continue
        if event.thread_name not in {"XLA Ops", "Async XLA Ops"}:
            continue

        path_parts = _hierarchical_parts(event)
        if not path_parts:
            continue
        leaf_path = "=>".join(path_parts)
        for depth in range(1, len(path_parts) + 1):
            path = "=>".join(path_parts[:depth])
            bucket = aggregate.setdefault(
                path,
                {"depth": depth, "count": 0, "inclusive_duration": 0.0, "exclusive_duration": 0.0},
            )
            bucket["count"] = int(bucket["count"]) + 1
            bucket["inclusive_duration"] = float(bucket["inclusive_duration"]) + exclusive_duration

        # "Exclusive" for a region excludes child regions. We approximate this by
        # assigning event time only to the deepest semantic path segment.
        leaf_bucket = aggregate[leaf_path]
        leaf_bucket["exclusive_duration"] = float(leaf_bucket["exclusive_duration"]) + exclusive_duration

    ranked = sorted(
        aggregate.items(),
        key=lambda item: (
            -float(item[1]["inclusive_duration"]),
            -float(item[1]["exclusive_duration"]),
            item[0],
        ),
    )

    result: list[RegionAggregate] = []
    for path, stats in ranked[:limit]:
        result.append(
            RegionAggregate(
                path=path,
                depth=int(stats["depth"]),
                count=int(stats["count"]),
                inclusive_duration=float(stats["inclusive_duration"]),
                exclusive_duration=float(stats["exclusive_duration"]),
            )
        )
    return result


def _summarize_gap_region_contexts(events: list[_CompleteTraceEvent], *, limit: int) -> list[GapRegionContext]:
    aggregate: dict[tuple[str, str], dict[str, float | int]] = {}
    preferred_paths = _preferred_region_path_by_op(events)

    by_track: dict[tuple[int, int], list[_CompleteTraceEvent]] = defaultdict(list)
    for event in events:
        if not _is_device_event(event):
            continue
        if event.thread_name not in {"XLA Ops", "Async XLA Ops"}:
            continue
        by_track[(event.pid, event.tid)].append(event)

    for track_events in by_track.values():
        sorted_events = sorted(track_events, key=lambda event: (event.ts, event.ts + event.dur))
        previous_end: float | None = None
        for event in sorted_events:
            if previous_end is not None and event.ts > previous_end:
                gap = event.ts - previous_end
                region_path = _event_gap_region_path(event, preferred_paths=preferred_paths)
                region_path = _format_gap_region_context_label(event.name, region_path)
                key = (event.name, region_path)
                bucket = aggregate.setdefault(
                    key,
                    {
                        "count": 0,
                        "total_gap_duration": 0.0,
                        "total_overlap_duration": 0.0,
                    },
                )
                bucket["count"] = int(bucket["count"]) + 1
                bucket["total_gap_duration"] = float(bucket["total_gap_duration"]) + gap
                # We use the op's hierarchical semantic path as a deterministic context label.
                bucket["total_overlap_duration"] = float(bucket["total_overlap_duration"]) + gap
            end = event.ts + event.dur
            previous_end = end if previous_end is None else max(previous_end, end)

    ranked = sorted(
        aggregate.items(),
        key=lambda item: (
            -float(item[1]["total_gap_duration"]),
            -float(item[1]["total_overlap_duration"]),
            item[0][0],
            item[0][1],
        ),
    )

    result: list[GapRegionContext] = []
    for (op_name, region_path), stats in ranked[:limit]:
        count = int(stats["count"])
        total_gap_duration = float(stats["total_gap_duration"])
        total_overlap_duration = float(stats["total_overlap_duration"])
        result.append(
            GapRegionContext(
                op_name=op_name,
                region_path=region_path,
                count=count,
                total_gap_duration=total_gap_duration,
                total_overlap_duration=total_overlap_duration,
                avg_gap_duration=(total_gap_duration / count) if count else 0.0,
            )
        )
    return result


def _derive_optimization_candidates(summary: ProfileSummary) -> list[OptimizationCandidate]:
    candidates: list[OptimizationCandidate] = []

    breakdown = summary.time_breakdown
    total_duration = breakdown.total_duration or 1.0

    if breakdown.communication.share_of_total >= 0.15:
        top_collective = summary.communication_ops[0].collective if summary.communication_ops else "collectives"
        candidates.append(
            OptimizationCandidate(
                candidate_id="communication-heavy",
                title="Communication appears dominant",
                rationale=(
                    f"Communication accounts for {breakdown.communication.share_of_total:.1%} of "
                    "exclusive profiled duration."
                ),
                evidence=[
                    f"Communication share: {breakdown.communication.share_of_total:.1%}",
                    f"Top collective: {top_collective}",
                ],
                suggestions=[
                    "Evaluate sharding/layout choices to reduce collective volume.",
                    "Try overlapping collectives with compute where possible.",
                    "Inspect all-reduce/all-gather callsites for avoidable synchronization.",
                ],
            )
        )

    if breakdown.stall.share_of_total >= 0.20:
        candidates.append(
            OptimizationCandidate(
                candidate_id="stall-heavy",
                title="Stall/wait time is significant",
                rationale=f"Stall-like events account for {breakdown.stall.share_of_total:.1%} of profiled duration.",
                evidence=[f"Stall share: {breakdown.stall.share_of_total:.1%}"],
                suggestions=[
                    "Investigate dependency waits and synchronization barriers.",
                    "Check host input pipeline and device dispatch overlap.",
                    "Reduce unnecessary host-side blocking calls around step execution.",
                ],
            )
        )

    if summary.hot_ops:
        hottest = summary.hot_ops[0]
        hot_share = hottest.exclusive_duration / total_duration
        if hot_share >= 0.08:
            candidates.append(
                OptimizationCandidate(
                    candidate_id="single-hot-op",
                    title="Single op has outsized exclusive time",
                    rationale=f"Top op '{hottest.name}' contributes {hot_share:.1%} of profiled exclusive duration.",
                    evidence=[
                        f"Top op: {hottest.name}",
                        f"Top op exclusive duration: {hottest.exclusive_duration:.3f}",
                        f"Top op share: {hot_share:.1%}",
                    ],
                    suggestions=[
                        "Inspect kernel implementation and tiling/fusion opportunities for this op.",
                        "Try alternative algorithmic variants or precision/layout adjustments.",
                        "Run a focused microbenchmark on this op before/after tuning changes.",
                    ],
                )
            )

    if summary.gap_before_ops:
        top_gap = summary.gap_before_ops[0]
        if top_gap.total_gap_duration > 0:
            gap_share = top_gap.total_gap_duration / total_duration
            if gap_share >= 0.05 or top_gap.max_gap_duration >= 1_000.0:
                candidates.append(
                    OptimizationCandidate(
                        candidate_id="pre-op-gap",
                        title="Large idle gaps appear before specific ops",
                        rationale=(
                            f"Op '{top_gap.name}' accumulates significant pre-op idle gap "
                            f"({gap_share:.1%} of total profiled exclusive duration)."
                        ),
                        evidence=[
                            f"Op with largest pre-gap: {top_gap.name}",
                            f"Total pre-gap: {top_gap.total_gap_duration:.3f}",
                            f"Max pre-gap: {top_gap.max_gap_duration:.3f}",
                            f"Occurrences: {top_gap.count}",
                        ],
                        suggestions=[
                            "Inspect upstream dependencies immediately before this op.",
                            "Look for host dispatch or synchronization barriers causing the gap.",
                            "Use hierarchical region totals to localize where the waiting accumulates.",
                        ],
                    )
                )

    steady = summary.step_time.steady_state_steps
    if steady.count >= 4 and steady.median and steady.p90 and steady.median > 0:
        jitter = steady.p90 / steady.median
        if jitter >= 1.4:
            candidates.append(
                OptimizationCandidate(
                    candidate_id="step-jitter",
                    title="Steady-state step time has high jitter",
                    rationale=(f"Steady-state p90/median ratio is {jitter:.2f}, indicating intermittent slow steps."),
                    evidence=[
                        f"Steady median: {steady.median:.3f}",
                        f"Steady p90: {steady.p90:.3f}",
                        f"p90/median: {jitter:.2f}",
                    ],
                    suggestions=[
                        "Correlate slow steps with collective spikes and host wait events.",
                        "Check for periodic checkpoint/eval/input stalls during profiled range.",
                        "Compare traces before/after disabling optional callbacks or host work.",
                    ],
                )
            )

    if not candidates:
        candidates.append(
            OptimizationCandidate(
                candidate_id="no-dominant-bottleneck",
                title="No single dominant bottleneck found",
                rationale="Compute, communication, host, and stall shares are relatively balanced.",
                evidence=[
                    f"Compute: {breakdown.compute.share_of_total:.1%}",
                    f"Communication: {breakdown.communication.share_of_total:.1%}",
                    f"Host: {breakdown.host.share_of_total:.1%}",
                    f"Stall: {breakdown.stall.share_of_total:.1%}",
                ],
                suggestions=[
                    "Prioritize low-risk wins on top 3 hot ops and re-profile.",
                    "Use before/after summary comparison to confirm throughput impact.",
                ],
            )
        )

    return candidates


def _event_category(event: _CompleteTraceEvent) -> str:
    if _STALL_PATTERN.search(event.name):
        return "stall"
    if event.process_name and event.process_name.startswith("/host:"):
        return "host"
    if _is_communication_name(event.name):
        return "communication"
    if _is_device_event(event):
        return "compute"
    return "other"


def _op_category(name: str) -> str:
    if _STALL_PATTERN.search(name):
        return "stall"
    if _is_communication_name(name):
        return "communication"
    return "compute"


def _is_communication_name(name: str) -> bool:
    lowered = name.lower()
    return any(pattern in lowered for pattern in _COMM_PATTERNS)


def _is_device_event(event: _CompleteTraceEvent) -> bool:
    return bool(event.process_name and event.process_name.startswith("/device:"))


def _collective_kind(name: str) -> str:
    lowered = name.lower()
    if "all-reduce" in lowered or "psum" in lowered:
        return "all-reduce"
    if "all-gather" in lowered or "all_gather" in lowered:
        return "all-gather"
    if "reduce-scatter" in lowered:
        return "reduce-scatter"
    if "all-to-all" in lowered or "alltoall" in lowered:
        return "all-to-all"
    if "collective-permute" in lowered or "permute" in lowered:
        return "collective-permute"
    if "async-collective" in lowered:
        return "async-collective"
    if "send" in lowered or "recv" in lowered:
        return "send-recv"
    return "other-collective"


def _hierarchical_parts(event: _CompleteTraceEvent) -> list[str]:
    if event.tf_op:
        parts = _filter_hierarchy_parts([_canonical_tf_op_part(part) for part in event.tf_op.split("/") if part.strip()])
        if parts:
            return parts

    delimiter_used: str | None = None
    for delimiter in _HIERARCHY_DELIMITERS:
        if delimiter in event.name:
            delimiter_used = delimiter
            break
    if delimiter_used is not None:
        parts = _filter_hierarchy_parts([part.strip() for part in event.name.split(delimiter_used) if part.strip()])
        if parts:
            return parts

    return [_canonical_name_part(event.name)]


def _event_gap_region_path(
    event: _CompleteTraceEvent,
    *,
    preferred_paths: dict[str, str] | None = None,
    max_depth: int = 4,
) -> str:
    parts = _hierarchical_parts(event)
    if preferred_paths is not None and _is_fallback_parts_for_event(parts, event):
        preferred = preferred_paths.get(event.name)
        if preferred:
            return preferred
    if not parts:
        return "unknown"
    return "=>".join(parts[:max_depth])


def _canonical_tf_op_part(part: str) -> str:
    trimmed = part.strip().strip(":")
    if not trimmed:
        return ""

    # Strip stackable wrappers so semantic ops such as apply_rotary_embedding surface directly when present.
    current = trimmed
    while True:
        wrapper_match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", current)
        if wrapper_match is None:
            break
        wrapper = wrapper_match.group(1)
        inner = wrapper_match.group(2).strip()
        if wrapper not in _TF_OP_WRAPPERS:
            break
        if not inner:
            return ""
        current = inner
    if current in _TF_OP_WRAPPERS:
        return ""
    normalized = current.strip().strip(":")
    if normalized.startswith("dynamic_donated"):
        first_dot = normalized.find(".")
        if first_dot >= 0 and first_dot + 1 < len(normalized):
            normalized = normalized[first_dot + 1 :]
    return normalized


def _canonical_name_part(name: str) -> str:
    stripped = name.strip().lstrip("%")
    return re.sub(r"\.\d+$", "", stripped)


def _filter_hierarchy_parts(parts: list[str]) -> list[str]:
    filtered: list[str] = []
    for part in parts:
        if not part:
            continue
        if _is_blacklisted_hierarchy_segment(part):
            continue
        filtered.append(part)
    return filtered


def _is_blacklisted_hierarchy_segment(part: str) -> bool:
    lowered = part.lower().strip()
    if not lowered:
        return True

    normalized = re.sub(r"[^a-z0-9_]+", "_", lowered).strip("_")
    if not normalized:
        return True
    if normalized in _HIERARCHY_SEGMENT_BLACKLIST_EXACT:
        return True
    if any(normalized.startswith(prefix) for prefix in _HIERARCHY_SEGMENT_BLACKLIST_PREFIX):
        return True
    if any(token in normalized for token in _HIERARCHY_SEGMENT_BLACKLIST_CONTAINS):
        return True
    return False


def _preferred_region_path_by_op(events: list[_CompleteTraceEvent], *, max_depth: int = 4) -> dict[str, str]:
    counters: dict[str, dict[str, int]] = defaultdict(dict)

    for event in events:
        if not _is_device_event(event):
            continue
        if event.thread_name not in {"XLA Ops", "Async XLA Ops"}:
            continue
        if not event.tf_op:
            continue
        parts = _hierarchical_parts(event)
        if not parts or _is_fallback_parts_for_event(parts, event):
            continue
        path = "=>".join(parts[:max_depth])
        op_counter = counters[event.name]
        op_counter[path] = op_counter.get(path, 0) + 1

    preferred: dict[str, str] = {}
    for op_name, path_counts in counters.items():
        best_path = sorted(path_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        preferred[op_name] = best_path
    return preferred


def _is_fallback_parts_for_event(parts: list[str], event: _CompleteTraceEvent) -> bool:
    return len(parts) == 1 and parts[0] == _canonical_name_part(event.name)


def _format_gap_region_context_label(op_name: str, region_path: str) -> str:
    canonical_op = _canonical_name_part(op_name).lower()
    if canonical_op.startswith("copy") and region_path and not region_path.startswith("copy("):
        return f"copy({region_path})"
    return region_path


def _finalize_top(
    *,
    stack: list[int],
    child_durations: dict[int, float],
    exclusive: list[float],
    events: list[_CompleteTraceEvent],
) -> None:
    idx = stack.pop()
    duration = events[idx].dur
    nested = child_durations.pop(idx, 0.0)
    exclusive[idx] = max(0.0, duration - nested)
    if stack:
        parent = stack[-1]
        child_durations[parent] = child_durations.get(parent, 0.0) + duration


def _breakdown_part(value: float, total: float) -> BreakdownPart:
    share = (value / total) if total > 0 else 0.0
    return BreakdownPart(total_duration=value, share_of_total=share)


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


def _run_metadata_from_run(run: Any, *, artifact_ref: str, artifact_name: str) -> RunMetadata:
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


def _string_arg(args_value: Any, key: str) -> str | None:
    if not isinstance(args_value, dict):
        return None
    value = args_value.get(key)
    return value if isinstance(value, str) and value else None
