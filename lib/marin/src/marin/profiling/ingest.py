# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ingest JAX profile artifacts into a normalized profile summary."""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import wandb

from marin.profiling.semantics import (
    canonical_op_name,
    classify_semantic_family,
    estimate_flop_proxy,
    extract_shape_signature,
)
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
    SemanticFamilyAggregate,
    StepClassSummary,
    StepTimeSummary,
    TimeBreakdown,
    TraceProvenance,
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
    canonical_name: str
    deduplicated_name: str | None
    pid: int
    tid: int
    ts: float
    dur: float
    tf_op: str | None
    source: str | None
    source_stack: str | None
    hlo_category: str | None
    long_name: str | None
    run_id: str | None
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
    trace_path = find_profile_trace(profile_dir)
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
    payload = _load_trace_payload(trace_path)
    display_time_unit = payload.get("displayTimeUnit")
    all_events = payload.get("traceEvents", [])
    if not isinstance(all_events, list):
        raise ValueError(f"Trace at '{trace_path}' does not contain a list under 'traceEvents'.")

    parsed_events, process_names, thread_names = _parse_complete_events(all_events)
    exclusive_durations = _compute_exclusive_durations(parsed_events)
    trace_sha256 = _sha256_for_path(trace_path)

    trace_overview = _make_trace_overview(
        display_time_unit=display_time_unit if isinstance(display_time_unit, str) else None,
        all_events=all_events,
        complete_events=parsed_events,
        process_names=process_names,
        thread_names=thread_names,
    )
    trace_provenance = _make_trace_provenance(parsed_events, trace_sha256=trace_sha256)
    step_time = _summarize_step_times(parsed_events, warmup_steps=warmup_steps)
    time_breakdown = _summarize_breakdown(parsed_events, exclusive_durations, mode=breakdown_mode)
    hot_ops = _summarize_hot_ops(parsed_events, exclusive_durations, limit=hot_op_limit)
    semantic_families = _summarize_semantic_families(
        hot_ops,
        total_duration=time_breakdown.total_duration,
        limit=max(hot_op_limit, 50),
    )
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
        trace_provenance=trace_provenance,
        step_time=step_time,
        time_breakdown=time_breakdown,
        hot_ops=hot_ops,
        semantic_families=semantic_families,
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
        trace_provenance=summary.trace_provenance,
        step_time=summary.step_time,
        time_breakdown=summary.time_breakdown,
        hot_ops=summary.hot_ops,
        semantic_families=summary.semantic_families,
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
                canonical_name=canonical_op_name(name),
                deduplicated_name=_string_arg(event.get("args"), "deduplicated_name"),
                pid=pid,
                tid=tid,
                ts=float(ts),
                dur=float(dur),
                tf_op=_string_arg(event.get("args"), "tf_op"),
                source=_string_arg(event.get("args"), "source"),
                source_stack=_string_arg(event.get("args"), "source_stack"),
                hlo_category=_string_arg(event.get("args"), "hlo_category"),
                long_name=_string_arg(event.get("args"), "long_name"),
                run_id=_string_like_arg(event.get("args"), "run_id"),
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


def _sha256_for_path(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def _make_trace_provenance(events: list[_CompleteTraceEvent], *, trace_sha256: str) -> TraceProvenance:
    run_ids = Counter(event.run_id for event in events if event.run_id)
    source_files = Counter(event.source for event in events if event.source)
    return TraceProvenance(
        trace_sha256=trace_sha256,
        run_ids=[name for name, _ in run_ids.most_common(20)],
        source_file_hints=[name for name, _ in source_files.most_common(20)],
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
        classes=_classify_step_patterns(averaged_steps, warmup_steps=warmup_steps),
    )


def _summarize_breakdown(
    events: list[_CompleteTraceEvent],
    exclusive: list[float],
    *,
    mode: str,
) -> TimeBreakdown:
    if mode == "exclusive_per_track":
        return _summarize_breakdown_per_track(events, exclusive)
    if mode == "exclusive_global":
        return _summarize_breakdown_global(events)
    raise ValueError(f"Unsupported breakdown mode: {mode}")


def _summarize_breakdown_per_track(events: list[_CompleteTraceEvent], exclusive: list[float]) -> TimeBreakdown:
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


def _summarize_breakdown_global(events: list[_CompleteTraceEvent]) -> TimeBreakdown:
    totals = {
        "compute": 0.0,
        "communication": 0.0,
        "host": 0.0,
        "stall": 0.0,
        "other": 0.0,
    }
    points: list[tuple[float, int, str]] = []
    for event in events:
        if event.thread_name == "Steps":
            continue
        category = _event_category(event)
        points.append((event.ts, 1, category))
        points.append((event.ts + event.dur, -1, category))

    if not points:
        return TimeBreakdown(
            duration_basis="exclusive_duration_global_timeline",
            total_duration=0.0,
            compute=_breakdown_part(0.0, 0.0),
            communication=_breakdown_part(0.0, 0.0),
            host=_breakdown_part(0.0, 0.0),
            stall=_breakdown_part(0.0, 0.0),
            other=_breakdown_part(0.0, 0.0),
        )

    points.sort(key=lambda item: (item[0], item[1]))
    active = {
        "compute": 0,
        "communication": 0,
        "host": 0,
        "stall": 0,
        "other": 0,
    }
    previous_ts: float | None = None
    index = 0
    while index < len(points):
        timestamp = points[index][0]
        if previous_ts is not None and timestamp > previous_ts:
            category = _active_category(active)
            if category is not None:
                totals[category] += timestamp - previous_ts

        while index < len(points) and points[index][0] == timestamp:
            _, delta, category = points[index]
            active[category] = max(0, active[category] + delta)
            index += 1
        previous_ts = timestamp

    total_duration = sum(totals.values())
    return TimeBreakdown(
        duration_basis="exclusive_duration_global_timeline",
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
    aggregate: dict[str, dict[str, float | int | str | Counter[str] | list[float]]] = {}

    for event, exclusive_duration in zip(events, exclusive, strict=True):
        if not _is_device_event(event):
            continue
        if event.thread_name not in {"XLA Ops", "Async XLA Ops"}:
            continue

        bucket = aggregate.setdefault(
            event.name,
            {
                "name": event.name,
                "canonical_name": event.canonical_name,
                "category": _op_category(event.name),
                "count": 0,
                "total_duration": 0.0,
                "exclusive_duration": 0.0,
                "shape_counts": Counter(),
                "source_counts": Counter(),
                "tf_op_counts": Counter(),
                "flop_samples": [],
            },
        )
        bucket["count"] = int(bucket["count"]) + 1
        bucket["total_duration"] = float(bucket["total_duration"]) + event.dur
        bucket["exclusive_duration"] = float(bucket["exclusive_duration"]) + exclusive_duration
        shape_signature = extract_shape_signature(event.long_name)
        if shape_signature:
            cast(Counter[str], bucket["shape_counts"])[shape_signature] += 1
            flop_proxy = estimate_flop_proxy(classify_semantic_family(event.name), shape_signature)
            if flop_proxy is not None:
                cast(list[float], bucket["flop_samples"]).append(flop_proxy)
        if event.source:
            cast(Counter[str], bucket["source_counts"])[event.source] += 1
        if event.tf_op:
            cast(Counter[str], bucket["tf_op_counts"])[event.tf_op] += 1

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
        shape_counts = cast(Counter[str], item["shape_counts"])
        source_counts = cast(Counter[str], item["source_counts"])
        tf_op_counts = cast(Counter[str], item["tf_op_counts"])
        flop_samples = cast(list[float], item["flop_samples"])
        result.append(
            HotOp(
                name=str(item["name"]),
                canonical_name=str(item["canonical_name"]),
                category=str(item["category"]),
                count=count,
                total_duration=total_duration,
                exclusive_duration=exclusive_duration,
                avg_duration=(total_duration / count) if count else 0.0,
                shape_signature=shape_counts.most_common(1)[0][0] if shape_counts else None,
                source_file=source_counts.most_common(1)[0][0] if source_counts else None,
                tf_op_path=tf_op_counts.most_common(1)[0][0] if tf_op_counts else None,
                flop_proxy_per_invocation=(sum(flop_samples) / len(flop_samples)) if flop_samples else None,
            )
        )

    return result


def _summarize_semantic_families(
    hot_ops: list[HotOp],
    *,
    total_duration: float,
    limit: int,
) -> list[SemanticFamilyAggregate]:
    aggregate: dict[str, dict[str, float | int | Counter[str] | str]] = {}
    for op in hot_ops:
        family = classify_semantic_family(op.name)
        bucket = aggregate.setdefault(
            family,
            {
                "count": 0,
                "total_duration": 0.0,
                "exclusive_duration": 0.0,
                "shape_counts": Counter(),
                "example_op": op.name,
                "flop_proxy_total": 0.0,
                "flop_proxy_count": 0,
            },
        )
        bucket["count"] = int(bucket["count"]) + op.count
        bucket["total_duration"] = float(bucket["total_duration"]) + op.total_duration
        bucket["exclusive_duration"] = float(bucket["exclusive_duration"]) + op.exclusive_duration
        if op.shape_signature:
            cast(Counter[str], bucket["shape_counts"])[op.shape_signature] += op.count
        if op.flop_proxy_per_invocation is not None and op.count > 0:
            bucket["flop_proxy_total"] = float(bucket["flop_proxy_total"]) + (op.flop_proxy_per_invocation * op.count)
            bucket["flop_proxy_count"] = int(bucket["flop_proxy_count"]) + op.count

    ranked = sorted(
        aggregate.items(),
        key=lambda item: (-float(item[1]["exclusive_duration"]), item[0]),
    )
    result: list[SemanticFamilyAggregate] = []
    for family, stats in ranked[:limit]:
        count = int(stats["count"])
        total = float(stats["total_duration"])
        exclusive = float(stats["exclusive_duration"])
        flop_proxy_total = float(stats["flop_proxy_total"])
        flop_proxy_count = int(stats["flop_proxy_count"])
        shape_counts = cast(Counter[str], stats["shape_counts"])
        dominant_shape = shape_counts.most_common(1)[0][0] if shape_counts else None
        time_per_flop_proxy = (exclusive / flop_proxy_total) if flop_proxy_total > 0 else None
        result.append(
            SemanticFamilyAggregate(
                family=family,
                count=count,
                total_duration=total,
                exclusive_duration=exclusive,
                share_of_total=(exclusive / total_duration) if total_duration > 0 else 0.0,
                avg_duration=(total / count) if count else 0.0,
                avg_exclusive_duration=(exclusive / count) if count else 0.0,
                example_op=cast(str, stats["example_op"]),
                dominant_shape_signature=dominant_shape,
                flop_proxy_total=flop_proxy_total if flop_proxy_count > 0 else None,
                time_per_flop_proxy=time_per_flop_proxy,
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

    _prune_redundant_unary_hierarchy_paths(aggregate)

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


def _prune_redundant_unary_hierarchy_paths(aggregate: dict[str, dict[str, float | int]]) -> None:
    children_by_parent: dict[str, set[str]] = defaultdict(set)
    for path in aggregate:
        if "=>" not in path:
            continue
        parent = path.rsplit("=>", 1)[0]
        children_by_parent[parent].add(path)

    redundant: set[str] = set()
    for path, stats in aggregate.items():
        depth = int(stats["depth"])
        if depth <= 1:
            continue
        children = children_by_parent.get(path)
        if children is None or len(children) != 1:
            continue
        child = next(iter(children))
        parent_inclusive = float(stats["inclusive_duration"])
        child_inclusive = float(aggregate[child]["inclusive_duration"])
        if math.isclose(parent_inclusive, child_inclusive, rel_tol=1e-9, abs_tol=1e-6):
            redundant.add(path)

    for path in redundant:
        aggregate.pop(path, None)


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
                    },
                )
                bucket["count"] = int(bucket["count"]) + 1
                bucket["total_gap_duration"] = float(bucket["total_gap_duration"]) + gap
            end = event.ts + event.dur
            previous_end = end if previous_end is None else max(previous_end, end)

    ranked = sorted(
        aggregate.items(),
        key=lambda item: (
            -float(item[1]["total_gap_duration"]),
            item[0][0],
            item[0][1],
        ),
    )

    result: list[GapRegionContext] = []
    for (op_name, region_path), stats in ranked[:limit]:
        count = int(stats["count"])
        total_gap_duration = float(stats["total_gap_duration"])
        result.append(
            GapRegionContext(
                op_name=op_name,
                region_path=region_path,
                count=count,
                total_gap_duration=total_gap_duration,
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


def _classify_step_patterns(averaged_steps: list[tuple[int, float]], *, warmup_steps: int) -> list[StepClassSummary]:
    steady = [(step, duration) for step, duration in averaged_steps if step >= warmup_steps]
    if not steady:
        return []

    if len(steady) < 6:
        stats = DurationStats.from_values([duration for _, duration in steady])
        representative_step, representative_duration = _representative_step(steady, stats.median)
        return [
            StepClassSummary(
                name="typical",
                count=len(steady),
                fraction_of_steady=1.0,
                duration_stats=stats,
                representative_step=representative_step,
                representative_duration=representative_duration,
                periodicity=None,
            )
        ]

    clusters = _kmeans_two_clusters(steady)
    if clusters is None:
        stats = DurationStats.from_values([duration for _, duration in steady])
        representative_step, representative_duration = _representative_step(steady, stats.median)
        return [
            StepClassSummary(
                name="typical",
                count=len(steady),
                fraction_of_steady=1.0,
                duration_stats=stats,
                representative_step=representative_step,
                representative_duration=representative_duration,
                periodicity=None,
            )
        ]

    low_cluster, high_cluster = clusters
    low_stats = DurationStats.from_values([duration for _, duration in low_cluster])
    high_stats = DurationStats.from_values([duration for _, duration in high_cluster])
    if (
        low_stats.median is None
        or high_stats.median is None
        or low_stats.median <= 0
        or (high_stats.median / low_stats.median) < 1.5
    ):
        stats = DurationStats.from_values([duration for _, duration in steady])
        representative_step, representative_duration = _representative_step(steady, stats.median)
        return [
            StepClassSummary(
                name="typical",
                count=len(steady),
                fraction_of_steady=1.0,
                duration_stats=stats,
                representative_step=representative_step,
                representative_duration=representative_duration,
                periodicity=None,
            )
        ]

    light_rep_step, light_rep_duration = _representative_step(low_cluster, low_stats.median)
    heavy_rep_step, heavy_rep_duration = _representative_step(high_cluster, high_stats.median)
    heavy_periodicity = _estimate_periodicity([step for step, _ in high_cluster])
    total = len(steady)
    return [
        StepClassSummary(
            name="light",
            count=len(low_cluster),
            fraction_of_steady=(len(low_cluster) / total),
            duration_stats=low_stats,
            representative_step=light_rep_step,
            representative_duration=light_rep_duration,
            periodicity=None,
        ),
        StepClassSummary(
            name="heavy",
            count=len(high_cluster),
            fraction_of_steady=(len(high_cluster) / total),
            duration_stats=high_stats,
            representative_step=heavy_rep_step,
            representative_duration=heavy_rep_duration,
            periodicity=heavy_periodicity,
        ),
    ]


def _kmeans_two_clusters(
    step_durations: list[tuple[int, float]],
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]] | None:
    values = [duration for _, duration in step_durations]
    if len(step_durations) < 6:
        return None
    minimum = min(values)
    maximum = max(values)
    if minimum <= 0 or maximum / minimum < 1.25:
        return None

    logs = [math.log(value) for value in values]
    center_a = min(logs)
    center_b = max(logs)
    labels = [0 for _ in step_durations]
    for _ in range(40):
        changed = False
        for index, value in enumerate(logs):
            dist_a = abs(value - center_a)
            dist_b = abs(value - center_b)
            label = 0 if dist_a <= dist_b else 1
            if label != labels[index]:
                labels[index] = label
                changed = True
        group_a = [value for value, label in zip(logs, labels, strict=True) if label == 0]
        group_b = [value for value, label in zip(logs, labels, strict=True) if label == 1]
        if not group_a or not group_b:
            return None
        next_a = sum(group_a) / len(group_a)
        next_b = sum(group_b) / len(group_b)
        if abs(next_a - center_a) < 1e-9 and abs(next_b - center_b) < 1e-9 and not changed:
            break
        center_a = next_a
        center_b = next_b

    group_a_pairs = [
        (index, value) for index, (value, label) in enumerate(zip(values, labels, strict=True)) if label == 0
    ]
    group_b_pairs = [
        (index, value) for index, (value, label) in enumerate(zip(values, labels, strict=True)) if label == 1
    ]
    if len(group_a_pairs) < 2 or len(group_b_pairs) < 2:
        return None
    if (len(group_a_pairs) / len(step_durations)) < 0.1 or (len(group_b_pairs) / len(step_durations)) < 0.1:
        return None
    cluster_a = [step_durations[index] for index, _ in group_a_pairs]
    cluster_b = [step_durations[index] for index, _ in group_b_pairs]
    if center_a <= center_b:
        return cluster_a, cluster_b
    return cluster_b, cluster_a


def _representative_step(steps: list[tuple[int, float]], target: float | None) -> tuple[int | None, float | None]:
    if not steps or target is None:
        return None, None
    step, duration = min(steps, key=lambda pair: (abs(pair[1] - target), pair[0]))
    return step, duration


def _estimate_periodicity(steps: list[int]) -> int | None:
    if len(steps) < 3:
        return None
    sorted_steps = sorted(steps)
    differences = [current - previous for previous, current in pairwise(sorted_steps)]
    positive = [difference for difference in differences if difference > 1]
    if len(positive) < 2:
        return None
    counts = Counter(positive)
    best_diff, best_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    if best_count < 2:
        return None
    return best_diff


def _active_category(active: dict[str, int]) -> str | None:
    for category in ("communication", "compute", "stall", "host", "other"):
        if active[category] > 0:
            return category
    return None


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


def _string_like_arg(args_value: Any, key: str) -> str | None:
    if not isinstance(args_value, dict):
        return None
    value = args_value.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value if value else None
    if isinstance(value, (int, float)):
        return str(value)
    return None
