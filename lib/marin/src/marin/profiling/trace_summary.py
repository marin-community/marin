# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trace-summary engine: turn parsed complete-trace events into a ProfileSummary.

This module holds the format-agnostic summarization core shared by the
Perfetto/Chrome trace ingester (`marin.profiling.ingest`) and the XPlane
ingester (`marin.profiling.xplane`).
"""

import gzip
import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import cast

import msgspec

from marin.profiling.schema import (
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
    TraceOverview,
    TraceProvenance,
    empty_category_totals,
    make_time_breakdown,
)
from marin.profiling.semantics import (
    canonical_op_name,
    classify_semantic_family,
    estimate_flop_proxy,
    extract_shape_signature,
)

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
    # GPU/NCCL-style (no separators)
    "nccl",
    "allgather",
    "allreduce",
    "reducescatter",
)


_DEVICE_OP_THREAD_NAMES = frozenset({"XLA Ops", "Async XLA Ops"})


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


_TRACE_COMPLETE_EVENT_TRUNCATION_THRESHOLDS = frozenset({1_000_000, 5_000_000})


_GAP_PAYLOAD_LOOKAHEAD_EVENTS = 8


_GAP_MARKER_CANONICAL_NAMES = {
    "iota",
    "constant",
    "bitcast",
    "get-tuple-element",
    "parameter",
    "tuple",
    "after-all",
}


_GAP_MARKER_PREFIXES = (
    "copy-start",
    "copy-done",
)


class TraceEventArgs(msgspec.Struct, gc=False):
    """Trace-event arguments used by profile summarization."""

    name: str | None = None
    tf_op: str | None = None
    source: str | None = None
    long_name: str | None = None
    run_id: str | int | float | None = None
    step_num: int | str | None = None


class TraceEvent(msgspec.Struct, gc=False):
    """Compact event decoded directly from Perfetto JSON or an XPlane line."""

    ph: str = ""
    name: str = ""
    pid: int = -1
    tid: int = -1
    ts: float = 0.0
    dur: float = 0.0
    args: TraceEventArgs | None = None
    process_name: str | None = None
    thread_name: str | None = None

    @property
    def tf_op(self) -> str | None:
        return self.args.tf_op if self.args is not None else None

    @property
    def source(self) -> str | None:
        return self.args.source if self.args is not None else None

    @property
    def long_name(self) -> str | None:
        return self.args.long_name if self.args is not None else None

    @property
    def run_id(self) -> str | None:
        if self.args is None or self.args.run_id is None:
            return None
        value = self.args.run_id
        return value if isinstance(value, str) else str(value)

    @property
    def step_num(self) -> int | None:
        if self.args is None:
            return None
        value = self.args.step_num
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None


class TracePayload(msgspec.Struct, gc=False):
    """Typed subset of a Perfetto trace payload."""

    traceEvents: list[TraceEvent] = msgspec.field(default_factory=list)
    displayTimeUnit: str | None = None


@dataclass(frozen=True, slots=True)
class TraceSummaryContext:
    """Input identity, metadata, and summary settings shared by trace ingesters."""

    source_format: str
    source_path: Path
    display_time_unit: str | None
    num_events_total: int
    process_names: dict[int, str]
    thread_names: dict[tuple[int, int], str]
    trace_sha256: str
    run_metadata: RunMetadata | None
    warmup_steps: int
    hot_op_limit: int
    breakdown_mode: str
    extra_quality_warnings: Sequence[str] = ()


@dataclass
class _PreOpGapStats:
    count: int = 0
    total_gap_duration: float = 0.0
    max_gap_duration: float = 0.0
    marker_counts: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True, slots=True)
class TraceEventTrack:
    """One process/thread timeline in exclusive-time sort order."""

    pid: int
    tid: int
    process_name: str | None
    thread_name: str | None
    events: Iterable[TraceEvent]


@dataclass(frozen=True, slots=True)
class TraceTrackAggregate:
    """Pre-aggregated contribution from a track that needs no device-op detail."""

    num_complete_events: int
    profile_start: float | None
    profile_end: float | None
    run_ids: Counter[str]
    source_files: Counter[str]
    step_events: list[TraceEvent]
    breakdown_totals: dict[str, float]


@dataclass(slots=True)
class _OpenTraceEvent:
    event: TraceEvent
    child_duration: float
    device_event_index: int | None


class _TraceSummaryBuilder:
    def __init__(self, *, breakdown_mode: str) -> None:
        self.breakdown_mode = breakdown_mode
        self.num_complete_events = 0
        self.profile_start: float | None = None
        self.profile_end: float | None = None
        self.run_ids: Counter[str] = Counter()
        self.source_files: Counter[str] = Counter()
        self.step_events: list[TraceEvent] = []
        self.breakdown_totals = empty_category_totals()
        self.global_events: list[TraceEvent] = []
        self.device_events: list[TraceEvent] = []
        self.device_exclusive_durations: list[float] = []
        self.device_track_indices: list[list[int]] = []

    def add_track(self, track: TraceEventTrack) -> None:
        stack: list[_OpenTraceEvent] = []
        device_track = bool(
            track.process_name and track.process_name.startswith("/device:") and is_device_op_thread(track.thread_name)
        )
        device_indices: list[int] = []
        previous_key: tuple[float, float] | None = None

        for event in track.events:
            event.process_name = track.process_name
            event.thread_name = track.thread_name
            key = (event.ts, -(event.ts + event.dur))
            if previous_key is not None and key < previous_key:
                raise ValueError(f"Events on trace track ({track.pid}, {track.tid}) are not in timeline order.")
            previous_key = key

            self._record_event(event)
            device_event_index: int | None = None
            if device_track:
                device_event_index = len(self.device_events)
                self.device_events.append(event)
                self.device_exclusive_durations.append(event.dur)
                device_indices.append(device_event_index)

            start = event.ts
            end = start + event.dur
            while stack and start >= stack[-1].event.ts + stack[-1].event.dur:
                self._finalize_event(stack)
            while stack and end > stack[-1].event.ts + stack[-1].event.dur:
                self._finalize_event(stack)
            stack.append(_OpenTraceEvent(event=event, child_duration=0.0, device_event_index=device_event_index))

        while stack:
            self._finalize_event(stack)
        if device_indices:
            self.device_track_indices.append(device_indices)

    def add_track_aggregate(self, aggregate: TraceTrackAggregate) -> None:
        self.num_complete_events += aggregate.num_complete_events
        if aggregate.profile_start is not None:
            self.profile_start = (
                aggregate.profile_start
                if self.profile_start is None
                else min(self.profile_start, aggregate.profile_start)
            )
        if aggregate.profile_end is not None:
            self.profile_end = (
                aggregate.profile_end if self.profile_end is None else max(self.profile_end, aggregate.profile_end)
            )
        self.run_ids.update(aggregate.run_ids)
        self.source_files.update(aggregate.source_files)
        self.step_events.extend(aggregate.step_events)
        for category, duration in aggregate.breakdown_totals.items():
            self.breakdown_totals[category] += duration

    def _record_event(self, event: TraceEvent) -> None:
        self.num_complete_events += 1
        self.profile_start = event.ts if self.profile_start is None else min(self.profile_start, event.ts)
        event_end = event.ts + event.dur
        self.profile_end = event_end if self.profile_end is None else max(self.profile_end, event_end)

        if event.run_id:
            self.run_ids[event.run_id] += 1
        if event.source:
            self.source_files[event.source] += 1

        if _is_device_event(event) and event.thread_name == "Steps":
            self.step_events.append(event)
        elif (
            event.process_name
            and event.process_name.startswith("/host:")
            and event.name == "train"
            and event.step_num is not None
        ):
            self.step_events.append(event)

        if self.breakdown_mode == "exclusive_global" and event.thread_name != "Steps" and _is_device_event(event):
            category = _event_category(event)
            if category in {"compute", "communication"}:
                self.global_events.append(event)

    def _finalize_event(self, stack: list[_OpenTraceEvent]) -> None:
        entry = stack.pop()
        exclusive_duration = max(0.0, entry.event.dur - entry.child_duration)
        if stack:
            stack[-1].child_duration += entry.event.dur
        if entry.event.thread_name != "Steps":
            self.breakdown_totals[_event_category(entry.event)] += exclusive_duration
        if entry.device_event_index is not None:
            self.device_exclusive_durations[entry.device_event_index] = exclusive_duration


def summarize_event_tracks(
    tracks: Iterable[TraceEventTrack | TraceTrackAggregate],
    *,
    context: TraceSummaryContext,
) -> ProfileSummary:
    builder = _TraceSummaryBuilder(breakdown_mode=context.breakdown_mode)
    for track in tracks:
        if isinstance(track, TraceTrackAggregate):
            builder.add_track_aggregate(track)
        else:
            builder.add_track(track)

    suspected_truncation, quality_warnings = trace_quality_warnings(num_complete_events=builder.num_complete_events)
    quality_warnings.extend(context.extra_quality_warnings)
    if context.source_format == "xplane_pb" and builder.num_complete_events == 0:
        quality_warnings.append("XPlane protobuf contained no direct timeline events with offset/duration data.")
    trace_overview = TraceOverview(
        display_time_unit=context.display_time_unit,
        num_events_total=context.num_events_total,
        num_complete_events=builder.num_complete_events,
        num_processes=len(context.process_names),
        num_threads=len(context.thread_names),
        profile_start_ts=builder.profile_start,
        profile_end_ts=builder.profile_end,
        duration_basis="exclusive_duration_per_track",
        suspected_truncation=suspected_truncation,
        quality_warnings=quality_warnings,
    )
    trace_provenance = TraceProvenance(
        trace_sha256=context.trace_sha256,
        run_ids=[name for name, _ in builder.run_ids.most_common(20)],
        source_file_hints=[name for name, _ in builder.source_files.most_common(20)],
    )
    step_time = _summarize_step_times(builder.step_events, warmup_steps=context.warmup_steps)
    if context.breakdown_mode == "exclusive_per_track":
        time_breakdown = make_time_breakdown(
            "exclusive_duration_per_track",
            builder.breakdown_totals,
            sum(builder.breakdown_totals.values()),
        )
    elif context.breakdown_mode == "exclusive_global":
        time_breakdown = _summarize_breakdown_global(builder.global_events)
    else:
        raise ValueError(f"Unsupported breakdown mode: {context.breakdown_mode}")

    hot_ops = _summarize_hot_ops(
        builder.device_events,
        builder.device_exclusive_durations,
        limit=context.hot_op_limit,
    )
    semantic_families = summarize_semantic_families(
        hot_ops,
        total_duration=time_breakdown.total_duration,
        limit=max(context.hot_op_limit, 50),
    )
    communication_ops = _summarize_communication(builder.device_events, builder.device_exclusive_durations)
    device_op_gaps = list(_iter_device_op_gaps(builder.device_events, builder.device_track_indices))
    gap_before_ops = _summarize_pre_op_gaps(device_op_gaps, limit=max(context.hot_op_limit, 500))
    hierarchical_regions = _summarize_hierarchical_regions(
        builder.device_events,
        builder.device_exclusive_durations,
        limit=max(context.hot_op_limit, 500),
    )
    gap_region_contexts = _summarize_gap_region_contexts(
        builder.device_events,
        device_op_gaps,
        limit=max(context.hot_op_limit, 500),
    )

    summary = ProfileSummary(
        source_format=context.source_format,
        source_path=str(context.source_path),
        run_metadata=context.run_metadata or RunMetadata(),
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
    return replace(summary, optimization_candidates=derive_optimization_candidates(summary))


def summarize_complete_events(
    parsed_events: list[TraceEvent],
    *,
    context: TraceSummaryContext,
) -> ProfileSummary:
    by_track: dict[tuple[int, int], list[TraceEvent]] = defaultdict(list)
    for event in parsed_events:
        by_track[(event.pid, event.tid)].append(event)
    tracks = (
        TraceEventTrack(
            pid=pid,
            tid=tid,
            process_name=context.process_names.get(pid),
            thread_name=context.thread_names.get((pid, tid)),
            events=sorted(events, key=lambda event: (event.ts, -(event.ts + event.dur))),
        )
        for (pid, tid), events in by_track.items()
    )
    return summarize_event_tracks(
        tracks,
        context=context,
    )


def load_trace_payload(trace_path: Path) -> TracePayload:
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file does not exist: {trace_path}")

    if trace_path.suffix == ".gz":
        with gzip.open(trace_path, "rb") as handle:
            encoded = handle.read()
    else:
        encoded = trace_path.read_bytes()

    try:
        return msgspec.json.decode(encoded, type=TracePayload)
    except msgspec.DecodeError as error:
        raise ValueError(f"Invalid Perfetto JSON in trace file '{trace_path}': {error}") from error


def parse_complete_events(
    events: list[TraceEvent],
) -> tuple[list[TraceEvent], dict[int, str], dict[tuple[int, int], str]]:
    process_names: dict[int, str] = {}
    thread_names: dict[tuple[int, int], str] = {}

    for event in events:
        if event.ph != "M":
            continue

        args = event.args

        if args is None or event.pid < 0:
            continue

        if event.name == "process_name" and args.name is not None:
            process_names[event.pid] = args.name
        elif event.name == "thread_name" and event.tid >= 0 and args.name is not None:
            thread_names[(event.pid, event.tid)] = args.name

    complete_events: list[TraceEvent] = []
    for event in events:
        if event.ph != "X":
            continue
        if event.pid < 0 or event.tid < 0:
            continue
        if not event.name or event.dur <= 0:
            continue

        event.process_name = process_names.get(event.pid)
        event.thread_name = thread_names.get((event.pid, event.tid))
        complete_events.append(event)

    return complete_events, process_names, thread_names


def sha256_for_path(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def trace_quality_warnings(*, num_complete_events: int) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    suspected_truncation = num_complete_events in _TRACE_COMPLETE_EVENT_TRUNCATION_THRESHOLDS
    if suspected_truncation:
        warnings.append(
            f"Trace contains exactly {num_complete_events:,} complete events; "
            "this often indicates export truncation at a collector cap."
        )
    return suspected_truncation, warnings


def _summarize_step_times(events: list[TraceEvent], *, warmup_steps: int) -> StepTimeSummary:
    per_step: dict[int, list[float]] = defaultdict(list)

    # TPU path: device "Steps" thread with numeric event names.
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

    # GPU fallback: host-side StepTraceAnnotation events (step_num in args).
    # Filter to name="train" on /host:CPU to avoid averaging unrelated spans
    # (e.g. device-side events that also carry step_num).
    if not per_step:
        for event in events:
            if event.step_num is None:
                continue
            if event.name != "train":
                continue
            if not event.process_name or not event.process_name.startswith("/host:"):
                continue
            per_step[event.step_num].append(event.dur)

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


def _summarize_breakdown_global(events: list[TraceEvent]) -> TimeBreakdown:
    totals = empty_category_totals()

    window = _global_stall_window(events)
    if window is None:
        return make_time_breakdown("exclusive_duration_global_timeline", totals, 0.0)
    window_start, window_end = window
    window_duration = max(0.0, window_end - window_start)

    points: list[tuple[float, int, str]] = []
    for event in events:
        if event.thread_name == "Steps":
            continue
        if not _is_device_event(event):
            continue
        category = _event_category(event)
        if category not in {"compute", "communication"}:
            continue
        start = max(event.ts, window_start)
        end = min(event.ts + event.dur, window_end)
        if end <= start:
            continue
        points.append((start, 1, category))
        points.append((end, -1, category))

    active = {"compute": 0, "communication": 0}
    uncovered_duration = 0.0
    points.sort(key=lambda item: (item[0], item[1]))

    previous_ts: float = window_start
    index = 0
    while index < len(points):
        timestamp = points[index][0]
        if timestamp > previous_ts:
            category = _active_device_category(active)
            if category is not None:
                totals[category] += timestamp - previous_ts
            else:
                uncovered_duration += timestamp - previous_ts

        while index < len(points) and points[index][0] == timestamp:
            _, delta, category = points[index]
            active[category] = max(0, active[category] + delta)
            index += 1
        previous_ts = timestamp

    if previous_ts < window_end:
        category = _active_device_category(active)
        if category is not None:
            totals[category] += window_end - previous_ts
        else:
            uncovered_duration += window_end - previous_ts

    totals["stall"] = max(0.0, uncovered_duration)
    return make_time_breakdown("exclusive_duration_global_timeline", totals, window_duration)


def _global_stall_window(events: list[TraceEvent]) -> tuple[float, float] | None:
    compute_events = [event for event in events if event.thread_name != "Steps" and _event_category(event) == "compute"]
    if not compute_events:
        return None
    start = min(event.ts for event in compute_events)
    end = max(event.ts + event.dur for event in compute_events)
    if end <= start:
        return None
    return start, end


def _summarize_hot_ops(
    events: list[TraceEvent],
    exclusive: list[float],
    *,
    limit: int,
) -> list[HotOp]:
    aggregate: dict[str, dict[str, float | int | str | Counter[str]]] = {}

    for event, exclusive_duration in zip(events, exclusive, strict=True):
        if not _is_device_op_event(event):
            continue

        bucket = aggregate.get(event.name)
        if bucket is None:
            bucket = {
                "name": event.name,
                "canonical_name": canonical_op_name(event.name),
                "category": op_category(event.name),
                "count": 0,
                "total_duration": 0.0,
                "exclusive_duration": 0.0,
                "shape_counts": Counter(),
                "source_counts": Counter(),
                "tf_op_counts": Counter(),
                "flop_proxy_total": 0.0,
                "flop_proxy_count": 0,
            }
            aggregate[event.name] = bucket
        bucket["count"] = int(bucket["count"]) + 1
        bucket["total_duration"] = float(bucket["total_duration"]) + event.dur
        bucket["exclusive_duration"] = float(bucket["exclusive_duration"]) + exclusive_duration
        shape_signature = extract_shape_signature(event.long_name)
        if shape_signature:
            cast(Counter[str], bucket["shape_counts"])[shape_signature] += 1
            flop_proxy = estimate_flop_proxy(classify_semantic_family(event.name), shape_signature)
            if flop_proxy is not None:
                bucket["flop_proxy_total"] = float(bucket["flop_proxy_total"]) + flop_proxy
                bucket["flop_proxy_count"] = int(bucket["flop_proxy_count"]) + 1
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
        flop_proxy_total = float(item["flop_proxy_total"])
        flop_proxy_count = int(item["flop_proxy_count"])
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
                flop_proxy_per_invocation=(flop_proxy_total / flop_proxy_count) if flop_proxy_count else None,
            )
        )

    return result


def summarize_semantic_families(
    hot_ops: list[HotOp],
    *,
    total_duration: float,
    limit: int,
) -> list[SemanticFamilyAggregate]:
    # Semantic-family aggregates are computed from per-op exclusive durations.
    # When the overall breakdown uses a global non-overlap basis, that total can
    # be smaller than summed per-op exclusive durations, which would otherwise
    # yield >100% shares. Use a denominator consistent with the aggregated basis.
    op_exclusive_total = sum(op.exclusive_duration for op in hot_ops)
    if total_duration > 0 and op_exclusive_total > 0:
        semantic_total_duration = max(total_duration, op_exclusive_total)
    elif total_duration > 0:
        semantic_total_duration = total_duration
    else:
        semantic_total_duration = op_exclusive_total

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
                share_of_total=(exclusive / semantic_total_duration) if semantic_total_duration > 0 else 0.0,
                avg_duration=(total / count) if count else 0.0,
                avg_exclusive_duration=(exclusive / count) if count else 0.0,
                example_op=cast(str, stats["example_op"]),
                dominant_shape_signature=dominant_shape,
                flop_proxy_total=flop_proxy_total if flop_proxy_count > 0 else None,
                time_per_flop_proxy=time_per_flop_proxy,
            )
        )
    return result


def _summarize_communication(events: list[TraceEvent], exclusive: list[float]) -> list[CommunicationOp]:
    aggregate: dict[str, tuple[int, float]] = {}

    for event, duration in zip(events, exclusive, strict=True):
        if not _is_device_op_event(event):
            continue
        if not _is_communication_name(event.name):
            continue

        collective = collective_kind(event.name)
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


def _iter_device_op_gaps(
    events: list[TraceEvent],
    sorted_track_indices: list[list[int]],
) -> Iterator[tuple[TraceEvent, TraceEvent, float]]:
    """Yield ``(marker_event, payload_event, gap)`` for every idle window on a device-op track.

    Device ops are grouped per ``(pid, tid)`` track and walked in start order. A gap is the
    idle time between the running maximum end of the preceding ops and the next op's start;
    the op that follows the gap is the marker, and ``_resolve_gap_payload_event`` maps it to
    the op that actually carries the payload.
    """
    for sorted_indices in sorted_track_indices:
        if not sorted_indices or not _is_device_op_event(events[sorted_indices[0]]):
            continue
        gap_indices = _gap_order_indices(events, sorted_indices)
        previous_end: float | None = None
        for position, event_index in enumerate(gap_indices):
            event = events[event_index]
            if previous_end is not None and event.ts > previous_end:
                marker_event, payload_event = _resolve_gap_payload_event(
                    events,
                    gap_indices,
                    marker_position=position,
                )
                yield marker_event, payload_event, event.ts - previous_end
            end = event.ts + event.dur
            previous_end = end if previous_end is None else max(previous_end, end)


def _gap_order_indices(events: list[TraceEvent], exclusive_order: list[int]) -> list[int]:
    """Restore the stable ``(start, end)`` order used by gap payload lookahead."""
    gap_order: list[int] = []
    group_start = 0
    while group_start < len(exclusive_order):
        timestamp = events[exclusive_order[group_start]].ts
        group_end = group_start + 1
        while group_end < len(exclusive_order) and events[exclusive_order[group_end]].ts == timestamp:
            group_end += 1
        if group_end == group_start + 1:
            gap_order.append(exclusive_order[group_start])
        else:
            gap_order.extend(
                sorted(exclusive_order[group_start:group_end], key=lambda index: events[index].ts + events[index].dur)
            )
        group_start = group_end
    return gap_order


def _summarize_pre_op_gaps(gaps: list[tuple[TraceEvent, TraceEvent, float]], *, limit: int) -> list[GapBeforeOp]:
    aggregate: dict[str, _PreOpGapStats] = {}

    for marker_event, payload_event, gap in gaps:
        bucket = aggregate.get(payload_event.name)
        if bucket is None:
            bucket = _PreOpGapStats()
            aggregate[payload_event.name] = bucket
        bucket.count += 1
        bucket.total_gap_duration += gap
        bucket.max_gap_duration = max(bucket.max_gap_duration, gap)
        bucket.marker_counts[marker_event.name] += 1

    ranked = sorted(
        aggregate.items(),
        key=lambda item: (
            -item[1].total_gap_duration,
            -item[1].max_gap_duration,
            item[0],
        ),
    )

    result: list[GapBeforeOp] = []
    for name, stats in ranked[:limit]:
        count = stats.count
        total_gap_duration = stats.total_gap_duration
        max_gap_duration = stats.max_gap_duration
        marker_counts = stats.marker_counts
        marker_op = sorted(marker_counts.items(), key=lambda item: (-item[1], item[0]))[0][0] if marker_counts else name
        result.append(
            GapBeforeOp(
                name=name,
                count=count,
                total_gap_duration=total_gap_duration,
                max_gap_duration=max_gap_duration,
                avg_gap_duration=(total_gap_duration / count) if count else 0.0,
                payload_op=name,
                marker_op=marker_op,
            )
        )
    return result


def _summarize_hierarchical_regions(
    events: list[TraceEvent],
    exclusive: list[float],
    *,
    limit: int,
) -> list[RegionAggregate]:
    aggregate: dict[str, dict[str, float | int]] = {}

    for event, exclusive_duration in zip(events, exclusive, strict=True):
        if not _is_device_op_event(event):
            continue

        path_parts = _hierarchical_parts(event)
        if not path_parts:
            continue
        leaf_path = "=>".join(path_parts)
        for depth in range(1, len(path_parts) + 1):
            path = "=>".join(path_parts[:depth])
            bucket = aggregate.get(path)
            if bucket is None:
                bucket = {"depth": depth, "count": 0, "inclusive_duration": 0.0, "exclusive_duration": 0.0}
                aggregate[path] = bucket
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


def _summarize_gap_region_contexts(
    events: list[TraceEvent],
    gaps: list[tuple[TraceEvent, TraceEvent, float]],
    *,
    limit: int,
) -> list[GapRegionContext]:
    aggregate: dict[tuple[str, str], dict[str, float | int]] = {}
    preferred_paths = _preferred_region_path_by_op(events)

    for _, payload_event, gap in gaps:
        region_path = _event_gap_region_path(payload_event, preferred_paths=preferred_paths)
        region_path = _format_gap_region_context_label(payload_event.name, region_path)
        key = (payload_event.name, region_path)
        bucket = aggregate.get(key)
        if bucket is None:
            bucket = {
                "count": 0,
                "total_gap_duration": 0.0,
            }
            aggregate[key] = bucket
        bucket["count"] = int(bucket["count"]) + 1
        bucket["total_gap_duration"] = float(bucket["total_gap_duration"]) + gap

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


def _resolve_gap_payload_event(
    events: list[TraceEvent],
    sorted_indices: list[int],
    *,
    marker_position: int,
) -> tuple[TraceEvent, TraceEvent]:
    marker_event = events[sorted_indices[marker_position]]
    if not _is_likely_gap_marker_op(marker_event):
        return marker_event, marker_event

    marker_chain_end = marker_event.ts + marker_event.dur
    upper = min(len(sorted_indices), marker_position + 1 + _GAP_PAYLOAD_LOOKAHEAD_EVENTS)
    for position in range(marker_position + 1, upper):
        candidate = events[sorted_indices[position]]
        if candidate.ts > marker_chain_end:
            # A second idle gap starts before we found payload work; do not bridge over it.
            break
        marker_chain_end = max(marker_chain_end, candidate.ts + candidate.dur)
        if _is_likely_gap_marker_op(candidate):
            continue
        return marker_event, candidate
    return marker_event, marker_event


def _is_likely_gap_marker_op(event: TraceEvent) -> bool:
    return _is_likely_gap_marker_name(event.name)


@cache
def _is_likely_gap_marker_name(name: str) -> bool:
    canonical = canonical_op_name(name).lower()
    if canonical in _GAP_MARKER_CANONICAL_NAMES:
        return True
    return any(canonical.startswith(prefix) for prefix in _GAP_MARKER_PREFIXES)


def derive_optimization_candidates(summary: ProfileSummary) -> list[OptimizationCandidate]:
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
                payload_name = top_gap.payload_op or top_gap.name
                marker_name = top_gap.marker_op or payload_name
                candidates.append(
                    OptimizationCandidate(
                        candidate_id="pre-op-gap",
                        title="Large idle gaps appear before specific ops",
                        rationale=(
                            f"Op '{payload_name}' accumulates significant pre-op idle gap "
                            f"({gap_share:.1%} of total profiled exclusive duration)."
                        ),
                        evidence=[
                            f"Payload op with largest pre-gap: {payload_name}",
                            f"Observed first op after gap (marker): {marker_name}",
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


def _single_step_class(steady: list[tuple[int, float]]) -> list[StepClassSummary]:
    """Classify every steady-state step as one ``typical`` class."""
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


def _classify_step_patterns(averaged_steps: list[tuple[int, float]], *, warmup_steps: int) -> list[StepClassSummary]:
    steady = [(step, duration) for step, duration in averaged_steps if step >= warmup_steps]
    if not steady:
        return []

    if len(steady) < 6:
        return _single_step_class(steady)

    clusters = _kmeans_two_clusters(steady)
    if clusters is None:
        return _single_step_class(steady)

    low_cluster, high_cluster = clusters
    low_stats = DurationStats.from_values([duration for _, duration in low_cluster])
    high_stats = DurationStats.from_values([duration for _, duration in high_cluster])
    if (
        low_stats.median is None
        or high_stats.median is None
        or low_stats.median <= 0
        or (high_stats.median / low_stats.median) < 1.5
    ):
        # The two clusters are too close together to be meaningfully different step shapes.
        return _single_step_class(steady)

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


def _active_device_category(active: dict[str, int]) -> str | None:
    for category in ("communication", "compute"):
        if active[category] > 0:
            return category
    return None


def _event_category(event: TraceEvent) -> str:
    return trace_event_category(event.name, event.process_name)


@cache
def trace_event_category(name: str, process_name: str | None) -> str:
    if _STALL_PATTERN.search(name):
        return "stall"
    if process_name and process_name.startswith("/host:"):
        return "host"
    if _is_communication_name(name):
        return "communication"
    if process_name and process_name.startswith("/device:"):
        return "compute"
    return "other"


@cache
def op_category(name: str) -> str:
    if _STALL_PATTERN.search(name):
        return "stall"
    if _is_communication_name(name):
        return "communication"
    return "compute"


@cache
def _is_communication_name(name: str) -> bool:
    lowered = name.lower()
    return any(pattern in lowered for pattern in _COMM_PATTERNS)


def _is_device_event(event: TraceEvent) -> bool:
    return bool(event.process_name and event.process_name.startswith("/device:"))


def is_device_op_thread(thread_name: str | None) -> bool:
    if thread_name is None:
        return False
    if thread_name in _DEVICE_OP_THREAD_NAMES:
        return True
    if thread_name.startswith("Stream #"):
        return True
    return False


def _is_device_op_event(event: TraceEvent) -> bool:
    return _is_device_event(event) and is_device_op_thread(event.thread_name)


@cache
def collective_kind(name: str) -> str:
    lowered = name.lower()
    if "all-reduce" in lowered or "allreduce" in lowered or "psum" in lowered:
        return "all-reduce"
    if "all-gather" in lowered or "all_gather" in lowered or "allgather" in lowered:
        return "all-gather"
    if "reduce-scatter" in lowered or "reducescatter" in lowered:
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


def _hierarchical_parts(event: TraceEvent) -> tuple[str, ...]:
    return _hierarchical_parts_for_event(event.name, event.tf_op)


@cache
def _hierarchical_parts_for_event(name: str, tf_op: str | None) -> tuple[str, ...]:
    if tf_op:
        parts = _filter_hierarchy_parts([_canonical_tf_op_part(part) for part in tf_op.split("/") if part.strip()])
        if parts:
            return tuple(parts)

    delimiter_used: str | None = None
    for delimiter in _HIERARCHY_DELIMITERS:
        if delimiter in name:
            delimiter_used = delimiter
            break
    if delimiter_used is not None:
        parts = _filter_hierarchy_parts([part.strip() for part in name.split(delimiter_used) if part.strip()])
        if parts:
            return tuple(parts)

    return (_canonical_name_part(name),)


def _event_gap_region_path(
    event: TraceEvent,
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


@cache
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


def _preferred_region_path_by_op(events: list[TraceEvent], *, max_depth: int = 4) -> dict[str, str]:
    counters: dict[str, dict[str, int]] = defaultdict(dict)

    for event in events:
        if not _is_device_op_event(event):
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


def _is_fallback_parts_for_event(parts: Sequence[str], event: TraceEvent) -> bool:
    return len(parts) == 1 and parts[0] == _canonical_name_part(event.name)


def _format_gap_region_context_label(op_name: str, region_path: str) -> str:
    canonical_op = _canonical_name_part(op_name).lower()
    if canonical_op.startswith("copy"):
        normalized = region_path.strip()
        if not normalized:
            return "copy"
        if normalized.startswith("copy("):
            return normalized
        if normalized.lower() == "copy":
            return "copy"
        return f"copy({normalized})"
    return region_path
