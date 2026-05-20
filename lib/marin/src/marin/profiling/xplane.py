# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ingest XPlane protobuf profiles directly and through xprof tables."""

from __future__ import annotations

import json
import logging
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from marin.profiling.ingest import (
    _collective_kind,
    _CompleteTraceEvent,
    _derive_optimization_candidates,
    _op_category,
    _sha256_for_path,
    _summarize_complete_events,
    _summarize_semantic_families,
    _trace_quality_warnings,
)
from marin.profiling.schema import (
    BreakdownPart,
    CommunicationOp,
    DurationStats,
    HotOp,
    ProfileSummary,
    RunMetadata,
    StepClassSummary,
    StepTimeSummary,
    TimeBreakdown,
    TraceOverview,
    TraceProvenance,
)
from marin.profiling.semantics import canonical_op_name

logger = logging.getLogger(__name__)

XPROF_TABLE_TOOLS = (
    "overview_page",
    "kernel_stats",
    "framework_op_stats",
    "op_profile",
    "hlo_stats",
    "memory_profile",
    "input_pipeline_analyzer",
)

_TRACE_COMPLETE_EVENT_TRUNCATION_THRESHOLD = 1_000_000
_PICoseconds_PER_MICROSECOND = 1_000_000.0
_XSPACE_MESSAGE_CLASS: Any | None = None


@dataclass(frozen=True)
class XPlaneTimeline:
    """Normalized timeline events parsed directly from an XPlane protobuf."""

    events: list[_CompleteTraceEvent]
    process_names: dict[int, str]
    thread_names: dict[tuple[int, int], str]
    num_events_total: int
    quality_warnings: list[str]


@dataclass(frozen=True)
class XPlaneTableExport:
    """Paths and sizes produced while converting an XPlane protobuf with xprof."""

    output_dir: Path
    table_sizes: dict[str, int]
    trace_event_count: int | None


class MultipleXPlaneFilesError(ValueError):
    """Raised when an artifact contains one XPlane protobuf per host."""


def find_xplane_file(profile_dir: Path) -> Path:
    """Locate an XPlane protobuf inside a downloaded JAX profile artifact."""
    if not profile_dir.exists():
        raise FileNotFoundError(f"Profile directory does not exist: {profile_dir}")

    candidates = sorted(profile_dir.rglob("*.xplane.pb"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        joined = ", ".join(str(path) for path in candidates)
        raise MultipleXPlaneFilesError(f"Found multiple *.xplane.pb files under '{profile_dir}': {joined}")

    raise FileNotFoundError(f"No *.xplane.pb file found under '{profile_dir}'.")


def export_xplane_tables(
    xplane_path: Path,
    output_dir: Path,
    *,
    count_trace_events: bool = False,
) -> XPlaneTableExport:
    """Convert an XPlane protobuf into compact xprof table JSON files."""
    from xprof.convert import raw_to_tool_data  # noqa: PLC0415

    if not xplane_path.exists():
        raise FileNotFoundError(f"XPlane protobuf does not exist: {xplane_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    for tool in XPROF_TABLE_TOOLS:
        data, _content_type = raw_to_tool_data.xspace_to_tool_data(
            [str(xplane_path)],
            tool,
            {"use_saved_result": False},
        )
        if data is None:
            continue
        target = output_dir / f"{tool}.json"
        encoded = _tool_data_to_bytes(data)
        target.write_bytes(encoded)
        sizes[tool] = len(encoded)

    trace_event_count = _trace_event_count_from_xprof(xplane_path) if count_trace_events else None
    return XPlaneTableExport(output_dir=output_dir, table_sizes=sizes, trace_event_count=trace_event_count)


def summarize_xplane(
    xplane_path: Path,
    *,
    output_dir: Path | None = None,
    run_metadata: RunMetadata | None = None,
    warmup_steps: int = 5,
    hot_op_limit: int = 25,
    count_trace_events: bool = False,
    breakdown_mode: str = "exclusive_per_track",
) -> ProfileSummary:
    """Summarize an XPlane protobuf into the normalized profile summary schema."""
    timeline_summary = summarize_xplane_timeline(
        xplane_path,
        run_metadata=run_metadata,
        warmup_steps=warmup_steps,
        hot_op_limit=hot_op_limit,
        breakdown_mode=breakdown_mode,
    )
    table_summary = _try_summarize_xprof_tables(
        xplane_path,
        output_dir=output_dir,
        run_metadata=run_metadata,
        warmup_steps=warmup_steps,
        hot_op_limit=hot_op_limit,
        count_trace_events=count_trace_events,
    )
    if table_summary is None:
        return timeline_summary
    return _merge_timeline_and_xprof_summaries(timeline_summary, table_summary, hot_op_limit=hot_op_limit)


def summarize_xplane_timeline(
    xplane_path: Path,
    *,
    run_metadata: RunMetadata | None = None,
    warmup_steps: int = 5,
    hot_op_limit: int = 25,
    breakdown_mode: str = "exclusive_per_track",
) -> ProfileSummary:
    """Summarize directly parsed XPlane timeline events."""
    timeline = parse_xplane_timeline(xplane_path)
    return _summarize_complete_events(
        timeline.events,
        source_format="xplane_pb",
        source_path=xplane_path,
        display_time_unit="us",
        num_events_total=timeline.num_events_total,
        process_names=timeline.process_names,
        thread_names=timeline.thread_names,
        trace_sha256=_sha256_for_path(xplane_path),
        run_metadata=run_metadata,
        warmup_steps=warmup_steps,
        hot_op_limit=hot_op_limit,
        breakdown_mode=breakdown_mode,
        extra_quality_warnings=timeline.quality_warnings,
    )


def parse_xplane_timeline(xplane_path: Path) -> XPlaneTimeline:
    """Parse XPlane protobuf timeline events into Marin's normalized event model."""
    if not xplane_path.exists():
        raise FileNotFoundError(f"XPlane protobuf does not exist: {xplane_path}")

    xspace_class = _xspace_message_class()
    xspace = xspace_class()
    xspace.ParseFromString(xplane_path.read_bytes())

    process_names: dict[int, str] = {}
    thread_names: dict[tuple[int, int], str] = {}
    events: list[_CompleteTraceEvent] = []
    num_events_total = 0
    quality_warnings = [str(warning) for warning in getattr(xspace, "warnings", [])]
    quality_warnings.extend(str(error) for error in getattr(xspace, "errors", []))

    for plane_index, plane in enumerate(xspace.planes):
        pid = plane_index + 1
        process_name = str(plane.name or f"xplane:{plane_index}")
        process_names[pid] = process_name
        stat_names = {int(key): str(value.name) for key, value in plane.stat_metadata.items()}

        for line_index, line in enumerate(plane.lines):
            tid = int(line.id or line_index + 1)
            thread_name = str(line.display_name or line.name or f"xline:{line_index}")
            thread_names[(pid, tid)] = thread_name
            num_events_total += len(line.events)
            line_start_us = float(line.timestamp_ns) / 1_000.0

            for event in line.events:
                if event.WhichOneof("data") != "offset_ps":
                    continue
                if event.duration_ps <= 0:
                    continue

                metadata = plane.event_metadata.get(event.metadata_id)
                if metadata is None:
                    continue

                event_stats = _xplane_stats_to_mapping(
                    list(metadata.stats) + list(event.stats),
                    stat_names=stat_names,
                )
                display_name = str(metadata.display_name or "")
                metadata_name = str(metadata.name or "")
                name = display_name or metadata_name
                if not name:
                    continue

                long_name = _string_stat(event_stats, "long_name") or (metadata_name if metadata_name != name else None)
                events.append(
                    _CompleteTraceEvent(
                        name=name,
                        canonical_name=canonical_op_name(name),
                        deduplicated_name=_string_stat(event_stats, "deduplicated_name"),
                        pid=pid,
                        tid=tid,
                        ts=line_start_us + (float(event.offset_ps) / _PICoseconds_PER_MICROSECOND),
                        dur=float(event.duration_ps) / _PICoseconds_PER_MICROSECOND,
                        tf_op=_string_stat(event_stats, "tf_op"),
                        source=_string_stat(event_stats, "source", "source_file", "file_name"),
                        source_stack=_string_stat(event_stats, "source_stack", "stack_frame"),
                        hlo_category=_string_stat(event_stats, "hlo_category"),
                        long_name=long_name,
                        run_id=_string_like_stat(event_stats, "run_id"),
                        process_name=process_name,
                        thread_name=thread_name,
                        step_num=_int_like_stat(event_stats, "step_num", "step"),
                    )
                )

    if not events:
        quality_warnings.append("XPlane protobuf contained no direct timeline events with offset/duration data.")

    return XPlaneTimeline(
        events=events,
        process_names=process_names,
        thread_names=thread_names,
        num_events_total=num_events_total,
        quality_warnings=quality_warnings,
    )


def summarize_xplane_tables(
    table_dir: Path,
    *,
    xplane_path: Path,
    run_metadata: RunMetadata | None = None,
    warmup_steps: int = 5,
    hot_op_limit: int = 25,
    trace_event_count: int | None = None,
) -> ProfileSummary:
    """Build a profile summary from xprof table JSON already exported from XPlane."""
    step_props, step_rows = _step_rows(table_dir)
    kernel_rows = _kernel_rows(table_dir)
    hot_ops = _hot_ops_from_kernel_rows(kernel_rows, limit=hot_op_limit)
    communication_ops = _communication_ops_from_kernel_rows(kernel_rows)
    step_time = _step_time_from_xprof_rows(step_rows, warmup_steps=warmup_steps)
    time_breakdown = _time_breakdown_from_xprof_rows(step_rows, kernel_rows)
    semantic_families = _summarize_semantic_families(
        hot_ops,
        total_duration=time_breakdown.total_duration,
        limit=max(hot_op_limit, 50),
    )
    trace_overview = _trace_overview_from_xprof_tables(
        table_dir=table_dir,
        step_props=step_props,
        step_rows=step_rows,
        kernel_rows=kernel_rows,
        trace_event_count=trace_event_count,
    )

    summary = ProfileSummary.create(
        source_format="xplane_pb_xprof_tables",
        source_path=str(xplane_path),
        run_metadata=run_metadata or RunMetadata(),
        trace_overview=trace_overview,
        trace_provenance=TraceProvenance(trace_sha256=_sha256_for_path(xplane_path)),
        step_time=step_time,
        time_breakdown=time_breakdown,
        hot_ops=hot_ops,
        semantic_families=semantic_families,
        communication_ops=communication_ops,
        gap_before_ops=[],
        hierarchical_regions=[],
        gap_region_contexts=[],
        optimization_candidates=[],
    )
    return replace(summary, optimization_candidates=_derive_optimization_candidates(summary))


def _try_summarize_xprof_tables(
    xplane_path: Path,
    *,
    output_dir: Path | None,
    run_metadata: RunMetadata | None,
    warmup_steps: int,
    hot_op_limit: int,
    count_trace_events: bool,
) -> ProfileSummary | None:
    try:
        if output_dir is not None:
            export = export_xplane_tables(xplane_path, output_dir, count_trace_events=count_trace_events)
            return summarize_xplane_tables(
                export.output_dir,
                xplane_path=xplane_path,
                run_metadata=run_metadata,
                warmup_steps=warmup_steps,
                hot_op_limit=hot_op_limit,
                trace_event_count=export.trace_event_count,
            )

        with tempfile.TemporaryDirectory(prefix="marin-xplane-tables-") as temp_dir:
            export = export_xplane_tables(xplane_path, Path(temp_dir), count_trace_events=count_trace_events)
            return summarize_xplane_tables(
                export.output_dir,
                xplane_path=xplane_path,
                run_metadata=run_metadata,
                warmup_steps=warmup_steps,
                hot_op_limit=hot_op_limit,
                trace_event_count=export.trace_event_count,
            )
    except ImportError:
        if output_dir is not None:
            raise RuntimeError("--xplane-output-dir requires the optional xprof package.") from None
        logger.info("xprof is not installed; continuing with direct XPlane timeline parsing only.")
        return None


def _merge_timeline_and_xprof_summaries(
    timeline_summary: ProfileSummary,
    table_summary: ProfileSummary,
    *,
    hot_op_limit: int,
) -> ProfileSummary:
    quality_warnings = list(timeline_summary.trace_overview.quality_warnings)
    quality_warnings.extend(_xprof_quality_warnings(table_summary))

    trace_overview = TraceOverview(
        display_time_unit=timeline_summary.trace_overview.display_time_unit,
        num_events_total=timeline_summary.trace_overview.num_events_total,
        num_complete_events=timeline_summary.trace_overview.num_complete_events,
        num_processes=timeline_summary.trace_overview.num_processes,
        num_threads=timeline_summary.trace_overview.num_threads,
        profile_start_ts=timeline_summary.trace_overview.profile_start_ts,
        profile_end_ts=timeline_summary.trace_overview.profile_end_ts,
        duration_basis=f"{timeline_summary.trace_overview.duration_basis}+xprof_aggregate_tables",
        suspected_truncation=(
            timeline_summary.trace_overview.suspected_truncation or table_summary.trace_overview.suspected_truncation
        ),
        quality_warnings=quality_warnings,
    )
    step_time = table_summary.step_time if table_summary.step_time.all_steps.count > 0 else timeline_summary.step_time
    time_breakdown = (
        table_summary.time_breakdown
        if table_summary.time_breakdown.total_duration > 0
        else timeline_summary.time_breakdown
    )
    hot_ops = _merge_hot_ops(timeline_summary.hot_ops, table_summary.hot_ops, limit=hot_op_limit)
    communication_ops = (
        table_summary.communication_ops if table_summary.communication_ops else timeline_summary.communication_ops
    )
    semantic_families = _summarize_semantic_families(
        hot_ops,
        total_duration=time_breakdown.total_duration,
        limit=max(hot_op_limit, 50),
    )

    summary = ProfileSummary.create(
        source_format="xplane_pb",
        source_path=timeline_summary.source_path,
        run_metadata=timeline_summary.run_metadata,
        trace_overview=trace_overview,
        trace_provenance=timeline_summary.trace_provenance,
        step_time=step_time,
        time_breakdown=time_breakdown,
        hot_ops=hot_ops,
        semantic_families=semantic_families,
        communication_ops=communication_ops,
        gap_before_ops=timeline_summary.gap_before_ops,
        hierarchical_regions=timeline_summary.hierarchical_regions,
        gap_region_contexts=timeline_summary.gap_region_contexts,
        optimization_candidates=[],
    )
    return replace(summary, optimization_candidates=_derive_optimization_candidates(summary))


def _xprof_quality_warnings(summary: ProfileSummary) -> list[str]:
    warnings = []
    for warning in summary.trace_overview.quality_warnings:
        if "pre-op gap and hierarchical region analysis are unavailable" in warning:
            continue
        warnings.append(warning)
    if summary.step_time.all_steps.count > 0:
        warnings.append("Step timing was augmented from xprof overview aggregate tables.")
    if summary.hot_ops:
        warnings.append("Kernel hotspot rows from xprof aggregate tables were merged into hot_ops.")
    if summary.communication_ops:
        warnings.append("Collective timing was augmented from xprof kernel aggregate tables.")
    return warnings


def _merge_hot_ops(timeline_hot_ops: list[HotOp], table_hot_ops: list[HotOp], *, limit: int) -> list[HotOp]:
    merged = list(timeline_hot_ops)
    seen = {op.name for op in merged}
    for op in table_hot_ops:
        if op.name in seen:
            continue
        merged.append(op)
        seen.add(op.name)
    return sorted(
        merged,
        key=lambda op: (-op.exclusive_duration, -op.total_duration, op.name),
    )[:limit]


def _xspace_message_class() -> Any:
    global _XSPACE_MESSAGE_CLASS
    if _XSPACE_MESSAGE_CLASS is None:
        pool = descriptor_pool.DescriptorPool()
        pool.Add(_xplane_file_descriptor())
        _XSPACE_MESSAGE_CLASS = message_factory.GetMessageClass(pool.FindMessageTypeByName("tensorflow.profiler.XSpace"))
    return _XSPACE_MESSAGE_CLASS


def _xplane_file_descriptor() -> descriptor_pb2.FileDescriptorProto:
    file_descriptor = descriptor_pb2.FileDescriptorProto(
        name="tensorflow/profiler/xplane.proto",
        package="tensorflow.profiler",
        syntax="proto3",
    )
    label = descriptor_pb2.FieldDescriptorProto.Label
    field_type = descriptor_pb2.FieldDescriptorProto.Type

    xspace = file_descriptor.message_type.add(name="XSpace")
    _add_field(xspace, "planes", 1, label.LABEL_REPEATED, field_type.TYPE_MESSAGE, ".tensorflow.profiler.XPlane")
    _add_field(xspace, "errors", 2, label.LABEL_REPEATED, field_type.TYPE_STRING)
    _add_field(xspace, "warnings", 3, label.LABEL_REPEATED, field_type.TYPE_STRING)
    _add_field(xspace, "hostnames", 4, label.LABEL_REPEATED, field_type.TYPE_STRING)

    xplane = file_descriptor.message_type.add(name="XPlane")
    _add_field(xplane, "id", 1, label.LABEL_OPTIONAL, field_type.TYPE_INT64)
    _add_field(xplane, "name", 2, label.LABEL_OPTIONAL, field_type.TYPE_STRING)
    _add_field(xplane, "lines", 3, label.LABEL_REPEATED, field_type.TYPE_MESSAGE, ".tensorflow.profiler.XLine")
    event_entry = xplane.nested_type.add(name="EventMetadataEntry")
    event_entry.options.map_entry = True
    _add_field(event_entry, "key", 1, label.LABEL_OPTIONAL, field_type.TYPE_INT64)
    _add_field(
        event_entry,
        "value",
        2,
        label.LABEL_OPTIONAL,
        field_type.TYPE_MESSAGE,
        ".tensorflow.profiler.XEventMetadata",
    )
    _add_field(
        xplane,
        "event_metadata",
        4,
        label.LABEL_REPEATED,
        field_type.TYPE_MESSAGE,
        ".tensorflow.profiler.XPlane.EventMetadataEntry",
    )
    stat_entry = xplane.nested_type.add(name="StatMetadataEntry")
    stat_entry.options.map_entry = True
    _add_field(stat_entry, "key", 1, label.LABEL_OPTIONAL, field_type.TYPE_INT64)
    _add_field(
        stat_entry,
        "value",
        2,
        label.LABEL_OPTIONAL,
        field_type.TYPE_MESSAGE,
        ".tensorflow.profiler.XStatMetadata",
    )
    _add_field(
        xplane,
        "stat_metadata",
        5,
        label.LABEL_REPEATED,
        field_type.TYPE_MESSAGE,
        ".tensorflow.profiler.XPlane.StatMetadataEntry",
    )
    _add_field(xplane, "stats", 6, label.LABEL_REPEATED, field_type.TYPE_MESSAGE, ".tensorflow.profiler.XStat")

    xline = file_descriptor.message_type.add(name="XLine")
    _add_field(xline, "id", 1, label.LABEL_OPTIONAL, field_type.TYPE_INT64)
    _add_field(xline, "name", 2, label.LABEL_OPTIONAL, field_type.TYPE_STRING)
    _add_field(xline, "timestamp_ns", 3, label.LABEL_OPTIONAL, field_type.TYPE_INT64)
    _add_field(xline, "events", 4, label.LABEL_REPEATED, field_type.TYPE_MESSAGE, ".tensorflow.profiler.XEvent")
    _add_field(xline, "duration_ps", 9, label.LABEL_OPTIONAL, field_type.TYPE_INT64)
    _add_field(xline, "display_id", 10, label.LABEL_OPTIONAL, field_type.TYPE_INT64)
    _add_field(xline, "display_name", 11, label.LABEL_OPTIONAL, field_type.TYPE_STRING)

    xevent = file_descriptor.message_type.add(name="XEvent")
    xevent.oneof_decl.add(name="data")
    _add_field(xevent, "metadata_id", 1, label.LABEL_OPTIONAL, field_type.TYPE_INT64)
    _add_field(xevent, "offset_ps", 2, label.LABEL_OPTIONAL, field_type.TYPE_INT64, oneof_index=0)
    _add_field(xevent, "duration_ps", 3, label.LABEL_OPTIONAL, field_type.TYPE_INT64)
    _add_field(xevent, "stats", 4, label.LABEL_REPEATED, field_type.TYPE_MESSAGE, ".tensorflow.profiler.XStat")
    _add_field(xevent, "num_occurrences", 5, label.LABEL_OPTIONAL, field_type.TYPE_INT64, oneof_index=0)

    xstat = file_descriptor.message_type.add(name="XStat")
    xstat.oneof_decl.add(name="value")
    _add_field(xstat, "metadata_id", 1, label.LABEL_OPTIONAL, field_type.TYPE_INT64)
    _add_field(xstat, "double_value", 2, label.LABEL_OPTIONAL, field_type.TYPE_DOUBLE, oneof_index=0)
    _add_field(xstat, "uint64_value", 3, label.LABEL_OPTIONAL, field_type.TYPE_UINT64, oneof_index=0)
    _add_field(xstat, "int64_value", 4, label.LABEL_OPTIONAL, field_type.TYPE_INT64, oneof_index=0)
    _add_field(xstat, "str_value", 5, label.LABEL_OPTIONAL, field_type.TYPE_STRING, oneof_index=0)
    _add_field(xstat, "bytes_value", 6, label.LABEL_OPTIONAL, field_type.TYPE_BYTES, oneof_index=0)
    _add_field(xstat, "ref_value", 7, label.LABEL_OPTIONAL, field_type.TYPE_UINT64, oneof_index=0)

    event_metadata = file_descriptor.message_type.add(name="XEventMetadata")
    _add_field(event_metadata, "id", 1, label.LABEL_OPTIONAL, field_type.TYPE_INT64)
    _add_field(event_metadata, "name", 2, label.LABEL_OPTIONAL, field_type.TYPE_STRING)
    _add_field(event_metadata, "metadata", 3, label.LABEL_OPTIONAL, field_type.TYPE_BYTES)
    _add_field(event_metadata, "display_name", 4, label.LABEL_OPTIONAL, field_type.TYPE_STRING)
    _add_field(
        event_metadata,
        "stats",
        5,
        label.LABEL_REPEATED,
        field_type.TYPE_MESSAGE,
        ".tensorflow.profiler.XStat",
    )
    _add_field(event_metadata, "child_id", 6, label.LABEL_REPEATED, field_type.TYPE_INT64)

    stat_metadata = file_descriptor.message_type.add(name="XStatMetadata")
    _add_field(stat_metadata, "id", 1, label.LABEL_OPTIONAL, field_type.TYPE_INT64)
    _add_field(stat_metadata, "name", 2, label.LABEL_OPTIONAL, field_type.TYPE_STRING)
    _add_field(stat_metadata, "description", 3, label.LABEL_OPTIONAL, field_type.TYPE_STRING)
    return file_descriptor


def _add_field(
    message: descriptor_pb2.DescriptorProto,
    name: str,
    number: int,
    label: descriptor_pb2.FieldDescriptorProto.Label.ValueType,
    field_type: descriptor_pb2.FieldDescriptorProto.Type.ValueType,
    type_name: str | None = None,
    *,
    oneof_index: int | None = None,
) -> None:
    field = message.field.add()
    field.name = name
    field.number = number
    field.label = label
    field.type = field_type
    if type_name is not None:
        field.type_name = type_name
    if oneof_index is not None:
        field.oneof_index = oneof_index


def _xplane_stats_to_mapping(stats: list[Any], *, stat_names: dict[int, str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for stat in stats:
        stat_name = stat_names.get(int(stat.metadata_id))
        if not stat_name:
            continue
        value = _xplane_stat_value(stat, stat_names=stat_names)
        if value is not None:
            values[stat_name] = value
    return values


def _xplane_stat_value(stat: Any, *, stat_names: dict[int, str]) -> Any:
    value_field = stat.WhichOneof("value")
    if value_field is None:
        return None
    if value_field == "ref_value":
        return stat_names.get(int(stat.ref_value), str(stat.ref_value))
    value = getattr(stat, value_field)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _string_stat(stats: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = stats.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _string_like_stat(stats: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = stats.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return value or None
        if isinstance(value, (int, float)):
            return str(int(value))
    return None


def _int_like_stat(stats: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = stats.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                continue
    return None


def _trace_event_count_from_xprof(xplane_path: Path) -> int | None:
    from xprof.convert import raw_to_tool_data  # noqa: PLC0415

    data, _content_type = raw_to_tool_data.xspace_to_tool_data(
        [str(xplane_path)],
        "trace_viewer@",
        {"use_saved_result": False},
    )
    if data is None:
        return None
    encoded = _tool_data_to_bytes(data)
    match = re.search(rb'"returnedEventsSize"\s*:\s*(\d+)', encoded[:4096])
    if match is None:
        return None
    return int(match.group(1))


def _tool_data_to_bytes(data: bytes | bytearray | str) -> bytes:
    if isinstance(data, str):
        return data.encode("utf-8")
    return bytes(data)


def _step_time_from_xprof_rows(rows: list[dict[str, Any]], *, warmup_steps: int) -> StepTimeSummary:
    step_durations: list[tuple[int, float]] = []
    for index, row in enumerate(rows):
        duration_ms = _float_value(row, "stepTimeMs")
        if duration_ms is None:
            continue
        step = _int_value(row, "stepnum")
        step_durations.append((index if step is None else step, duration_ms * 1000.0))

    all_values = [duration for _, duration in step_durations]
    steady_values = [duration for step, duration in step_durations if step >= warmup_steps]
    return StepTimeSummary(
        warmup_steps_ignored=warmup_steps,
        all_steps=DurationStats.from_values(all_values),
        steady_state_steps=DurationStats.from_values(steady_values),
        classes=_step_classes(step_durations, warmup_steps=warmup_steps),
    )


def _step_classes(step_durations: list[tuple[int, float]], *, warmup_steps: int) -> list[StepClassSummary]:
    steady = [(step, duration) for step, duration in step_durations if step >= warmup_steps]
    if len(steady) < 4:
        return []

    durations = [duration for _, duration in steady]
    median = DurationStats.from_values(durations).median
    if median is None or median <= 0:
        return []

    light = [(step, duration) for step, duration in steady if duration <= 1.25 * median]
    heavy = [(step, duration) for step, duration in steady if duration > 1.25 * median]
    if not heavy:
        return []

    return [
        _make_step_class("light", light, total_count=len(steady)),
        _make_step_class("heavy", heavy, total_count=len(steady)),
    ]


def _make_step_class(name: str, rows: list[tuple[int, float]], *, total_count: int) -> StepClassSummary:
    durations = [duration for _, duration in rows]
    return StepClassSummary(
        name=name,
        count=len(rows),
        fraction_of_steady=(len(rows) / total_count) if total_count else 0.0,
        duration_stats=DurationStats.from_values(durations),
        representative_step=rows[0][0] if rows else None,
        representative_duration=rows[0][1] if rows else None,
        periodicity=_periodicity([step for step, _ in rows]),
    )


def _periodicity(steps: list[int]) -> int | None:
    if len(steps) < 3:
        return None
    deltas = [right - left for left, right in pairwise(steps)]
    if not deltas:
        return None
    if len(set(deltas)) == 1:
        return deltas[0]
    return None


def _time_breakdown_from_xprof_rows(rows: list[dict[str, Any]], kernel_rows: list[dict[str, Any]]) -> TimeBreakdown:
    totals = {
        "compute": 0.0,
        "communication": 0.0,
        "host": 0.0,
        "stall": 0.0,
        "other": 0.0,
    }
    for row in rows:
        totals["compute"] += (_float_value(row, "deviceComputeTimeMs") or 0.0) * 1000.0
        totals["communication"] += (_float_value(row, "deviceCollectivesTimeMs") or 0.0) * 1000.0
        totals["host"] += (_float_value(row, "infeedTimeMs") or 0.0) * 1000.0
        totals["stall"] += (_float_value(row, "otherTimeMs") or 0.0) * 1000.0

    total_from_steps = sum((_float_value(row, "stepTimeMs") or 0.0) * 1000.0 for row in rows)
    if total_from_steps > 0:
        bucket_total = sum(totals.values())
        totals["other"] = max(0.0, total_from_steps - bucket_total)
        return _make_time_breakdown("xprof_overview_step_time_us", totals, total=total_from_steps)

    communication = sum(
        _float_value(row, "total_duration_us") or 0.0
        for row in kernel_rows
        if _op_category(str(row.get("kernel_name") or "")) == "communication"
    )
    total = sum(_float_value(row, "total_duration_us") or 0.0 for row in kernel_rows)
    totals["communication"] = communication
    totals["compute"] = max(0.0, total - communication)
    return _make_time_breakdown("xprof_kernel_duration_us", totals, total=total)


def _make_time_breakdown(duration_basis: str, totals: dict[str, float], *, total: float) -> TimeBreakdown:
    return TimeBreakdown(
        duration_basis=duration_basis,
        total_duration=total,
        compute=_breakdown_part(totals["compute"], total),
        communication=_breakdown_part(totals["communication"], total),
        host=_breakdown_part(totals["host"], total),
        stall=_breakdown_part(totals["stall"], total),
        other=_breakdown_part(totals["other"], total),
    )


def _breakdown_part(value: float, total: float):
    share = (value / total) if total > 0 else 0.0
    return BreakdownPart(total_duration=value, share_of_total=share)


def _hot_ops_from_kernel_rows(rows: list[dict[str, Any]], *, limit: int) -> list[HotOp]:
    hot_ops: list[HotOp] = []
    for row in _ranked_kernel_rows(rows)[:limit]:
        name = str(row.get("kernel_name") or "unknown")
        total_duration = _float_value(row, "total_duration_us") or 0.0
        count = _int_value(row, "occurrences") or 0
        hot_ops.append(
            HotOp(
                name=name,
                canonical_name=canonical_op_name(name),
                category=_op_category(name),
                count=count,
                total_duration=total_duration,
                exclusive_duration=total_duration,
                avg_duration=(total_duration / count) if count else 0.0,
            )
        )
    return hot_ops


def _communication_ops_from_kernel_rows(rows: list[dict[str, Any]]) -> list[CommunicationOp]:
    aggregate: dict[str, tuple[int, float]] = {}
    for row in rows:
        name = str(row.get("kernel_name") or "")
        if _op_category(name) != "communication":
            continue
        collective = _collective_kind(name)
        count, total = aggregate.get(collective, (0, 0.0))
        aggregate[collective] = (
            count + (_int_value(row, "occurrences") or 0),
            total + (_float_value(row, "total_duration_us") or 0.0),
        )

    return [
        CommunicationOp(
            collective=collective,
            count=count,
            total_duration=total,
            avg_duration=(total / count) if count else 0.0,
        )
        for collective, (count, total) in sorted(aggregate.items(), key=lambda item: (-item[1][1], item[0]))
    ]


def _trace_overview_from_xprof_tables(
    *,
    table_dir: Path,
    step_props: dict[str, Any],
    step_rows: list[dict[str, Any]],
    kernel_rows: list[dict[str, Any]],
    trace_event_count: int | None,
) -> TraceOverview:
    suspected_truncation = False
    warnings = [
        "Summary was built from xprof aggregate tables, not a Perfetto trace; "
        "pre-op gap and hierarchical region analysis are unavailable."
    ]
    if trace_event_count is not None:
        suspected_truncation = trace_event_count == _TRACE_COMPLETE_EVENT_TRUNCATION_THRESHOLD
        if suspected_truncation:
            _, trace_warnings = _trace_quality_warnings(num_complete_events=trace_event_count)
            warnings.extend(trace_warnings)

    statement = step_props.get("device_collectives_statement") or step_props.get("statement")
    bottleneck = step_props.get("bottleneck")
    if bottleneck:
        warnings.append(f"xprof bottleneck: {bottleneck}.")
    if statement:
        warnings.append(f"xprof statement: {statement}")

    return TraceOverview(
        display_time_unit="us",
        num_events_total=trace_event_count if trace_event_count is not None else _table_row_count(table_dir),
        num_complete_events=len(kernel_rows),
        num_processes=0,
        num_threads=0,
        profile_start_ts=None,
        profile_end_ts=None,
        duration_basis="xprof_aggregate_tables",
        suspected_truncation=suspected_truncation,
        quality_warnings=warnings,
    )


def _table_row_count(table_dir: Path) -> int:
    return sum(len(rows) for _cols, rows, _props in _iter_tables(table_dir))


def _step_rows(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    step_props: dict[str, Any] = {}
    step_rows = []
    for cols, rows, props in _load_tables(output_dir / "overview_page.json"):
        if "stepTimeMs" in cols:
            step_props.update(props)
            step_rows = rows
        if "bottleneck" in props:
            step_props.update(props)
    return step_props, step_rows


def _kernel_rows(output_dir: Path) -> list[dict[str, Any]]:
    tables = _load_tables(output_dir / "kernel_stats.json")
    if not tables:
        return []
    _cols, rows, _props = tables[0]
    return rows


def _ranked_kernel_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -(_float_value(row, "total_duration_us") or 0.0),
            str(row.get("kernel_name") or ""),
        ),
    )


def _load_tables(path: Path) -> list[tuple[list[str], list[dict[str, Any]], dict[str, Any]]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed xprof table JSON: {path}") from error
    tables = data if isinstance(data, list) else [data]
    return [_table_rows(table) for table in tables if isinstance(table, dict) and "cols" in table]


def _iter_tables(table_dir: Path) -> Iterable[tuple[list[str], list[dict[str, Any]], dict[str, Any]]]:
    for path in sorted(table_dir.glob("*.json")):
        yield from _load_tables(path)


def _table_rows(table: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    cols = [str(col.get("id") or col.get("label")) for col in table.get("cols", []) if isinstance(col, dict)]
    rows = []
    for row in table.get("rows", []):
        if not isinstance(row, dict):
            continue
        values = [cell.get("v") if isinstance(cell, dict) else cell for cell in row.get("c", [])]
        rows.append(dict(zip(cols, values, strict=False)))
    props = table.get("p", {})
    return cols, rows, cast(dict[str, Any], props if isinstance(props, dict) else {})


def _float_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _int_value(row: dict[str, Any], key: str) -> int | None:
    value = _float_value(row, key)
    if value is None:
        return None
    return int(value)
