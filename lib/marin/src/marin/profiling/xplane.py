# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ingest XPlane protobuf profiles through xprof tables."""

from __future__ import annotations

import json
import logging
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from marin.profiling.ingest import (
    _collective_kind,
    _derive_optimization_candidates,
    _op_category,
    _sha256_for_path,
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


@dataclass(frozen=True)
class XPlaneTableExport:
    """Paths and sizes produced while converting an XPlane protobuf with xprof."""

    output_dir: Path
    table_sizes: dict[str, int]
    trace_event_count: int | None


def find_xplane_file(profile_dir: Path) -> Path:
    """Locate an XPlane protobuf inside a downloaded JAX profile artifact."""
    if not profile_dir.exists():
        raise FileNotFoundError(f"Profile directory does not exist: {profile_dir}")

    candidates = sorted(profile_dir.rglob("*.xplane.pb"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"No *.xplane.pb file found under '{profile_dir}'.")


def export_xplane_tables(
    xplane_path: Path,
    output_dir: Path,
    *,
    count_trace_events: bool = False,
) -> XPlaneTableExport:
    """Convert an XPlane protobuf into compact xprof table JSON files."""
    from xprof.convert import raw_to_tool_data

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
) -> ProfileSummary:
    """Summarize an XPlane protobuf into the normalized profile summary schema."""
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
        optimization_candidates=_derive_optimization_candidates(summary),
    )


def _trace_event_count_from_xprof(xplane_path: Path) -> int | None:
    from xprof.convert import raw_to_tool_data

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
    data = json.loads(path.read_text(encoding="utf-8"))
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
