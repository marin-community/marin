# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read profiler summary and xprof table artifacts for roofline attribution."""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from marin.profiling.schema import profile_summary_from_dict
from marin.profiling.xplane import find_xplane_file, summarize_xplane, summarize_xplane_tables
from marin.tools.roofline.types import ObservedTimeBasis


@dataclass(frozen=True)
class ObservedProfileRow:
    name: str
    total_time: float
    count: int | None
    avg_time: float | None
    source: str
    basis: str
    op_name: str | None = None
    kernel_name: str | None = None

    @property
    def match_text(self) -> str:
        return " ".join(part for part in [self.op_name, self.name, self.kernel_name] if part)


@dataclass(frozen=True)
class ProfileIngestResult:
    rows: list[ObservedProfileRow]
    source_path: str | None
    warnings: list[str]
    profile_devices: int | None = None
    profile_steps: int | None = None


def ingest_profile(profile: str | None) -> ProfileIngestResult:
    if profile is None:
        return ProfileIngestResult(rows=[], source_path=None, warnings=[])
    if _looks_like_wandb_artifact(profile):
        return ProfileIngestResult(
            rows=[],
            source_path=profile,
            warnings=[
                "W&B profiler artifact refs are recorded but not downloaded by roofline v1; "
                "pass a local profile path."
            ],
        )

    path = Path(profile)
    if not path.exists():
        raise FileNotFoundError(f"Profile path does not exist: {path}")
    if path.is_file() and path.suffix == ".json":
        return _ingest_summary_or_table(path)
    if path.is_file() and path.name.endswith(".xplane.pb"):
        summary = summarize_xplane(path, output_dir=None)
        return _rows_from_profile_summary_dict(summary.to_dict(), source_path=str(path))
    if path.is_dir():
        return _ingest_profile_dir(path)
    raise ValueError(f"Unsupported profile input: {path}")


def _ingest_profile_dir(path: Path) -> ProfileIngestResult:
    xprof_dir = _find_xprof_dir(path)
    warnings: list[str] = []
    rows: list[ObservedProfileRow] = []
    if xprof_dir is not None:
        rows.extend(_rows_from_xprof_dir(xprof_dir))
        if not rows:
            xplane_path = _find_xplane_for_tables(path)
            if xplane_path is not None:
                summary = summarize_xplane_tables(xprof_dir, xplane_path=xplane_path)
                summary_result = _rows_from_profile_summary_dict(summary.to_dict(), source_path=str(xprof_dir))
                rows.extend(summary_result.rows)
                warnings.extend(summary_result.warnings)

    summary_path = _find_summary_json(path)
    if summary_path is not None and not rows:
        rows.extend(_ingest_summary_or_table(summary_path).rows)

    if not rows:
        try:
            xplane_path = find_xplane_file(path)
            summary = summarize_xplane(xplane_path, output_dir=None)
            rows.extend(_rows_from_profile_summary_dict(summary.to_dict(), source_path=str(xplane_path)).rows)
        except FileNotFoundError:
            warnings.append("No profile_summary JSON, xprof table directory, or XPlane protobuf was found.")

    profile_devices = _infer_profile_devices(xprof_dir)
    profile_steps = _infer_profile_steps(path)
    unaccounted = _infer_unaccounted_time(path)
    if unaccounted is not None:
        rows.append(unaccounted)
    return ProfileIngestResult(
        rows=_dedupe_rows(rows),
        source_path=str(path),
        warnings=warnings,
        profile_devices=profile_devices,
        profile_steps=profile_steps,
    )


def _ingest_summary_or_table(path: Path) -> ProfileIngestResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("schema_version") == "profile_summary.v1":
        return _rows_from_profile_summary_dict(payload, source_path=str(path))
    if isinstance(payload, dict) and "cols" in payload:
        return ProfileIngestResult(
            rows=_rows_from_xprof_table(payload, source=str(path)),
            source_path=str(path),
            warnings=[],
        )
    if isinstance(payload, list):
        rows = []
        for item in payload:
            if isinstance(item, dict) and "cols" in item:
                rows.extend(_rows_from_xprof_table(item, source=str(path)))
        return ProfileIngestResult(rows=rows, source_path=str(path), warnings=[])
    raise ValueError(f"Unsupported profile JSON schema: {path}")


def _rows_from_profile_summary_dict(payload: dict[str, Any], *, source_path: str) -> ProfileIngestResult:
    summary = profile_summary_from_dict(payload)
    rows = [
        ObservedProfileRow(
            name=op.name,
            op_name=op.tf_op_path,
            kernel_name=op.name,
            total_time=op.exclusive_duration,
            count=op.count,
            avg_time=op.avg_duration,
            source="profile_summary",
            basis="track_summed_profile_hot_op_time_us",
        )
        for op in summary.hot_ops
    ]
    warnings = list(summary.trace_overview.quality_warnings)
    return ProfileIngestResult(rows=rows, source_path=source_path, warnings=warnings)


def _rows_from_xprof_dir(path: Path) -> list[ObservedProfileRow]:
    for name in ("kernel_stats.json", "framework_op_stats.json", "hlo_stats.json"):
        table_path = path / name
        if table_path.exists():
            rows = _ingest_summary_or_table(table_path).rows
            if rows:
                return rows
    return []


def _infer_profile_devices(xprof_dir: Path | None) -> int | None:
    if xprof_dir is None:
        return None
    overview_path = xprof_dir / "overview_page.json"
    if not overview_path.exists():
        return None
    payload = json.loads(overview_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return None
    for table in payload:
        if not isinstance(table, dict):
            continue
        props = table.get("p")
        if not isinstance(props, dict):
            continue
        devices = _int_text_value(props.get("device_core_count"))
        if devices is not None and devices > 0:
            return devices
    return None


def _infer_profile_steps(profile_dir: Path) -> int | None:
    for trace_path in _candidate_trace_paths(profile_dir):
        steps = _profile_steps_from_trace(trace_path)
        if steps is not None:
            return steps
    return None


def _candidate_trace_paths(profile_dir: Path) -> list[Path]:
    return [
        *sorted(profile_dir.rglob("perfetto_trace.json.gz")),
        *sorted(profile_dir.rglob("*.trace.json.gz")),
    ]


def _infer_unaccounted_time(profile_dir: Path) -> ObservedProfileRow | None:
    for trace_path in _candidate_trace_paths(profile_dir):
        row = _unaccounted_time_from_trace(trace_path)
        if row is not None:
            return row
    return None


def _unaccounted_time_from_trace(trace_path: Path) -> ObservedProfileRow | None:
    opener = gzip.open if trace_path.name.endswith(".gz") else open
    with opener(trace_path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    events = payload.get("traceEvents")
    if not isinstance(events, list):
        return None

    gpu_pids = _gpu_process_ids(events)
    if not gpu_pids:
        return None

    device_steps = _device_train_step_intervals(events)
    if not device_steps:
        return None

    events_by_gpu_pid: dict[int, list[tuple[float, float]]] = {pid: [] for pid in gpu_pids.values()}
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        pid = _int_text_value(event.get("pid"))
        if pid not in events_by_gpu_pid:
            continue
        timestamp = _float_text_value(event.get("ts"))
        duration = _float_text_value(event.get("dur"))
        if timestamp is None or duration is None or duration <= 0:
            continue
        events_by_gpu_pid[pid].append((timestamp, timestamp + duration))

    unaccounted_time = 0.0
    count = 0
    for device_id, start, end in device_steps:
        pid = gpu_pids.get(device_id)
        if pid is None:
            continue
        clipped_intervals = [
            (max(event_start, start), min(event_end, end))
            for event_start, event_end in events_by_gpu_pid[pid]
            if event_start < end and event_end > start
        ]
        active_time = _interval_union(clipped_intervals)
        unaccounted_time += max(0.0, end - start - active_time)
        count += 1

    if count == 0 or unaccounted_time <= 0:
        return None
    return ObservedProfileRow(
        name="unaccounted_for",
        op_name="unaccounted_for",
        kernel_name="empty_device_train_step_time",
        total_time=unaccounted_time,
        count=count,
        avg_time=unaccounted_time / count,
        source="perfetto_trace",
        basis=ObservedTimeBasis.TRACE_EMPTY_TRAIN_STEP_TIME.value,
    )


def _gpu_process_ids(events: list[object]) -> dict[int, int]:
    gpu_pids = {}
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "M" or event.get("name") != "process_name":
            continue
        args = event.get("args")
        if not isinstance(args, dict):
            continue
        process_name = str(args.get("name") or "")
        match = re.fullmatch(r"/device:GPU:(\d+)", process_name)
        if match is None:
            continue
        pid = _int_text_value(event.get("pid"))
        if pid is not None:
            gpu_pids[int(match.group(1))] = pid
    return gpu_pids


def _device_train_step_intervals(events: list[object]) -> list[tuple[int, float, float]]:
    intervals = []
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        name = str(event.get("name") or "")
        args = event.get("args")
        if "CommonPjRtLoadedExecutable::Execute (jit_train_step)" not in name or not isinstance(args, dict):
            continue
        device_id = _int_text_value(args.get("global_device_id"))
        timestamp = _float_text_value(event.get("ts"))
        duration = _float_text_value(event.get("dur"))
        if device_id is None or timestamp is None or duration is None or duration <= 0:
            continue
        intervals.append((device_id, timestamp, timestamp + duration))
    return intervals


def _interval_union(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    merged = sorted(intervals)
    total = 0.0
    start, end = merged[0]
    for next_start, next_end in merged[1:]:
        if next_start <= end:
            end = max(end, next_end)
            continue
        total += end - start
        start, end = next_start, next_end
    return total + end - start


def _profile_steps_from_trace(trace_path: Path) -> int | None:
    opener = gzip.open if trace_path.name.endswith(".gz") else open
    with opener(trace_path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    events = payload.get("traceEvents")
    if not isinstance(events, list):
        return None

    top_level_run_ids: set[str] = set()
    top_level_timestamps: list[float] = []
    pjit_timestamps: list[float] = []
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        name = str(event.get("name") or "")
        args = event.get("args")
        if _is_top_level_jit_train_step(name, args):
            if isinstance(args, dict) and args.get("run_id") is not None:
                top_level_run_ids.add(str(args["run_id"]))
                continue
            timestamp = _float_text_value(event.get("ts"))
            if timestamp is not None:
                top_level_timestamps.append(timestamp)
            continue
        if name == "PjitFunction(train_step)":
            timestamp = _float_text_value(event.get("ts"))
            if timestamp is not None:
                pjit_timestamps.append(timestamp)

    if top_level_run_ids:
        return len(top_level_run_ids)
    clustered_top_level = _clustered_timestamp_count(top_level_timestamps)
    if clustered_top_level is not None:
        return clustered_top_level
    return _clustered_timestamp_count(pjit_timestamps)


def _is_top_level_jit_train_step(name: str, args: object) -> bool:
    if name != "CommonPjRtLoadedExecutable::Execute (jit_train_step)":
        return False
    if not isinstance(args, dict):
        return True
    return "global_device_id" not in args


def _clustered_timestamp_count(timestamps: list[float]) -> int | None:
    if not timestamps:
        return None
    count = 0
    previous: float | None = None
    for timestamp in sorted(timestamps):
        if previous is None or timestamp - previous > 1000.0:
            count += 1
        previous = timestamp
    return count


def _rows_from_xprof_table(table: dict[str, Any], *, source: str) -> list[ObservedProfileRow]:
    column_names = [str(col.get("id") or col.get("label")) for col in table.get("cols", []) if isinstance(col, dict)]
    parsed = []
    for row in table.get("rows", []):
        if not isinstance(row, dict):
            continue
        values = [cell.get("v") if isinstance(cell, dict) else cell for cell in row.get("c", [])]
        parsed.append(dict(zip(column_names, values, strict=False)))

    rows = []
    for row in parsed:
        kernel_name = _string_value(row, "kernel_name", "Kernel name", "name")
        op_name = _string_value(row, "op_name", "framework_op_name", "hlo_op_name", "tf_op", "operation")
        display_name = op_name or kernel_name
        if display_name is None:
            continue
        total_time = _float_value(row, "total_duration_us", "total_time_us", "total_time", "duration_us")
        count = _int_value(row, "occurrences", "count", "calls")
        avg_time = _float_value(row, "avg_duration_us", "avg_time_us", "mean_duration_us")
        if total_time is None and avg_time is not None and count is not None:
            total_time = avg_time * count
        if avg_time is None and total_time is not None and count:
            avg_time = total_time / count
        if total_time is None:
            continue
        rows.append(
            ObservedProfileRow(
                name=display_name,
                op_name=op_name,
                kernel_name=kernel_name,
                total_time=total_time,
                count=count,
                avg_time=avg_time,
                source="xprof",
                basis="track_summed_xprof_kernel_time_us",
            )
        )
    return rows


def _dedupe_rows(rows: list[ObservedProfileRow]) -> list[ObservedProfileRow]:
    seen: set[tuple[str, float, int | None, str]] = set()
    deduped = []
    for row in rows:
        key = (row.name, row.total_time, row.count, row.basis)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _find_summary_json(path: Path) -> Path | None:
    direct = path / "profile_summary.json"
    if direct.exists():
        return direct
    adjacent = path.parent / f"{path.name}_summary.json"
    if adjacent.exists():
        return adjacent
    candidates = sorted(path.glob("*summary.json"))
    return candidates[0] if candidates else None


def _find_xprof_dir(path: Path) -> Path | None:
    if (path / "kernel_stats.json").exists():
        return path
    adjacent = path.parent / f"{path.name}_xprof_tables"
    if adjacent.exists():
        return adjacent
    candidates = sorted(candidate for candidate in path.glob("*xprof*") if candidate.is_dir())
    return candidates[0] if candidates else None


def _find_xplane_for_tables(path: Path) -> Path | None:
    candidates = sorted(path.rglob("*.xplane.pb"))
    return candidates[0] if len(candidates) == 1 else None


def _looks_like_wandb_artifact(value: str) -> bool:
    return ":" in value and value.count("/") >= 2 and not Path(value).exists()


def _string_value(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _float_value(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
    return None


def _float_text_value(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _int_text_value(value: object) -> int | None:
    numeric = _float_text_value(value)
    if numeric is None:
        return None
    return int(numeric)


def _int_value(row: dict[str, Any], *keys: str) -> int | None:
    value = _float_value(row, *keys)
    if value is None:
        return None
    return int(value)
