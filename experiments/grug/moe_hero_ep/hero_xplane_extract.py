# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reduce one XProf XPlane protobuf to a compact JSON record.

Written for the MoK vs `fixed_pooled_wave_all_to_all` hero head-to-head profiling round. One
XPlane per JAX process, 64 processes per arm, so the whole-file parse has to happen where the
files are (a CPU Iris task next to the bucket), and only this record travels.

Usage::

    <interpreter> hero_xplane_extract.py <xplane.pb|s3://...> --output rec.json [--label rank-07]

The record carries, for the capture window:

* file identity (size, sha256) and plane/line inventory, so a device-plane-less capture is
  detectable without opening the file again;
* the host ``CommonPjRtLoadedExecutable::Execute (jit_train_step)`` events -- count, run ids,
  durations -- which are the step times;
* per kernel-name totals on the device compute streams, with the mangled name, the HLO op the
  kernel was emitted for, theoretical occupancy, and grid/block geometry;
* summed vs interval-union busy time (the difference is observable concurrency), computed over
  the whole capture and per step;
* memcpy/memset byte and time totals by direction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

_NS_PER_US = 1_000.0
_PS_PER_US = 1_000_000.0
_EXECUTE_EVENT = "CommonPjRtLoadedExecutable::Execute (jit_train_step)"
_GRID_RE = re.compile(r"grid:\s*<?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", re.IGNORECASE)
_BLOCK_RE = re.compile(r"block:\s*<?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", re.IGNORECASE)
_BYTES_RE = re.compile(r"(?:num_bytes|size):\s*(\d+)")
# Kernels whose *individual* launches are kept, in launch order, so a per-layer / per-barrier
# comparison across ranks is possible. Everything else is aggregated. Keep this list short:
# one entry per rank per launch, and the whole point of the reduction is that it stays small.
_TRACE_INSTANCES = re.compile(r"barrier_all|dispatch_mlp_swiglu_combine_(?:fwd|bwd)_kernel|ncclDevKernel_SendRecv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_value(stat: Any, stat_names: dict[int, str]) -> Any:
    which = stat.WhichOneof("value")
    if which is None:
        return None
    if which == "ref_value":
        return stat_names.get(int(stat.ref_value), int(stat.ref_value))
    return getattr(stat, which)


def _resolve_stats(stats: Any, stat_names: dict[int, str]) -> dict[str, Any]:
    return {stat_names.get(int(s.metadata_id), str(s.metadata_id)): _stat_value(s, stat_names) for s in stats}


def _union_length(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        elif end > cur_end:
            cur_end = end
    return total + (cur_end - cur_start)


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)

    def pick(q: float) -> float:
        return ordered[min(n - 1, max(0, round(q * (n - 1))))]

    return {
        "min": ordered[0],
        "p50": pick(0.5),
        "p90": pick(0.9),
        "p99": pick(0.99),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def extract(xplane_path: Path, *, label: str, source_uri: str | None) -> dict[str, Any]:
    from marin.profiling.xplane import _xspace_message_class  # noqa: PLC0415

    xspace = _xspace_message_class()()
    xspace.ParseFromString(xplane_path.read_bytes())

    record: dict[str, Any] = {
        "label": label,
        "source_uri": source_uri or str(xplane_path),
        "file_bytes": xplane_path.stat().st_size,
        "sha256": _sha256(xplane_path),
        "planes": {},
        "warnings": [str(w) for w in getattr(xspace, "warnings", [])],
        "errors": [str(e) for e in getattr(xspace, "errors", [])],
    }

    # ---- host plane: step (Execute) events -------------------------------------------------
    steps: list[dict[str, Any]] = []
    host_events = 0
    for plane in xspace.planes:
        record["planes"][str(plane.name)] = {
            "lines": len(plane.lines),
            "events": sum(len(line.events) for line in plane.lines),
        }
        if not str(plane.name).startswith("/host:CPU"):
            continue
        stat_names = {int(k): str(v.name) for k, v in plane.stat_metadata.items()}
        names = {int(k): str(v.display_name or "") or str(v.name or "") for k, v in plane.event_metadata.items()}
        for line in plane.lines:
            base_us = float(line.timestamp_ns) / _NS_PER_US
            host_events += len(line.events)
            for event in line.events:
                if names.get(int(event.metadata_id)) != _EXECUTE_EVENT:
                    continue
                stats = _resolve_stats(event.stats, stat_names)
                steps.append(
                    {
                        "start_us": base_us + float(event.offset_ps) / _PS_PER_US,
                        "dur_us": float(event.duration_ps) / _PS_PER_US,
                        "run_id": str(stats.get("run_id", "")),
                    }
                )
    steps.sort(key=lambda s: s["start_us"])
    record["host_events"] = host_events
    record["steps"] = steps
    record["step_count"] = len(steps)
    record["step_durations_us"] = [s["dur_us"] for s in steps]
    run_ids = [int(s["run_id"]) for s in steps if s["run_id"].lstrip("-").isdigit()]
    record["step_run_ids_consecutive"] = bool(run_ids) and sorted(run_ids) == list(
        range(min(run_ids), min(run_ids) + len(run_ids))
    )

    # ---- device planes ---------------------------------------------------------------------
    device_planes = [p for p in xspace.planes if str(p.name).startswith("/device:") and p.lines]
    record["active_device_planes"] = [str(p.name) for p in device_planes]
    if not device_planes:
        record["device_plane_present"] = False
        return record
    record["device_plane_present"] = True

    step_bounds = [(s["start_us"], s["start_us"] + s["dur_us"]) for s in steps]

    def step_of(ts: float) -> int | None:
        for index, (lo, hi) in enumerate(step_bounds):
            if lo <= ts < hi:
                return index
        return None

    kernels: dict[str, dict[str, Any]] = {}
    memory_ops: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "us": 0.0, "bytes": 0.0})
    compute_intervals: list[tuple[float, float]] = []
    nccl_intervals: list[tuple[float, float]] = []
    nccl_by_name: dict[str, list[tuple[float, float]]] = {}
    noncomm_intervals: list[tuple[float, float]] = []
    compute_summed_us = 0.0
    all_intervals: list[tuple[float, float]] = []
    per_line: dict[str, dict[str, float]] = {}
    kernel_event_count = 0
    memory_event_count = 0
    device_event_count = 0
    window_start = float("inf")
    window_end = float("-inf")

    for plane in device_planes:
        stat_names = {int(k): str(v.name) for k, v in plane.stat_metadata.items()}
        names = {int(k): str(v.display_name or "") or str(v.name or "") for k, v in plane.event_metadata.items()}
        for line in plane.lines:
            line_name = str(line.name or line.display_name or f"line-{line.id}")
            base_us = float(line.timestamp_ns) / _NS_PER_US
            is_compute = "Compute" in line_name
            line_stats = per_line.setdefault(line_name, {"events": 0, "us": 0.0})
            for event in line.events:
                if event.duration_ps <= 0:
                    continue
                start = base_us + float(event.offset_ps) / _PS_PER_US
                dur = float(event.duration_ps) / _PS_PER_US
                end = start + dur
                name = names.get(int(event.metadata_id), "<unknown>")
                device_event_count += 1
                line_stats["events"] += 1
                line_stats["us"] += dur
                window_start = min(window_start, start)
                window_end = max(window_end, end)
                all_intervals.append((start, end))

                if name.startswith("Memcpy") or name.startswith("Memset"):
                    memory_event_count += 1
                    stats = _resolve_stats(event.stats, stat_names)
                    detail = str(stats.get("memcpy_details") or stats.get("memset_details") or "")
                    match = _BYTES_RE.search(detail)
                    bucket = memory_ops[name]
                    bucket["count"] += 1
                    bucket["us"] += dur
                    if match:
                        bucket["bytes"] += float(match.group(1))
                    continue

                kernel_event_count += 1
                if is_compute:
                    compute_summed_us += dur
                    compute_intervals.append((start, end))
                    if name.startswith("ncclDevKernel"):
                        nccl_intervals.append((start, end))
                        nccl_by_name.setdefault(name, []).append((start, end))
                    else:
                        noncomm_intervals.append((start, end))
                stats = _resolve_stats(event.stats, stat_names)
                entry = kernels.get(name)
                if entry is None:
                    entry = kernels[name] = {
                        "count": 0,
                        "total_us": 0.0,
                        "durations": [],
                        "mangled": str(stats.get("name") or ""),
                        "hlo_ops": defaultdict(int),
                        "hlo_module": str(stats.get("hlo_module") or ""),
                        "details": defaultdict(int),
                        "instances": [] if _TRACE_INSTANCES.search(name) else None,
                        "per_step_us": [0.0] * max(1, len(steps)),
                        "per_step_count": [0] * max(1, len(steps)),
                        "line": line_name,
                    }
                entry["count"] += 1
                entry["total_us"] += dur
                if entry["instances"] is not None:
                    entry["instances"].append((round(start, 3), round(dur, 3)))
                entry["hlo_ops"][str(stats.get("hlo_op") or "")] += 1
                entry["details"][str(stats.get("kernel_details") or "")] += 1
                if len(entry["durations"]) < 200_000:
                    entry["durations"].append(dur)
                step_index = step_of(start)
                if step_index is not None:
                    entry["per_step_us"][step_index] += dur
                    entry["per_step_count"][step_index] += 1

    record["device_events"] = device_event_count
    record["kernel_events"] = kernel_event_count
    record["memory_events"] = memory_event_count
    record["capture_window_us"] = [window_start, window_end]
    record["capture_span_us"] = window_end - window_start
    record["per_line"] = per_line

    for name, entry in kernels.items():
        durations = entry.pop("durations")
        entry["stats_us"] = _quantiles(durations)
        entry["name"] = name
        entry["hlo_ops"] = dict(sorted(entry["hlo_ops"].items(), key=lambda kv: -kv[1])[:8])
        entry["details"] = dict(sorted(entry["details"].items(), key=lambda kv: -kv[1])[:8])
    record["kernels"] = kernels
    record["memory_ops"] = {k: dict(v) for k, v in memory_ops.items()}

    record["compute_summed_us"] = compute_summed_us
    record["compute_union_us"] = _union_length(list(compute_intervals))
    record["compute_concurrency_surplus_us"] = compute_summed_us - record["compute_union_us"]
    record["device_union_us"] = _union_length(list(all_intervals))
    record["device_idle_us"] = record["capture_span_us"] - record["device_union_us"]

    # Observable comm/compute overlap: seconds where an NCCL kernel and a non-NCCL kernel are
    # both resident. `union(A) + union(B) - union(A u B)` is exact for interval sets.
    nccl_union = _union_length(list(nccl_intervals))
    noncomm_union = _union_length(list(noncomm_intervals))
    record["nccl_summed_us"] = sum(e - s for s, e in nccl_intervals)
    record["nccl_union_us"] = nccl_union
    record["noncomm_summed_us"] = sum(e - s for s, e in noncomm_intervals)
    record["noncomm_union_us"] = noncomm_union
    record["nccl_compute_overlap_us"] = nccl_union + noncomm_union - record["compute_union_us"]

    # Per-collective overlap: how much of each NCCL kernel's residency coincides with compute.
    # `union(A) + union(B) - union(A u B)` again, one collective family at a time.
    per_collective = {}
    for name, spans in nccl_by_name.items():
        own = _union_length(list(spans))
        both = _union_length(list(spans) + list(noncomm_intervals))
        per_collective[name] = {
            "count": len(spans),
            "summed_us": sum(e - s for s, e in spans),
            "union_us": own,
            "overlap_with_compute_us": own + noncomm_union - both,
        }
    record["nccl_overlap_by_name"] = per_collective

    # Per-step device busy, clipped to each host Execute range.
    per_step = []
    for index, step in enumerate(steps):
        lo = step["start_us"]
        hi = lo + step["dur_us"]
        clipped = [(max(lo, s), min(hi, e)) for s, e in compute_intervals if e > lo and s < hi]
        summed = sum(e - s for s, e in clipped)
        union = _union_length(clipped)
        per_step.append(
            {
                "index": index,
                "host_dur_us": step["dur_us"],
                "compute_summed_us": summed,
                "compute_union_us": union,
                "compute_surplus_us": summed - union,
                "idle_us": step["dur_us"] - union,
            }
        )
    record["per_step"] = per_step
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Reduce an XPlane protobuf to a JSON record.")
    parser.add_argument("xplane", help="Local path or s3:// URI of the xplane.pb")
    parser.add_argument("--output", required=True, help="Where to write the JSON record")
    parser.add_argument("--label", default="", help="Label carried into the record (e.g. rank id)")
    args = parser.parse_args()

    source = args.xplane
    if "://" in source:
        import fsspec  # noqa: PLC0415
        from rigging.filesystem.s3_compat import configure_coreweave_s3  # noqa: PLC0415

        # Without this the s3 filesystem has no CoreWeave credentials and fails deep inside s3fs.
        configure_coreweave_s3()
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "xplane.pb"
            fs, _, paths = fsspec.get_fs_token_paths(source)
            fs.get(paths[0], str(local))
            record = extract(local, label=args.label or source, source_uri=source)
    else:
        record = extract(Path(source), label=args.label or source, source_uri=None)

    Path(args.output).write_text(json.dumps(record))
    print(
        f"{args.label or source}: device_plane={record.get('device_plane_present')} "
        f"kernels={record.get('kernel_events')} steps={record.get('step_count')}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
