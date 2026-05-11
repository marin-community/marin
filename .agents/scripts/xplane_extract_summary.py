#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract xprof tables from a JAX XPlane profile and write a compact report."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

SMALL_XPROF_TOOLS = (
    "overview_page",
    "kernel_stats",
    "framework_op_stats",
    "op_profile",
    "hlo_stats",
    "memory_profile",
    "input_pipeline_analyzer",
)


def _table_rows(table: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    cols = [col.get("id") or col.get("label") for col in table.get("cols", [])]
    rows = []
    for row in table.get("rows", []):
        values = [cell.get("v") if isinstance(cell, dict) else cell for cell in row.get("c", [])]
        rows.append(dict(zip(cols, values, strict=False)))
    return cols, rows


def _load_tables(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    tables = data if isinstance(data, list) else [data]
    return [table for table in tables if isinstance(table, dict) and "cols" in table]


def _extract_xprof_tables(xplane: Path, output_dir: Path) -> dict[str, int]:
    from xprof.convert import raw_to_tool_data

    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for tool in SMALL_XPROF_TOOLS:
        data, _content_type = raw_to_tool_data.xspace_to_tool_data(
            [str(xplane)],
            tool,
            {"use_saved_result": False},
        )
        if data is None:
            continue
        target = output_dir / f"{tool}.json"
        target.write_bytes(data)
        sizes[tool] = len(data)
    return sizes


def _trace_event_count(xplane: Path) -> int | None:
    from xprof.convert import raw_to_tool_data

    data, _content_type = raw_to_tool_data.xspace_to_tool_data(
        [str(xplane)],
        "trace_viewer@",
        {"use_saved_result": False},
    )
    if data is None:
        return None
    match = re.search(rb'"returnedEventsSize"\s*:\s*(\d+)', data[:4096])
    if match is None:
        return None
    return int(match.group(1))


def _step_rows(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    step_props: dict[str, Any] = {}
    step_rows = []
    for table in _load_tables(output_dir / "overview_page.json"):
        cols, rows = _table_rows(table)
        if "stepTimeMs" in cols:
            step_props.update(table.get("p", {}))
            step_rows = rows
        props = table.get("p", {})
        if "bottleneck" in props:
            step_props.update(props)
    return step_props, step_rows


def _kernel_summary(output_dir: Path) -> dict[str, Any]:
    kernel_tables = _load_tables(output_dir / "kernel_stats.json")
    if not kernel_tables:
        return {}

    _cols, rows = _table_rows(kernel_tables[0])

    def duration_if(predicate: Callable[[dict[str, Any]], bool]) -> float:
        return sum(float(row.get("total_duration_us") or 0.0) for row in rows if predicate(row))

    total_us = duration_if(lambda _row: True)
    nccl_us = duration_if(lambda row: "nccl" in str(row.get("kernel_name", "")).lower())
    allgather_us = duration_if(lambda row: "allgather" in str(row.get("kernel_name", "")).lower())
    reduce_scatter_us = duration_if(lambda row: "reducescatter" in str(row.get("kernel_name", "")).lower())
    allreduce_us = duration_if(lambda row: "allreduce" in str(row.get("kernel_name", "")).lower())
    lambda_us = duration_if(lambda row: str(row.get("kernel_name", "")).startswith("_lambda_"))
    nvjet_us = duration_if(lambda row: str(row.get("kernel_name", "")).startswith("nvjet_"))
    tensorcore_us = duration_if(lambda row: bool(row.get("is_kernel_using_tensor_core")))

    return {
        "rows": len(rows),
        "total_s": total_us / 1_000_000,
        "nccl_s": nccl_us / 1_000_000,
        "nccl_pct": 100.0 * nccl_us / total_us if total_us else 0.0,
        "allgather_s": allgather_us / 1_000_000,
        "reduce_scatter_s": reduce_scatter_us / 1_000_000,
        "allreduce_s": allreduce_us / 1_000_000,
        "lambda_s": lambda_us / 1_000_000,
        "nvjet_s": nvjet_us / 1_000_000,
        "tensorcore_s": tensorcore_us / 1_000_000,
        "top_kernels": rows[:12],
    }


def _format_seconds(value: float | int | None) -> str:
    if value is None:
        return "unknown"
    return f"{float(value):.3f}s"


def _write_report(
    report: Path,
    *,
    xplane: Path,
    run_url: str | None,
    trace_events: int | None,
    step_props: dict[str, Any],
    steps: list[dict[str, Any]],
    kernels: dict[str, Any],
) -> None:
    lines = ["# XPlane Summary", ""]
    if run_url:
        lines.extend([f"Run: {run_url}", ""])
    lines.extend([f"XPlane: `{xplane}`", ""])

    if trace_events is not None:
        lines.extend([f"- Direct xprof trace events: `{trace_events:,}`.", ""])

    lines.append("## Step Timing")
    lines.append("")
    if not steps:
        lines.append("- No xprof step rows were recovered.")
    else:
        for step in steps:
            lines.append(
                "- Step {step}: total `{total}`, collectives `{collectives}`, compute `{compute}`, "
                "input `{input_time}`, other `{other}`.".format(
                    step=step.get("stepnum"),
                    total=_format_seconds(float(step.get("stepTimeMs") or 0.0) / 1000.0),
                    collectives=_format_seconds(float(step.get("deviceCollectivesTimeMs") or 0.0) / 1000.0),
                    compute=_format_seconds(float(step.get("deviceComputeTimeMs") or 0.0) / 1000.0),
                    input_time=_format_seconds(float(step.get("infeedTimeMs") or 0.0) / 1000.0),
                    other=_format_seconds(float(step.get("otherTimeMs") or 0.0) / 1000.0),
                )
            )
    if step_props:
        lines.append(f"- xprof bottleneck: `{step_props.get('bottleneck', 'unknown')}`.")
        statement = step_props.get("device_collectives_statement") or step_props.get("statement")
        if statement:
            lines.append(f"- xprof statement: {statement}")
    lines.append("")

    lines.append("## Kernel Summary")
    lines.append("")
    if not kernels:
        lines.append("- No kernel table was recovered.")
    else:
        lines.extend(
            [
                f"- Kernel rows: `{kernels['rows']}`.",
                f"- Visible kernel duration: `{kernels['total_s']:.3f}s`.",
                f"- NCCL collectives: `{kernels['nccl_s']:.3f}s` / `{kernels['nccl_pct']:.2f}%`.",
                f"- All-gather: `{kernels['allgather_s']:.3f}s`.",
                f"- Reduce-scatter: `{kernels['reduce_scatter_s']:.3f}s`.",
                f"- All-reduce: `{kernels['allreduce_s']:.3f}s`.",
                f"- `_lambda_` kernels: `{kernels['lambda_s']:.3f}s`.",
                f"- `nvjet_*` kernels: `{kernels['nvjet_s']:.3f}s`.",
                f"- Tensor-core kernels: `{kernels['tensorcore_s']:.3f}s`.",
                "",
                "Top kernels:",
                "",
            ]
        )
        for row in kernels["top_kernels"]:
            lines.append(
                "- `{rank}` `{name}`: `{duration:.3f}s` over `{occurrences}` launches.".format(
                    rank=int(float(row.get("rank") or 0)),
                    name=row.get("kernel_name"),
                    duration=float(row.get("total_duration_us") or 0.0) / 1_000_000,
                    occurrences=int(float(row.get("occurrences") or 0)),
                )
            )

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("xplane", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--run-url")
    parser.add_argument("--count-trace-events", action="store_true")
    args = parser.parse_args()

    _extract_xprof_tables(args.xplane, args.output_dir)
    trace_events = _trace_event_count(args.xplane) if args.count_trace_events else None
    step_props, steps = _step_rows(args.output_dir)
    kernels = _kernel_summary(args.output_dir)

    report = args.report or args.output_dir / "xplane-summary.md"
    _write_report(
        report,
        xplane=args.xplane,
        run_url=args.run_url,
        trace_events=trace_events,
        step_props=step_props,
        steps=steps,
        kernels=kernels,
    )
    print(report)


if __name__ == "__main__":
    main()
