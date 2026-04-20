# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serial ablation runner for the prod-scale loadtest scenario.

Each step launches ``iris-loadtest scenario prod-scale`` as a fresh
subprocess so that import-time env-gate reads pick up the cumulative
toggle set. Between steps the runner SIGTERMs / SIGKILLs any stray child
processes and removes ``/tmp/loadtest-*`` scratch dirs — a previous
step's runaway worker will NOT bleed CPU into the next step.

After every step the runner parses the step's ``prod-scale-metrics.json``
and counts ``Slow *`` warnings in its log, then writes a combined
``REPORT.md`` under the output directory comparing all steps side-by-side.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import click

DEFAULT_STEPS: list[tuple[str, list[str]]] = [
    ("0-baseline", []),
]

# Slow-log categories the harness emits via ``slow_log``. Tracked so the
# combined report can surface pathological RPCs (the metrics reservoir
# bounds RPC samples; the slow-log is an unbounded per-event counter).
SLOW_CATEGORIES = (
    "heartbeat",
    "provider",
    "scheduling",
    "buffer_assignments",
    "building_counts",
    "dispatch_assignments_direct",
)


@dataclass
class StepResult:
    name: str
    exit_code: int
    wall_seconds: float
    metrics_path: Path | None
    log_path: Path
    slow_counts: dict[str, int]


def _cleanup() -> None:
    """SIGTERM then SIGKILL any stray scenario / worker processes + scratch dirs.

    Patterns are deliberately chosen so they do NOT match this driver's own
    process — if the user launches via ``uv run iris-loadtest ablation`` the
    parent's argv contains ``iris-loadtest``, and a broad ``pkill -f
    iris-loadtest`` would target the driver itself. Match only the child
    shapes we actually spawn:

    * ``iris.loadtest.cli`` — the ``-m`` module path used for scenario
      subprocesses (see ``run_series``).
    * ``iris.loadtest.synthetic_worker_main`` — synthetic worker subprocess.

    pkill returns non-zero when no match; we ignore the exit code. We also
    pass ``-P 1`` (descend only from init) is not portable, so we accept
    the theoretical risk that a scenario child spawned ``iris.loadtest.cli``
    in some indirect way — in practice the scenario child is the only one
    that matches that pattern.
    """
    self_pid = os.getpid()
    patterns = ("iris.loadtest.cli", "iris.loadtest.synthetic_worker_main")
    for signame in ("TERM", "KILL"):
        for pattern in patterns:
            # pgrep to get matching pids so we can exclude self_pid. ``pkill``
            # has no "exclude pid" flag; we replicate it via pgrep + kill.
            pgrep = subprocess.run(
                ["pgrep", "-f", pattern],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for line in pgrep.stdout.splitlines():
                try:
                    pid = int(line.strip())
                except ValueError:
                    continue
                if pid == self_pid:
                    continue
                try:
                    os.kill(pid, signal.SIGTERM if signame == "TERM" else signal.SIGKILL)
                except ProcessLookupError:
                    pass
        time.sleep(1 if signame == "TERM" else 0)
    for prefix in ("loadtest-db-", "loadtest-controller-state-"):
        for path in Path("/tmp").glob(f"{prefix}*"):
            shutil.rmtree(path, ignore_errors=True)


def _count_slow(log_path: Path) -> dict[str, int]:
    if not log_path.exists():
        return {}
    counts: dict[str, int] = {k: 0 for k in SLOW_CATEGORIES}
    pattern = re.compile(r"Slow (\w+)")
    with log_path.open("r", errors="replace") as fh:
        for line in fh:
            m = pattern.search(line)
            if m and m.group(1) in counts:
                counts[m.group(1)] += 1
    return counts


def _run_step(
    step_name: str,
    step_flags: list[str],
    *,
    base_cmd: list[str],
    out_dir: Path,
    timeout_seconds: int,
) -> StepResult:
    step_dir = out_dir / f"step-{step_name}"
    log_path = out_dir / f"step-{step_name}.log"
    cmd = [*base_cmd, "--output-dir", str(step_dir), *step_flags]

    click.echo(f"=== {time.strftime('%H:%M:%S')} step {step_name} START ===")
    t0 = time.monotonic()
    exit_code = 0
    with log_path.open("w") as fh:
        try:
            result = subprocess.run(
                cmd,
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
                # New session so we can SIGKILL the whole pgroup on timeout.
                start_new_session=True,
            )
            exit_code = result.returncode
        except subprocess.TimeoutExpired as exc:
            exit_code = 124
            # TimeoutExpired leaves the child alive; its pgroup was created
            # by start_new_session. Kill the whole group.
            if exc.pid is not None:
                try:
                    os.killpg(os.getpgid(exc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
    wall = time.monotonic() - t0
    click.echo(f"=== {time.strftime('%H:%M:%S')} step {step_name} END rc={exit_code} wall={wall:.1f}s ===")

    metrics_path = step_dir / "prod-scale-metrics.json"
    slow_counts = _count_slow(log_path)
    return StepResult(
        name=step_name,
        exit_code=exit_code,
        wall_seconds=wall,
        metrics_path=metrics_path if metrics_path.exists() else None,
        log_path=log_path,
        slow_counts=slow_counts,
    )


def _fmt(value: float | None, width: int = 10) -> str:
    if value is None:
        return "—".rjust(width)
    return f"{value:.1f}".rjust(width)


def _percentiles(metrics: dict | None, key: str) -> dict:
    if metrics is None:
        return {}
    entry = metrics.get(key, {})
    return entry.get("percentiles", {}) or {}


def _rpc_percentiles(metrics: dict | None, method: str) -> dict:
    if metrics is None:
        return {}
    rpc = metrics.get("rpc_ms", {}).get(method, {})
    return rpc.get("percentiles", {}) or {}


def _load_metrics(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


def write_report(out_dir: Path, results: list[StepResult], header: str) -> Path:
    report_path = out_dir / "REPORT.md"
    metrics = {r.name: _load_metrics(r.metrics_path) for r in results}

    lines: list[str] = [f"# {header}", ""]
    lines += [
        "## writer_lock_hold_ms / dashboard_query_ms",
        "",
        "| step | rc | wall | writer P50 | writer P95 | writer max | writer acqs | dash P50 | dash P95 | dash max |",
        "| ---- | -- | ---- | ---------- | ---------- | ---------- | ----------- | -------- | -------- | -------- |",
    ]
    for r in results:
        m = metrics[r.name]
        wp = _percentiles(m, "writer_lock_hold_ms")
        dp = _percentiles(m, "dashboard_query_ms")
        total = (m or {}).get("writer_lock_hold_ms", {}).get("total", "—")
        lines.append(
            f"| {r.name} | {r.exit_code} | {r.wall_seconds:.0f} |"
            f" {_fmt(wp.get('p50'))} | {_fmt(wp.get('p95'))} | {_fmt(wp.get('max'))} |"
            f" {total} | {_fmt(dp.get('p50'))} | {_fmt(dp.get('p95'))} | {_fmt(dp.get('max'))} |"
        )
    lines.append("")

    lines += [
        "## rpc_ms (per WorkerProvider method, batch wall-clock ms)",
        "",
        "| step | method | calls | P50 | P95 | P99 | max |",
        "| ---- | ------ | ----- | --- | --- | --- | --- |",
    ]
    for r in results:
        m = metrics[r.name]
        rpc = (m or {}).get("rpc_ms", {})
        if not rpc:
            lines.append(f"| {r.name} | — | 0 | — | — | — | — |")
            continue
        for method in sorted(rpc):
            entry = rpc[method]
            p = entry.get("percentiles", {}) or {}
            lines.append(
                f"| {r.name} | {method} | {entry.get('total', 0)} |"
                f" {_fmt(p.get('p50'))} | {_fmt(p.get('p95'))} | {_fmt(p.get('p99'))} | {_fmt(p.get('max'))} |"
            )
    lines.append("")

    lines += [
        "## Slow-log counts (from controller log output)",
        "",
        "| step | " + " | ".join(SLOW_CATEGORIES) + " |",
        "| ---- | " + " | ".join("-" * len(c) for c in SLOW_CATEGORIES) + " |",
    ]
    for r in results:
        row = [r.name] + [str(r.slow_counts.get(c, 0)) for c in SLOW_CATEGORIES]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    report_path.write_text("\n".join(lines))
    return report_path


def run_series(
    *,
    steps: list[tuple[str, list[str]]],
    scenario: str,
    preload_workers: int,
    duration: int,
    burst_jobs: int,
    cpu_jobs: int,
    cpu_tasks_per_job: int,
    out_dir: Path,
    timeout_seconds: int,
    common_flags: list[str],
    iris_loadtest: list[str] | None = None,
) -> list[StepResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd_prefix = iris_loadtest or [sys.executable, "-m", "iris.loadtest.cli"]
    base_cmd = [
        *cmd_prefix,
        "scenario",
        scenario,
        "--preload-workers",
        str(preload_workers),
        "--duration",
        str(duration),
        "--burst-jobs",
        str(burst_jobs),
        "--cpu-jobs",
        str(cpu_jobs),
        "--cpu-tasks-per-job",
        str(cpu_tasks_per_job),
        *common_flags,
    ]

    results: list[StepResult] = []
    for step_name, step_flags in steps:
        _cleanup()
        results.append(
            _run_step(
                step_name,
                step_flags,
                base_cmd=base_cmd,
                out_dir=out_dir,
                timeout_seconds=timeout_seconds,
            )
        )
        _cleanup()
    return results
