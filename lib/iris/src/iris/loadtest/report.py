# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Human-readable rendering of a :class:`ScenarioResult` + metrics artifacts.

The CLI hands a raw ``ScenarioResult`` (path to metrics.json) plus the live
harness over; this module pulls out the fields callers actually want to see in
a terminal (percentiles, counters, fleet size, fix flags in effect) and writes
a ``summary.md`` next to the metrics JSON.

The format is the "config / timings / counters" layout from the CLI-refactor
spec. Probe files, if present, are rendered as a P50/P95/P99/max table keyed
by the per-client/per-RPC label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from iris.loadtest.harness import LoadtestHarness
from iris.loadtest.scenarios import ScenarioResult


@dataclass(frozen=True)
class RunConfig:
    """Echo of the knobs a scenario was launched with."""

    scenario: str
    duration_s: float
    preload_workers: int
    burst_jobs: int
    probe_mix: str = "default"


def _percentile_line(label: str, stats: dict) -> str:
    return (
        f"  {label:<28s}: "
        f"P50={stats.get('p50', 0.0):>7.1f}  "
        f"P95={stats.get('p95', 0.0):>7.1f}  "
        f"P99={stats.get('p99', 0.0):>7.1f}  "
        f"max={stats.get('max', 0.0):>7.1f}  "
        f"n={stats.get('count', 0)}"
    )


def _extract_probe_stats(probe_path: Path | None) -> dict[str, dict] | None:
    if probe_path is None or not probe_path.exists():
        return None
    data = json.loads(probe_path.read_text())
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def render_summary(
    result: ScenarioResult,
    *,
    config: RunConfig,
    harness: LoadtestHarness | None = None,
    probe_path: Path | None = None,
) -> str:
    """Render the human-readable run summary (stdout + summary.md).

    ``harness`` is optional so this can also run against an already-finished
    run (metrics.json present, harness torn down). When provided, it surfaces
    live counters (peak thread count, synthetic workers, RSS) — otherwise we
    fall back to the numbers embedded in the metrics file.
    """
    data = json.loads(result.metrics_path.read_text())
    samples = data["samples"]
    peak_threads = max((s["active_scale_up_threads"] for s in samples), default=0)
    rss_peak = data.get("rss_peak_bytes", 0)
    create_attempts = samples[-1]["create_attempts"] if samples else 0
    create_failures = samples[-1]["create_failures"] if samples else 0

    synth_workers: int | None = None
    if harness is not None and harness.worker_pool is not None:
        synth_workers = harness.worker_pool.active_count()

    lines: list[str] = []
    lines.append("config:")
    lines.append(f"  scenario          : {config.scenario}")
    lines.append(f"  duration_s        : {config.duration_s:.0f}")
    lines.append(f"  preload_workers   : {config.preload_workers}")
    lines.append(f"  burst_jobs        : {config.burst_jobs}")
    lines.append(f"  probes            : {config.probe_mix}")
    lines.append("")

    lines.append("timings (P50 / P95 / P99 / max, ms):")
    lock_pct = data["writer_lock_hold_ms"]["percentiles"]
    query_pct = data["dashboard_query_ms"]["percentiles"]
    lines.append(_percentile_line("writer_lock_hold_ms", lock_pct))
    lines.append(_percentile_line("dashboard_query_ms", query_pct))

    probe_stats = _extract_probe_stats(probe_path)
    if probe_stats:
        for label in sorted(probe_stats.keys()):
            lines.append(_percentile_line(label, probe_stats[label]))
    lines.append("")

    lines.append("counters:")
    lines.append(f"  create_attempts            : {create_attempts}")
    lines.append(f"  create_failures            : {create_failures}")
    lines.append(f"  peak_scale_up_threads      : {peak_threads}")
    if synth_workers is not None:
        lines.append(f"  synthetic_workers_registered: {synth_workers}")
    lines.append(f"  rss_peak_mb                : {rss_peak / 1e6:.1f}")
    return "\n".join(lines) + "\n"


def write_summary(
    result: ScenarioResult,
    *,
    config: RunConfig,
    out_dir: Path,
    harness: LoadtestHarness | None = None,
    probe_path: Path | None = None,
) -> Path:
    """Render and write ``summary.md`` to ``out_dir``; return its path."""
    text = render_summary(result, config=config, harness=harness, probe_path=probe_path)
    path = out_dir / "summary.md"
    path.write_text(text)
    return path
