# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unified ``iris-loadtest`` CLI.

Subcommands:

- ``scenario NAME`` — run one of the canonical scenarios.
- ``ablation``      — sweep toggle combinations against prod-scale.
- ``list``          — list scenarios and toggle flags.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import click

SCENARIOS = ("burst", "api-timeouts", "incident", "fleet-wide", "prod-scale")


def _default_output_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(f"logs/autoscaler-loadtest/run-{ts}")


def _common_options(func):
    """Shared click options for scenario + ablation commands."""
    decorators = [
        click.option("--duration", type=int, default=300, show_default=True, help="Wall-clock seconds."),
        click.option("--preload-workers", type=int, default=600, show_default=True),
        click.option("--burst-jobs", type=int, default=500, show_default=True),
        click.option(
            "--cpu-jobs",
            type=int,
            default=10,
            show_default=True,
            help="prod-scale only: number of CPU-only jobs.",
        ),
        click.option(
            "--cpu-tasks-per-job",
            type=int,
            default=500,
            show_default=True,
            help="prod-scale only: replicas per CPU job.",
        ),
        click.option(
            "--output-dir",
            type=click.Path(file_okay=False, path_type=Path),
            default=None,
            help="Default: logs/autoscaler-loadtest/run-<timestamp>",
        ),
        click.option(
            "--snapshot",
            type=click.Path(dir_okay=False, path_type=Path),
            default=None,
            help="Controller sqlite snapshot. Default: /tmp/iris-marin.sqlite3",
        ),
        click.option("--latency-seconds", type=float, default=45.0, show_default=True),
        # Probe mix: repeat --probe name:hz to override the scenario's default mix.
        click.option(
            "--probe",
            "probe_tokens",
            type=str,
            multiple=True,
            help="Override scenario probe mix (repeatable). Format: name:hz (e.g. list_jobs:1)",
        ),
        click.option(
            "--log-level",
            type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
            default="INFO",
            show_default=True,
        ),
    ]
    for dec in reversed(decorators):
        func = dec(func)
    return func


def _configure_logging(level: str) -> None:
    import logging

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@click.group()
def main() -> None:
    """Iris autoscaler load-test harness."""


@main.command("probes")
def probes_cmd() -> None:
    """List available probe names for use with --probe name:hz."""
    from iris.loadtest.probes import PROBES

    click.echo("Available probes (use as --probe name:hz):")
    for name in sorted(PROBES):
        click.echo(f"  {name}")


@main.command("list")
def list_cmd() -> None:
    """List scenarios and available toggle flags."""
    click.echo("Scenarios:")
    click.echo("  burst         — burst only (baseline)")
    click.echo("  api-timeouts  — API timeouts only (tpu_create hangs)")
    click.echo("  incident      — combined: burst + bad API + periodic preemption (2026-04-18 repro)")
    click.echo("  fleet-wide    — prod-magnitude: all zones, all sizes, synthetic workers + probes")
    click.echo("  prod-scale    — full-controller: 600 preload, real RPC dashboard, dashboard probe mix")
    click.echo()
    click.echo("Ablation steps (see `iris-loadtest ablation --help` to run the series):")
    from iris.loadtest.ablation import DEFAULT_STEPS

    for idx, (name, flags) in enumerate(DEFAULT_STEPS):
        click.echo(f"  step {idx}: {name}  flags: {' '.join(flags) or '(baseline)'}")


def _run_scenario(
    *,
    scenario: str,
    duration: int,
    preload_workers: int,
    burst_jobs: int,
    cpu_jobs: int,
    cpu_tasks_per_job: int,
    output_dir: Path,
    snapshot: Path,
    latency_seconds: float,
    probe_tokens: tuple[str, ...],
) -> int:
    """Execute a scenario end-to-end. Env vars must already be populated."""
    # Deferred imports: env gates are read at import time.
    from iris.loadtest.harness import HarnessConfig, LoadtestHarness
    from iris.loadtest.probes import parse_probe_specs
    from iris.loadtest.report import RunConfig, write_summary
    from iris.loadtest.scenarios import SCENARIO_RUNNERS

    # Dwell = 5 s so task lifecycle is visible inside the default 5-minute duration.
    dwell_seconds = 5.0

    config = HarnessConfig(
        synthetic_worker_building_seconds=dwell_seconds,
        synthetic_worker_running_seconds=dwell_seconds,
        heartbeat_interval_seconds=5.0,
        use_split_heartbeat=True,
    )

    probe_specs = parse_probe_specs(list(probe_tokens)) if probe_tokens else None

    runner = SCENARIO_RUNNERS[scenario]
    scenario_kwargs: dict = {"duration_s": float(duration)}
    if scenario in ("api-timeouts", "incident", "fleet-wide", "prod-scale"):
        scenario_kwargs["latency_seconds"] = latency_seconds
    if probe_specs is not None:
        scenario_kwargs["probes"] = probe_specs
    if scenario == "prod-scale":
        scenario_kwargs["preload_count"] = preload_workers
        scenario_kwargs["snapshot_db"] = snapshot
        scenario_kwargs["burst_job_count"] = burst_jobs
        scenario_kwargs["cpu_job_count"] = cpu_jobs
        scenario_kwargs["cpu_tasks_per_job"] = cpu_tasks_per_job

    output_dir.mkdir(parents=True, exist_ok=True)

    stop = threading.Event()
    with tempfile.TemporaryDirectory(prefix="loadtest-db-") as td:
        db_dir = Path(td)
        shutil.copy2(snapshot, db_dir / "controller.sqlite3")
        harness = LoadtestHarness(db_dir, config=config)
        harness.start()
        if harness.controller_url is not None:
            click.echo(f"Controller URL: {harness.controller_url}")
        start = time.monotonic()
        try:
            result = runner(harness, output_dir, **scenario_kwargs)
        finally:
            harness.stop()
        elapsed = time.monotonic() - start
        _ = stop  # reserved for signal handling; kept for future wiring

    probe_path = output_dir / f"{result.name}-probe.json"
    probe_mix_str = ",".join(probe_tokens) if probe_tokens else "default"
    run_config = RunConfig(
        scenario=scenario,
        duration_s=float(duration),
        preload_workers=preload_workers if scenario == "prod-scale" else 0,
        burst_jobs=burst_jobs if scenario in ("incident", "fleet-wide", "prod-scale") else 0,
        probe_mix=probe_mix_str,
    )
    summary_path = write_summary(result, config=run_config, out_dir=output_dir, harness=None, probe_path=probe_path)
    click.echo()
    click.echo(summary_path.read_text())
    click.echo(f"metrics JSON: {result.metrics_path}")
    click.echo(f"summary MD  : {summary_path}")
    click.echo(f"wall seconds: {elapsed:.1f}")
    return 0


@main.command("scenario")
@click.argument("name", type=click.Choice(SCENARIOS, case_sensitive=False))
@_common_options
def scenario_cmd(
    name: str,
    duration: int,
    preload_workers: int,
    burst_jobs: int,
    cpu_jobs: int,
    cpu_tasks_per_job: int,
    output_dir: Path | None,
    snapshot: Path | None,
    latency_seconds: float,
    probe_tokens: tuple[str, ...],
    log_level: str,
) -> None:
    """Run one of the canonical scenarios (burst/api-timeouts/incident/fleet-wide/prod-scale)."""
    _configure_logging(log_level)

    # Deferred import: DEFAULT_SNAPSHOT_PATH is safe to import without gates.
    from iris.loadtest.configs import DEFAULT_SNAPSHOT_PATH

    snap = snapshot or DEFAULT_SNAPSHOT_PATH
    if not snap.exists():
        raise click.ClickException(f"Snapshot not found: {snap}")

    out = output_dir or _default_output_dir()
    rc = _run_scenario(
        scenario=name.lower(),
        duration=duration,
        preload_workers=preload_workers,
        burst_jobs=burst_jobs,
        cpu_jobs=cpu_jobs,
        cpu_tasks_per_job=cpu_tasks_per_job,
        output_dir=out,
        snapshot=snap,
        latency_seconds=latency_seconds,
        probe_tokens=probe_tokens,
    )
    sys.exit(rc)


@main.command("ablation")
@click.option("--preload-workers", type=int, default=100, show_default=True)
@click.option("--duration", type=int, default=300, show_default=True)
@click.option("--burst-jobs", type=int, default=100, show_default=True)
@click.option("--cpu-jobs", type=int, default=10, show_default=True)
@click.option("--cpu-tasks-per-job", type=int, default=100, show_default=True)
@click.option(
    "--step-timeout-seconds",
    type=int,
    default=540,
    show_default=True,
    help="Hard wall-clock cap per step; on expiry the child pgroup is SIGKILLed.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Default: logs/autoscaler-loadtest/run-<timestamp>",
)
@click.option(
    "--snapshot",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Controller sqlite snapshot (passed to each scenario subprocess).",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    show_default=True,
)
def ablation_cmd(
    preload_workers: int,
    duration: int,
    burst_jobs: int,
    cpu_jobs: int,
    cpu_tasks_per_job: int,
    step_timeout_seconds: int,
    output_dir: Path | None,
    snapshot: Path | None,
    log_level: str,
) -> None:
    """Run the serial toggle ablation against prod-scale.

    Each step spawns a fresh ``iris-loadtest scenario prod-scale`` subprocess
    (so import-time env gates are reread cleanly), with cleanup between
    steps. After the series a combined ``REPORT.md`` is written comparing
    writer/dashboard latencies, per-RPC batch wall-clock, and Slow-log
    counts across steps.
    """
    _configure_logging(log_level)

    from iris.loadtest.ablation import DEFAULT_STEPS, run_series, write_report

    out = (output_dir or _default_output_dir()).resolve()
    common: list[str] = []
    if snapshot is not None:
        common += ["--snapshot", str(snapshot)]

    results = run_series(
        steps=DEFAULT_STEPS,
        scenario="prod-scale",
        preload_workers=preload_workers,
        duration=duration,
        burst_jobs=burst_jobs,
        cpu_jobs=cpu_jobs,
        cpu_tasks_per_job=cpu_tasks_per_job,
        out_dir=out,
        timeout_seconds=step_timeout_seconds,
        common_flags=common,
    )
    header = (
        f"Serial ablation — {preload_workers} preload workers, {duration}s duration, "
        f"{burst_jobs} burst jobs, {cpu_jobs}x{cpu_tasks_per_job} CPU tasks"
    )
    report_path = write_report(out, results, header)
    click.echo(f"\nCombined report: {report_path}")


if __name__ == "__main__":
    main()
