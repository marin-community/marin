# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unified ``iris-loadtest`` CLI.

Subcommands:

- ``scenario NAME`` — run one of the canonical scenarios.
- ``ablation``      — sweep toggle combinations against prod-scale.
- ``list``          — list scenarios and toggle flags.

Toggle flags set environment variables read by the controller / autoscaler /
service at import time; all ``iris.loadtest.*`` imports are deferred until
after env vars have been populated.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import click

# NOTE: deliberate lack of iris.loadtest imports at module scope. The
# controller and autoscaler paths read several env vars once at import — we
# set those env vars from CLI flags and only THEN import the harness.

SCENARIOS = ("burst", "api-timeouts", "incident", "fleet-wide", "prod-scale")
ABLATION_STEPS: dict[int, tuple[str, dict[str, str]]] = {
    0: ("baseline", {}),
    1: ("sqlite-tuning", {"IRIS_DB_MMAP_BYTES": "268435456", "IRIS_DB_CACHE_KB": "1048576"}),
    2: ("controller-yield", {"IRIS_CONTROLLER_YIELD": "1"}),
    3: ("job-status-cache", {"IRIS_JOB_STATUS_CACHE_TTL_MS": "1000"}),
}


def _default_output_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(f"logs/autoscaler-loadtest/run-{ts}")


def _apply_fix_env(
    *,
    sqlite_tuning: bool,
    controller_yield: bool,
    job_status_cache: bool,
    cache_ttl_ms: int,
) -> list[str]:
    """Populate os.environ for each enabled toggle; return a label list.

    Must run BEFORE any iris.loadtest import — the controller / autoscaler
    / service read these env vars once at import time.
    """
    labels: list[str] = []
    if sqlite_tuning:
        os.environ["IRIS_DB_MMAP_BYTES"] = "268435456"
        os.environ["IRIS_DB_CACHE_KB"] = "1048576"
        labels.append("sqlite-tuning")
    if controller_yield:
        os.environ["IRIS_CONTROLLER_YIELD"] = "1"
        labels.append("controller-yield")
    if job_status_cache:
        os.environ["IRIS_JOB_STATUS_CACHE_TTL_MS"] = str(cache_ttl_ms)
        labels.append(f"job-status-cache:{cache_ttl_ms}ms")
    return labels


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
        # Toggle flags
        click.option("--sqlite-tuning/--no-sqlite-tuning", default=False),
        click.option(
            "--controller-yield/--no-controller-yield",
            "controller_yield",
            default=False,
            help="Set IRIS_CONTROLLER_YIELD=1 so scheduler + autoscaler loops sleep(0) between phases.",
        ),
        click.option("--job-status-cache/--no-job-status-cache", default=False),
        click.option("--cache-ttl-ms", type=int, default=1000, show_default=True),
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
    click.echo("Toggle flags (cumulative; each sets env vars the controller reads at import time):")
    for step, (name, env) in sorted(ABLATION_STEPS.items()):
        env_str = " ".join(f"{k}={v}" for k, v in env.items()) or "(baseline)"
        click.echo(f"  step {step}: --{name}")
        click.echo(f"    env: {env_str}")


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
    fix_labels: list[str],
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
        fixes=fix_labels,
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
    sqlite_tuning: bool,
    controller_yield: bool,
    job_status_cache: bool,
    cache_ttl_ms: int,
    log_level: str,
) -> None:
    """Run one of the canonical scenarios (burst/api-timeouts/incident/fleet-wide/prod-scale)."""
    _configure_logging(log_level)
    fix_labels = _apply_fix_env(
        sqlite_tuning=sqlite_tuning,
        controller_yield=controller_yield,
        job_status_cache=job_status_cache,
        cache_ttl_ms=cache_ttl_ms,
    )

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
        fix_labels=fix_labels,
    )
    sys.exit(rc)


@main.command("ablation")
@click.option(
    "--steps",
    type=str,
    default="0,1,2,3,4,5",
    show_default=True,
    help="Comma-separated step ids to run (cumulative).",
)
@_common_options
def ablation_cmd(
    steps: str,
    duration: int,
    preload_workers: int,
    burst_jobs: int,
    cpu_jobs: int,
    cpu_tasks_per_job: int,
    output_dir: Path | None,
    snapshot: Path | None,
    latency_seconds: float,
    probe_tokens: tuple[str, ...],
    sqlite_tuning: bool,  # ignored; ablation computes these from --steps
    controller_yield: bool,
    job_status_cache: bool,
    cache_ttl_ms: int,
    log_level: str,
) -> None:
    """Run the sequential ablation across --steps.

    Each step applies its env-gated toggle in addition to every prior step's
    toggles, then launches the prod-scale scenario against a fresh DB copy
    and writes ``step-<N>-<name>-{summary.md,metrics.json}`` under --output-dir.
    """
    _configure_logging(log_level)

    try:
        chosen = sorted({int(s) for s in steps.split(",") if s.strip()})
    except ValueError as exc:
        raise click.ClickException(f"--steps must be comma-separated ints, got {steps!r}") from exc
    unknown = [s for s in chosen if s not in ABLATION_STEPS]
    if unknown:
        raise click.ClickException(f"Unknown ablation steps: {unknown}")

    from iris.loadtest.configs import DEFAULT_SNAPSHOT_PATH

    snap = snapshot or DEFAULT_SNAPSHOT_PATH
    if not snap.exists():
        raise click.ClickException(f"Snapshot not found: {snap}")

    base_out = output_dir or _default_output_dir()
    base_out.mkdir(parents=True, exist_ok=True)

    # Cumulative env: by step N, every env var from steps <= N is set.
    cumulative_env: dict[str, str] = {}
    cumulative_labels: list[str] = []
    for step in sorted(ABLATION_STEPS):
        if step not in chosen:
            # Still accumulate so later steps include prior fixes.
            cumulative_env.update(ABLATION_STEPS[step][1])
            if ABLATION_STEPS[step][1]:
                cumulative_labels.append(ABLATION_STEPS[step][0])
            continue

        name, env = ABLATION_STEPS[step]
        cumulative_env.update(env)
        for k, v in cumulative_env.items():
            os.environ[k] = v
        if env:
            cumulative_labels.append(name)

        step_dir = base_out / f"step-{step}-{name}"
        click.echo(f"=== Ablation step {step}: {name} (cumulative env: {sorted(cumulative_env)}) ===")
        _run_scenario(
            scenario="prod-scale",
            duration=duration,
            preload_workers=preload_workers,
            burst_jobs=burst_jobs,
            cpu_jobs=cpu_jobs,
            cpu_tasks_per_job=cpu_tasks_per_job,
            output_dir=step_dir,
            snapshot=snap,
            latency_seconds=latency_seconds,
            probe_tokens=probe_tokens,
            fix_labels=list(cumulative_labels),
        )


if __name__ == "__main__":
    main()
