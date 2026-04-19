# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage-3 scenario runners.

Scenario runners — :func:`run_burst`, :func:`run_api_timeouts`,
:func:`run_incident`, :func:`run_fleet_wide`, :func:`run_prod_scale` — each take
an already-started :class:`LoadtestHarness` and an output directory, run the
scenario for a bounded wall time, write the time-series JSON + human summary,
and return a :class:`ScenarioResult`.

Each scenario:
  1. Resets scale-group state for the target pattern (fresh backoff).
  2. Starts metrics collection (sampler + probe threads).
  3. Runs stimuli on a timed loop; the harness's own tick thread drives
     the autoscaler.
  4. Emits ``<scenario>-metrics.json`` and ``<scenario>-summary.md``.

Probe read-pressure is expressed as a list of :class:`ProbeSpec` — each spec
spawns a dedicated RPC-client thread at the specified Hz. The runner
keys latency samples by ``"<name>@<hz>hz"`` so the same RPC polled at two
different rates is distinguishable in the report. Defaults per scenario live
in :mod:`iris.loadtest.probes`; callers can override with an explicit list.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from iris.cluster.controller.controller import compute_demand_entries

from iris.loadtest.configs import DEFAULT_SNAPSHOT_PATH
from iris.loadtest.harness import LoadtestHarness
from iris.loadtest.metrics import ScenarioMetrics
from iris.loadtest.probes import (
    API_TIMEOUTS_PROBES,
    BURST_PROBES,
    ClusterState,
    FLEET_WIDE_PROBES,
    INCIDENT_PROBES,
    PROD_SCALE_PROBES,
    ProbeResult,
    ProbeSpec,
    load_live_pending_running_job_ids,
    run_probes,
)
from iris.loadtest.stimuli import (
    bad_tpu_api,
    preempt_workers,
    submit_burst,
    submit_cpu_burst,
)
from iris.cluster.types import get_tpu_topology

logger = logging.getLogger(__name__)


# Group patterns. Stage-2 observed that the autoscaler routes size-8
# v6e-preemptible demand to us-east5-b first (lowest tier). To concentrate
# failure pressure we target every zone that appears in the v6e-preemptible
# pools so routing can't duck to a healthy zone.
V6E_PREEMPTIBLE_SLICE_PATTERN = r"tpu-v6e-preemptible-(4|8|16)-"
V6E_PREEMPTIBLE_GROUP_PATTERN_LIKE = "tpu_v6e-preemptible_%"


@dataclass
class ScenarioResult:
    name: str
    duration_s: float
    peak_active_scale_up_threads: int
    total_create_attempts: int
    total_create_failures: int
    writer_lock_p95_ms: float
    writer_lock_p99_ms: float
    dashboard_query_p95_ms: float
    dashboard_query_p99_ms: float
    rss_delta_bytes: int
    metrics_path: Path
    summary_path: Path


def _sleep_until(deadline: float, stop: threading.Event) -> None:
    remaining = deadline - time.monotonic()
    if remaining > 0:
        stop.wait(remaining)


def _submit_burst_over(
    harness: LoadtestHarness,
    *,
    job_count: int,
    spread_seconds: float,
    tpu_kind: str,
    size: int,
    user: str,
    seed: int,
    stop: threading.Event,
) -> None:
    """Submit ``job_count`` jobs spread uniformly over ``spread_seconds``."""
    if job_count <= 0:
        return
    batch = max(1, job_count // 20)
    start = time.monotonic()
    submitted = 0
    i = 0
    while submitted < job_count and not stop.is_set():
        target_done = min(job_count, int((time.monotonic() - start) / max(spread_seconds, 1e-6) * job_count) + batch)
        to_submit = max(0, min(batch, target_done - submitted, job_count - submitted))
        if to_submit:
            submit_burst(
                harness,
                user=user,
                job_count=to_submit,
                tpu_kind=tpu_kind,
                size=size,
                seed=seed + i,
            )
            submitted += to_submit
            i += 1
        if stop.wait(min(0.25, spread_seconds / max(job_count / batch, 1.0))):
            return


def _finalize(
    metrics: ScenarioMetrics,
    *,
    name: str,
    duration_s: float,
    out_dir: Path,
) -> ScenarioResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"{name}-metrics.json"
    summary_path = out_dir / f"{name}-summary.md"

    metrics.dump_json(metrics_path)
    summary_md = metrics.summary_markdown(name, duration_s)
    summary_path.write_text(summary_md)

    data = json.loads(metrics_path.read_text())
    lock_p = data["writer_lock_hold_ms"]["percentiles"]
    query_p = data["dashboard_query_ms"]["percentiles"]
    samples = data["samples"]
    peak_threads = max((s["active_scale_up_threads"] for s in samples), default=0)
    total_attempts = samples[-1]["create_attempts"] if samples else 0
    total_failures = samples[-1]["create_failures"] if samples else 0
    rss_delta = metrics.rss_delta_bytes()

    return ScenarioResult(
        name=name,
        duration_s=duration_s,
        peak_active_scale_up_threads=peak_threads,
        total_create_attempts=total_attempts,
        total_create_failures=total_failures,
        writer_lock_p95_ms=lock_p["p95"],
        writer_lock_p99_ms=lock_p["p99"],
        dashboard_query_p95_ms=query_p["p95"],
        dashboard_query_p99_ms=query_p["p99"],
        rss_delta_bytes=rss_delta,
        metrics_path=metrics_path,
        summary_path=summary_path,
    )


def _write_probe_json(result: ProbeResult, path: Path, *, scenario: str) -> None:
    data = {key: result.percentiles(key) for key in result.latencies_ms.keys()}
    path.write_text(json.dumps(data, indent=2))
    logger.info("%s probe results: %s", scenario, data)


def _build_cluster_state(harness: LoadtestHarness, *, seed: int = 0) -> ClusterState:
    """Snapshot pending/running job ids from the harness DB for probe consumption.

    Runs after preload/seed burst so the probes see live job ids the
    controller knows about. Requires the harness to expose a Connect/RPC
    URL (full-controller mode).
    """
    url = harness.controller_url
    if url is None:
        raise RuntimeError("probes require a full Controller; set HarnessConfig(enable_full_controller=True)")
    pending, running = load_live_pending_running_job_ids(harness.db)
    return ClusterState(
        controller_url=url,
        pending_job_ids=pending,
        running_job_ids=running,
        rng_seed=seed,
    )


def _run_probes_thread(
    harness: LoadtestHarness,
    specs: list[ProbeSpec],
    *,
    duration_s: float,
    stop: threading.Event,
    holder: dict[str, ProbeResult],
    seed: int,
) -> None:
    """Build the cluster state and run probes, stashing the result in ``holder``."""
    state = _build_cluster_state(harness, seed=seed)
    holder["result"] = run_probes(state, specs, duration_seconds=duration_s, stop=stop)


# ---------------------------------------------------------------------------
# burst: burst only
# ---------------------------------------------------------------------------


def run_burst(
    harness: LoadtestHarness,
    out_dir: Path,
    *,
    duration_s: float = 120.0,
    job_count: int = 240,
    burst_spread_s: float = 60.0,
    probes: list[ProbeSpec] | None = None,
) -> ScenarioResult:
    """Submit a large v6e-preemptible-8 burst; no API failures."""
    harness.reset_scale_group_state(name_pattern=V6E_PREEMPTIBLE_GROUP_PATTERN_LIKE)
    harness.pause_ticks()

    specs = probes if probes is not None else BURST_PROBES

    metrics = ScenarioMetrics(harness)
    metrics.start()
    stop = threading.Event()
    start = time.monotonic()
    deadline = start + duration_s
    probe_result_holder: dict[str, ProbeResult] = {}

    try:
        burst_deadline = start + min(burst_spread_s, duration_s)
        submitter = threading.Thread(
            target=_submit_burst_over,
            name="burst-burst",
            kwargs=dict(
                harness=harness,
                job_count=job_count,
                spread_seconds=burst_spread_s,
                tpu_kind="v6e-preemptible",
                size=8,
                user="loadtest-burst",
                seed=1000,
                stop=stop,
            ),
            daemon=True,
        )
        submitter.start()

        pump = threading.Thread(
            target=_demand_pump,
            name="burst-demand-pump",
            kwargs=dict(harness=harness, stop=stop, deadline=deadline),
            daemon=True,
        )
        pump.start()

        probe_runner = _maybe_start_probes(
            harness, specs, duration_s=duration_s, stop=stop, holder=probe_result_holder, seed=1001
        )

        _sleep_until(deadline, stop)
        stop.set()
        submitter.join(timeout=5.0)
        _ = burst_deadline  # documentation only
        pump.join(timeout=5.0)
        if probe_runner is not None:
            probe_runner.join(timeout=10.0)
    finally:
        metrics.stop()

    result = _finalize(metrics, name="burst", duration_s=duration_s, out_dir=out_dir)
    probe_result = probe_result_holder.get("result")
    if probe_result is not None:
        _write_probe_json(probe_result, out_dir / "burst-probe.json", scenario="burst")
    return result


def _maybe_start_probes(
    harness: LoadtestHarness,
    specs: list[ProbeSpec],
    *,
    duration_s: float,
    stop: threading.Event,
    holder: dict[str, ProbeResult],
    seed: int,
) -> threading.Thread | None:
    """Start a probe thread if the harness has a Controller URL and specs is non-empty.

    Scenarios that boot only the tick-only harness (no full Controller) skip
    probes entirely — there is no RPC endpoint to probe.
    """
    if not specs:
        return None
    if harness.controller_url is None:
        logger.info("probes disabled: harness has no Controller URL (tick-only mode)")
        return None
    t = threading.Thread(
        target=_run_probes_thread,
        name=f"probes-{seed}",
        kwargs=dict(harness=harness, specs=specs, duration_s=duration_s, stop=stop, holder=holder, seed=seed),
        daemon=True,
    )
    t.start()
    return t


def _demand_pump(harness: LoadtestHarness, stop: threading.Event, deadline: float) -> None:
    """Drive the autoscaler with *real* demand on a 1 Hz cadence.

    The harness's own ``_run_tick_loop`` calls ``run_once(demand_entries=[])``,
    which means the autoscaler has nothing to scale up even if jobs are
    submitted. In full-controller mode the Controller's own loop already
    computes real demand each cycle, so this becomes a no-op.
    """
    if harness.controller_url is not None:
        stop.wait(max(0.0, deadline - time.monotonic()))
        return
    while not stop.is_set() and time.monotonic() < deadline:
        try:
            demand = compute_demand_entries(harness.db)
            harness.tick(demand_entries=demand)
        except Exception:
            logger.exception("demand pump tick failed")
        if stop.wait(1.0):
            return


# ---------------------------------------------------------------------------
# api-timeouts: API timeouts only
# ---------------------------------------------------------------------------


def run_api_timeouts(
    harness: LoadtestHarness,
    out_dir: Path,
    *,
    duration_s: float = 360.0,
    latency_seconds: float = 60.0,
    probes: list[ProbeSpec] | None = None,
) -> ScenarioResult:
    """No new submissions; tpu_create timeouts for the whole duration."""
    harness.reset_scale_group_state(name_pattern=V6E_PREEMPTIBLE_GROUP_PATTERN_LIKE)
    harness.pause_ticks()

    specs = probes if probes is not None else API_TIMEOUTS_PROBES

    bad_tpu_api(
        harness,
        group_pattern=V6E_PREEMPTIBLE_SLICE_PATTERN,
        failure_mode="timeout",
        duration_seconds=duration_s,
        latency_seconds=latency_seconds,
    )

    submit_burst(
        harness,
        user="loadtest-api-timeouts-seed",
        job_count=24,
        tpu_kind="v6e-preemptible",
        size=8,
        seed=2000,
    )

    metrics = ScenarioMetrics(harness)
    metrics.start()
    stop = threading.Event()
    start = time.monotonic()
    deadline = start + duration_s
    probe_result_holder: dict[str, ProbeResult] = {}

    try:
        pump = threading.Thread(
            target=_demand_pump,
            name="api-timeouts-demand-pump",
            kwargs=dict(harness=harness, stop=stop, deadline=deadline),
            daemon=True,
        )
        pump.start()
        probe_runner = _maybe_start_probes(
            harness, specs, duration_s=duration_s, stop=stop, holder=probe_result_holder, seed=2001
        )
        _sleep_until(deadline, stop)
        stop.set()
        pump.join(timeout=5.0)
        if probe_runner is not None:
            probe_runner.join(timeout=10.0)
    finally:
        metrics.stop()

    result = _finalize(metrics, name="api-timeouts", duration_s=duration_s, out_dir=out_dir)
    probe_result = probe_result_holder.get("result")
    if probe_result is not None:
        _write_probe_json(probe_result, out_dir / "api-timeouts-probe.json", scenario="api-timeouts")
    return result


# ---------------------------------------------------------------------------
# incident: combined (prod-like)
# ---------------------------------------------------------------------------


def run_incident(
    harness: LoadtestHarness,
    out_dir: Path,
    *,
    duration_s: float = 600.0,
    burst_job_count: int = 240,
    burst_spread_s: float = 60.0,
    latency_seconds: float = 60.0,
    preempt_fraction: float = 0.1,
    preempt_interval_s: float = 300.0,
    probes: list[ProbeSpec] | None = None,
) -> ScenarioResult:
    """Burst + bad API + periodic preemption."""
    harness.reset_scale_group_state(name_pattern=V6E_PREEMPTIBLE_GROUP_PATTERN_LIKE)
    harness.pause_ticks()

    specs = probes if probes is not None else INCIDENT_PROBES

    bad_tpu_api(
        harness,
        group_pattern=V6E_PREEMPTIBLE_SLICE_PATTERN,
        failure_mode="timeout",
        duration_seconds=duration_s,
        latency_seconds=latency_seconds,
    )

    metrics = ScenarioMetrics(harness)
    metrics.start()
    stop = threading.Event()
    start = time.monotonic()
    deadline = start + duration_s
    probe_result_holder: dict[str, ProbeResult] = {}

    try:
        submitter = threading.Thread(
            target=_submit_burst_over,
            name="incident-burst",
            kwargs=dict(
                harness=harness,
                job_count=burst_job_count,
                spread_seconds=burst_spread_s,
                tpu_kind="v6e-preemptible",
                size=8,
                user="loadtest-incident-burst",
                seed=3000,
                stop=stop,
            ),
            daemon=True,
        )
        submitter.start()

        pump = threading.Thread(
            target=_demand_pump,
            name="incident-demand-pump",
            kwargs=dict(harness=harness, stop=stop, deadline=deadline),
            daemon=True,
        )
        pump.start()

        preempter = threading.Thread(
            target=_preempt_loop,
            name="incident-preempter",
            kwargs=dict(
                harness=harness,
                fraction=preempt_fraction,
                interval_s=preempt_interval_s,
                stop=stop,
                deadline=deadline,
            ),
            daemon=True,
        )
        preempter.start()

        probe_runner = _maybe_start_probes(
            harness, specs, duration_s=duration_s, stop=stop, holder=probe_result_holder, seed=3001
        )

        _sleep_until(deadline, stop)
        stop.set()
        submitter.join(timeout=5.0)
        pump.join(timeout=5.0)
        preempter.join(timeout=5.0)
        if probe_runner is not None:
            probe_runner.join(timeout=10.0)
    finally:
        metrics.stop()

    result = _finalize(metrics, name="incident", duration_s=duration_s, out_dir=out_dir)
    probe_result = probe_result_holder.get("result")
    if probe_result is not None:
        _write_probe_json(probe_result, out_dir / "incident-probe.json", scenario="incident")
    return result


def _preempt_loop(
    harness: LoadtestHarness,
    *,
    fraction: float,
    interval_s: float,
    stop: threading.Event,
    deadline: float,
) -> None:
    next_fire = time.monotonic() + interval_s
    while not stop.is_set() and time.monotonic() < deadline:
        if time.monotonic() >= next_fire:
            try:
                preempt_workers(
                    harness,
                    group_pattern=r"tpu_v6e-preemptible_(4|8|16)-",
                    fraction=fraction,
                    seed=int(next_fire),
                )
            except Exception:
                logger.exception("preempt loop failed")
            next_fire = time.monotonic() + interval_s
        if stop.wait(1.0):
            return


# ---------------------------------------------------------------------------
# fleet-wide: prod-magnitude (all zones, all sizes, synthetic workers + probes)
# ---------------------------------------------------------------------------


V6E_PREEMPTIBLE_SIZES = [4, 8, 16, 32, 64]


def run_fleet_wide(
    harness: LoadtestHarness,
    out_dir: Path,
    *,
    duration_s: float = 120.0,
    burst_job_count: int = 500,
    latency_seconds: float = 60.0,
    preempt_fraction: float = 0.1,
    preempt_interval_s: float = 60.0,
    probes: list[ProbeSpec] | None = None,
) -> ScenarioResult:
    """Prod-magnitude repro: all zones, all sizes, synthetic workers + probes."""
    harness.reset_scale_group_state(name_pattern=V6E_PREEMPTIBLE_GROUP_PATTERN_LIKE)
    harness.pause_ticks()

    specs = probes if probes is not None else FLEET_WIDE_PROBES

    bad_tpu_api(
        harness,
        group_pattern=V6E_PREEMPTIBLE_SLICE_PATTERN,
        failure_mode="timeout",
        duration_seconds=duration_s,
        latency_seconds=latency_seconds,
    )

    submit_burst(
        harness,
        user="loadtest-fleet-wide-seed",
        job_count=burst_job_count,
        tpu_kind="v6e-preemptible",
        size=V6E_PREEMPTIBLE_SIZES,
        seed=4000,
    )

    metrics = ScenarioMetrics(harness)
    metrics.start()
    stop = threading.Event()
    start = time.monotonic()
    deadline = start + duration_s
    probe_result_holder: dict[str, ProbeResult] = {}

    try:
        pump = threading.Thread(
            target=_demand_pump,
            name="fleet-wide-demand-pump",
            kwargs=dict(harness=harness, stop=stop, deadline=deadline),
            daemon=True,
        )
        pump.start()

        preempter = threading.Thread(
            target=_preempt_loop,
            name="fleet-wide-preempter",
            kwargs=dict(
                harness=harness,
                fraction=preempt_fraction,
                interval_s=preempt_interval_s,
                stop=stop,
                deadline=deadline,
            ),
            daemon=True,
        )
        preempter.start()

        probe_runner = _maybe_start_probes(
            harness, specs, duration_s=duration_s, stop=stop, holder=probe_result_holder, seed=4001
        )

        _sleep_until(deadline, stop)
        stop.set()
        pump.join(timeout=5.0)
        preempter.join(timeout=5.0)
        if probe_runner is not None:
            probe_runner.join(timeout=10.0)
    finally:
        metrics.stop()

    result = _finalize(metrics, name="fleet-wide", duration_s=duration_s, out_dir=out_dir)
    probe_result = probe_result_holder.get("result")
    if probe_result is not None:
        _write_probe_json(probe_result, out_dir / "fleet-wide-probe.json", scenario="fleet-wide")
    return result


# ---------------------------------------------------------------------------
# prod-scale: prod-scale fleet + scale-up storm + preemption churn (Stage 7)
# ---------------------------------------------------------------------------


SCALE_STORM_SLICE_PATTERN = r"tpu-(v6e|v5p)-preemptible-(4|8|16|32|64)-"
SCALE_STORM_GROUP_PATTERN_LIKE = "tpu_%preemptible%"

# Prod-mix sizing — weights loosely follow michaelryan's extract-v2 shape: v6e
# dominates with most mass at small-to-mid slice sizes; v5p is a long tail.
# The fractions are (tpu_kind, size, weight). Weights sum-normalised to the
# requested TPU job total at call time.
PROD_MIX_TPU_WEIGHTS: tuple[tuple[str, int, float], ...] = (
    ("v6e-preemptible", 4, 0.10),
    ("v6e-preemptible", 8, 0.50),
    ("v6e-preemptible", 16, 0.20),
    ("v6e-preemptible", 32, 0.05),
    ("v5p-preemptible", 8, 0.10),
    ("v5p-preemptible", 16, 0.05),
)


def _split_counts(total: int, weights: tuple[float, ...]) -> list[int]:
    """Distribute ``total`` across entries using largest-remainder rounding."""
    scaled = [total * w for w in weights]
    floors = [int(x) for x in scaled]
    remaining = total - sum(floors)
    order = sorted(range(len(scaled)), key=lambda i: scaled[i] - floors[i], reverse=True)
    for j in range(remaining):
        floors[order[j % len(order)]] += 1
    return floors


def run_prod_scale(
    harness: LoadtestHarness,
    out_dir: Path,
    *,
    duration_s: float = 180.0,
    preload_count: int = 600,
    burst_job_count: int = 1000,
    cpu_job_count: int = 10,
    cpu_tasks_per_job: int = 500,
    latency_seconds: float = 60.0,
    preempt_fraction: float = 0.1,
    preempt_interval_s: float = 30.0,
    snapshot_db: Path | None = None,
    probes: list[ProbeSpec] | None = None,
) -> ScenarioResult:
    """Prod-scale fleet + realistic TPU/CPU mix + preemption churn.

    Workload shape (defaults match prod magnitude):
      - ``burst_job_count`` TPU jobs (default 1000) distributed across v6e and
        v5p preemptible variants via ``PROD_MIX_TPU_WEIGHTS``. Multi-host
        variants (e.g. v6e-32 with vm_count=8) submit one task per VM.
      - ``cpu_job_count`` x ``cpu_tasks_per_job`` CPU tasks (default 10 x 500
        = 5000) that pack onto the same TPU hosts via leftover CPU+RAM.
    """
    snapshot = snapshot_db or DEFAULT_SNAPSHOT_PATH
    preloaded = harness.preload_workers(count=preload_count, snapshot_db=snapshot)
    logger.info("prod-scale: pre-loaded %d synthetic workers from %s", preloaded, snapshot)

    harness.reset_scale_group_state(name_pattern=SCALE_STORM_GROUP_PATTERN_LIKE)
    harness.pause_ticks()

    specs = probes if probes is not None else PROD_SCALE_PROBES

    bad_tpu_api(
        harness,
        group_pattern=SCALE_STORM_SLICE_PATTERN,
        failure_mode="timeout",
        duration_seconds=duration_s,
        latency_seconds=latency_seconds,
    )

    # Split the TPU demand across the prod-mix variants.
    weights = tuple(w for _, _, w in PROD_MIX_TPU_WEIGHTS)
    per_bucket = _split_counts(burst_job_count, weights)
    total_tpu_tasks = 0
    for seed_offset, ((tpu_kind, size, _w), count) in enumerate(zip(PROD_MIX_TPU_WEIGHTS, per_bucket, strict=True)):
        if count <= 0:
            continue
        submit_burst(
            harness,
            user=f"loadtest-prod-{tpu_kind}-{size}",
            job_count=count,
            tpu_kind=tpu_kind,
            size=size,
            seed=5000 + seed_offset * 1000,
        )
        # Task count per job is vm_count of the variant (multi-host slices).
        variant = f"{'v6e' if tpu_kind.startswith('v6e') else 'v5p'}-{size}"
        try:
            replicas = get_tpu_topology(variant).vm_count
        except ValueError:
            replicas = 1
        total_tpu_tasks += count * replicas

    cpu_ids = submit_cpu_burst(
        harness,
        user="loadtest-prod-cpu",
        job_count=cpu_job_count,
        tasks_per_job=cpu_tasks_per_job,
        seed=9000,
    )
    total_cpu_tasks = len(cpu_ids) * cpu_tasks_per_job
    logger.info(
        "prod-scale workload: %d TPU jobs (%d tasks) + %d CPU jobs (%d tasks)",
        burst_job_count,
        total_tpu_tasks,
        cpu_job_count,
        total_cpu_tasks,
    )

    metrics = ScenarioMetrics(harness)
    metrics.start()
    stop = threading.Event()
    start = time.monotonic()
    deadline = start + duration_s
    probe_result_holder: dict[str, ProbeResult] = {}

    try:
        pump = threading.Thread(
            target=_demand_pump,
            name="prod-scale-demand-pump",
            kwargs=dict(harness=harness, stop=stop, deadline=deadline),
            daemon=True,
        )
        pump.start()

        preempter = threading.Thread(
            target=_preempt_loop,
            name="prod-scale-preempter",
            kwargs=dict(
                harness=harness,
                fraction=preempt_fraction,
                interval_s=preempt_interval_s,
                stop=stop,
                deadline=deadline,
            ),
            daemon=True,
        )
        preempter.start()

        probe_runner = _maybe_start_probes(
            harness, specs, duration_s=duration_s, stop=stop, holder=probe_result_holder, seed=5001
        )

        _sleep_until(deadline, stop)
        stop.set()
        pump.join(timeout=5.0)
        preempter.join(timeout=5.0)
        if probe_runner is not None:
            probe_runner.join(timeout=10.0)
    finally:
        metrics.stop()

    result = _finalize(metrics, name="prod-scale", duration_s=duration_s, out_dir=out_dir)
    probe_result = probe_result_holder.get("result")
    if probe_result is not None:
        _write_probe_json(probe_result, out_dir / "prod-scale-probe.json", scenario="prod-scale")
    return result


SCENARIO_RUNNERS = {
    "burst": run_burst,
    "api-timeouts": run_api_timeouts,
    "incident": run_incident,
    "fleet-wide": run_fleet_wide,
    "prod-scale": run_prod_scale,
}
