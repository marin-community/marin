# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cluster gauges: the worker-fleet rollup (healthy-only resource totals + the
per-region head count) and the job-state split into in-flight vs terminal-24h
buckets. These exercise the pure ``aggregate_*`` over synthetic inputs, no live
controller."""

import json

import pytest
from cluster import (
    FLEET,
    METRIC_JOB_INFLIGHT,
    METRIC_JOB_TERMINAL,
    METRIC_WORKER_CPU_MILLICORES,
    METRIC_WORKER_HEALTHY,
    METRIC_WORKER_MEMORY_BYTES,
    METRIC_WORKER_TPU_CHIPS,
    StateCount,
    WorkerInfo,
    aggregate_jobs,
    aggregate_workers,
)

GIB = 1024**3


def _find(samples, metric, **labels):
    want = json.dumps(labels, sort_keys=True)
    vals = [s.value for s in samples if s.metric == metric and s.labels == want]
    assert len(vals) == 1, f"{metric} {labels}: {vals}"
    return vals[0]


def _missing(samples, metric, **labels):
    want = json.dumps(labels, sort_keys=True)
    return not any(s.metric == metric and s.labels == want for s in samples)


# ---- workers --------------------------------------------------------------


def test_worker_fleet_totals_sum_healthy_only():
    workers = [
        WorkerInfo(healthy=True, cpu_count=96, memory_bytes=8 * GIB, tpu_chips=4, region="us-east5"),
        WorkerInfo(healthy=True, cpu_count=4, memory_bytes=GIB, tpu_chips=0, region="us-east5"),
        # Unhealthy worker contributes nothing to any total.
        WorkerInfo(healthy=False, cpu_count=96, memory_bytes=8 * GIB, tpu_chips=4, region="us-east5"),
    ]
    samples = aggregate_workers(workers)
    assert _find(samples, METRIC_WORKER_HEALTHY, scope=FLEET) == 2
    assert _find(samples, METRIC_WORKER_CPU_MILLICORES, scope=FLEET) == 100_000  # (96 + 4) * 1000
    assert _find(samples, METRIC_WORKER_MEMORY_BYTES, scope=FLEET) == 9 * GIB
    assert _find(samples, METRIC_WORKER_TPU_CHIPS, scope=FLEET) == 4


def test_worker_per_region_head_count_healthy_only():
    workers = [
        WorkerInfo(healthy=True, cpu_count=1, memory_bytes=0, tpu_chips=0, region="us-east5"),
        WorkerInfo(healthy=True, cpu_count=1, memory_bytes=0, tpu_chips=0, region="us-east5"),
        WorkerInfo(healthy=True, cpu_count=1, memory_bytes=0, tpu_chips=0, region="europe-west4"),
        WorkerInfo(healthy=False, cpu_count=1, memory_bytes=0, tpu_chips=0, region="europe-west4"),
    ]
    samples = aggregate_workers(workers)
    assert _find(samples, METRIC_WORKER_HEALTHY, region="us-east5") == 2
    assert _find(samples, METRIC_WORKER_HEALTHY, region="europe-west4") == 1


def test_worker_empty_fleet_emits_zero_totals_no_regions():
    samples = aggregate_workers([])
    assert _find(samples, METRIC_WORKER_HEALTHY, scope=FLEET) == 0
    assert _find(samples, METRIC_WORKER_CPU_MILLICORES, scope=FLEET) == 0
    # No region series when there are no healthy workers.
    assert _missing(samples, METRIC_WORKER_HEALTHY, region="us-east5")


# ---- jobs -----------------------------------------------------------------


@pytest.fixture
def job_samples():
    # state ids: 3=running, 1=pending, 4=succeeded, 5=failed, 6=killed.
    rows = [
        StateCount(3, 7),  # running   (in-flight)
        StateCount(1, 2),  # pending   (in-flight)
        StateCount(4, 40),  # succeeded (terminal, 24h)
        StateCount(5, 3),  # failed    (terminal, 24h)
        StateCount(6, 1),  # killed    (terminal, 24h)
    ]
    return aggregate_jobs(rows)


def test_job_inflight_split(job_samples):
    assert _find(job_samples, METRIC_JOB_INFLIGHT, state="running") == 7
    assert _find(job_samples, METRIC_JOB_INFLIGHT, state="pending") == 2
    assert _find(job_samples, METRIC_JOB_INFLIGHT, scope=FLEET) == 9


def test_job_terminal_split(job_samples):
    assert _find(job_samples, METRIC_JOB_TERMINAL, state="succeeded") == 40
    assert _find(job_samples, METRIC_JOB_TERMINAL, state="failed") == 3
    assert _find(job_samples, METRIC_JOB_TERMINAL, state="killed") == 1
    assert _find(job_samples, METRIC_JOB_TERMINAL, scope=FLEET) == 44


def test_job_unknown_state_falls_through_as_state_n():
    # An enum value the probe doesn't know about must surface, not vanish.
    samples = aggregate_jobs([StateCount(99, 5)])
    assert _find(samples, METRIC_JOB_TERMINAL, state="state_99") == 5
    assert _find(samples, METRIC_JOB_TERMINAL, scope=FLEET) == 5
    assert _find(samples, METRIC_JOB_INFLIGHT, scope=FLEET) == 0
