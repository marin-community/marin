# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Provisioning rollup: counting outcomes by pool, the stockout/error split,
runtime deaths (preemptions) kept out of the success rate, and latency from READY
rows. These exercise the pure ``aggregate`` over synthetic iris.provisioning rows."""

from __future__ import annotations

import json

import pytest
from provisioning import (
    METRIC_ERROR,
    METRIC_LATENCY_SECONDS,
    METRIC_OUTCOMES,
    METRIC_POOLS_PLACING,
    METRIC_POOLS_STOCKOUT_DEAD,
    METRIC_PREEMPTED,
    METRIC_READY,
    METRIC_STOCKOUT,
    METRIC_SUCCESS_RATIO,
    Row,
    aggregate,
)

# (resource_type, scale_group, zone) prefixes; rows add (outcome, latency_ms).
P1 = ("tpu", "tpu_v6e-preemptible_8-us-east5-b", "us-east5-b")
P2 = ("tpu", "tpu_v5e-serving_8-europe-west4-b", "europe-west4-b")
P1_LABELS = {"resource_type": "tpu", "scale_group": P1[1], "zone": "us-east5-b"}
P2_LABELS = {"resource_type": "tpu", "scale_group": P2[1], "zone": "europe-west4-b"}
FLEET = {"scope": "fleet"}


def _find(samples, metric, **labels):
    want = json.dumps(labels, sort_keys=True)
    vals = [s.value for s in samples if s.metric == metric and s.labels == want]
    assert len(vals) == 1, f"{metric} {labels}: {vals}"
    return vals[0]


@pytest.fixture
def samples():
    rows = [
        Row(*P1, "ready", 300_000),  # success, 300s latency
        Row(*P1, "ready", 100_000),  # success, 100s latency
        Row(*P1, "stockout", 0),  # one stockout in an otherwise-placing pool
        Row(*P1, "preempted", 0),  # ran then preempted — not a create failure
        Row(*P2, "stockout", 0),
        Row(*P2, "error", 0),  # a real fault, not capacity
    ]
    return aggregate(rows, window_hours=3.0)


def test_per_pool_counts(samples):
    assert _find(samples, METRIC_READY, **P1_LABELS) == 2
    assert _find(samples, METRIC_STOCKOUT, **P1_LABELS) == 1
    assert _find(samples, METRIC_PREEMPTED, **P1_LABELS) == 1
    assert _find(samples, METRIC_OUTCOMES, **P1_LABELS) == 3  # ready + stockout + error; preempted excluded

    assert _find(samples, METRIC_READY, **P2_LABELS) == 0
    assert _find(samples, METRIC_STOCKOUT, **P2_LABELS) == 1
    assert _find(samples, METRIC_ERROR, **P2_LABELS) == 1


def test_fleet_rollup(samples):
    assert _find(samples, METRIC_READY, **FLEET) == 2
    assert _find(samples, METRIC_STOCKOUT, **FLEET) == 2
    assert _find(samples, METRIC_ERROR, **FLEET) == 1
    assert _find(samples, METRIC_PREEMPTED, **FLEET) == 1
    assert _find(samples, METRIC_OUTCOMES, **FLEET) == 5
    # success rate excludes runtime deaths: 2 ready / (2 ready + 2 stockout + 1 error)
    assert _find(samples, METRIC_SUCCESS_RATIO, **FLEET) == pytest.approx(2 / 5)
    assert _find(samples, METRIC_POOLS_PLACING, **FLEET) == 1  # P1
    assert _find(samples, METRIC_POOLS_STOCKOUT_DEAD, **FLEET) == 1  # P2


def test_latency_from_ready_rows_only(samples):
    # P1 has two READY rows (100s, 300s); nearest-rank p50 -> 100, p95 -> 300.
    assert _find(samples, METRIC_LATENCY_SECONDS, quantile="p50", **P1_LABELS) == pytest.approx(100.0)
    assert _find(samples, METRIC_LATENCY_SECONDS, quantile="p95", **P1_LABELS) == pytest.approx(300.0)
    # P2 never reached ready -> no latency series for it
    p2_latency = [
        s for s in samples if s.metric == METRIC_LATENCY_SECONDS and json.loads(s.labels).get("scale_group") == P2[1]
    ]
    assert not p2_latency


def test_success_ratio_is_fleet_scoped_only(samples):
    ratios = [s for s in samples if s.metric == METRIC_SUCCESS_RATIO]
    assert ratios and all(json.loads(s.labels) == FLEET for s in ratios)


def test_empty_inputs_emit_fleet_zeros_no_ratio():
    samples = aggregate([], window_hours=3.0)
    assert _find(samples, METRIC_OUTCOMES, **FLEET) == 0
    assert _find(samples, METRIC_POOLS_PLACING, **FLEET) == 0
    assert not [s for s in samples if s.metric == METRIC_SUCCESS_RATIO]
