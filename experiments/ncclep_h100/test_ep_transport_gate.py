# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.ncclep_h100.ep_transport_gate import (
    HISTORICAL_RING_FULL_ROUTED_MLP_MEDIAN_MS,
    MATERIAL_HEADROOM_RATIO,
    RECV_CAPACITY_PER_RANK,
    TimingSummary,
    balanced_route_table,
    build_summary,
    remote_wire_bytes,
    summarize_times,
    validate_route_capacity,
)


def test_balanced_route_capacity_matches_fixed_ep8_shape() -> None:
    report = validate_route_capacity(balanced_route_table())

    assert report["destination_counts"] == [65_536] * 8
    assert report["aligned_destination_counts"] == [65_536] * 8
    assert report["headroom_rows"] == RECV_CAPACITY_PER_RANK - 65_536
    assert report["validated_before_dispatch"] is True


def test_route_capacity_rejects_overflow_before_dispatch() -> None:
    routes = np.zeros_like(balanced_route_table())

    with pytest.raises(ValueError, match="exceeds NCCL_EP receive capacity before dispatch"):
        validate_route_capacity(routes)


def test_timing_summary_reports_quantiles_and_effective_wire_rate() -> None:
    timing = summarize_times([0.001, 0.002, 0.003], round_trip_count=1)

    assert timing.median_ms == pytest.approx(2.0)
    assert timing.p10_ms == pytest.approx(1.2)
    assert timing.p90_ms == pytest.approx(2.8)
    assert timing.remote_wire_bytes_per_rank == remote_wire_bytes(1)
    assert timing.effective_wire_gbps == pytest.approx(remote_wire_bytes(1) / 0.002 / 1e9)


def test_summary_labels_historical_ring_value_as_unpaired_bound() -> None:
    threshold_ms = HISTORICAL_RING_FULL_ROUTED_MLP_MEDIAN_MS * MATERIAL_HEADROOM_RATIO
    timing = TimingSummary(
        iterations=2,
        median_ms=threshold_ms,
        p10_ms=threshold_ms,
        p90_ms=threshold_ms,
        remote_wire_bytes_per_rank=1,
        effective_wire_gbps=1.0,
    )
    summary = build_summary(
        timing,
        timing,
        runtime={},
        routing_capacity=validate_route_capacity(balanced_route_table()),
    )

    assert summary["status"] == "pass"
    assert summary["decision_gate"]["comparison_kind"] == "unpaired_historical_hard_sanity_bound"
    assert summary["decision_gate"]["not_apples_to_apples"] is True
