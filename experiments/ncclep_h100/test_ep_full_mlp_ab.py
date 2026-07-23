# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

import pytest

from experiments.ncclep_h100.ep_full_mlp_ab import (
    ARM_RING,
    ARM_TE,
    PROMOTION_SPEEDUP,
    RECV_CAPACITY_PER_RANK,
    TimingSummary,
    balanced_route_table,
    build_summary,
    count_stablehlo_operations,
    routing_capacity_report,
    summarize_times,
    timing_orders,
)

_SCRIPT = Path(__file__).with_name("run_full_mlp_ab.sh")


def _summary(*, ring_ms: float, te_ms: float, parity_passed: bool = True) -> dict:
    timings = {
        ARM_RING: TimingSummary(3, ring_ms, ring_ms, ring_ms),
        ARM_TE: TimingSummary(3, te_ms, te_ms, te_ms),
    }
    finite = {
        ARM_RING: {"output": True, "gradients": True},
        ARM_TE: {"output": True, "gradients": True},
    }
    return build_summary(
        timings=timings,
        parity={"passed": parity_passed},
        finite=finite,
        runtime={},
        routing=routing_capacity_report(balanced_route_table()),
        stablehlo={},
    )


def test_balanced_routes_fit_identical_ep8_capacity_without_drops() -> None:
    report = routing_capacity_report(balanced_route_table())

    assert report["destination_counts"] == [65_536] * 8
    assert report["aligned_destination_counts"] == [65_536] * 8
    assert report["capacity_padding_rows_per_rank"] == [RECV_CAPACITY_PER_RANK - 65_536] * 8
    assert report["validated_before_dispatch"] is True


def test_interleaved_schedule_balances_first_arm_bias() -> None:
    orders = timing_orders(6)

    assert orders == [
        (ARM_RING, ARM_TE),
        (ARM_TE, ARM_RING),
        (ARM_RING, ARM_TE),
        (ARM_TE, ARM_RING),
        (ARM_RING, ARM_TE),
        (ARM_TE, ARM_RING),
    ]


def test_timing_summary_reports_requested_percentiles() -> None:
    timing = summarize_times([0.001, 0.002, 0.003])

    assert timing.median_ms == pytest.approx(2.0)
    assert timing.p10_ms == pytest.approx(1.2)
    assert timing.p90_ms == pytest.approx(2.8)


def test_promotion_requires_ten_percent_speedup_and_valid_numerics() -> None:
    passing = _summary(ring_ms=11.0, te_ms=10.0)
    parity_failure = _summary(ring_ms=11.0, te_ms=10.0, parity_passed=False)
    too_slow = _summary(ring_ms=10.9, te_ms=10.0)

    assert passing["comparison"]["ring_over_te_speedup"] == pytest.approx(PROMOTION_SPEEDUP)
    assert passing["promotion_criterion"]["passed"] is True
    assert passing["status"] == "promote"
    assert parity_failure["promotion_criterion"]["passed"] is False
    assert too_slow["promotion_criterion"]["passed"] is False


def test_stablehlo_report_counts_collectives_and_custom_call_targets() -> None:
    stablehlo = """
      %0 = stablehlo.all_gather %arg0
      %1 = stablehlo.reduce_scatter %0
      %2 = stablehlo.custom_call @foo(%1) {call_target_name = "te_dispatch"}
      %3 = stablehlo.custom_call @bar(%2) {call_target_name = "te_dispatch"}
    """

    report = count_stablehlo_operations(stablehlo)

    assert report["operations"]["all_gather"] == 1
    assert report["operations"]["reduce_scatter"] == 1
    assert report["operations"]["custom_call"] == 2
    assert report["custom_call_targets"] == {"te_dispatch": 2}


def test_launcher_has_valid_bash_and_dry_run_contract() -> None:
    syntax = subprocess.run(["bash", "-n", _SCRIPT], check=False, capture_output=True, text=True)
    dry_run = subprocess.run(["bash", _SCRIPT, "--dry-run"], check=False, capture_output=True, text=True)

    assert syntax.returncode == 0, syntax.stderr
    assert dry_run.returncode == 0, dry_run.stderr
    assert "8 processes x 1 GPU" in dry_run.stdout
    assert "TE value_and_grad p50 >= 1.10x ring" in dry_run.stdout
