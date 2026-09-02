# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.domain_phase_mix.materialize_delphi_tpp40_multiregion_assignment import (
    EXECUTOR_INFO_MARKER,
    LEGACY_EAST5_PARENT,
    _orders_with_marker,
    assign_remaining_rows,
    compact_orders,
    estimated_training_compute,
    freeze_snapshot,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import executor_status_succeeded


def test_assignment_preserves_completed_and_resumable_east5_rows() -> None:
    assignments = assign_remaining_rows(
        completed_orders={0, 1, 2},
        phase0_orders={0, 1, 2, 7, 9},
        expected_runs=12,
    )

    assert assignments["completed"] == (0, 1, 2)
    assert assignments["resumable_east5"] == (7, 9)
    assert set(assignments["completed"]).isdisjoint(assignments["east5"])
    assert set(assignments["completed"]).isdisjoint(assignments["europe"])
    assert set(assignments["east5"]).isdisjoint(assignments["europe"])
    assert set(assignments["completed"] + assignments["east5"] + assignments["europe"]) == set(range(12))
    assert abs(len(assignments["east5"]) - len(assignments["europe"])) <= 1


def test_compact_orders_round_trips_launcher_ranges() -> None:
    assert compact_orders((0, 1, 2, 5, 8, 9)) == "0-2,5,8-9"


def test_assignment_balances_panel_source_strata() -> None:
    strata = {order: "qsplit" if order < 8 else "deletion" for order in range(12)}

    assignments = assign_remaining_rows(
        completed_orders={0},
        phase0_orders={0, 1},
        strata_by_order=strata,
        expected_runs=12,
    )

    for stratum in set(strata.values()):
        east5_count = sum(strata[order] == stratum for order in assignments["east5"])
        europe_count = sum(strata[order] == stratum for order in assignments["europe"])
        assert abs(east5_count - europe_count) <= 1


def test_executor_status_uses_last_nonempty_line() -> None:
    assert executor_status_succeeded("RUNNING\nSUCCESS\n")
    assert executor_status_succeeded('RUNNING\n{"status":"SUCCESS"}\n')
    assert not executor_status_succeeded("SUCCESS\nFAILED\n")
    assert executor_status_succeeded('{"status":"SUCCESS"}\n{"message":"cleanup complete"}\n')
    assert not executor_status_succeeded('{"message":"SUCCESS"}\n')


def test_assignment_reports_remaining_training_compute() -> None:
    assignments = {
        "completed": (0,),
        "east5": (1, 2),
        "europe": (3, 4),
        "resumable_east5": (1,),
    }

    compute = estimated_training_compute(assignments)

    assert compute["east5"]["fresh_rows"] == 1
    assert compute["east5"]["resumable_rows"] == 1
    assert 1 < compute["east5"]["estimated_full_run_equivalents"] < 2
    assert compute["europe"]["estimated_full_run_equivalents"] == 2


def test_executor_info_marks_europe_namespace_as_used(tmp_path) -> None:
    run_root = tmp_path / "fit_007_example"
    run_root.mkdir()
    (run_root / EXECUTOR_INFO_MARKER).write_text("{}")

    assert _orders_with_marker(str(tmp_path), EXECUTOR_INFO_MARKER) == {7}


def test_assignment_freeze_requires_the_named_legacy_parent_to_be_terminal() -> None:
    snapshot = freeze_snapshot(
        legacy_parent_job=LEGACY_EAST5_PARENT,
        legacy_parent_state="killed",
        observed_at_utc="2026-08-31T05:00:00Z",
    )

    assert snapshot["legacy_parent_observed_at_utc"] == "2026-08-31T05:00:00+00:00"
    with pytest.raises(ValueError, match="not terminal"):
        freeze_snapshot(
            legacy_parent_job=LEGACY_EAST5_PARENT,
            legacy_parent_state="running",
            observed_at_utc="2026-08-31T05:00:00Z",
        )
    with pytest.raises(ValueError, match="Expected legacy parent"):
        freeze_snapshot(
            legacy_parent_job="/calvinxu/wrong-parent",
            legacy_parent_state="killed",
            observed_at_utc="2026-08-31T05:00:00Z",
        )
