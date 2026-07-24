# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.datakit.scripts.dedup_ab_finelog import (
    build_report,
    expected_executions,
)


def _row(root: str, order: int, execution_id: str, cpu: float) -> dict:
    return {
        "root_key": f"{root}/0:0",
        "epoch_ms": order,
        "execution_id": execution_id,
        "stage_name": "stage-0",
        "status": "END",
        "elapsed": 10.0,
        "items": 100,
        "bytes_processed": 1_000,
        "total_shards": 2,
        "cpu_pct_avg": 50.0,
        "cpu_time_total": cpu,
        "mem_bytes_avg": 100,
        "mem_peak_bytes_max": 200,
        "cluster": "test",
    }


def test_expected_executions_distinguishes_cap_and_convergence() -> None:
    expected = expected_executions(
        baseline_root="/baseline",
        baseline_continuation_root="/baseline-continuation",
        treatment_root="/treatment",
        baseline_capped_iterations=2,
        baseline_converged_iterations=4,
        treatment_iterations=1,
    )

    assert [(item.root_job_id, item.phase, item.iteration) for item in expected] == [
        ("/baseline", "minhash", None),
        ("/baseline", "initial_graph", None),
        ("/baseline", "connected_components", 1),
        ("/baseline", "connected_components", 2),
        ("/baseline", "marker_cap", None),
        ("/baseline-continuation", "connected_components", 3),
        ("/baseline-continuation", "connected_components", 4),
        ("/baseline-continuation", "marker_converged", None),
        ("/treatment", "minhash", None),
        ("/treatment", "initial_graph", None),
        ("/treatment", "connected_components", 1),
        ("/treatment", "marker_converged", None),
    ]


def test_report_accounts_for_every_execution_and_compares_identical_minhash_work() -> None:
    expected = expected_executions(
        baseline_root="/baseline",
        baseline_continuation_root="/baseline-continuation",
        treatment_root="/treatment",
        baseline_capped_iterations=1,
        baseline_converged_iterations=2,
        treatment_iterations=1,
    )
    order_by_root: dict[str, int] = {}
    rows = []
    for index, item in enumerate(expected):
        order = order_by_root.get(item.root_job_id, 0) + 1
        order_by_root[item.root_job_id] = order
        cpu = 10.0 if item.phase == "minhash" else 1.0
        rows.append(_row(item.root_job_id, order, f"execution-{index}", cpu))

    report = build_report(rows, expected)

    assert report["scenarios"]["baseline_cap"] == {
        "executions": 4,
        "stages": 4,
        "cpu_time_total": 13.0,
        "mem_peak_bytes_max": 200,
        "non_end_rows": 0,
    }
    assert report["scenarios"]["baseline_converged"] == {
        "executions": 5,
        "stages": 5,
        "cpu_time_total": 14.0,
        "mem_peak_bytes_max": 200,
        "non_end_rows": 0,
    }
    assert report["scenarios"]["treatment"]["cpu_time_total"] == 13.0
    assert report["minhash_comparison"]["cpu_delta_fraction"] == 0.0
    assert report["minhash_comparison"]["items_match"] is True
