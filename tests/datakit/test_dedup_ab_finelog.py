# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime
from types import SimpleNamespace

from iris.rpc import job_pb2

import experiments.datakit.scripts.dedup_ab_finelog as finelog
from experiments.datakit.scripts.dedup_ab_finelog import (
    _coordinator_jobs,
    _coordinator_stage_rows,
    _recovered_worker_stage_rows,
    build_report,
    expected_executions,
    query_archive,
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
        "worker_recovered_stages": 0,
        "cpu_time_total_observed": 13.0,
        "mem_peak_bytes_max_observed": 200,
        "resource_metrics_complete": True,
        "resource_metrics_missing_execution_ids": [],
        "non_end_rows": 0,
    }
    assert report["scenarios"]["baseline_converged"] == {
        "executions": 5,
        "stages": 5,
        "worker_recovered_stages": 0,
        "cpu_time_total_observed": 14.0,
        "mem_peak_bytes_max_observed": 200,
        "resource_metrics_complete": True,
        "resource_metrics_missing_execution_ids": [],
        "non_end_rows": 0,
    }
    assert report["scenarios"]["treatment"]["cpu_time_total_observed"] == 13.0
    assert report["minhash_comparison"]["cpu_delta_fraction"] == 0.0
    assert report["minhash_comparison"]["items_match"] is True


def test_archive_query_combines_archived_and_live_exact_stage_ids(monkeypatch) -> None:
    calls = []

    def fake_query_namespace(*, finelog_config: str, namespace: str, sql: str) -> list[dict]:
        calls.append((finelog_config, namespace, sql))
        if namespace == "log":
            return [{"root_key": "/baseline/0:0", "epoch_ms": 1, "execution_id": "execution-1"}]
        assert "execution-1" in sql
        assert "execution-2" in sql
        stage = _row("/ignored", 0, "execution-1", 5.0)
        stage.pop("root_key")
        stage.pop("epoch_ms")
        return [stage]

    def fake_live_roots(root_job_ids: list[str]) -> list[dict]:
        assert root_job_ids == ["/baseline", "/continuation"]
        return [{"root_key": "/continuation/0:0", "epoch_ms": 2, "execution_id": "execution-2"}]

    def fake_query_live(sql: str) -> list[dict]:
        assert "execution-1" in sql
        assert "execution-2" in sql
        stage = _row("/ignored", 0, "execution-2", 7.0)
        stage.pop("root_key")
        stage.pop("epoch_ms")
        return [stage]

    monkeypatch.setattr(finelog, "_query_namespace", fake_query_namespace)
    monkeypatch.setattr(finelog, "_live_roots", fake_live_roots)
    monkeypatch.setattr(finelog, "_query_live", fake_query_live)

    rows = query_archive(finelog_config="config.yaml", root_job_ids=["/baseline", "/continuation"])

    assert [(row["root_key"], row["cpu_time_total"]) for row in rows] == [
        ("/baseline/0:0", 5.0),
        ("/continuation/0:0", 7.0),
    ]
    assert [namespace for _, namespace, _ in calls] == ["log", "zephyr.stage"]


def test_live_roots_extracts_execution_ids(monkeypatch) -> None:
    monkeypatch.setattr(
        finelog,
        "_query_live",
        lambda _: [
            {
                "root_key": "/continuation/0:0",
                "epoch_ms": 2,
                "data": "Starting zephyr pipeline: 20260725-000001-deadbeef",
            }
        ],
    )

    rows = finelog._live_roots(["/continuation"])

    assert rows == [
        {
            "root_key": "/continuation/0:0",
            "epoch_ms": 2,
            "execution_id": "20260725-000001-deadbeef",
        }
    ]


def test_report_marks_missing_resource_metrics_without_underreporting_totals() -> None:
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
    missing_execution_id = "execution-9"
    for index, item in enumerate(expected):
        order = order_by_root.get(item.root_job_id, 0) + 1
        order_by_root[item.root_job_id] = order
        execution_id = f"execution-{index}"
        if execution_id == missing_execution_id:
            row = _row(item.root_job_id, order, execution_id, 0)
            for key in (
                "stage_name",
                "status",
                "elapsed",
                "items",
                "bytes_processed",
                "total_shards",
                "cpu_pct_avg",
                "cpu_time_total",
                "mem_bytes_avg",
                "mem_peak_bytes_max",
            ):
                row[key] = None
        else:
            row = _row(item.root_job_id, order, execution_id, 10.0 if item.phase == "minhash" else 1.0)
        rows.append(row)

    report = build_report(rows, expected)

    assert report["scenarios"]["treatment"]["resource_metrics_complete"] is False
    assert report["scenarios"]["treatment"]["resource_metrics_missing_execution_ids"] == [missing_execution_id]
    assert report["scenarios"]["treatment"]["cpu_time_total_observed"] == 12.0
    missing = next(execution for execution in report["executions"] if execution["execution_id"] == missing_execution_id)
    assert missing["resource_metrics_available"] is False
    assert missing["cpu_time_total"] is None


def test_worker_rows_recover_exact_resource_reductions_and_stage_span() -> None:
    rows = [
        {
            "execution_id": "execution",
            "stage_name": "stage",
            "shard_idx": 0,
            "status": "START",
            "ts_ms": 1_000,
            "items": 0,
            "bytes_processed": 0,
            "cpu_time_total": 0.0,
            "cpu_avg_pct": 0.0,
            "mem_avg_bytes": 0,
            "mem_peak_bytes": 0,
        },
        {
            "execution_id": "execution",
            "stage_name": "stage",
            "shard_idx": 1,
            "status": "START",
            "ts_ms": 2_000,
            "items": 0,
            "bytes_processed": 0,
            "cpu_time_total": 0.0,
            "cpu_avg_pct": 0.0,
            "mem_avg_bytes": 0,
            "mem_peak_bytes": 0,
        },
        {
            "execution_id": "execution",
            "stage_name": "stage",
            "shard_idx": 0,
            "status": "END",
            "ts_ms": 9_000,
            "items": 10,
            "bytes_processed": 100,
            "cpu_time_total": 3.0,
            "cpu_avg_pct": 20.0,
            "mem_avg_bytes": 100,
            "mem_peak_bytes": 200,
        },
        {
            "execution_id": "execution",
            "stage_name": "stage",
            "shard_idx": 1,
            "status": "END",
            "ts_ms": 11_000,
            "items": 20,
            "bytes_processed": 200,
            "cpu_time_total": 4.0,
            "cpu_avg_pct": 40.0,
            "mem_avg_bytes": 300,
            "mem_peak_bytes": 500,
        },
    ]

    recovered = _recovered_worker_stage_rows(rows, execution_ids={"execution"})

    assert recovered == [
        {
            "execution_id": "execution",
            "stage_name": "stage",
            "status": "END",
            "elapsed": 10.0,
            "items": 30,
            "bytes_processed": 300,
            "total_shards": 2,
            "cpu_pct_avg": 30.0,
            "cpu_time_total": 7.0,
            "mem_bytes_avg": 200.0,
            "mem_peak_bytes_max": 500,
            "metric_source": "worker_end_rows",
        }
    ]


def test_worker_rows_normalize_archive_and_live_timestamps() -> None:
    common = {
        "execution_id": "execution",
        "stage_name": "stage",
        "shard_idx": 0,
        "items": 1,
        "bytes_processed": 2,
        "cpu_time_total": 3.0,
        "cpu_avg_pct": 4.0,
        "mem_avg_bytes": 5,
        "mem_peak_bytes": 6,
    }
    rows = [
        {
            **common,
            "status": "START",
            "ts": "2026-07-25T08:00:00+00:00",
        },
        {
            **common,
            "status": "END",
            "ts": datetime(2026, 7, 25, 8, 0, 1, tzinfo=UTC),
        },
    ]

    recovered = _recovered_worker_stage_rows(rows, execution_ids={"execution"})

    assert recovered[0]["elapsed"] == 1.0


def test_worker_rows_recover_elapsed_when_start_rows_are_missing() -> None:
    rows = [
        {
            "execution_id": "execution",
            "stage_name": "stage",
            "shard_idx": 0,
            "status": "END",
            "ts_ms": 11_000,
            "items": 10,
            "bytes_processed": 100,
            "item_rate": 2.0,
            "byte_rate": 20.0,
            "cpu_time_total": 3.0,
            "cpu_avg_pct": 4.0,
            "mem_avg_bytes": 5,
            "mem_peak_bytes": 6,
        }
    ]

    recovered = _recovered_worker_stage_rows(rows, execution_ids={"execution"})

    assert recovered[0]["elapsed"] == 5.0


def test_coordinator_final_counters_recover_missing_execution_totals() -> None:
    roots = [
        {
            "root_key": "/baseline/0:0",
            "epoch_ms": 1_000,
            "execution_id": "execution",
        }
    ]
    final_rows = [
        {
            "key": "/baseline/zephyr-minhash/0:0",
            "epoch_ms": 11_000,
            "data": (
                "I20260725 zephyr.execution Final counters: "
                "{'zephyr/item_count': 10, 'zephyr/bytes_processed': 20, "
                "'zephyr/worker/cpu_pct_average': 30.0, 'zephyr/worker/cpu_time': 40.0, "
                "'zephyr/worker/mem_average_bytes': 50, 'zephyr/worker/mem_peak_bytes': 60}"
            ),
        }
    ]

    recovered = _coordinator_stage_rows(
        roots,
        final_rows,
        coordinator_jobs=[
            {
                "root_job_id": "/baseline",
                "job_id": "/baseline/zephyr-minhash",
                "submitted_epoch_ms": 2_000,
            }
        ],
        root_job_ids=["/baseline"],
        execution_ids={"execution"},
    )

    assert recovered == [
        {
            "execution_id": "execution",
            "stage_name": "coordinator_pipeline_total",
            "status": "END",
            "elapsed": 10.0,
            "items": 10,
            "bytes_processed": 20,
            "total_shards": 0,
            "cpu_pct_avg": 30.0,
            "cpu_time_total": 40.0,
            "mem_bytes_avg": 50.0,
            "mem_peak_bytes_max": 60,
            "metric_source": "coordinator_final_counters",
            "coordinator_key": "/baseline/zephyr-minhash/0:0",
        }
    ]


def test_coordinator_jobs_are_exact_succeeded_direct_children(monkeypatch) -> None:
    submitted = SimpleNamespace(epoch_ms=1_000)
    jobs = [
        SimpleNamespace(job_id="/baseline/zephyr-first", state=job_pb2.JOB_STATE_SUCCEEDED, submitted_at=submitted),
        SimpleNamespace(
            job_id="/baseline/zephyr-first/workers",
            state=job_pb2.JOB_STATE_KILLED,
            submitted_at=submitted,
        ),
        SimpleNamespace(job_id="/baseline/zephyr-failed", state=job_pb2.JOB_STATE_FAILED, submitted_at=submitted),
        SimpleNamespace(job_id="/baseline/other", state=job_pb2.JOB_STATE_SUCCEEDED, submitted_at=submitted),
    ]
    client = SimpleNamespace(list_jobs=lambda *, prefix, limit: jobs)
    monkeypatch.setattr(finelog, "get_iris_ctx", lambda: SimpleNamespace(client=client))

    assert _coordinator_jobs(["/baseline"]) == [
        {
            "root_job_id": "/baseline",
            "job_id": "/baseline/zephyr-first",
            "submitted_epoch_ms": 1_000,
        }
    ]
