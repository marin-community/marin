# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover and validate every archived Zephyr stage row for the dedup A/B."""

import argparse
import ast
import contextlib
import json
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from finelog.client import LogClient
from iris.client import get_iris_ctx
from iris.rpc import job_pb2
from rigging.filesystem import StoragePath
from zephyr.stats import StatsWriter

EXECUTION_ID_PATTERN = re.compile(r"Starting zephyr pipeline: ([0-9]{8}-[0-9]{6}-[0-9a-f]+)")
FINAL_COUNTERS_PATTERN = re.compile(r"Final counters: (\{.*\})")


@dataclass(frozen=True)
class ExpectedExecution:
    """One Zephyr execution in an exact root-log sequence."""

    root_job_id: str
    variant: str
    phase: str
    iteration: int | None = None


def expected_executions(
    *,
    baseline_root: str,
    baseline_continuation_root: str,
    treatment_root: str,
    baseline_capped_iterations: int,
    baseline_converged_iterations: int,
    treatment_iterations: int,
) -> list[ExpectedExecution]:
    """Return the required execution order for all three root jobs."""
    if baseline_capped_iterations <= 0:
        raise ValueError("Baseline cap must be positive")
    if baseline_converged_iterations <= baseline_capped_iterations:
        raise ValueError("Converged baseline must extend beyond the capped iteration")
    if treatment_iterations <= 0:
        raise ValueError("Treatment iterations must be positive")

    result = [
        ExpectedExecution(baseline_root, "baseline", "minhash"),
        ExpectedExecution(baseline_root, "baseline", "initial_graph"),
        *(
            ExpectedExecution(baseline_root, "baseline", "connected_components", iteration)
            for iteration in range(1, baseline_capped_iterations + 1)
        ),
        ExpectedExecution(baseline_root, "baseline", "marker_cap"),
        *(
            ExpectedExecution(
                baseline_continuation_root,
                "baseline",
                "connected_components",
                iteration,
            )
            for iteration in range(baseline_capped_iterations + 1, baseline_converged_iterations + 1)
        ),
        ExpectedExecution(baseline_continuation_root, "baseline", "marker_converged"),
        ExpectedExecution(treatment_root, "treatment", "minhash"),
        ExpectedExecution(treatment_root, "treatment", "initial_graph"),
        *(
            ExpectedExecution(treatment_root, "treatment", "connected_components", iteration)
            for iteration in range(1, treatment_iterations + 1)
        ),
        ExpectedExecution(treatment_root, "treatment", "marker_converged"),
    ]
    return result


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _archive_root_query(root_job_ids: Iterable[str]) -> str:
    root_keys = ", ".join(_sql_literal(f"{root_job_id}/0:0") for root_job_id in root_job_ids)
    return f"""
SELECT DISTINCT
    key AS root_key,
    epoch_ms,
    regexp_extract(
        data,
        'Starting zephyr pipeline: ([0-9]{{8}}-[0-9]{{6}}-[0-9a-f]+)',
        1
    ) AS execution_id
FROM "log"
WHERE key IN ({root_keys})
  AND data LIKE '%Starting zephyr pipeline:%'
ORDER BY root_key, epoch_ms
""".strip()


def _live_root_query(root_job_ids: Iterable[str]) -> str:
    root_keys = ", ".join(_sql_literal(f"{root_job_id}/0:0") for root_job_id in root_job_ids)
    return f"""
SELECT DISTINCT
    key AS root_key,
    epoch_ms,
    data
FROM "log"
WHERE key IN ({root_keys})
  AND data LIKE '%Starting zephyr pipeline:%'
ORDER BY root_key, epoch_ms
""".strip()


def _archive_stage_query(execution_ids: Iterable[str]) -> str:
    ids = ", ".join(_sql_literal(execution_id) for execution_id in execution_ids)
    return f"""
    SELECT DISTINCT
        execution_id,
        stage_name,
        status,
        elapsed,
        items,
        bytes_processed,
        total_shards,
        cpu_pct_avg,
        cpu_time_total,
        mem_bytes_avg,
        mem_peak_bytes_max
    FROM "zephyr.stage"
    WHERE execution_id IN ({ids})
    ORDER BY execution_id, stage_name, status
""".strip()


def _archive_worker_query(execution_ids: Iterable[str]) -> str:
    ids = ", ".join(_sql_literal(execution_id) for execution_id in execution_ids)
    return f"""
    SELECT DISTINCT
        execution_id,
        stage_name,
        shard_idx,
        status,
        ts,
        items,
        bytes_processed,
        item_rate,
        byte_rate,
        cpu_time_total,
        cpu_avg_pct,
        mem_avg_bytes,
        mem_peak_bytes
    FROM "zephyr.worker"
    WHERE execution_id IN ({ids})
      AND status IN ('START', 'END')
    ORDER BY execution_id, stage_name, shard_idx, status, ts
""".strip()


def _coordinator_jobs(root_job_ids: Iterable[str]) -> list[dict[str, Any]]:
    iris = get_iris_ctx()
    if iris is None or iris.client is None:
        raise RuntimeError("Iris client is unavailable outside an Iris job")
    result = []
    for root_job_id in root_job_ids:
        prefix = root_job_id.rstrip("/") + "/"
        jobs = iris.client.list_jobs(prefix=prefix, limit=1_000)
        result.extend(
            {
                "root_job_id": root_job_id,
                "job_id": job.job_id,
                "submitted_epoch_ms": int(job.submitted_at.epoch_ms),
            }
            for job in jobs
            if job.state == job_pb2.JOB_STATE_SUCCEEDED
            and "/" not in job.job_id.removeprefix(prefix)
            and job.job_id.removeprefix(prefix).startswith("zephyr-")
        )
    return sorted(result, key=lambda job: (job["root_job_id"], job["submitted_epoch_ms"], job["job_id"]))


def _coordinator_final_query(coordinator_job_ids: Iterable[str]) -> str:
    keys = ", ".join(_sql_literal(job_id.rstrip("/") + "/0:0") for job_id in coordinator_job_ids)
    return f"""
SELECT DISTINCT
    key,
    epoch_ms,
    data
FROM "log"
WHERE key IN ({keys})
  AND data LIKE '%Final counters:%'
ORDER BY key, epoch_ms
""".strip()


def _query_namespace(
    *,
    finelog_config: str,
    namespace: str,
    sql: str,
    max_rows: int = 10_000,
) -> list[dict[str, Any]]:
    command = [
        "uv",
        "run",
        "finelog",
        "gcs-query",
        finelog_config,
        "--namespace",
        namespace,
        "--format",
        "jsonl",
        "--max-rows",
        str(max_rows),
        sql,
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            f"Finelog archive query failed with exit {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]


def _query_live(sql: str, *, max_rows: int = 10_000) -> list[dict[str, Any]]:
    url = StatsWriter.resolve_url()
    if url is None:
        raise RuntimeError("Live finelog URL is unavailable outside an Iris job")
    with contextlib.closing(LogClient.connect(url)) as client:
        return client.query(sql, max_rows=max_rows).to_pylist()


def _live_roots(root_job_ids: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in _query_live(_live_root_query(root_job_ids)):
        match = EXECUTION_ID_PATTERN.search(row["data"])
        if match is None:
            raise AssertionError(f"Live root row has no execution ID: {row}")
        result.append(
            {
                "root_key": row["root_key"],
                "epoch_ms": row["epoch_ms"],
                "execution_id": match.group(1),
            }
        )
    return result


def _coordinator_stage_rows(
    roots: list[dict[str, Any]],
    final_rows: Iterable[dict[str, Any]],
    *,
    coordinator_jobs: list[dict[str, Any]],
    root_job_ids: list[str],
    execution_ids: set[str],
) -> list[dict[str, Any]]:
    starts_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for root in roots:
        root_job_id = root["root_key"].removesuffix("/0:0")
        starts_by_root[root_job_id].append(root)

    final_by_key: dict[str, dict[str, Any]] = {}
    for row in _unique_rows(final_rows):
        prior = final_by_key.get(row["key"])
        if prior is not None and prior != row:
            raise AssertionError(f"Coordinator key has conflicting final counters: {row['key']}")
        final_by_key[row["key"]] = row

    recovered = []
    for root_job_id in root_job_ids:
        starts = sorted(starts_by_root[root_job_id], key=lambda row: (int(row["epoch_ms"]), row["execution_id"]))
        jobs = [job for job in coordinator_jobs if job["root_job_id"] == root_job_id]
        if len(starts) != len(jobs):
            raise AssertionError(
                f"Coordinator job coverage mismatch for {root_job_id}: starts={len(starts)}, jobs={len(jobs)}"
            )
        for start, job in zip(starts, jobs, strict=True):
            execution_id = start["execution_id"]
            if execution_id not in execution_ids:
                continue
            final = final_by_key.get(job["job_id"].rstrip("/") + "/0:0")
            if final is None:
                continue
            match = FINAL_COUNTERS_PATTERN.search(final["data"])
            if match is None:
                raise AssertionError(f"Coordinator final row has no counters: {final}")
            final_counters = ast.literal_eval(match.group(1))
            elapsed = (int(final["epoch_ms"]) - int(start["epoch_ms"])) / 1_000
            if elapsed < 0:
                raise AssertionError(f"Coordinator final predates execution start: {execution_id}")
            recovered.append(
                {
                    "execution_id": execution_id,
                    "stage_name": "coordinator_pipeline_total",
                    "status": "END",
                    "elapsed": elapsed,
                    "items": int(final_counters.get("zephyr/item_count", 0)),
                    "bytes_processed": int(final_counters.get("zephyr/bytes_processed", 0)),
                    "total_shards": 0,
                    "cpu_pct_avg": float(final_counters.get("zephyr/worker/cpu_pct_average", 0)),
                    "cpu_time_total": float(final_counters.get("zephyr/worker/cpu_time", 0)),
                    "mem_bytes_avg": float(final_counters.get("zephyr/worker/mem_average_bytes", 0)),
                    "mem_peak_bytes_max": int(final_counters.get("zephyr/worker/mem_peak_bytes", 0)),
                    "metric_source": "coordinator_final_counters",
                    "coordinator_key": final["key"],
                }
            )
    return recovered


def _unique_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = json.dumps(row, separators=(",", ":"), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _timestamp_ms(value: datetime | str | int | float) -> int:
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        return int(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return int(timestamp.timestamp() * 1_000)


def _worker_rows_with_timestamp(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        normalized_row = dict(row)
        if "ts_ms" in normalized_row:
            normalized_row["ts_ms"] = int(normalized_row["ts_ms"])
        else:
            normalized_row["ts_ms"] = _timestamp_ms(normalized_row.pop("ts"))
        normalized.append(normalized_row)
    return normalized


def _worker_elapsed(row: dict[str, Any]) -> float:
    candidates = []
    for total_key, rate_key in (("items", "item_rate"), ("bytes_processed", "byte_rate")):
        total = float(row[total_key])
        rate = float(row.get(rate_key, 0))
        if total > 0 and rate > 0:
            candidates.append(total / rate)
    if not candidates:
        return 0.0
    if max(candidates) - min(candidates) > max(candidates) * 1e-6:
        raise AssertionError(f"Worker item and byte rates imply different elapsed times: {row}")
    return sum(candidates) / len(candidates)


def _recovered_worker_stage_rows(
    rows: Iterable[dict[str, Any]],
    *,
    execution_ids: set[str],
) -> list[dict[str, Any]]:
    by_stage: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _unique_rows(_worker_rows_with_timestamp(rows)):
        execution_id = row["execution_id"]
        if execution_id not in execution_ids:
            raise AssertionError(f"Unexpected worker execution {execution_id}")
        by_stage[execution_id, row["stage_name"]].append(row)

    recovered: list[dict[str, Any]] = []
    for (execution_id, stage_name), stage_rows in sorted(by_stage.items()):
        starts = [int(row["ts_ms"]) for row in stage_rows if row["status"] == "START"]
        ends_by_shard: dict[int, dict[str, Any]] = {}
        for row in stage_rows:
            if row["status"] != "END":
                continue
            shard_idx = int(row["shard_idx"])
            prior = ends_by_shard.get(shard_idx)
            if prior is not None and prior != row:
                raise AssertionError(
                    f"Execution {execution_id} stage {stage_name} has conflicting END rows for shard {shard_idx}"
                )
            ends_by_shard[shard_idx] = row
        if not ends_by_shard:
            raise AssertionError(f"Execution {execution_id} stage {stage_name} lacks END worker rows")
        ends = list(ends_by_shard.values())
        if not starts:
            starts = [int(row["ts_ms"] - _worker_elapsed(row) * 1_000) for row in ends]
        elapsed = (max(int(row["ts_ms"]) for row in ends) - min(starts)) / 1_000
        if elapsed < 0:
            raise AssertionError(f"Execution {execution_id} stage {stage_name} has negative worker span")
        recovered.append(
            {
                "execution_id": execution_id,
                "stage_name": stage_name,
                "status": "END",
                "elapsed": elapsed,
                "items": sum(int(row["items"]) for row in ends),
                "bytes_processed": sum(int(row["bytes_processed"]) for row in ends),
                "total_shards": len(ends),
                "cpu_pct_avg": sum(float(row["cpu_avg_pct"]) for row in ends) / len(ends),
                "cpu_time_total": sum(float(row["cpu_time_total"]) for row in ends),
                "mem_bytes_avg": sum(float(row["mem_avg_bytes"]) for row in ends) / len(ends),
                "mem_peak_bytes_max": max(int(row["mem_peak_bytes"]) for row in ends),
                "metric_source": "worker_end_rows",
            }
        )
    return recovered


def query_archive(*, finelog_config: str, root_job_ids: list[str]) -> list[dict[str, Any]]:
    """Recover exact root and stage rows across archive and live retention."""
    archived_roots = _query_namespace(
        finelog_config=finelog_config, namespace="log", sql=_archive_root_query(root_job_ids)
    )
    roots = _unique_rows([*archived_roots, *_live_roots(root_job_ids)])
    if not roots:
        raise FileNotFoundError("No archived or live root execution rows found")
    execution_ids = sorted({row["execution_id"] for row in roots})
    archived_stages = _query_namespace(
        finelog_config=finelog_config,
        namespace="zephyr.stage",
        sql=_archive_stage_query(execution_ids),
    )
    live_stages = _query_live(_archive_stage_query(execution_ids))
    stage_rows = _unique_rows([*archived_stages, *live_stages])
    for row in stage_rows:
        row["metric_source"] = "stage_stat"

    stages_by_execution = _execution_rows(stage_rows)
    missing_execution_ids = set(execution_ids) - stages_by_execution.keys()
    if missing_execution_ids:
        coordinator_jobs = _coordinator_jobs(root_job_ids)
        coordinator_sql = _coordinator_final_query(job["job_id"] for job in coordinator_jobs)
        archived_finals = _query_namespace(
            finelog_config=finelog_config,
            namespace="log",
            sql=coordinator_sql,
            max_rows=1_000,
        )
        live_finals = _query_live(coordinator_sql, max_rows=1_000)
        stage_rows.extend(
            _coordinator_stage_rows(
                roots,
                [*archived_finals, *live_finals],
                coordinator_jobs=coordinator_jobs,
                root_job_ids=root_job_ids,
                execution_ids=missing_execution_ids,
            )
        )
    stages_by_execution = _execution_rows(stage_rows)
    missing_execution_ids = set(execution_ids) - stages_by_execution.keys()
    if missing_execution_ids:
        worker_sql = _archive_worker_query(sorted(missing_execution_ids))
        archived_workers = _query_namespace(
            finelog_config=finelog_config,
            namespace="zephyr.worker",
            sql=worker_sql,
            max_rows=100_000,
        )
        live_workers = _query_live(worker_sql, max_rows=100_000)
        recovered = _recovered_worker_stage_rows(
            [*archived_workers, *live_workers],
            execution_ids=missing_execution_ids,
        )
        stage_rows.extend(recovered)
    stages_by_execution = _execution_rows(stage_rows)
    missing_stage = {
        "stage_name": None,
        "status": None,
        "elapsed": None,
        "items": None,
        "bytes_processed": None,
        "total_shards": None,
        "cpu_pct_avg": None,
        "cpu_time_total": None,
        "mem_bytes_avg": None,
        "mem_peak_bytes_max": None,
    }
    return [
        {
            **root,
            **stage,
        }
        for root in roots
        for stage in stages_by_execution.get(root["execution_id"], [missing_stage])
    ]


def _execution_rows(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_execution: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_execution[row["execution_id"]].append(row)
    return dict(by_execution)


def _execution_summary(
    *,
    expected: ExpectedExecution,
    execution_id: str,
    epoch_ms: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if any(row["stage_name"] is None for row in rows):
        return {
            **asdict(expected),
            "execution_id": execution_id,
            "epoch_ms": epoch_ms,
            "stages": [],
            "non_end_rows": [],
            "cpu_time_total": None,
            "mem_peak_bytes_max": None,
            "resource_metrics_available": False,
        }

    unique_end_rows: dict[str, dict[str, Any]] = {}
    non_end_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["status"] != "END":
            non_end_rows.append(row)
            continue
        stage_name = row["stage_name"]
        prior = unique_end_rows.get(stage_name)
        if prior is not None and prior != row:
            raise AssertionError(f"Execution {execution_id} has conflicting END rows for {stage_name}")
        unique_end_rows[stage_name] = row
    if not unique_end_rows:
        raise AssertionError(f"Execution {execution_id} has no successful stage rows")

    stages = []
    for _, row in sorted(unique_end_rows.items()):
        stage = {
            key: row[key]
            for key in (
                "stage_name",
                "elapsed",
                "items",
                "bytes_processed",
                "total_shards",
                "cpu_pct_avg",
                "cpu_time_total",
                "mem_bytes_avg",
                "mem_peak_bytes_max",
            )
        }
        stage["metric_source"] = row.get("metric_source", "stage_stat")
        stages.append(stage)
    return {
        **asdict(expected),
        "execution_id": execution_id,
        "epoch_ms": epoch_ms,
        "stages": stages,
        "non_end_rows": non_end_rows,
        "cpu_time_total": sum(float(stage["cpu_time_total"]) for stage in stages),
        "mem_peak_bytes_max": max(int(stage["mem_peak_bytes_max"]) for stage in stages),
        "resource_metrics_available": True,
    }


def _scenario_summary(executions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(executions)
    available_rows = [execution for execution in rows if execution["resource_metrics_available"]]
    missing_execution_ids = [
        execution["execution_id"] for execution in rows if not execution["resource_metrics_available"]
    ]
    return {
        "executions": len(rows),
        "stages": sum(len(execution["stages"]) for execution in rows),
        "worker_recovered_stages": sum(
            stage.get("metric_source", "stage_stat") == "worker_end_rows"
            for execution in rows
            for stage in execution["stages"]
        ),
        "cpu_time_total_observed": sum(execution["cpu_time_total"] for execution in available_rows),
        "mem_peak_bytes_max_observed": max(execution["mem_peak_bytes_max"] for execution in available_rows),
        "resource_metrics_complete": not missing_execution_ids,
        "resource_metrics_missing_execution_ids": missing_execution_ids,
        "non_end_rows": sum(len(execution["non_end_rows"]) for execution in rows),
    }


def _minhash_comparison(executions: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant = {execution["variant"]: execution for execution in executions if execution["phase"] == "minhash"}
    if by_variant.keys() != {"baseline", "treatment"}:
        raise AssertionError(f"Expected one MinHash execution per variant, found {sorted(by_variant)}")
    baseline = by_variant["baseline"]
    treatment = by_variant["treatment"]
    baseline_stages = {stage["stage_name"]: stage for stage in baseline["stages"]}
    treatment_stages = {stage["stage_name"]: stage for stage in treatment["stages"]}
    if baseline_stages.keys() != treatment_stages.keys():
        raise AssertionError("MinHash stage names differ between baseline and treatment")
    item_mismatches = {
        name: [baseline_stages[name]["items"], treatment_stages[name]["items"]]
        for name in baseline_stages
        if baseline_stages[name]["items"] != treatment_stages[name]["items"]
    }
    byte_mismatches = {
        name: [baseline_stages[name]["bytes_processed"], treatment_stages[name]["bytes_processed"]]
        for name in baseline_stages
        if baseline_stages[name]["bytes_processed"] != treatment_stages[name]["bytes_processed"]
    }
    if item_mismatches or byte_mismatches:
        raise AssertionError(f"MinHash work differs: items={item_mismatches}, bytes={byte_mismatches}")
    baseline_cpu = float(baseline["cpu_time_total"])
    treatment_cpu = float(treatment["cpu_time_total"])
    return {
        "baseline_cpu_time_total": baseline_cpu,
        "treatment_cpu_time_total": treatment_cpu,
        "cpu_delta": treatment_cpu - baseline_cpu,
        "cpu_delta_fraction": (treatment_cpu - baseline_cpu) / baseline_cpu,
        "baseline_mem_peak_bytes_max": baseline["mem_peak_bytes_max"],
        "treatment_mem_peak_bytes_max": treatment["mem_peak_bytes_max"],
        "items_match": True,
        "bytes_match": True,
    }


def build_report(
    rows: list[dict[str, Any]],
    expected: list[ExpectedExecution],
) -> dict[str, Any]:
    """Assign archive rows to the exact root sequence and aggregate scenarios."""
    expected_by_root: dict[str, list[ExpectedExecution]] = defaultdict(list)
    for item in expected:
        expected_by_root[item.root_job_id].append(item)

    rows_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        root_key = row["root_key"]
        if not root_key.endswith("/0:0"):
            raise AssertionError(f"Unexpected root log key {root_key}")
        rows_by_root[root_key.removesuffix("/0:0")].append(row)
    if rows_by_root.keys() != expected_by_root.keys():
        raise AssertionError(f"Root mismatch: archive={sorted(rows_by_root)}, expected={sorted(expected_by_root)}")

    summaries: list[dict[str, Any]] = []
    for root_job_id, root_expected in expected_by_root.items():
        root_rows = rows_by_root[root_job_id]
        execution_metadata = sorted({(int(row["epoch_ms"]), row["execution_id"]) for row in root_rows})
        if len(execution_metadata) != len(root_expected):
            raise AssertionError(
                f"Root {root_job_id} has {len(execution_metadata)} executions, expected {len(root_expected)}"
            )
        by_execution = _execution_rows(root_rows)
        for item, (epoch_ms, execution_id) in zip(root_expected, execution_metadata, strict=True):
            summaries.append(
                _execution_summary(
                    expected=item,
                    execution_id=execution_id,
                    epoch_ms=epoch_ms,
                    rows=by_execution[execution_id],
                )
            )

    baseline_cap = [
        execution
        for execution in summaries
        if execution["variant"] == "baseline"
        and (
            execution["phase"] in {"minhash", "initial_graph", "marker_cap"}
            or (execution["phase"] == "connected_components" and execution["root_job_id"] == expected[0].root_job_id)
        )
    ]
    baseline_converged = [
        execution for execution in summaries if execution["variant"] == "baseline" and execution["phase"] != "marker_cap"
    ]
    treatment = [execution for execution in summaries if execution["variant"] == "treatment"]
    return {
        "version": "v3",
        "executions": summaries,
        "scenarios": {
            "baseline_cap": _scenario_summary(baseline_cap),
            "baseline_converged": _scenario_summary(baseline_converged),
            "treatment": _scenario_summary(treatment),
        },
        "minhash_comparison": _minhash_comparison(summaries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finelog-config", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--baseline-continuation-root", required=True)
    parser.add_argument("--treatment-root", required=True)
    parser.add_argument("--baseline-capped-iterations", type=int, required=True)
    parser.add_argument("--baseline-converged-iterations", type=int, required=True)
    parser.add_argument("--treatment-iterations", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    expected = expected_executions(
        baseline_root=args.baseline_root,
        baseline_continuation_root=args.baseline_continuation_root,
        treatment_root=args.treatment_root,
        baseline_capped_iterations=args.baseline_capped_iterations,
        baseline_converged_iterations=args.baseline_converged_iterations,
        treatment_iterations=args.treatment_iterations,
    )
    root_job_ids = list(dict.fromkeys(item.root_job_id for item in expected))
    rows = query_archive(finelog_config=args.finelog_config, root_job_ids=root_job_ids)
    report = build_report(rows, expected)
    StoragePath(args.output).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report["scenarios"], indent=2, sort_keys=True))
    print(json.dumps(report["minhash_comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
