# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover and validate every archived Zephyr stage row for the dedup A/B."""

import argparse
import json
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from rigging.filesystem import StoragePath


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


def _query_namespace(*, finelog_config: str, namespace: str, sql: str) -> list[dict[str, Any]]:
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
        "10000",
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


def query_archive(*, finelog_config: str, root_job_ids: list[str]) -> list[dict[str, Any]]:
    """Recover root execution IDs, then query only their archived stage rows."""
    roots = _query_namespace(
        finelog_config=finelog_config,
        namespace="log",
        sql=_archive_root_query(root_job_ids),
    )
    if not roots:
        raise FileNotFoundError("No archived root execution rows found")
    stage_rows = _query_namespace(
        finelog_config=finelog_config,
        namespace="zephyr.stage",
        sql=_archive_stage_query(sorted({row["execution_id"] for row in roots})),
    )
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
        raise AssertionError(f"Execution {execution_id} has no archived stage rows")

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

    stages = [
        {
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
        for _, row in sorted(unique_end_rows.items())
    ]
    return {
        **asdict(expected),
        "execution_id": execution_id,
        "epoch_ms": epoch_ms,
        "stages": stages,
        "non_end_rows": non_end_rows,
        "cpu_time_total": sum(float(stage["cpu_time_total"]) for stage in stages),
        "mem_peak_bytes_max": max(int(stage["mem_peak_bytes_max"]) for stage in stages),
    }


def _scenario_summary(executions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(executions)
    return {
        "executions": len(rows),
        "stages": sum(len(execution["stages"]) for execution in rows),
        "cpu_time_total": sum(execution["cpu_time_total"] for execution in rows),
        "mem_peak_bytes_max": max(execution["mem_peak_bytes_max"] for execution in rows),
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
        "version": "v1",
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
