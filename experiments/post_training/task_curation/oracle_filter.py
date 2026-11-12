# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect full-population production Harbor ORACLE trials into an admission receipt."""

import argparse
import json
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec, decode_json_value
from taskcompendium.pipeline.oracle_filter import (
    OracleFilterReceipt,
    OracleTaskReceipt,
    OracleTrialReceipt,
    has_oracle_solution,
)


def harbor_inspection(kind: str, paths: list[str]) -> dict:
    """Validate in Harbor's locked runtime without importing it into the curation environment."""
    project = Path(__file__).resolve().parents[3] / "vendor/harbor"
    result = subprocess.run(
        ["uv", "run", "--project", str(project), "--frozen", "python", "-m", "harbor.cli.curation_inspect", kind],
        input=json.dumps(paths),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def row_task(row: dict) -> TaskSpec | None:
    if "lowered_task_spec" in row:
        return TaskSpec.model_validate(json.loads(row["lowered_task_spec"])["task"])
    serialized = row.get("task_json", row.get("task_spec"))
    return TaskSpec.model_validate_json(serialized) if serialized is not None else None


def row_runtime_hash(row: dict) -> str | None:
    return canonical_sha256(json.loads(row["lowered_task_spec"])["runtime"]) if "lowered_task_spec" in row else None


def collect_trials(jobs: Path) -> dict[tuple[str, str], list[OracleTrialReceipt]]:
    trials: dict[tuple[str, str], list[OracleTrialReceipt]] = {}
    seen = {}
    paths = sorted(jobs.rglob("result.json"))
    validated = harbor_inspection("results", [str(path) for path in paths])
    for path in paths:
        document = decode_json_value(path.read_text())
        if str(path) not in validated:
            continue
        digest = canonical_sha256(document)
        if document["id"] in seen:
            if seen[document["id"]] != digest:
                raise ValueError(f"Conflicting copies of oracle trial {document["id"]}")
            continue
        seen[document["id"]] = digest
        if document["config"]["agent"]["name"] != "oracle":
            raise ValueError(f"Non-oracle trial in filter job: {path}")
        verdict_path = path.parent / "verifier/verdict.json"
        verdict = decode_json_value(verdict_path.read_text()) if verdict_path.exists() else None
        if verdict is not None and not isinstance(verdict, dict):
            raise ValueError(f"Structured verdict must be an object: {verdict_path}")
        exit_path = path.parent / "agent/exit-code.txt"
        exit_code = int(exit_path.read_text()) if exit_path.exists() else None
        receipt = OracleTrialReceipt(
            path=str(path),
            result=document,
            verdict=verdict,
            oracle_exit_code=exit_code,
            result_sha256=canonical_sha256(document),
        )
        trials.setdefault((document["task_name"], document["task_checksum"]), []).append(receipt)
    return trials


def filter_cohort(
    cohort: Path, task_map: Path, jobs: Path, output: Path, conversion_manifest: Path | None = None
) -> OracleFilterReceipt:
    """Filter exact converted bytes; missing tasks, stale maps and incomplete populations fail closed."""
    mappings = decode_json_value(task_map.read_text())
    if not isinstance(mappings, dict):
        raise ValueError("Task map must map TaskSpec IDs to Harbor task_name and task_checksum")
    trials = collect_trials(jobs)
    paths = sorted(cohort.glob("*.parquet")) if cohort.is_dir() else [cohort]
    kept = []
    entries = {}
    matched = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    schema = pq.read_schema(paths[0])
    for path in paths:
        if pq.read_schema(path) != schema:
            raise ValueError("Cohort shards have different schemas")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                task = row_task(row)
                if task is None:
                    continue
                if task.id in entries:
                    raise ValueError(f"Duplicate task ID: {task.id}")
                oracle = has_oracle_solution(task)
                task_trials = []
                checksum = ""
                if oracle:
                    mapping = mappings[task.id]
                    if mapping["task_sha256"] != canonical_sha256(task.model_dump(mode="json")):
                        raise ValueError(f"Task changed after Harbor export: {task.id}")
                    if mapping.get("runtime_sha256") != row_runtime_hash(row):
                        raise ValueError(f"Runtime changed after Harbor export: {task.id}")
                    key = (mapping["task_name"], mapping["task_checksum"])
                    if key in matched:
                        raise ValueError("Two TaskSpecs are mapped to the same Harbor trial")
                    matched.add(key)
                    checksum = key[1]
                    task_trials = trials.get(key, [])
                    if not task_trials:
                        raise ValueError(f"No production Harbor ORACLE trial for {task.id}")
                reason = None
                if oracle and not any(trial.passed(checksum) for trial in task_trials):
                    unresolved = [trial for trial in task_trials if trial.result.get("exception_info") is not None]
                    if unresolved and len(task_trials) < 3:
                        raise ValueError(f"Infrastructure retry budget is incomplete for {task.id}")
                    reason = "oracle_failed_or_unscored_after_retry"
                entry = OracleTaskReceipt(
                    task_sha256=canonical_sha256(task.model_dump(mode="json")),
                    runtime_sha256=row_runtime_hash(row),
                    harbor_task_checksum=checksum,
                    has_oracle=oracle,
                    trials=tuple(task_trials),
                    exclusion_reason=reason,
                )
                entries[task.id] = entry
                if entry.passed:
                    kept.append(row)
    extra = set(mappings) - {task_id for task_id, entry in entries.items() if entry.has_oracle}
    if extra:
        raise ValueError(f"Task map contains unknown or non-oracle TaskSpecs: {sorted(extra)[:5]}")
    manifest = json.loads(conversion_manifest.read_text()) if conversion_manifest is not None else None
    if manifest is not None and manifest["converted_rows"] != len(entries):
        raise ValueError("Converted cohort does not match the full conversion manifest")
    input_rows = manifest["input_rows"] if manifest else len(entries)
    cohort_hash = (
        file_sha256(StoragePath(str(paths[0])))
        if len(paths) == 1
        else canonical_sha256({path.name: file_sha256(StoragePath(str(path))) for path in paths})
    )
    receipt = OracleFilterReceipt(
        recipe_identity=manifest["oracle_recipe_identity"] if manifest else None,
        input_cohort_sha256=cohort_hash,
        input_rows=input_rows,
        conversion_rejections=input_rows - len(entries),
        tasks=entries,
    )
    pq.write_table(pa.Table.from_pylist(kept, schema=schema), output)
    receipt_path = output.parent / "oracle-filter.json"
    receipt_path.write_text(receipt.model_dump_json(indent=2) + "\n")
    summary = {
        "input_rows": input_rows,
        "converted_rows": len(entries),
        "oracle_tasks": sum(e.has_oracle for e in entries.values()),
        "oracle_passed": sum(e.has_oracle and e.passed for e in entries.values()),
        "oracle_excluded": sum(e.has_oracle and not e.passed for e in entries.values()),
        "output_rows": len(kept),
        "receipt": str(receipt_path),
        "output_sha256": file_sha256(StoragePath(str(output))),
    }
    (output.parent / "oracle-filter-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return receipt


def write_task_map(cohort: Path, tasks_dir: Path, output: Path) -> None:
    """Bind each oracle TaskSpec to its already-exported native Harbor task."""
    root = tasks_dir.resolve()
    mappings = {}
    native = harbor_inspection("tasks", [str(path.parent.resolve()) for path in root.rglob("task.toml")])
    paths = sorted(cohort.glob("*.parquet")) if cohort.is_dir() else [cohort]
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                task = row_task(row)
                if task is None or not has_oracle_solution(task):
                    continue
                candidates = {(root / name).resolve() for name in (task.id, row.get("path", task.id))}
                candidates = {path for path in candidates if path.is_relative_to(root) and str(path) in native}
                if len(candidates) != 1:
                    raise ValueError(
                        f"Expected one native task directory for {task.id}; supply an explicit task map otherwise"
                    )
                package = native[str(candidates.pop())]
                mappings[task.id] = {
                    **package,
                    "task_sha256": canonical_sha256(task.model_dump(mode="json")),
                    "runtime_sha256": row_runtime_hash(row),
                }
    output.write_text(json.dumps(mappings, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    mapping = commands.add_parser("map")
    mapping.add_argument("--cohort", type=Path, required=True)
    mapping.add_argument("--tasks-dir", type=Path, required=True)
    mapping.add_argument("--output", type=Path, required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--cohort", type=Path, required=True)
    collect.add_argument("--task-map", type=Path, required=True)
    collect.add_argument("--jobs-dir", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--conversion-manifest", type=Path)
    args = parser.parse_args()
    if args.command == "map":
        write_task_map(args.cohort, args.tasks_dir, args.output)
        return
    filter_cohort(args.cohort, args.task_map, args.jobs_dir, args.output, args.conversion_manifest)


if __name__ == "__main__":
    main()
