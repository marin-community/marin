# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect full-population production Harbor ORACLE trials into an admission receipt."""

import argparse
import json
import shutil
import subprocess
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import tomlkit
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.grading_environment import GradingEnvironmentMode
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExecutionRequirements, TaskSpec, decode_json_value
from taskcompendium.pipeline.oracle_filter import (
    OracleFilterReceipt,
    OracleTaskReceipt,
    OracleTrialReceipt,
    ResourceCalibrationReceipt,
    VerifierResourceProfile,
    has_oracle_solution,
    next_trial_requirements,
)
from taskcompendium.pipeline.resource_calibration import (
    INITIAL_REQUIREMENTS,
    FailureSignal,
    ResourceDimension,
    ResourceUsage,
)
from taskcompendium.runtime.resources import inline_resource

HARBOR_PROJECT = Path(__file__).resolve().parents[3] / "vendor/harbor"
FAILURE_LOG_BYTES = 64 * 1024


def declared_native_requirements(path: Path) -> dict[ResourceDimension, float]:
    document = tomllib.loads((path / "task.toml").read_text())
    values = {name: document.get("environment", {}).get(name) for name in ("cpus", "memory_mb", "storage_mb")}
    values.update(
        {f"{phase}_timeout_sec": document.get(phase, {}).get("timeout_sec") for phase in ("agent", "verifier")}
    )
    return {ResourceDimension(name): value for name, value in values.items() if value is not None}


def native_requirements(path: Path) -> ExecutionRequirements:
    document = tomllib.loads((path / "task.toml").read_text())
    environment = document["environment"]
    return ExecutionRequirements(
        **{name: environment[name] for name in ("cpus", "memory_mb", "storage_mb")},
        agent_timeout_sec=document["agent"]["timeout_sec"],
        verifier_timeout_sec=document["verifier"]["timeout_sec"],
    )


def native_verifier_profile(path: Path) -> VerifierResourceProfile:
    verifier = tomllib.loads((path / "task.toml").read_text()).get("verifier", {})
    mode = verifier.get("environment_mode") or ("separate" if "environment" in verifier else "shared")
    environment = verifier.get("environment", {})
    return VerifierResourceProfile(
        mode=GradingEnvironmentMode(mode),
        fixed={
            ResourceDimension(name): environment[name]
            for name in ("cpus", "memory_mb", "storage_mb")
            if environment.get(name) is not None
        },
    )


def calibration_package_identity(path: Path, oracle_kind: str) -> str:
    """Identify the faithful package while allowing declared capacity/deadline additions."""
    document = tomllib.loads((path / "task.toml").read_text())
    for environment in (document.get("environment", {}), document.get("verifier", {}).get("environment", {})):
        for name in ("cpus", "memory_mb", "storage_mb"):
            environment.pop(name, None)
    for phase in ("agent", "verifier"):
        if phase in document:
            document[phase].pop("timeout_sec", None)
            if not document[phase]:
                document.pop(phase)
    files = {
        str(file.relative_to(path)): {
            "sha256": file_sha256(StoragePath(str(file))),
            "mode": file.stat().st_mode & 0o777,
        }
        for file in path.rglob("*")
        if file.is_file()
        and file != path / "task.toml"
        and not (oracle_kind == "calibration_only" and file.relative_to(path).parts[0] == "solution")
    }
    return canonical_sha256({"task": document, "files": files})


def trial_receipt(path: Path) -> OracleTrialReceipt:
    document = decode_json_value(path.read_text())
    native = Path(document["config"]["task"]["path"])
    package = harbor_inspection("tasks", [str(native)])[str(native)]
    if package["task_checksum"] != document["task_checksum"]:
        raise ValueError(f"Native task changed after its oracle trial: {native}")
    verdict_path = path.parent / "verifier/verdict.json"
    verdict = decode_json_value(verdict_path.read_text()) if verdict_path.exists() else None
    exit_path = path.parent / "agent/exit-code.txt"
    usage_path = path.parent / "agent/resource-usage.json"
    verifier_usage_path = path.parent / "verifier/resource-usage.json"
    logs = []
    for name in ("agent/oracle.txt", "verifier/test-stdout.txt", "verifier/test-stderr.txt"):
        log = path.parent / name
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - FAILURE_LOG_BYTES))
                logs.append(stream.read().decode(errors="replace"))
    return OracleTrialReceipt(
        path=str(path),
        result=document,
        verdict=verdict,
        oracle_exit_code=int(exit_path.read_text()) if exit_path.exists() else None,
        result_sha256=canonical_sha256(document),
        execution_requirements=native_requirements(native),
        resource_usage=(
            ResourceUsage.model_validate_json(usage_path.read_text()) if usage_path.exists() else ResourceUsage()
        ),
        verifier_resource_usage=(
            ResourceUsage.model_validate_json(verifier_usage_path.read_text())
            if verifier_usage_path.exists()
            else ResourceUsage()
        ),
        failure_log="\n".join(logs),
    )


def harbor_inspection(kind: str, paths: list[str]) -> dict:
    """Validate in Harbor's locked runtime without importing it into the curation environment."""
    result = subprocess.run(
        ["uv", "run", "--project", str(HARBOR_PROJECT), "--frozen", "python", "-m", "harbor.cli.curation_inspect", kind],
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
        receipt = trial_receipt(path)
        trials.setdefault((document["task_name"], document["task_checksum"]), []).append(receipt)
    return trials


def filter_cohort(
    cohort: Path, task_map: Path, jobs: Path, output: Path, conversion_manifest: Path | None = None
) -> OracleFilterReceipt:
    """Filter exact converted bytes; missing tasks, stale maps and incomplete populations fail closed."""
    for path in (output, output.parent / "oracle-filter.json", output.parent / "oracle-filter-summary.json"):
        if path.exists():
            raise FileExistsError(f"Preserve existing oracle artifacts; select a new output version: {path}")
    mappings = decode_json_value(task_map.read_text())
    if not isinstance(mappings, dict):
        raise ValueError("Task map must map TaskSpec IDs to Harbor task_name and task_checksum")
    trials = collect_trials(jobs)
    original_packages = harbor_inspection("tasks", [mapping["task_path"] for mapping in mappings.values()])
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
                if task.id in mappings:
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
                else:
                    raise ValueError(f"No resource calibration mapping for {task.id}")
                calibration = (
                    ResourceCalibrationReceipt.model_validate(mapping["calibration"])
                    if "calibration" in mapping
                    else ResourceCalibrationReceipt(
                        declared_requirements=task.execution_requirements,
                        preserved_requirements=declared_native_requirements(Path(mapping["task_path"])),
                        verifier=native_verifier_profile(Path(mapping["task_path"])),
                        oracle_kind="original" if oracle else "calibration_only",
                        attempts=tuple(task_trials),
                    )
                )
                original = Path(mapping["task_path"])
                if original_packages[str(original)]["task_checksum"] != mapping.get("source_task_checksum", checksum):
                    raise ValueError(f"Source package changed after Harbor export: {task.id}")
                if calibration.preserved_requirements != declared_native_requirements(original):
                    raise ValueError(f"Calibration changed source-declared limits: {task.id}")
                if calibration.verifier != native_verifier_profile(original):
                    raise ValueError(f"Calibration changed the declared verifier resource profile: {task.id}")
                package_identity = calibration_package_identity(original, calibration.oracle_kind)
                for attempt in calibration.attempts:
                    key = (attempt.result["task_name"], attempt.result["task_checksum"])
                    if attempt not in trials.get(key, []):
                        raise ValueError(f"Calibration trial missing from production jobs: {task.id}")
                    package = Path(attempt.result["config"]["task"]["path"])
                    if calibration_package_identity(package, calibration.oracle_kind) != package_identity:
                        raise ValueError(f"Calibration changed the task environment or scripts: {task.id}")
                requirements = calibration.execution_requirements
                if task.execution_requirements is not None and task.execution_requirements != requirements:
                    raise ValueError(f"Calibration overrides task specification: {task.id}")
                task = task.model_copy(update={"execution_requirements": requirements})
                final_package = Path(calibration.attempts[-1].result["config"]["task"]["path"])
                native_config = inline_resource("task.toml", (final_package / "task.toml").read_bytes())
                task = task.model_copy(
                    update={
                        "resources": task.resources.model_copy(
                            update={
                                "verifier": tuple(
                                    (
                                        resource.model_copy(update={"source": native_config.source})
                                        if resource.path == "task.toml"
                                        else resource
                                    )
                                    for resource in task.resources.verifier
                                )
                            }
                        )
                    }
                )
                row = row_with_task(row, task)
                reason = (
                    None
                    if calibration.passed
                    else (f"resource_calibration:{calibration.attempts[-1].failure_signal.value}")
                )
                entry = OracleTaskReceipt(
                    task_sha256=canonical_sha256(task.model_dump(mode="json")),
                    runtime_sha256=row_runtime_hash(row),
                    harbor_task_checksum=checksum,
                    has_oracle=oracle,
                    trials=(calibration.attempts[-1],) if oracle else (),
                    calibration=calibration,
                    exclusion_reason=reason,
                )
                entries[task.id] = entry
                if entry.passed:
                    kept.append(row)
    extra = set(mappings) - set(entries)
    if extra:
        raise ValueError(f"Task map contains unknown TaskSpecs: {sorted(extra)[:5]}")
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
        "uniform_execution_requirements": (
            receipt.uniform_requirements.model_dump(mode="json") if receipt.uniform_requirements else None
        ),
    }
    (output.parent / "oracle-filter-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return receipt


def write_task_map(cohort: Path, tasks_dir: Path, output: Path) -> None:
    """Bind every TaskSpec to its already-exported native Harbor task."""
    if output.exists():
        raise FileExistsError(f"Preserve the existing task map; select a new output version: {output}")
    root = tasks_dir.resolve()
    mappings = {}
    native = harbor_inspection("tasks", [str(path.parent.resolve()) for path in root.rglob("task.toml")])
    paths = sorted(cohort.glob("*.parquet")) if cohort.is_dir() else [cohort]
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                task = row_task(row)
                if task is None:
                    continue
                candidates = {(root / name).resolve() for name in (task.id, row.get("path", task.id))}
                candidates = {path for path in candidates if path.is_relative_to(root) and str(path) in native}
                if len(candidates) != 1:
                    raise ValueError(
                        f"Expected one native task directory for {task.id}; supply an explicit task map otherwise"
                    )
                task_path = candidates.pop()
                package = native[str(task_path)]
                mappings[task.id] = {
                    **package,
                    "task_path": str(task_path),
                    "task_sha256": canonical_sha256(task.model_dump(mode="json")),
                    "runtime_sha256": row_runtime_hash(row),
                }
    output.write_text(json.dumps(mappings, indent=2) + "\n")


def row_with_task(row: dict, task: TaskSpec) -> dict:
    row = dict(row)
    if "lowered_task_spec" in row:
        envelope = json.loads(row["lowered_task_spec"])
        envelope["task"] = task.model_dump(mode="json")
        row["lowered_task_spec"] = json.dumps(envelope)
    else:
        row["task_json" if "task_json" in row else "task_spec"] = task.model_dump_json()
    return row


def calibrated_task(
    task: TaskSpec,
    mapping: dict,
    destination: Path,
    environment: str,
    environment_kwargs: dict,
    calibration_solutions: Path | None,
) -> dict:
    """Run the production Trial path once per shape, creating a fresh task package each time."""
    original = Path(mapping["task_path"])
    native = harbor_inspection("tasks", [str(original)])[str(original)]
    if native["task_checksum"] != mapping["task_checksum"]:
        raise ValueError(f"Task changed after Harbor export: {task.id}")
    declared = task.execution_requirements
    preserved = declared_native_requirements(original)
    verifier = native_verifier_profile(original)
    shape = declared or ExecutionRequirements.model_validate({**INITIAL_REQUIREMENTS.model_dump(), **preserved})
    if declared is not None and native_requirements(original) != declared:
        raise ValueError(f"Harbor package differs from declared execution requirements: {task.id}")
    attempts = []
    infrastructure_attempts = 0
    while True:
        attempt = destination / canonical_sha256(task.id) / f"attempt-{len(attempts):02d}"
        attempt.mkdir(parents=True, exist_ok=False)
        package = attempt / original.name
        shutil.copytree(original, package)
        if not has_oracle_solution(task):
            if calibration_solutions is None:
                raise ValueError(f"Write a calibration-only solution for {task.id} and supply --calibration-solutions")
            shutil.copytree(calibration_solutions / task.id, package / "solution", dirs_exist_ok=True)
        if declared is None:
            document = tomlkit.parse((package / "task.toml").read_text())
            for name in ("cpus", "memory_mb", "storage_mb"):
                document.setdefault("environment", {})[name] = shape.model_dump()[name]
                if "environment" in document.get("verifier", {}):
                    document["verifier"]["environment"].setdefault(name, shape.model_dump()[name])
            for phase in ("agent", "verifier"):
                document.setdefault(phase, {})["timeout_sec"] = shape.model_dump()[f"{phase}_timeout_sec"]
            (package / "task.toml").write_text(tomlkit.dumps(document))
        config = {
            "task": {"path": str(package)},
            "agent": {"name": "oracle", "kwargs": {"resource_usage_filename": "resource-usage.json"}},
            "environment": {"type": environment, "kwargs": environment_kwargs},
            "trials_dir": str(attempt / "jobs"),
            "trial_name": "oracle",
        }
        config_path = attempt / "trial.json"
        config_path.write_text(json.dumps(config))
        with (attempt / "harbor.log").open("w") as log:
            subprocess.run(
                [
                    "uv",
                    "run",
                    "--isolated",
                    "--project",
                    str(HARBOR_PROJECT),
                    "--frozen",
                    "--extra",
                    "runtime",
                    "harbor",
                    "trials",
                    "start",
                    "--config",
                    str(config_path),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        paths = list((attempt / "jobs").rglob("result.json"))
        if len(paths) != 1:
            raise ValueError(f"Expected exactly one persisted production trial for {task.id}")
        trial = trial_receipt(paths[0])
        attempts.append(trial)
        if trial.failure_signal is FailureSignal.INFRASTRUCTURE:
            infrastructure_attempts += 1
            if infrastructure_attempts < 3:
                continue
            break
        infrastructure_attempts = 0
        if trial.passed(trial.result["task_checksum"]) or declared is not None:
            break
        updated = next_trial_requirements(trial, preserved, verifier)
        if updated is None:
            break
        shape = updated
    calibration = ResourceCalibrationReceipt(
        declared_requirements=declared,
        preserved_requirements=preserved,
        verifier=verifier,
        oracle_kind="original" if has_oracle_solution(task) else "calibration_only",
        attempts=tuple(attempts),
    )
    final = attempts[-1].result
    result = {
        **mapping,
        "task_name": final["task_name"],
        "task_checksum": final["task_checksum"],
        "source_task_checksum": mapping["task_checksum"],
        "calibration": calibration.model_dump(mode="json"),
    }
    (destination / canonical_sha256(task.id) / "task-map.json").write_text(json.dumps({task.id: result}, indent=2))
    return result


def calibrate_cohort(
    cohort: Path,
    task_map: Path,
    destination: Path,
    environment: str,
    environment_kwargs: dict,
    concurrency: int,
    calibration_solutions: Path | None = None,
) -> None:
    """Fold calibration into full-population oracle filtering; collect the same trials afterward."""
    mappings = json.loads(task_map.read_text())
    tasks = []
    paths = sorted(cohort.glob("*.parquet")) if cohort.is_dir() else [cohort]
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                task = row_task(row)
                if task is None:
                    continue
                mapping = mappings[task.id]
                if mapping["task_sha256"] != canonical_sha256(task.model_dump(mode="json")):
                    raise ValueError(f"Task changed after Harbor export: {task.id}")
                if mapping.get("runtime_sha256") != row_runtime_hash(row):
                    raise ValueError(f"Runtime changed after Harbor export: {task.id}")
                tasks.append(task)
    if len({task.id for task in tasks}) != len(tasks) or set(mappings) != {task.id for task in tasks}:
        raise ValueError("Calibration task map must cover exactly the full converted population")
    destination.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            task.id: pool.submit(
                calibrated_task,
                task,
                mappings[task.id],
                destination,
                environment,
                environment_kwargs,
                calibration_solutions,
            )
            for task in tasks
        }
        calibrated = {task_id: future.result() for task_id, future in futures.items()}
    (destination / "task-map.json").write_text(json.dumps(calibrated, indent=2) + "\n")


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
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--cohort", type=Path, required=True)
    calibrate.add_argument("--task-map", type=Path, required=True)
    calibrate.add_argument("--output-dir", type=Path, required=True)
    calibrate.add_argument("--environment", choices=("local", "daytona"), required=True)
    calibrate.add_argument("--environment-kwargs", type=json.loads, default={})
    calibrate.add_argument("--concurrency", type=int, required=True)
    calibrate.add_argument("--calibration-solutions", type=Path)
    args = parser.parse_args()
    if args.command == "map":
        write_task_map(args.cohort, args.tasks_dir, args.output)
        return
    if args.command == "calibrate":
        calibrate_cohort(
            args.cohort,
            args.task_map,
            args.output_dir,
            args.environment,
            args.environment_kwargs,
            args.concurrency,
            args.calibration_solutions,
        )
        return
    filter_cohort(args.cohort, args.task_map, args.jobs_dir, args.output, args.conversion_manifest)


if __name__ == "__main__":
    main()
