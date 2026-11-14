# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect full-population production Harbor ORACLE trials into an admission receipt."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import tomllib
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import tomlkit
from finestore.rl.curation import ContentEvidenceReference
from iris.cluster.client.bundle import collect_workspace_files
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import ExponentialBackoff
from shellbox.backends.local.cgroup import check_cgroup_parent, delegated_cgroup_parent
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.grading_environment import GradingEnvironmentMode
from taskcompendium.harbor.adapter import TASK_SPEC_FILE, declared_execution_limits
from taskcompendium.harbor.execution import native_execution_limits
from taskcompendium.harbor.oracle import read_oracle_command
from taskcompendium.harbor.protected_runtime import read_runtime_binding
from taskcompendium.harbor.runtime import TaskNetworkPolicyReceipt, TaskSpecRuntimeBinding, read_source_runtime
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExecutionRequirements, TaskSpec, decode_json_value
from taskcompendium.pipeline.oracle_filter import (
    LOCAL_EXECUTION_HOST_FIELDS,
    BackendFixIdentity,
    ExecutabilityRunReceipt,
    InfrastructureRetryApproval,
    InfrastructureRetryContext,
    InfrastructureRetryPlan,
    InfrastructureRetryTransition,
    LocalExecutionHostTransition,
    MechanicalOracleTaskReceipt,
    NativeOracleSuccessPolicy,
    NativeVerifierGrade,
    NegativeControlApplicability,
    OracleExclusionReason,
    OracleFilterReceipt,
    OracleNegativeControlReceipt,
    OracleTaskReceipt,
    OracleTrialReceipt,
    PriorPhaseFailureReceipt,
    ResourceCalibrationReceipt,
    SourcePytestEnvironment,
    VerifierResourceProfile,
    negative_control_applicability,
    next_trial_requirements,
    oracle_failure_reason,
    pre_oracle_host_unavailable,
    require_infrastructure_retry_parent,
    source_environment_incompatibility,
    trial_network_qualified,
)
from taskcompendium.pipeline.resource_calibration import (
    INITIAL_REQUIREMENTS,
    FailureSignal,
    ResourceDimension,
    ResourceUsage,
)
from taskcompendium.runtime.resources import has_oracle_solution, inline_resource

from experiments.post_training.task_curation.harbor_inspection import harbor_inspection
from experiments.post_training.task_curation.local_host_admission import LocalHostAdmission, LocalHostAdmissionConfig
from experiments.post_training.task_curation.negative_controls import require_negative_control

from experiments.post_training.task_curation.prior_phase_failures import (
    bind_prior_phase_failures,
    require_prior_phase_failure,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HARBOR_PROJECT = PROJECT_ROOT / "vendor/harbor"
BACKEND_CODE_ROOTS = (
    "lib/shellbox/",
    "lib/rigging/",
    "lib/rolloutengine/",
    "lib/verifyit/",
    "lib/taskcompendium/",
    "lib/finestore/",
    "vendor/harbor/",
)
BACKEND_CODE_FILES = {
    "uv.lock",
    "pyproject.toml",
    "experiments/post_training/task_curation/oracle_filter.py",
    "experiments/post_training/task_curation/local_host_admission.py",
    "experiments/post_training/task_curation/prior_phase_failures.py",
    "experiments/post_training/task_curation/archive_evidence.py",
    "experiments/post_training/task_curation/negative_controls.py",
    "experiments/post_training/task_curation/harbor_inspection.py",

}
FAILURE_LOG_BYTES = 64 * 1024
CALIBRATION_ATTEMPT_TIMEOUT = 72 * 60 * 60


def declared_native_requirements(path: Path) -> dict[ResourceDimension, float]:
    if (path / TASK_SPEC_FILE).is_file():
        task = TaskSpec.model_validate_json((path / TASK_SPEC_FILE).read_text())
        return {ResourceDimension(name): value for name, value in declared_execution_limits(task).items()}
    document = tomllib.loads((path / "task.toml").read_text())
    return {ResourceDimension(name): value for name, value in native_execution_limits(document).declared.items()}


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
    for owner in (document, document.get("verifier", {})):
        if owner.get("environment") == {}:
            owner.pop("environment")
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
    return canonical_sha256(
        {"task": document, "files": files, "grading_environment_mode": native_verifier_profile(path).mode.value}
    )


def receipt_from_trial(
    path: Path,
    document: dict,
    native_checksum: str,
    native_grade: NativeVerifierGrade | None = None,
) -> OracleTrialReceipt:
    """Read a trial's evidence against its measured native checksum."""
    native = Path(document["config"]["task"]["path"])
    if native_checksum != document["task_checksum"]:
        raise ValueError(f"Native task changed after its oracle trial: {native}")
    verdict_path = path.parent / "verifier/verdict.json"
    verdict = decode_json_value(verdict_path.read_text()) if verdict_path.exists() else None
    exit_path = path.parent / "agent/exit-code.txt"
    usage_path = path.parent / "agent/resource-usage.json"
    verifier_usage_path = path.parent / "verifier/resource-usage.json"
    source_environment_path = path.parent / "verifier/source-environment.json"
    network_path = path.parent / "network-policy.json"
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
        native_grade=native_grade,
        resource_usage=(
            ResourceUsage.model_validate_json(usage_path.read_text()) if usage_path.exists() else ResourceUsage()
        ),
        verifier_resource_usage=(
            ResourceUsage.model_validate_json(verifier_usage_path.read_text())
            if verifier_usage_path.exists()
            else ResourceUsage()
        ),
        failure_log="\n".join(logs),
        source_environment=(
            SourcePytestEnvironment.model_validate_json(source_environment_path.read_text())
            if source_environment_path.exists()
            else None
        ),
        network_policy=(
            TaskNetworkPolicyReceipt.model_validate_json(network_path.read_text()) if network_path.exists() else None
        ),
    )


def row_task(row: dict) -> TaskSpec | None:
    if "lowered_task_spec" in row:
        return TaskSpec.model_validate(json.loads(row["lowered_task_spec"])["task"])
    serialized = row.get("task_json", row.get("task_spec"))
    return TaskSpec.model_validate_json(serialized) if serialized is not None else None


def row_runtime(row: dict) -> dict | None:
    if "lowered_task_spec" in row:
        return json.loads(row["lowered_task_spec"])["runtime"]
    runtime = row.get("runtime")
    return json.loads(runtime) if runtime is not None else None


def row_runtime_hash(row: dict) -> str | None:
    runtime = row_runtime(row)
    return canonical_sha256(runtime) if runtime is not None else None


def cohort_sha256(cohort: Path) -> str:
    paths = sorted(cohort.glob("*.parquet")) if cohort.is_dir() else [cohort]
    return (
        file_sha256(StoragePath(str(paths[0])))
        if len(paths) == 1
        else canonical_sha256({path.name: file_sha256(StoragePath(str(path))) for path in paths})
    )


def trial_result_paths(jobs: Path) -> list[Path]:
    """Find persisted trials without descending into task-controlled outputs."""
    results = []
    for directory, subdirectories, files in jobs.walk():
        if "task.toml" in files and "instruction.md" in files:
            subdirectories.clear()
            continue
        if "config.json" not in files:
            continue
        config = decode_json_value((directory / "config.json").read_text())
        if "task" not in config or "agent" not in config:
            continue
        subdirectories[:] = [name for name in subdirectories if name == "attempts"]
        if "result.json" in files:
            results.append(directory / "result.json")
        elif "lifecycle-result.json" in files:
            results.append(directory / "lifecycle-result.json")
    return sorted(results)


def collect_trials(jobs: Path) -> dict[tuple[str, str], list[OracleTrialReceipt]]:
    trials: dict[tuple[str, str], list[OracleTrialReceipt]] = {}
    seen = {}
    paths = trial_result_paths(jobs)
    validated = harbor_inspection("results", [str(path) for path in paths])
    documents = {}
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
        if document["config"]["agent"]["name"] not in {"oracle", "task-spec-chat", "terminus-2"}:
            raise ValueError(f"Non-oracle trial in filter job: {path}")
        documents[path] = document
    native_paths = sorted({document["config"]["task"]["path"] for document in documents.values()})
    packages = harbor_inspection("tasks", native_paths) if native_paths else {}
    native_grades = harbor_inspection("native-grades", [str(path) for path in documents]) if documents else {}
    for path, document in documents.items():
        native = document["config"]["task"]["path"]
        grade = native_grades.get(str(path))
        receipt = receipt_from_trial(
            path,
            document,
            packages[native]["task_checksum"],
            NativeVerifierGrade.model_validate(grade) if grade is not None else None,
        )
        trials.setdefault((document["task_name"], document["task_checksum"]), []).append(receipt)
    return trials


def require_retry_provenance(
    task_id: str,
    history: tuple[InfrastructureRetryTransition, ...],
    parent_maps: dict[Path, tuple[str, dict[str, dict]]],
    retry_plans: dict[Path, tuple[str, InfrastructureRetryPlan]],
) -> None:
    """Reject changed retry inputs, reusing previously verified parent maps and plans."""
    for transition in history:
        parent_path = Path(transition.parent_map_path)
        if parent_path not in parent_maps:
            parent_maps[parent_path] = (
                file_sha256(StoragePath(str(parent_path))),
                json.loads(parent_path.read_text()),
            )
        parent_hash, parent_document = parent_maps[parent_path]
        if parent_hash != transition.parent_map_sha256:
            raise ValueError(f"Infrastructure parent task map changed: {task_id}")
        plan_path = Path(transition.plan_path)
        if plan_path not in retry_plans:
            checked_plan = InfrastructureRetryPlan.model_validate_json(plan_path.read_text())
            for frozen_path, expected in (
                (checked_plan.cohort_path, checked_plan.cohort_sha256),
                (checked_plan.source_parent_map_path, checked_plan.source_parent_map_sha256),
            ):
                if file_sha256(StoragePath(frozen_path)) != expected:
                    raise ValueError(f"Infrastructure retry input changed: {frozen_path}")
            retry_plans[plan_path] = (
                file_sha256(StoragePath(str(plan_path))),
                checked_plan,
            )
        plan_hash, plan = retry_plans[plan_path]
        if (
            plan_hash != transition.plan_sha256
            or plan.backend_fix != transition.backend_fix
            or plan.approvals[task_id] != transition.approval
            or plan.parent_map_path != str(parent_path)
            or plan.parent_map_sha256 != transition.parent_map_sha256
            or plan.local_host != transition.local_host
        ):
            raise ValueError(f"Infrastructure retry plan changed: {task_id}")
        parent = parent_document[task_id]
        if canonical_sha256(parent) != transition.parent_mapping_sha256:
            raise ValueError(f"Infrastructure parent mapping changed: {task_id}")
        if (
            parent["task_sha256"] != transition.task_sha256
            or parent.get("runtime_sha256") != transition.runtime_sha256
            or parent.get("source_task_checksum", parent["task_checksum"]) != transition.source_task_checksum
        ):
            raise ValueError(f"Infrastructure source identity changed: {task_id}")
        if ResourceCalibrationReceipt.model_validate(parent["calibration"]) != transition.parent_calibration:
            raise ValueError(f"Infrastructure parent attempts changed: {task_id}")


def filter_cohort(
    cohort: Path,
    task_map: Path,
    jobs: Path,
    output: Path,
    conversion_manifest: Path | None = None,
    executability_runs: Path | None = None,
    mechanical_outcomes: Path | None = None,
    input_evidence: ContentEvidenceReference | None = None,
    native_success_policies: Path | None = None,
    negative_controls: Path | None = None,
) -> OracleFilterReceipt:
    """Filter exact converted bytes; missing tasks, stale maps and incomplete populations fail closed."""
    for path in (
        output,
        output.parent / "oracle-filter.json",
        output.parent / "oracle-filter-summary.json",
        output.parent / "excluded-tasks.parquet",
    ):
        if path.exists():
            raise FileExistsError(f"Preserve existing oracle artifacts; select a new output version: {path}")
    mappings = decode_json_value(task_map.read_text())
    if not isinstance(mappings, dict):
        raise ValueError("Task map must map TaskSpec IDs to Harbor task_name and task_checksum")
    policies = {}
    checked_evidence = set()
    if native_success_policies is not None:
        for line in native_success_policies.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = record["task_id"]
            if task_id in policies or task_id not in mappings:
                raise ValueError(f"Duplicate or unknown native success policy: {task_id}")
            policy = NativeOracleSuccessPolicy.model_validate(record["policy"])
            for evidence in (policy.consumer, policy.source_census, policy.decision):
                key = (evidence.uri, evidence.sha256)
                if key in checked_evidence:
                    continue
                if hashlib.sha256(StoragePath(evidence.uri).read_bytes()).hexdigest() != evidence.sha256:
                    raise ValueError("Native success policy evidence bytes changed")
                checked_evidence.add(key)
            policies[task_id] = policy
    trials = collect_trials(jobs)
    controls = {}
    if executability_runs is not None:
        for line in executability_runs.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = record["task_id"]
            if task_id in controls:
                raise ValueError(f"Duplicate executability control task: {task_id}")
            controls[task_id] = ExecutabilityRunReceipt.model_validate(record["receipt"])
    negative = {}
    if negative_controls is not None:
        for line in negative_controls.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = record["task_id"]
            if task_id in negative or task_id not in mappings:
                raise ValueError(f"Duplicate or unknown physical negative control: {task_id}")
            control = OracleNegativeControlReceipt.model_validate(record["receipt"])
            require_negative_control(control)
            negative[task_id] = control
    mechanical = {}
    if mechanical_outcomes is not None:
        for line in mechanical_outcomes.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = record["task_id"]
            if task_id in mechanical or task_id in mappings:
                raise ValueError(f"Duplicate mechanical/trial outcome: {task_id}")
            mechanical[task_id] = MechanicalOracleTaskReceipt.model_validate(record["receipt"])
    original_packages = (
        harbor_inspection("tasks", [mapping["task_path"] for mapping in mappings.values()]) if mappings else {}
    )
    paths = sorted(cohort.glob("*.parquet")) if cohort.is_dir() else [cohort]
    cohort_hash = cohort_sha256(cohort)
    if input_evidence is not None and input_evidence.sha256 != cohort_hash:
        raise ValueError("Input evidence differs from the frozen cohort")
    kept = []
    excluded = []
    entries: dict[str, OracleTaskReceipt | MechanicalOracleTaskReceipt] = {}
    parent_maps = {}
    retry_plans = {}
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
                if task.id in mechanical or task.id not in mappings:
                    entry = mechanical.get(task.id)
                    if entry is None:
                        if input_evidence is None:
                            raise ValueError(
                                f"No resource calibration mapping or immutable input evidence for {task.id}"
                            )
                        entry = MechanicalOracleTaskReceipt(
                            task_sha256=canonical_sha256(task.model_dump(mode="json")),
                            runtime_sha256=row_runtime_hash(row),
                            has_oracle=oracle,
                            exclusion_reason=OracleExclusionReason.EXECUTABILITY_PROOF_MISSING,
                            evidence=(input_evidence,),
                        )
                    proof = OracleFilterReceipt(
                        recipe_identity=None,
                        input_cohort_sha256=cohort_hash,
                        input_rows=1,
                        conversion_rejections=0,
                        tasks={task.id: entry},
                    )
                    runtime = row_runtime(row)
                    assert not proof.admits(task, runtime)
                    entries[task.id] = entry
                    excluded.append(row)
                    continue
                original_row = row
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
                history = tuple(
                    InfrastructureRetryTransition.model_validate(value)
                    for value in mapping.get("infrastructure_retries", [])
                )
                require_retry_provenance(task.id, history, parent_maps, retry_plans)
                phases = [*(transition.parent_calibration for transition in history), calibration]
                for attempt in (attempt for phase in phases for attempt in phase.attempts):
                    key = (attempt.result["task_name"], attempt.result["task_checksum"])
                    if attempt.prior_phase_failure is not None:
                        require_prior_phase_failure(attempt.prior_phase_failure, attempt.result)
                    collected_attempt = attempt.model_copy(
                        update={"infrastructure_retry": None, "prior_phase_failure": None}
                    )
                    if collected_attempt not in trials.get(key, []):
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
                source_setup = (
                    tomllib.loads((original / "task.toml").read_text())
                    .get("environment", {})
                    .get("setup", {})
                    .get("command", "")
                )
                incompatibility = source_environment_incompatibility(calibration.attempts[-1], source_setup)
                final = calibration.attempts[-1]
                control = controls.get(task.id)
                policy = policies.get(task.id)
                if policy is not None and (
                    policy.source_task_sha256 != mapping["task_sha256"]
                    or policy.source_task_checksum != mapping.get("source_task_checksum", checksum)
                    or policy.runtime_sha256 != row_runtime_hash(original_row)
                    or policy.input_cohort_sha256 != cohort_hash
                ):
                    raise ValueError(f"Native success policy differs from the frozen source contract: {task.id}")
                oracle_passed = policy.passes(final, checksum) if policy is not None else calibration.passed
                if oracle_passed or (not oracle and control is not None and control.passed and control.trial == final):
                    reason = None
                else:
                    reason = oracle_failure_reason(final, incompatibility)
                if reason is None and not trial_network_qualified(task, row_runtime(row), final):
                    reason = OracleExclusionReason.NETWORK_POLICY_UNSUPPORTED
                if (
                    not oracle
                    and reason in {None, OracleExclusionReason.ORACLE_FAILED}
                    and not (control is not None and control.passed and control.trial == final)
                ):
                    reason = (
                        OracleExclusionReason.EXECUTABILITY_PROOF_MISSING
                        if control is None
                        else OracleExclusionReason.EXECUTABILITY_RUN_FAILED
                    )
                applicability = negative_control_applicability(task) if oracle else None
                negative_control = negative.get(task.id)
                if reason is None and applicability is NegativeControlApplicability.WORKSPACE:
                    if negative_control is None or (
                        not negative_control.conclusive and negative_control.annotation is None
                    ):
                        reason = OracleExclusionReason.NEGATIVE_CONTROL_PENDING
                    elif negative_control.passed(policy):
                        reason = OracleExclusionReason.NEGATIVE_CONTROL_PASSED
                entry = OracleTaskReceipt(
                    task_sha256=canonical_sha256(task.model_dump(mode="json")),
                    runtime_sha256=row_runtime_hash(row),
                    harbor_task_checksum=checksum,
                    source_task_sha256=mapping["task_sha256"],
                    source_task_checksum=mapping.get("source_task_checksum", checksum),
                    native_success_policy=policy,
                    negative_control=negative_control,
                    negative_control_applicability=applicability,
                    has_oracle=oracle,
                    trials=(calibration.attempts[-1],) if oracle else (),
                    calibration=calibration,
                    executability_run=control,
                    exclusion_reason=reason,
                    source_environment_incompatibility=incompatibility,
                    infrastructure_retries=history,
                )
                entries[task.id] = entry
                if entry.passed:
                    kept.append(row)
                else:
                    excluded.append(original_row)
    extra = set(mappings) - set(entries)
    if extra:
        raise ValueError(f"Task map contains unknown TaskSpecs: {sorted(extra)[:5]}")
    extra_controls = set(controls) - {task_id for task_id, entry in entries.items() if not entry.has_oracle}
    if extra_controls:
        raise ValueError(f"Executability controls do not belong to no-oracle tasks: {sorted(extra_controls)[:5]}")
    if set(mechanical) - entries.keys():
        raise ValueError("Mechanical outcomes contain tasks outside the frozen cohort")
    manifest = json.loads(conversion_manifest.read_text()) if conversion_manifest is not None else None
    if manifest is not None and manifest["converted_rows"] != len(entries):
        raise ValueError("Converted cohort does not match the full conversion manifest")
    input_rows = (
        manifest["input_rows"] - manifest["within_source_dedup"]["within_source_duplicates"]
        if manifest
        else len(entries)
    )
    receipt = OracleFilterReceipt(
        recipe_identity=manifest["oracle_recipe_identity"] if manifest else None,
        input_cohort_sha256=cohort_hash,
        input_rows=input_rows,
        conversion_rejections=input_rows - len(entries),
        tasks=entries,
    )
    pq.write_table(pa.Table.from_pylist(kept, schema=schema), output)
    excluded_path = output.parent / "excluded-tasks.parquet"
    pq.write_table(pa.Table.from_pylist(excluded, schema=schema), excluded_path)
    receipt_path = output.parent / "oracle-filter.json"
    receipt_path.write_text(receipt.model_dump_json(indent=2) + "\n")
    summary = {
        "input_rows": input_rows,
        "converted_rows": len(entries),
        "oracle_tasks": sum(e.has_oracle for e in entries.values()),
        "oracle_passed": sum(e.has_oracle and e.passed for e in entries.values()),
        "oracle_excluded": sum(e.has_oracle and not e.passed for e in entries.values()),
        "infrastructure_pending": sum(
            e.exclusion_reason is OracleExclusionReason.INFRASTRUCTURE for e in entries.values()
        ),
        "exclusions_by_reason": dict(Counter(e.exclusion_reason.value for e in entries.values() if e.exclusion_reason)),
        "output_rows": len(kept),
        "receipt": str(receipt_path),
        "output_sha256": file_sha256(StoragePath(str(output))),
        "excluded_archive": str(excluded_path),
        "excluded_archive_sha256": file_sha256(StoragePath(str(excluded_path))),
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
        serialized = task.model_dump_json()
        row["task_json" if "task_json" in row else "task_spec"] = serialized
        if "task_sha256" in row:
            row["task_sha256"] = hashlib.sha256(serialized.encode()).hexdigest()
    return row


def calibrated_task(
    task: TaskSpec,
    mapping: dict,
    destination: Path,
    environment: str,
    environment_kwargs: dict,
    calibration_solutions: Path | None,
    retry_context: InfrastructureRetryContext | None = None,
    prepare_attempt: Callable[[Path, ExecutionRequirements], None] | None = None,
    initial_requirements: ExecutionRequirements | None = None,
    source_runtime: dict | None = None,
    host_admission: LocalHostAdmission | None = None,
) -> dict:
    """Run the production Trial path once per shape, creating a fresh task package each time."""
    original = Path(mapping["task_path"])
    if host_admission is not None and environment != "local":
        raise ValueError("Physical-host admission applies only to LOCAL trials")
    configured_runtime = None
    if source_runtime is not None and (original / TASK_SPEC_FILE).is_file():
        configured_runtime = TaskSpecRuntimeBinding(
            task_sha256=canonical_sha256(task.model_dump(mode="json")),
            runtime_sha256=canonical_sha256(source_runtime),
            runtime=source_runtime,
        )
        if configured_runtime.runtime_sha256 != mapping.get("runtime_sha256"):
            raise ValueError(f"Configured runtime changed after Harbor export: {task.id}")
    for label, read_binding in (
        ("OracleCommand", read_oracle_command),
        ("Protected", read_runtime_binding),
        ("Source", read_source_runtime),
    ):
        binding = read_binding(original)
        if binding is None:
            continue
        binding.require_task(task)
        if binding.runtime_sha256 != mapping.get("runtime_sha256"):
            raise ValueError(f"{label} runtime changed after Harbor export: {task.id}")
    native = harbor_inspection("tasks", [str(original)])[str(original)]
    source_checksum = mapping.get("source_task_checksum", mapping["task_checksum"])
    if native["task_checksum"] != source_checksum:
        raise ValueError(f"Task changed after Harbor export: {task.id}")
    declared = task.execution_requirements
    preserved = declared_native_requirements(original)
    verifier = native_verifier_profile(original)
    shape = (
        declared
        or initial_requirements
        or ExecutionRequirements.model_validate({**INITIAL_REQUIREMENTS.model_dump(), **preserved})
    )
    if declared is not None and native_requirements(original) != declared:
        raise ValueError(f"Harbor package differs from declared execution requirements: {task.id}")
    attempts = []
    infrastructure_attempts = 0
    host_backoff = ExponentialBackoff(initial=1, maximum=120, factor=2, jitter=0.5)
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
                if (package / TASK_SPEC_FILE).is_file():
                    document["environment"][name] = shape.model_dump()[name]
                else:
                    document.setdefault("environment", {}).setdefault(name, shape.model_dump()[name])
                if "environment" in document.get("verifier", {}):
                    document["verifier"]["environment"].setdefault(name, shape.model_dump()[name])
            for phase in ("agent", "verifier"):
                if (package / TASK_SPEC_FILE).is_file():
                    document[phase]["timeout_sec"] = shape.model_dump()[f"{phase}_timeout_sec"]
                else:
                    document.setdefault(phase, {}).setdefault("timeout_sec", shape.model_dump()[f"{phase}_timeout_sec"])
            (package / "task.toml").write_text(tomlkit.dumps(document))
        if prepare_attempt is not None:
            prepare_attempt(package, shape)
        config = {
            "task": {
                "path": str(package),
                **({"source_runtime": configured_runtime.model_dump(mode="json")} if configured_runtime else {}),
            },
            "agent": {"name": "oracle", "kwargs": {"resource_usage_filename": "resource-usage.json"}},
            "environment": {"type": environment, "kwargs": environment_kwargs},
            "trials_dir": str(attempt / "jobs"),
            "trial_name": "oracle",
            "trial_attempt_timeout_sec": CALIBRATION_ATTEMPT_TIMEOUT,
        }
        config_path = attempt / "trial.json"
        config_path.write_text(json.dumps(config))
        admission = (
            host_admission.claim(task.id, deadline=time.monotonic() + CALIBRATION_ATTEMPT_TIMEOUT)
            if host_admission is not None
            else nullcontext()
        )
        with admission as slot, (attempt / "harbor.log").open("w") as log:
            if slot is not None:
                assert host_admission is not None
                (attempt / "host-admission.json").write_text(
                    json.dumps(
                        {
                            "boot_id": str(host_admission.boot_id),
                            "node_id": host_admission.node_id,
                            "control": slot.control.path,
                            "max_trials_per_host": host_admission.config.max_trials_per_host,
                            "claim": slot.record.model_dump(mode="json"),
                        }
                    )
                    + "\n"
                )
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
            trials = [trial for group in collect_trials(attempt / "jobs").values() for trial in group]
            if len(trials) != 1:
                raise ValueError(f"Expected exactly one persisted production trial for {task.id}")
            trial = trials[0]
            if slot is not None:
                if trial.result.get("completion") is None:
                    raise RuntimeError(f"LOCAL resource cleanup is unconfirmed; host claim retained: {task.id}")
                slot.release()
        if retry_context is not None:
            trial = trial.model_copy(update={"infrastructure_retry": retry_context})
        attempts.append(trial)
        if trial.failure_signal is FailureSignal.INFRASTRUCTURE:
            infrastructure_attempts += 1
            if infrastructure_attempts < 3:
                if pre_oracle_host_unavailable(trial.result, trial.verdict, trial.oracle_exit_code):
                    time.sleep(host_backoff.next_interval())
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
        initial_requirements=initial_requirements,
        verifier=verifier,
        oracle_kind="original" if has_oracle_solution(task) else "calibration_only",
        attempts=tuple(attempts),
    )
    final = attempts[-1].result
    result = {
        **mapping,
        "task_name": final["task_name"],
        "task_checksum": final["task_checksum"],
        "source_task_checksum": source_checksum,
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
    retry_contexts: dict[str, InfrastructureRetryContext] | None = None,
    prepare_attempt: Callable[[Path, ExecutionRequirements], None] | None = None,
    initial_requirements: dict[str, ExecutionRequirements] | None = None,
    host_admission: LocalHostAdmission | None = None,
) -> None:
    """Fold calibration into full-population oracle filtering; collect the same trials afterward."""
    mappings = json.loads(task_map.read_text())
    tasks = []
    runtimes = {}
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
                runtimes[task.id] = row_runtime(row)
    if len({task.id for task in tasks}) != len(tasks) or set(mappings) != {task.id for task in tasks}:
        raise ValueError("Calibration task map must cover exactly the full converted population")
    if retry_contexts is not None and retry_contexts.keys() != mappings.keys():
        raise ValueError("Infrastructure phase context must cover exactly the approved partition")
    if initial_requirements is not None and (retry_contexts is None or initial_requirements.keys() != mappings.keys()):
        raise ValueError("Resumed allocations must cover the approved retry partition")
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
                None if retry_contexts is None else retry_contexts[task.id],
                prepare_attempt,
                initial_requirements=None if initial_requirements is None else initial_requirements[task.id],
                source_runtime=runtimes[task.id],
                host_admission=host_admission,
            )
            for task in tasks
        }
        calibrated = {task_id: future.result() for task_id, future in futures.items()}
    (destination / "task-map.json").write_text(json.dumps(calibrated, indent=2) + "\n")


def backend_code_files(root: Path) -> dict[str, str]:
    """Fingerprint the locked Harbor runtime and its maintained source dependencies."""
    selected = {}
    for path in collect_workspace_files(root):
        relative = path.relative_to(root).as_posix()
        if relative in BACKEND_CODE_FILES or relative.startswith(BACKEND_CODE_ROOTS):
            selected[relative] = file_sha256(StoragePath(str(path)))
    return selected


def write_backend_fix(revision: str, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    if actual != revision:
        raise ValueError("Backend fix revision must be the current committed checkout")
    changed = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *BACKEND_CODE_ROOTS,
            *sorted(BACKEND_CODE_FILES),
        ],
        cwd=PROJECT_ROOT,
        text=True,
    )
    if changed.strip():
        raise ValueError("Commit the backend code before freezing a retry phase")
    identity = BackendFixIdentity(revision=revision, files_sha256=backend_code_files(PROJECT_ROOT))
    output.write_text(identity.model_dump_json(indent=2) + "\n")


def require_backend_fix(identity: BackendFixIdentity) -> None:
    if backend_code_files(PROJECT_ROOT) != identity.files_sha256:
        raise ValueError("Runtime source files differ from the pinned backend fix")


def write_retry_plan(
    cohort: Path,
    task_map: Path,
    approvals_path: Path,
    backend_fix_path: Path,
    output: Path,
    local_host: LocalExecutionHostTransition | None = None,
) -> None:
    """Freeze only explicitly approved infrastructure cases; never change their native packages."""
    if output.exists():
        raise FileExistsError(output)
    backend_fix = BackendFixIdentity.model_validate_json(backend_fix_path.read_text())
    require_backend_fix(backend_fix)
    approvals = {
        key: InfrastructureRetryApproval.model_validate(value)
        for key, value in json.loads(approvals_path.read_text()).items()
    }
    if not approvals:
        raise ValueError("An infrastructure retry requires explicit per-task approvals")
    mappings = json.loads(task_map.read_text())
    paths = sorted(cohort.glob("*.parquet")) if cohort.is_dir() else [cohort]
    schema = pq.read_schema(paths[0])
    rows = []
    selected = {}
    environment = None
    environment_kwargs = None
    for path in paths:
        if pq.read_schema(path) != schema:
            raise ValueError("Retry input cohort shards have different schemas")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                task = row_task(row)
                if task is None or task.id not in approvals:
                    continue
                if task.id in selected:
                    raise ValueError(f"Duplicate retry task: {task.id}")
                mapping = mappings[task.id]
                if mapping["task_sha256"] != canonical_sha256(task.model_dump(mode="json")) or mapping.get(
                    "runtime_sha256"
                ) != row_runtime_hash(row):
                    raise ValueError(f"Retry input task changed: {task.id}")
                calibration = ResourceCalibrationReceipt.model_validate(mapping["calibration"])
                require_infrastructure_retry_parent(calibration, approvals[task.id])
                if not has_oracle_solution(task) or task.execution_requirements != calibration.declared_requirements:
                    raise ValueError(f"Retry changed the original oracle or its declared execution limits: {task.id}")
                native = Path(mapping["task_path"])
                setup = (
                    tomllib.loads((native / "task.toml").read_text())
                    .get("environment", {})
                    .get("setup", {})
                    .get("command", "")
                )
                if source_environment_incompatibility(calibration.attempts[-1], setup) is not None:
                    raise ValueError(f"Source/grader incompatibility is not our infrastructure: {task.id}")
                expected_checksum = mapping.get("source_task_checksum", mapping["task_checksum"])
                measured = harbor_inspection("tasks", [str(native)])[str(native)]["task_checksum"]
                if measured != expected_checksum:
                    raise ValueError(f"Retry native files or environment changed: {task.id}")
                final = calibration.attempts[-1]
                if canonical_sha256(json.loads(Path(final.path).read_text())) != final.result_sha256:
                    raise ValueError(f"Retry parent result changed: {task.id}")
                final_package = Path(final.result["config"]["task"]["path"])
                final_checksum = harbor_inspection("tasks", [str(final_package)])[str(final_package)]["task_checksum"]
                if final_checksum != final.result["task_checksum"]:
                    raise ValueError(f"Retry parent package changed: {task.id}")
                if calibration_package_identity(final_package, "original") != calibration_package_identity(
                    native, "original"
                ):
                    raise ValueError(f"Retry parent changed source files or grader: {task.id}")
                config = calibration.attempts[-1].result["config"]["environment"]
                current = (config["type"], config.get("kwargs", {}))
                if environment is not None and current != (environment, environment_kwargs):
                    raise ValueError("One retry partition must use the same original backend configuration")
                environment, environment_kwargs = current
                rows.append(row)
                selected[task.id] = mapping
    if set(selected) != set(approvals):
        raise ValueError("Retry approvals must identify exactly the frozen task partition")
    assert environment_kwargs is not None
    if local_host is not None:
        if environment != "local":
            raise ValueError("Physical LOCAL host transitions require the LOCAL backend")
        physical = {key: value for key, value in environment_kwargs.items() if key in LOCAL_EXECUTION_HOST_FIELDS}
        if physical != local_host.parent_kwargs:
            raise ValueError("Physical host transition belongs to another parent worker")
        environment_kwargs = {
            **{key: value for key, value in environment_kwargs.items() if key not in LOCAL_EXECUTION_HOST_FIELDS},
            **local_host.current_kwargs,
        }
    output.mkdir(parents=True, exist_ok=False)
    frozen_cohort, frozen_map = output / "cohort.parquet", output / "parent-map.json"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), frozen_cohort)
    frozen_map.write_text(json.dumps(selected, indent=2) + "\n")
    plan = InfrastructureRetryPlan(
        cohort_path=str(frozen_cohort.resolve()),
        cohort_sha256=file_sha256(StoragePath(str(frozen_cohort))),
        parent_map_path=str(frozen_map.resolve()),
        parent_map_sha256=file_sha256(StoragePath(str(frozen_map))),
        source_parent_map_path=str(task_map.resolve()),
        source_parent_map_sha256=file_sha256(StoragePath(str(task_map))),
        backend_fix=backend_fix,
        environment=environment,
        environment_kwargs=environment_kwargs,
        approvals=approvals,
        local_host=local_host,
    )
    (output / "plan.json").write_text(plan.model_dump_json(indent=2) + "\n")


def require_current_local_host(transition: LocalExecutionHostTransition) -> None:
    """Verify immutable provenance and current worker containment before retry dispatch."""
    for evidence in transition.scope_evidence:
        if file_sha256(StoragePath(evidence.uri)) != evidence.sha256:
            raise ValueError("LOCAL worker scope evidence changed after approval")
    cgroup = transition.current_kwargs["cgroup_parent"]
    assert cgroup is not None
    current = Path(cgroup).resolve()
    delegated = delegated_cgroup_parent()
    if delegated is None or not current.is_relative_to(delegated):
        raise ValueError("Replacement cgroup is outside the current worker's own delegated scope")
    check_cgroup_parent(current)
    if not {"cpu", "memory"}.issubset((current / "cgroup.controllers").read_text().split()):
        raise ValueError("Replacement LOCAL scope does not delegate CPU and memory controllers")


def retry_cohort(
    plan_path: Path,
    destination: Path,
    concurrency: int,
    prepare_attempt: Callable[[Path, ExecutionRequirements], None] | None = None,
    host_admission: LocalHostAdmission | None = None,
) -> None:
    """Run one approved fixed-backend phase using the normal three-attempt infrastructure budget."""
    if destination.exists():
        raise FileExistsError(destination)
    plan = InfrastructureRetryPlan.model_validate_json(plan_path.read_text())
    require_backend_fix(plan.backend_fix)
    if plan.local_host is not None:
        require_current_local_host(plan.local_host)
    cohort, parent_map = Path(plan.cohort_path), Path(plan.parent_map_path)
    for path, expected in (
        (cohort, plan.cohort_sha256),
        (parent_map, plan.parent_map_sha256),
        (Path(plan.source_parent_map_path), plan.source_parent_map_sha256),
    ):
        if file_sha256(StoragePath(str(path))) != expected:
            raise ValueError(f"Retry phase input changed: {path}")
    parents = json.loads(parent_map.read_text())
    if parents.keys() != plan.approvals.keys():
        raise ValueError("Retry plan does not cover its exact parent partition")
    transitions = {}
    for task_id, parent in parents.items():
        native = Path(parent["task_path"])
        checksum = harbor_inspection("tasks", [str(native)])[str(native)]["task_checksum"]
        if checksum != parent.get("source_task_checksum", parent["task_checksum"]):
            raise ValueError(f"Retry native task changed after approval: {task_id}")
        transitions[task_id] = InfrastructureRetryTransition(
            task_sha256=parent["task_sha256"],
            runtime_sha256=parent.get("runtime_sha256"),
            source_task_checksum=checksum,
            plan_path=str(plan_path.resolve()),
            plan_sha256=file_sha256(StoragePath(str(plan_path))),
            parent_map_path=str(parent_map),
            parent_map_sha256=plan.parent_map_sha256,
            parent_mapping_sha256=canonical_sha256(parent),
            parent_history_sha256=canonical_sha256({"phases": parent.get("infrastructure_retries", [])}),
            parent_calibration=ResourceCalibrationReceipt.model_validate(parent["calibration"]),
            backend_fix=plan.backend_fix,
            approval=plan.approvals[task_id],
            local_host=plan.local_host,
        )
    phase = destination / "phase"
    contexts = {
        task_id: InfrastructureRetryContext(
            parent_result_sha256=approval.parent_result_sha256,
            decision_ref=approval.decision_ref,
            backend_fix_revision=plan.backend_fix.revision,
        )
        for task_id, approval in plan.approvals.items()
    }
    calibrate_cohort(
        cohort,
        parent_map,
        phase,
        plan.environment,
        plan.environment_kwargs,
        concurrency,
        retry_contexts=contexts,
        prepare_attempt=prepare_attempt,
        host_admission=host_admission,
        initial_requirements={
            task_id: ResourceCalibrationReceipt.model_validate(parent["calibration"]).execution_requirements
            for task_id, parent in parents.items()
        },
    )
    require_backend_fix(plan.backend_fix)
    current = json.loads((phase / "task-map.json").read_text())
    for task_id, mapping in current.items():
        mapping["infrastructure_retries"] = [
            *parents[task_id].get("infrastructure_retries", []),
            transitions[task_id].model_dump(mode="json"),
        ]
    (destination / "task-map.json").write_text(json.dumps(current, indent=2) + "\n")


def merge_retry_maps(original: Path, retries: list[Path], output: Path) -> None:
    """Replace a task only when a retry extends that exact parent; retain every phase in its receipt."""
    if output.exists():
        raise FileExistsError(output)
    merged = json.loads(original.read_text())
    for path in retries:
        for task_id, mapping in json.loads(path.read_text()).items():
            parent = merged[task_id]
            history = mapping.get("infrastructure_retries", [])
            if len(history) != len(parent.get("infrastructure_retries", [])) + 1 or history[:-1] != parent.get(
                "infrastructure_retries", []
            ):
                raise ValueError(f"Retry does not extend the current parent history: {task_id}")
            transition = InfrastructureRetryTransition.model_validate(history[-1])
            if transition.parent_mapping_sha256 != canonical_sha256(parent):
                raise ValueError(f"Retry belongs to another parent mapping: {task_id}")
            for name in ("task_path", "task_sha256", "runtime_sha256"):
                if mapping.get(name) != parent.get(name):
                    raise ValueError(f"Retry changed task/environment binding {name}: {task_id}")
            if mapping.get("source_task_checksum", mapping["task_checksum"]) != parent.get(
                "source_task_checksum", parent["task_checksum"]
            ):
                raise ValueError(f"Retry changed source task/environment checksum: {task_id}")
            merged[task_id] = mapping
    output.write_text(json.dumps(merged, indent=2) + "\n")


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
    collect.add_argument("--executability-runs", type=Path, help="strong-model production-control JSONL receipts")
    collect.add_argument("--mechanical-outcomes", type=Path, help="typed per-task mechanical outcome JSONL")
    collect.add_argument("--input-evidence-uri", help="immutable cohort URI for tasks with no execution proof")
    collect.add_argument("--native-success-policies", type=Path)
    collect.add_argument("--negative-controls", type=Path)
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--cohort", type=Path, required=True)
    calibrate.add_argument("--task-map", type=Path, required=True)
    calibrate.add_argument("--output-dir", type=Path, required=True)
    calibrate.add_argument("--environment", choices=("local", "daytona", "iris-sandbox"), required=True)
    calibrate.add_argument("--environment-kwargs", type=json.loads, default={})
    calibrate.add_argument("--concurrency", type=int, required=True)
    calibrate.add_argument("--calibration-solutions", type=Path)
    calibrate.add_argument("--host-admission", type=Path)
    calibrate.add_argument("--host-worker-id")
    calibrate.add_argument("--host-node-id")
    backend = commands.add_parser("backend-fix")
    backend.add_argument("--revision", required=True)
    backend.add_argument("--output", type=Path, required=True)
    plan = commands.add_parser("retry-plan")
    plan.add_argument("--cohort", type=Path, required=True)
    plan.add_argument("--task-map", type=Path, required=True)
    plan.add_argument("--approvals", type=Path, required=True)
    plan.add_argument("--backend-fix", type=Path, required=True)
    plan.add_argument("--local-host", type=Path, help="approved physical LOCAL host transition JSON")
    plan.add_argument("--output-dir", type=Path, required=True)
    retry = commands.add_parser("retry")
    retry.add_argument("--plan", type=Path, required=True)
    retry.add_argument("--output-dir", type=Path, required=True)
    retry.add_argument("--concurrency", type=int, required=True)
    retry.add_argument("--host-admission", type=Path)
    retry.add_argument("--host-worker-id")
    retry.add_argument("--host-node-id")
    merge = commands.add_parser("merge-retries")
    merge.add_argument("--task-map", type=Path, required=True)
    merge.add_argument("--retry-map", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    bind = commands.add_parser("bind-phase-failures")
    bind.add_argument("--task-map", type=Path, required=True)
    bind.add_argument("--evidence", type=Path, required=True, help="JSON list of PriorPhaseFailureReceipt objects")
    bind.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    host_admission = None
    if args.command in {"calibrate", "retry"}:
        if args.host_admission is None and (args.host_worker_id is not None or args.host_node_id is not None):
            parser.error("Host identities require --host-admission")
        if args.host_admission is not None:
            if args.host_worker_id is None or args.host_node_id is None:
                parser.error("Host admission requires --host-worker-id and --host-node-id")
            host_admission = LocalHostAdmission.current(
                LocalHostAdmissionConfig.model_validate_json(args.host_admission.read_text()),
                args.host_node_id,
                args.host_worker_id,
            )
    if args.command == "bind-phase-failures":
        evidence = [PriorPhaseFailureReceipt.model_validate(item) for item in json.loads(args.evidence.read_text())]
        failures = {item.result_sha256: item for item in evidence}
        if len(failures) != len(evidence):
            raise ValueError("Duplicate phase-failure evidence for the same production Trial")
        bind_prior_phase_failures(args.task_map, failures, args.output)
        return
    if args.command == "backend-fix":
        write_backend_fix(args.revision, args.output)
        return
    if args.command == "retry-plan":
        write_retry_plan(
            args.cohort,
            args.task_map,
            args.approvals,
            args.backend_fix,
            args.output_dir,
            LocalExecutionHostTransition.model_validate_json(args.local_host.read_text()) if args.local_host else None,
        )
        return
    if args.command == "retry":
        retry_cohort(args.plan, args.output_dir, args.concurrency, host_admission=host_admission)
        return
    if args.command == "merge-retries":
        merge_retry_maps(args.task_map, args.retry_map, args.output)
        return
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
            host_admission=host_admission,
        )
        return
    filter_cohort(
        args.cohort,
        args.task_map,
        args.jobs_dir,
        args.output,
        args.conversion_manifest,
        args.executability_runs,
        args.mechanical_outcomes,
        (
            ContentEvidenceReference(uri=args.input_evidence_uri, sha256=cohort_sha256(args.cohort))
            if args.input_evidence_uri
            else None
        ),
        args.native_success_policies,
        args.negative_controls,
    )


if __name__ == "__main__":
    main()
