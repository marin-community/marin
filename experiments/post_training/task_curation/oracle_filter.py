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
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from itertools import islice
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
from taskcompendium.execution_applicability import is_nonagentic_task
from taskcompendium.grading_environment import GradingEnvironmentMode
from taskcompendium.harbor.adapter import TASK_SPEC_FILE, declared_execution_limits
from taskcompendium.harbor.archive_paths import resolve_archive_path
from taskcompendium.harbor.execution import native_execution_limits
from taskcompendium.harbor.native_network import ReconstructedNativeNetworkPolicy, verify_native_network_canary
from taskcompendium.harbor.oracle import read_oracle_command
from taskcompendium.harbor.protected_runtime import read_runtime_binding
from taskcompendium.harbor.runtime import TaskNetworkPolicyReceipt, TaskSpecRuntimeBinding, read_source_runtime
from taskcompendium.identity import canonical_sha256, task_spec_json, task_spec_sha256
from taskcompendium.models import ExecutionRequirements, TaskSpec, decode_json_value
from taskcompendium.pipeline.legacy_producer_1485 import LegacyProducer1485QualificationReceipt
from taskcompendium.pipeline.legacy_roles import LegacyRoleQualificationReceipt
from taskcompendium.pipeline.oracle_filter import (
    LOCAL_EXECUTION_HOST_FIELDS,
    BackendFixIdentity,
    ExecutabilityRunReceipt,
    InfrastructureRetryApproval,
    InfrastructureRetryContext,
    InfrastructureRetryPlan,
    InfrastructureRetryTransition,
    LegacyVerifierInitialization,
    LocalExecutionHostTransition,
    MechanicalOracleTaskReceipt,
    NativeOracleSuccessPolicy,
    NativeVerifierGrade,
    NegativeControlApplicability,
    NegativeControlPolicy,
    OracleExclusionReason,
    OracleFilterReceipt,
    OracleNegativeControlReceipt,
    OracleTaskReceipt,
    OracleTrialReceipt,
    PostGradeTeardownEvidence,
    PriorPhaseFailureReceipt,
    RecollectionWaiver,
    ResourceCalibrationReceipt,
    SourcePytestEnvironment,
    SourceVerifierError,
    VerifierResourceProfile,
    initial_verifier_requirements,
    initialized_verifier_requirements,
    negative_control_applicability,
    next_trial_allocations,
    oracle_failure_reason,
    pre_oracle_host_unavailable,
    recovered_source_runtime_requirements,
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
from taskcompendium.runtime.resources import has_oracle_solution

from experiments.post_training.task_curation.calibration_projection import calibrated_task_spec
from experiments.post_training.task_curation.harbor_inspection import harbor_inspection
from experiments.post_training.task_curation.legacy_producer_1485_qualification import (
    LegacyProducer1485QualificationEvidence,
    legacy_producer_1485_qualifications,
)
from experiments.post_training.task_curation.legacy_role_qualification import (
    LegacyRoleQualificationEvidence,
    legacy_role_qualifications,
)
from experiments.post_training.task_curation.local_host_admission import LocalHostAdmission, LocalHostAdmissionConfig
from experiments.post_training.task_curation.negative_controls import require_negative_control
from experiments.post_training.task_curation.oracle_bootstrap_failure import require_bootstrap_failure
from experiments.post_training.task_curation.post_grade_teardowns import (
    bind_post_grade_teardowns,
    require_post_grade_teardown,
)
from experiments.post_training.task_curation.prior_phase_failures import (
    bind_prior_phase_failures,
    require_prior_phase_failure,
)
from experiments.post_training.task_curation.source_verifier_errors import require_source_verifier_error

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
    "experiments/post_training/task_curation/post_grade_teardowns.py",
    "experiments/post_training/task_curation/archive_evidence.py",
    "experiments/post_training/task_curation/oracle_bootstrap_failure.py",
    "experiments/post_training/task_curation/negative_controls.py",
    "experiments/post_training/task_curation/harbor_inspection.py",
    "experiments/post_training/task_curation/calibration_projection.py",
    "experiments/post_training/task_curation/legacy_role_qualification.py",
    "experiments/post_training/task_curation/legacy_producer_1485_qualification.py",
}
FAILURE_LOG_BYTES = 64 * 1024
CALIBRATION_ATTEMPT_TIMEOUT = 72 * 60 * 60


def declared_native_requirements(
    path: Path, source_runtime: TaskSpecRuntimeBinding | None = None
) -> dict[ResourceDimension, float]:
    if (path / TASK_SPEC_FILE).is_file():
        task = TaskSpec.model_validate_json((path / TASK_SPEC_FILE).read_text())
        declared = declared_execution_limits(task)
        if source_runtime is not None:
            source_runtime.require_task(task)
    else:
        document = tomllib.loads((path / "task.toml").read_text())
        declared = native_execution_limits(document).declared
    for name, value in source_runtime.resource_limits("task_machine").items() if source_runtime else ():
        if name in declared and declared[name] != value:
            raise ValueError(f"Source runtime actor {name} conflicts with the original task declaration")
        declared[name] = value
    return {ResourceDimension(name): value for name, value in declared.items()}


def native_requirements(path: Path) -> ExecutionRequirements:
    from harbor_config.models.task.config import EnvironmentConfig as NativeEnvironmentConfig  # noqa: PLC0415

    document = tomllib.loads((path / "task.toml").read_text())
    environment = NativeEnvironmentConfig.model_validate(document["environment"])
    return ExecutionRequirements(
        cpus=environment.cpus,
        memory_mb=environment.memory_mb,
        storage_mb=environment.storage_mb,
        agent_timeout_sec=document["agent"]["timeout_sec"],
        verifier_timeout_sec=document["verifier"]["timeout_sec"],
    )


def native_verifier_profile(path: Path, source_runtime: TaskSpecRuntimeBinding | None = None) -> VerifierResourceProfile:
    verifier = tomllib.loads((path / "task.toml").read_text()).get("verifier", {})
    mode = verifier.get("environment_mode") or ("separate" if "environment" in verifier else "shared")
    environment = verifier.get("environment", {})
    fixed = {
        ResourceDimension(name): environment[name]
        for name in ("cpus", "memory_mb", "storage_mb")
        if environment.get(name) is not None
    }
    specification = path / TASK_SPEC_FILE
    if specification.is_file():
        declared = TaskSpec.model_validate_json(specification.read_text()).verifier_execution_requirements
        if declared is not None:
            for name in ("cpus", "memory_mb", "storage_mb"):
                value = getattr(declared, name)
                if name in fixed and fixed[name] != value:
                    raise ValueError(f"Published verifier {name} conflicts with its original grader declaration")
                fixed[ResourceDimension(name)] = value
    for name, value in source_runtime.resource_limits("verifier_machine").items() if source_runtime else ():
        dimension = ResourceDimension(name)
        if dimension in fixed and fixed[dimension] != value:
            raise ValueError(f"Source runtime verifier {name} conflicts with the original grader declaration")
        fixed[dimension] = value
    return VerifierResourceProfile(mode=GradingEnvironmentMode(mode), fixed=fixed)


def source_runtime_binding(task: TaskSpec, runtime: dict | None) -> TaskSpecRuntimeBinding | None:
    """Validate original role declarations once against their exact frozen TaskSpec."""
    if runtime is None:
        return None
    binding = TaskSpecRuntimeBinding(
        task_sha256=task_spec_sha256(task),
        runtime_sha256=canonical_sha256(runtime),
        runtime=runtime,
    )
    binding.require_task(task)
    return binding


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
    *,
    recollection: RecollectionWaiver | None = None,
    archive_roots: Mapping[str, str] | None = None,
) -> OracleTrialReceipt:
    """Read a trial's evidence against its measured native checksum."""
    recorded_path = path
    roots = archive_roots or {}
    path = resolve_archive_path(path, roots)
    native = resolve_archive_path(Path(document["config"]["task"]["path"]), roots)
    if recollection is None:
        if native_checksum != document["task_checksum"]:
            raise ValueError(f"Native task changed after its oracle trial: {native}")
    else:
        # Typed re-collection (ruling 2026-11-19): the sealed trial document binds the
        # campaign-era package checksum, which the preserved rebuilt package cannot
        # reproduce; the binding of record is the waiver's sealed, control-validated
        # checksum, accepted only for the trial it names.
        sealed = recollection.binding(document["task_name"])
        if document["task_checksum"] != sealed.package_checksum:
            raise ValueError(f"Re-collected trial differs from its sealed checksum binding: {native}")
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
        path=str(recorded_path),
        result=document,
        verdict=verdict,
        oracle_exit_code=int(exit_path.read_text()) if exit_path.exists() else None,
        result_sha256=canonical_sha256(document),
        execution_requirements=native_requirements(native),
        verifier_execution_requirements=(
            ExecutionRequirements.model_validate(document["config"]["task"]["verifier_execution_requirements"])
            if document["config"].get("task", {}).get("verifier_execution_requirements") is not None
            else None
        ),
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


def require_recollection_binding(task: TaskSpec, mapping: Mapping, recollection: RecollectionWaiver) -> None:
    """Typed re-collection: the mapping must bind exactly the sealed entry's content hashes.

    The rebuilt package is accepted only when its mapping carries the sealed entry's task
    and runtime content hashes and the sealed, control-validated package checksum (ruling
    2026-11-19, option a); the rebuild's own measured checksum travels separately as
    ``source_task_checksum``.
    """
    binding = recollection.binding(task.id)
    if (
        mapping["task_sha256"] != binding.task_sha256
        or mapping.get("runtime_sha256") != binding.runtime_sha256
        or mapping["task_checksum"] != binding.package_checksum
    ):
        raise ValueError(f"Re-collection mapping differs from its sealed binding: {task.id}")


def row_task(row: dict) -> TaskSpec | None:
    if "lowered_task_spec" in row:
        return TaskSpec.model_validate_json(json.dumps(json.loads(row["lowered_task_spec"])["task"]))
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
    """Read an exact Result path or find trials without descending into task-controlled outputs."""
    if jobs.name in {"result.json", "lifecycle-result.json"} and not jobs.exists():
        raise FileNotFoundError(jobs)
    if jobs.is_file():
        if jobs.name not in {"result.json", "lifecycle-result.json"}:
            raise ValueError(f"Explicit trial input must name a production Result: {jobs}")
        return [jobs]
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


def collect_trials(
    jobs: Path,
    *,
    additional_inputs: tuple[Path, ...] = (),
    recollection: RecollectionWaiver | None = None,
    archive_roots: Mapping[str, str] | None = None,
) -> dict[tuple[str, str], list[OracleTrialReceipt]]:
    trials: dict[tuple[str, str], list[OracleTrialReceipt]] = {}
    seen = {}
    roots = archive_roots or {}
    reverse = {restored: original for original, restored in roots.items()}
    paths = sorted(
        {
            resolve_archive_path(path, reverse)
            for source in (jobs, *additional_inputs)
            for path in trial_result_paths(resolve_archive_path(source, roots))
        }
    )
    validated = harbor_inspection("results", [str(path) for path in paths], archive_roots=roots)
    documents = {}
    for path in paths:
        document = decode_json_value(resolve_archive_path(path, roots).read_text())
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
    packages = harbor_inspection("tasks", native_paths, archive_roots=roots) if native_paths else {}
    native_grades = (
        harbor_inspection("native-grades", [str(path) for path in documents], archive_roots=roots) if documents else {}
    )
    for path, document in documents.items():
        native = document["config"]["task"]["path"]
        grade = native_grades.get(str(path))
        receipt = receipt_from_trial(
            path,
            document,
            packages[native]["task_checksum"],
            NativeVerifierGrade.model_validate(grade) if grade is not None else None,
            recollection=recollection,
            archive_roots=roots,
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
            or plan.source_runtime_resource_recovery != transition.source_runtime_resource_recovery
            or plan.legacy_verifier_initialization != transition.legacy_verifier_initialization
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
        if transition.legacy_verifier_initialization is not None:
            require_legacy_verifier_scope(transition.legacy_verifier_initialization, task_id, parent)


def verify_native_network_reconstructions(proofs: Mapping[str, ReconstructedNativeNetworkPolicy]) -> None:
    """Authenticate canary, execution and ruling evidence before native reconstruction."""
    verified_network_canaries = set()
    verified_network_authorities = set()
    verified_network_executions = set()
    for proof in proofs.values():
        canary = proof.canary.evidence
        canary_key = (canary.uri, canary.sha256, canonical_sha256(proof.canary.model_dump(mode="json")))
        if canary_key not in verified_network_canaries:
            verify_native_network_canary(proof.canary, StoragePath(canary.uri).read_bytes())
            verified_network_canaries.add(canary_key)
        execution = proof.inputs.execution_provenance
        execution_key = (execution.uri, execution.sha256, proof.inputs.implementation_revision)
        if execution_key not in verified_network_executions:
            raw = StoragePath(execution.uri).read_bytes()
            if hashlib.sha256(raw).hexdigest() != execution.sha256:
                raise ValueError("Native network execution provenance bytes changed")
            if json.loads(raw)["source_revision"] != proof.inputs.implementation_revision:
                raise ValueError("Native reconstruction implementation differs from its recorded execution")
            verified_network_executions.add(execution_key)
        authority = proof.authority
        if (authority.uri, authority.sha256) not in verified_network_authorities:
            if file_sha256(StoragePath(authority.uri)) != authority.sha256:
                raise ValueError("Native network reconstruction authority bytes changed")
            verified_network_authorities.add((authority.uri, authority.sha256))


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
    negative_control_policy: NegativeControlPolicy | None = None,
    legacy_role_qualification_evidence: LegacyRoleQualificationEvidence | None = None,
    additional_trial_inputs: tuple[Path, ...] = (),
    legacy_producer_1485_qualification_evidence: LegacyProducer1485QualificationEvidence | None = None,
    source_verifier_errors: Path | None = None,
    recollection: RecollectionWaiver | None = None,
    archive_roots: Mapping[str, str] | None = None,
    native_network_reconstructions: Mapping[str, ReconstructedNativeNetworkPolicy] | None = None,
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
    if negative_control_policy is None:
        raise ValueError("Oracle filter wire version 4 requires an explicit global negative-control policy")
    decision = negative_control_policy.decision
    if file_sha256(StoragePath(decision.uri)) != decision.sha256:
        raise ValueError("Standing negative-control policy decision bytes changed")
    verify_native_network_reconstructions(native_network_reconstructions or {})
    mappings = decode_json_value(task_map.read_text())
    if not isinstance(mappings, dict):
        raise ValueError("Task map must map TaskSpec IDs to Harbor task_name and task_checksum")
    legacy: dict[str, LegacyRoleQualificationReceipt | LegacyProducer1485QualificationReceipt] = dict(
        legacy_role_qualifications(legacy_role_qualification_evidence, frozenset(mappings))
        if legacy_role_qualification_evidence is not None
        else {}
    )
    older = (
        legacy_producer_1485_qualifications(legacy_producer_1485_qualification_evidence, frozenset(mappings))
        if legacy_producer_1485_qualification_evidence is not None
        else {}
    )
    if legacy.keys() & older.keys():
        raise ValueError("A source task cannot receive qualifications from two different producers")
    legacy.update(older)
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
    source_errors = {}
    if source_verifier_errors is not None:
        for line in source_verifier_errors.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = record["task_id"]
            if task_id in source_errors or task_id not in mappings:
                raise ValueError(f"Duplicate or unknown source-verifier exception: {task_id}")
            source_errors[task_id] = SourceVerifierError.model_validate(record["receipt"])
    roots = archive_roots or {}
    trials = collect_trials(
        jobs, additional_inputs=additional_trial_inputs, recollection=recollection, archive_roots=roots
    )
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
        harbor_inspection("tasks", [mapping["task_path"] for mapping in mappings.values()], archive_roots=roots)
        if mappings
        else {}
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
                            task_sha256=task_spec_sha256(task),
                            runtime_sha256=row_runtime_hash(row),
                            has_oracle=oracle,
                            exclusion_reason=OracleExclusionReason.EXECUTABILITY_PROOF_MISSING,
                            evidence=(input_evidence,),
                        )
                    proof = OracleFilterReceipt(
                        negative_control_policy=negative_control_policy,
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
                source_binding = source_runtime_binding(task, row_runtime(row))
                original_row = row
                task_trials = []
                checksum = ""
                if task.id in mappings:
                    mapping = mappings[task.id]
                    if mapping["task_sha256"] != task_spec_sha256(task):
                        raise ValueError(f"Task changed after Harbor export: {task.id}")
                    if mapping.get("runtime_sha256") != row_runtime_hash(row):
                        raise ValueError(f"Runtime changed after Harbor export: {task.id}")
                    if recollection is not None:
                        require_recollection_binding(task, mapping, recollection)
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
                        preserved_requirements=declared_native_requirements(
                            resolve_archive_path(Path(mapping["task_path"]), roots), source_binding
                        ),
                        verifier=native_verifier_profile(
                            resolve_archive_path(Path(mapping["task_path"]), roots), source_binding
                        ),
                        oracle_kind="original" if oracle else "calibration_only",
                        attempts=tuple(task_trials),
                    )
                )
                original = resolve_archive_path(Path(mapping["task_path"]), roots)
                if original_packages[mapping["task_path"]]["task_checksum"] != mapping.get(
                    "source_task_checksum", checksum
                ):
                    raise ValueError(f"Source package changed after Harbor export: {task.id}")
                if not is_nonagentic_task(task) and calibration.preserved_requirements != declared_native_requirements(
                    original, source_binding
                ):
                    raise ValueError(f"Calibration changed source-declared limits: {task.id}")
                qualification = legacy.get(task.id)
                verifier_profile = native_verifier_profile(original, source_binding)
                if calibration.verifier.mode != verifier_profile.mode:
                    raise ValueError(f"Calibration changed the source grading boundary: {task.id}")
                if qualification is None:
                    if not is_nonagentic_task(task) and calibration.verifier != verifier_profile:
                        raise ValueError(f"Calibration changed the declared verifier resource profile: {task.id}")
                elif calibration.verifier.mode != verifier_profile.mode or any(
                    getattr(qualification.verifier.execution_requirements, name) != value
                    for name, value in verifier_profile.fixed.items()
                ):
                    raise ValueError(f"Qualified legacy allocation changed the original verifier contract: {task.id}")
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
                    if attempt.post_grade_teardown is not None:
                        require_post_grade_teardown(attempt.post_grade_teardown, attempt)
                    collected_attempt = attempt.model_copy(
                        update={"infrastructure_retry": None, "prior_phase_failure": None, "post_grade_teardown": None}
                    )
                    if collected_attempt not in trials.get(key, []):
                        raise ValueError(f"Calibration trial missing from production jobs: {task.id}")
                    package = resolve_archive_path(Path(attempt.result["config"]["task"]["path"]), roots)
                    if calibration_package_identity(package, calibration.oracle_kind) != package_identity:
                        raise ValueError(f"Calibration changed the task environment or scripts: {task.id}")
                requirements = calibration.execution_requirements
                final = calibration.attempts[-1]
                policy = policies.get(task.id)
                oracle_passed = policy.passes(final, checksum) if policy is not None else calibration.passed
                scored_failure = oracle and calibration.oracle_kind == "original" and final.scored and not oracle_passed
                if (
                    not is_nonagentic_task(task)
                    and task.execution_requirements is not None
                    and task.execution_requirements != requirements
                ):
                    raise ValueError(f"Calibration overrides task specification: {task.id}")
                verifier_requirements = (
                    qualification.verifier.execution_requirements
                    if qualification is not None
                    else calibration.attempts[-1].verifier_execution_requirements
                )
                if (
                    not is_nonagentic_task(task)
                    and not scored_failure
                    and (original / TASK_SPEC_FILE).is_file()
                    and verifier_requirements is None
                ):
                    raise ValueError(f"Separate-role allocation is unqualified without verifier evidence: {task.id}")
                final_package = resolve_archive_path(
                    Path(calibration.attempts[-1].result["config"]["task"]["path"]), roots
                )
                # Physical NoP receipts bind the calibrated packet even when the oracle fails.
                if (qualification is not None or not is_nonagentic_task(task)) and (
                    not scored_failure or task.id in negative
                ):
                    task = calibrated_task_spec(
                        task, requirements, verifier_requirements, (final_package / "task.toml").read_bytes()
                    )
                row = row_with_task(row, task)
                source_setup = (
                    tomllib.loads((original / "task.toml").read_text())
                    .get("environment", {})
                    .get("setup", {})
                    .get("command", "")
                )
                incompatibility = source_environment_incompatibility(calibration.attempts[-1], source_setup)
                control = controls.get(task.id)
                if policy is not None and (
                    policy.source_task_sha256 != mapping["task_sha256"]
                    or policy.source_task_checksum != mapping.get("source_task_checksum", checksum)
                    or policy.runtime_sha256 != row_runtime_hash(original_row)
                    or policy.input_cohort_sha256 != cohort_hash
                ):
                    raise ValueError(f"Native success policy differs from the frozen source contract: {task.id}")
                source_error = source_errors.get(task.id)
                if source_error is not None:
                    require_source_verifier_error(row_task(original_row), final, source_error)
                if oracle_passed or (not oracle and control is not None and control.passed and control.trial == final):
                    reason = None
                else:
                    reason = oracle_failure_reason(final, incompatibility, source_error)
                reconstruction = (native_network_reconstructions or {}).get(task.id)
                if reconstruction is not None or reason is None:
                    network_qualified = trial_network_qualified(task, row_runtime(row), final, reconstruction)
                    if reason is None and not network_qualified:
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
                    task_sha256=task_spec_sha256(task),
                    runtime_sha256=row_runtime_hash(row),
                    harbor_task_checksum=checksum,
                    source_task_sha256=mapping["task_sha256"],
                    source_task_checksum=(
                        # Typed re-collection: the control and oracle trials consumed the
                        # sealed, control-validated package, so the original source anchor
                        # stays the sealed checksum; the rebuild's own measured checksum
                        # travels on the task map as ``source_task_checksum``.
                        recollection.binding(task.id).package_checksum
                        if recollection is not None
                        else mapping.get("source_task_checksum", checksum)
                    ),
                    native_success_policy=policy,
                    legacy_role_qualification=qualification,
                    negative_control=negative_control,
                    negative_control_applicability=applicability,
                    has_oracle=oracle,
                    trials=(calibration.attempts[-1],) if oracle else (),
                    calibration=calibration,
                    executability_run=control,
                    exclusion_reason=reason,
                    source_environment_incompatibility=incompatibility,
                    source_verifier_error=source_error,
                    infrastructure_retries=history,
                    network_policy_reconstructed_native=reconstruction,
                )
                entries[task.id] = entry
                if entry.passed:
                    kept.append(row)
                else:
                    excluded.append(original_row)
    if set(native_network_reconstructions or {}) - entries.keys():
        raise ValueError("Native network reconstructions contain tasks outside the frozen cohort")
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
        negative_control_policy=negative_control_policy,
        recollection=recollection,
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
                    "task_sha256": task_spec_sha256(task),
                    "runtime_sha256": row_runtime_hash(row),
                }
    output.write_text(json.dumps(mappings, indent=2) + "\n")


def row_with_task(row: dict, task: TaskSpec) -> dict:
    row = dict(row)
    if "lowered_task_spec" in row:
        envelope = json.loads(row["lowered_task_spec"])
        envelope["task"] = task.source_document()
        row["lowered_task_spec"] = json.dumps(envelope)
    else:
        serialized = task_spec_json(task)
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
    initial_verifier_execution_requirements: ExecutionRequirements | None = None,
    *,
    trial_cleanup_grace_sec: float | None = None,
) -> dict:
    """Run the production Trial path once per shape, creating a fresh task package each time.

    An explicit cleanup grace changes harness teardown only; omitted values use Harbor's default.
    """
    original = Path(mapping["task_path"])
    if host_admission is not None and environment != "local":
        raise ValueError("Physical-host admission applies only to LOCAL trials")
    source_binding = source_runtime_binding(task, source_runtime) or read_source_runtime(original)
    if source_binding is not None:
        source_binding.require_task(task)
    if (original / TASK_SPEC_FILE).is_file() and mapping.get("runtime_sha256") is not None and source_binding is None:
        raise ValueError(f"Original frozen source runtime is required before execution: {task.id}")
    configured_runtime = source_binding if (original / TASK_SPEC_FILE).is_file() else None
    if source_binding is not None and source_binding.runtime_sha256 != mapping.get("runtime_sha256"):
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
    preserved = declared_native_requirements(original, source_binding)
    verifier = native_verifier_profile(original, source_binding)
    shape = (
        declared
        or initial_requirements
        or ExecutionRequirements.model_validate({**INITIAL_REQUIREMENTS.model_dump(), **preserved})
    )
    if declared is not None and native_requirements(original) != declared:
        raise ValueError(f"Harbor package differs from declared execution requirements: {task.id}")
    verifier_shape = (
        (
            initial_verifier_execution_requirements
            or task.verifier_execution_requirements
            or initial_verifier_requirements(shape, verifier)
        )
        if (original / TASK_SPEC_FILE).is_file()
        else None
    )
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
                    document.setdefault("environment", {})[name] = shape.model_dump()[name]
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
                **(
                    {"verifier_execution_requirements": verifier_shape.model_dump(mode="json")} if verifier_shape else {}
                ),
            },
            "agent": {"name": "oracle", "kwargs": {"resource_usage_filename": "resource-usage.json"}},
            "environment": {"type": environment, "kwargs": environment_kwargs},
            "trials_dir": str(attempt / "jobs"),
            "trial_name": "oracle",
            "trial_attempt_timeout_sec": CALIBRATION_ATTEMPT_TIMEOUT,
            **({"trial_cleanup_grace_sec": trial_cleanup_grace_sec} if trial_cleanup_grace_sec is not None else {}),
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
                    "--sandbox-lifecycle-log",
                    str(attempt / "sandbox-lifecycle.jsonl"),
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
        if trial.passed(trial.result["task_checksum"]) or (declared is not None and verifier_shape is None):
            break
        updated = next_trial_allocations(trial, preserved, verifier)
        if updated is None:
            break
        shape, verifier_shape = updated
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


def calibration_contracts(cohort: Path, mappings: dict) -> Iterator[tuple[TaskSpec, dict | None]]:
    """Stream and verify one complete frozen assignment without retaining its packets."""
    seen = set()
    paths = sorted(cohort.glob("*.parquet")) if cohort.is_dir() else [cohort]
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                task = row_task(row)
                if task is None:
                    continue
                if task.id in seen:
                    raise ValueError("Calibration task map must cover exactly the full converted population")
                seen.add(task.id)
                mapping = mappings[task.id]
                if mapping["task_sha256"] != task_spec_sha256(task):
                    raise ValueError(f"Task changed after Harbor export: {task.id}")
                runtime = row_runtime(row)
                runtime_sha256 = canonical_sha256(runtime) if runtime is not None else None
                if mapping.get("runtime_sha256") != runtime_sha256:
                    raise ValueError(f"Runtime changed after Harbor export: {task.id}")
                yield task, runtime
    if seen != mappings.keys():
        raise ValueError("Calibration task map must cover exactly the full converted population")


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
    initial_verifier_execution_requirements: dict[str, ExecutionRequirements | None] | None = None,
    *,
    trial_cleanup_grace_sec: float | None = None,
) -> None:
    """Fold calibration into full-population oracle filtering; collect the same trials afterward."""
    mappings = json.loads(task_map.read_text())
    # Finish the complete identity preflight before any trial can create a machine.
    task_ids = [task.id for task, _runtime in calibration_contracts(cohort, mappings)]
    if retry_contexts is not None and retry_contexts.keys() != mappings.keys():
        raise ValueError("Infrastructure phase context must cover exactly the approved partition")
    if initial_requirements is not None and (retry_contexts is None or initial_requirements.keys() != mappings.keys()):
        raise ValueError("Resumed allocations must cover the approved retry partition")
    destination.mkdir(parents=True, exist_ok=False)
    contracts = calibration_contracts(cohort, mappings)
    calibrated = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending: dict[Future[dict], str] = {}
        while True:
            for task, runtime in islice(contracts, concurrency - len(pending)):
                future = pool.submit(
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
                    source_runtime=runtime,
                    host_admission=host_admission,
                    initial_verifier_execution_requirements=(
                        initial_verifier_execution_requirements[task.id]
                        if initial_verifier_execution_requirements
                        else None
                    ),
                    trial_cleanup_grace_sec=trial_cleanup_grace_sec,
                )
                pending[future] = task.id
            if not pending:
                break
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                calibrated[pending.pop(future)] = future.result()
    graded = 0
    for result_path in destination.glob("*/attempt-*/jobs/*/result.json"):
        document = json.loads(result_path.read_text())
        if document.get("exception_info") is None and (document.get("verifier_result") or {}).get("rewards"):
            graded += 1
    if graded == 0:
        raise RuntimeError(
            f"Calibration graded no oracle trials across {len(calibrated)} tasks: every control failed "
            "before its verifier ran. Read the attempts' result.json exception_info for the cause."
        )
    ordered = {task_id: calibrated[task_id] for task_id in task_ids}
    (destination / "task-map.json").write_text(json.dumps(ordered, indent=2) + "\n")


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


def require_legacy_verifier_scope(initialization: LegacyVerifierInitialization, task_id: str, parent: dict) -> None:
    """Bind the fresh allocation to its immutable scope and original infrastructure approval."""
    documents = []
    for evidence in (initialization.scope, initialization.approval):
        path = StoragePath(evidence.uri)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != evidence.sha256:
            raise ValueError("Legacy verifier initialization authority changed")
        documents.append(json.loads(raw))
    scope, approval = documents
    if (
        approval.get("approved") is not True
        or approval.get("classification") != "our_infrastructure"
        or scope["approval_uri"] != initialization.approval.uri
        or scope["approval_sha256"] != initialization.approval.sha256
        or scope["parent_index_uri"] != approval["index_uri"]
        or scope["parent_index_sha256"] != approval["index_sha256"]
    ):
        raise ValueError("Legacy verifier initialization lacks its original approved scope")
    original = StoragePath(approval["index_uri"]).read_bytes()
    if hashlib.sha256(original).hexdigest() != approval["index_sha256"]:
        raise ValueError("Legacy verifier initialization parent index changed")
    selected = [entry for entry in scope["entries"] if entry["task_id"] == task_id]
    approved = [entry for entry in json.loads(original)["entries"] if entry["task_id"] == task_id]
    if len(selected) != 1 or len(approved) != 1:
        raise ValueError("Legacy verifier initialization requires exact approved task membership")
    entry, original_entry = selected[0], approved[0]
    checksum = parent.get("source_task_checksum", parent["task_checksum"])
    anchors = {
        "task_sha256": parent["task_sha256"],
        "runtime_sha256": parent.get("runtime_sha256"),
        "source_task_checksum": checksum,
        "parent_mapping_sha256": canonical_sha256(parent),
        "parent_result_sha256": initialization.parent_result_sha256,
    }
    if any(entry[name] != value or original_entry[name] != value for name, value in anchors.items()):
        raise ValueError("Legacy verifier initialization belongs to another frozen parent")
    if initialization.parent_mapping_sha256 != anchors["parent_mapping_sha256"]:
        raise ValueError("Legacy verifier initialization parent mapping changed")
    calibration = ResourceCalibrationReceipt.model_validate(parent["calibration"])
    if (
        entry["raw_parent_verifier_execution_requirements"] is not None
        or entry["scored"] is not False
        or entry["passed"] is not False
        or entry["source_commands_executed"] != 0
        or ExecutionRequirements.model_validate(entry["preserved_actor_execution_requirements"])
        != calibration.execution_requirements
        or ExecutionRequirements.model_validate(entry["fresh_verifier_initial_execution_requirements"])
        != initialization.verifier_execution_requirements
    ):
        raise ValueError("Legacy verifier initialization changed the approved old or fresh allocations")
    initialized_verifier_requirements(calibration, initialization)


def write_retry_plan(
    cohort: Path,
    task_map: Path,
    approvals_path: Path,
    backend_fix_path: Path,
    output: Path,
    local_host: LocalExecutionHostTransition | None = None,
    source_runtime_resource_recovery: TaskSpecRuntimeBinding | None = None,
    legacy_verifier_initialization: LegacyVerifierInitialization | None = None,
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
    if source_runtime_resource_recovery is not None and len(approvals) != 1:
        raise ValueError("A source runtime recovery plan must bind one exact original task")
    if legacy_verifier_initialization is not None and (
        source_runtime_resource_recovery is not None or len(approvals) != 1
    ):
        raise ValueError("Legacy verifier initialization requires one exact task and no runtime recovery")
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
                if mapping["task_sha256"] != task_spec_sha256(task) or mapping.get("runtime_sha256") != row_runtime_hash(
                    row
                ):
                    raise ValueError(f"Retry input task changed: {task.id}")
                calibration = ResourceCalibrationReceipt.model_validate(mapping["calibration"])
                require_infrastructure_retry_parent(calibration, approvals[task.id])
                require_bootstrap_failure(task, calibration.attempts[-1], approvals[task.id])
                if not has_oracle_solution(task) or task.execution_requirements != calibration.declared_requirements:
                    raise ValueError(f"Retry changed the original oracle or its declared execution limits: {task.id}")
                native = Path(mapping["task_path"])
                source_binding = source_runtime_binding(task, row_runtime(row))
                recovery = source_runtime_resource_recovery
                if (
                    (native / TASK_SPEC_FILE).is_file()
                    and recovery is None
                    and legacy_verifier_initialization is None
                    and calibration.attempts[-1].verifier_execution_requirements is None
                ):
                    raise ValueError(
                        f"Legacy separate verifier allocation requires an approved recovery transition: {task.id}"
                    )
                if legacy_verifier_initialization is not None:
                    legacy_verifier_initialization.source_runtime.require_task(task)
                    if legacy_verifier_initialization.source_runtime != source_binding:
                        raise ValueError(f"Verifier initialization changed the frozen source runtime: {task.id}")
                    require_legacy_verifier_scope(legacy_verifier_initialization, task.id, mapping)
                if recovery is not None:
                    recovery.require_task(task)
                    if recovery != source_binding:
                        raise ValueError(f"Recovery differs from the original frozen runtime: {task.id}")
                    recovered = recovered_source_runtime_requirements(calibration, recovery)
                    preserved, verifier = recovered.preserved_requirements, recovered.verifier
                else:
                    preserved, verifier = calibration.preserved_requirements, calibration.verifier
                if preserved != declared_native_requirements(
                    native, source_binding
                ) or verifier != native_verifier_profile(native, source_binding):
                    raise ValueError(f"Source role resources require an approved recovery transition: {task.id}")
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
        source_runtime_resource_recovery=source_runtime_resource_recovery,
        legacy_verifier_initialization=legacy_verifier_initialization,
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
    verifier_starts = {}
    for task_id, parent in parents.items():
        calibration = ResourceCalibrationReceipt.model_validate(parent["calibration"])
        if plan.legacy_verifier_initialization is not None:
            require_legacy_verifier_scope(plan.legacy_verifier_initialization, task_id, parent)
            verifier_starts[task_id] = initialized_verifier_requirements(
                calibration, plan.legacy_verifier_initialization
            )
        elif plan.source_runtime_resource_recovery is not None:
            recovered = recovered_source_runtime_requirements(calibration, plan.source_runtime_resource_recovery)
            verifier_starts[task_id] = initial_verifier_requirements(
                recovered.execution_requirements, recovered.verifier
            )
        else:
            verifier_starts[task_id] = calibration.attempts[-1].verifier_execution_requirements
        native = Path(parent["task_path"])
        if plan.approvals[task_id].bootstrap_failure is not None:
            task = TaskSpec.model_validate_json((native / TASK_SPEC_FILE).read_bytes())
            require_bootstrap_failure(task, calibration.attempts[-1], plan.approvals[task_id])
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
            source_runtime_resource_recovery=plan.source_runtime_resource_recovery,
            legacy_verifier_initialization=plan.legacy_verifier_initialization,
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
        initial_verifier_execution_requirements=verifier_starts,
        initial_requirements={
            task_id: (
                recovered_source_runtime_requirements(
                    ResourceCalibrationReceipt.model_validate(parent["calibration"]),
                    plan.source_runtime_resource_recovery,
                ).execution_requirements
                if plan.source_runtime_resource_recovery is not None
                else ResourceCalibrationReceipt.model_validate(parent["calibration"]).execution_requirements
            )
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
    collect.add_argument("--jobs-dir", type=Path)
    collect.add_argument(
        "--trial-input",
        type=Path,
        action="append",
        default=[],
        help="original trial root or exact Result path; repeat for multiple inputs",
    )
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--conversion-manifest", type=Path)
    collect.add_argument("--executability-runs", type=Path, help="strong-model production-control JSONL receipts")
    collect.add_argument("--mechanical-outcomes", type=Path, help="typed per-task mechanical outcome JSONL")
    collect.add_argument("--input-evidence-uri", help="immutable cohort URI for tasks with no execution proof")
    collect.add_argument("--native-success-policies", type=Path)
    collect.add_argument("--negative-control-policy", type=Path, required=True, help="global standing policy JSON")
    collect.add_argument("--negative-controls", type=Path)
    collect.add_argument("--source-verifier-errors", type=Path, help="proved pinned upstream parser exception JSONL")
    collect.add_argument("--legacy-role-qualification-evidence", type=Path)
    collect.add_argument("--legacy-producer-1485-qualification-evidence", type=Path)
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
    plan.add_argument("--source-runtime-resource-recovery", type=Path, help="exact approved original runtime binding")
    plan.add_argument("--legacy-verifier-initialization", type=Path, help="exact approved fresh verifier initialization")
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
    completed = commands.add_parser("bind-post-grade-teardowns")
    completed.add_argument("--task-map", type=Path, required=True)
    completed.add_argument("--evidence", type=Path, required=True, help="JSON list of PostGradeTeardownEvidence objects")
    completed.add_argument("--output", type=Path, required=True)
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
    if args.command == "bind-post-grade-teardowns":
        proofs = [PostGradeTeardownEvidence.model_validate(item) for item in json.loads(args.evidence.read_text())]
        by_result = {item.result_sha256: item for item in proofs}
        if len(by_result) != len(proofs):
            raise ValueError("Duplicate recorded-grade evidence for the same production Trial")
        bind_post_grade_teardowns(args.task_map, by_result, args.output)
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
            (
                TaskSpecRuntimeBinding.model_validate_json(args.source_runtime_resource_recovery.read_text())
                if args.source_runtime_resource_recovery
                else None
            ),
            (
                LegacyVerifierInitialization.model_validate_json(args.legacy_verifier_initialization.read_text())
                if args.legacy_verifier_initialization
                else None
            ),
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
    trial_inputs = ([args.jobs_dir] if args.jobs_dir is not None else []) + args.trial_input
    if not trial_inputs:
        raise ValueError("Collection requires an original trial root or exact Result path")
    filter_cohort(
        args.cohort,
        args.task_map,
        trial_inputs[0],
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
        NegativeControlPolicy.model_validate_json(args.negative_control_policy.read_text()),
        (
            LegacyRoleQualificationEvidence.model_validate_json(args.legacy_role_qualification_evidence.read_text())
            if args.legacy_role_qualification_evidence is not None
            else None
        ),
        additional_trial_inputs=tuple(trial_inputs[1:]),
        legacy_producer_1485_qualification_evidence=(
            LegacyProducer1485QualificationEvidence.model_validate_json(
                args.legacy_producer_1485_qualification_evidence.read_text()
            )
            if args.legacy_producer_1485_qualification_evidence is not None
            else None
        ),
        source_verifier_errors=args.source_verifier_errors,
    )


if __name__ == "__main__":
    main()
