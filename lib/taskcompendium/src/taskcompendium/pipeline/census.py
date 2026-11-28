# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure the complete converted population without executing tasks or graders.

The census is the source-level measurement the catalog declarations rest on: how
many tasks the converter emits, what each role demands of a machine backend, how
many distinct Daytona snapshots the population provisions, and which grading
boundaries and integrity contracts the population declares. Nothing here calls a
provider: declared environments are mapped through the engine's production
identity path, and every refusal is recorded rather than repaired.

The engine protocol keeps the machine-identity machinery at the caller side
(the experiments layer owns Harbor/shellbox lowering); the stage itself reads the
converted population, joins the measurement with the grading-boundary and
integrity-contract censuses, and publishes the evidence.
"""

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from finestore.rl.curation import (
    DAYTONA_SAFE_SNAPSHOT_LIMIT,
    ActorCensusInput,
    DaytonaMeasurement,
    DaytonaUnsafeReason,
    GradingEnvironmentEvidence,
    IntegrityContractExample,
    IntegrityContractGroup,
    IntegrityContractSummary,
)
from pydantic import ValidationError
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.grading_environment import (
    assess_grading_environment,
    count_grading_environment,
    grading_environment_census,
)
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec

logger = logging.getLogger(__name__)

CENSUS_REVISION = "1"
CONTRACT_FIELDS = ("id", "version", "dependency_state", "harness_policy")
HARNESS_POLICY_FIELDS = (
    "harness",
    "harness_sha256",
    "protected_root",
    "agent_uid",
    "grader_uid",
    "protected_test_patterns",
    "pinned_plugins",
    "plugin_discovery",
)
CENSUS_JSON = "census.json"
CONTRACTS_JSON = "integrity-contracts.json"


@dataclass(frozen=True)
class TaskEnvelope:
    """One converted task document with its validated spec, as the engine measures it."""

    task_id: str
    task: dict[str, Any]
    spec: TaskSpec | None


@dataclass
class RoleOutcome:
    """One role's measured machine demand on one backend."""

    machine_slots: int = 0
    failures: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvelopeMeasurement:
    """The engine's aggregate over one population's declared machine envelopes."""

    converted: int = 0
    machine_slots: int = 0
    snapshot_sizes: dict[str, int] = field(default_factory=dict)
    execution_failures: dict[str, int] = field(default_factory=dict)
    mapping_failures: dict[str, int] = field(default_factory=dict)
    roles: dict[str, dict[str, RoleOutcome]] = field(default_factory=dict)
    actor_sources: dict[str, int] = field(default_factory=dict)
    unsafe_task_counts: dict[str, int] = field(default_factory=dict)
    """Per-task refusal reason counts, classified by the engine with each task's own environment sources."""


class CensusEngine(Protocol):
    """The caller-side machine measurement over a population's declared envelopes."""

    def measure(self, envelopes: Iterable[TaskEnvelope]) -> EnvelopeMeasurement: ...


def actor_failure_reason(detail: str, sources: Mapping[str, object]) -> str:
    """Classify one actor refusal by the environment source that caused it."""
    if sources.get("docker_build") and "Unresolved Docker build context" in detail:
        return DaytonaUnsafeReason.UNBUILT_IMAGE.value
    if sources.get("packages_lock") and "requires a registry image or Docker build context" in detail:
        return DaytonaUnsafeReason.HOST_IMAGE_REQUIRED.value
    return DaytonaUnsafeReason.EXECUTION_REQUIREMENT.value


def unsafe_reasons(measurement: EnvelopeMeasurement) -> dict[str, str]:
    """Classify actor refusals and the snapshot budget, preserving their actual details."""
    details: dict[str, set[str]] = {}
    if len(measurement.snapshot_sizes) > DAYTONA_SAFE_SNAPSHOT_LIMIT:
        details[DaytonaUnsafeReason.TOO_MANY_SNAPSHOTS.value] = {
            f"{len(measurement.snapshot_sizes)} actor snapshot identities exceed "
            f"the {DAYTONA_SAFE_SNAPSHOT_LIMIT}-snapshot budget"
        }
    for detail in (*measurement.execution_failures, *measurement.mapping_failures):
        details.setdefault(actor_failure_reason(detail, measurement.actor_sources), set()).add(detail)
    return {reason: "; ".join(sorted(values)) for reason, values in details.items()}


def unsafe_counts(measurement: EnvelopeMeasurement) -> dict[str, int]:
    """Count affected actor task slots once per reason, including the cohort budget."""
    counts = dict(measurement.unsafe_task_counts)
    if len(measurement.snapshot_sizes) > DAYTONA_SAFE_SNAPSHOT_LIMIT:
        counts[DaytonaUnsafeReason.TOO_MANY_SNAPSHOTS.value] = measurement.machine_slots
    return counts


@dataclass
class ContractCensus:
    """Group the population's declared integrity contracts by their published descriptor."""

    groups: dict[str, dict] = field(default_factory=dict)
    example: dict | None = None
    required_shared: int = 0
    shared_missing: int = 0
    task_count: int = 0

    def observe(self, task: dict, *, required_shared: bool) -> None:
        self.task_count += 1
        contract = task.get("integrity_contract")
        self.required_shared += required_shared
        self.shared_missing += required_shared and contract is None
        descriptor = {key: (contract or {}).get(key) for key in CONTRACT_FIELDS}
        descriptor["native_family"] = ((contract or {}).get("native") or {}).get("family")
        if contract is not None:
            descriptor["harness_policy"] = {name: contract[name] for name in HARNESS_POLICY_FIELDS if name in contract}
        descriptor["present"] = contract is not None
        key = canonical_sha256(descriptor)
        group = self.groups.setdefault(key, {**descriptor, "count": 0})
        group["count"] += 1
        if self.example is None and contract is not None and len(json.dumps(contract).encode()) <= 16384:
            self.example = {"task_id": task["id"], "contract": contract}

    def rows(self) -> list[dict]:
        return sorted(self.groups.values(), key=lambda group: (-group["count"], canonical_sha256(group)))


def population_digest(rows: Iterable[tuple[str, str]]) -> str:
    """The measured population's digest: a SHA-256 chain over ``[task_id, task_json]`` in task-ID order."""
    chained = hashlib.sha256()
    for task_id, task_json in sorted(rows):
        chained.update(json.dumps([task_id, task_json], ensure_ascii=False).encode())
    return chained.hexdigest()


def contract_summary(
    census: ContractCensus, evidence_url: str, dataset_id: str, dataset_revision: str, digest: str
) -> IntegrityContractSummary:
    """The bounded receipt summary; full contract groups remain in the evidence artifact."""
    rows = census.rows()
    return IntegrityContractSummary(
        groups=[
            IntegrityContractGroup(
                id=group.get("id"),
                version=group.get("version"),
                dependency_state=group.get("dependency_state"),
                native_family=group.get("native_family"),
                harness_policy_sha256=(
                    canonical_sha256(group["harness_policy"]) if group.get("harness_policy") else None
                ),
                present=group["present"],
                count=group["count"],
            )
            for group in rows[:64]
        ],
        distinct_groups=len(rows),
        absent_contract_tasks=sum(group["count"] for group in rows if not group["present"]),
        required_shared_tasks=census.required_shared,
        shared_missing_contract_tasks=census.shared_missing,
        example=(
            IntegrityContractExample(task_id=census.example["task_id"], contract_json=census.example["contract"])
            if census.example is not None
            else None
        ),
        evidence_url=evidence_url,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        task_specs_sha256=digest,
    )


def write_contract_evidence(census: ContractCensus, path: str) -> None:
    """Publish the complete contract groups beside the bounded receipt summary."""
    document = {
        "groups": census.rows(),
        "example": census.example,
        "required_shared_tasks": census.required_shared,
        "shared_missing_contract_tasks": census.shared_missing,
        "task_count": census.task_count,
    }
    StoragePath(path).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class CensusInput:
    """The measured population's identity: its dataset pin, converter revision and digest."""

    dataset_id: str
    dataset_revision: str
    converter_version: str
    converter_revision: str
    measured_tasks: int
    task_specs_sha256: str
    evidence_url: str

    def record(self) -> ActorCensusInput:
        """The typed receipt input this measurement binds."""
        return ActorCensusInput(
            dataset_id=self.dataset_id,
            dataset_revision=self.dataset_revision,
            converter_version=self.converter_version,
            converter_revision=self.converter_revision,
            measured_tasks=self.measured_tasks,
            task_specs_sha256=self.task_specs_sha256,
            evidence_url=self.evidence_url,
        )


@dataclass(frozen=True)
class CensusEvidence:
    """The stage's published measurement, typed by the finestore receipt records."""

    input: CensusInput
    measurement: DaytonaMeasurement
    grading_environment: GradingEnvironmentEvidence
    integrity_contracts: IntegrityContractSummary
    rejections: dict[str, int]

    def manifest_section(self) -> dict[str, Any]:
        return {
            "census_revision": CENSUS_REVISION,
            "input": self.input.record().model_dump(mode="json"),
            "measurement": self.measurement.model_dump(mode="json"),
            "grading_environment": self.grading_environment.model_dump(mode="json"),
            "integrity_contracts": self.integrity_contracts.model_dump(mode="json"),
            "rejections": self.rejections,
        }


def _daytona_measurement(
    measurement: EnvelopeMeasurement, dataset_id: str, dataset_revision: str, basis: str
) -> DaytonaMeasurement:
    details = unsafe_reasons(measurement)
    return DaytonaMeasurement(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        daytona_snapshot_count=len(measurement.snapshot_sizes),
        daytona_machine_slots=measurement.machine_slots,
        daytona_execution_failures=sum(measurement.execution_failures.values()),
        daytona_count_basis=basis,
        daytona_unsafe_reasons=[DaytonaUnsafeReason(reason) for reason in sorted(details)],
        daytona_unsafe_details=details,
        daytona_unsafe_counts=unsafe_counts(measurement),
        daytona_roles={
            role: {
                backend: {"machine_slots": outcome.machine_slots, "failures": dict(outcome.failures)}
                for backend, outcome in backends.items()
            }
            for role, backends in measurement.roles.items()
        },
    )


def measure_population(
    rows: Iterator[dict[str, Any]],
    *,
    source_name: str,
    dataset_id: str,
    dataset_revision: str,
    converter_revision: str,
    evidence_url: str,
    engine: CensusEngine,
) -> tuple[CensusEvidence, ContractCensus]:
    """Measure the complete converted population: machine envelopes, grading boundary and contracts.

    ``rows`` yields one normalized converted row per task, with ``task_id``,
    ``task_json``, ``normalization_kind`` and ``normalization_reason``.
    Rejected rows carry no task and are counted by their typed conversion failure.
    The caller publishes the returned contract census beside the bounded receipt
    summary so the complete groups stay readable.
    """
    contracts = ContractCensus()
    grading_counts: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    unassessable: Counter[str] = Counter()
    digests: list[tuple[str, str]] = []
    envelopes: list[TaskEnvelope] = []
    for row in rows:
        if row["task_json"] is None:
            rejections[f"{row['normalization_kind']}:{row['normalization_reason']}"] += 1
            continue
        task_id, task_json = row["task_id"], row["task_json"]
        digests.append((task_id, task_json))
        task = json.loads(task_json)
        try:
            spec = TaskSpec.model_validate(task)
        except ValidationError as error:
            unassessable[f"{type(error).__name__}: {error}"] += 1
            spec = None
        else:
            assessment = assess_grading_environment(spec)
            contracts.observe(task, required_shared=assessment.required.value == "shared")
            count_grading_environment(grading_counts, assessment.model_dump(mode="json"))
        envelopes.append(TaskEnvelope(task_id, task, spec))
    if unassessable:
        logger.warning("%s: %d tasks are unassessable for the grading census", source_name, sum(unassessable.values()))

    measurement = engine.measure(envelopes)
    if measurement.converted != len(digests):
        raise ValueError(f"{source_name}: engine measured {measurement.converted} of {len(digests)} converted tasks")
    digest = population_digest(digests)
    basis = (
        f"Complete {measurement.converted}-TaskSpec census from {dataset_id}@{dataset_revision}; "
        f"native Harbor task.toml or declared environment requirements; "
        f"actor on Daytona, package-lock grader on LOCAL; snapshot identity through the production "
        f"naming path; no provider calls; recipe revision {converter_revision}"
    )
    return (
        CensusEvidence(
            input=CensusInput(
                dataset_id=dataset_id,
                dataset_revision=dataset_revision,
                converter_version=converter_revision,
                converter_revision=converter_revision,
                measured_tasks=measurement.converted,
                task_specs_sha256=digest,
                evidence_url=evidence_url,
            ),
            measurement=_daytona_measurement(measurement, dataset_id, dataset_revision, basis),
            grading_environment=GradingEnvironmentEvidence(
                **grading_environment_census(grading_counts),
                missing_artifact_counts={},
                evidence_url=evidence_url,
            ),
            integrity_contracts=contract_summary(contracts, evidence_url, dataset_id, dataset_revision, digest),
            rejections=dict(rejections),
        ),
        contracts,
    )
