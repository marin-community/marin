# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure the complete converted population without executing tasks or graders.

The census is the source-level measurement the catalog declarations rest on: how
many tasks the converter emits, what each role demands of a machine backend, how
many distinct Daytona snapshots the population provisions, and which grading
boundaries the population declares. Nothing here calls a provider: declared
environments are mapped through the engine's production identity path, and every
refusal is recorded rather than repaired.

The engine protocol keeps the machine-identity machinery at the caller side
(the experiments layer owns Harbor/shellbox lowering); the stage itself reads the
converted population, joins the measurement with the grading-boundary census and
the integrity-contract summary, and publishes the evidence.
"""

import hashlib
import json
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
    IntegrityContractGroup,
    IntegrityContractSummary,
)

from taskcompendium.grading_environment import (
    assess_grading_environment,
    count_grading_environment,
    grading_environment_census,
)
from taskcompendium.models import TaskSpec

CENSUS_REVISION = "1"
CENSUS_JSON = "census.json"


@dataclass(frozen=True)
class TaskEnvelope:
    """One converted task document with its validated spec, as the engine measures it."""

    task_id: str
    task: dict[str, Any]
    spec: TaskSpec


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

    def describe(self) -> str:
        """The measurement rule this engine applies, for the published count basis."""
        ...


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


def population_digest(rows: Iterable[tuple[str, str]]) -> str:
    """The measured population's digest: a SHA-256 chain over ``[task_id, task_json]`` in task-ID order."""
    chained = hashlib.sha256()
    for task_id, task_json in sorted(rows):
        chained.update(json.dumps([task_id, task_json], ensure_ascii=False).encode())
    return chained.hexdigest()


def contract_summary(
    *,
    measured_tasks: int,
    required_shared: int,
    shared_missing: int,
    evidence_url: str,
    dataset_id: str,
    dataset_revision: str,
    digest: str,
) -> IntegrityContractSummary:
    """The population's contract summary: one absent group until the contract-bearing
    stage ports make declared contracts reachable in main's schema."""
    return IntegrityContractSummary(
        groups=[
            IntegrityContractGroup(
                id=None,
                version=None,
                dependency_state=None,
                native_family=None,
                harness_policy_sha256=None,
                present=False,
                count=measured_tasks,
            )
        ],
        distinct_groups=1,
        absent_contract_tasks=measured_tasks,
        required_shared_tasks=required_shared,
        shared_missing_contract_tasks=shared_missing,
        example=None,
        evidence_url=evidence_url,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        task_specs_sha256=digest,
    )


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
) -> CensusEvidence:
    """Measure the complete converted population: machine envelopes, grading boundary and contracts.

    ``rows`` yields one normalized converted row per task, with ``task_id``,
    ``task_json``, ``normalization_kind`` and ``normalization_reason``.
    Rejected rows carry no task and are counted by their typed conversion failure.
    """
    grading_counts: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    digests: list[tuple[str, str]] = []
    envelopes: list[TaskEnvelope] = []
    required_shared = 0
    shared_missing = 0
    for row in rows:
        if row["task_json"] is None:
            rejections[f"{row['normalization_kind']}:{row['normalization_reason']}"] += 1
            continue
        task_id, task_json = row["task_id"], row["task_json"]
        digests.append((task_id, task_json))
        task = json.loads(task_json)
        # The normalize view holds validated TaskSpec serializations; anything
        # else is exported data corruption, so refuse instead of measuring it.
        spec = TaskSpec.model_validate(task)
        assessment = assess_grading_environment(spec)
        # Main's TaskSpec schema carries no integrity contract, so a required-SHARED
        # boundary is always missing its contract; the contract-bearing stage ports
        # bring the declared contracts this census will then group.
        shared = assessment.required.value == "shared"
        required_shared += shared
        shared_missing += shared
        count_grading_environment(grading_counts, assessment.model_dump(mode="json"))
        envelopes.append(TaskEnvelope(task_id, task, spec))

    measurement = engine.measure(envelopes)
    if measurement.converted != len(digests):
        raise ValueError(f"{source_name}: engine measured {measurement.converted} of {len(digests)} converted tasks")
    digest = population_digest(digests)
    basis = (
        f"Complete {measurement.converted}-TaskSpec census from {dataset_id}@{dataset_revision}; "
        f"native Harbor task.toml or declared environment requirements; "
        f"{engine.describe()}; recipe revision {converter_revision}"
    )
    return CensusEvidence(
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
        integrity_contracts=contract_summary(
            measured_tasks=measurement.converted,
            required_shared=required_shared,
            shared_missing=shared_missing,
            evidence_url=evidence_url,
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            digest=digest,
        ),
        rejections=dict(rejections),
    )
