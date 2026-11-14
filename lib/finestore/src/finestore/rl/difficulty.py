# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable Atlas difficulty measurements and their retained-population receipts."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType

from pydantic import Field, field_validator, model_validator

from finestore.layout import OnConflict
from finestore.rl.curation import CatalogRecord, ContentEvidenceReference, DatasetRelease
from finestore.schema import arrow_schema
from finestore.store import DataStore

PROTOCOL_ID = "atlas-difficulty-v3"
MEASUREMENTS = "difficulty_measurements"


class Solver(StrEnum):
    SMALL = "small"
    LARGE = "large"
    HOSTED = "hosted"
    SNOWBALL = "snowball"


class ThinkingPolicy(StrEnum):
    OFF = "off"
    ON = "on"
    LOW = "low"
    CHECKPOINT = "checkpoint"


@dataclass(frozen=True)
class DifficultyProtocol:
    models: Mapping[Solver, str]
    thinking: Mapping[Solver, ThinkingPolicy]
    temperature: float
    sampling: tuple[int | float, ...]


# Add a new entry for a new protocol; keep existing contracts readable indefinitely.
PROTOCOLS = MappingProxyType(
    {
        "atlas-difficulty-v3": DifficultyProtocol(
            MappingProxyType(
                {
                    Solver.SMALL: "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                    Solver.LARGE: "Qwen/Qwen3.5-122B-A10B",
                    Solver.HOSTED: "zai-org/GLM-5.3",
                    Solver.SNOWBALL: "open-athena/Marin-Snowball-67B-A2B-v0.1",
                }
            ),
            MappingProxyType(
                {
                    Solver.SMALL: ThinkingPolicy.OFF,
                    Solver.LARGE: ThinkingPolicy.ON,
                    Solver.HOSTED: ThinkingPolicy.LOW,
                    Solver.SNOWBALL: ThinkingPolicy.CHECKPOINT,
                }
            ),
            0.7,
            (65536, 49152, 16384, 0.95, 20, 0.0, 1.0, 0.0, 0.0),
        )
    }
)
SOLVER_MODELS = PROTOCOLS[PROTOCOL_ID].models
SOLVER_THINKING = PROTOCOLS[PROTOCOL_ID].thinking


class MeasurementStatus(StrEnum):
    MEASURED = "measured"
    PENDING = "pending"
    NOT_MEASURED = "not_measured"


class SampleTask(CatalogRecord):
    task_id: str
    task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DifficultyPopulation(CatalogRecord):
    """Population and filter evidence used in compact catalog projections."""

    source_id: str
    release: DatasetRelease
    converter_version: str
    converter_revision: str
    population_count: int = Field(ge=0)
    task_specs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    drop_manifest: ContentEvidenceReference
    seed: int
    pre_oracle_filter: bool
    resample_after_oracle_filter: bool
    oracle_filter_receipt: ContentEvidenceReference | None = None

    @model_validator(mode="after")
    def _filter_state(self) -> "DifficultyPopulation":
        if self.pre_oracle_filter != self.resample_after_oracle_filter:
            raise ValueError("A pre-oracle-filter sample must be scheduled for resampling")
        if self.pre_oracle_filter and self.oracle_filter_receipt is not None:
            raise ValueError("A pre-oracle-filter sample cannot certify a completed oracle filter")
        return self


class DifficultySample(DifficultyPopulation):
    """Seeded selection from one frozen retained population, shared by every solver."""

    tasks: list[SampleTask]

    @model_validator(mode="after")
    def _selection(self) -> "DifficultySample":
        if len(self.tasks) != min(100, self.population_count):
            raise ValueError("Difficulty requires min(100, retained population) selected tasks")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("Difficulty sample contains duplicate task IDs")
        return self


class SolverIdentity(CatalogRecord):
    model: str
    revision: str = Field(min_length=1)
    tokenizer: str
    tokenizer_revision: str
    chat_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    serving_receipt: ContentEvidenceReference


class ConfidenceInterval(CatalogRecord):
    low: float
    high: float


class SamplingSettings(CatalogRecord):
    context_tokens: int = Field(gt=0)
    input_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    temperature: float
    top_p: float
    top_k: int | None
    min_p: float | None
    repetition_penalty: float | None
    presence_penalty: float
    frequency_penalty: float
    thinking: ThinkingPolicy
    deviations: list[str]

    @model_validator(mode="after")
    def _protocol(self) -> "SamplingSettings":
        if self.input_tokens + self.output_tokens > self.context_tokens:
            raise ValueError("Input and output budgets exceed the served context")
        return self

    def sampling_values(self) -> tuple[int | float | None, ...]:
        return (
            self.context_tokens,
            self.input_tokens,
            self.output_tokens,
            self.top_p,
            self.top_k,
            self.min_p,
            self.repetition_penalty,
            self.presence_penalty,
            self.frequency_penalty,
        )


class DifficultyObservation(CatalogRecord):
    """A compact source and solver observation without the selected task list."""

    measurement_id: str
    protocol_id: str
    created_at: str
    source_id: str
    release: DatasetRelease
    converter_version: str
    solver: Solver
    status: MeasurementStatus
    reason: str | None
    sample: DifficultyPopulation | None
    sample_receipt: ContentEvidenceReference | None
    model: SolverIdentity | None
    settings: SamplingSettings | None
    n: int = Field(ge=0)
    solved: int = Field(ge=0)
    wilson_95: ConfidenceInterval | None
    infrastructure_attempts: int = Field(ge=0)
    pending_tasks: int = Field(ge=0)
    attempts_receipt: ContentEvidenceReference | None
    gpu_hours: float = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)

    @field_validator("created_at")
    @classmethod
    def _utc_timestamp(cls, value: str) -> str:
        instant = datetime.fromisoformat(value)
        if instant.tzinfo is None:
            raise ValueError("Difficulty observation timestamps require a timezone")
        return instant.astimezone(UTC).isoformat()

    @model_validator(mode="after")
    def _coverage(self) -> "DifficultyObservation":
        if self.solved > self.n:
            raise ValueError("Solved count exceeds scored tasks")
        if self.status == MeasurementStatus.NOT_MEASURED:
            if not self.reason or self.n or self.solved or self.wilson_95 is not None:
                raise ValueError("Not-measured results need a reason and no solve statistics")
            return self
        if self.sample is None or self.sample_receipt is None or self.model is None or self.settings is None:
            raise ValueError("Pending and measured results require pinned sample, model and settings")
        protocol = PROTOCOLS.get(self.protocol_id)
        if protocol is not None:
            if (
                self.model.model != protocol.models[self.solver]
                or self.settings.thinking != protocol.thinking[self.solver]
            ):
                raise ValueError("Solver model or thinking policy differs from its protocol")
            if self.settings.temperature != protocol.temperature:
                raise ValueError("Solver temperature differs from its protocol")
            if self.settings.sampling_values() != protocol.sampling and not self.settings.deviations:
                raise ValueError("Record every deviation from the protocol token and sampling settings")
        if (self.source_id, self.release, self.converter_version) != (
            self.sample.source_id,
            self.sample.release,
            self.sample.converter_version,
        ):
            raise ValueError("Measurement and sample identify different source contracts")
        if self.n + self.pending_tasks != min(100, self.sample.population_count):
            raise ValueError("Every selected task must be scored or pending")
        if self.status == MeasurementStatus.MEASURED and (
            self.pending_tasks or self.n == 0 or self.attempts_receipt is None or self.wilson_95 is None
        ):
            raise ValueError("Measured results require complete scored coverage and attempt evidence")
        if self.wilson_95 is not None:
            low, high = self.wilson_95.low, self.wilson_95.high
            if not self.n or not 0 <= low <= self.solved / self.n <= high <= 1:
                raise ValueError("Confidence interval does not contain the measured solve rate")
        return self


class DifficultyMeasurement(DifficultyObservation):
    """A complete observation retaining every selected task identity for immutable storage."""

    sample: DifficultySample | None


class AttemptStatus(StrEnum):
    SCORED = "scored"
    INFRASTRUCTURE = "infrastructure"
    PENDING = "pending"


class DifficultyAttempt(CatalogRecord):
    """A classified Harbor attempt; infrastructure failure never supplies a reward."""

    task_id: str
    status: AttemptStatus
    reward: float | None = Field(strict=True)

    @model_validator(mode="after")
    def _reward(self) -> "DifficultyAttempt":
        if self.status == AttemptStatus.SCORED:
            if self.reward is None or not math.isfinite(self.reward):
                raise ValueError("Scored attempts require a finite grader reward")
        elif self.reward is not None:
            raise ValueError("Unscored attempts cannot carry a solver reward")
        return self


def measurement_from_attempts(
    measurement: DifficultyMeasurement, attempts: Sequence[DifficultyAttempt]
) -> DifficultyMeasurement:
    """Count each scored task once, exclude infrastructure attempts, and retain partial credit in n."""
    if measurement.sample is None or measurement.attempts_receipt is None:
        raise ValueError("Counting requires the selected population and immutable attempt evidence")
    selected = {task.task_id for task in measurement.sample.tasks}
    scored = {}
    infrastructure = 0
    for attempt in attempts:
        if attempt.task_id not in selected:
            raise ValueError(f"Attempt is outside the selected sample: {attempt.task_id}")
        if attempt.status == AttemptStatus.INFRASTRUCTURE:
            infrastructure += 1
        if attempt.status == AttemptStatus.SCORED:
            if attempt.task_id in scored:
                raise ValueError(f"Multiple scored attempts would count a task twice: {attempt.task_id}")
            scored[attempt.task_id] = attempt.reward
    n = len(scored)
    solved = sum(reward == 1.0 for reward in scored.values())
    interval = None
    if n:
        z = 1.959963984540054
        rate = solved / n
        denominator = 1 + z * z / n
        center = (rate + z * z / (2 * n)) / denominator
        margin = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
        interval = {"low": max(0.0, center - margin), "high": min(1.0, center + margin)}
    return DifficultyMeasurement.model_validate(
        {
            **measurement.model_dump(mode="json"),
            "status": MeasurementStatus.MEASURED if n == len(selected) and n else MeasurementStatus.PENDING,
            "reason": None if n == len(selected) else "Selected tasks still lack scored solver attempts.",
            "n": n,
            "solved": solved,
            "pending_tasks": len(selected) - n,
            "infrastructure_attempts": infrastructure,
            "wilson_95": interval,
        }
    )


def register_difficulty(store: DataStore) -> None:
    """Add the immutable measurement table to a curation catalog or campaign archive."""
    store.table(
        MEASUREMENTS,
        primary_key=("measurement_id",),
        schema=arrow_schema(DifficultyMeasurement),
        on_conflict=OnConflict.ERROR,
    )
