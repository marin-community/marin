# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable Atlas difficulty measurements and their retained-population receipts."""

from datetime import UTC, datetime
from enum import StrEnum

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


SOLVER_MODELS = {
    Solver.SMALL: "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    Solver.LARGE: "Qwen/Qwen3.5-122B-A10B",
    Solver.HOSTED: "zai-org/GLM-5.3",
    Solver.SNOWBALL: "open-athena/Marin-Snowball-67B-A2B-v0.1",
}


class ThinkingPolicy(StrEnum):
    OFF = "off"
    ON = "on"
    LOW = "low"
    CHECKPOINT = "checkpoint"


SOLVER_THINKING = {
    Solver.SMALL: ThinkingPolicy.OFF,
    Solver.LARGE: ThinkingPolicy.ON,
    Solver.HOSTED: ThinkingPolicy.LOW,
    Solver.SNOWBALL: ThinkingPolicy.CHECKPOINT,
}


class MeasurementStatus(StrEnum):
    MEASURED = "measured"
    PENDING = "pending"
    NOT_MEASURED = "not_measured"


class SampleTask(CatalogRecord):
    task_id: str
    task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DifficultySample(CatalogRecord):
    """Seeded selection from one frozen retained population, shared by every solver."""

    source_id: str
    release: DatasetRelease
    converter_version: str
    converter_revision: str
    population_count: int = Field(ge=0)
    task_specs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    drop_manifest: ContentEvidenceReference
    seed: int
    tasks: list[SampleTask]
    pre_oracle_filter: bool
    resample_after_oracle_filter: bool
    oracle_filter_receipt: ContentEvidenceReference | None = None

    @model_validator(mode="after")
    def _selection(self) -> "DifficultySample":
        if len(self.tasks) != min(100, self.population_count):
            raise ValueError("Difficulty requires min(100, retained population) selected tasks")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("Difficulty sample contains duplicate task IDs")
        if self.pre_oracle_filter != self.resample_after_oracle_filter:
            raise ValueError("A pre-oracle-filter sample must be scheduled for resampling")
        if self.pre_oracle_filter and self.oracle_filter_receipt is not None:
            raise ValueError("A pre-oracle-filter sample cannot certify a completed oracle filter")
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
        if self.temperature != 0.7:
            raise ValueError("Atlas difficulty v3 requires temperature 0.7")
        if self.input_tokens + self.output_tokens > self.context_tokens:
            raise ValueError("Input and output budgets exceed the served context")
        expected = (65536, 49152, 16384, 0.95, 20, 0.0, 1.0, 0.0, 0.0)
        actual = (
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
        if actual != expected and not self.deviations:
            raise ValueError("Record every deviation from the v2 token and sampling settings")
        return self


class DifficultyMeasurement(CatalogRecord):
    """A source and solver observation with scored coverage or a missing reason."""

    measurement_id: str
    protocol_id: str
    created_at: str
    source_id: str
    release: DatasetRelease
    converter_version: str
    solver: Solver
    status: MeasurementStatus
    reason: str | None
    sample: DifficultySample | None
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
    def _coverage(self) -> "DifficultyMeasurement":
        if self.solved > self.n:
            raise ValueError("Solved count exceeds scored tasks")
        if self.status == MeasurementStatus.NOT_MEASURED:
            if not self.reason or self.n or self.solved or self.wilson_95 is not None:
                raise ValueError("Not-measured results need a reason and no solve statistics")
            return self
        if self.sample is None or self.sample_receipt is None or self.model is None or self.settings is None:
            raise ValueError("Pending and measured results require pinned sample, model and settings")
        if self.protocol_id == PROTOCOL_ID and self.model.model != SOLVER_MODELS[self.solver]:
            raise ValueError("The solver checkpoint does not match the difficulty v3 roster")
        if self.protocol_id == PROTOCOL_ID and self.settings.thinking != SOLVER_THINKING[self.solver]:
            raise ValueError("The solver thinking policy does not match the difficulty v3 roster")
        if (self.source_id, self.release, self.converter_version) != (
            self.sample.source_id,
            self.sample.release,
            self.sample.converter_version,
        ):
            raise ValueError("Measurement and sample identify different source contracts")
        if self.n + self.pending_tasks != len(self.sample.tasks):
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


def register_difficulty(store: DataStore) -> None:
    """Add the immutable measurement table to a curation catalog or campaign archive."""
    store.table(
        MEASUREMENTS,
        primary_key=("measurement_id",),
        schema=arrow_schema(DifficultyMeasurement),
        on_conflict=OnConflict.ERROR,
    )
