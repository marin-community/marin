# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable Atlas difficulty measurements and their retained-population receipts."""

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType

from pydantic import Field, field_validator, model_validator

from finestore.layout import OnConflict
from finestore.rl.curation import CatalogRecord, ContentEvidenceReference, DatasetRelease
from finestore.rl.status_taxonomy import ExclusionReason
from finestore.schema import SHA256_PATTERN, arrow_schema
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
SAMPLE_SIZE = 100
"""The per-measurement sample size every validator and the protocol agree on."""

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
"""The protocol's pinned solver models; the Marina server reads this directly."""
SOLVER_THINKING = PROTOCOLS[PROTOCOL_ID].thinking
"""The protocol's pinned thinking policies; the Marina server reads this directly."""


class MeasurementStatus(StrEnum):
    MEASURED = "measured"
    PENDING = "pending"
    BLOCKED = "blocked"
    NOT_MEASURED = "not_measured"


class DifficultyGapOwner(StrEnum):
    """Who has to act on a gap: the Difficulty pipeline, or the source itself."""

    PIPELINE = "pipeline"
    SOURCE = "source"


@dataclass(frozen=True)
class DifficultyGapSpec:
    """The contract behind one ``DifficultyGap`` member."""

    owner: DifficultyGapOwner
    meaning: str
    cleared_by: str
    exclusion: ExclusionReason | None = None
    """The Atlas exclusion reason the gap raises; only SOURCE-owned gaps may exclude."""

    def __post_init__(self) -> None:
        if self.exclusion is not None and self.owner is not DifficultyGapOwner.SOURCE:
            raise ValueError("Only SOURCE-owned gaps can exclude a source")


class DifficultyGap(StrEnum):
    """Every typed reason a Difficulty record may carry instead of free text.

    Each member names one blocking condition. ``DIFFICULTY_GAPS`` carries its
    owner, meaning and clear condition; the member value is the canonical
    reason token. A gap may carry one typed detail after ``:`` (a judge model,
    requirement URIs, a contract token) and one evidence segment
    ``qualification receipt: <uri>``. Historical records keep their original
    strings, so ``LEGACY_DIFFICULTY_GAP_PREFIXES`` and
    ``LEGACY_DIFFICULTY_GAP_REASONS`` map every spelling ever observed to the
    member that now owns it.
    """

    AWAITING_SCORED_COVERAGE = "awaiting_scored_coverage"
    PROTOCOL_INPUT_BUDGET_EXCEEDED = "protocol_input_budget_exceeded"
    PROVIDER_CAPACITY_WAIT = "provider_capacity_wait"
    DAYTONA_SNAPSHOT_PREFLIGHT_PENDING = "daytona_snapshot_preflight_pending"
    DAYTONA_TASK_SPEC_FACTORY_PENDING = "daytona_task_spec_factory_pending"
    DAYTONA_RUNTIME_QUALIFICATION_PENDING = "daytona_runtime_qualification_pending"
    PRODUCTION_HARBOR_TRIALS_PENDING = "production_harbor_trials_pending"
    EXECUTABILITY_ADMISSION_PENDING = "executability_admission_pending"
    FILTRATION_FINDINGS_ADMISSION_PENDING = "filtration_findings_admission_pending"
    JUDGE_QUALIFICATION_PENDING = "judge_qualification_pending"
    NO_QUALIFIED_MEASUREMENT_RECORDED = "no_qualified_measurement_recorded"
    MATH_VERIFIER_DEADLINE = "math_verifier_deadline"
    HOSTED_NATIVE_CONTEXT_UNQUALIFIED = "hosted_native_context_unqualified"
    HOSTED_TIMEOUT_GRADES_UNQUALIFIED = "hosted_timeout_grades_unqualified"
    HOSTED_DISPATCH_INTERRUPTED = "hosted_dispatch_interrupted"
    HOSTED_JUDGE_CANARY_INTERRUPTED = "hosted_judge_canary_interrupted"
    HOSTED_UNSERVED_RAW_ZERO_GRADES = "hosted_unserved_raw_zero_grades"
    ENVIRONMENT_ARTIFACT_MISSING = "environment_artifact_missing"
    UNSUPPORTED_COMPOSE_DOCUMENT = "unsupported_compose_document"
    NETWORK_POLICY_VIOLATION = "network_policy_violation"
    NO_EXECUTABLE_GRADER = "no_executable_task_spec_grader"
    STATE_CAPTURE_REQUIRED = "task_spec_state_capture_required"
    TOOL_PROVIDER_REQUIRED = "task_spec_tool_provider_required"
    SHARED_EXECUTOR_REQUIRED = "task_spec_shared_executor_required"
    PROVIDER_CONTRACT_INVALID = "task_spec_provider_contract_invalid"
    UNRESOLVED_DOCKER_BUILD = "task_spec_unresolved_build"
    JUDGE_ROUTE_UNQUALIFIED = "judge_route_unqualified"
    JUDGE_API_UNQUALIFIED = "judge_api_unqualified"
    INVALID_REFERENCE = "invalid_reference"


DIFFICULTY_GAPS: Mapping[DifficultyGap, DifficultyGapSpec] = MappingProxyType(
    {
        DifficultyGap.AWAITING_SCORED_COVERAGE: DifficultyGapSpec(
            DifficultyGapOwner.PIPELINE,
            "Selected tasks have no scored solver attempt yet.",
            "The scheduled campaign scores every selected task or records a terminal attempt.",
        ),
        DifficultyGap.PROTOCOL_INPUT_BUDGET_EXCEEDED: DifficultyGapSpec(
            DifficultyGapOwner.PIPELINE,
            "A candidate prompt exceeded the protocol input budget before dispatch; no solver score exists.",
            "The protocol budget accommodates the task, or the task is re-sampled within budget.",
        ),
        DifficultyGap.PROVIDER_CAPACITY_WAIT: DifficultyGapSpec(
            DifficultyGapOwner.PIPELINE,
            "The run is waiting for shared model-provider capacity; provider waits never score a task.",
            "Capacity is granted and the affected tasks are re-served.",
        ),
        DifficultyGap.DAYTONA_SNAPSHOT_PREFLIGHT_PENDING: DifficultyGapSpec(
            DifficultyGapOwner.PIPELINE,
            "The declarative Daytona source awaits its snapshot preflight.",
            "The snapshot preflight passes against the declarative Daytona executor.",
        ),
        DifficultyGap.DAYTONA_TASK_SPEC_FACTORY_PENDING: DifficultyGapSpec(
            DifficultyGapOwner.PIPELINE,
            "The TaskSpec Daytona source awaits its faithful actor factory.",
            "The TaskSpec Daytona executor produces and qualifies the actor environment.",
        ),
        DifficultyGap.DAYTONA_RUNTIME_QUALIFICATION_PENDING: DifficultyGapSpec(
            DifficultyGapOwner.PIPELINE,
            "The source awaits Daytona snapshot and original-runtime qualification.",
            "The snapshot identity and original runtime packet qualify in production.",
        ),
        DifficultyGap.PRODUCTION_HARBOR_TRIALS_PENDING: DifficultyGapSpec(
            DifficultyGapOwner.PIPELINE,
            "The source awaits qualified production Harbor trials with its original runtime packet.",
            "A qualified production Harbor trial run covers the source.",
        ),
        DifficultyGap.EXECUTABILITY_ADMISSION_PENDING: DifficultyGapSpec(
            DifficultyGapOwner.PIPELINE,
            "Task contracts failed static executability and await the re-filtered admission population.",
            "The executability-filtered admission population admits or rejects the contract.",
        ),
        DifficultyGap.FILTRATION_FINDINGS_ADMISSION_PENDING: DifficultyGapSpec(
            DifficultyGapOwner.PIPELINE,
            "Sampled filtration findings await a canonical executable admitted population.",
            "The sampled findings are resolved into a canonical executable admitted population.",
        ),
        DifficultyGap.JUDGE_QUALIFICATION_PENDING: DifficultyGapSpec(
            DifficultyGapOwner.PIPELINE,
            "The judge route or temperature qualification is pending for this source's judge.",
            "The route qualifies (recorded temperature deviation where the source declares another setting).",
        ),
        DifficultyGap.NO_QUALIFIED_MEASUREMENT_RECORDED: DifficultyGapSpec(
            DifficultyGapOwner.PIPELINE,
            "No qualified Harbor measurement has been recorded for this source and model.",
            "A qualified measurement for the pinned release and converter is published.",
        ),
        DifficultyGap.MATH_VERIFIER_DEADLINE: DifficultyGapSpec(
            DifficultyGapOwner.PIPELINE,
            "The math verifier deadline exhausted during grading.",
            "The shared grader deadline is fixed or the task is re-graded after repair.",
        ),
        DifficultyGap.HOSTED_NATIVE_CONTEXT_UNQUALIFIED: DifficultyGapSpec(
            DifficultyGapOwner.PIPELINE,
            "The hosted native client's effective context did not qualify.",
            "The native client context qualifies against the pinned serving receipt.",
        ),
        DifficultyGap.HOSTED_TIMEOUT_GRADES_UNQUALIFIED: DifficultyGapSpec(
            DifficultyGapOwner.PIPELINE,
            "A prior hosted aggregate contains timeout grades that do not qualify.",
            "Raw evidence is retained and a qualifying aggregate replaces it.",
        ),
        DifficultyGap.HOSTED_DISPATCH_INTERRUPTED: DifficultyGapSpec(
            DifficultyGapOwner.PIPELINE,
            "The hosted dispatch was interrupted before its traffic was sealed.",
            "A continuation recovers or re-serves the interrupted dispatch.",
        ),
        DifficultyGap.HOSTED_JUDGE_CANARY_INTERRUPTED: DifficultyGapSpec(
            DifficultyGapOwner.PIPELINE,
            "The hosted judge canary was interrupted before sealing.",
            "The judge canary is re-run and sealed.",
        ),
        DifficultyGap.HOSTED_UNSERVED_RAW_ZERO_GRADES: DifficultyGapSpec(
            DifficultyGapOwner.PIPELINE,
            "A prior hosted aggregate contains raw zero grades for requests that were never served.",
            "Unserved zeros are excluded and the aggregate is re-measured.",
        ),
        DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING: DifficultyGapSpec(
            DifficultyGapOwner.SOURCE,
            "A declared environment artifact (packages lock or prepared image) exists in no storage region.",
            "The artifact is rebuilt or repointed to a regional path (exclusion candidate otherwise).",
            exclusion=ExclusionReason.ENVIRONMENT_ARTIFACT_MISSING,
        ),
        DifficultyGap.UNSUPPORTED_COMPOSE_DOCUMENT: DifficultyGapSpec(
            DifficultyGapOwner.SOURCE,
            "The source declares a compose document no backend can execute.",
            "A supported backend executes the compose document (exclusion candidate otherwise).",
            exclusion=ExclusionReason.NO_SUPPORTED_BACKEND,
        ),
        DifficultyGap.NETWORK_POLICY_VIOLATION: DifficultyGapSpec(
            DifficultyGapOwner.SOURCE,
            "The source's declared runtime fails its production network-policy qualification.",
            "The declared runtime qualifies under the production network policy (exclusion candidate otherwise).",
            exclusion=ExclusionReason.NO_SUPPORTED_BACKEND,
        ),
        DifficultyGap.NO_EXECUTABLE_GRADER: DifficultyGapSpec(
            DifficultyGapOwner.SOURCE,
            "The task spec declares no executable grader.",
            "The source supplies an executable grader contract.",
            exclusion=ExclusionReason.NO_SUPPORTED_BACKEND,
        ),
        DifficultyGap.STATE_CAPTURE_REQUIRED: DifficultyGapSpec(
            DifficultyGapOwner.SOURCE,
            "The contract captures state that no supported executor returns.",
            "A supported executor captures the declared state.",
            exclusion=ExclusionReason.NO_SUPPORTED_BACKEND,
        ),
        DifficultyGap.TOOL_PROVIDER_REQUIRED: DifficultyGapSpec(
            DifficultyGapOwner.SOURCE,
            "The contract requires a tool provider no backend supports.",
            "A supported backend provides the tool interface (exclusion candidate otherwise).",
            exclusion=ExclusionReason.NO_SUPPORTED_BACKEND,
        ),
        DifficultyGap.SHARED_EXECUTOR_REQUIRED: DifficultyGapSpec(
            DifficultyGapOwner.SOURCE,
            "The contract requires a shared executor no backend provides.",
            "A supported backend provides the shared executor.",
        ),
        DifficultyGap.PROVIDER_CONTRACT_INVALID: DifficultyGapSpec(
            DifficultyGapOwner.SOURCE,
            "The declared tool-provider contract is invalid for its interface.",
            "The source declares a valid provider contract.",
            exclusion=ExclusionReason.NO_SUPPORTED_BACKEND,
        ),
        DifficultyGap.UNRESOLVED_DOCKER_BUILD: DifficultyGapSpec(
            DifficultyGapOwner.SOURCE,
            "The contract requires resolving a Docker build no supported executor performs.",
            "The Docker build is resolved declaratively or the contract changes.",
            exclusion=ExclusionReason.NO_SUPPORTED_BACKEND,
        ),
        DifficultyGap.JUDGE_ROUTE_UNQUALIFIED: DifficultyGapSpec(
            DifficultyGapOwner.SOURCE,
            "The source's own judge model has no qualified serving route.",
            "The judge is re-pinned to a served tier, substituted, or the source is excluded.",
            exclusion=ExclusionReason.GRADER_JUDGE_UNAVAILABLE,
        ),
        DifficultyGap.JUDGE_API_UNQUALIFIED: DifficultyGapSpec(
            DifficultyGapOwner.SOURCE,
            "The source's judge API is not the qualified chat-completions contract.",
            "The judge is re-pinned to a served API (exclusion candidate otherwise).",
            exclusion=ExclusionReason.GRADER_JUDGE_UNAVAILABLE,
        ),
        DifficultyGap.INVALID_REFERENCE: DifficultyGapSpec(
            DifficultyGapOwner.SOURCE,
            "A frozen sample task carries an invalid expected reference target.",
            "The source is repaired in a private remediation version, or excluded.",
            exclusion=ExclusionReason.INVALID_REFERENCE,
        ),
    }
)

# Legacy prefix forms with a typed detail after the prefix.
LEGACY_DIFFICULTY_GAP_PREFIXES: tuple[tuple[str, DifficultyGap], ...] = (
    ("awaiting_executability_filtered_admission_population:", DifficultyGap.EXECUTABILITY_ADMISSION_PENDING),
    (
        "Awaiting canonical executable admitted population after sampled filtration findings:",
        DifficultyGap.FILTRATION_FINDINGS_ADMISSION_PENDING,
    ),
    ("Production judge route or temperature qualification pending:", DifficultyGap.JUDGE_QUALIFICATION_PENDING),
    ("production_judge_route_not_qualified:", DifficultyGap.JUDGE_ROUTE_UNQUALIFIED),
    ("production_judge_api_not_qualified:", DifficultyGap.JUDGE_API_UNQUALIFIED),
    ("missing_environment_artifact:", DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING),
)

# Legacy spellings without a detail, mapped to the member that now owns them.
LEGACY_DIFFICULTY_GAP_REASONS: Mapping[str, DifficultyGap] = MappingProxyType(
    {
        "Awaiting scored coverage": DifficultyGap.AWAITING_SCORED_COVERAGE,
        "Selected tasks still lack scored solver attempts.": DifficultyGap.AWAITING_SCORED_COVERAGE,
        "A candidate prompt exceeded the protocol input budget before dispatch. No solver score was assigned.": (
            DifficultyGap.PROTOCOL_INPUT_BUDGET_EXCEEDED
        ),
        "Awaiting shared model provider capacity; provider waits are excluded from task attempts.": (
            DifficultyGap.PROVIDER_CAPACITY_WAIT
        ),
        "daytona_snapshot_preflight_pending": DifficultyGap.DAYTONA_SNAPSHOT_PREFLIGHT_PENDING,
        "faithful_daytona_task_spec_factory_pending": DifficultyGap.DAYTONA_TASK_SPEC_FACTORY_PENDING,
        "Faithful Daytona TaskSpec actor factory pending": DifficultyGap.DAYTONA_TASK_SPEC_FACTORY_PENDING,
        "Daytona snapshot and original runtime qualification pending": (
            DifficultyGap.DAYTONA_RUNTIME_QUALIFICATION_PENDING
        ),
        "Awaiting qualified production Harbor trials with the original source runtime packet": (
            DifficultyGap.PRODUCTION_HARBOR_TRIALS_PENDING
        ),
        "No qualified Harbor measurement has been recorded for this source/model in this snapshot.": (
            DifficultyGap.NO_QUALIFIED_MEASUREMENT_RECORDED
        ),
        "math_verifier_deadline": DifficultyGap.MATH_VERIFIER_DEADLINE,
        "native_client_effective_context_unqualified": DifficultyGap.HOSTED_NATIVE_CONTEXT_UNQUALIFIED,
        "prior_hosted_timeout_grades_unqualified": DifficultyGap.HOSTED_TIMEOUT_GRADES_UNQUALIFIED,
        "prior_hosted_timeout_grade_unqualified": DifficultyGap.HOSTED_TIMEOUT_GRADES_UNQUALIFIED,
        "interrupted_hosted_dispatch_unsealed": DifficultyGap.HOSTED_DISPATCH_INTERRUPTED,
        "interrupted_hosted_judge_canary_unsealed": DifficultyGap.HOSTED_JUDGE_CANARY_INTERRUPTED,
        "prior_hosted_unserved_requests_with_raw_zero_grades": DifficultyGap.HOSTED_UNSERVED_RAW_ZERO_GRADES,
        "unsupported_compose_document": DifficultyGap.UNSUPPORTED_COMPOSE_DOCUMENT,
        "network_policy_violation": DifficultyGap.NETWORK_POLICY_VIOLATION,
        "no_executable_task_spec_grader": DifficultyGap.NO_EXECUTABLE_GRADER,
        "task_spec_state_capture_required": DifficultyGap.STATE_CAPTURE_REQUIRED,
        "task_spec_tool_provider_required": DifficultyGap.TOOL_PROVIDER_REQUIRED,
        "task_spec_shared_executor_required": DifficultyGap.SHARED_EXECUTOR_REQUIRED,
        "task_spec_provider_contract_invalid": DifficultyGap.PROVIDER_CONTRACT_INVALID,
        "task_spec_unresolved_build": DifficultyGap.UNRESOLVED_DOCKER_BUILD,
        "invalid_expected_target_in_frozen_sample": DifficultyGap.INVALID_REFERENCE,
    }
)

QUALIFICATION_RECEIPT_PREFIX = "qualification receipt: "
GAP_DETAIL_SEPARATOR = ":"


def difficulty_gap_reason(gap: DifficultyGap, detail: str | None = None) -> str:
    """Compose one canonical reason segment for a gap."""
    return gap.value if detail is None else f"{gap.value}{GAP_DETAIL_SEPARATOR}{detail.strip()}"


def join_difficulty_gap_reasons(reasons: list[str]) -> str:
    """Join canonical reason segments; callers never hand-build separators."""
    return "; ".join(sorted(set(reasons)))


def _canonicalize_legacy_segments(reason: str) -> list[str]:
    """Split a reason into segments, tolerating a legacy join that lost its separator."""
    text = reason
    for prefix, _ in LEGACY_DIFFICULTY_GAP_PREFIXES:
        text = re.sub(rf"(?<=[\w/])\s+(?={re.escape(prefix)})", "; ", text)
    for legacy in LEGACY_DIFFICULTY_GAP_REASONS:
        text = re.sub(rf"(?<=[\w/])\s+(?={re.escape(legacy)})", "; ", text)
    return [segment.strip() for segment in text.split("; ") if segment.strip()]


def parse_difficulty_gap_segment(segment: str) -> tuple[DifficultyGap, str | None]:
    """Resolve one reason segment to its gap and typed detail, or raise."""
    if segment.startswith(QUALIFICATION_RECEIPT_PREFIX):
        raise ValueError(f"The evidence segment is not a gap: {segment!r}")
    if segment in LEGACY_DIFFICULTY_GAP_REASONS:
        return LEGACY_DIFFICULTY_GAP_REASONS[segment], None
    for prefix, gap in LEGACY_DIFFICULTY_GAP_PREFIXES:
        if segment.startswith(prefix):
            detail = segment[len(prefix) :].strip()
            return gap, detail or None
    for gap in DifficultyGap:
        if segment == gap.value:
            return gap, None
        if segment.startswith(gap.value + GAP_DETAIL_SEPARATOR):
            return gap, segment[len(gap.value) + len(GAP_DETAIL_SEPARATOR) :].strip() or None
    raise ValueError(f"Unmapped difficulty gap reason: {segment!r}")


def _hides_legacy_reason(text: str) -> bool:
    """True when a legacy spelling appears anywhere inside a longer segment."""
    if any(legacy in text and legacy != text for legacy in LEGACY_DIFFICULTY_GAP_REASONS):
        return True
    return any(prefix in text and not text.startswith(prefix) for prefix, _ in LEGACY_DIFFICULTY_GAP_PREFIXES)


def difficulty_gaps(reason: str) -> list[tuple[DifficultyGap, str | None]]:
    """Every (gap, typed detail) a reason carries, excluding its evidence segments.

    Evidence stays readable through `difficulty_gap_evidence`; a reason whose
    segments carry no gap raises through `require_difficulty_gaps`.
    """
    text = reason.strip()
    if not text:
        return []
    if text in LEGACY_DIFFICULTY_GAP_REASONS:
        return [(LEGACY_DIFFICULTY_GAP_REASONS[text], None)]
    for gap in DifficultyGap:
        if text == gap.value:
            return [(gap, None)]
    # Canonical reasons separate segments with "; "; parse each segment on its
    # own so one gap's detail can never swallow the next gap. The whole-string
    # parse stays only for single segments and legacy spellings that lost their
    # separator.
    segments = _canonicalize_legacy_segments(text)
    if len(segments) > 1:
        return [
            parse_difficulty_gap_segment(segment)
            for segment in segments
            if not segment.startswith(QUALIFICATION_RECEIPT_PREFIX)
        ]
    try:
        whole = parse_difficulty_gap_segment(text)
    except ValueError:
        whole = None
    if whole is not None and not _hides_legacy_reason(text):
        return [whole]
    return [
        parse_difficulty_gap_segment(segment)
        for segment in segments
        if not segment.startswith(QUALIFICATION_RECEIPT_PREFIX)
    ]


def difficulty_gap_evidence(reason: str) -> str | None:
    """The typed evidence URI a reason binds to, when one exists."""
    for segment in _canonicalize_legacy_segments(reason):
        if segment.startswith(QUALIFICATION_RECEIPT_PREFIX):
            return segment[len(QUALIFICATION_RECEIPT_PREFIX) :].strip()
    return None


def require_difficulty_gaps(reason: str) -> None:
    """Reject any reason a Difficulty record carries that is not a typed gap."""
    if not difficulty_gaps(reason):
        raise ValueError(f"Unmapped difficulty gap reason: {reason!r}")


class SampleTask(CatalogRecord):
    task_id: str
    task_sha256: str = Field(pattern=SHA256_PATTERN)


class DifficultyPopulation(CatalogRecord):
    """Population and filter evidence used in compact catalog projections."""

    source_id: str
    release: DatasetRelease
    converter_version: str
    converter_revision: str
    population_count: int = Field(ge=0)
    task_specs_sha256: str = Field(pattern=SHA256_PATTERN)
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
        if len(self.tasks) != min(SAMPLE_SIZE, self.population_count):
            raise ValueError("Difficulty requires min(SAMPLE_SIZE, retained population) selected tasks")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("Difficulty sample contains duplicate task IDs")
        return self


class SolverIdentity(CatalogRecord):
    model: str
    revision: str = Field(min_length=1)
    tokenizer: str
    tokenizer_revision: str
    chat_template_sha256: str = Field(pattern=SHA256_PATTERN)
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
    blocked_tasks: int = Field(default=0, ge=0)
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
        if self.reason is not None:
            require_difficulty_gaps(self.reason)
        if self.status == MeasurementStatus.NOT_MEASURED:
            if not self.reason or self.n or self.solved or self.wilson_95 is not None:
                raise ValueError("Not-measured results need a reason and no solve statistics")
            return self
        if self.sample is None or self.sample_receipt is None or self.model is None or self.settings is None:
            raise ValueError("Pending and measured results require pinned sample, model and settings")
        if self.protocol_id not in PROTOCOLS:
            raise ValueError(f"Unknown difficulty protocol: {self.protocol_id!r}")
        protocol = PROTOCOLS[self.protocol_id]
        if self.model.model != protocol.models[self.solver] or self.settings.thinking != protocol.thinking[self.solver]:
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
        if self.n + self.pending_tasks + self.blocked_tasks != min(SAMPLE_SIZE, self.sample.population_count):
            raise ValueError("Every selected task must be scored, pending or blocked")
        if self.status == MeasurementStatus.MEASURED and (
            self.pending_tasks
            or self.blocked_tasks
            or self.n == 0
            or self.attempts_receipt is None
            or self.wilson_95 is None
        ):
            raise ValueError("Measured results require complete scored coverage and attempt evidence")
        if self.status == MeasurementStatus.BLOCKED and (
            self.pending_tasks or not self.blocked_tasks or not self.reason
        ):
            raise ValueError("Blocked results require a reason, blocked tasks and no retryable pending tasks")
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
    BLOCKED = "blocked"


class DifficultyAttempt(CatalogRecord):
    """A classified Harbor attempt; infrastructure failure never supplies a reward."""

    task_id: str
    status: AttemptStatus
    reward: float | None = Field(strict=True)
    reason: str | None = None

    @model_validator(mode="after")
    def _reward(self) -> "DifficultyAttempt":
        if self.reason is not None:
            require_difficulty_gaps(self.reason)
        if self.status == AttemptStatus.SCORED:
            if self.reward is None or not math.isfinite(self.reward):
                raise ValueError("Scored attempts require a finite grader reward")
        elif self.reward is not None:
            raise ValueError("Unscored attempts cannot carry a solver reward")
        if self.status == AttemptStatus.BLOCKED and not self.reason:
            raise ValueError("Blocked attempts require a reason")
        return self


@dataclass(frozen=True)
class AttemptTally:
    """Validated attempt outcomes for one measurement."""

    scored: dict[str, float]
    """Reward per scored task id; each task scores at most once."""
    excluded: int
    """Attempt count excluded from scoring: infrastructure failures and blocked attempts.
    The recorded receipt key keeps the historical name ``infrastructure_attempts``."""
    blocked: frozenset[str]
    """Task ids whose attempts ended blocked."""
    blocked_reasons: frozenset[str]
    """Distinct blocked-attempt reasons."""


def _tally_attempts(measurement: DifficultyMeasurement, attempts: Sequence[DifficultyAttempt]) -> AttemptTally:
    """Validate attempts against the selected sample and tally their outcomes."""
    selected = {task.task_id for task in measurement.sample.tasks}
    # The recorded ``infrastructure_attempts`` receipt key counts infrastructure
    # failures AND blocked attempts; frozen receipts carry that semantics.
    excluded = 0
    scored: dict[str, float] = {}
    blocked: set[str] = set()
    blocked_reasons: set[str] = set()
    for attempt in attempts:
        if attempt.task_id not in selected:
            raise ValueError(f"Attempt is outside the selected sample: {attempt.task_id}")
        if attempt.task_id in scored and attempt.status == AttemptStatus.SCORED:
            raise ValueError(f"Multiple scored attempts would count a task twice: {attempt.task_id}")
        if attempt.task_id in blocked or attempt.task_id in scored:
            raise ValueError(f"Attempt follows a terminal result: {attempt.task_id}")
        if attempt.status in {AttemptStatus.INFRASTRUCTURE, AttemptStatus.BLOCKED}:
            excluded += 1
        if attempt.status == AttemptStatus.BLOCKED:
            blocked.add(attempt.task_id)
            assert attempt.reason is not None
            blocked_reasons.add(attempt.reason)
        if attempt.status == AttemptStatus.SCORED:
            assert attempt.reward is not None
            scored[attempt.task_id] = attempt.reward
    return AttemptTally(
        scored=scored,
        excluded=excluded,
        blocked=frozenset(blocked),
        blocked_reasons=frozenset(blocked_reasons),
    )


def _wilson_interval(solved: int, n: int) -> ConfidenceInterval:
    """Wilson 95% score interval for a solved share, clamped to the unit range."""
    z = 1.959963984540054
    rate = solved / n
    denominator = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    return ConfidenceInterval(
        low=0.0 if solved == 0 else max(0.0, center - margin),
        high=1.0 if solved == n else min(1.0, center + margin),
    )


def measurement_from_attempts(
    measurement: DifficultyMeasurement, attempts: Sequence[DifficultyAttempt]
) -> DifficultyMeasurement:
    """Count each scored task once, exclude infrastructure attempts, and retain partial credit in n."""
    if measurement.sample is None or measurement.attempts_receipt is None:
        raise ValueError("Counting requires the selected population and immutable attempt evidence")
    tally = _tally_attempts(measurement, attempts)
    n = len(tally.scored)
    solved = sum(reward == 1.0 for reward in tally.scored.values())
    interval = _wilson_interval(solved, n) if n else None
    pending = len(measurement.sample.tasks) - n - len(tally.blocked)
    status = MeasurementStatus.PENDING
    if not pending:
        status = MeasurementStatus.BLOCKED if tally.blocked else MeasurementStatus.MEASURED
    return DifficultyMeasurement.model_validate(
        {
            **measurement.model_dump(mode="json"),
            "status": status,
            "reason": (
                None
                if status is MeasurementStatus.MEASURED
                else (
                    join_difficulty_gap_reasons(sorted(tally.blocked_reasons))
                    if status is MeasurementStatus.BLOCKED
                    else DifficultyGap.AWAITING_SCORED_COVERAGE.value
                )
            ),
            "n": n,
            "solved": solved,
            "pending_tasks": pending,
            "blocked_tasks": len(tally.blocked),
            "infrastructure_attempts": tally.excluded,
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
