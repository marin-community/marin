"""The item protocol between evaluation sources, graders, and execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
import json
from dataclasses import asdict
from typing import Any, Literal


class RequestKind(StrEnum):
    GENERATE = "generate_until"
    LIKELIHOOD = "loglikelihood"
    ROLLING_LIKELIHOOD = "loglikelihood_rolling"


@dataclass(frozen=True)
class SourceRef:
    dataset: str
    split: str = "test"
    revision: str | None = None


@dataclass(frozen=True)
class ModelRequest:
    """One model operation, with optional preceding conversation turns."""

    kind: RequestKind
    prompt: Any
    generation: Mapping[str, Any] = field(default_factory=dict)
    continuation: str | None = None
    previous_turn: int | None = None
    context_length: int | None = None
    chat_template: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    prompt_format: Literal["fastchat"] | None = None
    # Source-selected budgets remain upper bounds after transport-level caller overrides.
    output_token_cap: int | None = None


@dataclass(frozen=True)
class GraderSpec:
    """A grading function and the dataset-provided reference data it needs."""

    name: str
    reference: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReducerSpec:
    """Select a common reduction and its public metric spellings once per source."""

    name: str = "mean"
    accuracy: str = "accuracy"
    stderr: str = "accuracy_stderr"
    repeat_ddof: int = 0
    groups: Mapping[str, str] = field(default_factory=dict)
    group_suffix: str = ""
    group_weights: Mapping[str, float] = field(default_factory=dict)
    trial_metric: str | None = "accuracy"
    selected_trial: int | None = None


@dataclass(frozen=True)
class EvalSpec:
    """One source example whose model operations and grade share an identity.

    Repetitions remain explicit operations. Their trial indices are independent
    of admission order, so scheduling cannot change a sampled request's seed.
    """

    task: str
    source: SourceRef
    source_id: Any
    ordinal: int
    requests: tuple[ModelRequest, ...]
    grader: GraderSpec
    doc: Any = field(default_factory=dict)
    target: Any = ""
    namespace: str = "default"
    trial: int | None = None
    tags: Mapping[str, str | tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self):
        # Specs cross process and storage boundaries. Reject executable state,
        # opaque task objects and non-finite numbers at the source boundary.
        json.dumps(asdict(self), allow_nan=False)


@dataclass(frozen=True)
class Grade:
    """Per-item reducer measurements and explicit sample output.

    Samples use metrics unless sample_metrics selects a primary filter's values.
    Audit fields carry supplementary evidence, never core sample fields.
    """

    metrics: Mapping[str, Any]
    audit: Mapping[str, Any] = field(default_factory=dict)
    weights: Mapping[str, float] = field(default_factory=dict)
    reducer_inputs: Mapping[str, Any] = field(default_factory=dict)
    sample_metrics: Mapping[str, Any] | None = None
    filtered_resps: Sequence[Any] | None = None


@dataclass(frozen=True)
class JudgePlan:
    """Judge operations, with separate semantic-retry and infrastructure policies.

    Only MalformedJudgeResponse permits another parse attempt. Infrastructure
    failures use on_error; they never consume the malformed-response fallback.
    """

    requests: tuple[ModelRequest, ...]
    grader: GraderSpec
    on_error: GraderSpec | None = None
    max_attempts: int = 1
    on_exhausted: GraderSpec | None = None

    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError("judge plans require at least one attempt")
