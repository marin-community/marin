# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluation protocols and per-task execution inputs, independent of TaskSpec's schema."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ConversationInput

EVALUATION_SCHEMA_VERSION = 3
type ResponseSelection = Literal[
    "combine", "final_only", "final_or_complete_reasoning", "final_or_complete_boxed_reasoning"
]


class OperationKind(StrEnum):
    CHAT_GENERATION = "chat_generation"
    TEXT_GENERATION = "text_generation"
    CHOICE_LIKELIHOOD = "choice_likelihood"
    CONTINUATION_LIKELIHOOD = "continuation_likelihood"


class EvaluationModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ChatPromptVariant(EvaluationModel):
    context: ConversationInput
    fewshot: int = Field(ge=0)


class ChatGeneration(EvaluationModel):
    """Send the task's conversation unchanged, without another submission instruction."""

    kind: Literal[OperationKind.CHAT_GENERATION] = OperationKind.CHAT_GENERATION
    variants: tuple[ChatPromptVariant, ...] = ()
    prompt_selection: Literal["fixed", "fit_context"] = "fixed"
    response_selection: ResponseSelection = "combine"
    historical_assistant_reasoning: Literal["omit", "empty"] = "omit"

    @model_validator(mode="after")
    def check_variants(self) -> "ChatGeneration":
        if self.prompt_selection == "fit_context" and not self.variants:
            raise ValueError("Context fitting requires explicit prompt variants")
        return self


class TextGeneration(EvaluationModel):
    kind: Literal[OperationKind.TEXT_GENERATION] = OperationKind.TEXT_GENERATION


class LikelihoodRequest(EvaluationModel):
    context: str
    continuation: str


class ChoiceLikelihood(EvaluationModel):
    """Score continuations; normalization uses the original choices, before adding the delimiter."""

    kind: Literal[OperationKind.CHOICE_LIKELIHOOD] = OperationKind.CHOICE_LIKELIHOOD
    choices: tuple[str, ...]
    requests: tuple[LikelihoodRequest, ...]
    gold: tuple[int, ...]

    @model_validator(mode="after")
    def check_choices(self) -> "ChoiceLikelihood":
        if (
            not self.choices
            or len(self.requests) != len(self.choices)
            or not self.gold
            or any(index < 0 or index >= len(self.choices) for index in self.gold)
        ):
            raise ValueError("Likelihood scoring requires choices and valid gold indices")
        return self


class ContinuationLikelihood(EvaluationModel):
    kind: Literal[OperationKind.CONTINUATION_LIKELIHOOD] = OperationKind.CONTINUATION_LIKELIHOOD
    request: LikelihoodRequest


Operation = Annotated[
    ChatGeneration | TextGeneration | ChoiceLikelihood | ContinuationLikelihood, Field(discriminator="kind")
]


class CohortOrder(StrEnum):
    SOURCE = "source"
    ROUND_ROBIN = "round_robin"


class LimitStage(StrEnum):
    CANDIDATES = "candidates"
    ELIGIBLE = "eligible"


class TaskGroup(EvaluationModel):
    """Match task annotations; declaration order defines round-robin group order."""

    tags: tuple[str, ...] = ()
    labels: dict[str, str] = Field(default_factory=dict)
    minimum_context: int | None = Field(default=None, gt=0)


class CohortPolicy(EvaluationModel):
    groups: tuple[TaskGroup, ...] = ()
    order: CohortOrder = CohortOrder.SOURCE
    limit_stage: LimitStage = LimitStage.CANDIDATES


class OutputFit(StrEnum):
    SHRINK = "shrink"
    REQUIRE = "require"


class OutputBudget(EvaluationModel):
    """Multiply the larger of the minimum and reference-token count plus reserve."""

    minimum: int = Field(gt=0)
    reserve: int = Field(ge=0)
    multiplier: int = Field(gt=0)


class PrefixProbe(EvaluationModel):
    message_index: int = Field(ge=0)
    characters_per_token: int = Field(gt=0)
    token_margin: int = Field(ge=0)


class ContextPolicy(EvaluationModel):
    """Resolve a generation budget and admit prompts using the run's tokenizer."""

    output_fit: OutputFit = OutputFit.SHRINK
    context_reserve: int = Field(default=64, ge=0)
    output_budget: OutputBudget | None = None
    prefix_probe: PrefixProbe | None = None

    @model_validator(mode="after")
    def check_prefix_probe(self) -> "ContextPolicy":
        if self.prefix_probe is not None and self.output_fit is not OutputFit.REQUIRE:
            raise ValueError("Prefix probes require a fixed output allowance")
        return self


class EvaluationItem(EvaluationModel):
    task_id: str
    source_row: str
    operation: Operation
    groups: dict[str, str] = Field(default_factory=dict)
    reference_text: str | None = None
    prompt_characters: int | None = Field(default=None, ge=0)

    @property
    def source_ordinal(self) -> int:
        return int(self.source_row.rsplit(":", 1)[1])


class Sampling(EvaluationModel):
    temperature: float = Field(ge=0)
    top_p: float | None = Field(gt=0, le=1)
    """None preserves the served model's generation default."""
    top_k: int | None
    max_tokens: int | None = Field(gt=0)
    """None uses the explicit run's model budget; the resolved budget is saved with the run."""
    stop: tuple[str, ...]
    chat_template_kwargs: dict[str, bool | str]


class TrialPolicy(EvaluationModel):
    """Legacy source repetitions use the same base-plus-trial seed for every source row."""

    count: int = Field(gt=0)
    seed: int | None

    def seed_for(self, trial: int) -> int | None:
        if not 0 <= trial < self.count:
            raise ValueError("Trial is outside the declared cohort")
        return None if self.seed is None else self.seed + trial


class MetricKind(StrEnum):
    BINARY = "binary"
    CONTINUOUS = "continuous"


class Metric(EvaluationModel):
    name: str
    kind: MetricKind
    higher_is_better: bool


class Metrics:
    ACCURACY = Metric(name="accuracy", kind=MetricKind.BINARY, higher_is_better=True)
    CLASSIFICATION = (
        ACCURACY,
        *(
            Metric(name=name, kind=MetricKind.BINARY, higher_is_better=False)
            for name in ("incorrect", "not_attempted", "judge_failed")
        ),
    )


class SummaryKind(StrEnum):
    """Consumer-side reductions; the driver always writes individual observations.

    Groups use the same reduction independently of the headline population.
    Weighted means read each metric's explicit weights from grading detail.
    Classification derives conditional rates from correct/incorrect/abstention counts.
    Perplexity transforms mean log likelihood with exp(-mean), leaving accuracy as a mean.
    """

    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    CLASSIFICATION = "classification"
    PERPLEXITY = "perplexity"


class LikelihoodNormalization(StrEnum):
    RAW = "raw"
    CHARACTERS = "characters"
    BYTES = "bytes"


class LikelihoodMetric(EvaluationModel):
    name: str
    normalization: LikelihoodNormalization
    scoring: Literal["most_likely", "probability_mass", "greedy_accuracy", "log_likelihood"] = "most_likely"


class EvaluationProtocol(EvaluationModel):
    """The immutable benchmark settings needed to execute and interpret individual attempts."""

    schema_version: Literal[3] = EVALUATION_SCHEMA_VERSION
    benchmark: str
    version: str
    operation: OperationKind
    sampling: Sampling | None
    trials: TrialPolicy
    metrics: tuple[Metric, ...]
    primary_metric: str
    summary: SummaryKind
    expected_population: int = Field(gt=0)
    context_window: int | None = Field(default=None, gt=0)
    cohort: CohortPolicy = Field(default_factory=CohortPolicy)
    context_policy: ContextPolicy | None = None
    likelihood_metrics: tuple[LikelihoodMetric, ...] = ()

    @model_validator(mode="after")
    def check_protocol(self) -> "EvaluationProtocol":
        names = [metric.name for metric in self.metrics]
        if len(names) != len(set(names)) or self.primary_metric not in names:
            raise ValueError("Evaluation metrics must be unique and include the primary metric")
        if (self.operation in ("chat_generation", "text_generation")) != (self.sampling is not None):
            raise ValueError("Only generation protocols declare sampling settings")
        if self.operation in ("choice_likelihood", "continuation_likelihood"):
            if {metric.name for metric in self.likelihood_metrics} != set(names):
                raise ValueError("Each likelihood metric must declare its normalization")
        elif self.likelihood_metrics:
            raise ValueError("Generation protocols cannot declare likelihood metrics")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))
