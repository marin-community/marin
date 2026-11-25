# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed declarations of Marin's eval policy: which candidates run which evaluations, with which exceptions.

Candidates wrap model catalog entries and evaluations name benchmark catalog protocols.
"""

import re
from collections import Counter
from dataclasses import KW_ONLY, dataclass

from evalrecords.records import EvaluationOverrides, EvaluationSelection, PromptFormat, TaskMachineSettings
from marin.evaluation.model_config import ModelConfig

from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.evaluation.curated import requires_judge
from experiments.evaluation.evals import JUDGES

HF_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, kw_only=True)
class PolicyCandidate:
    """A model catalog entry under evaluation.

    ``max_context`` is the largest context window the model supports; no evaluation runs it with more.
    A Hugging Face model must pin a commit SHA so the policy names one checkpoint.
    """

    model: ModelConfig
    max_context: int

    def __post_init__(self) -> None:
        if "://" not in self.model.location and (
            self.model.revision is None or HF_COMMIT.fullmatch(self.model.revision) is None
        ):
            raise ValueError(f"{self.name}: pin {self.model.location} to a Hugging Face commit SHA")

    @property
    def name(self) -> str:
        return self.model.name


@dataclass(frozen=True)
class PolicyEvaluation:
    """One catalog benchmark and every setting the policy runs it with.

    The benchmark choice (``protocol``, ``num_fewshot``, ``prompt_format``) picks a catalog variant, which
    owns prompts, thinking mode, trials, output caps, metrics and any window it enforces. Unset settings leave
    the variant's value; ``label`` names the evaluation where the benchmark's own name would not, such as an
    earlier run's name. Construction fails unless the catalog declares the variant and a judge is named exactly
    when the benchmark needs one.
    """

    benchmark: str
    _: KW_ONLY
    label: str | None = None
    protocol: str | None = None
    num_fewshot: int | None = None
    prompt_format: PromptFormat | None = None
    max_output: int | None = None
    """None leaves output limited only by the context and the benchmark protocol."""
    seed: int | None = None
    """A base seed for every trial; None leaves sampling seeds to the benchmark protocol."""
    judge: str | None = None
    """The name of the judge in ``configs/evaluations.yaml`` that grades the benchmark, for benchmarks that need one."""
    concurrency: int | None = None
    """The most attempts that run at once."""
    task_machine: TaskMachineSettings | None = None
    """The task machines agent episodes run on."""

    def __post_init__(self) -> None:
        if self.protocol == "":
            raise ValueError(f"{self.benchmark}: name a protocol or leave it unset")
        if self.judge is not None and self.judge not in JUDGES:
            raise ValueError(f"{self.benchmark}: no judge {self.judge!r}; known: {sorted(JUDGES)}")
        catalog = evaluation_benchmarks()
        if self.benchmark not in catalog:
            raise ValueError(f"no benchmark {self.benchmark!r} in the benchmark catalog")
        catalog[self.benchmark].resolve(self.protocol, self.num_fewshot, self.prompt_format)
        if (self.judge is not None) != requires_judge(self.selection):
            needs = "needs a judge" if self.judge is None else "takes no judge"
            raise ValueError(f"{self.name}: the benchmark {needs}")

    @property
    def selection(self) -> EvaluationSelection:
        """The catalog selection with the evaluation's judge, concurrency and task machines, without the output
        limit and seed, which can differ by candidate."""
        judge = {} if self.judge is None else JUDGES[self.judge].model_dump()
        run = EvaluationOverrides.model_validate(
            {**judge, "concurrency": self.concurrency, "task_machine": self.task_machine}
        )
        return EvaluationSelection(
            benchmark=self.benchmark,
            label=self.label,
            protocol=self.protocol,
            num_fewshot=self.num_fewshot,
            prompt_format=self.prompt_format,
            run=run,
        )

    @property
    def name(self) -> str:
        return self.selection.name


@dataclass(frozen=True, kw_only=True)
class PolicyExclusion:
    candidate: PolicyCandidate
    evaluation: PolicyEvaluation
    reason: str


@dataclass(frozen=True, kw_only=True)
class PolicyOverride:
    """A different output limit for one candidate on one evaluation."""

    candidate: PolicyCandidate
    evaluation: PolicyEvaluation
    reason: str
    max_output: int


@dataclass(frozen=True, kw_only=True)
class EvalPolicy:
    """One version of the eval policy. Every evaluation runs for every candidate unless excluded.

    ``serving_window`` is the context window every candidate is served at, or the candidate's own window if
    smaller.
    """

    version: str
    serving_window: int
    candidates: tuple[PolicyCandidate, ...]
    in_distribution: tuple[PolicyEvaluation, ...]
    exclusions: tuple[PolicyExclusion, ...] = ()
    overrides: tuple[PolicyOverride, ...] = ()

    def __post_init__(self) -> None:
        exceptions = (*self.exclusions, *self.overrides)
        _require_unique("candidate", [candidate.name for candidate in self.candidates])
        _require_unique("evaluation", [evaluation.name for evaluation in self.in_distribution])
        _require_unique("exception for", [f"{item.candidate.name} on {item.evaluation.name}" for item in exceptions])
        # Full-value comparison rejects exceptions built from stale evaluations.
        for item in exceptions:
            if item.candidate not in self.candidates or item.evaluation not in self.in_distribution:
                raise ValueError(f"exception for {item.candidate.name} on {item.evaluation.name} is not in this policy")


def defaults(*benchmarks: str) -> tuple[PolicyEvaluation, ...]:
    """Evaluations of each benchmark at its catalog defaults, with no policy settings."""
    return tuple(PolicyEvaluation(benchmark) for benchmark in benchmarks)


def _require_unique(kind: str, names: list[str]) -> None:
    if duplicates := sorted(name for name, count in Counter(names).items() if count > 1):
        raise ValueError(f"more than one {kind} {duplicates}")
