# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed declarations of Marin's eval policy: which candidates run which evaluations, with which exceptions.

Candidates wrap model catalog entries and evaluations hold benchmark catalog selections.
"""

import re
from collections import Counter
from dataclasses import dataclass

from evalrecords.records import EvaluationSelection
from marin.evaluation.model_config import ModelConfig

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


@dataclass(frozen=True, kw_only=True)
class PolicyEvaluation:
    """One catalog selection and the settings the policy runs it with.

    The policy owns ``max_output`` and ``seed``, so a selection may not set them in its run overrides.
    """

    selection: EvaluationSelection
    context: int
    max_output: int | None
    """None leaves output limited only by the context and the benchmark protocol."""
    seed: int | None
    """A base seed for every trial; None leaves sampling seeds to the benchmark protocol."""

    def __post_init__(self) -> None:
        if self.selection.run.max_tokens is not None or self.selection.run.seed is not None:
            raise ValueError(f"{self.name}: set max_output and seed on the policy evaluation, not the selection")

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
    """Settings for one candidate on one evaluation; unset ones keep the evaluation's value."""

    candidate: PolicyCandidate
    evaluation: PolicyEvaluation
    reason: str
    context: int | None = None
    max_output: int | None = None

    def __post_init__(self) -> None:
        if self.context is not None and self.context > self.candidate.max_context:
            raise ValueError(
                f"override for {self.candidate.name} on {self.evaluation.name} "
                "sets a context larger than the candidate's window"
            )


@dataclass(frozen=True, kw_only=True)
class EvalPolicy:
    """One version of the eval policy. Every evaluation runs for every candidate unless excluded."""

    version: str
    candidates: tuple[PolicyCandidate, ...]
    in_distribution: tuple[PolicyEvaluation, ...]
    exclusions: tuple[PolicyExclusion, ...] = ()
    overrides: tuple[PolicyOverride, ...] = ()
    release_tag: str | None = None
    """The git tag a released version launches from; every run of it must name that tag's commit."""

    def __post_init__(self) -> None:
        exceptions = (*self.exclusions, *self.overrides)
        _require_unique("candidate", [candidate.name for candidate in self.candidates])
        _require_unique("evaluation", [evaluation.name for evaluation in self.in_distribution])
        _require_unique("exception for", [f"{item.candidate.name} on {item.evaluation.name}" for item in exceptions])
        # Full-value comparison rejects exceptions built from stale evaluations.
        for item in exceptions:
            if item.candidate not in self.candidates or item.evaluation not in self.in_distribution:
                raise ValueError(f"exception for {item.candidate.name} on {item.evaluation.name} is not in this policy")


def _require_unique(kind: str, names: list[str]) -> None:
    if duplicates := sorted(name for name, count in Counter(names).items() if count > 1):
        raise ValueError(f"more than one {kind} {duplicates}")
