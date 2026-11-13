# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed declarations of Marin's eval policy: which candidates run which evaluations, with which exceptions.

These are the policy's own view of models and benchmarks. Adapters outside this package translate them
into launches.
"""

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpecCorpus:
    """A published TaskSpec corpus, named by its benchmark."""

    benchmark: str


@dataclass(frozen=True)
class VendoredEvalchemy:
    """A named config for the vendored Evalchemy runtime."""

    config: str


@dataclass(frozen=True)
class VendoredHarbor:
    """A named config for the vendored Harbor runtime."""

    config: str


Runtime = TaskSpecCorpus | VendoredEvalchemy | VendoredHarbor


@dataclass(frozen=True, kw_only=True)
class PolicyCandidate:
    """A model under evaluation.

    ``model`` names its model catalog entry. ``max_context`` is the largest context window the model
    supports; no evaluation runs it with more.
    """

    name: str
    model: str
    max_context: int


@dataclass(frozen=True, kw_only=True)
class PolicyEvaluation:
    """One benchmark. Unset settings use the policy's defaults."""

    name: str
    run: Runtime
    context: int | None = None
    max_output: int | None = None
    seed: int | None = None


@dataclass(frozen=True, kw_only=True)
class PolicyDefaults:
    context: int
    max_output: int | None
    """None leaves output limited only by the context."""
    seed: int


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
    defaults: PolicyDefaults
    candidates: tuple[PolicyCandidate, ...]
    in_distribution: tuple[PolicyEvaluation, ...]
    exclusions: tuple[PolicyExclusion, ...] = ()
    overrides: tuple[PolicyOverride, ...] = ()

    def __post_init__(self) -> None:
        exceptions = (*self.exclusions, *self.overrides)
        _require_unique("candidate", [candidate.name for candidate in self.candidates])
        _require_unique("evaluation", [evaluation.name for evaluation in self.in_distribution])
        _require_unique("exception for", [f"{item.candidate.name} on {item.evaluation.name}" for item in exceptions])
        # Comparing whole definitions, not names, rejects an exception built from a stale copy.
        for item in exceptions:
            if item.candidate not in self.candidates or item.evaluation not in self.in_distribution:
                raise ValueError(f"exception for {item.candidate.name} on {item.evaluation.name} is not in this policy")


def _require_unique(kind: str, names: list[str]) -> None:
    if duplicates := sorted(name for name, count in Counter(names).items() if count > 1):
        raise ValueError(f"more than one {kind} {duplicates}")
