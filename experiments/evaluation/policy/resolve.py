# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combine a policy's evaluations and exceptions into the settings for one candidate."""

from dataclasses import dataclass
from enum import StrEnum

from evalrecords.records import EvaluationSelection
from marin.evaluation.model_config import prompt_format

from experiments.evaluation.curated import selection_dataset
from experiments.evaluation.policy.schema import EvalPolicy, PolicyCandidate, PolicyEvaluation


@dataclass(frozen=True, kw_only=True)
class ResolvedPolicyEvaluation:
    """One candidate on one evaluation, with the candidate's exceptions applied.

    Build these with ``resolve``. The top-level settings are the ones to run with; ``evaluation``
    keeps the evaluation as the policy declares it.
    """

    candidate: PolicyCandidate
    evaluation: PolicyEvaluation
    context: int
    max_output: int | None
    seed: int | None

    @property
    def selection(self) -> EvaluationSelection:
        """The catalog selection this candidate runs, with every policy setting applied."""
        selection = self.evaluation.selection
        run = selection.run.model_copy(update={"max_tokens": self.max_output, "seed": self.seed})
        return selection.model_copy(update={"run": run})

    def __post_init__(self) -> None:
        if self.context > self.candidate.max_context:
            raise ValueError(
                f"{self.candidate.name} on {self.evaluation.name}: context {self.context} is larger than "
                f"the candidate's window {self.candidate.max_context}; add an override"
            )
        if self.max_output is not None and self.max_output >= self.context:
            raise ValueError(
                f"{self.candidate.name} on {self.evaluation.name}: output limit {self.max_output} does not fit "
                f"in context {self.context}; add an override"
            )


class WindowKind(StrEnum):
    """How a benchmark's declared window constrains a run's context."""

    PROTOCOL = "protocol"
    """The window its items were built for; a run uses exactly this window."""
    AGENT_LIMIT = "agent_limit"
    """An agent's input limit; the agent gets the smaller of it and the served window, so a run may use less."""


@dataclass(frozen=True)
class BenchmarkWindow:
    """A window a benchmark declares: its protocol's window, or its agent's input limit."""

    tokens: int
    kind: WindowKind


def benchmark_window(evaluation: PolicyEvaluation, candidate: PolicyCandidate) -> BenchmarkWindow | None:
    """Return the benchmark's agent input limit or protocol window, if declared."""
    dataset = selection_dataset(evaluation.selection, prompt_format(candidate.model))
    if dataset.agent is not None and dataset.agent.max_input_tokens is not None:
        return BenchmarkWindow(tokens=dataset.agent.max_input_tokens, kind=WindowKind.AGENT_LIMIT)
    if dataset.protocol.context_window is not None:
        return BenchmarkWindow(tokens=dataset.protocol.context_window, kind=WindowKind.PROTOCOL)
    return None


def _context(policy: EvalPolicy, candidate: PolicyCandidate, evaluation: PolicyEvaluation, context: int | None) -> int:
    """The context an evaluation runs with: ``context`` if set, else its benchmark's window, else the policy's.

    A benchmark that declares a window runs on the candidate's served window, not one sized for it, so the
    candidate must be served at least that large.
    """
    window = benchmark_window(evaluation, candidate)
    if window is None:
        return policy.context if context is None else context
    name = f"{candidate.name} on {evaluation.name}"
    served = min(candidate.max_context, candidate.model.serve.max_model_len or candidate.max_context)
    if window.kind is WindowKind.PROTOCOL and window.tokens > served:
        raise ValueError(
            f"{name}: the benchmark's window {window.tokens} is larger than the served window {served}; exclude it"
        )
    if context is None:
        context = window.tokens
    elif window.kind is WindowKind.PROTOCOL and context != window.tokens:
        raise ValueError(f"{name}: context {context} is not the benchmark's window {window.tokens}")
    elif context > window.tokens:
        raise ValueError(f"{name}: context {context} is above the agent input limit {window.tokens}")
    if context > served:
        raise ValueError(f"{name}: context {context} is larger than the served window {served}; add an override")
    return context


def resolve(policy: EvalPolicy, candidate: PolicyCandidate) -> tuple[ResolvedPolicyEvaluation, ...]:
    """Return every evaluation the candidate runs under the policy.

    Excluded evaluations are left out. Each setting comes from the candidate's override, else the
    evaluation. An unset context is the benchmark's window if it declares one, else the policy's. A
    context may be below an agent's input limit but must equal a protocol window, so a candidate whose
    window cannot hold a protocol window has to be excluded from that benchmark.
    """
    if candidate not in policy.candidates:
        raise ValueError(f"{candidate.name} is not a candidate in policy {policy.version}")
    excluded = {item.evaluation.name for item in policy.exclusions if item.candidate == candidate}
    overrides = {item.evaluation.name: item for item in policy.overrides if item.candidate == candidate}
    resolved = []
    for evaluation in policy.in_distribution:
        if evaluation.name in excluded:
            continue
        context, max_output = evaluation.context, evaluation.max_output
        if (override := overrides.get(evaluation.name)) is not None:
            context = context if override.context is None else override.context
            max_output = max_output if override.max_output is None else override.max_output
        resolved.append(
            ResolvedPolicyEvaluation(
                candidate=candidate,
                evaluation=evaluation,
                context=_context(policy, candidate, evaluation, context),
                max_output=max_output,
                seed=evaluation.seed,
            )
        )
    return tuple(resolved)
