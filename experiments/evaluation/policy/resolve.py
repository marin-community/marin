# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combine a policy's evaluations and exceptions into the settings for one candidate."""

from dataclasses import dataclass

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


def benchmark_window(evaluation: PolicyEvaluation, candidate: PolicyCandidate) -> int | None:
    """The input window the benchmark enforces on any server: its agent's input limit, else its protocol's window."""
    dataset = selection_dataset(evaluation.selection, prompt_format(candidate.model))
    if dataset.agent is not None:
        return dataset.agent.max_input_tokens
    return dataset.protocol.context_window


def resolve(policy: EvalPolicy, candidate: PolicyCandidate) -> tuple[ResolvedPolicyEvaluation, ...]:
    """Return every evaluation the candidate runs under the policy.

    Excluded evaluations are left out. Each setting comes from the candidate's override, else the
    evaluation. An unset context is the benchmark's window if it declares one, else the policy's; a
    context that differs from the benchmark's window is rejected. A context larger than the candidate's
    window needs an override.
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
        window = benchmark_window(evaluation, candidate)
        if context is None:
            context = policy.context if window is None else window
        elif window is not None and context != window:
            raise ValueError(
                f"{candidate.name} on {evaluation.name}: context {context} is not the window {window} "
                "the benchmark enforces"
            )
        resolved.append(
            ResolvedPolicyEvaluation(
                candidate=candidate,
                evaluation=evaluation,
                context=context,
                max_output=max_output,
                seed=evaluation.seed,
            )
        )
    return tuple(resolved)
