# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combine a policy's evaluations and exceptions into the settings for one candidate."""

from dataclasses import dataclass

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
    seed: int

    def __post_init__(self) -> None:
        if self.context > self.candidate.max_context:
            raise ValueError(
                f"{self.candidate.name} on {self.evaluation.name}: context {self.context} is larger than "
                f"the candidate's window {self.candidate.max_context}"
            )
        if self.max_output is not None and self.max_output >= self.context:
            raise ValueError(
                f"{self.candidate.name} on {self.evaluation.name}: output limit {self.max_output} does not fit "
                f"in context {self.context}; add an override"
            )


def resolve(policy: EvalPolicy, candidate: PolicyCandidate) -> tuple[ResolvedPolicyEvaluation, ...]:
    """Return every evaluation the candidate runs under the policy, in policy order.

    Excluded evaluations are left out. Each setting comes from the candidate's override, else the
    evaluation, and context is capped at the candidate's window.
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
                context=min(context, candidate.max_context),
                max_output=max_output,
                seed=evaluation.seed,
            )
        )
    return tuple(resolved)
