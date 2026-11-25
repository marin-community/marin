# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combine a policy's evaluations and exceptions into the settings for one candidate."""

from dataclasses import dataclass

from evalrecords.records import EvaluationSelection
from marin.evaluation.model_config import prompt_format

from experiments.evaluation.curated import selection_dataset
from experiments.evaluation.policy.schema import EvalPolicy, PolicyCandidate, PolicyEvaluation
from experiments.evaluation.preparation import EvaluationCorpusRecipe


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
        if self.max_output is not None and self.max_output >= self.context:
            raise ValueError(
                f"{self.candidate.name} on {self.evaluation.name}: output limit {self.max_output} does not fit "
                f"in context {self.context}; add an override"
            )


def catalog_variant(evaluation: PolicyEvaluation, candidate: PolicyCandidate) -> EvaluationCorpusRecipe:
    """The catalog variant the candidate runs: the evaluation's, in the candidate's chat or completion format."""
    return selection_dataset(evaluation.selection, prompt_format(candidate.model))


def _context(policy: EvalPolicy, candidate: PolicyCandidate, evaluation: PolicyEvaluation, context: int | None) -> int:
    """The context an evaluation runs with.

    A protocol window is the one its items were built for, so a run uses exactly it, and a candidate served
    below it must be excluded. Otherwise the run gets the smallest of the window the candidate is served at,
    the requested context (else the policy's) and any agent input limit, as the worker gives an agent.
    """
    variant = catalog_variant(evaluation, candidate)
    served = min(candidate.max_context, candidate.model.serve.max_model_len or candidate.max_context)
    name = f"{candidate.name} on {evaluation.name}"
    if (window := variant.protocol.context_window) is not None:
        if window > served:
            raise ValueError(
                f"{name}: the benchmark's window {window} is larger than the served window {served}; exclude it"
            )
        if context is not None and context != window:
            raise ValueError(f"{name}: context {context} is not the benchmark's window {window}")
        return window
    limits = [served, policy.context if context is None else context]
    if variant.agent is not None and variant.agent.max_input_tokens is not None:
        limits.append(variant.agent.max_input_tokens)
    return min(limits)


def resolve(policy: EvalPolicy, candidate: PolicyCandidate) -> tuple[ResolvedPolicyEvaluation, ...]:
    """Return every evaluation the candidate runs under the policy.

    Excluded evaluations are left out. Each setting comes from the candidate's override, else the
    evaluation; the context is then fitted to the benchmark and the candidate (``_context``).
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
