# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combine a policy's evaluations and exceptions into the settings for one candidate."""

from collections.abc import Collection
from dataclasses import dataclass

from evalrecords.records import EvaluationOverrides, EvaluationSelection
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
        run = EvaluationOverrides.model_validate(
            {**selection.run.model_dump(), "max_tokens": self.max_output, "seed": self.seed}
        )
        return selection.model_copy(update={"run": run})

    def __post_init__(self) -> None:
        if self.max_output is not None and self.max_output >= self.context:
            raise ValueError(
                f"{self.candidate.name} on {self.evaluation.name}: output limit {self.max_output} does not fit "
                f"in context {self.context}"
            )


def catalog_variant(evaluation: PolicyEvaluation, candidate: PolicyCandidate) -> EvaluationCorpusRecipe:
    """The catalog variant the candidate runs: the evaluation's, in the candidate's chat or completion format."""
    return selection_dataset(evaluation.selection, prompt_format(candidate.model))


def serving_window(policy: EvalPolicy, candidate: PolicyCandidate) -> int:
    """The window the candidate is served at: the policy's, or the candidate's own if smaller."""
    own = min(candidate.max_context, candidate.model.serve.max_model_len or candidate.max_context)
    return min(own, policy.serving_window)


def _context(policy: EvalPolicy, candidate: PolicyCandidate, evaluation: PolicyEvaluation) -> int:
    """The context an evaluation runs with on the candidate's server.

    A protocol window is the one its items were built for, so a run uses exactly it, and a candidate served below
    it must be excluded. An agent gets the smaller of its input limit and the serving window, as the worker gives
    it; any other evaluation runs at the serving window.
    """
    window = serving_window(policy, candidate)
    variant = catalog_variant(evaluation, candidate)
    if (protocol_window := variant.protocol.context_window) is not None:
        if protocol_window > window:
            raise ValueError(
                f"{candidate.name} on {evaluation.name}: the benchmark's window {protocol_window} is larger than "
                f"the serving window {window}; exclude it"
            )
        return protocol_window
    if variant.agent is not None and variant.agent.max_input_tokens is not None:
        return min(variant.agent.max_input_tokens, window)
    return window


def resolve(
    policy: EvalPolicy, candidate: PolicyCandidate, evaluations: Collection[str] | None = None
) -> tuple[ResolvedPolicyEvaluation, ...]:
    """Return every evaluation the candidate runs under the policy.

    ``evaluations`` limits it to those names, whose constraints alone are checked. Excluded evaluations are left
    out, and a candidate's override replaces an evaluation's output limit. A candidate with no matching
    exclusions or overrides uses every policy evaluation.
    """
    excluded = {item.evaluation.name for item in policy.exclusions if item.candidate == candidate}
    overrides = {item.evaluation.name: item for item in policy.overrides if item.candidate == candidate}
    resolved = []
    for evaluation in policy.in_distribution:
        if evaluation.name in excluded or (evaluations is not None and evaluation.name not in evaluations):
            continue
        override = overrides.get(evaluation.name)
        max_output = evaluation.max_output if override is None else override.max_output
        resolved.append(
            ResolvedPolicyEvaluation(
                candidate=candidate,
                evaluation=evaluation,
                context=_context(policy, candidate, evaluation),
                max_output=max_output,
                seed=evaluation.seed,
            )
        )
    return tuple(resolved)
