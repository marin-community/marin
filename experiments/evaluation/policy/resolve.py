# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combine a policy's evaluations and exceptions into the settings for one candidate."""

from dataclasses import dataclass

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


@dataclass(frozen=True)
class BenchmarkWindow:
    """A window a benchmark declares: its protocol's fixed window, or its agent's input limit.

    An agent's input limit is a ``cap``: the worker gives the agent the smaller of it and the served
    window, so a smaller context still runs the benchmark. A protocol window is the one its items were
    built for.
    """

    tokens: int
    cap: bool


def benchmark_window(evaluation: PolicyEvaluation, candidate: PolicyCandidate) -> BenchmarkWindow | None:
    """Return the benchmark's agent input limit or protocol window, if declared."""
    dataset = selection_dataset(evaluation.selection, prompt_format(candidate.model))
    if dataset.agent is not None and dataset.agent.max_input_tokens is not None:
        return BenchmarkWindow(tokens=dataset.agent.max_input_tokens, cap=True)
    if dataset.protocol.context_window is not None:
        return BenchmarkWindow(tokens=dataset.protocol.context_window, cap=False)
    return None


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
        window = benchmark_window(evaluation, candidate)
        if window is not None:
            # A benchmark that declares its window runs on the candidate's served window, not one sized for it.
            served = min(candidate.max_context, candidate.model.serve.max_model_len or candidate.max_context)
            if not window.cap and window.tokens > served:
                raise ValueError(
                    f"{candidate.name} on {evaluation.name}: the benchmark's window {window.tokens} is larger than "
                    f"the candidate's served window {served}; exclude it"
                )
        if context is None:
            context = policy.context if window is None else window.tokens
        elif window is not None and (context > window.tokens if window.cap else context != window.tokens):
            limit = "above the agent input limit" if window.cap else "not the window"
            raise ValueError(f"{candidate.name} on {evaluation.name}: context {context} is {limit} {window.tokens}")
        if window is not None and context > served:
            raise ValueError(
                f"{candidate.name} on {evaluation.name}: context {context} is larger than the candidate's served "
                f"window {served}; add an override"
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
