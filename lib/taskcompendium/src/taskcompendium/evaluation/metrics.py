# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-example likelihood metrics."""

from verifyit.adapters.harness_native import likelihood_choice
from verifyit.modes.grade_mcq import LikelihoodScoring, grade_mcq_likelihoods

from taskcompendium.evaluation.spec import ChoiceLikelihood, ContinuationLikelihood, EvaluationProtocol


def likelihood_metrics(
    operation: ChoiceLikelihood | ContinuationLikelihood,
    scores: tuple[float, ...],
    protocol: EvaluationProtocol,
    *,
    is_greedy: tuple[bool, ...] = (),
) -> dict[str, float]:
    """Grade saved model likelihoods using the shared native harness comparison policy."""
    metrics = {}
    for metric in protocol.likelihood_metrics:
        if metric.scoring == "log_likelihood":
            if len(scores) != 1:
                raise ValueError("Continuation likelihood requires exactly one score")
            metrics[metric.name] = scores[0]
        elif metric.scoring == "greedy_accuracy":
            if len(is_greedy) != 1:
                raise ValueError("Greedy continuation accuracy requires token agreement evidence")
            metrics[metric.name] = float(is_greedy[0])
        elif isinstance(operation, ChoiceLikelihood):
            if metric.scoring == "probability_mass":
                metrics[metric.name] = grade_mcq_likelihoods(
                    scores,
                    operation.gold,
                    normalization_lengths=[1] * len(scores),
                    policy=LikelihoodScoring.PROBABILITY_MASS,
                ).reward
            else:
                metrics[metric.name] = likelihood_choice(
                    operation.choices, scores, operation.gold, normalization=metric.normalization
                ).reward
        else:
            raise ValueError("Choice selection requires a choice likelihood operation")
    return metrics
