# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Choice terms fold several likelihood requests into one score per choice."""

import math

import pytest
from pydantic import ValidationError

from taskcompendium.evaluation.metrics import likelihood_metrics
from taskcompendium.evaluation.spec import (
    ChoiceLikelihood,
    EvaluationProtocol,
    LikelihoodContext,
    LikelihoodMetric,
    LikelihoodNormalization,
    LikelihoodRequest,
    Metric,
    MetricKind,
    SummaryKind,
    TrialPolicy,
    UnscorableChoice,
)

ALL_SINGLE = (True,) * 5
PROTOCOL = EvaluationProtocol(
    benchmark="terms",
    version="1",
    operation="choice_likelihood",
    sampling=None,
    trials=TrialPolicy(count=1, seed=None),
    metrics=(Metric(name="acc", kind=MetricKind.BINARY, higher_is_better=True),),
    primary_metric="acc",
    summary=SummaryKind.MEAN,
    expected_population=1,
    likelihood_metrics=(LikelihoodMetric(name="acc", normalization=LikelihoodNormalization.RAW),),
)


def request(context: str, continuation: str) -> LikelihoodRequest:
    return LikelihoodRequest(context=context, continuation=continuation)


# Two option orders; the first order pools the "A" and " A" spellings of option 0.
ROTATED = ChoiceLikelihood(
    choices=("first", "second"),
    requests=(request("x", "A"), request("x", " A"), request("y", "B"), request("x", "B"), request("y", "A")),
    gold=(0,),
    terms=(((0, 1), (2,)), ((3,), (4,))),
)


def test_pooled_spellings_change_the_selected_choice():
    likelihoods = tuple(map(math.log, (0.2, 0.1, 0.3, 0.4, 0.2)))

    first, second = ROTATED.choice_scores(likelihoods, ALL_SINGLE)

    assert first == pytest.approx(math.log(0.3 * 0.3))
    assert second == pytest.approx(math.log(0.4 * 0.2))
    # Without pooling, option 0 would score log(0.2 * 0.3) and lose to option 1.
    assert likelihood_metrics(ROTATED, likelihoods, PROTOCOL, single_token=ALL_SINGLE) == {"acc": 1.0}


def test_pooling_drops_alternatives_the_tokenizer_splits():
    # " A" is two tokens on this tokenizer, so option 0 keeps only the bare "A" in the first order.
    likelihoods = tuple(map(math.log, (0.2, 0.1, 0.3, 0.4, 0.2)))
    split_space = (True, False, True, True, True)

    first, _ = ROTATED.choice_scores(likelihoods, split_space)

    assert first == pytest.approx(math.log(0.2 * 0.3))
    assert likelihood_metrics(ROTATED, likelihoods, PROTOCOL, single_token=split_space) == {"acc": 0.0}


def test_a_pooled_group_without_a_single_token_alternative_is_unscorable():
    with pytest.raises(UnscorableChoice, match="encodes as one token"):
        ROTATED.scored_terms((False, False, True, True, True))


def test_counterbalanced_ties_select_the_first_choice():
    # Each choice is labelled "1" in one order and "2" in the other; a label-only preference ties exactly.
    counterbalanced = ChoiceLikelihood(
        choices=("act", "refrain"),
        requests=(request("n", " 1"), request("n", " 2"), request("s", " 1"), request("s", " 2")),
        gold=(1,),
        terms=(((0,), (3,)), ((1,), (2,))),
    )
    likelihoods = (-0.576, -0.826, -0.576, -0.826)

    first, second = counterbalanced.choice_scores(likelihoods)

    assert first == second
    assert likelihood_metrics(counterbalanced, likelihoods, PROTOCOL) == {"acc": 0.0}


def test_default_fields_are_omitted_so_existing_items_serialize_unchanged():
    plain = ChoiceLikelihood(choices=("a", "b"), requests=(request("q", " a"), request("q", " b")), gold=(1,))
    chat = ROTATED.model_copy(update={"context_format": LikelihoodContext.USER_TURN})

    assert set(plain.model_dump(mode="json")) == {"kind", "choices", "requests", "gold"}
    assert ChoiceLikelihood.model_validate(chat.model_dump(mode="json")) == chat


@pytest.mark.parametrize(
    "terms",
    [
        (((0, 1),), ((3,), (4,))),  # request 2 is never scored
        (((0, 1), (2,)),),  # one choice has no terms
        (((0, 1), ()), ((3,), (4,), (2,))),  # an empty group
    ],
)
def test_terms_must_cover_every_choice_and_request(terms):
    with pytest.raises(ValidationError, match="Choice terms"):
        ChoiceLikelihood.model_validate({**ROTATED.model_dump(mode="json"), "terms": terms})
