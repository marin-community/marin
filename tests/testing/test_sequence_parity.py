# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest
from marin.testing.inference.sequence_parity import (
    AccuracyObservation,
    ScoredSequence,
    ScoredToken,
    SequenceThresholds,
    compare_accuracy,
    compare_sequences,
)


def observations(delta_after_window=0.0):
    return ScoredSequence(
        "long",
        "fixed-input",
        tuple(
            ScoredToken(i, None if i == 0 else -1.0 - (delta_after_window if i >= 2048 else 0), None if i == 0 else i)
            for i in range(4097)
        ),
    )


def test_window_regression_is_localized_even_when_pooled_mean_passes():
    reference = observations()
    broken = observations(0.06)
    result = compare_sequences(
        [broken], [reference], boundaries=(0, 1024, 2048, 2304, 4097), thresholds=SequenceThresholds(0.04, 0.04, 0.03)
    )
    assert result.pooled.mean_absolute_delta < 0.04
    assert result.failures == ("band_mean:2048:2304", "band_mean:2304:4097")
    assert result.bands[1].mean_absolute_delta == 0
    assert result.bands[2].mean_absolute_delta == pytest.approx(0.06)


@pytest.mark.parametrize("fault", ["missing", "shifted", "nan"])
def test_incomplete_or_invalid_teacher_response_cannot_pass(fault):
    reference = observations()
    tokens = list(reference.tokens)
    if fault == "missing":
        tokens.pop()
    elif fault == "shifted":
        tokens[2048] = dataclasses.replace(tokens[2048], token_id=99999)
    else:
        tokens[2048] = dataclasses.replace(tokens[2048], logprob=float("nan"))
    with pytest.raises(ValueError):
        compare_sequences(
            [dataclasses.replace(reference, tokens=tuple(tokens))],
            [reference],
            boundaries=(0, 2048, 4097),
            thresholds=SequenceThresholds(0.025, 0.04, 0.03),
        )


def test_equal_accuracy_does_not_hide_changed_truncation():
    reference = [AccuracyObservation("a", True, "stop"), AccuracyObservation("b", False, "length")]
    candidate = [AccuracyObservation("a", True, "length"), AccuracyObservation("b", False, "stop")]
    result = compare_accuracy(candidate, reference, max_z=2)
    assert result.z == 0
    assert result.changed_length_finishes == ("a", "b")
    assert result.failures == ("length_finishes",)


def test_zero_variance_opposite_scores_fail_and_identical_scores_pass():
    perfect = [AccuracyObservation(str(i), True, "stop") for i in range(10)]
    wrong = [dataclasses.replace(item, correct=False) for item in perfect]
    assert compare_accuracy(wrong, perfect, max_z=2).failures == ("accuracy_z",)
    assert compare_accuracy(perfect, perfect, max_z=2).failures == ()
