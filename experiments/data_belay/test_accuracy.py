# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from experiments.data_belay.accuracy import Question, accuracies


def test_acc_norm_divides_each_choice_by_its_bytes():
    # The short choice has the higher total log-likelihood, the long one the higher per byte.
    question = Question("Q:", (" a", " a much longer answer"), gold=1)
    result = accuracies([question], [-2.0, -3.0])
    assert result.acc == 0.0
    assert result.acc_norm == 1.0


def test_rows_are_grouped_back_into_questions_in_order():
    questions = [Question("Q1:", (" x", " y"), gold=1), Question("Q2:", (" x", " y", " z"), gold=0)]
    result = accuracies(questions, [-3.0, -1.0, -1.0, -2.0, -3.0])
    assert (result.acc, result.acc_norm, result.questions) == (1.0, 1.0, 2)


def test_choice_bits_track_the_gold_margin_that_accuracy_ignores():
    question = Question("Q:", (" x", " y"), gold=0)
    tied = accuracies([question], [-1.0, -1.0])
    narrow = accuracies([question], [-1.0, -1.5])
    wide = accuracies([question], [-1.0, -4.0])
    assert tied.choice_bits == pytest.approx(1.0)  # chance among two choices
    assert narrow.acc == wide.acc == 1.0
    assert wide.choice_bits < narrow.choice_bits < tied.choice_bits
    assert wide.choice_bits == pytest.approx(math.log2(1 + math.exp(-3.0)))
