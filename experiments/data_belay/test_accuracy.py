# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.data_belay.accuracy import Question, accuracies


def test_acc_norm_divides_each_choice_by_its_bytes():
    # The short choice has the higher total log-likelihood, the long one the higher per byte.
    question = Question("Q:", (" a", " a much longer answer"), gold=1)
    result = accuracies([question], [-2.0, -3.0])
    assert result["acc"] == 0.0
    assert result["acc_norm"] == 1.0


def test_rows_are_grouped_back_into_questions_in_order():
    questions = [Question("Q1:", (" x", " y"), gold=1), Question("Q2:", (" x", " y", " z"), gold=0)]
    result = accuracies(questions, [-3.0, -1.0, -1.0, -2.0, -3.0])
    assert result == {"acc": 1.0, "acc_norm": 1.0, "questions": 2}
