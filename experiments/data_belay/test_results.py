# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from experiments.data_belay.accuracy import TASKS, Question, accuracies
from experiments.data_belay.judge import CHOICE_METRIC_PREFIX, GROUP_METRICS
from experiments.data_belay.results import (
    DILUTION_REFERENCES,
    dilution_references,
    metrics_from,
    parse_scores,
    scores_json,
)


def test_report_reads_choice_loss_in_the_format_the_scorer_writes():
    question = Question("Q:", (" a", " b"), gold=0)
    written = parse_scores(scores_json({task.name: accuracies([question], [-1.0, -2.0]) for task in TASKS}))
    summary = {key: 2.0 for keys in GROUP_METRICS.values() for key in keys if not key.startswith(CHOICE_METRIC_PREFIX)}
    metrics = metrics_from(summary, written)
    assert metrics["choice/mmlu_pro"] == metrics["choice/truthful_qa"] == pytest.approx(math.log2(1 + math.exp(-1.0)))
    assert metrics["eval/eval_policy/knowledge/trivia_qa/bpb"] == 2.0


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ("fineweb-edu-sample-10bt-p20-006b29", DILUTION_REFERENCES[20]),
        # The first pilot caches' labels carry the share without a cache hash.
        ("finemath-p20", DILUTION_REFERENCES[20]),
        # A reference is judged against the others.
        (DILUTION_REFERENCES[20][0], DILUTION_REFERENCES[20][1:]),
        # No references at 50%, and none for the control.
        ("fineweb-sample-10bt-p50-d497e2", ()),
        ("control", ()),
    ],
)
def test_a_candidate_is_judged_against_the_dilution_references_at_its_share(candidate, expected):
    assert dilution_references(candidate) == expected
