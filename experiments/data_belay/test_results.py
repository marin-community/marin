# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math

import pytest

from experiments.data_belay.__main__ import LAUNCH_VERSION, rung_tokens
from experiments.data_belay.accuracy import TASKS, Question, accuracies
from experiments.data_belay.judge import CHOICE_METRIC_PREFIX, GROUP_METRICS
from experiments.data_belay.results import (
    DILUTION_REFERENCES,
    TrainedMix,
    dilution_references,
    metrics_from,
    parse_scores,
    scores_json,
    trained_mix,
)
from experiments.grug.fast_track.launch import TrainMix, build_h100_ladder_run


def test_report_reads_choice_loss_in_the_format_the_scorer_writes():
    question = Question("Q:", (" a", " b"), gold=0)
    written = parse_scores(scores_json({task.name: accuracies([question], [-1.0, -2.0]) for task in TASKS}))
    summary = {key: 2.0 for keys in GROUP_METRICS.values() for key in keys if not key.startswith(CHOICE_METRIC_PREFIX)}
    metrics = metrics_from(summary, written)
    assert metrics["choice/mmlu_pro"] == metrics["choice/truthful_qa"] == pytest.approx(math.log2(1 + math.exp(-1.0)))
    assert metrics["eval/eval_policy/knowledge/trivia_qa/bpb"] == 2.0


@pytest.mark.parametrize(
    ("candidate", "share", "expected"),
    [
        ("fineweb-edu-sample-10bt-p20-006b29", 0.2, DILUTION_REFERENCES[0.2]),
        # A reference is judged against the others.
        (DILUTION_REFERENCES[0.2][0], 0.2, DILUTION_REFERENCES[0.2][1:]),
        # A label that rounds the share does not match: runs at 19.6% have no references.
        ("fineweb-sample-10bt-p20-abcdef", 0.196, ()),
        # No references at 50%, and none for the control, which gives up no share.
        ("fineweb-sample-10bt-p50-d497e2", 0.5, ()),
        ("control", 0.0, ()),
    ],
)
def test_a_candidate_is_judged_against_the_dilution_references_at_the_share_its_runs_trained_at(
    candidate, share, expected
):
    assert dilution_references(candidate, share) == expected


def test_report_reads_the_share_cache_and_tokens_that_the_launcher_gives_a_run():
    # The launcher's config is what a run logs to W&B, where report reads it.
    root = "s3://marin-us-east-02a/marin/data_belay/caches/finemath/e92b25a61673/finemath-4plus-train-4ab2d8f8d6"
    treatment = build_h100_ladder_run(
        run_id="t", size="d512", dense=True, version=LAUNCH_VERSION, mix=TrainMix(f"{root}/train", 0.2)
    )
    control = build_h100_ladder_run(run_id="c", size="d512", dense=True, version=LAUNCH_VERSION)
    assert trained_mix(json.loads(treatment.fingerprint_payload())) == TrainedMix(0.2, root, rung_tokens("d512"))
    assert trained_mix(json.loads(control.fingerprint_payload())) is None
