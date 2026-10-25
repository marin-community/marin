# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Automatic policy completion keeps raw evidence and reports corrected scores."""

from dataclasses import replace

import pytest
from finestore.eval import EvalSample, EvaluationStore, Grading, SampleKind
from marin.evaluation.records import BenchmarkMetadataRef, BenchmarkMetricRef, EvalTaskRef, MetricKind, TaskCoverage
from marin.evaluation.runner import EvaluationOutcome

from experiments.evaluation.policy_finalization import trial_coverage


def outcome(score: float) -> EvaluationOutcome:
    return EvaluationOutcome(
        metrics={"sotopia-hard": {"reward": score}},
        canonical_metrics={"sotopia-hard": {"reward": score}},
        coverage={"sotopia-hard": TaskCoverage(n_benchmark=100, n_attempted=100, n_scored=100)},
        tasks=(
            EvalTaskRef(
                name="sotopia-hard",
                num_fewshot=None,
                benchmark=BenchmarkMetadataRef(
                    schema_version=1,
                    task="sotopia-hard",
                    primary_metric="reward",
                    metric_kind=MetricKind.CONTINUOUS,
                    metrics=(
                        BenchmarkMetricRef(
                            name="reward", source_name="reward", kind=MetricKind.CONTINUOUS, higher_is_better=True
                        ),
                    ),
                    n_benchmark=100,
                    n_attempted=100,
                ),
            ),
        ),
    )


@pytest.mark.parametrize("observed_repetitions", [2, 3])
def test_repeated_judgments_count_declared_trials_including_missing_repetitions(tmp_path, observed_repetitions):
    original = outcome(0.5)
    reference = original.tasks[0].model_copy(update={"name": "OlympiadBench"})
    original = replace(
        original,
        metrics={"OlympiadBench": {"num_repeat": observed_repetitions, "accuracy_avg": 0.5}},
        canonical_metrics={"OlympiadBench": {"accuracy_avg": 0.5}},
        coverage={"OlympiadBench": TaskCoverage(n_attempted=2, n_benchmark=2, n_scored=2)},
        tasks=(reference,),
    )
    with EvaluationStore.open(str(tmp_path), writer_id="evalchemy") as store:
        for doc in range(2):
            for trial in range(observed_repetitions):
                score = float((doc + trial) % 2)
                store.add_sample(
                    EvalSample(
                        task="OlympiadBench",
                        doc_id=str(doc),
                        kind=SampleKind.GENERATION,
                        extracted="answer",
                        correct=bool(score),
                        grading=Grading(method="llm_judge", metric="accuracy", score=score),
                    ),
                    trial_id=str(trial),
                )
        store.seal()
    final = trial_coverage(original, str(tmp_path), repetitions=3)
    assert final.coverage["OlympiadBench"].n_scored == 2 * observed_repetitions
    assert final.coverage["OlympiadBench"].n_attempted == 6
    assert final.coverage["OlympiadBench"].n_correct == observed_repetitions
    assert final.tasks[0].benchmark.n_attempted == 6
    assert final.canonical_metrics == original.canonical_metrics
