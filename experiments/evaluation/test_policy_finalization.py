# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Automatic policy completion keeps raw evidence and reports corrected scores."""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
from finestore.eval import EvalSample, EvaluationStore, Grading, SampleKind
from finestore.reader import ReadView
from marin.evaluation.records import BenchmarkMetadataRef, BenchmarkMetricRef, EvalTaskRef, MetricKind, TaskCoverage
from marin.evaluation.runner import EvaluationOutcome

from experiments.evaluation.policy_finalization import SOTOPIA_ARTIFACT, CohortSource, finalize_sotopia, trial_coverage

RANGES = {
    "believability": (0, 10),
    "relationship": (-5, 5),
    "knowledge": (0, 10),
    "secret": (-10, 0),
    "social_rules": (-10, 0),
    "financial_and_material_benefits": (-5, 5),
    "goal": (0, 10),
}


def sotopia_archive(root: Path, length_offset: int) -> float:
    scores = []
    with EvaluationStore.open(str(root), writer_id="harbor") as store:
        for index in range(100):
            length = 100 + length_offset + index * (2 if length_offset else 1)
            score = 1 / (1 + math.exp(-(0.5 + index % 5 * 0.03 + 0.8 * math.tanh((length - 200) / 150))))
            scores.append(score)
            task = f"task-{index}"
            trial = f"trial-{index}"
            metrics = {"sotopia_" + name: score for name in RANGES}
            store.add_sample(
                EvalSample(
                    task="sotopia-hard",
                    doc_id=task,
                    kind=SampleKind.AGENTIC,
                    metrics=metrics,
                    grading=Grading(method="harbor", metric="reward", score=score),
                ),
                trial_id=trial,
            )
            summary = {
                "evaluated_agent_index": 0,
                "scores": [{name: minimum + score * (maximum - minimum) for name, (minimum, maximum) in RANGES.items()}],
                "transcript": [
                    [{"sender": "Environment", "receiver": "Alice"}, {"sender": "Environment", "receiver": "Bob"}],
                    [{"sender": "Alice", "receiver": "Bob", "message": "x" * length}],
                ],
            }
            path = root / "harbor_jobs" / "job" / trial / "attempts" / "0" / "agent" / "sotopia-summary.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(summary))
        store.seal()
    return sum(scores) / len(scores)


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


def test_sotopia_finalization_preserves_raw_rewards_and_updates_the_canonical_score(tmp_path):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    sotopia_archive(baseline, 0)
    raw = sotopia_archive(candidate, 200)
    final = finalize_sotopia(outcome(raw), str(candidate), "candidate", (CohortSource("baseline", str(baseline)),))
    reader = ReadView(str(candidate))
    rows = reader.scan("samples").to_pylist(maps_as_pydicts="strict")
    adjusted = [row["metrics"]["length_controlled_overall"] for row in rows]
    assert final.canonical_metrics["sotopia-hard"]["length_controlled_overall"] == pytest.approx(sum(adjusted) / 100)
    assert final.canonical_metrics["sotopia-hard"]["length_controlled_overall"] != pytest.approx(raw, abs=0.01)
    assert sum(row["metrics"]["sotopia_goal"] for row in rows) / 100 == pytest.approx(raw)
    assert final.tasks[0].benchmark.primary_metric == "length_controlled_overall"
    evidence = json.loads(reader.read_blob(SOTOPIA_ARTIFACT))
    assert {row["model"] for row in evidence["scores"]} == {"baseline", "candidate"}
    assert len(evidence["adjusted_episodes"]) == 200
    repeated = finalize_sotopia(outcome(raw), str(candidate), "candidate", (CohortSource("baseline", str(baseline)),))
    assert repeated.canonical_metrics == final.canonical_metrics


def test_incomplete_cohort_does_not_publish_an_uncorrected_score(tmp_path):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    sotopia_archive(baseline, 0)
    raw = sotopia_archive(candidate, 200)
    next((baseline / "harbor_jobs").rglob("sotopia-summary.json")).unlink()
    with pytest.raises(ValueError, match="missing SOTOPIA summary"):
        finalize_sotopia(outcome(raw), str(candidate), "candidate", (CohortSource("baseline", str(baseline)),))
    reader = ReadView(str(candidate))
    assert reader.read_blob(SOTOPIA_ARTIFACT) is None
    assert all(
        "length_controlled_overall" not in row["metrics"]
        for row in reader.scan("samples").to_pylist(maps_as_pydicts="strict")
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
