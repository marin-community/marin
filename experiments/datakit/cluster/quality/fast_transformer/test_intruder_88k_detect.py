# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for the 88k-join intruder detection driver."""

from dataclasses import dataclass

from experiments.datakit.cluster.intruder import DOCS_PER_TRIAL, Bucket, BucketPool, IntruderTrial
from experiments.datakit.cluster.quality.fast_transformer.intruder_88k_detect import (
    _select_sources,
    run_detection,
)


def _cells(source: str, level_sizes: dict[int, int]) -> dict[tuple[str, int], int]:
    return {(source, level): n for level, n in level_sizes.items()}


def test_select_sources_prefers_fuller_quality_coverage():
    cells = {
        **_cells("two-levels-big", {1: 500, 5: 500}),
        **_cells("all-levels", {q: 20 for q in range(1, 6)}),
        **_cells("one-level", {3: 500}),
        **_cells("thin-cells", {q: 5 for q in range(1, 6)}),
    }
    chosen = _select_sources(cells)
    # Five covered levels beat raw size; a single eligible level cannot form a
    # trial, and cells below the floor never count as coverage.
    assert chosen[0] == "all-levels"
    assert "two-levels-big" in chosen
    assert "one-level" not in chosen
    assert "thin-cells" not in chosen


def test_select_sources_stays_within_the_document_budget():
    # Nine full sources contribute 320 documents each; the 2,560 budget fits
    # exactly eight of them.
    cells = {}
    for k in range(9):
        cells.update(_cells(f"source-{k}", {q: 100 for q in range(1, 6)}))
    chosen = _select_sources(cells)
    assert len(chosen) == 8


def _pool() -> BucketPool:
    buckets = [
        Bucket(f"src|q{level}", [f"src q{level} doc {i} {'filler ' * 5}" for i in range(8)]) for level in range(3)
    ]
    return BucketPool("fake", buckets, stratum_of=lambda key: key.rsplit("|q", 1)[0])


@dataclass
class FixedPanelist:
    """Votes correctly with certainty, or always misses by one position."""

    name: str
    correct: bool

    def vote(self, trial: IntruderTrial, *, max_doc_chars: int) -> int:
        if self.correct:
            return trial.intruder_index
        return (trial.intruder_index + 1) % DOCS_PER_TRIAL


def test_detection_resolves_above_chance_for_a_sharp_panel():
    result = run_detection(_pool(), [FixedPanelist("sharp", correct=True)], max_trials=150)
    assert result["decision"] == "above_chance"
    assert result["detection_rate"] == 1.0
    assert result["interval"][0] > result["chance"]
    # The sequential test stops as soon as the interval clears the floor.
    assert result["n_trials"] < 150


def test_detection_resolves_below_chance_for_a_systematically_wrong_panel():
    result = run_detection(_pool(), [FixedPanelist("wrong", correct=False)], max_trials=150)
    assert result["decision"] == "below_chance"
    assert result["interval"][1] < result["chance"]


def test_detection_resumes_from_a_prior_unresolved_result():
    # An unresolved 100-trial run at the chance rate, resumed with a sharp panel:
    # the carried trials stay in the tallies (the rate lands strictly between the
    # prior 0.2 and the extension's 1.0) and only the extension is attempted.
    prior = {
        "n_trials": 100,
        "n_attempted": 100,
        "n_abstained": 3,
        "detection_rate": 0.2,
        "per_model": {"sharp": {"accuracy": 0.2, "votes": 100}},
    }
    result = run_detection(_pool(), [FixedPanelist("sharp", correct=True)], max_trials=200, prior=prior)
    assert result["n_trials"] > 100
    assert result["n_attempted"] <= 200
    assert result["n_abstained"] == 3
    assert 0.2 < result["detection_rate"] < 1.0
    votes = result["per_model"]["sharp"]["votes"]
    assert votes > 100
    assert result["per_model"]["sharp"]["accuracy"] == (20 + votes - 100) / votes
