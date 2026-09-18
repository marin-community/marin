# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
from marin.testing.inference.hero_forward_goldens import (
    REQUIRED_OBSERVATIONS,
    ComparisonTolerances,
    GoldenBundle,
    compare_observations,
)

FIXTURE = Path(__file__).parents[2] / "lib/marin/src/marin/testing/inference/resources/hero_forward_fixture_v1"
COMPARISON_TOLERANCE = 0.01
TOLERANCES = ComparisonTolerances(
    target_logprob=COMPARISON_TOLERANCE,
    top_logprob=COMPARISON_TOLERANCE,
    full_logit=COMPARISON_TOLERANCE,
    route_combine_weight=COMPARISON_TOLERANCE,
    route_cutoff_gap=COMPARISON_TOLERANCE,
)


def _observations(bundle: GoldenBundle) -> dict[str, np.ndarray]:
    return {name: bundle.arrays[name].copy() for name in REQUIRED_OBSERVATIONS}


def test_hero_forward_comparator_accepts_numeric_changes_within_bounds() -> None:
    bundle = GoldenBundle.load(FIXTURE)
    observations = _observations(bundle)
    observations["target_logprobs"][0] += COMPARISON_TOLERANCE / 2
    observations["top_logprobs"][0, 0] -= COMPARISON_TOLERANCE / 2
    observations["full_logits"][0, 0] += COMPARISON_TOLERANCE / 2
    observations["route_combine_weights"][0, 0, 0, 0] += COMPARISON_TOLERANCE / 2
    observations["route_cutoff_gaps"][0, 0, 0] += COMPARISON_TOLERANCE / 2

    report = compare_observations(bundle, observations, TOLERANCES)

    assert report.ok


def test_hero_forward_comparator_reports_missing_score() -> None:
    bundle = GoldenBundle.load(FIXTURE)
    observations = _observations(bundle)
    del observations["target_logprobs"]

    report = compare_observations(bundle, observations, TOLERANCES)

    assert [(issue.kind, issue.field) for issue in report.issues] == [("missing", "target_logprobs")]


def test_hero_forward_comparator_reports_shifted_alignment() -> None:
    bundle = GoldenBundle.load(FIXTURE)
    observations = _observations(bundle)
    observations["prediction_positions"][0] += 1

    report = compare_observations(bundle, observations, TOLERANCES)

    assert any(issue.kind == "alignment" and issue.field == "prediction_positions" for issue in report.issues)


def test_hero_forward_comparator_reports_numerical_error_beyond_bound() -> None:
    bundle = GoldenBundle.load(FIXTURE)
    observations = _observations(bundle)
    observations["target_logprobs"][1] += 0.02

    report = compare_observations(bundle, observations, TOLERANCES)

    issue = next(issue for issue in report.issues if issue.field == "target_logprobs")
    assert issue.kind == "numerical"
    assert issue.count == 1
    assert issue.max_abs_error == pytest.approx(0.02)


def test_hero_forward_comparator_reports_well_separated_route_change() -> None:
    bundle = GoldenBundle.load(FIXTURE)
    observations = _observations(bundle)
    observations["route_expert_ids"][0, 0, 1, 0] = 7

    report = compare_observations(bundle, observations, TOLERANCES)

    issue = next(issue for issue in report.issues if issue.field == "route_expert_ids")
    assert issue.kind == "routing"
    assert issue.count == 1
    assert issue.well_separated_count == 1


def test_hero_forward_comparator_does_not_excuse_small_gap_route_change() -> None:
    bundle = GoldenBundle.load(FIXTURE)
    observations = _observations(bundle)
    observations["route_expert_ids"][0, 0, 0, 0] = 7

    report = compare_observations(bundle, observations, TOLERANCES)

    issue = next(issue for issue in report.issues if issue.field == "route_expert_ids")
    assert issue.count == 1
    assert issue.well_separated_count == 0


def test_hero_forward_report_raises_with_structured_issue_summary() -> None:
    bundle = GoldenBundle.load(FIXTURE)
    observations = _observations(bundle)
    observations["full_logits"][0, 0] += 0.02

    with pytest.raises(AssertionError):
        compare_observations(bundle, observations, TOLERANCES).raise_for_errors()
