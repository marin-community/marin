# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.snowball_inventory_evaluate import EvaluationOutcome, PanelTask, summarize_panel


@pytest.fixture
def panel():
    return [
        PanelTask("calendar-a", "calendar", "contract-a"),
        PanelTask("calendar-b", "calendar", "contract-b"),
        PanelTask("native-c", "structural", "contract-c"),
    ]


def test_missing_grade_remains_in_family_and_aggregate_denominators(panel):
    report = summarize_panel(
        panel,
        [
            EvaluationOutcome("calendar-a", 1.0, "stop", "verified"),
            EvaluationOutcome("native-c", None, "length", "unavailable", "ProviderRateLimitError"),
        ],
    )

    assert report["complete"] is False
    assert report["aggregate_fixed_denominator_mean"] == pytest.approx(1 / 3)
    assert report["macro_family_mean"] == 0.25
    assert report["families"]["calendar"]["fixed_denominator_mean"] == 0.5
    assert report["families"]["calendar"]["coverage"] == 0.5
    assert report["families"]["structural"]["scored"] == 0
    assert report["families"]["structural"]["length_stops"] == 1
    assert report["families"]["structural"]["failure_categories"] == {"ProviderRateLimitError": 1}
    assert report["families"]["calendar"]["failure_categories"] == {}
    assert [row["attempted"] for row in report["tasks"]] == [True, False, True]


def test_pairing_identity_does_not_depend_on_model_scores_or_result_order(panel):
    baseline = summarize_panel(panel, [EvaluationOutcome(task.task_id, 0.0, "length") for task in panel])
    checkpoint = summarize_panel(panel, [EvaluationOutcome(task.task_id, 1.0, "stop") for task in reversed(panel)])

    assert baseline["pairing_identity"] == checkpoint["pairing_identity"]
    assert baseline["complete"] is True
    assert baseline["aggregate_fixed_denominator_mean"] == 0
    assert baseline["families"]["calendar"]["length_stops"] == 2
    assert checkpoint["aggregate_fixed_denominator_mean"] == 1
    assert [row["task_id"] for row in checkpoint["tasks"]] == [task.task_id for task in panel]


def test_duplicate_outcomes_cannot_inflate_fixed_panel_score(panel):
    with pytest.raises(ValueError, match="repeated evaluation task"):
        summarize_panel(panel, [EvaluationOutcome("calendar-a", 1.0, "stop")] * 2)
