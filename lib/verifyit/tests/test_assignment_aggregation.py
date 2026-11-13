# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from verifyit.grade import Reward, Status, aggregate_first_fit, infra_error, invalid_task, scored
from verifyit.modes.grade_json_schema import grade_json_schema_candidate


def test_reference_first_assignment_preserves_source_greedy_order():
    schemas = [{"pattern": "cat"}, {"pattern": "caterpillar"}]
    candidates = ["caterpillar", "cat"]
    edges = [[grade_json_schema_candidate(schema, candidate) for candidate in candidates] for schema in schemas]
    # A perfect assignment exists, but the source consumes the first broad match.
    assert aggregate_first_fit(edges, expected_total=2).reward == 0
    reordered = [[row[1], row[0]] for row in edges]
    assert aggregate_first_fit(reordered, expected_total=2).reward == 1


def test_candidate_first_assignment_preserves_source_alternative_order():
    schemas = [{"enum": ["cat", "dog"]}, {"enum": ["cat"]}]
    candidates = ["cat", "dog"]
    edges = [[grade_json_schema_candidate(schema, candidate) for schema in schemas] for candidate in candidates]
    assert aggregate_first_fit(edges, expected_total=2).reward == 0
    assert aggregate_first_fit(list(reversed(edges)), expected_total=2).reward == 1


@pytest.mark.parametrize("failure", [invalid_task("bad reference"), infra_error("grader failed")])
def test_unvisited_edge_failure_discards_complete_assignment(failure):
    result = aggregate_first_fit([[scored(1), failure], [scored(0), scored(1)]], expected_total=2)
    assert result.status == failure.status
    assert result.reward == 0


def test_infrastructure_failure_takes_precedence_over_invalid_task():
    result = aggregate_first_fit([[invalid_task("bad task"), infra_error("failed")]], expected_total=1)
    assert result.status == Status.INFRA_ERROR
    assert result.reward == 0


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 0.5])
def test_nonbinary_or_invalid_edge_is_not_accepted(value):
    result = aggregate_first_fit([[Reward(value, Status.SCORED)]], expected_total=1)
    assert result.status == Status.INFRA_ERROR
    assert result.reward == 0


@pytest.mark.parametrize("shape", [[], [[]], [[1, 1]], [[1], [1]]])
def test_missing_or_extra_assignment_components_score_zero(shape):
    result = aggregate_first_fit([[scored(value) for value in row] for row in shape], expected_total=1)
    assert result.status == Status.SCORED
    assert result.reward == 0


def test_ragged_assignment_is_an_infrastructure_error():
    result = aggregate_first_fit([[scored(1)], [scored(1), scored(0)]], expected_total=2)
    assert result.status == Status.INFRA_ERROR
    assert result.reward == 0


def test_empty_trusted_assignment_is_invalid_before_candidate_handling():
    result = aggregate_first_fit([], expected_total=0)
    assert result.status == Status.INVALID_TASK
    assert result.reward == 0
