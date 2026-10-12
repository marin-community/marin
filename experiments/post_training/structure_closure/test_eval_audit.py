# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from experiments.post_training.structure_closure.eval_audit import audit_evaluation
from experiments.post_training.structure_closure.token_data import TokenizedCaseRecord


@pytest.fixture
def cohort():
    return [
        TokenizedCaseRecord(
            case_id=identity,
            task_id="task",
            split="validation",
            family="json_object",
            control="close_required",
            format="json",
            instruction="Close the supplied object.",
            prefix='{"task":"x"',
            expected='{"task":"x"}',
            prompt_token_ids=[17, 23],
            target_token_ids=[29],
            prompt_sha256="unused-in-row-audit",
            canonical_joint_encoding_matches=True,
        )
        for identity in ("first", "second")
    ]


@pytest.fixture
def rows():
    return [
        {
            "uid": str(index),
            "prompt_token_ids": [17, 23],
            "output_response": response,
            "env_class": "structure_closure",
            "data_source": "structure-closure",
            "verification": {"status": "verified", "score": score},
            "score": score,
            "env_metrics": {
                "closure/json_object/close_required/syntax_complete": score,
                "closure/json_object/close_required/content_preserved": score,
                "closure/json_object/close_required/success": score,
            },
        }
        for index, (response, score) in enumerate((("}", 1.0), ("", 0.0)))
    ]


def test_audit_keeps_identical_prompts_distinct_and_counts_model_failure(cohort, rows):
    audit = audit_evaluation(cohort, reversed(rows))
    assert audit.complete
    assert [(row.case_id, row.success) for row in audit.results] == [("second", False), ("first", True)]
    assert audit.summary()["groups"]["json_object"] == {
        "scored": 2,
        "syntax_complete": 1,
        "content_preserved": 1,
        "success": 1,
        "failure": 1,
    }


@pytest.mark.parametrize("failure", ["missing", "duplicate", "prefix", "unavailable", "metric", "reward", "infra"])
def test_audit_rejects_incomplete_or_inconsistent_evidence(cohort, rows, failure):
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif failure == "prefix":
        rows[1]["prompt_token_ids"] = [17]
    elif failure == "unavailable":
        rows[1]["verification"] = {"status": "unavailable", "score": None}
    elif failure == "metric":
        rows[1]["env_metrics"]["closure/json_object/close_required/success"] = 1.0
    elif failure == "reward":
        rows[1]["score"] = 1.0
    else:
        rows[1]["exception_type"] = "TimeoutError"
    audit = audit_evaluation(cohort, rows)
    assert not audit.complete
    if failure == "missing":
        assert audit.missing_uids == ("1",)
    else:
        assert len(audit.failures) == 1
    assert audit.results[0].case_id == "first"


def test_audit_reports_every_bad_row_in_one_pass(cohort, rows):
    rows[0]["prompt_token_ids"] = []
    rows[1]["verification"] = None
    audit = audit_evaluation(cohort, rows)
    assert [failure.uid for failure in audit.failures] == ["0", "1"]
    assert not audit.complete
    assert audit.results == ()
