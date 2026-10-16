# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from experiments.post_training.structure_closure.eval_audit import UidSource, audit_evaluation
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
            "stop_reason": "stop",
            "env_class": "structure_closure",
            "data_source": "structure-closure",
            "verification": {"status": "verified", "score": score},
            "score": [0.0, score],
            "env_metrics": {
                "closure_region_v1/json_object/close_required/syntax_complete": score,
                "closure_region_v1/json_object/close_required/content_preserved": score,
                "closure_region_v1/json_object/close_required/success": score,
                "closure_region_v1/json_object/close_required/trailing_content": score,
                "closure_region_v1/json_object/close_required/whole_response_success": 0.0,
            },
        }
        for index, (response, score) in enumerate((("}\nextra output<|eot_id|>", 1.0), ("<|eot_id|>", 0.0)))
    ]


def test_audit_keeps_identical_prompts_distinct_and_counts_model_failure(cohort, rows):
    audit = audit_evaluation(cohort, reversed(rows), uid_source=UidSource.ROW_INDEX)
    assert audit.complete
    assert [(row.case_id, row.success) for row in audit.results] == [("second", False), ("first", True)]
    assert audit.summary()["groups"]["json_object"] == {
        "scored": 2,
        "syntax_complete": 1,
        "content_preserved": 1,
        "success": 1,
        "failure": 1,
        "trailing_content": 1,
        "whole_response_success": 0,
    }


def test_audit_reports_early_control_closure_without_changing_recorded_reward(cohort, rows):
    case = cohort[1]
    case.update(control="continue_required", prefix='{"task":"x', expected='{"task":"xyz"}')
    row = rows[1]
    row["output_response"] = '"}<|eot_id|>'
    row["env_metrics"] = {
        f"closure_region_v1/json_object/continue_required/{name}": value
        for name, value in {
            "syntax_complete": 1.0,
            "content_preserved": 0.0,
            "success": 0.0,
            "trailing_content": 0.0,
            "whole_response_success": 0.0,
        }.items()
    }
    audit = audit_evaluation(cohort, rows, uid_source=UidSource.ROW_INDEX)
    assert audit.complete
    assert not audit.results[1].success
    assert audit.summary()["control_outcomes"] == {"json_object/continue_required": {"reference_prefix_closed_early": 1}}


@pytest.mark.parametrize("marker", ["<|eot_id|>", "<|end_of_text|>"])
def test_audit_matches_training_reward_for_signed_zero_and_terminal_markers(cohort, rows, marker):
    case = cohort[0]
    case.update(prefix='{"a":', expected='{"a":0.0}')
    row = rows[0]
    row["output_response"] = "-0.0}" + marker
    row["env_metrics"]["closure_region_v1/json_object/close_required/trailing_content"] = 0.0
    row["env_metrics"]["closure_region_v1/json_object/close_required/whole_response_success"] = 1.0
    audit = audit_evaluation([case], [row], uid_source=UidSource.ROW_INDEX)
    assert audit.complete
    assert audit.results[0].success
    assert audit.results[0].whole_response_success


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "prefix", "unavailable", "metric", "reward", "infra", "legacy_scorer"]
)
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
        rows[1]["env_metrics"]["closure_region_v1/json_object/close_required/success"] = 1.0
    elif failure == "reward":
        rows[1]["score"] = [0.0, 1.0]
    elif failure == "legacy_scorer":
        rows[1]["env_metrics"] = {
            name.replace("closure_region_v1/", "closure/"): value for name, value in rows[1]["env_metrics"].items()
        }
    else:
        rows[1]["exception_type"] = "TimeoutError"
    audit = audit_evaluation(cohort, rows, uid_source=UidSource.ROW_INDEX)
    assert not audit.complete
    if failure == "missing":
        assert audit.missing_uids == ("1",)
    else:
        assert len(audit.failures) == 1
    assert audit.results[0].case_id == "first"


def test_audit_reports_every_bad_row_in_one_pass(cohort, rows):
    rows[0]["prompt_token_ids"] = []
    rows[1]["verification"] = None
    audit = audit_evaluation(cohort, rows, uid_source=UidSource.ROW_INDEX)
    assert [failure.uid for failure in audit.failures] == ["0", "1"]
    assert not audit.complete
    assert audit.results == ()


def test_audit_matches_stable_case_ids_when_output_order_changes(cohort, rows):
    for case, row in zip(cohort, rows, strict=True):
        row["uid"] = "structure-closure:" + case["case_id"]
    audit = audit_evaluation(cohort, reversed(rows), uid_source=UidSource.CASE_ID)
    assert audit.complete
    assert [(row.case_id, row.success) for row in audit.results] == [("second", False), ("first", True)]


def test_audit_separates_repeated_openers_from_valid_calls_with_changed_content(cohort, rows):
    for case, row, response, complete in zip(
        cohort,
        rows,
        ('{"name":"other","arguments":{}}</tool_call>', "<tool_call><tool_call>"),
        (True, False),
        strict=True,
    ):
        case.update(
            family="repeated_opener_retention",
            control="opener_retention",
            format="tool",
            prefix="<tool_call>",
            expected='<tool_call>{"name":"expected","arguments":{}}</tool_call>',
        )
        row.update(output_response=response, verification={"status": "verified", "score": 0.0}, score=[0.0])
        row["env_metrics"] = {
            f"closure_region_v1/repeated_opener_retention/opener_retention/{name}": value
            for name, value in {
                "syntax_complete": float(complete),
                "content_preserved": 0.0,
                "success": 0.0,
                "trailing_content": 0.0,
                "whole_response_success": 0.0,
            }.items()
        }
    audit = audit_evaluation(cohort, rows, uid_source=UidSource.ROW_INDEX)
    assert audit.complete
    assert [result.success for result in audit.results] == [False, False]
    assert audit.summary()["tool_protocol"] == {
        "repeated_opener_retention/opener_retention": {"valid": 1, "repeated_opener": 1}
    }
