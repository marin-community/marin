import copy
import json

import pytest

from capability_pipeline.synthesis import (
    SynthesisError,
    _controls_pass,
    _validate_controls,
)


def controls():
    cases = [
        {"id": "gold", "class": "positive", "category": "known_correct"},
        {"id": "wrong", "class": "negative", "category": "plausible_wrong"},
        {"id": "shortcut", "class": "negative", "category": "task_specific_shortcut"},
        {
            "id": "empty",
            "class": "malformed",
            "category": "empty_or_malformed",
            "expect": {"status": "extraction_error"},
        },
        {
            "id": "c7-mutant",
            "class": "partial",
            "category": "criterion_mutation",
            "partial_credit_reason": "Admitted rubric awards six independent numeric criteria while C7 fails.",
            "expect": {
                "status": "graded",
                "reward_min": 6 / 7 - 1e-9,
                "reward_max": 6 / 7 + 1e-9,
                "assertions": [
                    {
                        "path": ["detail", "stdout", "criteria", "C7", "passed"],
                        "equals": False,
                    }
                ],
            },
        },
    ]
    for case in cases:
        case.update(source_author="fixture-author", response="fixture")
    return {"schema_version": "1", "cases": cases}


def evidence(document):
    cases = []
    for c in document["cases"]:
        result = {"status": "graded", "reward": 0.0, "detail": {}}
        if c["class"] == "positive":
            result["reward"] = 1.0
        if c["class"] == "malformed":
            result.update(status="extraction_error", reward=None)
        if c["class"] == "partial":
            result.update(
                reward=6 / 7,
                detail={"stdout": json.dumps({"criteria": {"C7": {"passed": False}}})},
            )
        cases.append(
            {
                **c,
                "result": result,
                "control_type": "independent_solver"
                if c["class"] == "positive"
                else "authored_adversarial_control",
            }
        )
    return {"cases": cases}


def test_partial_control_checks_failed_criterion_and_preserves_earned_credit():
    document = controls()
    _validate_controls(document, 1)
    outcomes = evidence(document)
    assert _controls_pass(document, outcomes, external=True) == (True, [])
    outcomes["cases"][-1]["result"]["detail"]["stdout"] = json.dumps(
        {"criteria": {"C7": {"passed": True}}}
    )
    assert any(
        "criterion assertion" in issue
        for issue in _controls_pass(document, outcomes)[1]
    )


def test_partial_needs_explicit_ranges_assertions_and_rationale():
    for field in ("reward_min", "reward_max", "assertions"):
        document = controls()
        document["cases"][-1]["expect"].pop(field)
        with pytest.raises(SynthesisError, match="partial control"):
            _validate_controls(document, 1)
    document = controls()
    document["cases"][-1]["expect"]["reward_max"] = 1
    with pytest.raises(SynthesisError, match="partial control"):
        _validate_controls(document, 1)


def test_partial_does_not_replace_whole_answer_negative_controls():
    document = controls()
    document["cases"] = [case for case in document["cases"] if case["id"] != "wrong"]
    with pytest.raises(SynthesisError, match="plausible"):
        _validate_controls(document, 1)
    document = controls()
    outcomes = evidence(document)
    outcomes["cases"][1]["result"]["reward"] = 6 / 7
    assert any(
        "negative control earned" in issue
        for issue in _controls_pass(document, outcomes)[1]
    )


def test_criterion_assertion_rejects_missing_values_and_boolean_number_confusion():
    document = controls()
    for detail in (
        {},
        {"stdout": "not JSON"},
        {"stdout": '{"criteria":{"C7":{"passed":0}}}'},
    ):
        outcomes = copy.deepcopy(evidence(document))
        outcomes["cases"][-1]["result"]["detail"] = detail
        assert any(
            "criterion assertion" in issue
            for issue in _controls_pass(document, outcomes)[1]
        )
