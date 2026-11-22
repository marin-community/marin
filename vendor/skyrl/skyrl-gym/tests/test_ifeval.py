"""IFEval rollout, preflight, and bounded grading use the same checks."""

import pytest
from rolloutengine.models import ModelTurn
from taskcompendium.grading_result import Outcome
from skyrl_gym.answer_tasks import grade_ifeval
from skyrl_gym.envs.data_contracts import get_data_contract

CASES = [
    (
        {"func_name": "verify_paragraph_count", "N": 2},
        "First.\n***\nSecond.",
        "First.\n\nSecond.",
    ),
    (
        {"func_name": "verify_bullet_points", "N": 2},
        "**Items:**\n* One\n* Two\n* * *",
        "**Items:**\n* One\n* * *",
    ),
    (
        {"func_name": "validate_sections", "N": 2, "section_splitter": "SECTION"},
        "<<Guide to SECTION 1>>\nSECTION 1\nFirst body.\nSECTION 2\nSecond body.",
        "<<Guide to SECTION 1>>\nSECTION 1\nFirst body.",
    ),
    (
        {"func_name": "validate_highlighted_sections", "N": 2},
        "**First** and **second**.",
        "**Only one**.",
    ),
]


@pytest.mark.parametrize(("descriptor", "passing", "failing"), CASES)
@pytest.mark.parametrize("verifyit_enabled", [False, True])
def test_ifeval_preflight_and_rollout_agree_on_compliant_and_ignored_replies(
    descriptor, passing, failing, verifyit_enabled
):
    contract = get_data_contract("ifeval")
    ground_truth = contract.validate_example(descriptor, passing, failing)
    assert contract.is_correct is not None
    assert contract.is_correct(passing, ground_truth)
    assert not contract.is_correct(failing, ground_truth)
    extras = {"reward_model": {"ground_truth": ground_truth}}
    for reply, reward in ((passing, 1.0), (failing, 0.0)):
        turn = ModelTurn({}, (), (), None, "stop", text=reply)
        result = grade_ifeval(turn, {"verifyit_enabled": verifyit_enabled}, extras)
        assert result.reward == reward


def test_ifeval_rollout_reports_invalid_constraints_as_verifier_errors():
    turn = ModelTurn({}, (), (), None, "stop", text="Any response")
    extras = {"reward_model": {"ground_truth": {"func_name": "validate_choice", "options": []}}}
    result = grade_ifeval(turn, {}, extras)
    assert result.reward == 0.0
    assert result.grade is not None
    assert result.grade.status is Outcome.INFRA_ERROR
