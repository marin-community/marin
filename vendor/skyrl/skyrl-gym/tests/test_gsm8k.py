import pytest

from skyrl_gym.answer_tasks import grade_gsm8k
from skyrl_gym.envs.gsm8k.utils import compute_score


@pytest.mark.parametrize(
    "output, ground_truth, expected",
    [
        ("The answer is #### 42", "42", 1.0),
        ("The answer is #### 42", "43", 0.0),
        ("The answer is #### $10", "10", 1.0),
        ("The answer is #### $5", "5", 1.0),
        ("The answer is #### $10", "11", 0.0),
        ("The answer is #### $1,234", "1234", 1.0),
        # answer is not in the expected format
        ("The answer is 42", "42", 0.0),
    ],
)
def test_task_grade_preserves_numeric_answer_formats(model_turn, output, ground_truth, expected):
    result = grade_gsm8k(
        model_turn(output), {"reward_method": "strict"}, {"reward_spec": {"ground_truth": ground_truth}}
    )
    assert result.reward == expected
    assert result.grade.reward == expected


@pytest.mark.parametrize(
    "output, ground_truth, stop_reason, expected",
    [
        ("Work.\n#### 42", "42", "stop", 1.0),
        ("Work.\n#### 42", "42", "complete", 1.0),
        ("Work.\n#### 42.0\n", "42", "eos", 1.0),
        ("Work.\n#### 1,234", "1234", "end_turn", 1.0),
        ("#### 42\nCorrection.\n#### 43", "42", "stop", 0.0),
        ("#### 43\nCorrection.\n#### 42", "42", "stop", 1.0),
        ("Work.\n#### 42\nMore text.", "42", "stop", 0.0),
        ("Work.\n#### 42", "42", "length", 0.0),
        ("Work.\n#### 42", "42", "error", 0.0),
        ("Work.\n#### 1,,234", "1234", "stop", 0.0),
        ("The answer is 42.", "42", "stop", 0.0),
        ("The answer is #### 42", "42", "stop", 0.0),
        ("#### $10", "10", "stop", 0.0),
    ],
)
def test_default_reward_grades_the_completed_final_line(model_turn, output, ground_truth, stop_reason, expected):
    result = grade_gsm8k(
        model_turn(output, stop_reason=stop_reason),
        {},
        {"reward_spec": {"ground_truth": ground_truth}},
    )
    assert result.reward == expected


def test_prepared_reward_model_extras_reach_the_verifier(model_turn):
    result = grade_gsm8k(
        model_turn("The answer is #### 42"), {"reward_method": "strict"}, {"reward_model": {"ground_truth": "42"}}
    )
    assert result.reward == 1.0


@pytest.mark.parametrize("method", ["strict", "flexible", "final_line"])
def test_invalid_reference_cannot_receive_format_credit(method):
    assert compute_score("#### 10", None, method=method, format_score=0.5) == 0
