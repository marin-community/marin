import pytest
from rolloutengine.models import ModelTurn
from skyrl_gym.answer_tasks import grade_sequence
from skyrl_gym.envs.sequence.reward import sequence_score

EVENS = [str(2 * i) for i in range(1, 11)]


@pytest.mark.parametrize(
    ("completion", "exact", "prefix", "reward"),
    [
        ("2 4 6 8 10 12 14 16 18 20", True, 10, 1.0),
        ("2 4 6 8 10 12 14 16 18 20<|im_end|>", True, 10, 1.0),
        ("2 4 6 8 10 12 14 16 18", False, 9, 0.7 * 0.9),
        ("2 4 6 8 10 12 14 16 18 20 22", False, 10, 0.7 - 0.25),
        ("8 10 12", False, 0, -0.25),
        ("", False, 0, -0.25),
    ],
)
def test_score_rewards_the_correct_prefix_and_exact_match(completion, exact, prefix, reward):
    score = sequence_score(completion, EVENS)
    assert score.exact is exact
    assert score.correct_prefix == prefix
    assert score.reward == pytest.approx(reward)


@pytest.mark.parametrize("completion", ["2\n4\n6", "2  4\t6", " 2 4 6 "])
def test_exact_requires_single_spaces_but_shaping_does_not(completion):
    score = sequence_score(completion, ["2", "4", "6"])
    assert score.correct_prefix == 3
    assert score.exact is (completion.strip() == "2 4 6")
    assert score.reward == pytest.approx(1.0 if score.exact else 0.7)


def test_truncation_costs_a_tenth():
    assert sequence_score("2 4 6 8 10 12 14 16 18 20", EVENS, stop_reason="length").reward == pytest.approx(0.9)


def test_task_reads_items_from_the_row():
    turn = ModelTurn({}, (), (), None, "stop", text="2 4 6")
    step = grade_sequence(turn, {}, {"extra_info": {"n": 3, "items": ["2", "4", "6"]}})
    assert step.grade.passed is True
    assert step.metrics["exact_n3"] == 1.0 and step.metrics["prefix_fraction"] == 1.0
    incorrect = ModelTurn({}, (), (), None, "stop", text="2 5")
    assert grade_sequence(incorrect, {}, {"extra_info": {"items": ["2", "4"]}}).reward == pytest.approx(0.35 - 0.25)
