import pytest
from rolloutengine.models import ModelTurn
from skyrl_gym.answer_tasks import grade_cat_count

from skyrl_gym.envs.cat_count.reward import cat_count_score

NS = [1, 2, 4, 7, 10, 20]


def reward(completion, n, *, stop_reason=None):
    return cat_count_score(completion, n, stop_reason=stop_reason).reward


def cats(k: int) -> str:
    return " ".join(["cat"] * k)


@pytest.mark.parametrize("n", NS)
def test_exact_scores_one_without_clipping(n):
    s = cat_count_score(cats(n), n)
    assert s.exact and s.reward == pytest.approx(1.0)
    assert reward(cats(n) + "<|im_end|>", n) == pytest.approx(1.0)
    assert reward(cats(n) + "<|im_end|> <|eot_id|>", n) == pytest.approx(1.0)


@pytest.mark.parametrize("n", [n for n in NS if n > 1])
def test_reward_rises_monotonically_toward_n(n):
    scores = [reward(cats(k), n) for k in range(0, n + 1)]
    assert all(a < b for a, b in zip(scores, scores[1:]))


@pytest.mark.parametrize("n", NS)
def test_more_junk_words_score_lower_until_the_cap(n):
    scores = [reward(" ".join(["junk"] * k + [cats(n)]), n) for k in range(0, 12)]
    assert all(a > b for a, b in zip(scores, scores[1:]))


@pytest.mark.parametrize("n", NS)
def test_overshoot_keeps_falling_across_the_whole_64_token_budget(n):
    scores = [reward(cats(k), n) for k in range(n, 64)]
    assert all(a > b for a, b in zip(scores, scores[1:]))


def test_n1_non_cat_replies_score_nonpositive():
    assert reward("dog", 1) <= 0.0
    assert reward("I can help with that", 1) < 0.0


def test_near_miss_formats_earn_shaping_but_not_exact():
    for text in ["Cat.", "cat!", "CAT"]:
        s = cat_count_score(text, 1)
        assert not s.exact and 0.0 < s.reward < 1.0
    s = cat_count_score("cat  cat\ncat", 3)
    assert not s.exact and s.reward == pytest.approx(0.7)


def test_character_pieces_earn_nothing():
    assert reward("ca at c a t", 4) < 0.0
    assert reward("exact concatenate", 4) < 0.0


def test_embedded_end_marker_cannot_verify_exact():
    score = cat_count_score("cat<|im_end|> cat", 2)
    assert not score.exact
    assert score.reward < 1.0


@pytest.mark.parametrize("n", NS)
def test_empty_reply_loses_to_any_reply_within_2n_cats(n):
    empty = reward("", n)
    assert empty == pytest.approx(-0.25)
    for k in range(1, 2 * n + 1):
        assert reward(cats(k), n) > empty


@pytest.mark.parametrize(
    "n,completion,stop_reason,expected_reward,exact,n_words,truncated",
    [
        (4, "cat cat cat", None, 0.4725, False, 3, False),
        (2, "cat cat", None, 1.0, True, 2, False),
        (2, "cat cat", "length", 0.9, True, 2, True),
        (2, "cat cat", "stop", 1.0, True, 2, False),
        (2, "cat cat", "unknown", 1.0, True, 2, False),
        (20, cats(25), None, 0.468347953216, False, 25, False),
        (20, cats(25), "length", 0.368347953216, False, 25, True),
    ],
)
def test_environment_scores_completion_and_reports_verification_and_metrics(
    n, completion, stop_reason, expected_reward, exact, n_words, truncated
):
    turn = ModelTurn({}, (), (), None, stop_reason or "unknown", text=completion)
    step = grade_cat_count(turn, {}, {"extra_info": {"n": n}})
    assert step.reward == pytest.approx(expected_reward)
    assert step.grade.reward == float(exact)
    assert step.grade.passed is exact
    metrics = step.metrics
    assert metrics[f"exact_n{n}"] == float(exact)
    assert metrics[f"n_words_n{n}"] == float(n_words)
    assert metrics["has_cat"] == 1.0
    assert metrics["truncated"] == float(truncated)


def test_environment_scores_the_row_word():
    extras = {"extra_info": {"n": 3, "word": "dog"}}
    correct = ModelTurn({}, (), (), None, "stop", text="dog dog dog")
    incorrect = ModelTurn({}, (), (), None, "stop", text="cat cat cat")
    assert grade_cat_count(correct, {}, extras).grade.passed is True
    assert grade_cat_count(incorrect, {}, extras).grade.passed is False
    assert cat_count_score("cat cat cat", 3, word="dog").reward < cat_count_score("dog dog dog", 3, word="dog").reward


def test_uppercase_target_keeps_exact_reward_and_case_insensitive_shaping():
    exact = cat_count_score("Dog Dog", 2, word="Dog")
    near_miss = cat_count_score("dog dog", 2, word="Dog")
    assert exact.exact and exact.reward == pytest.approx(1.0)
    assert exact.junk_words == 0
    assert not near_miss.exact and near_miss.reward == pytest.approx(0.7)
