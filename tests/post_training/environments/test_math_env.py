from marin.post_training.environments.math_env import MathEnv
from marin.post_training.environments.math_utils import grade_answer
from marin.post_training.environments.olym_math_env import OlymMathEnv


def test_math_env_loaded():
    """Test whether MathEnv examples are loaded correctly."""
    math_env = MathEnv(tokenizer=None)
    assert len(math_env.train_examples) == 7500, "MathEnv train examples should not be empty"
    assert len(math_env.eval_examples) == 5000, "MathEnv eval examples should not be empty"


def test_olym_math_env_loaded():
    """Test whether OlymMathEnv examples are loaded correctly."""
    olymp_math_env = OlymMathEnv(tokenizer=None, difficulty="easy", language="en")
    assert len(olymp_math_env.train_examples) == 80
    assert len(olymp_math_env.eval_examples) == 20

    # Ensure we get the same examples every time we load the environment
    assert olymp_math_env.train_examples[32]["prompt"].startswith(
        "Suppose 40 people vote anonymously, each with one ballot. "
        "Each person can vote for one or two candidates among three candidates. There are no invalid ballots"
    )
    assert olymp_math_env.eval_examples[16]["prompt"].startswith(
        "A frisbee toy is a circular disc divided into 20 sectors by 20 rays emanating from the center, "
        "with each sector colored either red or blue (only the front side is colored), and any two opposite "
        "sectors are colored differently. If frisbee toys that are the same after rotation are considered "
        "identical, how many different frisbee toys are there in total? (Answer with a specific number.)"
    )

    hard_olymp_math_env = OlymMathEnv(tokenizer=None, difficulty="hard", language="en")
    assert len(hard_olymp_math_env.train_examples) == 80
    assert len(hard_olymp_math_env.eval_examples) == 20
    assert hard_olymp_math_env.eval_examples[16]["prompt"].startswith(
        "If the inequality $2\\sin^2 C + \\sin A \\cdot \\sin B > k \\sin B \\cdot \\sin C$ holds for any "
        "triangle $\\triangle ABC$, find the maximum value of the real number $k$."
    )


def test_grade_answer_with_olym_math_env():
    """
    Test whether `grade_answer` works correctly with OlymMathEnv
    by ensuring a solution for one of the examples is verifiable.
    """
    hard_olymp_math_env = OlymMathEnv(tokenizer=None, difficulty="hard", language="en")

    example = hard_olymp_math_env.eval_examples[16]
    assert grade_answer(given_answer="2\\sqrt{2}-1", ground_truth=example["answer"]) is True
    assert grade_answer(given_answer=r"2\sqrt{2}-1", ground_truth=example["answer"]) is True
    assert grade_answer(given_answer=r"2*\sqrt{2} - 1", ground_truth=example["answer"]) is True
    assert grade_answer(given_answer=r"-1+2\sqrt{2}", ground_truth=example["answer"]) is True

    assert grade_answer(given_answer=r"2\sqrt{3}-1", ground_truth=example["answer"]) is False
    assert grade_answer(given_answer=r"2\sqrt{2} + 1", ground_truth=example["answer"]) is False
