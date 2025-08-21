import pytest

from marin.post_training.environments.math_utils import grade_answer


@pytest.mark.skip(reason="Need to fix environment import.")
def test_math_env_loaded():
    """Test whether MathEnv examples are loaded correctly."""
    from marin.post_training.environments.math_env import MathEnv

    math_env = MathEnv(tokenizer=None)
    assert len(math_env.train_examples) == 7500, "MathEnv train examples should not be empty"
    assert len(math_env.eval_examples) == 5000, "MathEnv eval examples should not be empty"


@pytest.mark.skip(reason="Need to fix environment import.")
def test_olym_math_env_loaded():
    """Test whether OlymMathEnv examples are loaded correctly."""
    from marin.post_training.environments.olym_math_env import OlymMathEnv

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


@pytest.mark.skip(reason="Need to fix environment import.")
def test_open_math_reasoning_env_loaded():
    from marin.post_training.environments.open_math_reasoning_env import OpenMathReasoningEnv

    env = OpenMathReasoningEnv(tokenizer=None)
    assert len(env.train_examples) == 234_572
    assert len(env.eval_examples) == 1000

    # Ensure we get the same examples every time we load the environment
    assert env.train_examples[0]["prompt"].startswith(
        "Solve for \\( x \\): \\( |ax - 2| \\geq bx \\) given \\( a > 0 \\) and \\( b > 0 \\)"
    )
    assert env.train_examples[0]["answer"] == (
        "\\( x \\geq \\frac{2}{a-b} \\) or \\( x \\leq \\frac{2}{a+b} \\) or \\( x \\leq 0 \\)"
    )
    assert env.eval_examples[16]["prompt"].startswith(
        "For an integer \\( a > 1 \\) that is not a prime number, find the maximum possible "
        "value of \\( \\frac{a}{p^2} \\) where \\( p \\) is the smallest prime divisor of \\( a \\)."
    )


@pytest.mark.skip(reason="Need to fix environment import.")
def test_numina_math_env_loaded():
    from marin.post_training.environments.numina_math_env import NuminaMathEnv

    env = NuminaMathEnv(tokenizer=None)

    assert len(env.train_examples) == 836_291
    assert len(env.eval_examples) == 98

    # Ensure we get the same examples every time we load the environment
    assert env.train_examples[0]["prompt"].startswith(
        "Consider the terms of an arithmetic sequence: $-\\frac{1}{3}, y+2, 4y, \\ldots$. Solve for $y$."
    )
    assert env.eval_examples[16]["prompt"].startswith(
        "Given that the sequence $\\{a_n\\}$ is an arithmetic sequence, if $a_3 + a_{11} = 24$ and "
        "$a_4 = 3$, then the common difference of the sequence $\\{a_n\\}$ is ______."
    )


@pytest.mark.skip(reason="Need to fix environment import.")
def test_aqua_rat_env_loaded():
    from marin.post_training.environments.aqua_rat_env import AquaRatEnv

    env = AquaRatEnv(tokenizer=None)
    assert len(env.train_examples) == 96_993
    assert len(env.eval_examples) == 252

    # Ensure we get the same examples every time we load the environment
    assert env.train_examples[0]["prompt"].startswith(
        "Two friends plan to walk along a 43-km trail, starting at opposite ends of the trail at the same time. "
        "If Friend P's rate is 15% faster than Friend Q's, how many kilometers will Friend P have walked when "
        "they pass each other?"
    )
    assert env.eval_examples[16]["prompt"].startswith(
        "Alex has enough money to buy 30 bricks. If the bricks each cost 20 cents less, "
        "Grace could buy 10 more bricks. How much money does Grace have to spend on bricks?"
    )


@pytest.mark.skip(reason="Need to fix environment import.")
def test_svamp_env_loaded():
    from marin.post_training.environments.svamp_env import SVAMPEnv

    env = SVAMPEnv(tokenizer=None)

    assert len(env.train_examples) == 19_690
    assert len(env.eval_examples) == 1000

    # Ensure we get the same examples every time we load the environment
    assert env.train_examples[0]["prompt"].startswith(
        "bobby ate some pieces of candy . then he ate 25 more . if he ate a total of 43 pieces of "
        "candy how many pieces of candy had he eaten at the start ?"
    )
    assert env.eval_examples[16]["prompt"].startswith(
        "every day ryan spends 3 hours on learning english and some more hours on learning chinese . "
        "if he spends a total of 4 hours on learning english and chinese everyday how many hours does he "
        "spend on learning chinese ?"
    )


@pytest.mark.skip(reason="Need to fix environment import.")
def test_olympiad_bench_env_math_loaded():
    from marin.post_training.environments.olympiad_bench_env import OlympiadBenchEnv

    env = OlympiadBenchEnv(tokenizer=None, subject="maths", language="en")
    assert len(env.train_examples) == 574
    assert len(env.eval_examples) == 100

    # Ensure we get the same examples every time we load the environment
    assert env.train_examples[0]["prompt"].startswith(
        "Find the smallest number $n$ such that there exist polynomials $f_{1}, f_{2}, \\ldots, f_{n}$ "
        "with rational coefficients satisfying\n\n$$\nx^{2}+7=f_{1}(x)^{2}+f_{2}(x)^{2}+\\cdots+f_{n}(x)^{2}"
    )
    assert env.eval_examples[16]["prompt"].startswith(
        "The Sieve of Sundaram uses the following infinite table of positive integers:"
        "\n\n| 4 | 7 | 10 | 13 | $\\cdots$ |"
        "\n| :---: | :---: | :---: | :---: | :---: |"
        "\n| 7 | 12 | 17 | 22 | $\\cdots$ |"
        "\n| 10 | 17 | 24 | 31 | $\\cdots$ |"
        "\n| 13 | 22 | 31 | 40 | $\\cdots$ |"
        "\n| $\\vdots$ | $\\vdots$ | $\\vdots$ | $\\vdots$ |  |"
        "\n\nThe numbers in each row in the table form an arithmetic sequence. "
        "The numbers in each column in the table form an arithmetic sequence. "
        "The first four entries in each of the first four rows and columns are shown."
        "\nDetermine the number in the 50th row and 40th column."
    )


@pytest.mark.skip(reason="Need to fix environment import.")
def test_olympiad_bench_env_physics_loaded():
    from marin.post_training.environments.olympiad_bench_env import OlympiadBenchEnv

    env = OlympiadBenchEnv(tokenizer=None, subject="physics", language="en")
    assert len(env.train_examples) == 136
    assert len(env.eval_examples) == 100

    # Ensure we get the same examples every time we load the environment
    assert env.train_examples[0]["prompt"].startswith(
        "3. The circular restricted three-body problem\n\nIn general, there is no exact solution of the "
        "three-body problem, in which three masses move under their mutual gravitational attraction. However, "
        "it is possible to make some progress by adding some constraints to the motion.\n\nTwo-body problem\n\n"
        "Let's start with the motion of two masses, $M_{1}$ and $M_{2}$. Assume both masses move in circular orbits "
        "about their center of mass."
    )
    assert env.eval_examples[16]["prompt"].startswith(
        "Problem T3. Protostar formation\n\nLet us model the formation of a star as follows. A spherical cloud "
        "of sparse interstellar gas, initially at rest, starts to collapse due to its own gravity. The initial "
        "radius of the ball is $r_{0}$ and the mass is $m$. The temperature of the surroundings (much sparser than "
        "the gas) and the initial temperature of the gas is uniformly $T_{0}$. The gas may be assumed to be ideal. "
        "The average molar mass of the gas is $\\mu$ and its adiabatic index is $\\gamma>\\frac{4}{3}$. Assume "
        "that $G \\frac{m \\mu}{r_{0}} \\gg R T_{0}$, where $R$ is the gas constant and $G$ is the gravitational "
        "constant.\ni. During much of the collapse, the gas is so transparent that any heat generated is "
        "immediately radiated away, i.e. the ball stays in thermodynamic equilibrium with its surroundings. "
        "What is the number of times, $n$, by which the pressure increases when the radius is halved to "
        "$r_{1}=0.5 r_{0}$ ?"
    )


@pytest.mark.skip(reason="Need to fix environment import.")
def test_orz_env_loaded():
    from marin.post_training.environments.orz_env import ORZEnv

    env = ORZEnv(tokenizer=None)
    assert len(env.train_examples) == 71444
    assert len(env.eval_examples) == ORZEnv.DEV_SET_SIZE

    # Ensure we get the same examples every time we load the environment
    assert env.train_examples[0]["prompt"].startswith(
        "10. A. Given positive real numbers $a$, $b$, $c$ satisfy $9a + 4b = abc$. Then the "
        "minimum value of $a + b + c$ is $\\qquad$"
    )
    assert env.eval_examples[16]["prompt"].startswith(
        "19. Suppose $x, y, z$ and $\\lambda$ are positive real numbers such that\n$$\n\\begin{aligned}\ny "
        "z & =6 \\lambda x \\\\\nx z & =6 \\lambda y \\\\\nx y & =6 \\lambda z "
        "\\\\\nx^{2}+y^{2}+z^{2} & =1\n\\end{aligned}\n$$\n\nFind the value of $(x y z \\lambda)^{-1}$."
    )


@pytest.mark.skip(reason="Need to fix environment import.")
def test_grade_answer_with_olym_math_env():
    """
    Test whether `grade_answer` works correctly with OlymMathEnv
    by ensuring a solution for one of the examples is verifiable.
    """
    from marin.post_training.environments.olym_math_env import OlymMathEnv

    hard_olymp_math_env = OlymMathEnv(tokenizer=None, difficulty="hard", language="en")

    example = hard_olymp_math_env.eval_examples[16]
    assert grade_answer(given_answer="2\\sqrt{2}-1", ground_truth=example["answer"]) is True
    assert grade_answer(given_answer=r"2\sqrt{2}-1", ground_truth=example["answer"]) is True
    assert grade_answer(given_answer=r"2*\sqrt{2} - 1", ground_truth=example["answer"]) is True
    assert grade_answer(given_answer=r"-1+2\sqrt{2}", ground_truth=example["answer"]) is True

    assert grade_answer(given_answer=r"2\sqrt{3}-1", ground_truth=example["answer"]) is False
    assert grade_answer(given_answer=r"2\sqrt{2} + 1", ground_truth=example["answer"]) is False


@pytest.mark.skip(reason="Need to fix environment import.")
def test_grade_answer_with_open_math_reasoning_env():
    """
    Test whether `grade_answer` works correctly with OpenMathReasoningEnv
    by ensuring a solution for one of the examples is verifiable.
    """
    from marin.post_training.environments.open_math_reasoning_env import OpenMathReasoningEnv

    env = OpenMathReasoningEnv(tokenizer=None)

    answer = env.train_examples[0]["answer"]
    assert (
        grade_answer(
            given_answer="\\( x \\geq \\frac{2}{a-b} \\) or \\( x \\leq \\frac{2}{a+b} \\) or \\( x \\leq 0 \\)",
            ground_truth=answer,
        )
        is True
    )
    assert (
        grade_answer(
            given_answer="\\( x \\geq \\frac{2}{a-b} \\), \\( x \\leq \\frac{2}{a+b} \\), \\( x \\leq 0 \\)",
            ground_truth=answer,
        )
        is True
    )
    assert (
        grade_answer(
            given_answer="\\( x \\geq \\frac{2}{a-b} \\) or \\( x \\leq \\frac{2}{a+b} \\) or \\( x \\geq 0 \\)",
            ground_truth=answer,
        )
        is False
    )

    answer = env.train_examples[1]["answer"]
    assert grade_answer(given_answer=" 20", ground_truth=answer) is True
    assert grade_answer(given_answer=" 19", ground_truth=answer) is False

    answer = env.train_examples[2]["answer"]
    assert grade_answer(given_answer="\\(-\\frac{2}{3}\\)", ground_truth=answer) is True
    assert grade_answer(given_answer="-\\frac{2}{3}", ground_truth=answer) is True
    assert grade_answer(given_answer="\\(-\\frac{4}{3}\\)", ground_truth=answer) is False
    assert grade_answer(given_answer="-\\frac{1}{3}", ground_truth=answer) is False


@pytest.mark.skip(reason="Need to fix environment import.")
def test_grade_answer_with_numina_math_env():
    from marin.post_training.environments.numina_math_env import NuminaMathEnv

    env = NuminaMathEnv(tokenizer=None)

    answer = env.train_examples[0]["answer"]
    assert grade_answer(given_answer="\\frac{13}{6}", ground_truth=answer) is True
    assert grade_answer(given_answer="+\\frac{13}{6}", ground_truth=answer) is True
    assert grade_answer(given_answer="-\\frac{13}{6}", ground_truth=answer) is False


@pytest.mark.skip(reason="Need to fix environment import.")
def test_grade_answer_with_aqua_rat_env():
    from marin.post_training.environments.aqua_rat_env import AquaRatEnv

    env = AquaRatEnv(tokenizer=None)

    answer = env.train_examples[0]["answer"]
    assert grade_answer(given_answer="23", ground_truth=answer) is True
    assert grade_answer(given_answer="23.0", ground_truth=answer) is True
    assert grade_answer(given_answer="23.01", ground_truth=answer) is False
    assert grade_answer(given_answer="24", ground_truth=answer) is False

    answer = env.train_examples[1]["answer"]
    assert grade_answer(given_answer="5 and 1", ground_truth=answer) is True
    assert grade_answer(given_answer="5 and   1", ground_truth=answer) is True
    assert grade_answer(given_answer="5 and   2", ground_truth=answer) is False

    answer = env.train_examples[2]["answer"]
    assert grade_answer(given_answer="I and II", ground_truth=answer) is True
    assert grade_answer(given_answer="I and III", ground_truth=answer) is False

    answer = env.train_examples[3]["answer"]
    assert grade_answer(given_answer="$1600", ground_truth=answer) is True
    assert grade_answer(given_answer="$1600.00", ground_truth=answer) is True
    assert grade_answer(given_answer="1600.00", ground_truth=answer) is True
    assert grade_answer(given_answer="1599.99", ground_truth=answer) is False
    assert grade_answer(given_answer="$1600.01", ground_truth=answer) is False


@pytest.mark.skip(reason="Need to fix environment import.")
def test_grade_answer_with_svamp_env():
    from marin.post_training.environments.svamp_env import SVAMPEnv

    env = SVAMPEnv(tokenizer=None)

    answer = env.train_examples[0]["answer"]
    assert grade_answer(given_answer="18.000", ground_truth=answer) is True
    assert grade_answer(given_answer=" 18", ground_truth=answer) is True
    assert grade_answer(given_answer="16", ground_truth=answer) is False

    answer = env.eval_examples[0]["answer"]
    assert grade_answer(given_answer="58.0", ground_truth=answer) is True
    assert grade_answer(given_answer=" 58", ground_truth=answer) is True
    assert grade_answer(given_answer="57.999", ground_truth=answer) is False


@pytest.mark.skip(reason="Need to fix environment import.")
def test_grade_answer_with_olympiad_bench_env_loaded():
    from marin.post_training.environments.olympiad_bench_env import OlympiadBenchEnv

    env = OlympiadBenchEnv(tokenizer=None, subject="maths", language="en")

    answer = env.train_examples[6]["answer"]
    assert grade_answer(given_answer="6m", ground_truth=answer) is True
    assert grade_answer(given_answer="$ 6m $", ground_truth=answer) is True
    assert grade_answer(given_answer="16m", ground_truth=answer) is False


@pytest.mark.skip(reason="Need to fix environment import.")
def test_grade_answer_with_orz_env_loaded():
    from marin.post_training.environments.orz_env import ORZEnv

    env = ORZEnv(tokenizer=None)

    answer = env.train_examples[300]["answer"]
    assert grade_answer(given_answer="8\\pi", ground_truth=answer) is True
    assert grade_answer(given_answer=r"8 \pi", ground_truth=answer) is True
    assert grade_answer(given_answer=r"8 * \pi", ground_truth=answer) is True
    assert grade_answer(given_answer=r"9 * \pi", ground_truth=answer) is False
    assert grade_answer(given_answer="9\\pi", ground_truth=answer) is False
