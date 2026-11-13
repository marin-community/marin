# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from skyrl_gym.envs.aime.utils import normalize_final_answer
from skyrl_gym.envs.aime.verifier import AIMEVerifier
from skyrl_gym.verification import RolloutEvidence


@pytest.mark.parametrize(
    "expected,correct,wrong",
    [
        (r"\left(3,\frac{\pi}{2}\right)", r"(3,\frac{\pi}{2})", r"(3,\pi)"),
        (
            r"\begin{pmatrix} -1/3 \\ 2/3 \\ 5/3 \end{pmatrix}",
            r"\begin{pmatrix}-1/3\\2/3\\5/3\end{pmatrix}",
            r"\begin{pmatrix}-1/3\\2/3\\4/3\end{pmatrix}",
        ),
        (r"\frac{270}7\text{ degrees}", r"\frac{270}{7}", r"\frac{271}{7}"),
        (r"\frac9{19}", r"\frac{9}{19}", r"\frac{9}{20}"),
        (r"\frac{3}{\pi}", r"\dfrac{3}{\pi}", r"\dfrac{4}{\pi}"),
        (r"-\frac{1}{4}", "-1/4", "1/4"),
        ("42", "42", "42.00001"),
    ],
)
def test_math500_answers_accept_equivalents_and_reject_wrong_answers(
    expected, correct, wrong
):
    judge = AIMEVerifier(normalize_final_answer(expected))
    for answer, passed in [(correct, True), (wrong, False), ("", False)]:
        result = judge.verify(RolloutEvidence(response=f"Answer: \\boxed{{{answer}}}"))
        assert result.passed is passed
