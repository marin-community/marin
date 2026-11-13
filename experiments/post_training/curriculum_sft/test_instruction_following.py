# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import random

import pytest

from experiments.post_training.curriculum_sft.finance import WORKING_CAPITAL, finance_problem
from experiments.post_training.curriculum_sft.instruction_following import (
    CONFLICTS,
    ConstraintKind,
    follows_constraints,
    instruction_following_problem,
    sample_constraints,
)


@pytest.mark.parametrize(
    ("kind", "kwargs", "compliant", "violating"),
    [
        (ConstraintKind.BULLET_COUNT, {"count": 2}, "Steps:\n* one\n- two\n**Total** \\boxed{3}", "* one\n* two\n* 3"),
        (
            ConstraintKind.END_PHRASE,
            {"phrase": "That concludes the analysis."},
            "\\boxed{3}\nThat concludes the analysis.  ",
            "That concludes the analysis. \\boxed{3}",
        ),
        (ConstraintKind.NO_COMMAS, {}, "Revenue was 1250 so \\boxed{1250}", "Revenue was 1,250 so \\boxed{1250}"),
        (ConstraintKind.LOWERCASE, {}, "the ratio is \\boxed{A}", "The ratio is \\boxed{2}"),
        (ConstraintKind.QUOTATION, {}, ' "the answer is \\boxed{2}"\n', '"the answer is" \\boxed{2}'),
        (
            ConstraintKind.KEYWORD_FREQUENCY,
            {"keyword": "ratio", "count": 2},
            "The Ratio of A to B; the ratio is \\boxed{2}",
            "The ratios of A to B; the ratio is \\boxed{2}",
        ),
        (ConstraintKind.WORD_LIMIT, {"limit": 5}, "It is \\boxed{2} exactly.", "It is \\boxed{2} exactly, I think."),
        (ConstraintKind.TITLE, {}, "<<current ratio>>\nIt is \\boxed{2}", "<<>>\nIt is \\boxed{2}"),
        (
            ConstraintKind.JSON_ANSWER,
            {},
            'It is \\boxed{2.5}\n```json\n{"answer": 2.5}\n```\n',
            'It is \\boxed{2.5}\n```json\n{"answer": "2.5"}\n```',
        ),
        (
            ConstraintKind.PARAGRAPH_COUNT,
            {"count": 2},
            "First we divide.\n***\nSo \\boxed{2}",
            "First we divide.\n***\n***\nSo \\boxed{2}",
        ),
    ],
)
def test_constraint_accepts_compliant_and_rejects_violating_response(kind, kwargs, compliant, violating):
    constraints = [{"kind": str(kind), "kwargs": kwargs}]

    assert follows_constraints(constraints, compliant)
    assert not follows_constraints(constraints, violating)


def test_overlay_is_deterministic_and_never_pairs_conflicting_constraints():
    draws = [sample_constraints(random.Random(seed)) for seed in range(300)]

    assert draws == [sample_constraints(random.Random(seed)) for seed in range(300)]
    assert {len(draw) for draw in draws} == {1, 2}
    for draw in draws:
        kinds = [ConstraintKind(constraint["kind"]) for constraint in draw]
        assert len(set(kinds)) == len(kinds)
        assert frozenset(kinds) not in CONFLICTS
    assert {constraint["kind"] for draw in draws for constraint in draw} == set(ConstraintKind)


def test_constrained_problem_keeps_the_finance_question_and_answer():
    base = finance_problem(WORKING_CAPITAL, 4, seed=17)
    row = instruction_following_problem(WORKING_CAPITAL, 4, seed=17)

    assert row == instruction_following_problem(WORKING_CAPITAL, 4, seed=17)
    assert row["answer"] == base["answer"]
    assert row["problem"].startswith(base["problem"])
    assert row["problem"] != base["problem"]
    assert json.loads(row["constraints"])
