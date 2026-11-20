# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hinted GSM8K and MATH-500 prompts match the Hinted Scaling Laws reference construction.

Expected hinted prompts were rendered by tatsu-lab/hinted-scaling-laws at
c0490e53ba8eae58bf5375f6b56956f0d57dad30 (GSM8KDelphiProblem.hinted_prompt_at_level with its
mask_word and truncate_word) on the synthetic rows below.
"""

import re

import pytest

from taskcompendium.importers.evalchemy.hinted import (
    HINT_LEVELS,
    HintMethod,
    gsm8k_block,
    gsm8k_example,
    gsm8k_rows,
    hint_block_prompt,
    mask_words,
    math_reference,
    minerva_exemplars,
    redact_answer,
    truncate_words,
)
from taskcompendium.importers.evalchemy.scorers.main import score
from taskcompendium.importers.evaluation_harness import HarnessTemplate, compile_generation_documents

EXEMPLARS = (
    {
        "question": "Ann has 3 apples and buys 4 more. How many apples does she have?",
        "answer": "Ann has 3 + 4 = <<3+4=7>>7 apples.\n#### 7",
    },
    {
        "question": "A box holds 2 rows of 6 eggs. How many eggs?",
        "answer": "There are 2 * 6 = <<2*6=12>>12 eggs.\n#### 12",
    },
)
TARGET = {
    "question": "Tom reads 9 pages a day for 2 days. How many pages does he read?",
    "answer": "Tom reads 9 * 2 = <<9*2=18>>18 pages in total, so 18 pages.\n#### 18",
}
EXEMPLAR_BLOCKS = (
    "Question: Ann has 3 apples and buys 4 more. How many apples does she have?\nHint: {first}\n"
    "Answer: Ann has 3 + 4 = <<3+4=7>>7 apples.\n#### 7\n\n"
    "Question: A box holds 2 rows of 6 eggs. How many eggs?\nHint: {second}\n"
    "Answer: There are 2 * 6 = <<2*6=12>>12 eggs.\n#### 12\n\n"
    "Question: Tom reads 9 pages a day for 2 days. How many pages does he read?\nHint: {target}\nAnswer:"
)


def gsm8k_prompt(method: HintMethod, level: float) -> str:
    exemplars = [gsm8k_example(document, f"train:{index}") for index, document in enumerate(EXEMPLARS)]
    prompt, _ = hint_block_prompt(gsm8k_example(TARGET, "0"), exemplars, method, level, render=gsm8k_block)
    return prompt


def test_truncation_shows_a_word_prefix_and_redacts_the_answer():
    assert gsm8k_prompt(HintMethod.TRUNCATION, 0.5) == EXEMPLAR_BLOCKS.format(
        first="Ann has 3 +", second="There are 2 *", target="Tom reads 9 * 2 = <<9*2=<mask>>><mask>"
    )


def test_masking_keeps_seeded_words_and_does_not_redact():
    assert gsm8k_prompt(HintMethod.MASKING, 0.5) == EXEMPLAR_BLOCKS.format(
        first="<mask> has 3 <mask> <mask> = <mask> apples.",
        second="<mask> are 2 * <mask> = <mask> <mask>",
        target="<mask> <mask> <mask> * 2 = <<9*2=18>>18 pages in total, <mask> <mask> <mask>",
    )


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [
        (0.5, "<mask> <mask>  gamma\ndelta epsilon <mask>"),
        (0.3, "<mask> <mask>  gamma\n<mask> <mask> zeta"),
    ],
)
def test_masking_matches_reference_draws_and_preserves_whitespace(fraction, expected):
    assert mask_words("alpha beta  gamma\ndelta epsilon zeta", fraction, seed="p|0") == expected


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [(0.1, "alpha"), (0.5, "alpha beta  gamma")],
)
def test_truncation_cuts_at_word_boundaries(fraction, expected):
    assert truncate_words("alpha beta  gamma\ndelta epsilon zeta", fraction) == expected


def test_redaction_spares_longer_numbers_and_matches_grouped_spellings():
    # The reference guard also spares a sentence-final "18.", as the paper's redaction does.
    assert redact_answer("18 of 180 and 1.18 then 18 or 18.", "18") == "<mask> of 180 and 1.18 then <mask> or 18."
    assert redact_answer("paid $2,125 or 2125", "2125") == "paid $<mask> or <mask>"
    assert redact_answer(r"so $x=\frac{2}{3}$ and \frac{2}{30}", r"\frac{2}{3}") == r"so $x=<mask>$ and \frac{2}{30}"


def test_hinted_gsm8k_rows_cover_each_level_with_the_plain_exemplars():
    pool = [{"question": f"Pool question {index}?", "answer": f"Work {index}.\n#### {index}"} for index in range(9)]
    [plain] = compile_generation_documents([TARGET], pool, HarnessTemplate(name="gsm8k", num_fewshot=5))
    rows = list(gsm8k_rows([TARGET], pool, HintMethod.TRUNCATION, source_row_prefix="gsm8k:test"))
    assert [row["hint_level"] for row in rows] == list(HINT_LEVELS)
    exemplar_questions = [pool[index]["question"] for index in plain["fewshot_indices"]]
    for row in rows:
        assert re.findall(r"Question: (.*)\n", row["prompt"]) == [*exemplar_questions, TARGET["question"]]
        assert row["reference_id"] == "gsm8k:test:0" and row["grading"] == {"expected": "18"}


SOLUTIONS = [
    ("So the answer is $\\boxed{5}$.", "", None),
    (
        "First, $x+1=6$. Therefore, the domain is $\\boxed{[2,5)}$.",
        "First, $x+1=6$. Therefore, the domain is",
        "[2,5)",
    ),
    (
        "We get\n\n$$-\\frac{3}{2}a=b$$Since $6y=b$, we have\n\n$$\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$",
        "We get\n\n$$-\\frac{3}{2}a=b$$Since $6y=b$, we have",
        "-\\frac{2}{3}",
    ),
    (
        "\\begin{align*}\n2x &= 4 \\\\\nx &= \\boxed{2}\n\\end{align*}",
        "\\begin{align*}\n2x &= 4 \\\\\nx &=",
        None,
    ),
    ("Hence $\\tan A = \\boxed{2}$.\n\nAlternatively, $\\tan A=\\boxed{2}$.", "Hence", "2"),
]


@pytest.mark.parametrize(("solution", "expected", "answer"), SOLUTIONS)
def test_math_reference_removes_the_boxed_answer(solution, expected, answer):
    reference = math_reference(solution)
    assert "boxed" not in reference
    if expected:
        assert reference == expected
    if answer is not None:
        assert answer not in reference


def test_minerva_exemplar_references_hold_no_final_answer():
    for exemplar in minerva_exemplars():
        assert "boxed" not in exemplar.reference and "Final Answer" not in exemplar.reference
        assert exemplar.answer not in exemplar.reference


# Cases from the paper's tests/benchmarks/test_gsm8k.py (strict_extract and verify).
@pytest.mark.parametrize(
    ("response", "expected", "reward", "extracted"),
    [
        ("so 18 eggs.\n#### 18\n", "18", 1.0, "18"),
        ("#### $460\n", "460", 1.0, "460"),
        ("####   26", "26", 1.0, "26"),
        ("#### 1,430.", "1430", 1.0, "1430"),
        ("#### 12\nQuestion: next one\n#### 99", "99", 0.0, "12"),
        ("the answer is 18", "18", 0.0, None),
        ("#### 12.00", "12", 1.0, "12.00"),
        ("#### 2,125", "2125", 1.0, "2125"),
        ("#### 13", "12", 0.0, "13"),
    ],
)
def test_first_hash_scorer_matches_the_paper_rule(response, expected, reward, extracted):
    result = score("gsm8k_first_hash", {"expected": expected}, response)
    assert (result.reward, result.detail["extracted"]) == (reward, extracted)
