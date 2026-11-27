# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from verifyit.grade import Status

from taskcompendium.importers.evalchemy.chembench import multiple_choice_prompt, numeric_prompt
from taskcompendium.importers.evalchemy.scorers.chembench import grade_chembench

MULTIPLE_CHOICE = {"kind": "multiple_choice", "letters": ["A", "B", "C", "D"], "correct": ["A", "C"]}

# ChemBench's INSTRUCT_TEMPLATE_1 for the synthetic row below, with markup wrappers removed.
MULTIPLE_CHOICE_RENDERED = (
    "The following is a multiple choice question about chemistry.\n"
    "Please answer by responding with the letter of the correct answer.\n\n"
    "Question: Which of H2O and CCO boils at 100 °C?\n\n"
    "Options:\nA. water\nB. ethanol {x}\n\n"
    "You MUST include the letter(s) of the correct answer (separated by comma if there are many) within the "
    "following tags: [ANSWER] and [/ANSWER].\n"
    "For example, '[ANSWER]<answer>[/ANSWER]', where <answer> is comma- or space-separated list of the correct "
    "letters. Always answer in exactly this format of comma-separated letters between the two tags, even if you "
    "are unsure. We require this because we use automatic parsing."
)
NUMERIC_RENDERED = (
    "The following is a question about chemistry.\n\n"
    "Question: What is the pH of 0.1 M HCl?\n\n"
    "You MUST include the final answer within the following tags: [ANSWER] and [/ANSWER].\n"
    "For example, '[ANSWER]<answer>[/ANSWER]', where <answer> is only one number. Always answer in exactly this "
    "format, with ONE NUMBER between the two tags, even if you are unsure. Use dots as decimal separator. We "
    "require this because we use automatic parsing."
)


def test_prompts_strip_markup_and_letter_options():
    question = "Which of \\ce{H2O} and [START_SMILES]CCO[END_SMILES] boils at $100$ \\pu{°C}?"
    assert multiple_choice_prompt(question, ["water", "ethanol {x}"]) == MULTIPLE_CHOICE_RENDERED
    assert numeric_prompt("What is the pH of 0.1 M \\ce{HCl}?") == NUMERIC_RENDERED


@pytest.mark.parametrize(
    "response,reward,extracted",
    [
        ("Reasoning.\n[ANSWER]A, C[/ANSWER]", 1, ["A", "C"]),
        ("[ANSWER]C,A[/ANSWER]", 1, ["A", "C"]),
        ("[ANSWER]A[/ANSWER]", 0, ["A"]),
        ("[ANSWER]A, B, C[/ANSWER]", 0, ["A", "B", "C"]),
        # Letters from every tag count, so a revised answer adds to the first.
        ("[ANSWER]A[/ANSWER] or rather [ANSWER]C[/ANSWER]", 1, ["A", "C"]),
        ("<ANS>A, C</ANS>", 1, ["A", "C"]),
        # Text after a period inside the tag is ignored.
        ("[ANSWER]A, C. water and more[/ANSWER]", 1, ["A", "C"]),
        # Letters outside the offered options are dropped.
        ("[ANSWER]A, C, Z[/ANSWER]", 1, ["A", "C"]),
        ("[ANSWER]a, c[/ANSWER]", 0, None),
        ("[ANSWER]A C[/ANSWER]", 1, ["A", "C"]),
        ("[ANSWER]A and C[/ANSWER]", 0, None),
        ("The answer is A and C.", 0, None),
        ("", 0, None),
    ],
)
def test_multiple_choice_requires_the_exact_letter_set(response, reward, extracted):
    result = grade_chembench(MULTIPLE_CHOICE, response)
    assert result.status == Status.SCORED
    assert result.reward == reward
    assert result.detail["extracted"] == extracted


@pytest.mark.parametrize(
    "target,response,reward,value",
    [
        (1500.0, "[ANSWER]1500[/ANSWER]", 1, 1500.0),
        (1500.0, "[ANSWER] 1.5 x 10^3 [/ANSWER]", 1, 1500.0),
        (1500.0, "[ANSWER]1.5 \\times 10^3[/ANSWER]", 1, 1500.0),
        (1500.0, "[ANSWER]1.5 \u00d7 10^3[/ANSWER]", 1, 1500.0),
        (1500.0, "[ANSWER]1.5e3[/ANSWER]", 1, 1500.0),
        (1500.0, "[ANSWER]\n1510\n[/ANSWER]", 1, 1510.0),
        (1500.0, "[ANSWER]1515[/ANSWER]", 0, 1515.0),
        # The first number inside the first tag is the answer.
        (1500.0, "[ANSWER]about 1500 kJ/mol[/ANSWER]", 1, 1500.0),
        (1500.0, "[ANSWER]1,500[/ANSWER]", 0, 1.0),
        (1500.0, "[ANSWER]1.5x10^{3}[/ANSWER]", 0, None),
        (0.001, "[ANSWER]10^-3[/ANSWER]", 1, 0.001),
        (-100.0, "[ANSWER]-10^2[/ANSWER]", 1, -100.0),
        (-100.0, "[ANSWER]-100.5[/ANSWER]", 1, -100.5),
        (-285.8, "[ANSWER]\u2212285.8[/ANSWER]", 1, -285.8),
        (285.8, "[ANSWER]\u2212285.8[/ANSWER]", 0, -285.8),
        (-285.8, "[ANSWER]\u2013285.8 kJ/mol[/ANSWER]", 1, -285.8),
        (0.0, "[ANSWER]0[/ANSWER]", 1, 0.0),
        (0.0, "[ANSWER]0.001[/ANSWER]", 0, 0.001),
        (1500.0, "[ANSWER]none[/ANSWER]", 0, None),
        (1500.0, "The answer is 1500.", 0, None),
    ],
)
def test_numeric_answers_are_within_one_percent(target, response, reward, value):
    result = grade_chembench({"kind": "numeric", "target": target}, response)
    assert result.status == Status.SCORED
    assert result.reward == reward
    assert result.detail["value"] == value
