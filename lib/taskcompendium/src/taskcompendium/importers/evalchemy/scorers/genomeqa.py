# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GenomeQA final-answer parsing and equal-cell scoring."""

from verifyit.grade import Reward, scored
from verifyit.modes.grade_mcq import answer_letters

MACRO_METRIC = "genomeqa_macro"


def grade_genomeqa(expected: str, options: str, cell_weight: float, response: str) -> Reward:
    # The upstream prompt puts a choice after the last heading, sometimes as a bare letter.
    selected = response.rsplit("### Answer", 1)[-1].strip()
    declarations = answer_letters(selected)
    candidate = (declarations[-1] if declarations else selected).upper()
    answer = candidate if candidate in tuple(options) else ""
    correct = float(answer == expected)
    return scored(
        correct,
        metrics={
            "accuracy": correct,
            MACRO_METRIC: correct,
            "unanswered": float(not answer),
        },
        metric_weights={MACRO_METRIC: cell_weight},
        genomeqa_parsing={"answer": answer},
    )
