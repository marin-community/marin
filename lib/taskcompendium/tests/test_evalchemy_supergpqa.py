# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from taskcompendium.importers.evalchemy.supergpqa import supergpqa_prompt

# The authors' zero-shot template filled by infer/data_loader.py for the synthetic row below.
SUPERGPQA_RENDERED = (
    "Answer the following multiple choice question. There is only one correct answer. The last line of your "
    "response should be in the format 'Answer: $LETTER' (without quotes), where LETTER is one of A, B, C, D, E, "
    "F, G, H, I, or J.\n\nWhich number is prime?\nPick one.\nA) 4\nB) 6\nC) 7\nD) {x}\n"
)


def test_prompt_matches_reference_rendering():
    assert supergpqa_prompt("Which number is prime?\nPick one.", ["4", "6", "7", "{x}"]) == SUPERGPQA_RENDERED
