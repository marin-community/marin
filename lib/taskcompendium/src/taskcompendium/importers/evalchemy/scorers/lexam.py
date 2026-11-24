# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LEXam multiple-choice extraction from ``evaluation.py`` in LEXam-Benchmark/LEXam.

The last ``###X###`` marker in the final answer decides; a response without one is wrong.
"""

import re

from verifyit.adapters.completion import final_text
from verifyit.grade import Reward, scored

ANSWER_MARKER = re.compile(r"###([A-Z])###")


def grade_lexam_mcq(expected: str, response: str) -> Reward:
    matches = ANSWER_MARKER.findall(final_text(response))
    answer = matches[-1] if matches else None
    return scored(float(answer == expected), extracted=answer, expected=expected)
