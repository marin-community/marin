# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn how often the student answered a question correctly into an answer, abstain or drop label.

A question the student usually answers correctly becomes an answer target; one it almost never answers
correctly becomes an abstain target. The uncertain middle is dropped, so neither target teaches a guess.
"""

import hashlib
from enum import StrEnum

SAMPLES = 8
ANSWER_MIN_CORRECT = 6
ABSTAIN_MAX_CORRECT = 1
HELD_OUT_FRACTION = 0.1


class Label(StrEnum):
    ANSWER = "answer"
    ABSTAIN = "abstain"
    DROP = "drop"


def knowledge_label(correct: int) -> Label:
    assert 0 <= correct <= SAMPLES, f"correct count {correct} outside 0..{SAMPLES}"
    if correct >= ANSWER_MIN_CORRECT:
        return Label.ANSWER
    if correct <= ABSTAIN_MAX_CORRECT:
        return Label.ABSTAIN
    return Label.DROP


def held_out(request_id: str) -> bool:
    """Hold out a stable tenth of questions by hashing the source id, independent of row order."""
    digest = hashlib.sha256(request_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < HELD_OUT_FRACTION
