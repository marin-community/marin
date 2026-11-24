# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Hinted Scaling Laws GSM8K rule: the number after the first ``####``, compared as a number.

Ported from tatsu-lab/hinted-scaling-laws ``src/hinted/benchmarks/gsm8k_delphi.py``
(commit c0490e53ba8eae58bf5375f6b56956f0d57dad30). Unlike lm-eval's strict match, any whitespace
may follow ``####`` and the number may carry a dollar sign.
"""

import re

from verifyit.grade import Reward, scored

_FIRST_HASH_NUMBER = re.compile(r"####\s*(-?[$0-9.,]+)")


def first_hash_answer(completion: str) -> str | None:
    """The normalized number after the first ``####``, or None without one."""
    match = _FIRST_HASH_NUMBER.search(completion)
    if match is None:
        return None
    return match.group(1).replace("$", "").replace(",", "").strip().rstrip(".")


def grade_gsm8k_first_hash(expected: str, response: str) -> Reward:
    extracted = first_hash_answer(response)
    if extracted is None:
        return scored(0, extracted=None, expected=expected)
    gold = expected.replace(",", "").replace("$", "").strip()
    try:
        correct = float(extracted) == float(gold)
    except ValueError:
        correct = extracted == gold
    return scored(float(correct), extracted=extracted, expected=expected)
