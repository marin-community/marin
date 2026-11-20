# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ARC-AGI-2 grid extraction and exact-match grading.

Extraction follows ``parse_and_validate_json`` in ARC Prize's ``utils/parsing.py``
(arcprize/arc-agi-benchmarking, MIT, at 78f2d3eacb86587d1438ab14f08fb56811eb1f1c): the first
``\\boxed{...}`` holding a JSON list of lists, else the last bracketed JSON value in the response if it is
a non-empty list of lists. The harness's fallback LLM extraction call is omitted. The extracted grid must
equal the reference exactly.
"""

import json
import re

from verifyit.adapters.harbor_arc_agi_2 import grade_grids
from verifyit.grade import Reward, scored

BOXED = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)
CLOSING_TO_OPENING = {"]": "[", "}": "{"}


def _list_of_lists(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(row, list) for row in value)


def _boxed_grid(response: str) -> list | None:
    match = BOXED.search(response)
    if match is None:
        return None
    try:
        value = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None
    return value if _list_of_lists(value) else None


def _backscan_grid(response: str) -> list | None:
    end = max(response.rfind("]"), response.rfind("}"))
    if end == -1:
        return None
    closing = response[end]
    opening = CLOSING_TO_OPENING[closing]
    depth = 1
    for start in range(end - 1, -1, -1):
        if response[start] == closing:
            depth += 1
        elif response[start] == opening:
            depth -= 1
            if depth == 0:
                break
    else:
        return None
    try:
        value = json.loads(response[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if value and _list_of_lists(value) else None


def extract_grid(response: str) -> list | None:
    """The reference harness's grid for ``response``, or None when neither parser finds one."""
    boxed = _boxed_grid(response)
    return boxed if boxed is not None else _backscan_grid(response)


def grade_arc_agi(expected: list[list[int]], weight: float, response: str) -> Reward:
    candidate = extract_grid(response)
    if candidate is None:
        return scored(0.0, reason="no_grid", metric_weights={"accuracy": weight})
    grade = grade_grids(expected, candidate)
    return scored(grade.reward, **grade.detail, extracted=candidate, metric_weights={"accuracy": weight})
