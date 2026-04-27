# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parse the `ground_truth` field of `allenai/IF_multi_constraints_upto5`.

The training-set `ground_truth` column is a Python repr-style string (single
quotes, `None` literals), not JSON. So we use `ast.literal_eval`, which
handles both Python and JSON-style quoting safely without executing arbitrary
code.

Example value:

    [{'instruction_id': ['detectable_format:sentence_hyphens',
                         'last_word:last_word_answer'],
      'kwargs':         [None, {'last_word': 'brief'}]}]

The IFBench_test schema uses `instruction_id_list` (plural) at the top level;
the training schema wraps the same payload in a one-element list with a
singular `instruction_id` key. We normalise to the IFBench_test shape so the
downstream scorer doesn't need to know which source it came from.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedConstraints:
    instruction_id_list: list[str]
    kwargs: list[dict[str, Any] | None]


def parse_ground_truth(raw: str) -> ParsedConstraints:
    """Parse the `ground_truth` string into the (id_list, kwargs) tuple.

    Raises ValueError on malformed input — never silently skip.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"ground_truth must be a non-empty string; got {type(raw).__name__}")

    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"could not parse ground_truth as Python literal: {e!r}") from e

    if not isinstance(parsed, list) or len(parsed) != 1 or not isinstance(parsed[0], dict):
        raise ValueError(
            f"ground_truth must be a 1-element list of dicts; got {type(parsed).__name__} "
            f"len={len(parsed) if isinstance(parsed, list) else 'n/a'}"
        )

    body = parsed[0]
    if "instruction_id" not in body or "kwargs" not in body:
        raise ValueError(f"ground_truth dict missing keys; got {sorted(body.keys())}")

    id_list = body["instruction_id"]
    kwargs = body["kwargs"]
    if not isinstance(id_list, list) or not isinstance(kwargs, list):
        raise ValueError("instruction_id and kwargs must both be lists")
    if len(id_list) != len(kwargs):
        raise ValueError(f"instruction_id and kwargs length mismatch: {len(id_list)} vs {len(kwargs)}")

    return ParsedConstraints(instruction_id_list=list(id_list), kwargs=list(kwargs))
