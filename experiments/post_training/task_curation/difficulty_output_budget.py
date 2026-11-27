# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The authorized Large v2 output policy, separate from frozen fixed-budget runs."""

from typing import Any

from finestore.rl.difficulty import (
    LARGE_DYNAMIC_OUTPUT_PROTOCOL_ID as LARGE_DYNAMIC_OUTPUT_PROTOCOL_ID,
)
from finestore.rl.difficulty import (
    PROTOCOL_ID,
    Solver,
)
from finestore.rl.difficulty import (
    DynamicOutputBudget as DynamicOutputBudget,
)

LARGE_DYNAMIC_REQUEST_TIMEOUT = 7200


def require_solver_protocol(solver: Solver, protocol_id: str) -> None:
    """Accept the frozen protocol or its authorized Large-only replacement."""
    if protocol_id != PROTOCOL_ID and (solver != Solver.LARGE or protocol_id != LARGE_DYNAMIC_OUTPUT_PROTOCOL_ID):
        raise ValueError(f"Unknown solver protocol: {solver}/{protocol_id}")


def dynamic_usage_matches(payload: dict[str, Any], prompt_tokens: int, max_tokens: int) -> bool:
    """Require exact served input usage and the full allowance at a length stop."""
    usage = payload.get("usage")
    choices = payload.get("choices")
    return (
        isinstance(usage, dict)
        and type(usage.get("prompt_tokens")) is int
        and type(usage.get("completion_tokens")) is int
        and usage["prompt_tokens"] == prompt_tokens
        and 0 <= usage["completion_tokens"] <= max_tokens
        and isinstance(choices, list)
        and len(choices) == 1
        and isinstance(choices[0], dict)
        and (choices[0].get("finish_reason") != "length" or usage["completion_tokens"] == max_tokens)
    )
