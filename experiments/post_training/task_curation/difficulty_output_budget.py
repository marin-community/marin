# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The authorized Large v2 output policy, separate from frozen fixed-budget runs."""

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


def require_solver_protocol(solver: Solver, protocol_id: str) -> None:
    """Accept the frozen protocol or its authorized Large-only replacement."""
    if protocol_id != PROTOCOL_ID and (solver != Solver.LARGE or protocol_id != LARGE_DYNAMIC_OUTPUT_PROTOCOL_ID):
        raise ValueError(f"Unknown solver protocol: {solver}/{protocol_id}")
