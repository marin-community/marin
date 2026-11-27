# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The authorized Large v2 output policy, separate from frozen fixed-budget runs."""

from dataclasses import dataclass

from finestore.rl.difficulty import PROTOCOL_ID, Solver

LARGE_DYNAMIC_OUTPUT_PROTOCOL_ID = "large-context-minus-prompt-20261127-v2"


def require_solver_protocol(solver: Solver, protocol_id: str) -> None:
    """Accept the frozen protocol or its authorized Large-only replacement."""
    if protocol_id != PROTOCOL_ID and (solver != Solver.LARGE or protocol_id != LARGE_DYNAMIC_OUTPUT_PROTOCOL_ID):
        raise ValueError(f"Unknown solver protocol: {solver}/{protocol_id}")


@dataclass(frozen=True)
class DynamicOutputBudget:
    protocol_id: str = LARGE_DYNAMIC_OUTPUT_PROTOCOL_ID
    context_tokens: int = 65536
    safety_margin_tokens: int = 64
    minimum_output_tokens: int = 16384

    def output_capacity(self, prompt_tokens: int) -> int:
        """Reject an unserved prompt when it leaves less than the admission floor."""
        capacity = self.context_tokens - prompt_tokens - self.safety_margin_tokens
        if prompt_tokens <= 0 or capacity < self.minimum_output_tokens:
            raise ValueError("Exact prompt leaves less than the Large v2 minimum output allowance")
        return capacity
