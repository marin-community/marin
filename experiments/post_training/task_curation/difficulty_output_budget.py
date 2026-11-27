# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The authorized Large v2 output policy, separate from frozen fixed-budget runs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicOutputBudget:
    protocol_id: str = "large-context-minus-prompt-20261127-v2"
    context_tokens: int = 65536
    safety_margin_tokens: int = 64
    minimum_output_tokens: int = 16384

    def output_capacity(self, prompt_tokens: int) -> int:
        """Reject an unserved prompt when it leaves less than the admission floor."""
        capacity = self.context_tokens - prompt_tokens - self.safety_margin_tokens
        if prompt_tokens <= 0 or capacity < self.minimum_output_tokens:
            raise ValueError("Exact prompt leaves less than the Large v2 minimum output allowance")
        return capacity
