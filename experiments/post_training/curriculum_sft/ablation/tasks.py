# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small fictional tasks with a machine-checkable finance oracle.

The task family is intentionally narrower than FinanceBench. It tests the
mechanics we need to separate: extracting supplied figures, applying a known
calculation, citing the relevant evidence, and following an output contract.
No benchmark questions or examples are imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

EVIDENCE_IDS = ("disclosure.revenue", "disclosure.operating_cost")
TASK_PAYLOAD_FIELDS = frozenset({"task_id", "issuer", "facts", "question", "answer", "evidence"})
FACT_FIELDS = frozenset({"revenue", "operating_cost"})
RESULT_FIELDS = frozenset({"gross_profit", "margin_bps"})
BASIS_POINTS_SCALE = 10_000


@dataclass(frozen=True)
class SyntheticFinanceTask:
    """A fictional disclosure and its exact expected calculations."""

    task_id: str
    revenue: int
    operating_cost: int

    @property
    def gross_profit(self) -> int:
        return self.revenue - self.operating_cost

    @property
    def margin_bps(self) -> int:
        quotient, remainder = divmod(self.gross_profit * BASIS_POINTS_SCALE, self.revenue)
        if remainder:
            raise ValueError("synthetic task does not have an integral margin in basis points")
        return quotient

    @property
    def evidence_ids(self) -> tuple[str, str]:
        return EVIDENCE_IDS

    @property
    def expected_result(self) -> dict[str, int]:
        return {"gross_profit": self.gross_profit, "margin_bps": self.margin_bps}


def task_from_payload(payload: dict[str, Any]) -> SyntheticFinanceTask:
    """Reconstruct a task after the caller validates its payload."""

    facts = payload["facts"]
    return SyntheticFinanceTask(
        task_id=payload["task_id"],
        revenue=facts["revenue"],
        operating_cost=facts["operating_cost"],
    )
