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


@dataclass(frozen=True)
class SyntheticFinanceTask:
    """A fictional disclosure and its exact expected calculations."""

    task_id: str
    issuer: str
    revenue: int
    operating_cost: int

    @property
    def gross_profit(self) -> int:
        return self.revenue - self.operating_cost

    @property
    def margin_bps(self) -> int:
        quotient, remainder = divmod(self.gross_profit * 10_000, self.revenue)
        if remainder:
            raise ValueError("synthetic task does not have an integral margin in basis points")
        return quotient

    @property
    def evidence_ids(self) -> tuple[str, str]:
        return EVIDENCE_IDS

    @property
    def question(self) -> str:
        templates = (
            "{issuer} reports revenue of {revenue} million and operating costs of {operating_cost} million. "
            "Compute gross profit and operating margin in basis points.",
            "Using {issuer}'s reported revenue of {revenue} million and operating cost of {operating_cost} "
            "million, calculate gross profit and the margin in basis points.",
            "A disclosure for {issuer} gives {revenue} million of revenue and {operating_cost} million of "
            "operating costs. What are gross profit and operating margin, in basis points?",
            "Find {issuer}'s gross profit and operating-margin basis points from revenue {revenue} million "
            "and operating cost {operating_cost} million.",
        )
        template_index = (self.revenue // 100 + self.operating_cost // 100) % len(templates)
        return (
            templates[template_index].format(
                issuer=self.issuer,
                revenue=self.revenue,
                operating_cost=self.operating_cost,
            )
            + " Use only those figures and cite disclosure.revenue and disclosure.operating_cost."
        )

    @property
    def expected_result(self) -> dict[str, int]:
        return {"gross_profit": self.gross_profit, "margin_bps": self.margin_bps}


def synthetic_task(index: int, *, seed: int = 17) -> SyntheticFinanceTask:
    """Return a deterministic fictional task whose arithmetic has an exact oracle."""

    if index < 0:
        raise ValueError("index must be non-negative")
    # Unique revenue values make fact tuples distinct across train/eval namespaces.
    # Multiples of 100 keep the exact basis-point oracle integral.
    revenue = 100_000 + 100 * (index + 1)
    margin_bps = 1_000 + 100 * ((seed * 31 + index * 17) % 70)
    gross_profit = revenue * margin_bps // 10_000
    return SyntheticFinanceTask(
        task_id=f"fictional-finance-{seed}-{index:04d}",
        issuer=f"Issuer {index + 1}",
        revenue=revenue,
        operating_cost=revenue - gross_profit,
    )


def held_out_task(index: int, *, seed: int = 17) -> SyntheticFinanceTask:
    """Return a deterministic evaluation task disjoint from training indices.

    The offset is part of the split contract: training uses indices starting at zero,
    while evaluation always uses a separate namespace even when both sets have the
    same seed and size.
    """

    if index < 0:
        raise ValueError("index must be non-negative")
    return synthetic_task(index + 10_000, seed=seed)


def task_payload(task: SyntheticFinanceTask) -> dict[str, Any]:
    """Serialize a task into the strict generator contract."""

    return {
        "task_id": task.task_id,
        "issuer": task.issuer,
        "facts": {
            "revenue": task.revenue,
            "operating_cost": task.operating_cost,
        },
        "question": task.question,
        "answer": task.expected_result,
        "evidence": list(task.evidence_ids),
    }


def task_from_payload(payload: dict[str, Any]) -> SyntheticFinanceTask:
    """Reconstruct a task after the caller validates its payload."""

    facts = payload["facts"]
    return SyntheticFinanceTask(
        task_id=payload["task_id"],
        issuer=payload["issuer"],
        revenue=facts["revenue"],
        operating_cost=facts["operating_cost"],
    )
