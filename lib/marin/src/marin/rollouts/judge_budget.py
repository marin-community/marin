# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reserve a trusted grader's maximum charge before starting the paid grader."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from decimal import ROUND_CEILING, Decimal
from typing import TypedDict

from evalrecords.records import JudgeCallBudget
from finestore.store import DataStore, Transaction
from taskcompendium.grading_result import GradeResult, Outcome

BUDGET_OBJECT = "evaluation/judge-budget.json"


class JudgeReservation(TypedDict):
    calls: int
    microdollars: int
    settled: bool


class JudgeBudget:
    """Per-archive ledger for the evaluation worker, including interrupted attempts."""

    def __init__(
        self,
        store: DataStore,
        limits: JudgeCallBudget,
        *,
        concurrency: int | None = None,
        commit: Callable[[Callable[[Transaction], object]], Awaitable[None]] | None = None,
    ):
        self.commit = commit
        self.concurrency = concurrency
        self.store = store
        self.limits = limits
        saved = store.read_view().read_blob(BUDGET_OBJECT)
        self.entries: dict[str, JudgeReservation] = json.loads(saved) if saved else {}
        self.active: set[str] = set()
        self.changed = asyncio.Condition()

    async def save(self) -> None:
        data = json.dumps(self.entries, allow_nan=False).encode()
        if self.commit is not None:
            await self.commit(lambda transaction: transaction.write_object(BUDGET_OBJECT, data))
        else:
            with self.store.transaction() as transaction:
                transaction.write_object(BUDGET_OBJECT, data)

    async def reserve(self, key: str, calls: int) -> None:
        cost = calls * self.limits.microdollars_per_call
        async with self.changed:
            if key in self.entries or calls <= 0:
                raise ValueError("Invalid or duplicate judge reservation")
            while (
                (self.concurrency is not None and len(self.active) >= self.concurrency)
                or calls + sum(entry["calls"] for entry in self.entries.values()) > self.limits.max_calls
                or cost + sum(entry["microdollars"] for entry in self.entries.values()) > self.limits.max_microdollars
            ):
                if not self.active:
                    raise RuntimeError("Judge budget exhausted, including outstanding reservations")
                await self.changed.wait()
            if key in self.entries:
                raise ValueError("Duplicate judge reservation")
            self.entries[key] = {"calls": calls, "microdollars": cost, "settled": False}
            await self.save()
            self.active.add(key)

    async def finish(self, key: str, grade: GradeResult | None) -> None:
        async with self.changed:
            if key not in self.active:
                if grade and (grade.detail or {}).get("judge_receipts"):
                    raise ValueError("Judge returned receipts before its reservation was committed")
                return
            self.active.remove(key)
            self.changed.notify_all()
            if grade is None:
                return
            if grade.diagnostics.get("grader_started") is False:
                receipts = []
            elif grade.status == Outcome.GRADED:
                receipts = (grade.detail or {}).get("judge_receipts")
            else:
                return
            if not isinstance(receipts, list):
                return
            entry = self.entries[key]
            if len(receipts) > entry["calls"] or len({receipt["id"] for receipt in receipts}) != len(receipts):
                raise ValueError("Unexpected judge receipt count or duplicate receipt")
            cost = Decimal(0)
            for receipt in receipts:
                amount = Decimal(str(receipt["usage"]["cost"])) * 1_000_000
                if not amount.is_finite() or not 0 <= amount <= self.limits.microdollars_per_call:
                    raise ValueError("Judge receipt exceeds its charge reservation")
                cost += amount
            self.entries[key] = {
                "calls": len(receipts),
                "microdollars": int(cost.to_integral_value(rounding=ROUND_CEILING)),
                "settled": True,
            }
            await self.save()
