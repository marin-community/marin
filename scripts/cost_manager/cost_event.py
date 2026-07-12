# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The cost-event schema written to finelog and the fetch window.

A :class:`CostEvent` is one provider cost line item for a single UTC usage
day. Every backend normalizes its provider's billing data into a list of
these, the runner stamps each with a shared ``collected_ts``, and the sink
appends them to the finelog ``cost.events`` namespace.

finelog is append-only: a daily job that re-fetches a trailing window writes a
fresh row for each ``(usage_date, provider, category, detail)`` on every run.
Readers take the latest snapshot by keeping the row with the greatest
``collected_ts`` (equivalently the greatest implicit ``seq``) per key. See the
README for the canonical "latest snapshot" SQL.
"""

import datetime as dt
import os
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import ClassVar


class CostFetchError(RuntimeError):
    """A backend could not fetch costs for an actionable, expected reason.

    Raised for missing credentials, provider auth/permission errors, and other
    misconfigurations the operator can fix. The runner reports these per
    provider and continues with the remaining backends rather than aborting.
    """


def require_env(var_name: str, *, provider: str, purpose: str) -> str:
    """Return the value of env var ``var_name`` or raise :class:`CostFetchError`."""
    value = os.environ.get(var_name)
    if not value:
        raise CostFetchError(f"{provider}: required env var {var_name!r} is unset ({purpose})")
    return value


# finelog namespace holding daily cost line items.
COST_EVENTS_NAMESPACE = "cost.events"

# Cost data is tiny (hundreds of rows/day) but valuable over a long horizon, so
# the namespace gets a generous byte cap to keep years of history rather than
# inheriting the server's high-volume eviction default.
COST_EVENTS_MAX_BYTES = 2 * 1024 * 1024 * 1024


class AmountKind(StrEnum):
    """Whether ``CostEvent.cost`` is an authoritative bill or an estimate.

    ``BILLED`` rows come from a provider cost/billing API and reconcile with an
    invoice. ``ESTIMATED`` rows are derived from usage metrics times a local
    rate card (CoreWeave has no dollar-denominated API) and will not match an
    invoice exactly.
    """

    BILLED = "billed"
    ESTIMATED = "estimated"


@dataclass
class CostEvent:
    """One provider cost line item for a single UTC usage day."""

    # Same-provider rows colocate for row-group pruning; the dominant query is
    # a per-provider time range.
    key_column: ClassVar[str] = "provider"

    # UTC midnight of the usage day; finelog's per-key ordering timestamp.
    ts: dt.datetime
    # The usage day as an ISO ``YYYY-MM-DD`` string for human-facing grouping.
    usage_date: str
    provider: str
    # Provider-natural coarse grouping: an OpenAI/Anthropic line item is "api";
    # a GCP row carries the service ("Compute Engine"); a CoreWeave row carries
    # the resource class ("compute"/"storage"/"network").
    category: str
    # Finer sub-detail within the category: model, SKU, region, or line item.
    detail: str
    cost: float
    currency: str
    # AmountKind value; str because finelog cannot serialize a StrEnum directly.
    amount_kind: str
    # When this row was produced. The runner stamps every row of a run with one
    # value so a single fetch is one atomic snapshot for latest-wins reads.
    collected_ts: dt.datetime | None = None


@dataclass(frozen=True)
class DateWindow:
    """An inclusive range of UTC usage days ``[start, end]`` to fetch."""

    start: dt.date
    end: dt.date

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"DateWindow end {self.end} precedes start {self.start}")

    @staticmethod
    def trailing(days: int, *, today: dt.date) -> "DateWindow":
        """The window of the last ``days`` UTC days ending at ``today``."""
        if days < 1:
            raise ValueError(f"lookback days must be >= 1, got {days}")
        return DateWindow(start=today - dt.timedelta(days=days - 1), end=today)

    @property
    def start_dt(self) -> dt.datetime:
        """UTC midnight at the start of the first day (inclusive)."""
        return _day_start(self.start)

    @property
    def end_exclusive_dt(self) -> dt.datetime:
        """UTC midnight after the last day, i.e. the exclusive upper bound."""
        return _day_start(self.end + dt.timedelta(days=1))

    def days(self) -> list[dt.date]:
        out: list[dt.date] = []
        day = self.start
        while day <= self.end:
            out.append(day)
            day += dt.timedelta(days=1)
        return out


def _day_start(day: dt.date) -> dt.datetime:
    return dt.datetime.combine(day, dt.time.min, tzinfo=dt.UTC)


def cost_event(
    *,
    provider: str,
    day: dt.date,
    category: str,
    detail: str,
    cost: float,
    currency: str = "USD",
    amount_kind: AmountKind = AmountKind.BILLED,
) -> CostEvent:
    """Build a :class:`CostEvent` for ``day``; ``collected_ts`` is stamped later."""
    return CostEvent(
        ts=_day_start(day),
        usage_date=day.isoformat(),
        provider=provider,
        category=category,
        detail=detail,
        cost=float(cost),
        currency=currency,
        amount_kind=str(amount_kind),
    )


def stamp_collected(events: list[CostEvent], collected_at: dt.datetime) -> list[CostEvent]:
    """Return ``events`` with ``collected_ts`` set to one shared run timestamp."""
    return [replace(e, collected_ts=collected_at) for e in events]
