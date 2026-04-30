# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Budget tracking: resource value function and per-user spend."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from rigging.timing import Timestamp

from iris.cluster.controller.db import ACTIVE_TASK_STATES, ControllerDB, QuerySnapshot
from iris.cluster.types import JobName
from iris.rpc import config_pb2, job_pb2

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _accel_from_device_json(device_json: str | None) -> int:
    """Count GPU + TPU accelerators from a device JSON column."""
    if not device_json:
        return 0
    data = json.loads(device_json)
    if "gpu" in data:
        return data["gpu"].get("count", 0)
    if "tpu" in data:
        return data["tpu"].get("count", 0)
    return 0


@dataclass(frozen=True)
class UserTask(Generic[T]):
    user_id: str
    task: T


# Task states that count as "active" for budget spend (re-exported from db for local use)
_ACTIVE_TASK_STATES = tuple(ACTIVE_TASK_STATES)


@dataclass
class UserBudgetDefaults:
    """Budget settings for users without an explicit user_budgets row.

    An absent row means "defaults apply" — we no longer stamp a row into the
    table at first-submit time, so the scheduler and launch-job guard both
    fall back to these values when the lookup misses.
    """

    budget_limit: int = 1000
    """Max budget value applied to users without an override row.

    A value of 0 means unlimited; any positive value caps accumulated spend
    before :func:`compute_effective_band` downgrades INTERACTIVE work to BATCH.
    """

    max_band: int = job_pb2.PRIORITY_BAND_INTERACTIVE
    """Default max priority band (proto int) for users without an override row."""


def resource_value(cpu_millicores: int, memory_bytes: int, accelerator_count: int) -> int:
    """Compute a scalar resource value for budget tracking.

    Formula: 1000 * accelerators + RAM_GB + 5 * CPU_cores.
    Uses integer division so that fractional cores/GB are truncated.
    """
    ram_gb = memory_bytes // (1024**3)
    cpu_cores = cpu_millicores // 1000
    return 1000 * accelerator_count + ram_gb + 5 * cpu_cores


def compute_user_spend(snapshot: QuerySnapshot) -> dict[str, int]:
    """Compute per-user budget spend from active tasks.

    Joins tasks (in ASSIGNED/BUILDING/RUNNING states) with job_config to get
    resource columns.  Groups by job, then sums resource_value * task_count per user.

    Returns ``{user_id: total_resource_value}`` for users with active tasks.
    """
    placeholders = ",".join("?" for _ in _ACTIVE_TASK_STATES)
    rows = snapshot.raw(
        f"SELECT jc.job_id, jc.res_cpu_millicores, jc.res_memory_bytes, jc.res_device_json, "
        f"COUNT(*) as task_count "
        f"FROM tasks t JOIN job_config jc ON t.job_id = jc.job_id "
        f"WHERE t.state IN ({placeholders}) "
        f"GROUP BY jc.job_id",
        tuple(_ACTIVE_TASK_STATES),
        decoders={"job_id": JobName.from_wire},
    )

    spend: dict[str, int] = defaultdict(int)
    for row in rows:
        user_id = row.job_id.user
        cpu = row.res_cpu_millicores
        mem = row.res_memory_bytes
        accel = _accel_from_device_json(row.res_device_json)
        value = resource_value(cpu, mem, accel)
        spend[user_id] += value * int(row.task_count)
    return dict(spend)


def compute_effective_band(
    task_band: int,
    user_id: str,
    user_spend: dict[str, int],
    user_budgets: dict[str, int],
    defaults: UserBudgetDefaults,
) -> int:
    """Downgrade task to BATCH if its user exceeds their budget.

    PRODUCTION tasks are never downgraded. Users without a ``user_budgets``
    row fall back to ``defaults.budget_limit``; a limit of 0 means unlimited.
    """
    if task_band == job_pb2.PRIORITY_BAND_PRODUCTION:
        return task_band
    limit = user_budgets.get(user_id, defaults.budget_limit)
    if limit > 0 and user_spend.get(user_id, 0) > limit:
        return max(task_band, job_pb2.PRIORITY_BAND_BATCH)
    return task_band


def interleave_by_user(
    tasks: list[UserTask[T]],
    user_spend: dict[str, int],
) -> list[T]:
    """Round-robin tasks across users, ordered by ascending budget spend.

    ``tasks`` is a list of :class:`UserTask` entries. The returned list
    contains only the task objects (user_id is stripped).

    Users who have spent less get their tasks earlier in each round.
    Must be called separately for each priority band to avoid cross-band
    reordering.
    """
    by_user: dict[str, list[T]] = defaultdict(list)
    for ut in tasks:
        by_user[ut.user_id].append(ut.task)

    sorted_users = sorted(by_user.keys(), key=lambda u: user_spend.get(u, 0))

    result: list[T] = []
    round_idx = 0
    while True:
        added = False
        for user in sorted_users:
            user_tasks = by_user[user]
            if round_idx < len(user_tasks):
                result.append(user_tasks[round_idx])
                added = True
        if not added:
            break
        round_idx += 1
    return result


# Bands accepted in user_budgets config entries. UNSPECIFIED is kept out of the
# set so a missing/zeroed max_band field surfaces as a config error rather than
# silently granting BATCH; callers must pick a real band.
_VALID_TIER_BANDS = frozenset(
    (
        job_pb2.PRIORITY_BAND_PRODUCTION,
        job_pb2.PRIORITY_BAND_INTERACTIVE,
        job_pb2.PRIORITY_BAND_BATCH,
    )
)


def reconcile_user_budget_tiers(
    db: ControllerDB,
    tiers: Iterable[config_pb2.UserBudgetTier],
    now: Timestamp,
) -> int:
    """Upsert per-user budgets from cluster config into the user_budgets table.

    Runs at controller startup after auth is resolved. Each tier entry lists
    a set of user_ids that all receive the same budget_limit and max_band.
    Tiers are applied in order, so later tiers override earlier ones for
    users listed in both — lets ops promote a user by appending a later tier
    without editing earlier ones.

    Unlisted users don't get a row; their effective budget and max_band come
    from :class:`UserBudgetDefaults` at read time (see
    :func:`compute_effective_band` and the launch-job guard in service.py).

    Returns the number of (user_id, tier) pairs applied; duplicate user_ids
    across tiers are counted per-apply since the later tier overwrites.
    """
    count = 0
    for tier in tiers:
        if tier.max_band not in _VALID_TIER_BANDS:
            raise ValueError(
                f"UserBudgetTier.max_band must be one of PRODUCTION/INTERACTIVE/BATCH; "
                f"got {tier.max_band} for users {list(tier.user_ids)}"
            )
        for user_id in tier.user_ids:
            if not user_id:
                raise ValueError("UserBudgetTier.user_ids contains an empty entry")
            db.ensure_user(user_id, now)
            db.set_user_budget(
                user_id=user_id,
                budget_limit=tier.budget_limit,
                max_band=tier.max_band,
                now=now,
            )
            count += 1
    if count:
        logger.info("Reconciled %d user budget assignment(s) from cluster config", count)
    return count
