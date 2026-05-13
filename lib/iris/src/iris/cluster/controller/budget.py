# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Budget tracking: resource value function and per-user spend."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from rigging.timing import Timestamp
from sqlalchemy import bindparam, func, select

from iris.cluster.controller import writes
from iris.cluster.controller.codec import device_counts_from_json
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.schema import job_config_table, tasks_table
from iris.cluster.controller.task_state import ACTIVE_TASK_STATES
from iris.rpc import config_pb2, job_pb2

logger = logging.getLogger(__name__)

T = TypeVar("T")


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


_USER_SPEND_QUERY = (
    select(
        tasks_table.c.job_id,
        job_config_table.c.res_cpu_millicores,
        job_config_table.c.res_memory_bytes,
        job_config_table.c.res_device_json,
        func.count().label("task_count"),
    )
    .select_from(tasks_table.join(job_config_table, job_config_table.c.job_id == tasks_table.c.job_id))
    .where(tasks_table.c.state.in_(bindparam("states", expanding=True)))
    .where(job_config_table.c.priority_band != job_pb2.PRIORITY_BAND_BATCH)
    .group_by(tasks_table.c.job_id)
)


def compute_user_spend(tx: Tx) -> dict[str, int]:
    """Compute per-user budget spend from active tasks.

    Joins tasks (in ASSIGNED/BUILDING/RUNNING states) with job_config to get
    resource columns.  Groups by job, then sums resource_value * task_count per user.

    Jobs whose requested band is ``PRIORITY_BAND_BATCH`` are excluded so users
    aren't billed for opportunistic work they explicitly submitted as batch.
    We key off ``job_config.priority_band`` (the user's requested band) rather
    than the stamped ``tasks.priority_band`` so jobs the scheduler downgraded
    to BATCH still count — otherwise a downgrade would drop the user under
    budget on the next tick and the band would oscillate.

    Returns ``{user_id: total_resource_value}`` for users with active tasks.
    """
    rows = tx.execute(_USER_SPEND_QUERY, {"states": list(_ACTIVE_TASK_STATES)}).all()

    spend: dict[str, int] = defaultdict(int)
    for row in rows:
        # job_id is decoded by JobNameType to JobName
        user_id = row.job_id.user
        cpu = row.res_cpu_millicores
        mem = row.res_memory_bytes
        counts = device_counts_from_json(row.res_device_json)
        accel = counts.gpu + counts.tpu
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

    Defense-in-depth: a leaked UNSPECIFIED (0) is normalized to INTERACTIVE
    so it cannot sort ahead of PRODUCTION under ``ORDER BY priority_band
    ASC``. Callers should resolve UNSPECIFIED upstream (parent inheritance,
    then INTERACTIVE default) — see ``reads.jobs.get_priority_bands``.
    """
    if task_band == job_pb2.PRIORITY_BAND_UNSPECIFIED:
        task_band = job_pb2.PRIORITY_BAND_INTERACTIVE
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
            with db.transaction() as _tx:
                writes.ensure_user(_tx, user_id, now)
                writes.set_user_budget(_tx, user_id, tier.budget_limit, tier.max_band, now)
            count += 1
    if count:
        logger.info("Reconciled %d user budget assignment(s) from cluster config", count)
    return count
