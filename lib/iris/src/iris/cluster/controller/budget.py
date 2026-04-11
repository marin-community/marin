# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Budget tracking: resource value function and per-user spend."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Generic, TypeVar

import json

from iris.cluster.controller.db import ACTIVE_TASK_STATES, QuerySnapshot
from iris.cluster.types import JobName
from iris.rpc import job_pb2

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
    """Defaults for new user budget rows created at job submission time."""

    budget_limit: int = 0
    """Max budget value (0 = unlimited)."""

    max_band: int = job_pb2.PRIORITY_BAND_INTERACTIVE
    """Default max priority band (proto int) for new users."""


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
        cpu = int(row.res_cpu_millicores or 0)
        mem = int(row.res_memory_bytes or 0)
        accel = _accel_from_device_json(row.res_device_json)
        value = resource_value(cpu, mem, accel)
        spend[user_id] += value * int(row.task_count)
    return dict(spend)


def compute_effective_band(
    task_band: int, user_id: str, user_spend: dict[str, int], user_budgets: dict[str, int]
) -> int:
    """Downgrade task to BATCH if its user exceeds their budget.

    PRODUCTION tasks are never downgraded.  A budget_limit of 0 means unlimited.
    """
    if task_band == job_pb2.PRIORITY_BAND_PRODUCTION:
        return task_band
    limit = user_budgets.get(user_id, 0)
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
