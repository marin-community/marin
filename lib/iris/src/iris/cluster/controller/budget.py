# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Budget tracking: resource value function, priority bands, and per-user spend."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from iris.cluster.controller.db import JOBS, TASKS, QuerySnapshot
from iris.cluster.types import get_gpu_count, get_tpu_count


class PriorityBand(StrEnum):
    PRODUCTION = "production"
    INTERACTIVE = "interactive"
    BATCH = "batch"


# Map proto enum int value → PriorityBand
BAND_FROM_PROTO: dict[int, PriorityBand] = {
    0: PriorityBand.INTERACTIVE,  # UNSPECIFIED defaults to INTERACTIVE
    1: PriorityBand.PRODUCTION,
    2: PriorityBand.INTERACTIVE,
    3: PriorityBand.BATCH,
}

# Band → integer sort key for DB (lower = higher priority)
BAND_SORT_KEY: dict[PriorityBand, int] = {
    PriorityBand.PRODUCTION: 1,
    PriorityBand.INTERACTIVE: 2,
    PriorityBand.BATCH: 3,
}

# Reverse: integer sort key → PriorityBand
BAND_FROM_SORT_KEY: dict[int, PriorityBand] = {v: k for k, v in BAND_SORT_KEY.items()}

# Task states that count as "active" for budget spend
_ACTIVE_TASK_STATES = (2, 3, 9)  # BUILDING, RUNNING, ASSIGNED


@dataclass
class UserBudgetDefaults:
    """Defaults for new user budget rows created at job submission time."""

    budget_limit: int = 0
    """Max budget value (0 = unlimited)."""

    max_band: str = "interactive"
    """Default max band for new users."""


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

    Joins tasks (in ASSIGNED/BUILDING/RUNNING states) with jobs to get user_id
    and the resource spec proto.  Resource values are computed in Python because
    the spec lives in a serialized proto blob.

    Returns ``{user_id: total_resource_value}`` for users with active tasks.
    """
    joined = TASKS.join(JOBS, on=TASKS.c.job_id == JOBS.c.job_id)
    rows = snapshot.select(
        joined,
        columns=[JOBS.c.job_id, JOBS.c.request_proto],
        where=TASKS.c.state.in_(_ACTIVE_TASK_STATES),
    )

    spend: dict[str, int] = defaultdict(int)
    for row in rows:
        user_id = row.job_id.user
        res = row.request_proto.resources
        accel = get_gpu_count(res.device) + get_tpu_count(res.device)
        spend[user_id] += resource_value(res.cpu_millicores, res.memory_bytes, accel)
    return dict(spend)


def interleave_by_user(
    tasks: list[tuple[str, Any]],
    user_spend: dict[str, int],
) -> list[Any]:
    """Round-robin tasks across users, ordered by ascending budget spend.

    ``tasks`` is a list of ``(user_id, task_object)`` pairs.  The returned list
    contains only the task objects (user_id is stripped).

    Users who have spent less get their tasks earlier in each round.
    Must be called separately for each priority band to avoid cross-band
    reordering.
    """
    by_user: dict[str, list[Any]] = defaultdict(list)
    for user_id, task in tasks:
        by_user[user_id].append(task)

    sorted_users = sorted(by_user.keys(), key=lambda u: user_spend.get(u, 0))

    result: list[Any] = []
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
