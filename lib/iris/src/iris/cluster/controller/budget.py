# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Budget tracking: resource value function and per-user spend."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Generic, TypeVar

from iris.cluster.controller.db import ACTIVE_TASK_STATES, QuerySnapshot
from iris.cluster.types import JobName
from iris.cluster.types import get_gpu_count, get_tpu_count
from iris.rpc import cluster_pb2

T = TypeVar("T")


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

    max_band: int = cluster_pb2.PRIORITY_BAND_INTERACTIVE
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

    Joins tasks (in ASSIGNED/BUILDING/RUNNING states) with jobs to get user_id
    and the resource spec proto.  Resource values are computed in Python because
    the spec lives in a serialized proto blob.

    Returns ``{user_id: total_resource_value}`` for users with active tasks.
    """
    placeholders = ",".join("?" for _ in _ACTIVE_TASK_STATES)
    rows = snapshot.raw(
        f"SELECT j.job_id, j.request_proto FROM tasks t "
        f"JOIN jobs j ON t.job_id = j.job_id "
        f"WHERE t.state IN ({placeholders})",
        tuple(_ACTIVE_TASK_STATES),
        decoders={"job_id": JobName.from_wire},
    )

    spend: dict[str, int] = defaultdict(int)
    for row in rows:
        user_id = row.job_id.user
        request = cluster_pb2.Controller.LaunchJobRequest()
        request.ParseFromString(row.request_proto)
        res = request.resources
        accel = get_gpu_count(res.device) + get_tpu_count(res.device)
        spend[user_id] += resource_value(res.cpu_millicores, res.memory_bytes, accel)
    return dict(spend)


def compute_effective_band(
    task_band: int, user_id: str, user_spend: dict[str, int], user_budgets: dict[str, int]
) -> int:
    """Downgrade task to BATCH if its user exceeds their budget.

    PRODUCTION tasks are never downgraded.  A budget_limit of 0 means unlimited.
    """
    if task_band == cluster_pb2.PRIORITY_BAND_PRODUCTION:
        return task_band
    limit = user_budgets.get(user_id, 0)
    if limit > 0 and user_spend.get(user_id, 0) > limit:
        return max(task_band, cluster_pb2.PRIORITY_BAND_BATCH)
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
