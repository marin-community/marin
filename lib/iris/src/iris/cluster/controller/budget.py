# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Budget tracking, priority bands, and fairness helpers for the scheduling loop."""

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum

from iris.cluster.controller.db import ControllerDB
from iris.cluster.types import get_gpu_count, get_tpu_count
from iris.rpc import cluster_pb2


class PriorityBand(StrEnum):
    PRODUCTION = "production"
    INTERACTIVE = "interactive"
    BATCH = "batch"


# Map proto enum value → PriorityBand (0 = UNSPECIFIED defaults to INTERACTIVE)
BAND_FROM_PROTO: dict[int, PriorityBand] = {
    0: PriorityBand.INTERACTIVE,
    1: PriorityBand.PRODUCTION,
    2: PriorityBand.INTERACTIVE,
    3: PriorityBand.BATCH,
}

# Band → DB sort key (lower = higher priority)
BAND_SORT_KEY: dict[PriorityBand, int] = {
    PriorityBand.PRODUCTION: 1,
    PriorityBand.INTERACTIVE: 2,
    PriorityBand.BATCH: 3,
}

# Reverse: DB sort key → PriorityBand
BAND_FROM_SORT_KEY: dict[int, PriorityBand] = {v: k for k, v in BAND_SORT_KEY.items()}


@dataclass
class UserBudgetDefaults:
    """Defaults for new user budget rows created at job submission time."""

    budget_limit: int = 0
    """Max budget value (0 = unlimited)."""

    max_band: str = "interactive"
    """Default max band for new users."""


def resource_value(resources: cluster_pb2.ResourceSpecProto) -> int:
    """Collapse heterogeneous resources into a single cost number.

    Weights: 1000 per accelerator chip, 1 per GB RAM, 5 per CPU core.
    """
    accel_count = get_gpu_count(resources.device) + get_tpu_count(resources.device)
    ram_gb = resources.memory_bytes // (1024**3)
    cpu_cores = resources.cpu_millicores // 1000
    return 1000 * accel_count + ram_gb + 5 * cpu_cores


def compute_user_spend(db: ControllerDB) -> dict[str, int]:
    """Compute per-user budget spend from currently active tasks.

    Returns {user_id: total_resource_value} for users with tasks in
    ASSIGNED, BUILDING, or RUNNING states. Uses the job's request proto
    to extract resource specs.
    """
    rows = db.fetchall(
        "SELECT j.user_id, j.request_proto "
        "FROM tasks t "
        "JOIN jobs j ON t.job_id = j.job_id "
        "WHERE t.state IN (?, ?, ?)",
        (
            cluster_pb2.TASK_STATE_ASSIGNED,
            cluster_pb2.TASK_STATE_BUILDING,
            cluster_pb2.TASK_STATE_RUNNING,
        ),
    )
    spend: dict[str, int] = defaultdict(int)
    for row in rows:
        request = cluster_pb2.Controller.LaunchJobRequest()
        request.ParseFromString(row["request_proto"])
        spend[row["user_id"]] += resource_value(request.resources)
    return dict(spend)


def interleave_by_user(
    user_task_pairs: list[tuple[str, object]],
    user_spend: dict[str, int],
) -> list:
    """Round-robin tasks across users, ordered by ascending budget spend.

    Takes (user_id, item) pairs and returns items interleaved so that
    low-spend users' tasks come first in each round.

    Must be called once per band to avoid cross-band reordering.
    """
    by_user: dict[str, list] = defaultdict(list)
    for user_id, item in user_task_pairs:
        by_user[user_id].append(item)

    sorted_users = sorted(by_user.keys(), key=lambda u: user_spend.get(u, 0))

    result: list = []
    round_idx = 0
    while True:
        added = False
        for user in sorted_users:
            user_items = by_user[user]
            if round_idx < len(user_items):
                result.append(user_items[round_idx])
                added = True
        if not added:
            break
        round_idx += 1
    return result
