# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Budget tracking: resource value function and per-user spend."""

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

from rigging.timing import Timestamp

from iris.cluster.config import UserBudgetTier
from iris.cluster.controller import reads, writes
from iris.cluster.controller.codec import device_counts_from_json
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.types import LOCAL_ADMIN_SUBMITTER, JobName, PendingTask, UserBudgetDefaults
from iris.rpc import job_pb2
from iris.rpc.proto_display import ADMIN_PRIORITY_BAND_VALUES, PRIORITY_BAND_VALUES

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class UserTask(Generic[T]):
    user_id: str
    task: T


def budget_user_id(job_id: JobName, submitting_user: str) -> str:
    """Use the authenticated submitter, with job-owner fallback for local or empty identities."""
    if submitting_user and submitting_user != LOCAL_ADMIN_SUBMITTER:
        return submitting_user
    return job_id.user


def resource_value(cpu_millicores: int, memory_bytes: int, accelerator_count: int) -> int:
    """Compute a scalar resource value for budget tracking.

    Formula: 1000 * accelerators + RAM_GB + 5 * CPU_cores.
    Uses integer division so that fractional cores/GB are truncated.
    """
    ram_gb = memory_bytes // (1024**3)
    cpu_cores = cpu_millicores // 1000
    return 1000 * accelerator_count + ram_gb + 5 * cpu_cores


def compute_user_spend(tx: Tx) -> dict[str, int]:
    """Compute per-user budget spend from active tasks.

    Sums ``resource_value * task_count`` per user over the active, non-BATCH
    task rows returned by :func:`reads.user_spend_rows`.

    Returns ``{budget_user_id: total_resource_value}`` for principals with active tasks.
    """
    rows = reads.user_spend_rows(tx)

    spend: dict[str, int] = defaultdict(int)
    for row in rows:
        user_id = budget_user_id(row.job_id, str(row.submitting_user))
        cpu = row.res_cpu_millicores
        mem = row.res_memory_bytes
        counts = device_counts_from_json(row.res_device_json)
        accel = counts.gpu + counts.tpu
        value = resource_value(cpu, mem, accel)
        spend[user_id] += value * int(row.task_count)
    return dict(spend)


def task_accelerators(res_device_json: str | None) -> int:
    """Accelerators (GPUs + TPU chips) one task with this device spec holds."""
    counts = device_counts_from_json(res_device_json)
    return counts.gpu + counts.tpu


def placement_accelerators(per_task: int, pending_tasks: int, coscheduled: bool) -> int:
    """Accelerators one placement decision charges: the whole gang if coscheduled, else one task."""
    return per_task * pending_tasks if coscheduled else per_task


def compute_user_accelerators(tx: Tx) -> dict[str, int]:
    """Return ``{budget_user_id: accelerators}`` held by each identity's active tasks.

    Counts GPUs plus TPU chips per task, times the number of active (assigned,
    building, or running) tasks, across every band including BATCH.
    """
    in_use: dict[str, int] = defaultdict(int)
    for row in reads.user_active_resource_rows(tx):
        accelerators = task_accelerators(row.res_device_json)
        if accelerators:
            in_use[budget_user_id(row.job_id, str(row.submitting_user))] += accelerators * int(row.task_count)
    return dict(in_use)


@dataclass
class AcceleratorCaps:
    """Hard per-identity accelerator caps for one placement pass.

    ``caps`` holds only identities with a positive cap. ``in_use`` starts at what
    each identity's active tasks hold and grows as the pass admits work, so a
    single pass never places past a cap. Lowering a cap below ``in_use`` holds new
    work but never touches running tasks.
    """

    caps: dict[str, int]
    in_use: dict[str, int]
    job_users: dict[JobName, str]

    def hold_reason(self, job_id: JobName, accelerators: int) -> str | None:
        """Why placing ``accelerators`` more for ``job_id`` would breach its identity's cap, or None."""
        if accelerators <= 0:
            return None
        user = self.job_users.get(job_id)
        cap = self.caps.get(user, 0) if user is not None else 0
        if cap <= 0:
            return None
        used = self.in_use.get(user, 0)
        if used + accelerators <= cap:
            return None
        return f"held: user cap {cap} accelerators ({used} in use, needs {accelerators})"

    def try_admit(self, job_id: JobName, accelerators: int) -> bool:
        """Charge ``accelerators`` to ``job_id``'s identity if it stays within the cap."""
        if self.hold_reason(job_id, accelerators) is not None:
            return False
        user = self.job_users.get(job_id)
        if accelerators > 0 and user is not None and user in self.caps:
            self.in_use[user] = self.in_use.get(user, 0) + accelerators
        return True

    def copy(self) -> "AcceleratorCaps":
        return AcceleratorCaps(caps=self.caps, in_use=dict(self.in_use), job_users=self.job_users)


def build_accelerator_caps(
    tx: Tx,
    job_users: Mapping[JobName, str],
    defaults: UserBudgetDefaults,
) -> AcceleratorCaps:
    """Resolve the caps of the identities owning ``job_users`` and their current usage.

    An identity's cap is its ``user_budgets.max_accelerators`` row value, else
    ``defaults.max_accelerators``. Usage is read only when some identity is capped.
    """
    row_caps = reads.get_all_user_accelerator_caps(tx)
    caps: dict[str, int] = {}
    for user in set(job_users.values()):
        cap = row_caps.get(user, defaults.max_accelerators)
        if cap > 0:
            caps[user] = cap
    in_use = compute_user_accelerators(tx) if caps else {}
    return AcceleratorCaps(caps=caps, in_use=in_use, job_users=dict(job_users))


@dataclass(frozen=True)
class UserCapStatus:
    user_id: str
    max_accelerators: int
    accelerators_in_use: int
    held_tasks: int
    held_jobs: int


def user_cap_status(tx: Tx, user_id: str, defaults: UserBudgetDefaults) -> UserCapStatus:
    """Report ``user_id``'s accelerator cap, usage, and pending work the cap holds.

    A pending job counts as held when placing one of its tasks (or its whole
    gang, if coscheduled) at current usage would breach the cap.
    """
    budget = reads.get_user_budget(tx, user_id)
    cap = budget.max_accelerators if budget is not None else defaults.max_accelerators
    in_use = compute_user_accelerators(tx).get(user_id, 0)
    pending_by_job: dict[JobName, list[PendingTask]] = defaultdict(list)
    for task in reads.pending_tasks_with_jobs(tx):
        if budget_user_id(task.job_id, task.submitting_user) == user_id:
            pending_by_job[task.job_id].append(task)
    caps = AcceleratorCaps(
        caps={user_id: cap} if cap > 0 else {},
        in_use={user_id: in_use},
        job_users={job_id: user_id for job_id in pending_by_job},
    )
    held_tasks = 0
    held_jobs = 0
    for job_id, tasks in pending_by_job.items():
        needed = placement_accelerators(
            task_accelerators(tasks[0].res_device_json), len(tasks), tasks[0].has_coscheduling
        )
        if caps.hold_reason(job_id, needed) is not None:
            held_jobs += 1
            held_tasks += len(tasks)
    return UserCapStatus(
        user_id=user_id,
        max_accelerators=cap,
        accelerators_in_use=in_use,
        held_tasks=held_tasks,
        held_jobs=held_jobs,
    )


def compute_effective_band(
    task_band: int,
    user_id: str,
    user_spend: dict[str, int],
    user_budgets: dict[str, int],
    defaults: UserBudgetDefaults,
) -> int:
    """Downgrade task to BATCH if its user exceeds their budget.

    SYSTEM and PRODUCTION tasks are never downgraded. Users without a
    ``user_budgets`` row fall back to ``defaults.budget_limit``; a limit of 0
    means unlimited.

    ``task_band`` is a real band: ``LaunchJob`` resolves INHERIT once at ingestion (see
    :func:`iris.cluster.controller.ops.job.resolve_priority_band`).
    """
    if task_band in ADMIN_PRIORITY_BAND_VALUES:
        return task_band
    limit = user_budgets.get(user_id, defaults.budget_limit)
    if limit > 0 and user_spend.get(user_id, 0) > limit:
        return job_pb2.PRIORITY_BAND_BATCH
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
# set so a missing/zeroed max_band field surfaces as a config error; callers
# must pick a real band.
_VALID_TIER_BANDS = frozenset(PRIORITY_BAND_VALUES)


def reconcile_user_budget_tiers(
    db: ControllerDB,
    tiers: Iterable[UserBudgetTier],
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
    :func:`compute_effective_band` and the launch guard in `jobs.py`).

    Config seeds the accelerator cap, the DB decides: a tier's
    ``max_accelerators`` is written only for users whose cap was never set at
    runtime (``iris user cap set``), so a restart cannot undo an admin's cap.

    Returns the number of (user_id, tier) pairs applied; duplicate user_ids
    across tiers are counted per-apply since the later tier overwrites.
    """
    count = 0
    # Startup-only: apply every tier under one transaction so a bad tier rolls the
    # whole reconcile back rather than leaving a half-applied budget state.
    with db.transaction() as _tx:
        for tier in tiers:
            if tier.max_band not in _VALID_TIER_BANDS:
                raise ValueError(
                    f"UserBudgetTier.max_band must be one of SYSTEM/PRODUCTION/INTERACTIVE/BATCH; "
                    f"got {tier.max_band} for users {list(tier.user_ids)}"
                )
            for user_id in tier.user_ids:
                if not user_id:
                    raise ValueError("UserBudgetTier.user_ids contains an empty entry")
                writes.set_user_budget(_tx, user_id, tier.budget_limit, tier.max_band, now)
                writes.seed_user_cap(_tx, user_id, tier.max_accelerators)
                count += 1
    if count:
        logger.info("Reconciled %d user budget assignment(s) from cluster config", count)
    return count
