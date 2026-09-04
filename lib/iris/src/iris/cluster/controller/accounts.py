# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller operations for users, roles, and budgets."""

from dataclasses import dataclass, field

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from rigging.server_auth import get_verified_identity, require_identity
from rigging.timing import Timestamp
from sqlalchemy import bindparam, func, select

from iris.cluster.controller import reads, writes
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.budget import compute_user_spend
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.schema import jobs_table, tasks_table, user_budgets_table
from iris.cluster.types import TERMINAL_JOB_STATES, USER_JOB_STATES
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.auth import AuthzAction, authorize
from iris.rpc.proto_display import job_state_friendly, task_state_friendly


@dataclass(frozen=True, slots=True)
class AccountDependencies:
    db: ControllerDB
    auth: ControllerAuth


@dataclass(frozen=True)
class UserStats:
    user: str
    task_state_counts: dict[int, int] = field(default_factory=dict)
    job_state_counts: dict[int, int] = field(default_factory=dict)


_ACTIVE_JOB_STATES = (
    job_pb2.JOB_STATE_PENDING,
    job_pb2.JOB_STATE_BUILDING,
    job_pb2.JOB_STATE_RUNNING,
)

_USER_TASK_STATES = (
    job_pb2.TASK_STATE_PENDING,
    job_pb2.TASK_STATE_ASSIGNED,
    job_pb2.TASK_STATE_BUILDING,
    job_pb2.TASK_STATE_RUNNING,
    job_pb2.TASK_STATE_SUCCEEDED,
    job_pb2.TASK_STATE_FAILED,
    job_pb2.TASK_STATE_KILLED,
    job_pb2.TASK_STATE_UNSCHEDULABLE,
    job_pb2.TASK_STATE_WORKER_FAILED,
    job_pb2.TASK_STATE_PREEMPTED,
    job_pb2.TASK_STATE_COSCHED_FAILED,
)


def list_users(
    dependencies: AccountDependencies,
    request: controller_pb2.Controller.ListUsersRequest,
    context: RequestContext,
) -> controller_pb2.Controller.ListUsersResponse:
    """Return current per-user workload counts and configured roles."""
    del request, context
    role_policy = dependencies.auth.role_policy
    users = sorted(
        _live_user_stats(dependencies.db),
        key=lambda entry: (
            -_active_job_count(entry.job_state_counts),
            -(entry.task_state_counts.get(job_pb2.TASK_STATE_RUNNING, 0)),
            entry.user,
        ),
    )
    return controller_pb2.Controller.ListUsersResponse(
        users=[
            controller_pb2.Controller.UserSummary(
                user=entry.user,
                task_state_counts=_task_state_counts_for_summary(entry.task_state_counts),
                job_state_counts=_job_state_counts_for_summary(entry.job_state_counts),
                role=role_policy.role_for(entry.user) if role_policy else "",
            )
            for entry in users
        ]
    )


def get_current_user(
    request: job_pb2.GetCurrentUserRequest,
    context: RequestContext,
) -> job_pb2.GetCurrentUserResponse:
    del request, context
    identity = get_verified_identity()
    if identity is None:
        return job_pb2.GetCurrentUserResponse(user_id="anonymous", role="")
    return job_pb2.GetCurrentUserResponse(user_id=identity.user_id, role=identity.role)


def set_user_budget(
    dependencies: AccountDependencies,
    request: controller_pb2.Controller.SetUserBudgetRequest,
    context: RequestContext,
) -> controller_pb2.Controller.SetUserBudgetResponse:
    """Set one user's budget limit and maximum priority band."""
    del context
    authorize(AuthzAction.MANAGE_BUDGETS)
    if not request.user_id:
        raise ConnectError(Code.INVALID_ARGUMENT, "user_id is required")
    max_band = request.max_band or job_pb2.PRIORITY_BAND_INTERACTIVE
    if max_band not in (
        job_pb2.PRIORITY_BAND_PRODUCTION,
        job_pb2.PRIORITY_BAND_SYSTEM,
        job_pb2.PRIORITY_BAND_INTERACTIVE,
        job_pb2.PRIORITY_BAND_BATCH,
    ):
        raise ConnectError(Code.INVALID_ARGUMENT, f"Invalid max_band: {request.max_band}")
    with dependencies.db.transaction() as tx:
        writes.set_user_budget(tx, request.user_id, request.budget_limit, max_band, Timestamp.now())
    return controller_pb2.Controller.SetUserBudgetResponse()


def get_user_budget(
    dependencies: AccountDependencies,
    request: controller_pb2.Controller.GetUserBudgetRequest,
    context: RequestContext,
) -> controller_pb2.Controller.GetUserBudgetResponse:
    """Return one user's budget configuration and current spend."""
    del context
    require_identity()
    if not request.user_id:
        raise ConnectError(Code.INVALID_ARGUMENT, "user_id is required")
    with dependencies.db.read_snapshot() as snapshot:
        budget = reads.get_user_budget(snapshot, request.user_id)
        spend = compute_user_spend(snapshot)
    if budget is None:
        raise ConnectError(Code.NOT_FOUND, f"No budget found for user {request.user_id}")
    return controller_pb2.Controller.GetUserBudgetResponse(
        user_id=budget.user_id,
        budget_limit=budget.budget_limit,
        budget_spent=spend.get(request.user_id, 0),
        max_band=budget.max_band,
    )


def list_user_budgets(
    dependencies: AccountDependencies,
    request: controller_pb2.Controller.ListUserBudgetsRequest,
    context: RequestContext,
) -> controller_pb2.Controller.ListUserBudgetsResponse:
    """Return every configured user budget with current spend."""
    del request, context
    require_identity()
    with dependencies.db.read_snapshot() as snapshot:
        budgets = reads.list_user_budgets(snapshot)
        spend = compute_user_spend(snapshot)
    return controller_pb2.Controller.ListUserBudgetsResponse(
        users=[
            controller_pb2.Controller.GetUserBudgetResponse(
                user_id=budget.user_id,
                budget_limit=budget.budget_limit,
                budget_spent=spend.get(budget.user_id, 0),
                max_band=budget.max_band,
            )
            for budget in budgets
        ]
    )


def _active_job_count(counts: dict[int, int]) -> int:
    return sum(count for state, count in counts.items() if state not in TERMINAL_JOB_STATES)


def _task_state_counts_for_summary(counts: dict[int, int]) -> dict[str, int]:
    result = {task_state_friendly(state): 0 for state in _USER_TASK_STATES}
    for state, count in counts.items():
        result[task_state_friendly(state)] = count
    return result


def _job_state_counts_for_summary(counts: dict[int, int]) -> dict[str, int]:
    result = {job_state_friendly(state): 0 for state in USER_JOB_STATES}
    for state, count in counts.items():
        result[job_state_friendly(state)] = count
    return result


def _live_user_stats(db: ControllerDB) -> list[UserStats]:
    active_states = list(_ACTIVE_JOB_STATES)
    with db.read_snapshot() as tx:
        user_rows = tx.execute(select(jobs_table.c.user_id).distinct()).all()
        budget_rows = tx.execute(select(user_budgets_table.c.user_id)).all()
        job_rows = tx.execute(
            select(jobs_table.c.user_id, jobs_table.c.state, func.count().label("cnt"))
            .where(jobs_table.c.state.in_(bindparam("active_states", expanding=True)))
            .group_by(jobs_table.c.user_id, jobs_table.c.state),
            {"active_states": active_states},
        ).all()
        task_rows = tx.execute(
            select(jobs_table.c.user_id, tasks_table.c.state, func.count().label("cnt"))
            .select_from(tasks_table.join(jobs_table, tasks_table.c.job_id == jobs_table.c.job_id))
            .where(jobs_table.c.state.in_(bindparam("active_states", expanding=True)))
            .group_by(jobs_table.c.user_id, tasks_table.c.state),
            {"active_states": active_states},
        ).all()
    by_user = {str(row.user_id): UserStats(user=str(row.user_id)) for row in user_rows}
    for row in budget_rows:
        by_user.setdefault(str(row.user_id), UserStats(user=str(row.user_id)))
    for row in job_rows:
        stats = by_user.setdefault(str(row.user_id), UserStats(user=str(row.user_id)))
        stats.job_state_counts[int(row.state)] = int(row.cnt)
    for row in task_rows:
        stats = by_user.setdefault(str(row.user_id), UserStats(user=str(row.user_id)))
        stats.task_state_counts[int(row.state)] = int(row.cnt)
    return list(by_user.values())
