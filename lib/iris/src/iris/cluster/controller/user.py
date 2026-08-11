# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical user-level resource reads."""

from iris.cluster.controller.budget import compute_user_spend
from iris.cluster.controller.dependencies import ResourceDependencies
from iris.cluster.controller.persistence import reads
from iris.resources.state import JobState, TaskState
from iris.resources.user import UserSummary

_ACTIVE_JOB_STATES = (JobState.PENDING, JobState.BUILDING, JobState.RUNNING)


class UserResources:
    """Read user summaries from canonical controller persistence."""

    def __init__(self, dependencies: ResourceDependencies) -> None:
        self._dependencies = dependencies

    def list_users(self) -> tuple[UserSummary, ...]:
        with self._dependencies.db.read_snapshot() as snapshot:
            rows = reads.live_user_state_counts(snapshot, _ACTIVE_JOB_STATES)
            budgets = {budget.user_id: budget for budget in reads.list_user_budgets(snapshot)}
            spend = compute_user_spend(snapshot)
        role_policy = self._dependencies.auth.role_policy
        return tuple(
            UserSummary(
                user_id=row.user_id,
                task_state_counts=tuple(sorted((TaskState(state), count) for state, count in row.task_states.items())),
                job_state_counts=tuple(sorted((JobState(state), count) for state, count in row.job_states.items())),
                role=role_policy.role_for(row.user_id) if role_policy is not None else "",
                budget_limit=budgets[row.user_id].budget_limit if row.user_id in budgets else None,
                budget_spent=spend.get(row.user_id, 0),
                max_band=budgets[row.user_id].max_band if row.user_id in budgets else None,
            )
            for row in sorted(rows, key=lambda row: row.user_id)
        )
