# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQL expressions for retry counts derived from persisted Attempt rows."""

from sqlalchemy import case, func
from sqlalchemy.sql.elements import ColumnElement

from iris.cluster.controller.persistence.schema import task_attempts_table
from iris.cluster.resources.attempt import PREEMPTION_ATTEMPT_STATES
from iris.cluster.resources.state import TaskState


def failure_count_expr() -> ColumnElement[int]:
    """Count FAILED Attempts in an aggregate over ``task_attempts``."""
    return func.coalesce(
        func.sum(case((task_attempts_table.c.state == TaskState.FAILED, 1), else_=0)),
        0,
    )


def preemption_count_expr() -> ColumnElement[int]:
    """Count executing-phase preemptions in an aggregate over ``task_attempts``."""
    return func.coalesce(
        func.sum(
            case(
                (
                    task_attempts_table.c.state.in_(PREEMPTION_ATTEMPT_STATES)
                    & task_attempts_table.c.started_at_ms.is_not(None),
                    1,
                ),
                else_=0,
            )
        ),
        0,
    )
