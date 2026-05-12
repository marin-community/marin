# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""User-budget and user-role read helpers (SA Core expression language).

All queries use ``select(table.c.col, ...)`` rather than ``text("SELECT
...")``. TypeDecorators on schema_v2 columns decode updated_at_ms to
Timestamp on read.

Return shapes:

* ``get_user_budget`` — ``UserBudget | None`` (lazy import to avoid cycle)
* ``list_user_budgets`` — ``list[UserBudget]``
* ``get_all_user_budget_limits`` — ``dict[str, int]``
* ``get_user_role`` — ``str`` (defaults to ``"user"`` if absent)
"""

from sqlalchemy import bindparam, select

from iris.cluster.controller.db_v2 import Tx
from iris.cluster.controller.schema_v2 import user_budgets_table, users_table

# ---------------------------------------------------------------------------
# User budgets
# ---------------------------------------------------------------------------

GET_USER_BUDGET_QUERY = select(
    user_budgets_table.c.user_id,
    user_budgets_table.c.budget_limit,
    user_budgets_table.c.max_band,
    user_budgets_table.c.updated_at_ms,
).where(user_budgets_table.c.user_id == bindparam("user_id"))

LIST_USER_BUDGETS_QUERY = select(
    user_budgets_table.c.user_id,
    user_budgets_table.c.budget_limit,
    user_budgets_table.c.max_band,
    user_budgets_table.c.updated_at_ms,
)


def get_user_budget(tx: Tx, user_id: str):
    """Return :class:`iris.cluster.controller.db.UserBudget` for ``user_id``, or None.

    UserBudget is imported lazily to avoid an import cycle. Row has
    updated_at_ms decoded to Timestamp by TimestampMsType.
    """
    from iris.cluster.controller.db import UserBudget

    row = tx.execute(GET_USER_BUDGET_QUERY, {"user_id": user_id}).first()
    if row is None:
        return None
    return UserBudget(
        user_id=str(row.user_id),
        budget_limit=int(row.budget_limit),
        max_band=int(row.max_band),
        updated_at=row.updated_at_ms,
    )


def list_user_budgets(tx: Tx) -> list:
    """Return every :class:`iris.cluster.controller.db.UserBudget` row."""
    from iris.cluster.controller.db import UserBudget

    rows = tx.execute(LIST_USER_BUDGETS_QUERY).all()
    return [
        UserBudget(
            user_id=str(row.user_id),
            budget_limit=int(row.budget_limit),
            max_band=int(row.max_band),
            updated_at=row.updated_at_ms,
        )
        for row in rows
    ]


def get_all_user_budget_limits(tx: Tx) -> dict[str, int]:
    """Return ``{user_id: budget_limit}`` for every user with a budget row."""
    rows = tx.execute(LIST_USER_BUDGETS_QUERY).all()
    return {str(row.user_id): int(row.budget_limit) for row in rows}


# ---------------------------------------------------------------------------
# User roles
# ---------------------------------------------------------------------------

GET_USER_ROLE_QUERY = select(users_table.c.role).where(users_table.c.user_id == bindparam("user_id"))


def get_user_role(tx: Tx, user_id: str) -> str:
    """Return the user's role, or ``"user"`` if not found (legacy default)."""
    row = tx.execute(GET_USER_ROLE_QUERY, {"user_id": user_id}).first()
    return str(row.role) if row is not None else "user"
