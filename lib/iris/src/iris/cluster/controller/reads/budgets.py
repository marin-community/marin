# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""User-budget and user-role read helpers.

Return shapes:

* ``get_user_budget`` — ``UserBudget | None`` (lazy import to avoid cycle)
* ``list_user_budgets`` — ``list[UserBudget]``
* ``get_all_user_budget_limits`` — ``dict[str, int]``
* ``get_user_role`` — ``str`` (defaults to ``"user"`` if absent)
"""

from sqlalchemy import bindparam, select

from iris.cluster.controller.db import Tx
from iris.cluster.controller.schema import user_budgets_table, users_table

# ---------------------------------------------------------------------------
# User budgets
# ---------------------------------------------------------------------------


def get_user_budget(tx: Tx, user_id: str):
    """Return :class:`iris.cluster.controller.db.UserBudget` for ``user_id``, or None.

    UserBudget is imported lazily to avoid an import cycle.
    """
    from iris.cluster.controller.db import UserBudget

    row = tx.execute(
        select(
            user_budgets_table.c.user_id,
            user_budgets_table.c.budget_limit,
            user_budgets_table.c.max_band,
            user_budgets_table.c.updated_at_ms,
        ).where(user_budgets_table.c.user_id == bindparam("user_id")),
        {"user_id": user_id},
    ).first()
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

    rows = tx.execute(
        select(
            user_budgets_table.c.user_id,
            user_budgets_table.c.budget_limit,
            user_budgets_table.c.max_band,
            user_budgets_table.c.updated_at_ms,
        )
    ).all()
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
    rows = tx.execute(
        select(
            user_budgets_table.c.user_id,
            user_budgets_table.c.budget_limit,
            user_budgets_table.c.max_band,
            user_budgets_table.c.updated_at_ms,
        )
    ).all()
    return {str(row.user_id): int(row.budget_limit) for row in rows}


# ---------------------------------------------------------------------------
# User roles
# ---------------------------------------------------------------------------


def get_user_role(tx: Tx, user_id: str) -> str:
    """Return the user's role, or ``"user"`` if not found."""
    row = tx.execute(
        select(users_table.c.role).where(users_table.c.user_id == bindparam("user_id")),
        {"user_id": user_id},
    ).first()
    return str(row.role) if row is not None else "user"
