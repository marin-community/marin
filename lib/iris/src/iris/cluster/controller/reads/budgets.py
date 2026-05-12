# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""User-budget and user-role read helpers (SA Core port).

Mirrors the budget/user accessors currently inline on
:class:`iris.cluster.controller.db.ControllerDB`
(``get_user_budget``, ``list_user_budgets``, ``get_all_user_budget_limits``,
``get_user_role``). The legacy methods open their own ``read_snapshot``
internally; the SA Core ports take an explicit :class:`Tx` so callers can
chain multiple reads onto a single snapshot.
"""

from rigging.timing import Timestamp
from sqlalchemy import text

from iris.cluster.controller.db_v2 import Tx

# ---------------------------------------------------------------------------
# User budgets
# ---------------------------------------------------------------------------

_GET_USER_BUDGET_SQL = text(
    "SELECT user_id, budget_limit, max_band, updated_at_ms " "FROM user_budgets WHERE user_id = :uid"
)

_LIST_USER_BUDGETS_SQL = text("SELECT user_id, budget_limit, max_band, updated_at_ms FROM user_budgets")


def get_user_budget(tx: Tx, user_id: str):
    """Return :class:`iris.cluster.controller.db.UserBudget` for ``user_id``, or None."""
    from iris.cluster.controller.db import UserBudget

    row = tx.execute(_GET_USER_BUDGET_SQL, {"uid": user_id}).first()
    if row is None:
        return None
    return UserBudget(
        user_id=str(row.user_id),
        budget_limit=int(row.budget_limit),
        max_band=int(row.max_band),
        updated_at=Timestamp.from_ms(int(row.updated_at_ms)),
    )


def list_user_budgets(tx: Tx) -> list:
    """Return every :class:`iris.cluster.controller.db.UserBudget` row."""
    from iris.cluster.controller.db import UserBudget

    rows = tx.execute(_LIST_USER_BUDGETS_SQL).all()
    return [
        UserBudget(
            user_id=str(row.user_id),
            budget_limit=int(row.budget_limit),
            max_band=int(row.max_band),
            updated_at=Timestamp.from_ms(int(row.updated_at_ms)),
        )
        for row in rows
    ]


def get_all_user_budget_limits(tx: Tx) -> dict[str, int]:
    """Return ``{user_id: budget_limit}`` for every user with a budget row."""
    rows = tx.execute(_LIST_USER_BUDGETS_SQL).all()
    return {str(row.user_id): int(row.budget_limit) for row in rows}


# ---------------------------------------------------------------------------
# User roles
# ---------------------------------------------------------------------------

_GET_USER_ROLE_SQL = text("SELECT role FROM users WHERE user_id = :uid")


def get_user_role(tx: Tx, user_id: str) -> str:
    """Return the user's role, or ``"user"`` if not found (legacy default)."""
    row = tx.execute(_GET_USER_ROLE_SQL, {"uid": user_id}).first()
    return str(row.role) if row is not None else "user"
