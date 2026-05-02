# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    """Clear the user_budgets table.

    Before this migration we eagerly seeded a row per user at job submission
    with budget_limit=0 / max_band=INTERACTIVE. Those rows are now stale: the
    controller treats an absent row as "apply UserBudgetDefaults" and
    reconciles explicit overrides from the cluster config's ``user_budgets``
    tier list at startup (see reconcile_user_budget_tiers). Any rows that
    should persist will be re-inserted by that reconcile step, which runs
    after all migrations.
    """
    conn.execute("DELETE FROM user_budgets")
