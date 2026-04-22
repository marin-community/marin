# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3

# Admins: standard budget, may submit PRIORITY_BAND_PRODUCTION for work that
# must not be downgraded. INTERACTIVE work still counts against the budget.
PRODUCTION_USERS = ("runner", "power", "dlwh", "rav", "romain", "held", "larry")

# Researchers: standard budget, capped at PRIORITY_BAND_INTERACTIVE.
STANDARD_BUDGET_USERS = (
    "ruili",
    "quevedo",
    "pc0618",
    "konwoo",
    "ahmedah",
    "ahmed",
    "rohith",
    "tim",
    "eczech",
    "tonyhlee",
    "kevin",
    "calvinxu",
    "moojink",
)

BUDGETED_USERS = PRODUCTION_USERS + STANDARD_BUDGET_USERS

STANDARD_BUDGET = 75000

# Priority band values mirror iris.job.PriorityBand. Hard-coded to keep
# migrations free of proto imports.
PRIORITY_BAND_PRODUCTION = 1
PRIORITY_BAND_INTERACTIVE = 2
PRIORITY_BAND_BATCH = 3


def migrate(conn: sqlite3.Connection) -> None:
    now_ms_sql = "CAST(strftime('%s','now') AS INTEGER) * 1000"

    # Force the standard budget on both admins and researchers, regardless of
    # prior state. Admins can still bypass the cap by submitting to PRODUCTION.
    budgeted_placeholders = ",".join("?" for _ in BUDGETED_USERS)
    conn.execute(
        f"UPDATE user_budgets SET budget_limit = ?, updated_at_ms = {now_ms_sql} "
        f"WHERE user_id IN ({budgeted_placeholders})",
        (STANDARD_BUDGET, *BUDGETED_USERS),
    )

    # Max-band policy: everyone defaults to BATCH, then researchers get
    # INTERACTIVE, then admins get PRODUCTION. The broad reset intentionally
    # stomps any pre-existing grants so this migration is prescriptive.
    conn.execute(
        f"UPDATE user_budgets SET max_band = ?, updated_at_ms = {now_ms_sql}",
        (PRIORITY_BAND_BATCH,),
    )
    researcher_placeholders = ",".join("?" for _ in STANDARD_BUDGET_USERS)
    conn.execute(
        f"UPDATE user_budgets SET max_band = ?, updated_at_ms = {now_ms_sql} "
        f"WHERE user_id IN ({researcher_placeholders})",
        (PRIORITY_BAND_INTERACTIVE, *STANDARD_BUDGET_USERS),
    )
    admin_placeholders = ",".join("?" for _ in PRODUCTION_USERS)
    conn.execute(
        f"UPDATE user_budgets SET max_band = ?, updated_at_ms = {now_ms_sql} "
        f"WHERE user_id IN ({admin_placeholders})",
        (PRIORITY_BAND_PRODUCTION, *PRODUCTION_USERS),
    )
