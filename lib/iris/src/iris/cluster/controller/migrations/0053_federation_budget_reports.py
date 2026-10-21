# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist peer budget usage independently of handed-off job mirrors."""


def migrate(raw_conn) -> None:
    raw_conn.execute(
        "CREATE TABLE IF NOT EXISTS federation_budget_reports ("
        "cluster TEXT PRIMARY KEY, authority_id TEXT NOT NULL, user_spend TEXT NOT NULL, "
        "reported_at_ms INTEGER NOT NULL, "
        "refresh_epoch TEXT NOT NULL, refresh_generation INTEGER NOT NULL)"
    )
    # Existing task mirrors have placeholder bands. Re-fetch the full active
    # subtree so unchanged running tasks also receive their peer's real band.
    raw_conn.execute("DELETE FROM federation_sync_state")
