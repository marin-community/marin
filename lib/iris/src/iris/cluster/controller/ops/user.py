# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate-scoped commands for users."""

from rigging.timing import Timestamp

from iris.cluster.controller import reads, writes
from iris.cluster.controller.db import Tx


def ensure_user_and_role(cur: Tx, user_id: str, now: Timestamp) -> str:
    """Idempotently create ``user_id`` and return its role, in one transaction.

    Used by the login and create-API-key paths so the user-row upsert and the
    role read happen atomically rather than across two separate transactions.
    """
    writes.ensure_user(cur, user_id, now)
    return reads.get_user_role(cur, user_id)
