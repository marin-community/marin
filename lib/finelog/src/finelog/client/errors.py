# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backwards-compatible re-export of the stats-service error types.

The canonical location is :mod:`finelog.errors`. This module exists so
``from finelog.client.errors import X`` keeps working without forcing
``finelog.client.__init__`` to be imported first (which would create a
cycle with :mod:`finelog.store.schema`).
"""

from finelog.errors import (
    InvalidNamespaceError,
    NamespaceNotFoundError,
    QueryResultTooLargeError,
    SchemaConflictError,
    SchemaValidationError,
    StatsError,
)

__all__ = [
    "InvalidNamespaceError",
    "NamespaceNotFoundError",
    "QueryResultTooLargeError",
    "SchemaConflictError",
    "SchemaValidationError",
    "StatsError",
]
