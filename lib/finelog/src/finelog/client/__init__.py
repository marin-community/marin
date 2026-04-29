# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finelog client APIs.

:class:`LogClient` is the single user-facing entry point; it covers both the
legacy log surface (``write_batch`` / ``query``) and the stats surface
(``get_table`` / ``drop_table``). :class:`RemoteLogHandler` plugs Python's
``logging`` into a :class:`LogClient`.

The error types live in :mod:`finelog.client.errors` and are re-exported
here so callers can ``from finelog.client import SchemaConflictError`` etc.
"""

from finelog.client.errors import (
    InvalidNamespaceError,
    NamespaceNotFoundError,
    QueryResultTooLargeError,
    SchemaConflictError,
    SchemaValidationError,
    StatsError,
)
from finelog.client.log_client import LogClient, Table, schema_from_dataclass
from finelog.client.remote_log_handler import RemoteLogHandler

__all__ = [
    "InvalidNamespaceError",
    "LogClient",
    "NamespaceNotFoundError",
    "QueryResultTooLargeError",
    "RemoteLogHandler",
    "SchemaConflictError",
    "SchemaValidationError",
    "StatsError",
    "Table",
    "schema_from_dataclass",
]
