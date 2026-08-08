# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public stats-service error types.

Raised by :mod:`finelog.client` to translate ConnectRPC error codes into
typed exceptions callers can catch.
"""


class StatsError(Exception):
    """Base error for stats-service operations."""


class SchemaConflictError(StatsError):
    """Requested schema disagrees with the registered one about the data's shape.

    Today that means a column type change. Disagreements about derived
    acceleration state — index policies, covering projections — degrade instead:
    the server adopts, supersedes, or keeps what it already has. A new column
    declared non-nullable is adopted as nullable, and a differing key column is
    a hint the server logs and ignores.
    """


class SchemaValidationError(StatsError):
    """A schema or write batch is structurally invalid.

    Raised at register time for missing ordering key / key column type
    mismatch / unknown column type, and at write time for missing
    non-nullable column / unknown column / type mismatch / nested or union
    type / oversized batch.
    """


class InvalidNamespaceError(StatsError):
    """Namespace name fails the regex or path-containment check."""


class NamespaceNotFoundError(StatsError):
    """Named namespace is not registered."""


class QueryResultTooLargeError(StatsError):
    """Raised by Table.query() when the result row count exceeds ``max_rows``.

    Caller should add a LIMIT, aggregate further, or pass a higher cap.
    """
