# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public stats-service error types.

The schema module re-exports these so server-side imports keep working
alongside the client API.
"""


class StatsError(Exception):
    """Base error for stats-service operations."""


class SchemaConflictError(StatsError):
    """Requested schema differs from the registered one in a non-additive way.

    Non-additive: a renamed column, a type change, a new non-nullable
    column, or a changed key column.
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
