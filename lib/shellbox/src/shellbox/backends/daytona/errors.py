# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Daytona provider causes expressed through the shared task error schema."""

from daytona import DaytonaConflictError, DaytonaConnectionError
from rigging.task_errors import ErrorIdentity, RetryDisposition, exception_identity


def daytona_error_identity(error: Exception, *, operation: str) -> ErrorIdentity:
    """Classify SDK failures, including conflicts that clear during deletion."""
    identity = exception_identity(error, operation=operation)
    if (
        isinstance(error, DaytonaConnectionError) and identity.provider_status_code is None
    ) or (isinstance(error, DaytonaConflictError) and operation == "delete"):
        return identity.model_copy(update={"retry_disposition": RetryDisposition.TRANSIENT})
    return identity
