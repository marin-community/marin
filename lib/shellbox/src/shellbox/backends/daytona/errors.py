# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shellbox types for Daytona failures to prepare, create or delete a sandbox.

Callers classify persisted failures by exception class name, and the SDK's
names (``ConnectionError``, ``RateLimitError``, ``ConflictError``) also name
other libraries' errors, so provisioning failures leave under these types with
the SDK error as their cause.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus

from daytona import DaytonaError, DaytonaRateLimitError

from shellbox.backends.daytona.transient import is_ambiguous_daytona_error
from shellbox.machine import MachineUnavailable


class DaytonaCapacityExhausted(RuntimeError):
    """Daytona refused a sandbox or image for the account's capacity or request rate."""


class DaytonaProvisioningError(RuntimeError):
    """Daytona could not prepare, create or delete a sandbox, or an image or sandbox failed."""


@asynccontextmanager
async def provisioning_failures(action: str) -> AsyncIterator[None]:
    """Raise a provider error from ``action`` under a Shellbox type.

    A lost or timed-out request, or a server error, raises ``MachineUnavailable``, as a command's
    does; a capacity or rate refusal raises ``DaytonaCapacityExhausted``; any other failure raises
    ``DaytonaProvisioningError``.
    """
    try:
        yield
    except DaytonaRateLimitError as error:
        raise DaytonaCapacityExhausted(f"Daytona cannot {action}: {error}") from error
    except DaytonaError as error:
        if is_ambiguous_daytona_error(error) or (error.status_code or 0) >= HTTPStatus.INTERNAL_SERVER_ERROR:
            raise MachineUnavailable(f"Lost the connection to Daytona while trying to {action}: {error}") from error
        raise DaytonaProvisioningError(f"Daytona could not {action}: {error}") from error
