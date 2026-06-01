# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ray-compatible exception types.

smallpond references ``ray.exceptions.RuntimeEnvSetupError`` (caught around
dispatch) and relies on a timeout error being raised by ``ray.get(..., timeout=0)``
on an unfinished object. We provide the minimal set with the Ray names so the
shim is drop-in.
"""

from __future__ import annotations


class RayError(Exception):
    """Base class for Ray-compatible errors."""


class GetTimeoutError(RayError, TimeoutError):
    """Raised by ``get`` when objects are not ready within the timeout.

    Subclasses ``TimeoutError`` so callers that catch either name work.
    """


class RuntimeEnvSetupError(RayError, RuntimeError):
    """Raised when a worker's runtime environment fails to set up.

    smallpond catches this specifically and retries; the shim never raises it
    in normal operation, but the type must exist for ``except`` clauses.
    """


class RayTaskError(RayError, RuntimeError):
    """Wraps an exception raised inside a remote task.

    The shim re-raises the *original* task exception from ``get`` rather than
    wrapping it, matching what smallpond's broad ``except Exception`` expects.
    This type exists for API completeness.
    """
