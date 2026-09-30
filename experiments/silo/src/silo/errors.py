# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Provider errors, shaped so the consumer's duck-typed classifier can read them.

The capability pipeline does not import provider exception classes. It inspects
``type(error).__name__``, ``getattr(error, "status_code", None)`` and the
lowercased ``str(error)`` (``capability_pipeline/daytona_snapshot.py:52-76``), and
it is deliberately strict: a 404 whose message does not name the resource is NOT
treated as a not-found, and a "create it" branch that misreads a generic error
becomes a retry loop.

So the message text is part of the contract, not decoration. Every error here
names its resource and carries the HTTP status the classifier expects.
"""

from __future__ import annotations

from collections.abc import Mapping

# Read by builder agents through dt.py's JSON output, so it says what to do next.
NETWORK_REFUSAL = (
    "this provider only creates network-blocked sandboxes (network_block_all=True; "
    "with dt: pass --no-network). Put anything that needs the network -- apt, pip, "
    "git clone -- in the snapshot's Dockerfile RUN steps, which build with network access."
)


class SiloError(Exception):
    """Base for provider errors. Carries an HTTP status for the classifier."""

    status_code: int | None = None

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code


class SiloNotFoundError(SiloError):
    """A named resource does not exist.

    ``resource`` must be the word the caller will search for -- "snapshot" or
    "sandbox". ``resource_not_found()`` upstream requires it to appear in the
    message, so it is built in rather than left to the call site.
    """

    status_code = 404

    def __init__(self, resource: str, name: str, detail: str = "") -> None:
        self.resource = resource
        self.name = name
        message = f"{resource} {name!r} not found"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


class SiloConflictError(SiloError):
    """A concurrent create won the race; the caller should resolve theirs."""

    status_code = 409

    def __init__(self, resource: str, name: str, detail: str = "") -> None:
        self.resource = resource
        self.name = name
        message = f"{resource} {name!r} already exists"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


class SiloRateLimitError(SiloError):
    """Capacity is exhausted; retry after the advertised delay.

    ``provider_retry.py:27-70`` reads ``.headers`` for ``Retry-After`` and
    ``.error_code`` for the receipt, so both are always present.
    """

    status_code = 429

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        error_code: str = "capacity_exhausted",
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        merged = dict(headers or {})
        if retry_after_seconds is not None and "Retry-After" not in merged:
            merged["Retry-After"] = str(int(retry_after_seconds))
        self.headers = merged


class SiloRecipeError(SiloError):
    """The Dockerfile recipe uses something this provider will not guess at.

    Refusing an unrecognised instruction is deliberate. A provider that half
    understands a recipe builds an environment that differs from the one the
    task was validated against, which is worse than not building it at all.
    """

    status_code = 422
