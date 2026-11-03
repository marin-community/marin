"""Retry and error-classification helpers for Daytona API calls."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from typing import Any

from tenacity import RetryCallState, wait_exponential_jitter

RetryCallback = Callable[[RetryCallState], bool]
WaitCallback = Callable[[RetryCallState], float]

# Default client-side HTTP connection-pool cap for the Daytona SDK. The SDK's own
# default is 250 -- a HARD aiohttp TCPConnector limit shared across EVERY sandbox's
# API calls. At high agentic concurrency (n_concurrent >> 250) the verifier's
# upload/exec/download calls starve for a connection and blow their timeout budget
# -> mass VerifierTimeoutError (the verifier isn't slow; its HTTP calls can't get a
# socket). We bake a high default so every harbor job type (RL / eval / datagen)
# is safe with zero config. 2048 = ~2.7x headroom over a 768-concurrency run;
# bounded (not unlimited) so we self-throttle rather than flood the control plane.
_DEFAULT_DAYTONA_CONNECTION_POOL_MAXSIZE = 2048


def daytona_connection_pool_maxsize() -> int | None:
    """Connection-pool cap to pass to EVERY ``DaytonaConfig`` harbor builds.

    Single knob for the whole fleet: override with
    ``HARBOR_DAYTONA_CONNECTION_POOL_MAXSIZE`` (a positive int, or
    ``none``/``unlimited`` for no limit). Applied at all Daytona client
    construction sites so the SDK's too-low 250 default can never bite again.
    """
    raw = os.environ.get("HARBOR_DAYTONA_CONNECTION_POOL_MAXSIZE")
    if not raw:
        return _DEFAULT_DAYTONA_CONNECTION_POOL_MAXSIZE
    if raw.strip().lower() in ("none", "unlimited"):
        return None
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_DAYTONA_CONNECTION_POOL_MAXSIZE
    return value if value > 0 else None


def _get_daytona() -> Any:
    """Lazy import daytona to avoid ImportError when not installed."""
    try:
        import daytona

        return daytona
    except ImportError as e:
        raise ImportError(
            "daytona package is not installed. Install it with: pip install daytona"
        ) from e


# --- Error-message classification patterns --------------------------------
# Daytona's SDK classifies errors by TYPED exception + numeric ``status_code``
# (daytona.common.errors: DaytonaError base carries ``status_code``; subclasses
# DaytonaNotFoundError / DaytonaRateLimitError / DaytonaTimeoutError), NOT by
# message strings. These named sets are harbor's FALLBACK for SDK paths that
# surface a condition only as a generic message (e.g. quota/capacity errors that
# arrive as a plain DaytonaError). Prefer type/status_code checks; reach for a
# pattern set only when the SDK gives nothing typed. One named set per condition
# (cf. the ``_TRANSIENT_PGRST_CODES`` frozenset convention in harbor/auth/retry.py).
_TRANSIENT_MESSAGE_PATTERNS: frozenset[str] = frozenset(
    {"limit exceeded", "too many requests", "capacity", "rate limit"}
)
_SNAPSHOT_QUOTA_PATTERNS: frozenset[str] = frozenset(
    {
        "snapshot limit",
        "snapshot quota",
        "maximum number of snapshots",
        "max snapshots",
        "quota exceeded",
        "snapshot cap",
        "out of snapshots",
    }
)
_BUILD_FAILURE_PATTERNS: frozenset[str] = frozenset({"build_failed", "failed to build"})
_TRANSIENT_BUILD_FAILURE_PATTERNS = (
    "429 too many requests",
    "504 gateway timeout",
    "image processing cancelled",
)
_BUILD_FAILURE_ATTEMPTS = 4
_BUILD_FAILURE_WAIT = wait_exponential_jitter(initial=30, max=300, jitter=30)
# Exact Daytona 401 response body — used ONLY as a last-resort fallback in
# is_daytona_auth_flake when the HTTP status can't be read off the exception
# chain. The primary signal is ``status_code == 401``, not this string.
_AUTH_FLAKE_MESSAGE = "bearer token is invalid"
# Daytona's error taxonomy is version-dependent: SDK 0.183.0 collapsed 401 to a
# plain ``DaytonaError`` with ``status_code == 401``, but OLDER versions still on
# some clusters raise a typed ``DaytonaAuthenticationError`` (which may not carry
# ``status_code``). Match by class NAME so detection is robust across versions
# without importing types that don't exist in the installed SDK.
_AUTH_ERROR_TYPE_NAMES: frozenset[str] = frozenset(
    {"DaytonaAuthenticationError", "DaytonaAuthorizationError"}
)


def _exception_chain(exception: BaseException | None):
    """Yield ``exception`` then its ``__cause__``/``__context__`` chain (bounded,
    cycle-safe). Daytona's ``intercept_errors`` re-raises the underlying transport
    error (e.g. an ``httpx.HTTPStatusError``) as its implicit ``__context__``, so
    the true HTTP status often lives one hop down the chain, not on the top error.
    """
    seen: set[int] = set()
    exc = exception
    depth = 0
    while exc is not None and id(exc) not in seen and depth < 10:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__
        depth += 1


def is_daytona_auth_flake(exception: BaseException | None) -> bool:
    """True for Daytona's intermittent toolbox 401 ("Bearer token is invalid").

    Detected by HTTP STATUS, not message substrings — Daytona's API-stable signal.
    Control-plane calls carry the 401 as ``DaytonaError.status_code``; the toolbox
    fs/process calls (raw httpx under ``intercept_errors``' catch-all) lose
    ``status_code`` (None) and expose the 401 only on the chained httpx error's
    ``.response.status_code``. Walk the chain and check both shapes. The exact
    response-body string is a last-resort fallback only.
    """
    if exception is None:
        return False
    for exc in _exception_chain(exception):
        if type(exc).__name__ in _AUTH_ERROR_TYPE_NAMES:  # older Daytona taxonomy
            return True
        if getattr(exc, "status_code", None) == 401:
            return True
        response = getattr(exc, "response", None)  # httpx.HTTPStatusError et al.
        if response is not None and getattr(response, "status_code", None) == 401:
            return True
    return _AUTH_FLAKE_MESSAGE in str(exception).lower()


def is_transient_daytona_error(exception: BaseException | None) -> bool:
    """True for rate limits and capacity errors that warrant extended retries."""
    if exception is None:
        return False

    try:
        daytona = _get_daytona()
        DaytonaError = daytona.common.errors.DaytonaError
        DaytonaRateLimitError = daytona.common.errors.DaytonaRateLimitError
    except (ImportError, AttributeError):
        return False

    if isinstance(exception, DaytonaRateLimitError):
        return True

    if isinstance(exception, DaytonaError):
        msg = str(exception).lower()
        if any(pattern in msg for pattern in _TRANSIENT_MESSAGE_PATTERNS):
            return True

    return False


def is_snapshot_quota_error(exception: BaseException | None) -> bool:
    """True when a snapshot create was refused because the org snapshot cap is full.

    Daytona returns a quota/limit error (not a build failure) when the org has hit
    its HARD snapshot cap. We classify these distinctly from generic transient
    rate limits so the auto-snapshot path can fall through to a *declarative build*
    (run the trial from the Dockerfile directly) rather than retrying forever
    against a cap that won't clear, or finalizing the trial unscored.

    Matched on the error message because the SDK does not expose a dedicated
    quota-exceeded exception class across versions.
    """
    if exception is None:
        return False

    try:
        daytona = _get_daytona()
        DaytonaError = daytona.common.errors.DaytonaError
    except (ImportError, AttributeError):
        DaytonaError = None  # type: ignore[assignment]

    # Accept any exception (some SDK paths wrap quota errors in plain Exceptions);
    # the message patterns below are specific enough to avoid false positives.
    msg = str(exception).lower()
    if any(pattern in msg for pattern in _SNAPSHOT_QUOTA_PATTERNS):
        return True

    # A bare "limit exceeded"/"quota" only counts as a snapshot-quota error when
    # it is an actual DaytonaError (so we don't misread unrelated strings).
    if DaytonaError is not None and isinstance(exception, DaytonaError):
        if "quota" in msg or ("limit" in msg and "exceeded" in msg):
            return True

    return False


def is_sandbox_build_failure(exception: BaseException) -> bool:
    """True when a Daytona sandbox/snapshot build failed."""
    try:
        from harbor.environments.base import SandboxBuildFailedError

        if isinstance(exception, SandboxBuildFailedError):
            return True
    except ImportError:
        pass

    try:
        daytona = _get_daytona()
        DaytonaError = daytona.common.errors.DaytonaError
    except (ImportError, AttributeError):
        return False

    if isinstance(exception, DaytonaError):
        msg = str(exception).lower()
        return any(pattern in msg for pattern in _BUILD_FAILURE_PATTERNS)

    return False


def _is_transient_build_failure(exception: BaseException | None) -> bool:
    if exception is None or not is_sandbox_build_failure(exception):
        return False
    message = str(exception).lower()
    return any(pattern in message for pattern in _TRANSIENT_BUILD_FAILURE_PATTERNS)


def _is_non_retryable(exception: BaseException) -> bool:
    # tenacity's retry loop catches BaseException, so without this check it
    # swallows the CancelledError injected by an enclosing asyncio.wait_for /
    # asyncio.timeout and retries the call. The retried call then parks on the
    # same dead connection with no further cancellation coming (asyncio
    # delivers the deadline cancellation exactly once), so the caller's
    # timeout never fires and the trial hangs for as long as the connection
    # stays severed. Cancellation must always propagate immediately.
    if isinstance(exception, asyncio.CancelledError):
        return True
    if isinstance(exception, TimeoutError):
        return True
    # A full org snapshot cap won't clear by retrying — let it escape immediately
    # so the auto-snapshot path can fall through to a declarative build.
    if is_snapshot_quota_error(exception):
        return True
    return is_sandbox_build_failure(exception)


def daytona_retry_callbacks(
    *,
    transient_linear_step: int = 60,
) -> tuple[RetryCallback, WaitCallback]:
    """Build tenacity retry/wait callbacks for Daytona API calls.

    Args:
        transient_linear_step: Seconds added per attempt for transient errors
            (60 for create operations, 30 for snapshot GET).
    """

    def retry_callback(retry_state: RetryCallState) -> bool:
        exception = retry_state.outcome.exception() if retry_state.outcome else None
        if _is_transient_build_failure(exception):
            return retry_state.attempt_number < _BUILD_FAILURE_ATTEMPTS
        if exception is None or _is_non_retryable(exception):
            return False
        if is_transient_daytona_error(exception):
            return retry_state.attempt_number < 10
        # Daytona's intermittent toolbox 401 flake ("Bearer token is invalid"):
        # a static-key server-side flake, not expiry. Bounded (~6 attempts, the
        # exp 2->30s default wait below) so it rides out the flake but still fails
        # fast on a genuinely rotated/revoked key rather than hanging a trial for
        # the better part of an hour. (Was hardcoded per-call-site as
        # _create_exec_session_with_retry / _sdk_upload_*; centralized here.)
        if is_daytona_auth_flake(exception):
            return retry_state.attempt_number < 6
        return retry_state.attempt_number < 3

    def wait_callback(retry_state: RetryCallState) -> float:
        exception = retry_state.outcome.exception() if retry_state.outcome else None
        if _is_transient_build_failure(exception):
            return _BUILD_FAILURE_WAIT(retry_state)
        if is_transient_daytona_error(exception):
            return transient_linear_step * retry_state.attempt_number
        # Auth-flake + default: bounded exponential 2->30s.
        return min(30, max(2, 2**retry_state.attempt_number))

    return retry_callback, wait_callback


SANDBOX_RETRY, SANDBOX_WAIT = daytona_retry_callbacks(transient_linear_step=60)
SNAPSHOT_CREATE_RETRY, SNAPSHOT_CREATE_WAIT = daytona_retry_callbacks(
    transient_linear_step=60
)
SNAPSHOT_GET_RETRY, SNAPSHOT_GET_WAIT = daytona_retry_callbacks(
    transient_linear_step=30
)
