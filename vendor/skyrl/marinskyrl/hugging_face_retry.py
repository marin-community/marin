"""Shared retry policy for Hugging Face Hub operations."""

from __future__ import annotations

from collections.abc import Callable
import os
from typing import TypeVar


T = TypeVar("T")

DEFAULT_HF_MAX_RETRIES = 5
DEFAULT_HF_BACKOFF_BASE_SECONDS = 2.0
DEFAULT_HF_BACKOFF_CAP_SECONDS = 32.0

_HF_TRANSIENT_MESSAGE_FRAGMENTS = (
    "incompleteread",
    "incomplete read",
    "eof",
    "connection reset",
    "connection aborted",
    "connection broken",
    "remote end closed",
    "broken pipe",
    "server disconnected",
    "disconnected without sending",
    "peer closed connection",
)


def is_transient_hugging_face_error(error: BaseException) -> bool:
    """Return whether a Hugging Face failure is safe to retry."""
    # The Iris runtime bundle imports this module in dependency-light dataset
    # commands, so transport dependencies stay lazy until a Hub call fails.
    import httpcore  # noqa: PLC0415
    import httpx  # noqa: PLC0415
    from huggingface_hub.errors import (  # noqa: PLC0415
        EntryNotFoundError,
        GatedRepoError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )
    import requests  # noqa: PLC0415
    import urllib3  # noqa: PLC0415

    fatal_errors = (RepositoryNotFoundError, RevisionNotFoundError, GatedRepoError)
    transient_errors = (
        OSError,
        httpx.TransportError,
        httpcore.ProtocolError,
        requests.exceptions.RequestException,
        urllib3.exceptions.HTTPError,
        HfHubHTTPError,
        EntryNotFoundError,
        LocalEntryNotFoundError,
    )
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, fatal_errors):
            return False
        if isinstance(current, transient_errors):
            return True
        message = str(current).lower()
        if any(fragment in message for fragment in _HF_TRANSIENT_MESSAGE_FRAGMENTS):
            return True
        if "safetensors" in message and any(
            fragment in message
            for fragment in ("no ", "not find", "cannot find", "couldn't find", "could not find", "does not")
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def call_with_hugging_face_retry(
    call: Callable[[], T],
    *,
    operation: str,
    max_retries: int = DEFAULT_HF_MAX_RETRIES,
    backoff_base: float = DEFAULT_HF_BACKOFF_BASE_SECONDS,
    backoff_cap: float = DEFAULT_HF_BACKOFF_CAP_SECONDS,
) -> T:
    """Call a Hub operation with the shared transient-failure policy."""
    # Rigging is absent from dependency-light runtime-bundle commands that
    # import the retry interface without invoking it.
    from rigging.timing import ExponentialBackoff, retry_with_backoff  # noqa: PLC0415

    return retry_with_backoff(
        call,
        retryable=is_transient_hugging_face_error,
        max_attempts=max_retries + 1,
        backoff=ExponentialBackoff(initial=backoff_base, maximum=backoff_cap, factor=2.0, jitter=0.0),
        operation=operation,
    )


def load_hugging_face_with_retry(
    call: Callable[[], T],
    *,
    resource_id: str,
    resource_kind: str,
    max_retries: int = DEFAULT_HF_MAX_RETRIES,
    backoff_base: float = DEFAULT_HF_BACKOFF_BASE_SECONDS,
    backoff_cap: float = DEFAULT_HF_BACKOFF_CAP_SECONDS,
) -> T:
    """Load one Hub resource through the shared Hugging Face retry policy."""
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
    return call_with_hugging_face_retry(
        call,
        operation=f"load Hugging Face {resource_kind} {resource_id} on rank {rank}",
        max_retries=max_retries,
        backoff_base=backoff_base,
        backoff_cap=backoff_cap,
    )
