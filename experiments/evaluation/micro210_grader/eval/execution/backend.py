"""Execute normalized model requests against local or HTTP model backends."""

from __future__ import annotations

from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponseError
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
def retryable_transport(error: Exception) -> bool:
    if isinstance(error, ClientResponseError):
        return error.status in RETRYABLE_STATUSES
    return isinstance(error, (ClientConnectionError, ClientPayloadError, ConnectionError, TimeoutError))


def transport_retry_delay(error: Exception) -> float:
    """Respect a server's Retry-After response within the operation deadline."""
    if not isinstance(error, ClientResponseError) or error.headers is None:
        return 0.0
    value = error.headers.get("Retry-After")
    if value is None:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
            return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return 0.0
