# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Partial-source status and keyset-paged resource results."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from rigging.timing import Timestamp

MAX_SOURCE_ERROR_MESSAGE: Final[int] = 512
MAX_PROVIDER_SNAPSHOT_ITEMS: Final[int] = 50_000


class SourceState(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"


class Freshness(StrEnum):
    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ResourceSourceStatus:
    source_id: str
    backend_id: str
    state: SourceState
    freshness: Freshness
    observed_at: Timestamp | None
    error_code: str
    error_message: str


@dataclass(frozen=True, slots=True)
class Page[T]:
    items: tuple[T, ...]
    next_page_token: str | None
    source_statuses: tuple[ResourceSourceStatus, ...]
