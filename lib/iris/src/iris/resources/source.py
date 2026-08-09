# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Partial-source status and keyset-paged resource results."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from rigging.timing import Timestamp

MAX_SOURCE_ERROR_CODE: Final[int] = 64
MAX_SOURCE_ERROR_MESSAGE: Final[int] = 512
MAX_ACTIVITY_MESSAGE: Final[int] = 2_048
MAX_ACTIVITY_ATTRIBUTES: Final[int] = 32
MAX_ACTIVITY_ATTRIBUTE_KEY: Final[int] = 64
MAX_ACTIVITY_ATTRIBUTE_VALUE: Final[int] = 512
MAX_ROOT_CAUSE_HIGHLIGHTS: Final[int] = 20
MAX_ROOT_CAUSE_HIGHLIGHT: Final[int] = 512
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

    def __post_init__(self) -> None:
        self._validate_source_id()
        if len(self.error_code) > MAX_SOURCE_ERROR_CODE:
            raise ValueError(f"error_code exceeds {MAX_SOURCE_ERROR_CODE} characters")
        if len(self.error_message) > MAX_SOURCE_ERROR_MESSAGE:
            raise ValueError(f"error_message exceeds {MAX_SOURCE_ERROR_MESSAGE} characters")
        if self.state is SourceState.AVAILABLE:
            if self.observed_at is None or self.error_code or self.error_message:
                raise ValueError("available sources require observed_at and empty error fields")
            return
        if self.state is SourceState.UNAVAILABLE:
            if not self.error_code:
                raise ValueError("unavailable sources require an error_code")
            return
        if self.state is SourceState.UNSUPPORTED:
            if (
                self.freshness is not Freshness.UNKNOWN
                or self.observed_at is not None
                or self.error_code != "unsupported"
            ):
                raise ValueError(
                    "unsupported sources require unknown freshness, no observation, and error_code 'unsupported'"
                )
            return
        raise ValueError(f"unknown source state: {self.state!r}")

    def _validate_source_id(self) -> None:
        prefix, separator, identifier = self.source_id.partition(":")
        if not separator or not identifier.strip():
            raise ValueError("source_id must contain a non-empty source prefix and identifier")
        if prefix == "backend":
            if not self.backend_id or self.backend_id != identifier:
                raise ValueError("backend source_id and backend_id must name the same backend")
            return
        if prefix not in {"controller", "finelog", "federation"}:
            raise ValueError(f"unsupported source_id prefix: {prefix!r}")
        if self.backend_id:
            raise ValueError(f"{prefix} sources must not carry backend_id")


@dataclass(frozen=True, slots=True)
class Page[T]:
    items: tuple[T, ...]
    next_page_token: str | None
    source_statuses: tuple[ResourceSourceStatus, ...]
