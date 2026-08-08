# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authorized resource activity queries and entries."""

from collections.abc import Mapping
from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.cluster.resources.identity import ResourceKey
from iris.cluster.resources.source import (
    MAX_ACTIVITY_ATTRIBUTE_KEY,
    MAX_ACTIVITY_ATTRIBUTE_VALUE,
    MAX_ACTIVITY_ATTRIBUTES,
    MAX_ACTIVITY_MESSAGE,
)


@dataclass(frozen=True, slots=True)
class ActivityQuery:
    target: ResourceKey
    attempt_uid: str | None = None
    after: Timestamp | None = None
    page_size: int = 200
    page_token: str | None = None


@dataclass(frozen=True, slots=True)
class ActivityEntry:
    entry_id: str
    occurred_at: Timestamp
    source: str
    severity: str
    kind: str
    message: str
    target: ResourceKey
    attempt_uid: str | None
    correlation_id: str | None
    attributes: Mapping[str, str]

    def __post_init__(self) -> None:
        self._validate_entry_id()
        if len(self.message) > MAX_ACTIVITY_MESSAGE:
            raise ValueError(f"activity message exceeds {MAX_ACTIVITY_MESSAGE} characters")
        if len(self.attributes) > MAX_ACTIVITY_ATTRIBUTES:
            raise ValueError(f"activity attributes exceeds {MAX_ACTIVITY_ATTRIBUTES} entries")
        for key, value in self.attributes.items():
            if len(key) > MAX_ACTIVITY_ATTRIBUTE_KEY:
                raise ValueError(f"activity attribute key exceeds {MAX_ACTIVITY_ATTRIBUTE_KEY} characters")
            if len(value) > MAX_ACTIVITY_ATTRIBUTE_VALUE:
                raise ValueError(f"activity attribute value exceeds {MAX_ACTIVITY_ATTRIBUTE_VALUE} characters")

    def _validate_entry_id(self) -> None:
        if self.entry_id.startswith("action:") and self.entry_id.removeprefix("action:").strip():
            return
        if self.entry_id.startswith("finelog:"):
            namespace, separator, sequence = self.entry_id.removeprefix("finelog:").rpartition(":")
            if separator and namespace and sequence.isdecimal():
                return
        raise ValueError("entry_id must be action:<id> or finelog:<namespace>:<sequence>")
