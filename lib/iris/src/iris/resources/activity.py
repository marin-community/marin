# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authorized resource activity queries and entries."""

from collections.abc import Mapping
from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.resources.identity import ResourceKey


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
