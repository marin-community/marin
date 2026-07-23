# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared health model for the main finelog and its k8s mirrors."""

from dataclasses import dataclass
from enum import StrEnum


class FinelogRole(StrEnum):
    """A server's place in the finelog forwarding topology."""

    HUB = "hub"
    MIRROR = "mirror"


@dataclass(frozen=True)
class FinelogHealth:
    """One finelog server's query or HTTP-readiness health."""

    cluster: str
    server: str
    role: FinelogRole
    responsive: bool
    ready: int
    desired: int
    latency_ms: int | None
    error_class: str
    error: str
