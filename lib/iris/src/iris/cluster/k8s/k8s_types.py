# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared data types for the k8s cluster layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


class KubectlError(RuntimeError):
    """Error raised for kubectl command failures."""


@dataclass
class KubectlLogLine:
    """A single parsed log line from kubectl logs --timestamps."""

    timestamp: datetime
    stream: str  # "stdout" or "stderr"
    data: str


@dataclass
class KubectlLogResult:
    """Result of an incremental log fetch."""

    lines: list[KubectlLogLine]
    byte_offset: int
