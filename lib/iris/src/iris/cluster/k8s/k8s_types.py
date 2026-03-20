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


def parse_k8s_cpu(value: str) -> int:
    """Parse Kubernetes CPU notation to millicores.

    Examples: '250m' -> 250, '1' -> 1000, '0.5' -> 500, '2500m' -> 2500
    """
    if value.endswith("m"):
        return int(value[:-1])
    return int(float(value) * 1000)


def parse_k8s_memory(value: str) -> int:
    """Parse Kubernetes memory notation to bytes.

    Examples: '512Mi' -> 536870912, '1Gi' -> 1073741824, '100Ki' -> 102400,
              '1000' -> 1000 (raw bytes)
    """
    units = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4}
    for suffix, multiplier in units.items():
        if value.endswith(suffix):
            return int(value[: -len(suffix)]) * multiplier
    si_units = {"K": 1000, "M": 1000**2, "G": 1000**3, "T": 1000**4}
    for suffix, multiplier in si_units.items():
        if value.endswith(suffix) and not value.endswith("i"):
            return int(value[: -len(suffix)]) * multiplier
    return int(value)
