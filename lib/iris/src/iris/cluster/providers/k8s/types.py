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
    last_timestamp: datetime | None


@dataclass(frozen=True)
class ExecResult:
    """Domain type replacing subprocess.CompletedProcess in the K8sService protocol."""

    returncode: int
    stdout: str
    stderr: str


def parse_k8s_quantity(val: str) -> int:
    """Parse K8s resource quantity strings like '4000m', '16Gi', '8'.

    Handles binary suffixes (Ki, Mi, Gi, Ti), SI suffixes (K, M, G, T),
    millicore 'm' suffix, and plain integers.
    """
    if not val:
        return 0
    binary_suffixes = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "Pi": 2**50}
    si_suffixes = {"K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12, "P": 10**15}
    for suffix, mult in binary_suffixes.items():
        if val.endswith(suffix):
            return int(float(val[: -len(suffix)]) * mult)
    for suffix, mult in si_suffixes.items():
        if val.endswith(suffix) and not val.endswith("i"):
            return int(float(val[: -len(suffix)]) * mult)
    if val.endswith("m"):
        return int(val[:-1])
    return int(float(val))


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
