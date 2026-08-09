# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native system-process observations."""

from dataclasses import dataclass

from rigging.provenance import Provenance


@dataclass(frozen=True, slots=True)
class ProcessInfo:
    hostname: str
    pid: int
    python_version: str
    uptime_ms: int
    memory_rss_bytes: int
    memory_vms_bytes: int
    thread_count: int
    open_fd_count: int
    memory_total_bytes: int
    cpu_count: int
    cpu_millicores: int
    provenance: Provenance
