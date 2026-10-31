# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native backend identity for a directly reachable task Pod."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskPod:
    namespace: str
    pod_name: str
    pod_uid: str
    pod_ip: str
    attempt_uid: str
    host_network: bool
