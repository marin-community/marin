# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral identity for a directly reachable task instance."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConnection:
    address: str
    instance_uid: str
    attempt_uid: str
