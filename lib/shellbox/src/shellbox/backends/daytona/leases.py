# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host-owned snapshot admission, outside sandbox authority."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SnapshotLease:
    name: str
    token: str
