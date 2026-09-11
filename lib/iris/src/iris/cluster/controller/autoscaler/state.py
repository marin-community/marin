# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistable autoscaler state.

The autoscaler keeps its slice/group tracking entirely in memory. After each
capacity call the controller mirrors that in-memory state into the ``slices`` /
``scaling_groups`` DB tables so a restarted controller can recover. These
dataclasses are the plain-data hand-off: the autoscaler produces an
:class:`AutoscalerState`, the controller persists it. No DB types appear here,
so :mod:`iris.cluster.controller.backend` can carry these in its result types
without importing the autoscaler runtime (which would cycle through
``controller.db``).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SlicePersist:
    """One row to mirror into the ``slices`` table."""

    slice_id: str
    scale_group: str
    lifecycle: str
    worker_ids: list[str]
    created_at_ms: int
    error_message: str


@dataclass(frozen=True)
class GroupPersist:
    """One row to mirror into the ``scaling_groups`` table."""

    name: str
    last_scale_up_ms: int
    last_scale_down_ms: int


@dataclass(frozen=True)
class AutoscalerState:
    """Snapshot of all autoscaler-tracked slices and groups.

    Authoritative copy lives in the in-memory ``ScalingGroup``s; the controller
    syncs the DB to match this after every capacity call. Empty for backends
    that manage their own capacity (k8s).
    """

    slices: list[SlicePersist] = field(default_factory=list)
    groups: list[GroupPersist] = field(default_factory=list)
