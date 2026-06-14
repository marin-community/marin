# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-namespace storage retention policy.

A :class:`StoragePolicy` lets a caller tighten or loosen the eviction
caps for a single namespace at register time. Fields left as ``None``
inherit the cluster-wide defaults baked into the server's compaction
config; explicit values override on a per-field basis.

The client encodes a policy onto the ``RegisterTable`` request; the
``finelog-server`` persists it and consults the effective values in its
eviction step.

``max_age_seconds`` is the only knob that introduces a behavior the
default config has no analogue for: it evicts any L>=1 BOTH segment
whose ``created_at_ms`` is older than ``now - max_age_seconds``, on
top of the existing size / count caps.
"""

from __future__ import annotations

from dataclasses import dataclass

from finelog.rpc import finelog_stats_pb2 as stats_pb2


@dataclass(frozen=True)
class StoragePolicy:
    """Per-namespace retention overrides.

    Any field left as ``None`` inherits the server-wide compaction config
    value. Explicit values are absolute (not deltas).

    Attributes:
        max_segments: Per-namespace count cap on locally-tracked segments.
        max_bytes: Per-namespace byte cap on locally-tracked segments.
        max_age_seconds: Drop any L>=1 BOTH segment whose
            ``created_at_ms`` is older than ``now - max_age_seconds``.
            No default-config analogue; ``None`` disables age-based
            eviction.
    """

    max_segments: int | None = None
    max_bytes: int | None = None
    max_age_seconds: int | None = None

    def to_proto(self) -> stats_pb2.StoragePolicy:
        """Encode a policy for the wire. ``None`` round-trips as proto3's zero."""
        return stats_pb2.StoragePolicy(
            max_segments=self.max_segments or 0,
            max_bytes=self.max_bytes or 0,
            max_age_seconds=self.max_age_seconds or 0,
        )
