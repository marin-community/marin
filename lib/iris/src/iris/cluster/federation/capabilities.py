# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What a controller advertises to peers that may delegate work to it.

A controller reports availability markers — ``available:<device>`` strings like
``available:H100`` — for the device types its backends can schedule. Federating
peers read these over the capability heartbeat so they route only to a peer that
can actually run the work, without the parent hardcoding the peer's hardware in
its own config.
"""

from collections.abc import Mapping
from typing import Protocol

from iris.cluster.types import WellKnownAttribute

# Attribute keys that name a schedulable device (as opposed to placement keys like
# region/zone/preemptible). Each advertised value becomes an ``available:`` marker.
_DEVICE_ATTRIBUTE_KEYS = (
    WellKnownAttribute.DEVICE_TYPE,
    WellKnownAttribute.DEVICE_VARIANT,
    WellKnownAttribute.GPU_VARIANT,
)

# Marker for a backend that advertises no device attributes (a CPU catch-all).
_DEFAULT_DEVICE = "cpu"

CAPABILITY_MARKER_PREFIX = "available:"


class CapabilityBackend(Protocol):
    """The slice of a backend the capability report reads."""

    def advertised_attributes(self) -> dict[str, set[str]]: ...


def cluster_capability_markers(backends: Mapping[str, CapabilityBackend]) -> list[str]:
    """Availability markers for every device type ``backends`` can schedule.

    Returns a sorted, de-duplicated list of ``available:<device>`` markers — one
    per distinct device type/variant advertised across all backends. A backend
    that advertises no device attributes contributes ``available:cpu`` (it is a
    CPU catch-all).
    """
    markers: set[str] = set()
    for backend in backends.values():
        advertised = backend.advertised_attributes()
        devices: set[str] = set()
        for key in _DEVICE_ATTRIBUTE_KEYS:
            devices |= advertised.get(key, set())
        if not devices:
            devices = {_DEFAULT_DEVICE}
        markers.update(f"{CAPABILITY_MARKER_PREFIX}{device}" for device in devices)
    return sorted(markers)
