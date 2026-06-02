# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``ray.util`` shim namespace (placement groups, scheduling strategies)."""

from fakeray._placement import PlacementGroup, placement_group, remove_placement_group

__all__ = [
    "PlacementGroup",
    "placement_group",
    "remove_placement_group",
]


def get_node_ip_address() -> str:
    """Best-effort local IP (Ray-compat helper used by some frameworks)."""
    import socket

    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"
