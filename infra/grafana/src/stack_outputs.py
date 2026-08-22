# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate cross-stack values consumed by the Grafana deployment."""

from collections.abc import Mapping


def workload_client(value: object, name: str) -> dict[str, object]:
    """Select one named Loom workload client from its Pulumi stack output."""
    if not isinstance(value, list):
        raise ValueError("Loom workloadClients output must be a list")
    matches = [item for item in value if isinstance(item, dict) and item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"Loom stack must export exactly one {name!r} workload client")
    client = matches[0]
    required = ("loomUrl", "profile")
    if not all(isinstance(client.get(field), str) and client[field] for field in required):
        raise ValueError(f"Loom workload client {name!r} is missing {required}")
    profiles = client.get("profiles")
    if not isinstance(profiles, list) or not profiles or not all(isinstance(profile, str) for profile in profiles):
        raise ValueError(f"Loom workload client {name!r} is missing profiles")
    if client["profile"] not in profiles:
        raise ValueError(f"Loom workload client {name!r} default profile is not granted")
    return {**{field: client[field] for field in required}, "profiles": profiles}


def require_workload_profile(client: Mapping[str, object], profile: str) -> str:
    """Return a named profile only when federation grants it to the workload."""
    profiles = client.get("profiles")
    if not isinstance(profiles, list) or profile not in profiles:
        raise ValueError(f"Loom workload client does not grant profile {profile!r}")
    return profile
