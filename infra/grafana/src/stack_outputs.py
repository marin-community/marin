# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate cross-stack values consumed by the Grafana deployment."""


def workload_client(value: object, name: str) -> dict[str, str]:
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
    return {field: client[field] for field in required}
