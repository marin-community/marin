# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Discovery helpers for services reached through Marina's VPC egress."""

from __future__ import annotations

from google.cloud import compute_v1


def resolve_internal_ip(project: str, zone: str, instance_filter: str, *, timeout: float | None = None) -> str:
    """Return the internal IP of the VM matching a GCE instance filter."""
    request = compute_v1.ListInstancesRequest(project=project, zone=zone, filter=instance_filter)
    for instance in compute_v1.InstancesClient().list(request=request, timeout=timeout):
        for interface in instance.network_interfaces:
            if interface.network_i_p:
                return interface.network_i_p
    raise RuntimeError(f"no VM with an internal IP for filter {instance_filter!r} in {zone}")
