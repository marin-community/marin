# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a VM's internal IP from a GCE list filter.

The Iris controller and the finelog hub are reached over Direct VPC egress by internal IP; the
address is a GCE instance lookup, so it resolves while the monitored cluster's control plane is down
and picks up a rebuilt VM's new IP on the next lookup.
"""

from __future__ import annotations

from google.cloud import compute_v1


def resolve_internal_ip(project: str, zone: str, instance_filter: str, *, timeout: float | None = None) -> str:
    """Return the internal IP of the VM matching ``instance_filter`` in ``project``/``zone``.

    ``timeout`` bounds the Compute API call so a caller outside the VPC (local dev) fails fast rather
    than hanging. Raises RuntimeError if no matching instance has an internal IP.
    """
    request = compute_v1.ListInstancesRequest(project=project, zone=zone, filter=instance_filter)
    for instance in compute_v1.InstancesClient().list(request=request, timeout=timeout):
        for interface in instance.network_interfaces:
            if interface.network_i_p:
                return interface.network_i_p
    raise RuntimeError(f"no VM with an internal IP for filter {instance_filter!r} in {zone}")
