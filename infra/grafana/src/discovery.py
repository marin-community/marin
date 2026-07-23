# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a VM's internal IP from a GCE list filter.

Both the finelog and the Iris controller sources reach their VM over Direct VPC
egress by internal IP; the address is a GCE instance lookup so it holds while the
monitored cluster's control plane is down.
"""

from google.cloud import compute_v1


class InstanceResolutionError(RuntimeError):
    """A configured VM filter did not resolve to an internal IP."""


def resolve_internal_ip(project: str, zone: str, instance_filter: str) -> str:
    """Return the internal IP of the VM matching instance_filter in project/zone.

    Raises InstanceResolutionError if no matching instance has an internal IP.
    """
    # list() flattens only project/zone; the filter rides on the request.
    request = compute_v1.ListInstancesRequest(project=project, zone=zone, filter=instance_filter)
    for instance in compute_v1.InstancesClient().list(request=request):
        for interface in instance.network_interfaces:
            if interface.network_i_p:
                return interface.network_i_p
    raise InstanceResolutionError(f"no VM with an internal IP for filter {instance_filter!r} in {zone}")
