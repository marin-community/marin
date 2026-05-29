# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure serialization for Iris cluster config: proto -> YAML-friendly dict."""

from __future__ import annotations

from google.protobuf.json_format import MessageToDict

from iris.rpc import config_pb2

# Reverse mapping for YAML serialization: proto enum name → friendly YAML name
_CAPACITY_TYPE_REVERSE_MAP = {
    "CAPACITY_TYPE_PREEMPTIBLE": "preemptible",
    "CAPACITY_TYPE_ON_DEMAND": "on-demand",
    "CAPACITY_TYPE_RESERVED": "reserved",
}


def config_to_dict(config: config_pb2.IrisClusterConfig) -> dict:
    """Convert config to dict for YAML serialization."""
    data = MessageToDict(config, preserving_proto_field_name=True)
    scale_groups = data.get("scale_groups")
    if isinstance(scale_groups, dict):
        for sg in scale_groups.values():
            if not isinstance(sg, dict):
                continue
            resources = sg.get("resources")
            if not isinstance(resources, dict):
                continue
            normalized: dict[str, object] = {}
            if "cpu_millicores" in resources:
                normalized["cpu"] = resources["cpu_millicores"] / 1000
            if "memory_bytes" in resources:
                normalized["ram"] = resources["memory_bytes"]
            if "disk_bytes" in resources:
                normalized["disk"] = resources["disk_bytes"]
            if "device_type" in resources:
                normalized["device_type"] = resources["device_type"]
            if "device_variant" in resources:
                normalized["device_variant"] = resources["device_variant"]
            if "device_count" in resources:
                normalized["device_count"] = resources["device_count"]
            if "capacity_type" in resources:
                raw_ct = resources["capacity_type"]
                normalized["capacity_type"] = _CAPACITY_TYPE_REVERSE_MAP.get(raw_ct, raw_ct)
            sg["resources"] = normalized
    return data
