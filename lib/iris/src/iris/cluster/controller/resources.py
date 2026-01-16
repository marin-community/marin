# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resource parsing and comparison utilities for scheduler resource matching."""

import humanfriendly

from iris.rpc import cluster_pb2


def parse_memory_string(memory_str: str) -> int:
    """Parse memory string like '8g', '16gb', '512m' to bytes.

    Uses humanfriendly library for robust parsing. Supports various formats:
    - "8G", "8GB", "8 GB", "8 gigabytes"
    - "512M", "512MB", "512 megabytes"
    - "1024K", "1024KB", "1024 kilobytes"
    - Plain numbers treated as bytes

    Args:
        memory_str: Memory string (e.g., "8g", "16gb", "512m", "1024mb")

    Returns:
        Memory in bytes

    Raises:
        ValueError: If format is invalid
    """
    if not memory_str:
        return 0

    memory_str = memory_str.strip()
    if not memory_str or memory_str == "0":
        return 0

    try:
        return humanfriendly.parse_size(memory_str, binary=True)
    except humanfriendly.InvalidSize as e:
        raise ValueError(str(e)) from e


def get_device_type(device: cluster_pb2.DeviceConfig) -> str:
    if device.HasField("cpu"):
        return "cpu"
    elif device.HasField("gpu"):
        return "gpu"
    elif device.HasField("tpu"):
        return "tpu"
    return "cpu"  # Default to CPU if no device specified


def get_device_variant(device: cluster_pb2.DeviceConfig) -> str | None:
    if device.HasField("gpu"):
        return device.gpu.variant if device.gpu.variant else None
    elif device.HasField("tpu"):
        return device.tpu.variant if device.tpu.variant else None
    return None


def get_gpu_count(device: cluster_pb2.DeviceConfig) -> int:
    if device.HasField("gpu"):
        return device.gpu.count or 1
    return 0
