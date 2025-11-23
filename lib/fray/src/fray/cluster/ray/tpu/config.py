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

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""TPU configuration database and utility functions for Ray TPU orchestration."""

from dataclasses import dataclass


# TPU configurations are complicated. The number of chips per host is
# always the same for a particular generation, but the number of VMs per host
# can vary based on the pod size.
#
# Even more confusingly, Google sometimes refers to TPU cores
# as chips and vice-versa: v4 and v5p topologies refer to "core", but
# v5e and v6e topologies refer to "chips". It's doubly confusing as some
# topologies split 2 VMs per host, while others do not. We just write them
# all down here.
@dataclass(frozen=True)
class TPUConfig:
    name: str
    chip_count: int
    host_count: int
    vm_count: int
    chips_per_vm: int


TPU_CONFIGS: list[TPUConfig] = [
    # https://cloud.google.com/tpu/docs/v4
    TPUConfig("v4-8", 4, 1, 1, 4),
    TPUConfig("v4-16", 8, 2, 2, 4),
    TPUConfig("v4-32", 16, 4, 4, 4),
    TPUConfig("v4-64", 32, 8, 8, 4),
    TPUConfig("v4-128", 64, 16, 16, 4),
    TPUConfig("v4-256", 128, 32, 32, 4),
    TPUConfig("v4-512", 256, 64, 64, 4),
    TPUConfig("v4-1024", 512, 128, 128, 4),
    TPUConfig("v4-2048", 1024, 256, 256, 4),
    TPUConfig("v4-4096", 2048, 512, 512, 4),
    # https://cloud.google.com/tpu/docs/v5e
    TPUConfig("v5litepod-1", 1, 1, 1, 1),
    TPUConfig("v5litepod-2", 2, 1, 1, 2),
    TPUConfig("v5litepod-4", 4, 1, 1, 4),
    TPUConfig("v5litepod-8", 8, 1, 1, 8),
    TPUConfig("v5litepod-16", 16, 2, 4, 4),
    TPUConfig("v5litepod-32", 32, 4, 8, 4),
    TPUConfig("v5litepod-64", 64, 8, 16, 4),
    TPUConfig("v5litepod-128", 128, 16, 32, 4),
    TPUConfig("v5litepod-256", 256, 32, 64, 4),
    # https://cloud.google.com/tpu/docs/v5p
    TPUConfig("v5p-8", 4, 1, 1, 4),
    TPUConfig("v5p-16", 8, 2, 2, 4),
    TPUConfig("v5p-32", 16, 4, 4, 4),
    TPUConfig("v5p-64", 32, 8, 8, 4),
    TPUConfig("v5p-128", 64, 16, 16, 4),
    TPUConfig("v5p-256", 128, 32, 32, 4),
    TPUConfig("v5p-512", 256, 64, 64, 4),
    TPUConfig("v5p-1024", 512, 128, 128, 4),
    TPUConfig("v5p-2048", 1024, 256, 256, 4),
    TPUConfig("v5p-4096", 2048, 512, 512, 4),
    TPUConfig("v5p-8192", 4096, 1024, 1024, 4),
    TPUConfig("v5p-12288", 6144, 1536, 1536, 4),
    # https://cloud.google.com/tpu/docs/v6e
    TPUConfig("v6e-1", 1, 1, 1, 1),
    TPUConfig("v6e-4", 4, 1, 1, 4),
    TPUConfig("v6e-8", 8, 1, 1, 8),
    TPUConfig("v6e-16", 16, 4, 4, 4),
    TPUConfig("v6e-32", 32, 8, 8, 4),
    TPUConfig("v6e-64", 64, 16, 16, 4),
    TPUConfig("v6e-128", 128, 32, 32, 4),
    TPUConfig("v6e-256", 256, 64, 64, 4),
]


def get_tpu_config(tpu_type: str) -> TPUConfig:
    """Get TPU configuration by type name."""
    for config in TPU_CONFIGS:
        if config.name == tpu_type:
            return config
    raise ValueError(f"Unknown TPU type: {tpu_type}")
