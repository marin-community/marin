# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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

"""Controller lifecycle factory.

This module provides the factory function to create controller instances
based on configuration (GCP, manual, or local).
"""

from __future__ import annotations

from iris.cluster.controller.local import LocalController
from iris.cluster.vm.controller_vm import ControllerProtocol, GcpController, ManualController
from iris.managed_thread import ThreadContainer
from iris.rpc import config_pb2


def create_controller_vm(
    config: config_pb2.IrisClusterConfig,
    threads: ThreadContainer | None = None,
) -> ControllerProtocol:
    """Factory function to create appropriate controller VM type.

    Dispatches based on the controller.controller oneof field:
    - gcp: Creates GcpController for GCP-managed VMs
    - manual: Creates ManualController for SSH bootstrap to pre-existing hosts
    - local: Creates LocalController for in-process testing

    Args:
        config: Cluster configuration.
        threads: Optional parent ThreadContainer. Only used by LocalController
            to integrate in-process threads into the caller's hierarchy.
    """
    controller = config.controller
    which = controller.WhichOneof("controller")
    if which == "gcp":
        return GcpController(config)
    if which == "manual":
        return ManualController(config)
    if which == "local":
        return LocalController(config, threads=threads)
    raise ValueError("No controller config specified in controller")
