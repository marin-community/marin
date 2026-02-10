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

"""VM-layer configuration utilities.

This module re-exports configuration utilities needed by the VM abstraction layer.
The main configuration logic now lives in iris.cluster.controller.config.
"""

from iris.cluster.controller.config import (
    DEFAULT_CONFIG,
    DEFAULT_SSH_PORT,
    IrisConfig,
    ScaleGroupSpec,
    config_to_dict,
    create_local_autoscaler,
    get_ssh_config,
    load_config,
    make_local_config,
)

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_SSH_PORT",
    "IrisConfig",
    "ScaleGroupSpec",
    "config_to_dict",
    "create_local_autoscaler",
    "get_ssh_config",
    "load_config",
    "make_local_config",
]
