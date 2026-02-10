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

"""Container runtime abstraction layer.

Provides ContainerRuntime protocol and implementations:
- DockerRuntime: runs containers via Docker CLI
- ProcessRuntime: runs commands as subprocesses (for local/testing use)
"""

from iris.cluster.runtime.docker import DockerImageBuilder, DockerRuntime
from iris.cluster.runtime.entrypoint import build_runtime_entrypoint, runtime_entrypoint_to_bash_script
from iris.cluster.runtime.types import (
    ContainerConfig,
    ContainerHandle,
    ContainerResult,
    ContainerRuntime,
    ContainerStats,
    ContainerStatus,
    ImageBuilder,
    ImageInfo,
)

__all__ = [
    "ContainerConfig",
    "ContainerHandle",
    "ContainerResult",
    "ContainerRuntime",
    "ContainerStats",
    "ContainerStatus",
    "DockerImageBuilder",
    "DockerRuntime",
    "ImageBuilder",
    "ImageInfo",
    "build_runtime_entrypoint",
    "runtime_entrypoint_to_bash_script",
]
