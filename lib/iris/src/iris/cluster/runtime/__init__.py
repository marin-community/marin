# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Container runtime abstraction layer.

Provides ContainerRuntime protocol and implementations:
- DockerRuntime: runs containers via Docker CLI
- ContainerdRuntime: runs containers via crictl (CRI CLI) for containerd-based hosts
- ProcessRuntime: runs commands as subprocesses (for local/testing use)
"""

from iris.cluster.runtime.containerd import ContainerdRuntime
from iris.cluster.runtime.docker import DockerImageBuilder, DockerRuntime
from iris.cluster.runtime.entrypoint import build_runtime_entrypoint, runtime_entrypoint_to_bash_script
from iris.cluster.runtime.kubernetes import KubernetesRuntime
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
    "ContainerdRuntime",
    "DockerImageBuilder",
    "DockerRuntime",
    "ImageBuilder",
    "ImageInfo",
    "KubernetesRuntime",
    "build_runtime_entrypoint",
    "runtime_entrypoint_to_bash_script",
]
