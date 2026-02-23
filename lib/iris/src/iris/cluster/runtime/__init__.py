# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Container runtime abstraction layer.

Provides ContainerRuntime protocol and implementations:
- DockerRuntime: runs containers via Docker CLI
- KubernetesRuntime: runs containers as Kubernetes Pods
- ProcessRuntime: runs commands as subprocesses (for local/testing use)
"""

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
    "DockerImageBuilder",
    "DockerRuntime",
    "ImageBuilder",
    "ImageInfo",
    "KubernetesRuntime",
    "build_runtime_entrypoint",
    "runtime_entrypoint_to_bash_script",
]
