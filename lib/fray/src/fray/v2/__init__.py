# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fray v2: minimal job and actor scheduling interface."""

from fray.v2.actor import ActorContext, ActorFuture, ActorGroup, ActorHandle, ActorMethod, current_actor
from fray.v2.client import Client, JobFailed, JobHandle, current_client, set_current_client, wait_all
from fray.v2.local_backend import LocalActorHandle, LocalActorMethod, LocalClient, LocalJobHandle
from fray.v2.types import (
    BinaryEntrypoint,
    CallableEntrypoint,
    CpuConfig,
    DeviceConfig,
    DeviceKind,
    Entrypoint,
    EnvironmentConfig,
    GpuConfig,
    GpuType,
    JobRequest,
    JobStatus,
    ResourceConfig,
    TpuConfig,
    TpuTopologyInfo,
    TpuType,
    create_environment,
    get_tpu_topology,
)

__all__ = [
    "ActorContext",
    "ActorFuture",
    "ActorGroup",
    "ActorHandle",
    "ActorMethod",
    "BinaryEntrypoint",
    "CallableEntrypoint",
    "Client",
    "CpuConfig",
    "DeviceConfig",
    "DeviceKind",
    "Entrypoint",
    "EnvironmentConfig",
    "GpuConfig",
    "GpuType",
    "JobFailed",
    "JobHandle",
    "JobRequest",
    "JobStatus",
    "LocalActorHandle",
    "LocalActorMethod",
    "LocalClient",
    "LocalJobHandle",
    "ResourceConfig",
    "TpuConfig",
    "TpuTopologyInfo",
    "TpuType",
    "create_environment",
    "current_actor",
    "current_client",
    "get_tpu_topology",
    "set_current_client",
    "wait_all",
]
