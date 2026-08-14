# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Static configuration for Marin's MoK-like backend."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from levanter.kernels.mixture_of_kittens.source import SUPPORTED_NUM_DEVICES


_CLUSTER_SIZE = 2
_TILE_ROWS = 256
_MAX_WORKSPACE_SLOTS = 2
# GPUs per GB200 node. An expert group at or below this size can be owned by a
# single JAX process; above it the group necessarily spans processes and hosts.
EXPERT_AXIS = "expert"
_DEVICES_PER_NODE = 4


class MokLikeForwardXStorage(StrEnum):
    """Storage used for the peer-read forward activation input.

    The experimental XLA mode depends on peer mappings supplied by the pinned
    GPU allocator. It does not imply symmetric virtual addresses.

    Neither mode determines whether peers outside the calling process are
    reachable: that is a property of the runtime workspace's transport, not of
    where the forward input is read from. See ``MokLikeWorkspaceTransport``.
    """

    RUNTIME_STAGED = "runtime_staged"
    XLA_PEER_EXPERIMENTAL = "xla_peer_experimental"

    @property
    def native_ffi_code(self) -> int:
        """Return the stable integer passed through the native FFI ABI."""

        if self is MokLikeForwardXStorage.RUNTIME_STAGED:
            return 0
        if self is MokLikeForwardXStorage.XLA_PEER_EXPERIMENTAL:
            return 1
        raise AssertionError(f"unhandled forward x storage {self}")

    @property
    def reads_xla_buffers_directly(self) -> bool:
        """Whether peers read XLA-owned memory instead of the staged workspace.

        Direct reads use space-0 peer mappings from the pinned GPU allocator and
        therefore cannot leave the calling process, regardless of transport.
        """

        return self is MokLikeForwardXStorage.XLA_PEER_EXPERIMENTAL


class MokLikeBackwardPeerStorage(StrEnum):
    """Storage used for the four peer-accessed backward buffers."""

    RUNTIME_STAGED = "runtime_staged"
    XLA_PEER_EXPERIMENTAL = "xla_peer_experimental"
    XLA_PEER_INPUTS_EXPERIMENTAL = "xla_peer_inputs_experimental"

    @property
    def native_ffi_code(self) -> int:
        """Return the stable integer passed through the native FFI ABI."""

        if self is MokLikeBackwardPeerStorage.RUNTIME_STAGED:
            return 0
        if self is MokLikeBackwardPeerStorage.XLA_PEER_EXPERIMENTAL:
            return 1
        if self is MokLikeBackwardPeerStorage.XLA_PEER_INPUTS_EXPERIMENTAL:
            return 2
        raise AssertionError(f"unhandled backward peer storage {self}")

    @property
    def reads_xla_buffers_directly(self) -> bool:
        """Whether peers read XLA-owned memory instead of the staged workspace."""

        return self in (
            MokLikeBackwardPeerStorage.XLA_PEER_EXPERIMENTAL,
            MokLikeBackwardPeerStorage.XLA_PEER_INPUTS_EXPERIMENTAL,
        )


class MokLikeWorkspaceTransport(StrEnum):
    """How the runtime workspace is allocated and made peer-visible.

    This, not the storage modes, decides how far an expert group can reach.

    ``IN_PROCESS_PEER`` is the sealed v15 behaviour: the workspace is plain
    device memory and peers are reached with ``cudaDeviceEnablePeerAccess`` over
    a pointer table held in a process-local registry. It cannot address a rank
    owned by another process, so it caps the expert group at the process's local
    device count.

    ``FABRIC_SYMMETRIC`` allocates the workspace as a symmetric heap with the
    CUDA VMM APIs, exporting each mapping as a fabric handle and exchanging the
    handles through a cross-process bootstrap. Within one NVLink domain this
    reaches ranks on other hosts, which is what an EP64 group on a GB200 rack
    requires. Not implemented yet; selecting it raises.
    """

    IN_PROCESS_PEER = "in_process_peer"
    FABRIC_SYMMETRIC = "fabric_symmetric"

    @property
    def crosses_processes(self) -> bool:
        """Whether this transport can address ranks outside the local process."""

        return self is MokLikeWorkspaceTransport.FABRIC_SYMMETRIC


@dataclass(frozen=True)
class MokLikeConfig:
    """Static controls for one native MoK-like forward/backward call."""

    num_comm_sms: int = 40
    bwd_num_comm_sms: int = 28
    minibatch_size: int = 4096
    macrobatch_size: int = 32768
    schedule_capacity_factor: float = 1.1
    workspace_slots: int = 2
    forward_x_storage: MokLikeForwardXStorage = MokLikeForwardXStorage.RUNTIME_STAGED
    backward_peer_storage: MokLikeBackwardPeerStorage = MokLikeBackwardPeerStorage.RUNTIME_STAGED
    # Size of the expert-parallel group. Must match the rank count the native
    # adapter was compiled for (`MokLikeBuildConfig.num_devices`).
    num_devices: int = 4
    workspace_transport: MokLikeWorkspaceTransport = MokLikeWorkspaceTransport.IN_PROCESS_PEER

    @property
    def requires_cross_process_transport(self) -> bool:
        """Whether this expert group must span JAX processes.

        A GB200 node exposes four GPUs, so any group wider than that necessarily
        crosses processes and hosts.
        """

        return self.num_devices > _DEVICES_PER_NODE

    def __post_init__(self) -> None:
        if self.num_comm_sms < _CLUSTER_SIZE or self.num_comm_sms % _CLUSTER_SIZE != 0:
            raise ValueError("num_comm_sms must be a positive multiple of the cluster size")
        if self.bwd_num_comm_sms < _CLUSTER_SIZE or self.bwd_num_comm_sms % _CLUSTER_SIZE != 0:
            raise ValueError("bwd_num_comm_sms must be a positive multiple of the cluster size")
        if self.minibatch_size < _TILE_ROWS or self.minibatch_size % _TILE_ROWS != 0:
            raise ValueError("minibatch_size must be a positive multiple of 256")
        if self.macrobatch_size < self.minibatch_size or self.macrobatch_size % self.minibatch_size != 0:
            raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
        if not math.isfinite(self.schedule_capacity_factor) or self.schedule_capacity_factor < 1.0:
            raise ValueError("schedule_capacity_factor must be finite and at least one")
        if type(self.workspace_slots) is not int or not 1 <= self.workspace_slots <= _MAX_WORKSPACE_SLOTS:
            raise ValueError(f"workspace_slots must be an integer from 1 through {_MAX_WORKSPACE_SLOTS}")
        if not isinstance(self.forward_x_storage, MokLikeForwardXStorage):
            raise TypeError("forward_x_storage must be a MokLikeForwardXStorage")
        if not isinstance(self.backward_peer_storage, MokLikeBackwardPeerStorage):
            raise TypeError("backward_peer_storage must be a MokLikeBackwardPeerStorage")
        if self.num_devices not in SUPPORTED_NUM_DEVICES:
            supported = ", ".join(str(value) for value in SUPPORTED_NUM_DEVICES)
            raise ValueError(f"num_devices must be one of {supported}, got {self.num_devices!r}")
        if not isinstance(self.workspace_transport, MokLikeWorkspaceTransport):
            raise TypeError("workspace_transport must be a MokLikeWorkspaceTransport")
        # Catch an unreachable-peer configuration here rather than as an illegal
        # remote access on device.
        if self.requires_cross_process_transport and not self.workspace_transport.crosses_processes:
            raise ValueError(
                f"num_devices={self.num_devices} spans processes and needs "
                f"{MokLikeWorkspaceTransport.FABRIC_SYMMETRIC}; "
                f"{self.workspace_transport} only reaches process-local ranks"
            )
        # The direct-read modes bypass the workspace and use space-0 peer mappings,
        # which stay process-local whatever the transport does. What makes them
        # unusable is the transport actually crossing processes, not the group
        # outgrowing one node: a four-rank group under FABRIC_SYMMETRIC is one
        # process per GPU, so a peer's XLA buffer is an address in another process
        # and only the arena is mapped across them.
        if self.workspace_transport.crosses_processes:
            if self.forward_x_storage.reads_xla_buffers_directly:
                raise ValueError(
                    f"forward_x_storage={self.forward_x_storage} reads XLA buffers over "
                    f"process-local peer mappings and cannot serve {self.workspace_transport}; "
                    f"use {MokLikeForwardXStorage.RUNTIME_STAGED}"
                )
            if self.backward_peer_storage.reads_xla_buffers_directly:
                raise ValueError(
                    f"backward_peer_storage={self.backward_peer_storage} reads XLA buffers over "
                    f"process-local peer mappings and cannot serve {self.workspace_transport}; "
                    f"use {MokLikeBackwardPeerStorage.RUNTIME_STAGED}"
                )


class MokLikeRuntime(Protocol):
    """Runtime capability required by traced FFI wrappers."""

    def require_compatible(self, *, num_tokens: int, hidden_dim: int, top_k: int, workspace_slots: int) -> None:
        """Raise unless this live runtime owns workspaces for the requested shape."""
