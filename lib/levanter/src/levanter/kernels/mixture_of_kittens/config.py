# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Static configuration for Marin's MoK-like backend."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


_CLUSTER_SIZE = 2
_TILE_ROWS = 256
_MAX_WORKSPACE_SLOTS = 2


class MokLikeTopology(StrEnum):
    """Expert-group topology owned by one native MoK runtime."""

    LOCAL_EP4 = "local_ep4"
    NVLINK_EP64 = "nvlink_ep64"

    @property
    def expert_axis_size(self) -> int:
        """Return the required expert mesh-axis size."""

        if self is MokLikeTopology.LOCAL_EP4:
            return 4
        if self is MokLikeTopology.NVLINK_EP64:
            return 64
        raise AssertionError(f"unhandled mok_like topology {self}")


class MokLikeForwardXStorage(StrEnum):
    """Storage used for the peer-read forward activation input.

    The experimental XLA mode depends on peer mappings supplied by the pinned
    GPU allocator. It does not imply symmetric virtual addresses.
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


@dataclass(frozen=True)
class MokLikeConfig:
    """Static controls for one native MoK-like forward/backward call."""

    topology: MokLikeTopology = MokLikeTopology.LOCAL_EP4
    num_comm_sms: int = 40
    bwd_num_comm_sms: int = 28
    minibatch_size: int = 4096
    macrobatch_size: int = 32768
    schedule_capacity_factor: float = 1.1
    workspace_slots: int = 2
    forward_x_storage: MokLikeForwardXStorage = MokLikeForwardXStorage.RUNTIME_STAGED
    backward_peer_storage: MokLikeBackwardPeerStorage = MokLikeBackwardPeerStorage.RUNTIME_STAGED

    def __post_init__(self) -> None:
        if not isinstance(self.topology, MokLikeTopology):
            raise TypeError("topology must be a MokLikeTopology")
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
        if self.topology is MokLikeTopology.NVLINK_EP64:
            if self.workspace_slots != 1:
                raise ValueError("NVLink EP64 requires exactly one workspace slot")
            if self.forward_x_storage is not MokLikeForwardXStorage.RUNTIME_STAGED:
                raise ValueError("NVLink EP64 requires runtime-staged forward inputs")
            if self.backward_peer_storage is not MokLikeBackwardPeerStorage.RUNTIME_STAGED:
                raise ValueError("NVLink EP64 requires runtime-staged backward buffers")


class MokLikeRuntime(Protocol):
    """Runtime capability required by traced FFI wrappers."""

    def require_compatible(self, *, num_tokens: int, hidden_dim: int, top_k: int, workspace_slots: int) -> None:
        """Raise unless this live runtime owns workspaces for the requested shape."""
