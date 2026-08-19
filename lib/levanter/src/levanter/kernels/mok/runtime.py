# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle for Torch symmetric-memory workspaces used by MoK FFI calls."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any

from levanter.kernels.mok.availability import require_mok_available


@dataclass(frozen=True)
class MokBf16Config:
    """Static MoK BF16 launch configuration."""

    workspace_id: int = 0
    fwd_num_comm_sms: int = 40
    bwd_num_comm_sms: int = 28
    minibatch_size: int = 4096
    macrobatch_size: int = 131_072
    schedule_capacity_multiplier: float = 0.5
    all_gather_top_experts_chunk_bytes: int = 2048
    # LatentMoE compresses before dispatch, so the routed path runs at ``latent_size`` while the
    # fused shared experts stay at the caller's hidden width. Zero disables both projections and
    # the pre-dispatch RMSNorm, collapsing the routed width back onto the hidden width.
    latent_size: int = 0
    latent_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.workspace_id < 0:
            raise ValueError("workspace_id must be non-negative")
        if self.fwd_num_comm_sms <= 0 or self.fwd_num_comm_sms % 2:
            raise ValueError("fwd_num_comm_sms must be positive and even")
        if self.bwd_num_comm_sms <= 0 or self.bwd_num_comm_sms % 2:
            raise ValueError("bwd_num_comm_sms must be positive and even")
        if self.minibatch_size <= 0 or self.minibatch_size % 256:
            raise ValueError("minibatch_size must be positive and divisible by 256")
        if self.macrobatch_size <= 0 or self.macrobatch_size % self.minibatch_size:
            raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
        if not math.isfinite(self.schedule_capacity_multiplier) or self.schedule_capacity_multiplier <= 0:
            raise ValueError("schedule_capacity_multiplier must be positive and finite")
        if self.all_gather_top_experts_chunk_bytes <= 0 or self.all_gather_top_experts_chunk_bytes % 16:
            raise ValueError("all_gather_top_experts_chunk_bytes must be positive and 16-byte aligned")
        if self.latent_size < 0 or self.latent_size % 256:
            raise ValueError("latent_size must be non-negative and divisible by 256 (0 disables the latent path)")
        if self.latent_size and not (math.isfinite(self.latent_norm_eps) and self.latent_norm_eps > 0):
            raise ValueError("latent_norm_eps must be positive and finite when latent_size is non-zero")


@dataclass
class _RuntimeState:
    workspace: Any
    native_config: Any


_RUNTIMES: dict[int, _RuntimeState] = {}


@dataclass
class MokRuntimeHandle:
    """Idempotent owner for one registered MoK workspace."""

    workspace_id: int
    _closed: bool = False

    def close(self) -> None:
        """Close the workspace once; subsequent calls have no effect."""

        if self._closed:
            return
        close_mok_runtime(self.workspace_id)
        self._closed = True

    def __enter__(self) -> "MokRuntimeHandle":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def _native_modules() -> tuple[Any, Any, Any, Any]:
    torch = importlib.import_module("torch")
    distributed = importlib.import_module("torch.distributed")
    functional = importlib.import_module("mok.functional")
    native = importlib.import_module("mok._C")
    return torch, distributed, functional, native


def initialize_mok_runtime(
    *,
    config: MokBf16Config,
    num_local_tokens: int,
    hidden_size: int,
    topk: int,
    latent_size: int | None = None,
    process_group: Any | None = None,
    device_index: int | None = None,
) -> MokRuntimeHandle:
    """Create and register one caller-owned symmetric-memory workspace.

    ``hidden_size`` is the caller's full token width, the one the fused shared experts read.
    ``latent_size`` is the routed width LatentMoE dispatches; ``None`` (or zero) means the routed
    width equals ``hidden_size``. The symmetric dispatch and combine buffers are sized at the
    routed width, which is what makes the latent hero shape fit, so passing the latent width as
    ``hidden_size`` -- as this call used to require -- is now rejected by the native handlers.

    Torch distributed must already be initialized with one process per GPU.
    Initialization is collective over ``process_group`` and must run on every
    participating rank before tracing a function that calls :func:`mok_bf16`.
    """

    require_mok_available()
    # Fail on a missing handler before creating collective symmetric memory.
    from levanter.kernels.mok.ffi import register_ffi_targets  # noqa: PLC0415

    register_ffi_targets()
    torch, distributed, functional, native = _native_modules()
    if not distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before the MoK runtime")
    if config.workspace_id in _RUNTIMES:
        raise RuntimeError(f"MoK workspace_id={config.workspace_id} is already initialized")

    if process_group is None:
        process_group = distributed.group.WORLD
    if device_index is None:
        device_index = int(torch.cuda.current_device())
    device = torch.device("cuda", device_index)
    native_config = functional.MoKConfig(
        fwd_num_comm_sms=config.fwd_num_comm_sms,
        bwd_num_comm_sms=config.bwd_num_comm_sms,
        minibatch_size=config.minibatch_size,
        macrobatch_size=config.macrobatch_size,
        schedule_capacity_multiplier=config.schedule_capacity_multiplier,
        all_gather_top_experts_chunk_bytes=config.all_gather_top_experts_chunk_bytes,
    )
    workspace = functional.get_workspace(
        native_config,
        process_group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
        latent_size=latent_size or None,
    )
    # Workspace construction and symmetric-memory rendezvous run on Torch's current stream.
    # Complete them before XLA first consumes the registered pointers on its own stream.
    torch.cuda.synchronize(device)
    native.levanter_mok_register_workspace_v1(config.workspace_id, workspace, native_config)
    _RUNTIMES[config.workspace_id] = _RuntimeState(workspace=workspace, native_config=native_config)
    return MokRuntimeHandle(config.workspace_id)


def close_mok_runtime(workspace_id: int) -> None:
    """Collectively close a registered workspace after outstanding JAX work finishes."""

    if workspace_id not in _RUNTIMES:
        raise RuntimeError(f"MoK workspace_id={workspace_id} is not initialized")
    torch, _, functional, native = _native_modules()
    torch.cuda.synchronize()
    native.levanter_mok_close_workspace_v1(workspace_id)
    del _RUNTIMES[workspace_id]
    if not _RUNTIMES:
        functional.clear_workspace_cache()


def mok_runtime_initialized(workspace_id: int) -> bool:
    """Return whether this process registered ``workspace_id``."""

    return workspace_id in _RUNTIMES
