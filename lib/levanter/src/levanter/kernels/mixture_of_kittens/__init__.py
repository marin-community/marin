# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin's PyTorch-free MoK-like fused Grug backend.

This package is a Marin reimplementation based on pinned Mixture-of-Kittens and
ThunderKittens sources. It is not the optional adapter to the upstream MoK
Python package.
"""

from levanter.kernels.mixture_of_kittens.api import (
    MOK_CONTEXT_CHECKPOINT_NAME as MOK_CONTEXT_CHECKPOINT_NAME,
    mok_like_mlp as mok_like_mlp,
    mok_like_reference as mok_like_reference,
    validate_mok_like_inputs as validate_mok_like_inputs,
)
from levanter.kernels.mixture_of_kittens.availability import (
    MokLikePreflightStatus as MokLikePreflightStatus,
    mok_like_preflight_status as mok_like_preflight_status,
    require_mok_like_available as require_mok_like_available,
)
from levanter.kernels.mixture_of_kittens.config import (
    MokLikeBackwardPeerStorage as MokLikeBackwardPeerStorage,
    MokLikeConfig as MokLikeConfig,
    MokLikeForwardXStorage as MokLikeForwardXStorage,
    MokLikeTopology as MokLikeTopology,
)
from levanter.kernels.mixture_of_kittens.runtime import (
    MokLikeDebugCounters as MokLikeDebugCounters,
    MokLikeMemoryPoolRankTelemetry as MokLikeMemoryPoolRankTelemetry,
    MokLikeMemoryPoolTrimTelemetry as MokLikeMemoryPoolTrimTelemetry,
    MokLikeRuntimeHandle as MokLikeRuntimeHandle,
    initialize_mok_like_runtime as initialize_mok_like_runtime,
    mok_like_runtime_initialized as mok_like_runtime_initialized,
    validate_mok_like_expert_groups as validate_mok_like_expert_groups,
    validate_mok_like_mesh_topology as validate_mok_like_mesh_topology,
)
from levanter.kernels.mixture_of_kittens.source import MokLikeBuildConfig as MokLikeBuildConfig
from levanter.kernels.mixture_of_kittens.symmetric_memory import (
    MOK_LIKE_EP64_ARENA_OFFSET_FIELDS as MOK_LIKE_EP64_ARENA_OFFSET_FIELDS,
    MOK_LIKE_EP64_ARENA_SCHEMA_VERSION as MOK_LIKE_EP64_ARENA_SCHEMA_VERSION,
    MokLikeSymmetricArenaLayout as MokLikeSymmetricArenaLayout,
    MokLikeSymmetricMemoryUnavailable as MokLikeSymmetricMemoryUnavailable,
    MokLikeSymmetricNativeArguments as MokLikeSymmetricNativeArguments,
    MokLikeSymmetricWorkspace as MokLikeSymmetricWorkspace,
    MokLikeTopologyError as MokLikeTopologyError,
    initialize_mok_like_symmetric_workspace as initialize_mok_like_symmetric_workspace,
    mok_like_symmetric_arena_layout as mok_like_symmetric_arena_layout,
)
