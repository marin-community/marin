# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
#
# pyrefly: ignore-errors

"""Benchmark source-push MoE MLP forward and forward+backward paths."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from itertools import product
from statistics import median
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import mosaic_gpu as mgpu
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

import levanter.grug._moe.source_push_mlp as source_push_mlp
from levanter.grug._moe.source_push_backward_dy_route import (
    SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_SOURCE_PUSH_JAX,
    SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_SOURCE_PUSH_PALLAS_MGPU,
    _source_push_backward_dy_to_expert_major,
    _source_push_backward_dy_to_expert_major_from_plan_source_push_jax,
    _source_push_backward_dy_to_expert_major_source_push_pallas_call,
    _source_push_backward_dy_to_h_rows,
)
from levanter.grug._moe.source_push_backward_dx13 import (
    SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU,
    SourcePushDx13PallasBlockSizes,
    SourcePushDx13SourceCompactOutput,
    _dx13_max_source_group_rows,
    _source_push_dx13_source_grouped_sharded_mgpu_kernel,
    source_push_dx13_pallas_resolved_block_sizes,
    source_push_dx13_compact_assignment_slots_from_fields,
    source_push_dx13_contrib_buffer_from_expert_reference,
    source_push_dx13_expert_major_store_zero_pallas_mgpu,
    source_push_dx13_push_compact,
    source_push_dx13_push_compact_contrib,
    source_push_dx13_push_compact_xla,
    source_push_dx13_push_contrib,
    source_push_dx13_push_contrib_block_contiguous_pallas_mgpu,
    source_push_dx13_push_route_buffer,
    source_push_dx13_source_route_buffer_reference,
    source_push_dx13_source_compact_combine_reference,
    source_push_dx13_source_compact_to_route_buffer_reference,
)
from levanter.grug._moe.source_push_backward_return import (
    source_push_backward_return,
    source_push_backward_return_flat,
    source_push_backward_return_flat_route_indices_jax,
    source_push_backward_return_route_indices_jax,
)
from levanter.grug._moe.source_push_backward_w13 import (
    SOURCE_PUSH_X_TO_W13_ROWS_IMPLEMENTATION_PALLAS_MGPU,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_COMPACT_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_COMPACT_DX_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_EXACT_FLAT_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_GATE_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_LINEAR_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_PERSISTENT_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_SPLIT_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_UP_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_PREFILLED_X_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_SOURCE_PADDED_PARTIALS_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_SOURCE_GATHER_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_XLA_COMPACT_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_XLA_LOCAL_SWIGLU_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_XLA_SOURCE_PADDED_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_IMPLEMENTATIONS,
    SOURCE_PUSH_W13_BACKWARD_EXPERIMENT_COMPACT_DX_SOURCE_GATHER_DW13,
    SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_PALLAS_MGPU_COMPACT,
    SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_TILED,
    MIN_MOSAIC_INT32_TRANSFER_ELEMENTS,
    SourcePushXToW13RowsPallasBlockSizes,
    SourcePushW13CompactBackwardOutput,
    SourcePushW13BackwardTiledBlockSizes,
    _source_push_w13_backward_expert_blocks_compact_dx_source_gather_dw13,
    _source_push_w13_backward_expert_blocks_dw13_only_pallas_mgpu,
    _source_push_w13_backward_expert_blocks_dw13_only_exact_flat_pallas_mgpu,
    _source_push_w13_backward_expert_blocks_dx_only_pallas_mgpu,
    _source_push_w13_backward_expert_blocks_local_swiglu_gate_dw13_only_pallas_mgpu,
    _source_push_w13_backward_expert_blocks_local_swiglu_dw13_only_pallas_mgpu,
    _source_push_w13_backward_expert_blocks_local_swiglu_persistent_dw13_only_pallas_mgpu,
    _source_push_w13_backward_expert_blocks_local_swiglu_split_dw13_only_pallas_mgpu,
    _source_push_w13_backward_expert_blocks_local_linear_dw13_only_pallas_mgpu,
    _source_push_w13_backward_expert_blocks_local_swiglu_up_dw13_only_pallas_mgpu,
    _source_push_w13_backward_expert_blocks_pallas_mgpu,
    _source_push_w13_backward_expert_blocks_prefilled_x_dw13_only_pallas_mgpu,
    _source_push_w13_dw13_source_padded_partials_pallas_mgpu,
    source_push_w13_dw13_default_block_sizes,
    source_push_x_to_w13_rows,
    source_push_w13_backward_expert_blocks_dw13_only_xla,
    source_push_w13_backward_expert_blocks_local_swiglu_dw13_only_xla,
    source_push_w13_backward,
    source_push_w13_backward_diagnostic_component,
    source_push_w13_backward_expert_blocks_source_gather_dw13_only,
    source_push_w13_backward_expert_blocks_source_padded_dw13_only_xla,
    source_push_w13_backward_is_diagnostic_only,
    source_push_w13_backward_uses_local_dw13_default_block_sizes,
    source_push_w13_backward_expert_blocks_tiled_reference,
)
from levanter.grug._moe.source_push_backward_w2 import (
    MIN_SOURCE_PUSH_W2_MATMUL_ROW_BLOCK,
    _SourcePushW2BackwardOutput,
    _SourcePushW2MatmulBackwardOutput,
    _dst_indices,
    _expert_flat_rows,
    _gather_flat_rows,
    _gather_flat_rows_by_expert_slice,
    _pad_w2_matmul_rows_for_pallas,
    _source_push_w2_d_weighted_activation_pallas_call,
    _source_push_w2_activation_and_weighted_activation_reference,
    _source_push_w2_backward_expert_blocks,
    _source_push_w2_backward_from_flat_h,
    _source_push_w2_dw2_pallas_call,
    _source_push_w2_matmul_backward,
    _source_push_w2_matmul_backward_inferred_block_sizes,
    _source_push_w2_swiglu_backward,
    _source_push_w2_valid_blocks_sharded,
)
from levanter.grug._moe.source_push_forward import (
    FORWARD_EXECUTION_STAGED_HOST_SYNC,
    FORWARD_STAGE_COMBINE,
    FORWARD_STAGE_TOTAL,
    FORWARD_STAGE_W13,
    FORWARD_STAGE_W2_RETURN,
    FORWARD_STAGES,
    SourcePushForwardDeviceInputs,
    _call_source_push_w13_h_expert_major_device_inputs,
    _shard_source_push_forward_inputs,
    _sharded_source_combine_kernel,
    _time_staged_source_push_forward,
    device_source_push_forward_inputs_from_plan,
    make_source_push_forward_plan_inputs,
    make_source_push_forward_source_plan_raw_inputs,
    source_push_forward_with_h_from_plan,
)
from levanter.grug._moe.source_push_inbox import (
    AXIS,
    BYTES_PER_BF16,
    DIAGNOSTIC_VARIANT_COMPUTE_ONLY_LOCAL,
    DIAGNOSTIC_VARIANT_FULL,
    DIAGNOSTIC_VARIANT_STORE_ZERO,
    PushInboxConfig,
    _block_until_ready,
    _compact_h_expert_capacity_from_metadata,
    _sharded_w13_h_compact_kernel,
    _sharded_raw_token_w13_h_kernel,
    _sharded_raw_token_w13_h_compact_kernel,
)
from levanter.grug._moe.source_push_inbox_profiles import (
    SOURCE_PUSH_PROFILE_STABLE_216,
    SOURCE_PUSH_PROFILES,
    source_push_profile_defaults,
)
from levanter.grug._moe.source_push_mlp import (
    SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU,
    SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
    source_push_mlp_route_table_from_plan,
    source_push_moe_mlp_from_plan,
)
from levanter.grug._moe.source_push_plan import (
    SOURCE_PUSH_MESH_AXIS,
    _source_push_out_sharding,
    pack_source_push_tokens_jax,
    source_push_h_row_route_weights_jax,
)
from levanter.grug._moe.source_push_token_pack import source_push_pack_tokens_pallas_mgpu
from levanter.grug._moe.source_push_w2_return import (
    _sharded_w2_from_compact_h_return_direct_to_source_kernel,
    _sharded_w2_from_h_return_direct_to_source_kernel,
)
from levanter.grug.grug_moe import moe_mlp
from levanter.utils.activation import ActivationFunctionEnum

KERNEL_NAME = "source_push_mlp_fwd_bwd"
SOURCE_PUSH_W13_STABLE_BASELINE_TFLOPS_PER_RANK = 216.949
MODE_FORWARD = "forward"
MODE_FORWARD_BACKWARD = "forward_backward"
MODE_FORWARD_BACKWARD_REDUCED = "forward_backward_reduced"
MODE_FORWARD_BACKWARD_DX_CHECKSUM = "forward_backward_dx_checksum"
MODE_FORWARD_BACKWARD_DROUTE_CHECKSUM = "forward_backward_droute_checksum"
MODE_FORWARD_BACKWARD_DW13_CHECKSUM = "forward_backward_dw13_checksum"
MODE_FORWARD_BACKWARD_DW2_CHECKSUM = "forward_backward_dw2_checksum"
MODE_FORWARD_DECOMPOSED = "forward_decomposed"
MODE_FORWARD_DECOMPOSED_RAW_TOKENS = "forward_decomposed_raw_tokens"
MODE_FORWARD_W13_DIRECT_COMPACT = "forward_w13_direct_compact"
MODE_FORWARD_W13_DIRECT_COMPACT_STORE_ZERO = "forward_w13_direct_compact_store_zero"
MODE_FORWARD_W13_DIRECT_COMPACT_COMPUTE_ONLY_LOCAL = "forward_w13_direct_compact_compute_only_local"
MODE_FORWARD_COMPACT_H_DECOMPOSED = "forward_compact_h_decomposed"
MODE_FORWARD_COMPACT_H_DECOMPOSED_WITH_PREP = "forward_compact_h_decomposed_with_prep"
MODE_FORWARD_COMPACT_H_DECOMPOSED_WITH_PALLAS_PACK = "forward_compact_h_decomposed_with_pallas_pack"
MODE_FORWARD_COMPACT_H_RAW_TOKENS_DECOMPOSED = "forward_compact_h_raw_tokens_decomposed"
MODE_FORWARD_PACK_TOTAL = "forward_pack_total"
MODE_FORWARD_PACK_TOKEN_PACK = "forward_pack_token_pack"
MODE_FORWARD_PACK_TOKEN_PACK_PALLAS = "forward_pack_token_pack_pallas"
MODE_FORWARD_PACK_H_ROUTE_WEIGHTS = "forward_pack_h_route_weights"
MODE_FORWARD_PACK_STATIC_SHARD = "forward_pack_static_shard"
MODE_BACKWARD_DECOMPOSED = "backward_decomposed"
MODE_BACKWARD_STAGED_FLAT = "backward_staged_flat"
MODE_BACKWARD_STAGED_BLOCKS = "backward_staged_blocks"
MODE_BACKWARD_DY_ROUTE_ONLY = "backward_dy_route_only"
MODE_BACKWARD_W2_ONLY = "backward_w2_only"
MODE_BACKWARD_W13_ONLY = "backward_w13_only"
MODE_BACKWARD_DX13_ONLY = "backward_dx13_only"
MODE_BACKWARD_DX13_STORE_ZERO_ONLY = "backward_dx13_store_zero_only"
MODE_BACKWARD_DX13_ROUTE_BUFFER_ONLY = "backward_dx13_route_buffer_only"
MODE_BACKWARD_DX13_PUSH_CONTRIB_ONLY = "backward_dx13_push_contrib_only"
MODE_BACKWARD_DX13_SOURCE_COMPACT_ONLY = "backward_dx13_source_compact_only"
MODE_BACKWARD_DX13_SOURCE_COMPACT_BLOCK_ONLY = "backward_dx13_source_compact_block_only"
MODE_BACKWARD_DX13_SOURCE_COMPACT_COMBINE_ONLY = "backward_dx13_source_compact_combine_only"
MODE_BACKWARD_DX13_XLA_SOURCE_COMPACT_DIRECT_ONLY = "backward_dx13_xla_source_compact_direct_only"
MODE_BACKWARD_DX13_XLA_ROUTE_BUFFER_DIRECT_ONLY = "backward_dx13_xla_route_buffer_direct_only"
MODE_BACKWARD_DX13_SOURCE_GROUPED_ONLY = "backward_dx13_source_grouped_only"
MODE_BACKWARD_RETURN_ONLY = "backward_return_only"
MODE_BACKWARD_RETURN_COMPONENTS_ONLY = "backward_return_components_only"
FORWARD_DECOMPOSED_STAGE_PACK_INPUTS = "pack_inputs"
FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_TOKEN_PACK = "pack_inputs_token_pack"
FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_H_ROUTE_WEIGHTS = "pack_inputs_h_route_weights"
FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_COMPACT_H_ROUTE_WEIGHTS = "pack_inputs_compact_h_route_weights"
FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_STATIC_SHARD = "pack_inputs_static_shard"
FORWARD_DECOMPOSED_STAGE_PREPARE_INPUTS = "prepare_inputs"
FORWARD_DECOMPOSED_STAGE_PREPACKED_INPUTS = "prepacked_inputs"
BACKWARD_STAGE_TOTAL = "backward_total"
BACKWARD_STAGE_FORWARD_H = "forward_h"
BACKWARD_STAGE_H_WEIGHT_GATHER = "h_weight_gather"
BACKWARD_STAGE_DY_ROUTE = "dy_route"
BACKWARD_STAGE_ACTIVATION = "activation"
BACKWARD_STAGE_W2 = "w2_backward"
BACKWARD_STAGE_W2_GATHER = "w2_h_weight_gather"
BACKWARD_STAGE_W2_ACTIVATION = "w2_activation_weight"
BACKWARD_STAGE_W2_MATMUL = "w2_matmul"
BACKWARD_STAGE_W2_D_WEIGHTED_ACTIVATION = "w2_d_weighted_activation"
BACKWARD_STAGE_W2_DW2 = "w2_dw2"
BACKWARD_STAGE_W2_SWIGLU = "w2_swiglu"
BACKWARD_STAGE_W2_SCATTER = "w2_scatter"
BACKWARD_STAGE_SWIGLU = "swiglu_backward"
BACKWARD_STAGE_X_REMAT = "x_rematerialization"
BACKWARD_STAGE_W13 = "w13_backward"
BACKWARD_STAGE_DX13_PUSH = "dx13_push_compact"
BACKWARD_STAGE_DX13_SOURCE_GROUPED = "dx13_source_grouped"
BACKWARD_STAGE_DX_COMBINE = "dx_return_combine"
BACKWARD_IMPLEMENTATION_DEFAULT = "default"
BACKWARD_STOP_AFTER_NONE = "none"
BACKWARD_STOP_AFTER_STAGES = (
    BACKWARD_STOP_AFTER_NONE,
    BACKWARD_STAGE_DY_ROUTE,
    BACKWARD_STAGE_W2,
    BACKWARD_STAGE_W13,
    BACKWARD_STAGE_DX_COMBINE,
)
BACKWARD_DY_ROUTE_IMPLEMENTATIONS = (
    BACKWARD_IMPLEMENTATION_DEFAULT,
    "reference",
    "pallas_mgpu",
    "source_push_pallas_mgpu",
    SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_SOURCE_PUSH_JAX,
)
BACKWARD_W2_IMPLEMENTATIONS = (
    BACKWARD_IMPLEMENTATION_DEFAULT,
    "reference",
    "reference_matmul_pallas_mgpu_swiglu",
    "pallas_mgpu_matmul_reference_swiglu",
    "pallas_mgpu_fused",
)
BACKWARD_W13_IMPLEMENTATIONS = (
    BACKWARD_IMPLEMENTATION_DEFAULT,
    "reference",
    "tiled",
    "pallas_mgpu",
    SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_PALLAS_MGPU_COMPACT,
    SOURCE_PUSH_W13_BACKWARD_EXPERIMENT_COMPACT_DX_SOURCE_GATHER_DW13,
    *SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_IMPLEMENTATIONS,
)
BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_LOCAL_SWIGLU_DX13_DW13 = "pallas_mgpu_local_swiglu_dx13_dw13"
BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_DX13_XLA_LOCAL_SWIGLU_DW13 = "pallas_mgpu_dx13_xla_local_swiglu_dw13"
BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_DX13_SPLIT_LOCAL_SWIGLU_DW13 = "pallas_mgpu_dx13_split_local_swiglu_dw13"
BACKWARD_W13_IMPLEMENTATION_XLA_DX13_XLA_LOCAL_SWIGLU_DW13 = "xla_dx13_xla_local_swiglu_dw13"
BACKWARD_W13_IMPLEMENTATION_XLA_DX13_ROUTE_BUFFER_XLA_LOCAL_SWIGLU_DW13 = "xla_dx13_route_buffer_xla_local_swiglu_dw13"
BACKWARD_W13_IMPLEMENTATION_XLA_DX13_SOURCE_GATHER_DW13 = "xla_dx13_source_gather_dw13"
BACKWARD_W13_IMPLEMENTATION_XLA_DX13_PALLAS_X_REMAT_XLA_LOCAL_SWIGLU_DW13 = (
    "xla_dx13_pallas_x_remat_xla_local_swiglu_dw13"
)
BACKWARD_W13_IMPLEMENTATIONS = (
    *BACKWARD_W13_IMPLEMENTATIONS,
    BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_LOCAL_SWIGLU_DX13_DW13,
    BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_DX13_XLA_LOCAL_SWIGLU_DW13,
    BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_DX13_SPLIT_LOCAL_SWIGLU_DW13,
    BACKWARD_W13_IMPLEMENTATION_XLA_DX13_XLA_LOCAL_SWIGLU_DW13,
    BACKWARD_W13_IMPLEMENTATION_XLA_DX13_ROUTE_BUFFER_XLA_LOCAL_SWIGLU_DW13,
    BACKWARD_W13_IMPLEMENTATION_XLA_DX13_SOURCE_GATHER_DW13,
    BACKWARD_W13_IMPLEMENTATION_XLA_DX13_PALLAS_X_REMAT_XLA_LOCAL_SWIGLU_DW13,
)
BACKWARD_DX13_IMPLEMENTATION_XLA_EXPERT_MAJOR = "xla_expert_major"
BACKWARD_DX13_IMPLEMENTATIONS = (
    SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU,
    BACKWARD_DX13_IMPLEMENTATION_XLA_EXPERT_MAJOR,
)
BACKWARD_W13_LOWERING_SEMANTICS = ("auto", "lane", "warpgroup")
BACKWARD_RETURN_IMPLEMENTATIONS = (
    BACKWARD_IMPLEMENTATION_DEFAULT,
    "jax",
    "pallas_mgpu",
)
BACKWARD_STAGES = (
    BACKWARD_STAGE_H_WEIGHT_GATHER,
    BACKWARD_STAGE_DY_ROUTE,
    BACKWARD_STAGE_ACTIVATION,
    BACKWARD_STAGE_W2,
    BACKWARD_STAGE_SWIGLU,
    BACKWARD_STAGE_X_REMAT,
    BACKWARD_STAGE_W13,
    BACKWARD_STAGE_DX_COMBINE,
)
BACKWARD_W2_SPLIT_STAGES = (
    BACKWARD_STAGE_W2_GATHER,
    BACKWARD_STAGE_W2_ACTIVATION,
    BACKWARD_STAGE_W2_MATMUL,
    BACKWARD_STAGE_W2_D_WEIGHTED_ACTIVATION,
    BACKWARD_STAGE_W2_DW2,
    BACKWARD_STAGE_W2_SWIGLU,
    BACKWARD_STAGE_W2_SCATTER,
)


def _w13_implementation_uses_separate_x_remat(w13_implementation: str) -> bool:
    return w13_implementation in (
        SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_DW13_ONLY,
        SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_PERSISTENT_DW13_ONLY,
        SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_SPLIT_DW13_ONLY,
        SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_XLA_LOCAL_SWIGLU_DW13_ONLY,
        BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_LOCAL_SWIGLU_DX13_DW13,
        BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_DX13_XLA_LOCAL_SWIGLU_DW13,
        BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_DX13_SPLIT_LOCAL_SWIGLU_DW13,
        BACKWARD_W13_IMPLEMENTATION_XLA_DX13_XLA_LOCAL_SWIGLU_DW13,
        BACKWARD_W13_IMPLEMENTATION_XLA_DX13_ROUTE_BUFFER_XLA_LOCAL_SWIGLU_DW13,
        BACKWARD_W13_IMPLEMENTATION_XLA_DX13_PALLAS_X_REMAT_XLA_LOCAL_SWIGLU_DW13,
    )


def _backward_staged_block_timed_stages(
    stop_after_stage: str,
    w13_implementation: str = BACKWARD_IMPLEMENTATION_DEFAULT,
) -> tuple[str, ...]:
    stages = (BACKWARD_STAGE_DY_ROUTE,)
    if stop_after_stage in (
        BACKWARD_STAGE_W2,
        BACKWARD_STAGE_W13,
        BACKWARD_STAGE_DX_COMBINE,
        BACKWARD_STOP_AFTER_NONE,
    ):
        stages = (*stages, BACKWARD_STAGE_W2)
    if stop_after_stage in (BACKWARD_STAGE_W13, BACKWARD_STAGE_DX_COMBINE, BACKWARD_STOP_AFTER_NONE):
        if _w13_implementation_uses_separate_x_remat(w13_implementation):
            stages = (*stages, BACKWARD_STAGE_X_REMAT)
        stages = (*stages, BACKWARD_STAGE_W13)
    if stop_after_stage in (BACKWARD_STAGE_DX_COMBINE, BACKWARD_STOP_AFTER_NONE):
        stages = (*stages, BACKWARD_STAGE_DX_COMBINE)
    return stages


MODES = (
    MODE_FORWARD,
    MODE_FORWARD_BACKWARD,
    MODE_FORWARD_BACKWARD_REDUCED,
    MODE_FORWARD_BACKWARD_DX_CHECKSUM,
    MODE_FORWARD_BACKWARD_DROUTE_CHECKSUM,
    MODE_FORWARD_BACKWARD_DW13_CHECKSUM,
    MODE_FORWARD_BACKWARD_DW2_CHECKSUM,
    MODE_FORWARD_DECOMPOSED,
    MODE_FORWARD_DECOMPOSED_RAW_TOKENS,
    MODE_FORWARD_W13_DIRECT_COMPACT,
    MODE_FORWARD_W13_DIRECT_COMPACT_STORE_ZERO,
    MODE_FORWARD_W13_DIRECT_COMPACT_COMPUTE_ONLY_LOCAL,
    MODE_FORWARD_COMPACT_H_DECOMPOSED,
    MODE_FORWARD_COMPACT_H_DECOMPOSED_WITH_PREP,
    MODE_FORWARD_COMPACT_H_DECOMPOSED_WITH_PALLAS_PACK,
    MODE_FORWARD_COMPACT_H_RAW_TOKENS_DECOMPOSED,
    MODE_FORWARD_PACK_TOTAL,
    MODE_FORWARD_PACK_TOKEN_PACK,
    MODE_FORWARD_PACK_TOKEN_PACK_PALLAS,
    MODE_FORWARD_PACK_H_ROUTE_WEIGHTS,
    MODE_FORWARD_PACK_STATIC_SHARD,
    MODE_BACKWARD_DECOMPOSED,
    MODE_BACKWARD_STAGED_FLAT,
    MODE_BACKWARD_STAGED_BLOCKS,
    MODE_BACKWARD_DY_ROUTE_ONLY,
    MODE_BACKWARD_W2_ONLY,
    MODE_BACKWARD_W13_ONLY,
    MODE_BACKWARD_DX13_ONLY,
    MODE_BACKWARD_DX13_STORE_ZERO_ONLY,
    MODE_BACKWARD_DX13_ROUTE_BUFFER_ONLY,
    MODE_BACKWARD_DX13_PUSH_CONTRIB_ONLY,
    MODE_BACKWARD_DX13_SOURCE_COMPACT_ONLY,
    MODE_BACKWARD_DX13_SOURCE_COMPACT_BLOCK_ONLY,
    MODE_BACKWARD_DX13_SOURCE_COMPACT_COMBINE_ONLY,
    MODE_BACKWARD_DX13_XLA_SOURCE_COMPACT_DIRECT_ONLY,
    MODE_BACKWARD_DX13_XLA_ROUTE_BUFFER_DIRECT_ONLY,
    MODE_BACKWARD_DX13_SOURCE_GROUPED_ONLY,
    MODE_BACKWARD_RETURN_ONLY,
    MODE_BACKWARD_RETURN_COMPONENTS_ONLY,
)
TARGET_KERNEL_SUITE_MODES = (
    MODE_FORWARD,
    MODE_FORWARD_DECOMPOSED,
    MODE_FORWARD_W13_DIRECT_COMPACT,
    MODE_FORWARD_COMPACT_H_DECOMPOSED,
    MODE_BACKWARD_DY_ROUTE_ONLY,
    MODE_BACKWARD_W2_ONLY,
    MODE_BACKWARD_W13_ONLY,
    MODE_BACKWARD_DX13_ONLY,
    MODE_BACKWARD_DX13_PUSH_CONTRIB_ONLY,
    MODE_BACKWARD_RETURN_ONLY,
    MODE_BACKWARD_RETURN_COMPONENTS_ONLY,
    MODE_BACKWARD_STAGED_BLOCKS,
    MODE_FORWARD_BACKWARD_REDUCED,
)
TARGET_KERNEL_SUITE_PROFILE_KEYS = (
    "entries_per_rank",
    "inbox_slots",
    "block_m",
    "block_n",
    "block_k",
    "n_group",
    "n_groups_per_job",
    "send_worker_programs_per_peer",
    "worker_programs_per_peer",
    "send_pipeline_depth",
    "routing",
    "separate_compile",
)
TARGET_KERNEL_SUITE_TARGET_SHAPE = {
    "ep_size": 8,
    "tokens_per_rank": 32768,
    "hidden_dim": 2560,
    "intermediate_dim": 1280,
    "experts_per_rank": 32,
    "topk": 4,
    "capacity_factor": 1.25,
}
FORWARD_BACKWARD_GRAD_CHECKSUM_ARGNUM = {
    MODE_FORWARD_BACKWARD_DX_CHECKSUM: 0,
    MODE_FORWARD_BACKWARD_DROUTE_CHECKSUM: 1,
    MODE_FORWARD_BACKWARD_DW13_CHECKSUM: 2,
    MODE_FORWARD_BACKWARD_DW2_CHECKSUM: 3,
}
FORWARD_PACK_PROBE_MODE_TO_STAGE = {
    MODE_FORWARD_PACK_TOTAL: FORWARD_DECOMPOSED_STAGE_PACK_INPUTS,
    MODE_FORWARD_PACK_TOKEN_PACK: FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_TOKEN_PACK,
    MODE_FORWARD_PACK_TOKEN_PACK_PALLAS: FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_TOKEN_PACK,
    MODE_FORWARD_PACK_H_ROUTE_WEIGHTS: FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_H_ROUTE_WEIGHTS,
    MODE_FORWARD_PACK_STATIC_SHARD: FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_STATIC_SHARD,
}
BACKEND_RING = "ring"
BACKEND_RAGGED_A2A = "ragged_all_to_all"
BACKEND_PUBLIC_SOURCE_PUSH = "public_source_push"
BACKEND_SOURCE_PUSH_REFERENCE = "source_push_reference"
BACKEND_SOURCE_PUSH_PALLAS = "source_push_pallas_mgpu"
BACKENDS = (
    BACKEND_RING,
    BACKEND_RAGGED_A2A,
    BACKEND_PUBLIC_SOURCE_PUSH,
    BACKEND_SOURCE_PUSH_REFERENCE,
    BACKEND_SOURCE_PUSH_PALLAS,
)
PUBLIC_BACKEND_TO_IMPLEMENTATION = {
    BACKEND_RING: "ring",
    BACKEND_RAGGED_A2A: "ragged_all_to_all",
    BACKEND_PUBLIC_SOURCE_PUSH: "pallas_mgpu_source_push",
}
SOURCE_PUSH_BACKEND_TO_IMPLEMENTATION = {
    BACKEND_SOURCE_PUSH_REFERENCE: SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
    BACKEND_SOURCE_PUSH_PALLAS: SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU,
}
OUTER_JIT_CHOICES = ("auto", "true", "false")
SUMMARY_METRICS = (
    "steady_state_time",
    "compile_time",
    "lower_compile_time",
    "first_run_time",
    "first_call_time",
    "useful_forward_tflops_per_rank",
    "rounded_forward_tflops_per_rank",
    "useful_fwd_bwd_tflops_per_rank",
    "rounded_fwd_bwd_tflops_per_rank",
    "useful_backward_tflops_per_rank",
    "rounded_backward_tflops_per_rank",
    "useful_tflops_per_rank",
    "rounded_tflops_per_rank",
    "dropped_routes",
    "w13_payload_send_bytes_per_rank",
    "w13_lhs_compute_read_bytes_per_rank",
    "w13_weight_read_bytes_per_rank",
    "w13_compact_h_store_bytes_per_rank",
    "w13_estimated_total_bytes_per_rank",
    "w13_estimated_total_gbps_per_rank",
)


class MlpTiming(NamedTuple):
    """Timing result for one MLP benchmark callable."""

    compile_time: float | None
    lower_compile_time: float | None
    first_run_time: float | None
    first_call_time: float
    steady_state_times: list[float]
    output: Any


class InputPackTiming(NamedTuple):
    """Timing result for dynamic source-push input packing and sharding."""

    first_call_time: float
    steady_state_times: list[float]
    output: Any
    stage_steady_state_times: dict[str, list[float]] | None = None


class RawTokenForwardInputs(NamedTuple):
    """Device inputs for the raw-token W13-H source-push decomposition."""

    x: jax.Array
    token_ids: jax.Array
    send_meta: jax.Array
    recv_meta: jax.Array
    expert_base: jax.Array
    src_base_by_expert: jax.Array
    w_gate_up: jax.Array
    h_route_weights: jax.Array
    w_down: jax.Array
    queue_dst_ord: jax.Array
    queue_entry: jax.Array
    queue_row: jax.Array
    route_combine_weights: jax.Array
    route_valid_mask: jax.Array
    use_exact_expert_major: bool


class RawTokenCompactHForwardInputs(NamedTuple):
    """Device inputs for raw-token W13 with compact expert-major H."""

    x: jax.Array
    token_ids: jax.Array
    send_meta: jax.Array
    recv_meta: jax.Array
    expert_base: jax.Array
    src_base_by_expert: jax.Array
    w_gate_up: jax.Array
    h_route_weights: jax.Array
    w_down: jax.Array
    queue_dst_ord: jax.Array
    queue_entry: jax.Array
    queue_row: jax.Array
    route_combine_weights: jax.Array
    route_valid_mask: jax.Array
    use_exact_expert_major: bool


class RawTokenForwardTiming(NamedTuple):
    """Timing result for raw-token staged source-push forward."""

    compile_time: float
    steady_state_times: list[float]
    output: Any
    stage_steady_state_times: dict[str, list[float]]
    stage_compile_times: dict[str, float]


class BackwardDecomposedTiming(NamedTuple):
    """Timing result for staged source-push MLP backward diagnostics."""

    first_call_time: float
    steady_state_times: list[float]
    output: Any
    stage_steady_state_times: dict[str, list[float]]


def _parse_int_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in value.split(",") if part)
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return values


def _profile_defaults(argv: Sequence[str] | None = None) -> dict[str, Any]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--source-push-profile", choices=SOURCE_PUSH_PROFILES, default="none")
    args, _ = pre_parser.parse_known_args(argv)
    return source_push_profile_defaults(args.source_push_profile)


def _cli_flag_present(argv: Sequence[str], flag: str) -> bool:
    return any(item == flag or item.startswith(f"{flag}=") for item in argv)


def _apply_target_kernel_suite_defaults(args: argparse.Namespace, argv: Sequence[str]) -> None:
    if not _cli_flag_present(argv, "--source-push-profile"):
        args.source_push_profile = SOURCE_PUSH_PROFILE_STABLE_216

    profile_defaults = source_push_profile_defaults(SOURCE_PUSH_PROFILE_STABLE_216)
    for name in TARGET_KERNEL_SUITE_PROFILE_KEYS:
        flag = f"--{name.replace('_', '-')}"
        if name in profile_defaults and not _cli_flag_present(argv, flag):
            setattr(args, name, profile_defaults[name])

    for name, value in TARGET_KERNEL_SUITE_TARGET_SHAPE.items():
        flag = f"--{name.replace('_', '-')}"
        if not _cli_flag_present(argv, flag):
            setattr(args, name, value)

    if not _cli_flag_present(argv, "--backends"):
        args.backends = BACKEND_SOURCE_PUSH_PALLAS
    if not _cli_flag_present(argv, "--modes"):
        args.modes = ",".join(TARGET_KERNEL_SUITE_MODES)


def parse_source_push_mlp_fwd_bwd_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the source-push MLP forward/backward benchmark arguments."""

    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    profile_defaults = _profile_defaults(argv)

    def default(name: str, fallback: Any) -> Any:
        return profile_defaults.get(name, fallback)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-push-profile", choices=SOURCE_PUSH_PROFILES, default="none")
    parser.add_argument("--ep-size", type=int, default=default("ep_size", 8))
    parser.add_argument("--entries-per-rank", type=int, default=default("entries_per_rank", 2))
    parser.add_argument("--inbox-slots", type=int, default=default("inbox_slots", 2))
    parser.add_argument("--sweep-inbox-slots", type=_parse_int_csv, default=None)
    parser.add_argument("--hidden-dim", type=int, default=default("hidden_dim", 2560))
    parser.add_argument("--intermediate-dim", type=int, default=default("intermediate_dim", 1280))
    parser.add_argument("--block-m", type=int, default=default("block_m", 64))
    parser.add_argument("--block-n", type=int, default=default("block_n", 128))
    parser.add_argument("--block-k", type=int, default=default("block_k", 128))
    parser.add_argument("--n-group", type=int, default=default("n_group", 1))
    parser.add_argument("--n-groups-per-job", type=int, default=default("n_groups_per_job", 1))
    parser.add_argument("--sweep-n-groups-per-job", type=_parse_int_csv, default=None)
    parser.add_argument("--experts-per-rank", type=int, default=default("experts_per_rank", 32))
    parser.add_argument(
        "--send-worker-programs-per-peer",
        type=int,
        default=default("send_worker_programs_per_peer", 4),
    )
    parser.add_argument("--sweep-send-worker-programs-per-peer", type=_parse_int_csv, default=None)
    parser.add_argument(
        "--worker-programs-per-peer",
        type=int,
        default=default("worker_programs_per_peer", 16),
    )
    parser.add_argument("--sweep-worker-programs-per-peer", type=_parse_int_csv, default=None)
    parser.add_argument("--send-pipeline-depth", type=int, default=default("send_pipeline_depth", 1))
    parser.add_argument("--sweep-send-pipeline-depth", type=_parse_int_csv, default=None)
    parser.add_argument("--routing", type=str, default=default("routing", "balanced"))
    parser.add_argument("--tokens-per-rank", type=int, default=default("tokens_per_rank", 32768))
    parser.add_argument("--topk", type=int, default=default("topk", 4))
    parser.add_argument("--routing-seed", type=int, default=default("routing_seed", 0))
    parser.add_argument("--capacity-factor", type=float, default=default("capacity_factor", 1.25))
    parser.add_argument("--warmup", type=int, default=default("warmup", 1))
    parser.add_argument("--steps", type=int, default=default("steps", 3))
    parser.add_argument("--repeat-runs", type=int, default=default("repeat_runs", 1))
    parser.add_argument("--backends", default=BACKEND_SOURCE_PUSH_PALLAS)
    parser.add_argument("--modes", default=f"{MODE_FORWARD},{MODE_FORWARD_BACKWARD}")
    parser.add_argument(
        "--target-kernel-suite",
        action="store_true",
        help=(
            "Apply the target-shape source-push kernel isolation suite. This sets the stable Hopper "
            "source-push profile knobs, target MoE shape, source-push backend, and curated modes unless "
            "those flags are explicitly provided."
        ),
    )
    parser.add_argument(
        "--use-exact-expert-major",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use exact expert-major flat-H metadata instead of source-padded rows for source-push paths.",
    )
    parser.add_argument(
        "--backward-dy-route-implementation", choices=BACKWARD_DY_ROUTE_IMPLEMENTATIONS, default="default"
    )
    parser.add_argument("--backward-w2-implementation", choices=BACKWARD_W2_IMPLEMENTATIONS, default="default")
    parser.add_argument(
        "--backward-w2-split-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Benchmark-only: split staged-flat W2 timing into gather, activation, matmul, SwiGLU, and scatter.",
    )
    parser.add_argument(
        "--backward-dx13-implementation",
        choices=BACKWARD_DX13_IMPLEMENTATIONS,
        default=SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU,
        help="Benchmark-only implementation selector for backward_dx13_only.",
    )
    parser.add_argument(
        "--sweep-backward-dx13-implementation",
        default=None,
        help="Comma-separated DX13 implementations to benchmark with otherwise identical config.",
    )
    parser.add_argument("--backward-w13-implementation", choices=BACKWARD_W13_IMPLEMENTATIONS, default="default")
    parser.add_argument(
        "--sweep-backward-w13-implementation",
        default=None,
        help="Comma-separated W13 backward implementations to benchmark with otherwise identical config.",
    )
    parser.add_argument("--backward-w13-row-block", type=int, default=None)
    parser.add_argument("--backward-w13-hidden-block", type=int, default=None)
    parser.add_argument("--backward-w13-output-block", type=int, default=None)
    parser.add_argument("--sweep-backward-w13-row-block", type=_parse_int_csv, default=None)
    parser.add_argument("--sweep-backward-w13-hidden-block", type=_parse_int_csv, default=None)
    parser.add_argument("--sweep-backward-w13-output-block", type=_parse_int_csv, default=None)
    parser.add_argument(
        "--backward-w13-lowering-semantics",
        choices=BACKWARD_W13_LOWERING_SEMANTICS,
        default="auto",
        help="Benchmark-only: lowering semantics for local compact W13 DW13 MGPU diagnostics.",
    )
    parser.add_argument("--backward-return-implementation", choices=BACKWARD_RETURN_IMPLEMENTATIONS, default="default")
    parser.add_argument(
        "--sweep-backward-return-implementation",
        default=None,
        help="Comma-separated backward return implementations to benchmark with otherwise identical config.",
    )
    parser.add_argument(
        "--backward-stop-after-stage",
        choices=BACKWARD_STOP_AFTER_STAGES,
        default=BACKWARD_STOP_AFTER_NONE,
        help="Benchmark-only diagnostic stop point for staged-flat backward decomposition.",
    )
    parser.add_argument(
        "--outer-jit",
        choices=OUTER_JIT_CHOICES,
        default="auto",
        help="Use an outer jax.jit around the measured callable. Auto jit-compiles ring/ragged/reference only.",
    )
    parser.add_argument("--separate-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-exceptions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--git-sha", type=str, default=None)
    parser.add_argument("--jsonl", type=str, default=None)
    args = parser.parse_args(argv)
    if args.target_kernel_suite:
        _apply_target_kernel_suite_defaults(args, raw_argv)
    return args


def _source_push_w13_backward_block_sizes_from_args(
    *,
    row_block: int | None,
    hidden_block: int | None,
    output_block: int | None,
) -> SourcePushW13BackwardTiledBlockSizes | None:
    if row_block is None and hidden_block is None and output_block is None:
        return None
    defaults = SourcePushW13BackwardTiledBlockSizes.get_default()
    return SourcePushW13BackwardTiledBlockSizes(
        row_block=defaults.row_block if row_block is None else row_block,
        hidden_block=defaults.hidden_block if hidden_block is None else hidden_block,
        output_block=defaults.output_block if output_block is None else output_block,
    )


def _source_push_w13_lowering_semantics_from_arg(value: str) -> mgpu.LoweringSemantics | None:
    if value == "auto":
        return None
    if value == "lane":
        return mgpu.LoweringSemantics.Lane
    if value == "warpgroup":
        return mgpu.LoweringSemantics.Warpgroup
    raise ValueError(f"unknown W13 lowering semantics {value!r}")


def _resolved_w13_only_block_sizes(
    w13_implementation: str,
    block_sizes: SourcePushW13BackwardTiledBlockSizes | None,
) -> SourcePushW13BackwardTiledBlockSizes:
    if block_sizes is not None:
        return block_sizes
    if source_push_w13_backward_uses_local_dw13_default_block_sizes(w13_implementation):
        return source_push_w13_dw13_default_block_sizes()
    if w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_EXACT_FLAT_DW13_ONLY:
        defaults = SourcePushW13BackwardTiledBlockSizes.get_default()
        return replace(defaults, row_block=max(defaults.row_block, MIN_MOSAIC_INT32_TRANSFER_ELEMENTS))
    return SourcePushW13BackwardTiledBlockSizes.get_default()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_source_push_mlp_fwd_bwd_args(argv)
    if args.jsonl:
        jsonl_dir = os.path.dirname(args.jsonl)
        if jsonl_dir:
            os.makedirs(jsonl_dir, exist_ok=True)

    inbox_slots_values = args.sweep_inbox_slots or (args.inbox_slots,)
    send_worker_programs_per_peer_values = args.sweep_send_worker_programs_per_peer or (
        args.send_worker_programs_per_peer,
    )
    worker_programs_per_peer_values = args.sweep_worker_programs_per_peer or (args.worker_programs_per_peer,)
    send_pipeline_depth_values = args.sweep_send_pipeline_depth or (args.send_pipeline_depth,)
    n_groups_per_job_values = args.sweep_n_groups_per_job or (args.n_groups_per_job,)
    backward_dx13_implementation_values = (
        _parse_csv_choices(
            args.sweep_backward_dx13_implementation,
            BACKWARD_DX13_IMPLEMENTATIONS,
            flag="--sweep-backward-dx13-implementation",
        )
        if args.sweep_backward_dx13_implementation is not None
        else (args.backward_dx13_implementation,)
    )
    backward_w13_row_block_values = args.sweep_backward_w13_row_block or (args.backward_w13_row_block,)
    backward_w13_hidden_block_values = args.sweep_backward_w13_hidden_block or (args.backward_w13_hidden_block,)
    backward_w13_output_block_values = args.sweep_backward_w13_output_block or (args.backward_w13_output_block,)
    backward_w13_implementation_values = (
        _parse_csv_choices(
            args.sweep_backward_w13_implementation,
            BACKWARD_W13_IMPLEMENTATIONS,
            flag="--sweep-backward-w13-implementation",
        )
        if args.sweep_backward_w13_implementation is not None
        else (args.backward_w13_implementation,)
    )
    backward_return_implementation_values = (
        _parse_csv_choices(
            args.sweep_backward_return_implementation,
            BACKWARD_RETURN_IMPLEMENTATIONS,
            flag="--sweep-backward-return-implementation",
        )
        if args.sweep_backward_return_implementation is not None
        else (args.backward_return_implementation,)
    )
    backends = _parse_csv_choices(args.backends, BACKENDS, flag="--backends")
    modes = _parse_csv_choices(args.modes, MODES, flag="--modes")

    for (
        inbox_slots,
        send_worker_programs_per_peer,
        worker_programs_per_peer,
        send_pipeline_depth,
        n_groups_per_job,
        backward_dx13_implementation,
        backward_w13_row_block,
        backward_w13_hidden_block,
        backward_w13_output_block,
        backward_w13_implementation,
        backward_return_implementation,
    ) in product(
        inbox_slots_values,
        send_worker_programs_per_peer_values,
        worker_programs_per_peer_values,
        send_pipeline_depth_values,
        n_groups_per_job_values,
        backward_dx13_implementation_values,
        backward_w13_row_block_values,
        backward_w13_hidden_block_values,
        backward_w13_output_block_values,
        backward_w13_implementation_values,
        backward_return_implementation_values,
    ):
        config = PushInboxConfig(
            ep_size=args.ep_size,
            entries_per_rank=args.entries_per_rank,
            inbox_slots=inbox_slots,
            hidden_dim=args.hidden_dim,
            intermediate_dim=args.intermediate_dim,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            n_group=args.n_group,
            n_groups_per_job=n_groups_per_job,
            experts_per_rank=args.experts_per_rank,
            send_worker_programs_per_peer=send_worker_programs_per_peer,
            worker_programs_per_peer=worker_programs_per_peer,
            send_pipeline_depth=send_pipeline_depth,
            routing=args.routing,
            tokens_per_rank=args.tokens_per_rank,
            topk=args.topk,
            routing_seed=args.routing_seed,
            capacity_factor=args.capacity_factor,
        )
        backward_w13_block_sizes = _source_push_w13_backward_block_sizes_from_args(
            row_block=backward_w13_row_block,
            hidden_block=backward_w13_hidden_block,
            output_block=backward_w13_output_block,
        )
        rows = run_source_push_mlp_fwd_bwd(
            config,
            backends=backends,
            modes=modes,
            warmup=args.warmup,
            steps=args.steps,
            repeat_runs=args.repeat_runs,
            outer_jit=args.outer_jit,
            separate_compile=args.separate_compile,
            use_exact_expert_major=args.use_exact_expert_major,
            debug_exceptions=args.debug_exceptions,
            backward_dy_route_implementation=args.backward_dy_route_implementation,
            backward_w2_implementation=args.backward_w2_implementation,
            backward_w2_split_timing=args.backward_w2_split_timing,
            backward_dx13_implementation=backward_dx13_implementation,
            backward_w13_implementation=backward_w13_implementation,
            backward_w13_block_sizes=backward_w13_block_sizes,
            backward_w13_lowering_semantics=args.backward_w13_lowering_semantics,
            backward_return_implementation=backward_return_implementation,
            backward_stop_after_stage=args.backward_stop_after_stage,
        )
        for row in rows:
            if args.git_sha is not None:
                row["git_sha"] = args.git_sha
            line = json.dumps(row, sort_keys=True)
            print(line, flush=True)
            if args.jsonl:
                with open(args.jsonl, "a", encoding="utf-8") as f:
                    print(line, file=f, flush=True)


def run_source_push_mlp_fwd_bwd(
    config: PushInboxConfig,
    *,
    backends: Sequence[str],
    modes: Sequence[str],
    warmup: int,
    steps: int,
    repeat_runs: int,
    outer_jit: str,
    separate_compile: bool,
    use_exact_expert_major: bool = False,
    debug_exceptions: bool = False,
    backward_dy_route_implementation: str = BACKWARD_IMPLEMENTATION_DEFAULT,
    backward_w2_implementation: str = BACKWARD_IMPLEMENTATION_DEFAULT,
    backward_w2_split_timing: bool = False,
    backward_dx13_implementation: str = SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU,
    backward_w13_implementation: str = BACKWARD_IMPLEMENTATION_DEFAULT,
    backward_w13_block_sizes: SourcePushW13BackwardTiledBlockSizes | None = None,
    backward_w13_lowering_semantics: str = "lane",
    backward_return_implementation: str = BACKWARD_IMPLEMENTATION_DEFAULT,
    backward_stop_after_stage: str = BACKWARD_STOP_AFTER_NONE,
) -> list[dict[str, Any]]:
    """Run public/preplanned MLP forward and forward+backward timings."""

    rows = []
    for backend in backends:
        for mode in modes:
            rows.extend(
                _run_one(
                    config,
                    backend=backend,
                    mode=mode,
                    warmup=warmup,
                    steps=steps,
                    repeat_runs=repeat_runs,
                    outer_jit=outer_jit,
                    separate_compile=separate_compile,
                    use_exact_expert_major=use_exact_expert_major,
                    debug_exceptions=debug_exceptions,
                    backward_dy_route_implementation=backward_dy_route_implementation,
                    backward_w2_implementation=backward_w2_implementation,
                    backward_w2_split_timing=backward_w2_split_timing,
                    backward_dx13_implementation=backward_dx13_implementation,
                    backward_w13_implementation=backward_w13_implementation,
                    backward_w13_block_sizes=backward_w13_block_sizes,
                    backward_w13_lowering_semantics=backward_w13_lowering_semantics,
                    backward_return_implementation=backward_return_implementation,
                    backward_stop_after_stage=backward_stop_after_stage,
                )
            )
    return rows


def _run_one(
    config: PushInboxConfig,
    *,
    backend: str,
    mode: str,
    warmup: int,
    steps: int,
    repeat_runs: int,
    outer_jit: str,
    separate_compile: bool,
    use_exact_expert_major: bool,
    debug_exceptions: bool,
    backward_dy_route_implementation: str,
    backward_w2_implementation: str,
    backward_w2_split_timing: bool,
    backward_dx13_implementation: str,
    backward_w13_implementation: str,
    backward_w13_block_sizes: SourcePushW13BackwardTiledBlockSizes | None,
    backward_w13_lowering_semantics: str,
    backward_return_implementation: str,
    backward_stop_after_stage: str,
) -> list[dict[str, Any]]:
    try:
        config.validate()
        if repeat_runs <= 0:
            raise ValueError(f"repeat_runs must be positive, got {repeat_runs}")
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        mesh = _make_public_ep_mesh(config.ep_size)
        raw_inputs = make_source_push_forward_source_plan_raw_inputs(config)
        host_inputs = make_source_push_forward_plan_inputs(
            config,
            raw_inputs.selected_experts,
            input_mode="exact_source_push_plan" if use_exact_expert_major else "source_push_plan",
            use_exact_expert_major=use_exact_expert_major,
        )
        route_table = source_push_mlp_route_table_from_plan(
            host_inputs.plan,
            src_base_by_expert=host_inputs.src_base_by_expert,
        )
        inputs = _device_benchmark_inputs(config, raw_inputs, mesh)
        use_outer_jit = _resolve_outer_jit(backend, outer_jit)
        if mode == MODE_BACKWARD_DECOMPOSED:
            if backend != BACKEND_SOURCE_PUSH_PALLAS:
                raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
            return _run_source_push_backward_decomposed(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                route_table=route_table,
                inputs=inputs,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
                backward_dy_route_implementation=backward_dy_route_implementation,
                backward_w2_implementation=backward_w2_implementation,
                backward_w2_split_timing=backward_w2_split_timing,
                backward_w13_implementation=backward_w13_implementation,
                backward_return_implementation=backward_return_implementation,
            )
        if mode == MODE_BACKWARD_STAGED_FLAT:
            if backend != BACKEND_SOURCE_PUSH_PALLAS:
                raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
            return _run_source_push_backward_staged_flat(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                route_table=route_table,
                inputs=inputs,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
                backward_dy_route_implementation=backward_dy_route_implementation,
                backward_w2_implementation=backward_w2_implementation,
                backward_w2_split_timing=backward_w2_split_timing,
                backward_w13_implementation=backward_w13_implementation,
                backward_return_implementation=backward_return_implementation,
                backward_stop_after_stage=backward_stop_after_stage,
            )
        if mode == MODE_BACKWARD_STAGED_BLOCKS:
            if backend != BACKEND_SOURCE_PUSH_PALLAS:
                raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
            return _run_source_push_backward_staged_blocks(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                route_table=route_table,
                inputs=inputs,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
                backward_dy_route_implementation=backward_dy_route_implementation,
                backward_w2_implementation=backward_w2_implementation,
                backward_w13_implementation=backward_w13_implementation,
                backward_return_implementation=backward_return_implementation,
                backward_stop_after_stage=backward_stop_after_stage,
            )
        if mode in (MODE_BACKWARD_DY_ROUTE_ONLY, MODE_BACKWARD_W2_ONLY):
            if backend != BACKEND_SOURCE_PUSH_PALLAS:
                raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
            stop_after_stage = BACKWARD_STAGE_DY_ROUTE if mode == MODE_BACKWARD_DY_ROUTE_ONLY else BACKWARD_STAGE_W2
            rows = _run_source_push_backward_staged_blocks(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                route_table=route_table,
                inputs=inputs,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
                backward_dy_route_implementation=backward_dy_route_implementation,
                backward_w2_implementation=backward_w2_implementation,
                backward_w13_implementation=backward_w13_implementation,
                backward_return_implementation=backward_return_implementation,
                backward_stop_after_stage=stop_after_stage,
            )
            for row in rows:
                row["mode"] = mode
            return rows
        if mode == MODE_BACKWARD_W13_ONLY:
            if backend != BACKEND_SOURCE_PUSH_PALLAS:
                raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
            return _run_source_push_backward_w13_only(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                route_table=route_table,
                inputs=inputs,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
                backward_w13_implementation=backward_w13_implementation,
                backward_w13_block_sizes=backward_w13_block_sizes,
                backward_w13_lowering_semantics=backward_w13_lowering_semantics,
            )
        if mode == MODE_BACKWARD_DX13_ONLY:
            if backend != BACKEND_SOURCE_PUSH_PALLAS:
                raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
            return _run_source_push_backward_dx13_only(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                route_table=route_table,
                inputs=inputs,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
                backward_dx13_implementation=backward_dx13_implementation,
            )
        if mode == MODE_BACKWARD_DX13_STORE_ZERO_ONLY:
            if backend != BACKEND_SOURCE_PUSH_PALLAS:
                raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
            return _run_source_push_backward_dx13_store_zero_only(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                route_table=route_table,
                inputs=inputs,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
            )
        if mode == MODE_BACKWARD_DX13_ROUTE_BUFFER_ONLY:
            if backend != BACKEND_SOURCE_PUSH_PALLAS:
                raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
            return _run_source_push_backward_dx13_route_buffer_only(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                route_table=route_table,
                inputs=inputs,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
            )
        if mode in (MODE_BACKWARD_DX13_PUSH_CONTRIB_ONLY, MODE_BACKWARD_DX13_SOURCE_COMPACT_ONLY):
            if backend != BACKEND_SOURCE_PUSH_PALLAS:
                raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
            return _run_source_push_backward_dx13_source_compact_only(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                route_table=route_table,
                inputs=inputs,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
                mode=mode,
            )
        if mode == MODE_BACKWARD_DX13_SOURCE_COMPACT_BLOCK_ONLY:
            if backend != BACKEND_SOURCE_PUSH_PALLAS:
                raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
            return _run_source_push_backward_dx13_source_compact_only(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                route_table=route_table,
                inputs=inputs,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
                block_contiguous=True,
            )
        if mode == MODE_BACKWARD_DX13_SOURCE_COMPACT_COMBINE_ONLY:
            if backend != BACKEND_SOURCE_PUSH_PALLAS:
                raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
            return _run_source_push_backward_dx13_source_compact_combine_only(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
            )
        if mode == MODE_BACKWARD_DX13_XLA_SOURCE_COMPACT_DIRECT_ONLY:
            if backend != BACKEND_SOURCE_PUSH_PALLAS:
                raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
            return _run_source_push_backward_dx13_xla_source_compact_direct_only(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                route_table=route_table,
                inputs=inputs,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
            )
        if mode == MODE_BACKWARD_DX13_XLA_ROUTE_BUFFER_DIRECT_ONLY:
            if backend != BACKEND_SOURCE_PUSH_PALLAS:
                raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
            return _run_source_push_backward_dx13_xla_route_buffer_direct_only(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                route_table=route_table,
                inputs=inputs,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
            )
        if mode == MODE_BACKWARD_DX13_SOURCE_GROUPED_ONLY:
            if backend != BACKEND_SOURCE_PUSH_PALLAS:
                raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
            return _run_source_push_backward_dx13_source_grouped_only(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                route_table=route_table,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
            )
        if mode == MODE_BACKWARD_RETURN_ONLY:
            if backend != BACKEND_SOURCE_PUSH_PALLAS:
                raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
            return _run_source_push_backward_return_only(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                route_table=route_table,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
                backward_return_implementation=backward_return_implementation,
            )
        if mode == MODE_BACKWARD_RETURN_COMPONENTS_ONLY:
            if backend != BACKEND_SOURCE_PUSH_PALLAS:
                raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
            return _run_source_push_backward_return_components_only(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                route_table=route_table,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
            )
        if mode in FORWARD_PACK_PROBE_MODE_TO_STAGE:
            if backend != BACKEND_SOURCE_PUSH_PALLAS:
                raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
            return _run_source_push_pack_probe(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                inputs=inputs,
                mode=mode,
                stage=FORWARD_PACK_PROBE_MODE_TO_STAGE[mode],
                use_pallas_token_pack=mode == MODE_FORWARD_PACK_TOKEN_PACK_PALLAS,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
            )
        if mode in (MODE_FORWARD_DECOMPOSED, MODE_FORWARD_DECOMPOSED_RAW_TOKENS):
            if backend != BACKEND_SOURCE_PUSH_PALLAS:
                raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
            if mode == MODE_FORWARD_DECOMPOSED_RAW_TOKENS:
                return _run_source_push_forward_raw_token_decomposed(
                    config,
                    mesh=mesh,
                    host_inputs=host_inputs,
                    inputs=inputs,
                    warmup=warmup,
                    steps=steps,
                    repeat_runs=repeat_runs,
                )
            return _run_source_push_forward_decomposed(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                inputs=inputs,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
            )
        if mode in (
            MODE_FORWARD_W13_DIRECT_COMPACT,
            MODE_FORWARD_W13_DIRECT_COMPACT_STORE_ZERO,
            MODE_FORWARD_W13_DIRECT_COMPACT_COMPUTE_ONLY_LOCAL,
        ):
            if backend != BACKEND_SOURCE_PUSH_PALLAS:
                raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
            diagnostic_variant = DIAGNOSTIC_VARIANT_FULL
            if mode == MODE_FORWARD_W13_DIRECT_COMPACT_STORE_ZERO:
                diagnostic_variant = DIAGNOSTIC_VARIANT_STORE_ZERO
            elif mode == MODE_FORWARD_W13_DIRECT_COMPACT_COMPUTE_ONLY_LOCAL:
                diagnostic_variant = DIAGNOSTIC_VARIANT_COMPUTE_ONLY_LOCAL
            return _run_source_push_forward_w13_direct_compact(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                inputs=inputs,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
                mode=mode,
                diagnostic_variant=diagnostic_variant,
            )
        if mode in (
            MODE_FORWARD_COMPACT_H_DECOMPOSED,
            MODE_FORWARD_COMPACT_H_DECOMPOSED_WITH_PREP,
            MODE_FORWARD_COMPACT_H_DECOMPOSED_WITH_PALLAS_PACK,
        ):
            if backend != BACKEND_SOURCE_PUSH_PALLAS:
                raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
            return _run_source_push_forward_compact_h_decomposed(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                route_table=route_table,
                inputs=inputs,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
                include_prepare=mode
                in (MODE_FORWARD_COMPACT_H_DECOMPOSED_WITH_PREP, MODE_FORWARD_COMPACT_H_DECOMPOSED_WITH_PALLAS_PACK),
                use_pallas_token_pack=mode == MODE_FORWARD_COMPACT_H_DECOMPOSED_WITH_PALLAS_PACK,
                mode=mode,
            )
        if mode == MODE_FORWARD_COMPACT_H_RAW_TOKENS_DECOMPOSED:
            if backend != BACKEND_SOURCE_PUSH_PALLAS:
                raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
            return _run_source_push_forward_compact_h_raw_token_decomposed(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                route_table=route_table,
                inputs=inputs,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
            )
        fn, call_args = _make_benchmark_callable(
            config,
            backend=backend,
            mode=mode,
            mesh=mesh,
            host_inputs=host_inputs,
            route_table=route_table,
            inputs=inputs,
            backward_dy_route_implementation=backward_dy_route_implementation,
            backward_w2_implementation=backward_w2_implementation,
            backward_w13_implementation=backward_w13_implementation,
            backward_return_implementation=backward_return_implementation,
        )
        timing = _time_callable(
            fn,
            *call_args,
            mesh=mesh,
            warmup=warmup,
            steps=steps,
            repeat_runs=repeat_runs,
            use_outer_jit=use_outer_jit,
            separate_compile=separate_compile,
        )
        return _timing_rows(
            config,
            backend=backend,
            mode=mode,
            timing=timing,
            queue_stats=host_inputs.queue_stats,
            repeat_runs=repeat_runs,
            outer_jit=use_outer_jit,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
        if debug_exceptions:
            raise
        return [
            {
                "kernel": KERNEL_NAME,
                "implementation": backend,
                "backend": backend,
                "mode": mode,
                "row_type": "error",
                "config": asdict(config),
                "outer_jit": _outer_jit_error_value(backend, outer_jit),
                "repeat_run": None,
                "repeat_runs": repeat_runs,
                "steady_state_time": None,
                "compile_time": None,
                "lower_compile_time": None,
                "first_run_time": None,
                "first_call_time": None,
                "dropped_routes": None,
                "error": f"{type(exc).__name__}: {exc}",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        ]
    finally:
        jax.clear_caches()


def _run_source_push_pack_probe(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    inputs: dict[str, jax.Array],
    mode: str,
    stage: str,
    use_pallas_token_pack: bool,
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> list[dict[str, Any]]:
    probe = _make_source_push_pack_probe(
        config,
        mesh=mesh,
        host_inputs=host_inputs,
        inputs=inputs,
        stage=stage,
        use_pallas_token_pack=use_pallas_token_pack,
    )

    start = time.perf_counter()
    output = probe()
    first_call_time = time.perf_counter() - start
    _block_until_ready(output)

    for _ in range(warmup):
        _block_until_ready(probe())

    steady_state_times = []
    for _ in range(repeat_runs):
        start = time.perf_counter()
        for _ in range(steps):
            _block_until_ready(probe())
        steady_state_times.append((time.perf_counter() - start) / steps)

    rows = [
        _pack_probe_row(
            config,
            queue_stats=host_inputs.queue_stats,
            mode=mode,
            stage=stage,
            repeat_run=repeat_run,
            repeat_runs=repeat_runs,
            steady_state_time=steady_state_time,
            first_call_time=first_call_time,
        )
        for repeat_run, steady_state_time in enumerate(steady_state_times)
    ]
    summary = _summary_row(rows)
    return [*rows, summary]


def _make_source_push_pack_probe(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    inputs: dict[str, jax.Array],
    stage: str,
    use_pallas_token_pack: bool = False,
) -> Callable[[], Any]:
    x = inputs["x_source"]
    route_weights = inputs["combine_source"]
    w13 = inputs["w13_source"]
    w2 = inputs["w2_source"]

    if stage == FORWARD_DECOMPOSED_STAGE_PACK_INPUTS:
        return lambda: _pack_probe_total(config, mesh, host_inputs, x, route_weights, w13, w2)
    if stage == FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_TOKEN_PACK:
        return lambda: _pack_probe_token_pack(mesh, host_inputs, x, use_pallas_token_pack=use_pallas_token_pack)
    if stage == FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_H_ROUTE_WEIGHTS:
        return lambda: _pack_probe_h_route_weights(config, mesh, host_inputs, route_weights)
    if stage == FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_STATIC_SHARD:
        with jax.set_mesh(mesh):
            packed_x = pack_source_push_tokens_jax(x, host_inputs.plan).astype(jnp.bfloat16)
            h_route_weights = source_push_h_row_route_weights_jax(
                route_weights,
                host_inputs.plan,
                host_inputs.send_meta,
                host_inputs.expert_base,
                host_inputs.src_base_by_expert,
                hidden_rows_per_rank=config.hidden_rows_per_rank,
                use_exact_expert_major=host_inputs.use_exact_expert_major,
            ).astype(jnp.bfloat16)
        _block_until_ready((packed_x, h_route_weights))
        return lambda: _pack_probe_static_shard(mesh, host_inputs, packed_x, h_route_weights, w13, w2)
    raise ValueError(f"unsupported pack probe stage {stage!r}")


def _pack_probe_total(config: PushInboxConfig, mesh: Mesh, host_inputs, x, route_weights, w13, w2):
    del route_weights
    compact_expert_capacity = _compact_h_expert_capacity_from_metadata(
        config,
        host_inputs.send_meta,
        host_inputs.expert_base,
        host_inputs.src_base_by_expert,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )
    with jax.set_mesh(mesh):
        packed_x = pack_source_push_tokens_jax(x, host_inputs.plan).astype(jnp.bfloat16)
        h_route_weights = jnp.zeros(
            (config.ep_size, config.experts_per_rank, compact_expert_capacity),
            dtype=jnp.bfloat16,
        )
        packed = _pack_compact_h_static_inputs(mesh, host_inputs, packed_x, h_route_weights, w13, w2, config)
    return packed


def _prepack_w13_h_expert_major_inputs(
    config: PushInboxConfig,
    mesh: Mesh,
    host_inputs,
    x: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
) -> tuple[SourcePushForwardDeviceInputs, int, int]:
    store_capacity = _compact_h_expert_capacity_from_metadata(
        config,
        host_inputs.send_meta,
        host_inputs.expert_base,
        host_inputs.src_base_by_expert,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )
    live_capacity = _compact_h_expert_capacity_from_metadata(
        config,
        host_inputs.send_meta,
        host_inputs.expert_base,
        host_inputs.src_base_by_expert,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
        include_store_padding=False,
    )
    with jax.set_mesh(mesh):
        packed_x = pack_source_push_tokens_jax(x, host_inputs.plan).astype(jnp.bfloat16)
        h_route_weights = jnp.zeros(
            (config.ep_size, config.experts_per_rank, store_capacity),
            dtype=jnp.bfloat16,
        )
        packed = _pack_compact_h_static_inputs(mesh, host_inputs, packed_x, h_route_weights, w13, w2, config)
    return packed, store_capacity, live_capacity


def _pack_probe_token_pack(mesh: Mesh, host_inputs, x, *, use_pallas_token_pack: bool):
    with jax.set_mesh(mesh):
        if use_pallas_token_pack:
            return source_push_pack_tokens_pallas_mgpu(x, host_inputs.plan, mesh=mesh)
        packed_x = pack_source_push_tokens_jax(x, host_inputs.plan).astype(jnp.bfloat16)
    return packed_x


def _pack_probe_h_route_weights(config: PushInboxConfig, mesh: Mesh, host_inputs, route_weights):
    with jax.set_mesh(mesh):
        return source_push_h_row_route_weights_jax(
            route_weights,
            host_inputs.plan,
            host_inputs.send_meta,
            host_inputs.expert_base,
            host_inputs.src_base_by_expert,
            hidden_rows_per_rank=config.hidden_rows_per_rank,
            use_exact_expert_major=host_inputs.use_exact_expert_major,
        ).astype(jnp.bfloat16)


def _pack_probe_static_shard(mesh: Mesh, host_inputs, packed_x, h_route_weights, w13, w2):
    with jax.set_mesh(mesh):
        packed = SourcePushForwardDeviceInputs(
            x=packed_x,
            send_meta=jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
            recv_meta=jnp.asarray(host_inputs.recv_meta, dtype=jnp.int32),
            expert_base=jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
            src_base_by_expert=jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
            w_gate_up=jnp.asarray(w13, dtype=jnp.bfloat16),
            w_down=jnp.asarray(w2, dtype=jnp.bfloat16),
            queue_dst_ord=jnp.asarray(host_inputs.queue_dst_ord, dtype=jnp.int32),
            queue_entry=jnp.asarray(host_inputs.queue_entry, dtype=jnp.int32),
            queue_row=jnp.asarray(host_inputs.queue_row, dtype=jnp.int32),
            h_route_weights=h_route_weights,
            route_combine_weights=jnp.asarray(host_inputs.route_combine_weights, dtype=jnp.bfloat16),
            route_valid_mask=jnp.asarray(host_inputs.route_valid_mask, dtype=jnp.bool_),
            queue_stats=host_inputs.queue_stats,
            use_exact_expert_major=host_inputs.use_exact_expert_major,
        )
        packed = _shard_source_push_forward_inputs(mesh, packed)
    return _block_source_push_forward_device_inputs(packed)


def _pack_probe_row(
    config: PushInboxConfig,
    *,
    queue_stats: dict[str, Any],
    mode: str,
    stage: str,
    repeat_run: int,
    repeat_runs: int,
    steady_state_time: float,
    first_call_time: float,
) -> dict[str, Any]:
    return {
        "kernel": KERNEL_NAME,
        "implementation": f"{BACKEND_SOURCE_PUSH_PALLAS}_{stage}",
        "backend": BACKEND_SOURCE_PUSH_PALLAS,
        "mode": mode,
        "stage": stage,
        "row_type": "repeat",
        "config": asdict(config),
        "queue_stats": queue_stats,
        **queue_stats,
        "outer_jit": False,
        "compile_time": None,
        "lower_compile_time": None,
        "first_run_time": None,
        "first_call_time": first_call_time,
        "repeat_run": repeat_run,
        "repeat_runs": repeat_runs,
        "steady_state_time": steady_state_time,
        "bytes_per_rank": None,
        "forward_gbps_per_rank": None,
        "useful_forward_tflops_per_rank": None,
        "rounded_forward_tflops_per_rank": None,
        "useful_fwd_bwd_tflops_per_rank": None,
        "rounded_fwd_bwd_tflops_per_rank": None,
        "useful_backward_tflops_per_rank": None,
        "rounded_backward_tflops_per_rank": None,
        "useful_tflops_per_rank": None,
        "rounded_tflops_per_rank": None,
        "dropped_routes": int(jax.device_get(queue_stats["dropped_routes"])),
        "error": None,
        "error_type": None,
        "error_message": None,
    }


def _run_source_push_forward_decomposed(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    inputs: dict[str, jax.Array],
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> list[dict[str, Any]]:
    pack_timing = _time_source_push_input_pack(
        config,
        mesh=mesh,
        host_inputs=host_inputs,
        x=inputs["x_source"],
        route_weights=inputs["combine_source"],
        w13=inputs["w13_source"],
        w2=inputs["w2_source"],
        warmup=warmup,
        steps=steps,
        repeat_runs=repeat_runs,
    )
    packed = pack_timing.output
    staged_timing = _time_staged_source_push_forward(
        mesh,
        config,
        packed.x,
        packed.send_meta,
        packed.recv_meta,
        packed.expert_base,
        packed.src_base_by_expert,
        packed.w_gate_up,
        packed.h_route_weights,
        packed.w_down,
        packed.queue_dst_ord,
        packed.queue_entry,
        packed.queue_row,
        packed.route_combine_weights,
        packed.route_valid_mask,
        warmup=warmup,
        steps=steps,
        repeat_runs=repeat_runs,
        use_exact_expert_major=packed.use_exact_expert_major,
    )
    return _decomposed_forward_rows(
        config,
        pack_timing=pack_timing,
        staged_timing=staged_timing,
        queue_stats=host_inputs.queue_stats,
        repeat_runs=repeat_runs,
        mode=MODE_FORWARD_DECOMPOSED,
        input_stage=FORWARD_DECOMPOSED_STAGE_PACK_INPUTS,
    )


def _run_source_push_forward_raw_token_decomposed(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    inputs: dict[str, jax.Array],
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> list[dict[str, Any]]:
    prepare_timing = _time_source_push_raw_token_input_prepare(
        config,
        mesh=mesh,
        host_inputs=host_inputs,
        x=inputs["x_source"],
        route_weights=inputs["combine_source"],
        w13=inputs["w13_source"],
        w2=inputs["w2_source"],
        warmup=warmup,
        steps=steps,
        repeat_runs=repeat_runs,
    )
    raw_inputs = prepare_timing.output
    staged_timing = _time_staged_source_push_forward_raw_tokens(
        mesh,
        config,
        raw_inputs,
        warmup=warmup,
        steps=steps,
        repeat_runs=repeat_runs,
    )
    return _decomposed_forward_rows(
        config,
        pack_timing=prepare_timing,
        staged_timing=staged_timing,
        queue_stats=host_inputs.queue_stats,
        repeat_runs=repeat_runs,
        mode=MODE_FORWARD_DECOMPOSED_RAW_TOKENS,
        input_stage=FORWARD_DECOMPOSED_STAGE_PREPARE_INPUTS,
    )


def _run_source_push_forward_compact_h_raw_token_decomposed(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    route_table,
    inputs: dict[str, jax.Array],
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> list[dict[str, Any]]:
    compact_expert_capacity = _compact_h_expert_capacity_from_metadata(
        config,
        host_inputs.send_meta,
        host_inputs.expert_base,
        host_inputs.src_base_by_expert,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )
    prepare_timing = _time_source_push_raw_token_compact_h_input_prepare(
        mesh=mesh,
        host_inputs=host_inputs,
        route_table=route_table,
        x=inputs["x_source"],
        route_weights=inputs["combine_source"],
        w13=inputs["w13_source"],
        w2=inputs["w2_source"],
        compact_expert_capacity=compact_expert_capacity,
        warmup=warmup,
        steps=steps,
        repeat_runs=repeat_runs,
    )
    raw_inputs = prepare_timing.output
    staged_timing = _time_staged_source_push_forward_raw_tokens_compact_h(
        mesh,
        config,
        raw_inputs,
        compact_expert_capacity=compact_expert_capacity,
        warmup=warmup,
        steps=steps,
        repeat_runs=repeat_runs,
    )
    rows = _decomposed_forward_rows(
        config,
        pack_timing=prepare_timing,
        staged_timing=staged_timing,
        queue_stats=host_inputs.queue_stats,
        repeat_runs=repeat_runs,
        mode=MODE_FORWARD_COMPACT_H_RAW_TOKENS_DECOMPOSED,
        input_stage=FORWARD_DECOMPOSED_STAGE_PREPARE_INPUTS,
    )
    for row in rows:
        row["compact_h_layout"] = "direct_padded_expert_major"
        row["compact_expert_capacity"] = compact_expert_capacity
    return rows


def _run_source_push_forward_w13_direct_compact(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    inputs: dict[str, jax.Array],
    warmup: int,
    steps: int,
    repeat_runs: int,
    mode: str = MODE_FORWARD_W13_DIRECT_COMPACT,
    diagnostic_variant: str = DIAGNOSTIC_VARIANT_FULL,
) -> list[dict[str, Any]]:
    packed = _pack_probe_total(
        config,
        mesh,
        host_inputs,
        inputs["x_source"],
        inputs["combine_source"],
        inputs["w13_source"],
        inputs["w2_source"],
    )
    compact_expert_capacity = _compact_h_expert_capacity_from_metadata(
        config,
        host_inputs.send_meta,
        host_inputs.expert_base,
        host_inputs.src_base_by_expert,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )
    timing = _time_source_push_w13_direct_compact(
        mesh,
        config,
        packed,
        compact_expert_capacity=compact_expert_capacity,
        warmup=warmup,
        steps=steps,
        repeat_runs=repeat_runs,
        diagnostic_variant=diagnostic_variant,
    )
    return _w13_direct_compact_rows(
        config,
        timing=timing,
        queue_stats=host_inputs.queue_stats,
        compact_expert_capacity=compact_expert_capacity,
        repeat_runs=repeat_runs,
        mode=mode,
        diagnostic_variant=diagnostic_variant,
    )


def _run_source_push_forward_compact_h_decomposed(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    route_table,
    inputs: dict[str, jax.Array],
    warmup: int,
    steps: int,
    repeat_runs: int,
    include_prepare: bool = False,
    use_pallas_token_pack: bool = False,
    mode: str = MODE_FORWARD_COMPACT_H_DECOMPOSED,
) -> list[dict[str, Any]]:
    compact_expert_capacity = _compact_h_expert_capacity_from_metadata(
        config,
        host_inputs.send_meta,
        host_inputs.expert_base,
        host_inputs.src_base_by_expert,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )
    if include_prepare:
        pack_timing = _time_source_push_compact_h_input_pack(
            config,
            mesh=mesh,
            host_inputs=host_inputs,
            route_table=route_table,
            x=inputs["x_source"],
            route_weights=inputs["combine_source"],
            w13=inputs["w13_source"],
            w2=inputs["w2_source"],
            compact_expert_capacity=compact_expert_capacity,
            use_pallas_token_pack=use_pallas_token_pack,
            warmup=warmup,
            steps=steps,
            repeat_runs=repeat_runs,
        )
        packed, h_route_weights = pack_timing.output
        input_stage = FORWARD_DECOMPOSED_STAGE_PREPARE_INPUTS
    else:
        packed, h_route_weights = _pack_compact_h_probe_total(
            config,
            mesh,
            host_inputs,
            route_table,
            inputs["x_source"],
            inputs["combine_source"],
            inputs["w13_source"],
            inputs["w2_source"],
            compact_expert_capacity=compact_expert_capacity,
        )
        pack_timing = InputPackTiming(
            first_call_time=0.0,
            steady_state_times=[0.0 for _ in range(repeat_runs)],
            output=(packed, h_route_weights),
        )
        input_stage = FORWARD_DECOMPOSED_STAGE_PREPACKED_INPUTS
    staged_timing = _time_staged_source_push_forward_compact_h(
        mesh,
        config,
        packed,
        h_route_weights,
        compact_expert_capacity=compact_expert_capacity,
        warmup=warmup,
        steps=steps,
        repeat_runs=repeat_runs,
    )
    rows = _decomposed_forward_rows(
        config,
        pack_timing=pack_timing,
        staged_timing=staged_timing,
        queue_stats=host_inputs.queue_stats,
        repeat_runs=repeat_runs,
        mode=mode,
        input_stage=input_stage,
    )
    for row in rows:
        row["compact_h_layout"] = "direct_padded_expert_major"
        row["compact_expert_capacity"] = compact_expert_capacity
    return rows


def _pack_compact_h_probe_total(
    config: PushInboxConfig,
    mesh: Mesh,
    host_inputs,
    route_table,
    x: jax.Array,
    route_weights: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    *,
    compact_expert_capacity: int,
    use_pallas_token_pack: bool = False,
) -> tuple[SourcePushForwardDeviceInputs, jax.Array]:
    with jax.set_mesh(mesh):
        if use_pallas_token_pack:
            packed_x = source_push_pack_tokens_pallas_mgpu(x, host_inputs.plan, mesh=mesh)
        else:
            packed_x = pack_source_push_tokens_jax(x, host_inputs.plan).astype(jnp.bfloat16)
        expert_route_weights = source_push_mlp._source_push_mlp_route_weights_to_all_expert_major(
            route_table,
            route_weights,
        )
        h_route_weights = source_push_mlp._source_push_mlp_pad_expert_route_weights(
            expert_route_weights,
            compact_expert_capacity,
        ).astype(jnp.bfloat16)
        packed = _pack_compact_h_static_inputs(mesh, host_inputs, packed_x, h_route_weights, w13, w2, config)
    return _block_compact_h_forward_inputs(packed, h_route_weights)


def _pack_compact_h_static_inputs(
    mesh: Mesh,
    host_inputs,
    packed_x: jax.Array,
    h_route_weights: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    config: PushInboxConfig,
) -> SourcePushForwardDeviceInputs:
    h_route_weights = jax.device_put(h_route_weights, NamedSharding(mesh, P(AXIS, None, None)))
    packed = SourcePushForwardDeviceInputs(
        x=packed_x,
        send_meta=jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        recv_meta=jnp.asarray(host_inputs.recv_meta, dtype=jnp.int32),
        expert_base=jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
        src_base_by_expert=jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        w_gate_up=jnp.asarray(w13, dtype=jnp.bfloat16),
        w_down=jnp.asarray(w2, dtype=jnp.bfloat16),
        queue_dst_ord=jnp.asarray(host_inputs.queue_dst_ord, dtype=jnp.int32),
        queue_entry=jnp.asarray(host_inputs.queue_entry, dtype=jnp.int32),
        queue_row=jnp.asarray(host_inputs.queue_row, dtype=jnp.int32),
        h_route_weights=jnp.zeros((config.ep_size, config.hidden_rows_per_rank), dtype=jnp.bfloat16),
        route_combine_weights=jnp.asarray(host_inputs.route_combine_weights, dtype=jnp.bfloat16),
        route_valid_mask=jnp.asarray(host_inputs.route_valid_mask, dtype=jnp.bool_),
        queue_stats=host_inputs.queue_stats,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )
    return _shard_source_push_forward_inputs(mesh, packed)


def _block_compact_h_forward_inputs(
    packed: SourcePushForwardDeviceInputs,
    h_route_weights: jax.Array,
) -> tuple[SourcePushForwardDeviceInputs, jax.Array]:
    return _block_source_push_forward_device_inputs(packed), _block_until_ready(h_route_weights)


def _run_source_push_backward_decomposed(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    route_table,
    inputs: dict[str, jax.Array],
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> list[dict[str, Any]]:
    def call_forward_h():
        out, h_flat, dropped_routes = source_push_forward_with_h_from_plan(
            config,
            host_inputs,
            inputs["x_source"],
            inputs["combine_source"],
            inputs["w13_source"],
            inputs["w2_source"],
            execution_mode=FORWARD_EXECUTION_STAGED_HOST_SYNC,
            mesh=mesh,
        )
        _block_until_ready((out, h_flat, dropped_routes))
        return out, h_flat, dropped_routes

    with jax.set_mesh(mesh):
        forward_start = time.perf_counter()
        out, h_flat, dropped_routes = call_forward_h()
        forward_h_first_call_time = time.perf_counter() - forward_start

        for _ in range(warmup):
            out, h_flat, dropped_routes = call_forward_h()

        forward_h_times = []
        for _ in range(repeat_runs):
            start = time.perf_counter()
            for _ in range(steps):
                out, h_flat, dropped_routes = call_forward_h()
            forward_h_times.append((time.perf_counter() - start) / steps)

    dy = jnp.ones_like(out, dtype=jnp.float32)
    with jax.set_mesh(mesh):
        expert_route_weights = source_push_mlp._source_push_mlp_route_weights_to_all_expert_major(
            route_table,
            inputs["combine_source"],
        )
        _block_until_ready(expert_route_weights)
        timing = _time_source_push_backward_decomposed(
            route_table,
            jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
            inputs["x_source"],
            expert_route_weights,
            inputs["w13_source"],
            inputs["w2_source"],
            h_flat,
            dy,
            warmup=warmup,
            steps=steps,
            repeat_runs=repeat_runs,
        )
    rows = _decomposed_backward_rows(
        config,
        timing=timing,
        queue_stats=host_inputs.queue_stats,
        repeat_runs=repeat_runs,
        dropped_routes=int(jax.device_get(dropped_routes)),
        forward_h_first_call_time=forward_h_first_call_time,
        forward_h_times=forward_h_times,
    )
    return rows


def _run_source_push_backward_staged_flat(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    route_table,
    inputs: dict[str, jax.Array],
    warmup: int,
    steps: int,
    repeat_runs: int,
    backward_dy_route_implementation: str,
    backward_w2_implementation: str,
    backward_w2_split_timing: bool,
    backward_w13_implementation: str,
    backward_return_implementation: str,
    backward_stop_after_stage: str,
) -> list[dict[str, Any]]:
    def call_forward_residuals():
        out, h_flat, dropped_routes = source_push_forward_with_h_from_plan(
            config,
            host_inputs,
            inputs["x_source"],
            inputs["combine_source"],
            inputs["w13_source"],
            inputs["w2_source"],
            execution_mode=FORWARD_EXECUTION_STAGED_HOST_SYNC,
            mesh=mesh,
        )
        h_route_weights = source_push_h_row_route_weights_jax(
            inputs["combine_source"],
            host_inputs.plan,
            host_inputs.send_meta,
            host_inputs.expert_base,
            host_inputs.src_base_by_expert,
            hidden_rows_per_rank=config.hidden_rows_per_rank,
            use_exact_expert_major=host_inputs.use_exact_expert_major,
        )
        _block_until_ready((out, h_flat, h_route_weights, dropped_routes))
        return out, h_flat, h_route_weights, dropped_routes

    with jax.set_mesh(mesh):
        forward_start = time.perf_counter()
        out, h_flat, h_route_weights, dropped_routes = call_forward_residuals()
        forward_h_first_call_time = time.perf_counter() - forward_start

        for _ in range(warmup):
            out, h_flat, h_route_weights, dropped_routes = call_forward_residuals()

        forward_h_times = []
        for _ in range(repeat_runs):
            start = time.perf_counter()
            for _ in range(steps):
                out, h_flat, h_route_weights, dropped_routes = call_forward_residuals()
            forward_h_times.append((time.perf_counter() - start) / steps)

    dy = jnp.ones_like(out, dtype=jnp.float32)
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    with jax.set_mesh(mesh):
        return_route_indices = None
        if backward_stop_after_stage == BACKWARD_STOP_AFTER_NONE:
            resolved_w13_for_return_indices = _resolve_backward_stage_implementation(
                backward_w13_implementation,
                source_push_mlp._source_push_mlp_backward_w13_implementation(
                    SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU
                ),
            )
            if resolved_w13_for_return_indices in (
                SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_PALLAS_MGPU_COMPACT,
                SOURCE_PUSH_W13_BACKWARD_EXPERIMENT_COMPACT_DX_SOURCE_GATHER_DW13,
                SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_COMPACT_DX_ONLY,
            ):
                return_route_indices = source_push_backward_return_route_indices_jax(
                    host_inputs.plan,
                    src_base_by_expert=host_inputs.src_base_by_expert,
                )
            else:
                return_route_indices = source_push_backward_return_flat_route_indices_jax(
                    host_inputs.plan,
                    expert_base=expert_base,
                    src_base_by_expert=host_inputs.src_base_by_expert,
                )
            _block_until_ready(return_route_indices)
        timing = _time_source_push_backward_staged_flat(
            config,
            host_inputs,
            route_table,
            expert_base,
            inputs["x_source"],
            h_route_weights,
            inputs["w13_source"],
            inputs["w2_source"],
            h_flat,
            dy,
            return_route_indices=return_route_indices,
            mesh=mesh,
            warmup=warmup,
            steps=steps,
            repeat_runs=repeat_runs,
            backward_dy_route_implementation=backward_dy_route_implementation,
            backward_w2_implementation=backward_w2_implementation,
            backward_w2_split_timing=backward_w2_split_timing,
            backward_w13_implementation=backward_w13_implementation,
            backward_return_implementation=backward_return_implementation,
            backward_stop_after_stage=backward_stop_after_stage,
        )
    stages = _staged_flat_backward_stages(backward_stop_after_stage)
    if backward_w2_split_timing and BACKWARD_STAGE_W2 in stages:
        stages = (*stages, *BACKWARD_W2_SPLIT_STAGES)
    rows = _decomposed_backward_rows(
        config,
        timing=timing,
        queue_stats=host_inputs.queue_stats,
        repeat_runs=repeat_runs,
        dropped_routes=int(jax.device_get(dropped_routes)),
        forward_h_first_call_time=forward_h_first_call_time,
        forward_h_times=forward_h_times,
        mode=MODE_BACKWARD_STAGED_FLAT,
        stages=stages,
        w13_backward_component=_w13_backward_component(backward_w13_implementation),
    )
    resolved_w2_implementation, resolved_w2_matmul_implementation, resolved_w2_swiglu_implementation = (
        _resolve_w2_backward_implementations(
            backward_w2_implementation,
            source_push_mlp._source_push_mlp_backward_w2_implementation(SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU),
        )
    )
    if backward_w2_implementation == BACKWARD_IMPLEMENTATION_DEFAULT:
        resolved_w2_matmul_implementation = source_push_mlp._source_push_mlp_backward_w2_matmul_implementation(
            SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU
        )
        resolved_w2_swiglu_implementation = source_push_mlp._source_push_mlp_backward_w2_swiglu_implementation(
            SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU
        )
    for row in rows:
        row["backward_dy_route_implementation"] = backward_dy_route_implementation
        row["backward_w2_implementation"] = backward_w2_implementation
        row["resolved_backward_w2_implementation"] = resolved_w2_implementation
        row["resolved_backward_w2_matmul_implementation"] = resolved_w2_matmul_implementation
        row["resolved_backward_w2_swiglu_implementation"] = resolved_w2_swiglu_implementation
        row["backward_w13_implementation"] = backward_w13_implementation
        row["w13_backward_component"] = _row_w13_backward_component(row, backward_w13_implementation)
        row["backward_return_implementation"] = backward_return_implementation
        row["backward_stop_after_stage"] = backward_stop_after_stage
    return rows


def _run_source_push_backward_staged_blocks(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    route_table,
    inputs: dict[str, jax.Array],
    warmup: int,
    steps: int,
    repeat_runs: int,
    backward_dy_route_implementation: str,
    backward_w2_implementation: str,
    backward_w13_implementation: str,
    backward_return_implementation: str,
    backward_stop_after_stage: str,
) -> list[dict[str, Any]]:
    """Benchmark W2 backward with expert-block H/dy inputs instead of flat-H reconstruction."""

    if backward_stop_after_stage not in BACKWARD_STOP_AFTER_STAGES:
        raise ValueError(
            f"{MODE_BACKWARD_STAGED_BLOCKS!r} supports stop-after {BACKWARD_STOP_AFTER_STAGES}, "
            f"got {backward_stop_after_stage!r}"
        )

    packed_w13_h_inputs, store_capacity, live_capacity = _prepack_w13_h_expert_major_inputs(
        config,
        mesh,
        host_inputs,
        inputs["x_source"],
        inputs["w13_source"],
        inputs["w2_source"],
    )

    def call_forward_residuals():
        h_blocks = _call_source_push_w13_h_expert_major_device_inputs(
            mesh,
            config,
            packed_w13_h_inputs,
            store_capacity=store_capacity,
            live_capacity=live_capacity,
        )
        _block_until_ready(h_blocks)
        return h_blocks, host_inputs.plan.dropped_routes

    with jax.set_mesh(mesh):
        expert_route_weights = source_push_mlp._source_push_mlp_route_weights_to_all_expert_major(
            route_table,
            inputs["combine_source"],
        )
        _block_until_ready(expert_route_weights)

    with jax.set_mesh(mesh):
        forward_start = time.perf_counter()
        h_blocks, dropped_routes = call_forward_residuals()
        forward_h_first_call_time = time.perf_counter() - forward_start

        for _ in range(warmup):
            h_blocks, dropped_routes = call_forward_residuals()

        forward_h_times = []
        for _ in range(repeat_runs):
            start = time.perf_counter()
            for _ in range(steps):
                h_blocks, dropped_routes = call_forward_residuals()
            forward_h_times.append((time.perf_counter() - start) / steps)

    dy = jax.device_put(
        jnp.ones((config.ep_size, config.tokens_per_rank, config.hidden_dim), dtype=jnp.float32),
        NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None)),
    )
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    with jax.set_mesh(mesh):
        return_route_indices = None
        if backward_stop_after_stage == BACKWARD_STOP_AFTER_NONE:
            resolved_w13_for_return_indices = _resolve_backward_stage_implementation(
                backward_w13_implementation,
                source_push_mlp._source_push_mlp_backward_w13_implementation(
                    SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU
                ),
            )
            if resolved_w13_for_return_indices in (
                SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_TILED,
                SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_PALLAS_MGPU_COMPACT,
                SOURCE_PUSH_W13_BACKWARD_EXPERIMENT_COMPACT_DX_SOURCE_GATHER_DW13,
                SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_COMPACT_DX_ONLY,
                BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_LOCAL_SWIGLU_DX13_DW13,
                BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_DX13_XLA_LOCAL_SWIGLU_DW13,
                BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_DX13_SPLIT_LOCAL_SWIGLU_DW13,
                BACKWARD_W13_IMPLEMENTATION_XLA_DX13_XLA_LOCAL_SWIGLU_DW13,
                BACKWARD_W13_IMPLEMENTATION_XLA_DX13_ROUTE_BUFFER_XLA_LOCAL_SWIGLU_DW13,
                BACKWARD_W13_IMPLEMENTATION_XLA_DX13_SOURCE_GATHER_DW13,
                BACKWARD_W13_IMPLEMENTATION_XLA_DX13_PALLAS_X_REMAT_XLA_LOCAL_SWIGLU_DW13,
            ):
                return_route_indices = source_push_backward_return_route_indices_jax(
                    host_inputs.plan,
                    src_base_by_expert=host_inputs.src_base_by_expert,
                )
            else:
                return_route_indices = source_push_backward_return_flat_route_indices_jax(
                    host_inputs.plan,
                    expert_base=expert_base,
                    src_base_by_expert=host_inputs.src_base_by_expert,
                )
            _block_until_ready(return_route_indices)
        timing = _time_source_push_backward_staged_blocks(
            config,
            host_inputs,
            route_table,
            expert_base,
            inputs["x_source"],
            expert_route_weights,
            inputs["w13_source"],
            inputs["w2_source"],
            h_blocks,
            dy,
            mesh=mesh,
            warmup=warmup,
            steps=steps,
            repeat_runs=repeat_runs,
            backward_dy_route_implementation=backward_dy_route_implementation,
            backward_w2_implementation=backward_w2_implementation,
            backward_w13_implementation=backward_w13_implementation,
            backward_return_implementation=backward_return_implementation,
            backward_stop_after_stage=backward_stop_after_stage,
            return_route_indices=return_route_indices,
        )
    resolved_w13_implementation = _resolve_backward_stage_implementation(
        backward_w13_implementation,
        source_push_mlp._source_push_mlp_backward_w13_implementation(SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU),
    )
    stages = _backward_staged_block_timed_stages(backward_stop_after_stage, resolved_w13_implementation)
    rows = _decomposed_backward_rows(
        config,
        timing=timing,
        queue_stats=host_inputs.queue_stats,
        repeat_runs=repeat_runs,
        dropped_routes=int(jax.device_get(dropped_routes)),
        forward_h_first_call_time=forward_h_first_call_time,
        forward_h_times=forward_h_times,
        mode=MODE_BACKWARD_STAGED_BLOCKS,
        stages=stages,
        w13_backward_component=_w13_backward_component(backward_w13_implementation),
    )
    resolved_w2_implementation, resolved_w2_matmul_implementation, resolved_w2_swiglu_implementation = (
        _resolve_w2_backward_implementations(
            backward_w2_implementation,
            source_push_mlp._source_push_mlp_backward_w2_implementation(SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU),
        )
    )
    if backward_w2_implementation == BACKWARD_IMPLEMENTATION_DEFAULT:
        resolved_w2_matmul_implementation = source_push_mlp._source_push_mlp_backward_w2_matmul_implementation(
            SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU
        )
        resolved_w2_swiglu_implementation = source_push_mlp._source_push_mlp_backward_w2_swiglu_implementation(
            SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU
        )
    for row in rows:
        row["backward_dy_route_implementation"] = backward_dy_route_implementation
        row["backward_w2_implementation"] = backward_w2_implementation
        row["resolved_backward_w2_implementation"] = resolved_w2_implementation
        row["resolved_backward_w2_matmul_implementation"] = resolved_w2_matmul_implementation
        row["resolved_backward_w2_swiglu_implementation"] = resolved_w2_swiglu_implementation
        row["backward_w13_implementation"] = backward_w13_implementation
        row["w13_backward_component"] = _row_w13_backward_component(row, backward_w13_implementation)
        row["backward_return_implementation"] = backward_return_implementation
        row["backward_stop_after_stage"] = backward_stop_after_stage
    return rows


def _run_source_push_backward_w13_only(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    route_table,
    inputs: dict[str, jax.Array],
    warmup: int,
    steps: int,
    repeat_runs: int,
    backward_w13_implementation: str,
    backward_w13_block_sizes: SourcePushW13BackwardTiledBlockSizes | None,
    backward_w13_lowering_semantics: str,
) -> list[dict[str, Any]]:
    """Benchmark W13 backward in isolation from forward H, dy routing, and W2."""

    w13_implementation = _resolve_backward_stage_implementation(
        backward_w13_implementation,
        source_push_mlp._source_push_mlp_backward_w13_implementation(SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU),
    )
    w13_block_sizes = backward_w13_block_sizes
    resolved_w13_block_sizes = _resolved_w13_only_block_sizes(w13_implementation, w13_block_sizes)
    w13_lowering_semantics = _source_push_w13_lowering_semantics_from_arg(backward_w13_lowering_semantics)
    compact_capacity = route_table.valid_by_expert.shape[-1]
    output_dim = 2 * config.intermediate_dim
    compact_block_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None))
    compact_meta_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    d_h_blocks = jax.device_put(
        np.ones(
            (config.ep_size, config.experts_per_rank, compact_capacity, output_dim),
            dtype=inputs["w13_source"].dtype,
        ),
        compact_block_sharding,
    )
    z_blocks = jax.device_put(
        np.ones(
            (config.ep_size, config.experts_per_rank, compact_capacity, output_dim),
            dtype=inputs["w13_source"].dtype,
        ),
        compact_block_sharding,
    )
    d_activation_blocks = jax.device_put(
        np.ones(
            (config.ep_size, config.experts_per_rank, compact_capacity, config.intermediate_dim),
            dtype=inputs["w13_source"].dtype,
        ),
        compact_block_sharding,
    )
    valid = jax.device_put(route_table.valid_by_expert, compact_meta_sharding)
    source_rank_by_expert = jax.device_put(route_table.source_rank_by_expert, compact_meta_sharding)
    token_id_by_expert = jax.device_put(route_table.token_id_by_expert, compact_meta_sharding)
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    prefilled_x_expert_major = None
    if w13_implementation in (
        SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_PREFILLED_X_DW13_ONLY,
        SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_DW13_ONLY,
        SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_PERSISTENT_DW13_ONLY,
        SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_LINEAR_DW13_ONLY,
        SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_GATE_DW13_ONLY,
        SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_UP_DW13_ONLY,
        SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_SPLIT_DW13_ONLY,
        SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_XLA_LOCAL_SWIGLU_DW13_ONLY,
    ):
        safe_src = jnp.where(valid, source_rank_by_expert, 0)
        safe_token = jnp.where(valid, token_id_by_expert, 0)
        prefilled_x_expert_major = inputs["x_source"].at[safe_src, safe_token].get(out_sharding=compact_block_sharding)
        prefilled_x_expert_major = jnp.where(
            valid[..., None],
            prefilled_x_expert_major,
            jnp.zeros_like(prefilled_x_expert_major),
        )
        _block_until_ready(prefilled_x_expert_major)

    def call_w13():
        with jax.set_mesh(mesh):
            if w13_implementation == SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_TILED:
                w13_grads = source_push_w13_backward_expert_blocks_tiled_reference(
                    inputs["x_source"],
                    d_h_blocks,
                    inputs["w13_source"],
                    source_rank_by_expert,
                    token_id_by_expert,
                    valid,
                    block_sizes=resolved_w13_block_sizes,
                )
            elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_PALLAS_MGPU_COMPACT:
                w13_grads = _source_push_w13_backward_expert_blocks_pallas_mgpu(
                    inputs["x_source"],
                    d_h_blocks,
                    inputs["w13_source"],
                    source_rank_by_expert,
                    token_id_by_expert,
                    valid,
                    block_sizes=resolved_w13_block_sizes,
                    mesh=mesh,
                )
            elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_EXPERIMENT_COMPACT_DX_SOURCE_GATHER_DW13:
                w13_grads = _source_push_w13_backward_expert_blocks_compact_dx_source_gather_dw13(
                    inputs["x_source"],
                    d_h_blocks,
                    inputs["w13_source"],
                    source_rank_by_expert,
                    token_id_by_expert,
                    valid,
                    block_sizes=resolved_w13_block_sizes,
                    mesh=mesh,
                )
            elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_COMPACT_DX_ONLY:
                w13_grads = _source_push_w13_backward_expert_blocks_dx_only_pallas_mgpu(
                    d_h_blocks,
                    inputs["w13_source"],
                    valid,
                    block_sizes=resolved_w13_block_sizes,
                    mesh=mesh,
                )
            elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_COMPACT_DW13_ONLY:
                w13_grads = _source_push_w13_backward_expert_blocks_dw13_only_pallas_mgpu(
                    inputs["x_source"],
                    d_h_blocks,
                    inputs["w13_source"],
                    source_rank_by_expert,
                    token_id_by_expert,
                    valid,
                    block_sizes=w13_block_sizes,
                    lowering_semantics=w13_lowering_semantics,
                    mesh=mesh,
                )
            elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_PREFILLED_X_DW13_ONLY:
                if prefilled_x_expert_major is None:
                    raise ValueError("prefilled x_expert_major was not prepared")
                w13_grads = _source_push_w13_backward_expert_blocks_prefilled_x_dw13_only_pallas_mgpu(
                    prefilled_x_expert_major,
                    d_h_blocks,
                    inputs["w13_source"],
                    valid,
                    block_sizes=w13_block_sizes,
                    lowering_semantics=w13_lowering_semantics,
                    mesh=mesh,
                )
            elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_EXACT_FLAT_DW13_ONLY:
                w13_grads = _source_push_w13_backward_expert_blocks_dw13_only_exact_flat_pallas_mgpu(
                    inputs["x_source"],
                    d_h_blocks,
                    inputs["w13_source"],
                    source_rank_by_expert,
                    token_id_by_expert,
                    valid,
                    block_sizes=resolved_w13_block_sizes,
                    mesh=mesh,
                )
            elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_DW13_ONLY:
                if prefilled_x_expert_major is None:
                    raise ValueError("prefilled x_expert_major was not prepared")
                w13_grads = _source_push_w13_backward_expert_blocks_local_swiglu_dw13_only_pallas_mgpu(
                    prefilled_x_expert_major,
                    d_activation_blocks,
                    z_blocks,
                    valid,
                    block_sizes=resolved_w13_block_sizes,
                    lowering_semantics=w13_lowering_semantics,
                    mesh=mesh,
                )
            elif (
                w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_PERSISTENT_DW13_ONLY
            ):
                if prefilled_x_expert_major is None:
                    raise ValueError("prefilled x_expert_major was not prepared")
                w13_grads = _source_push_w13_backward_expert_blocks_local_swiglu_persistent_dw13_only_pallas_mgpu(
                    prefilled_x_expert_major,
                    d_activation_blocks,
                    z_blocks,
                    valid,
                    block_sizes=resolved_w13_block_sizes,
                    lowering_semantics=w13_lowering_semantics,
                    mesh=mesh,
                )
            elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_LINEAR_DW13_ONLY:
                if prefilled_x_expert_major is None:
                    raise ValueError("prefilled x_expert_major was not prepared")
                w13_grads = _source_push_w13_backward_expert_blocks_local_linear_dw13_only_pallas_mgpu(
                    prefilled_x_expert_major,
                    d_activation_blocks,
                    z_blocks,
                    valid,
                    block_sizes=resolved_w13_block_sizes,
                    lowering_semantics=w13_lowering_semantics,
                    mesh=mesh,
                )
            elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_XLA_LOCAL_SWIGLU_DW13_ONLY:
                if prefilled_x_expert_major is None:
                    raise ValueError("prefilled x_expert_major was not prepared")
                w13_grads = source_push_w13_backward_expert_blocks_local_swiglu_dw13_only_xla(
                    prefilled_x_expert_major,
                    d_activation_blocks,
                    z_blocks,
                    valid,
                )
            elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_SPLIT_DW13_ONLY:
                if prefilled_x_expert_major is None:
                    raise ValueError("prefilled x_expert_major was not prepared")
                w13_grads = _source_push_w13_backward_expert_blocks_local_swiglu_split_dw13_only_pallas_mgpu(
                    prefilled_x_expert_major,
                    d_activation_blocks,
                    z_blocks,
                    valid,
                    block_sizes=resolved_w13_block_sizes,
                    lowering_semantics=w13_lowering_semantics,
                    mesh=mesh,
                )
            elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_GATE_DW13_ONLY:
                if prefilled_x_expert_major is None:
                    raise ValueError("prefilled x_expert_major was not prepared")
                w13_grads = _source_push_w13_backward_expert_blocks_local_swiglu_gate_dw13_only_pallas_mgpu(
                    prefilled_x_expert_major,
                    d_activation_blocks,
                    z_blocks,
                    valid,
                    block_sizes=resolved_w13_block_sizes,
                    lowering_semantics=w13_lowering_semantics,
                    mesh=mesh,
                )
            elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_UP_DW13_ONLY:
                if prefilled_x_expert_major is None:
                    raise ValueError("prefilled x_expert_major was not prepared")
                w13_grads = _source_push_w13_backward_expert_blocks_local_swiglu_up_dw13_only_pallas_mgpu(
                    prefilled_x_expert_major,
                    d_activation_blocks,
                    z_blocks,
                    valid,
                    block_sizes=resolved_w13_block_sizes,
                    lowering_semantics=w13_lowering_semantics,
                    mesh=mesh,
                )
            elif w13_implementation in (
                BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_LOCAL_SWIGLU_DX13_DW13,
                BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_DX13_XLA_LOCAL_SWIGLU_DW13,
                BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_DX13_SPLIT_LOCAL_SWIGLU_DW13,
                BACKWARD_W13_IMPLEMENTATION_XLA_DX13_XLA_LOCAL_SWIGLU_DW13,
                BACKWARD_W13_IMPLEMENTATION_XLA_DX13_ROUTE_BUFFER_XLA_LOCAL_SWIGLU_DW13,
                BACKWARD_W13_IMPLEMENTATION_XLA_DX13_SOURCE_GATHER_DW13,
                BACKWARD_W13_IMPLEMENTATION_XLA_DX13_PALLAS_X_REMAT_XLA_LOCAL_SWIGLU_DW13,
            ):
                raise ValueError(
                    f"{w13_implementation!r} requires real W2 "
                    f"`d_activation` and saved W13 preactivation; use {MODE_BACKWARD_STAGED_BLOCKS!r}."
                )
            elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_XLA_COMPACT_DW13_ONLY:
                w13_grads = source_push_w13_backward_expert_blocks_dw13_only_xla(
                    inputs["x_source"],
                    d_h_blocks,
                    inputs["w13_source"],
                    source_rank_by_expert,
                    token_id_by_expert,
                    valid,
                    block_sizes=resolved_w13_block_sizes,
                )
            elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_XLA_SOURCE_PADDED_DW13_ONLY:
                w13_grads = source_push_w13_backward_expert_blocks_source_padded_dw13_only_xla(
                    inputs["x_source"],
                    d_h_blocks,
                    source_rank_by_expert,
                    token_id_by_expert,
                    valid,
                    host_inputs.src_base_by_expert,
                    block_sizes=resolved_w13_block_sizes,
                )
            elif (
                w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_SOURCE_PADDED_PARTIALS_DW13_ONLY
            ):
                partials = _source_push_w13_dw13_source_padded_partials_pallas_mgpu(
                    inputs["x_source"],
                    d_h_blocks,
                    source_rank_by_expert,
                    token_id_by_expert,
                    valid,
                    host_inputs.src_base_by_expert,
                    block_sizes=resolved_w13_block_sizes,
                    mesh=mesh,
                )
                w13_grads = SourcePushW13CompactBackwardOutput(
                    x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
                    dx_expert_major=jnp.zeros((0,), dtype=jnp.float32),
                    dw13=jnp.sum(partials, axis=0),
                )
            elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_SOURCE_GATHER_DW13_ONLY:
                w13_grads = source_push_w13_backward_expert_blocks_source_gather_dw13_only(
                    inputs["x_source"],
                    d_h_blocks,
                    source_rank_by_expert,
                    token_id_by_expert,
                    valid,
                    block_sizes=resolved_w13_block_sizes,
                )
            else:
                d_h_flat = _flatten_expert_blocks_to_flat_rows(
                    expert_base,
                    d_h_blocks,
                    flat_rows_per_rank=config.hidden_rows_per_rank,
                    valid=valid,
                )
                w13_grads = source_push_w13_backward(
                    inputs["x_source"],
                    d_h_flat,
                    inputs["w13_source"],
                    host_inputs.plan,
                    host_inputs.send_meta,
                    expert_base,
                    host_inputs.src_base_by_expert,
                    use_exact_expert_major=host_inputs.use_exact_expert_major,
                    implementation=w13_implementation,
                    mesh=mesh,
                )
        _block_until_ready(w13_grads)
        return w13_grads

    start = time.perf_counter()
    output = call_w13()
    first_call_time = time.perf_counter() - start

    for _ in range(warmup):
        output = call_w13()

    steady_state_times = []
    for _ in range(repeat_runs):
        start = time.perf_counter()
        for _ in range(steps):
            output = call_w13()
        steady_state_times.append((time.perf_counter() - start) / steps)

    _block_until_ready(output)
    stage_useful_flops, stage_rounded_flops = _backward_stage_flops_per_rank(
        config,
        host_inputs.queue_stats,
        BACKWARD_STAGE_W13,
        w13_backward_component=_w13_backward_component(w13_implementation),
    )
    dropped_routes = int(jax.device_get(host_inputs.plan.dropped_routes))
    rows = [
        _decomposed_backward_row(
            config,
            queue_stats=host_inputs.queue_stats,
            repeat_run=repeat_run,
            repeat_runs=repeat_runs,
            stage=BACKWARD_STAGE_W13,
            steady_state_time=steady_state_time,
            first_call_time=first_call_time,
            useful_backward_flops=stage_useful_flops,
            rounded_backward_flops=stage_rounded_flops,
            dropped_routes=dropped_routes,
            mode=MODE_BACKWARD_W13_ONLY,
        )
        for repeat_run, steady_state_time in enumerate(steady_state_times)
    ]
    rows.append(_summary_row(rows))
    for row in rows:
        row["backward_w13_implementation"] = backward_w13_implementation
        row["resolved_backward_w13_implementation"] = w13_implementation
        row["backward_w13_row_block"] = resolved_w13_block_sizes.row_block
        row["backward_w13_hidden_block"] = resolved_w13_block_sizes.hidden_block
        row["backward_w13_output_block"] = resolved_w13_block_sizes.output_block
        row["backward_w13_lowering_semantics"] = backward_w13_lowering_semantics
        row["w13_backward_component"] = _row_w13_backward_component(row, w13_implementation)
        row["backward_stop_after_stage"] = BACKWARD_STAGE_W13
    return rows


def _run_source_push_backward_dx13_only(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    route_table,
    inputs: dict[str, jax.Array],
    warmup: int,
    steps: int,
    repeat_runs: int,
    backward_dx13_implementation: str,
) -> list[dict[str, Any]]:
    """Benchmark local DX13 compact contribution generation in isolation.

    This covers the structural target from the backward notes:
    recompute dSwiGLU from local W2 activation gradients plus saved W13
    preactivation, multiply by ``W13.T``, and return compact contribution rows
    with source-return metadata.  It deliberately excludes source-side combine.
    """

    compact_capacity = route_table.valid_by_expert.shape[-1]
    compact_block_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None))
    compact_meta_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    block_sizes = source_push_dx13_pallas_resolved_block_sizes(SourcePushDx13PallasBlockSizes.get_default())
    d_activation_blocks = jax.device_put(
        np.ones(
            (config.ep_size, config.experts_per_rank, compact_capacity, config.intermediate_dim),
            dtype=inputs["w13_source"].dtype,
        ),
        compact_block_sharding,
    )
    z_blocks = jax.device_put(
        np.ones(
            (config.ep_size, config.experts_per_rank, compact_capacity, 2 * config.intermediate_dim),
            dtype=inputs["w13_source"].dtype,
        ),
        compact_block_sharding,
    )
    valid = jax.device_put(route_table.valid_by_expert, compact_meta_sharding)
    source_rank_by_expert = jax.device_put(route_table.source_rank_by_expert, compact_meta_sharding)
    token_id_by_expert = jax.device_put(route_table.token_id_by_expert, compact_meta_sharding)
    route_slot_by_expert = jax.device_put(route_table.route_slot_by_expert, compact_meta_sharding)

    def call_dx13():
        with jax.set_mesh(mesh):
            if backward_dx13_implementation == SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU:
                output = source_push_dx13_push_compact(
                    d_activation_blocks,
                    z_blocks,
                    inputs["w13_source"],
                    source_rank_by_expert,
                    token_id_by_expert,
                    route_slot_by_expert,
                    valid,
                    implementation=SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU,
                    block_sizes=block_sizes,
                    mesh=mesh,
                )
            elif backward_dx13_implementation == BACKWARD_DX13_IMPLEMENTATION_XLA_EXPERT_MAJOR:
                output = source_push_dx13_push_compact_xla(
                    d_activation_blocks,
                    z_blocks,
                    inputs["w13_source"],
                    source_rank_by_expert,
                    token_id_by_expert,
                    route_slot_by_expert,
                    valid,
                )
            else:
                raise ValueError(f"unsupported backward DX13 implementation {backward_dx13_implementation!r}")
        _block_until_ready(output)
        return output

    start = time.perf_counter()
    output = call_dx13()
    first_call_time = time.perf_counter() - start

    for _ in range(warmup):
        output = call_dx13()

    steady_state_times = []
    for _ in range(repeat_runs):
        start = time.perf_counter()
        for _ in range(steps):
            output = call_dx13()
        steady_state_times.append((time.perf_counter() - start) / steps)

    _block_until_ready(output)
    stage_useful_flops, stage_rounded_flops = _backward_stage_flops_per_rank(
        config,
        host_inputs.queue_stats,
        BACKWARD_STAGE_DX13_PUSH,
    )
    dropped_routes = int(jax.device_get(host_inputs.plan.dropped_routes))
    rows = [
        _decomposed_backward_row(
            config,
            queue_stats=host_inputs.queue_stats,
            repeat_run=repeat_run,
            repeat_runs=repeat_runs,
            stage=BACKWARD_STAGE_DX13_PUSH,
            steady_state_time=steady_state_time,
            first_call_time=first_call_time,
            useful_backward_flops=stage_useful_flops,
            rounded_backward_flops=stage_rounded_flops,
            dropped_routes=dropped_routes,
            mode=MODE_BACKWARD_DX13_ONLY,
        )
        for repeat_run, steady_state_time in enumerate(steady_state_times)
    ]
    rows.append(_summary_row(rows))
    for row in rows:
        row["backward_dx13_implementation"] = backward_dx13_implementation
        row["backward_dx13_boundary"] = (
            "xla_expert_major_materialization"
            if backward_dx13_implementation == BACKWARD_DX13_IMPLEMENTATION_XLA_EXPERT_MAJOR
            else "expert_major_materialization_with_source_route_buffer_contract"
        )
        row["backward_dx13_row_block"] = block_sizes.row_block
        row["backward_dx13_hidden_block"] = block_sizes.hidden_block
        row["backward_dx13_output_block"] = block_sizes.output_block
        row["backward_stop_after_stage"] = BACKWARD_STAGE_DX13_PUSH
    return rows


def _run_source_push_backward_dx13_store_zero_only(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    route_table,
    inputs: dict[str, jax.Array],
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> list[dict[str, Any]]:
    """Benchmark the DX13 expert-major output write/allocation floor."""

    compact_capacity = route_table.valid_by_expert.shape[-1]
    compact_block_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None))
    compact_meta_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    block_sizes = source_push_dx13_pallas_resolved_block_sizes(SourcePushDx13PallasBlockSizes.get_default())
    d_activation_blocks = jax.device_put(
        np.ones(
            (config.ep_size, config.experts_per_rank, compact_capacity, config.intermediate_dim),
            dtype=inputs["w13_source"].dtype,
        ),
        compact_block_sharding,
    )
    z_blocks = jax.device_put(
        np.ones(
            (config.ep_size, config.experts_per_rank, compact_capacity, 2 * config.intermediate_dim),
            dtype=inputs["w13_source"].dtype,
        ),
        compact_block_sharding,
    )
    valid = jax.device_put(route_table.valid_by_expert, compact_meta_sharding)

    def call_dx13_store_zero():
        with jax.set_mesh(mesh):
            output = source_push_dx13_expert_major_store_zero_pallas_mgpu(
                d_activation_blocks,
                z_blocks,
                inputs["w13_source"],
                valid,
                block_sizes=block_sizes,
                mesh=mesh,
            )
        _block_until_ready(output)
        return output

    start = time.perf_counter()
    output = call_dx13_store_zero()
    first_call_time = time.perf_counter() - start

    for _ in range(warmup):
        output = call_dx13_store_zero()

    steady_state_times = []
    for _ in range(repeat_runs):
        start = time.perf_counter()
        for _ in range(steps):
            output = call_dx13_store_zero()
        steady_state_times.append((time.perf_counter() - start) / steps)

    _block_until_ready(output)
    dropped_routes = int(jax.device_get(host_inputs.plan.dropped_routes))
    output_bytes_per_rank = output.shape[1] * output.shape[2] * output.shape[3] * output.dtype.itemsize
    rows = [
        _decomposed_backward_row(
            config,
            queue_stats=host_inputs.queue_stats,
            repeat_run=repeat_run,
            repeat_runs=repeat_runs,
            stage=BACKWARD_STAGE_DX13_PUSH,
            steady_state_time=steady_state_time,
            first_call_time=first_call_time,
            useful_backward_flops=0.0,
            rounded_backward_flops=0.0,
            dropped_routes=dropped_routes,
            mode=MODE_BACKWARD_DX13_STORE_ZERO_ONLY,
        )
        for repeat_run, steady_state_time in enumerate(steady_state_times)
    ]
    rows.append(_summary_row(rows))
    for row in rows:
        steady = row.get("steady_state_time") or row.get("median_steady_state_time")
        row["backward_dx13_implementation"] = SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU
        row["backward_dx13_boundary"] = "expert_major_store_zero"
        row["backward_dx13_row_block"] = block_sizes.row_block
        row["backward_dx13_hidden_block"] = block_sizes.hidden_block
        row["backward_dx13_output_block"] = block_sizes.output_block
        row["backward_stop_after_stage"] = BACKWARD_STAGE_DX13_PUSH
        row["dx13_output_bytes_per_rank"] = output_bytes_per_rank
        row["dx13_output_gbps_per_rank"] = None if steady is None else output_bytes_per_rank / steady / 1e9
    return rows


def _run_source_push_backward_dx13_route_buffer_only(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    route_table,
    inputs: dict[str, jax.Array],
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> list[dict[str, Any]]:
    """Benchmark DX13 direct source route-buffer contribution generation."""

    compact_capacity = route_table.valid_by_expert.shape[-1]
    compact_block_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None))
    compact_meta_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    block_sizes = source_push_dx13_pallas_resolved_block_sizes(SourcePushDx13PallasBlockSizes.get_default())
    d_activation_blocks = jax.device_put(
        np.ones(
            (config.ep_size, config.experts_per_rank, compact_capacity, config.intermediate_dim),
            dtype=inputs["w13_source"].dtype,
        ),
        compact_block_sharding,
    )
    z_blocks = jax.device_put(
        np.ones(
            (config.ep_size, config.experts_per_rank, compact_capacity, 2 * config.intermediate_dim),
            dtype=inputs["w13_source"].dtype,
        ),
        compact_block_sharding,
    )
    valid = jax.device_put(route_table.valid_by_expert, compact_meta_sharding)
    source_rank_by_expert = jax.device_put(route_table.source_rank_by_expert, compact_meta_sharding)
    token_id_by_expert = jax.device_put(route_table.token_id_by_expert, compact_meta_sharding)
    route_slot_by_expert = jax.device_put(route_table.route_slot_by_expert, compact_meta_sharding)

    def call_dx13_route_buffer():
        with jax.set_mesh(mesh):
            output = source_push_dx13_push_route_buffer(
                d_activation_blocks,
                z_blocks,
                inputs["w13_source"],
                source_rank_by_expert,
                token_id_by_expert,
                route_slot_by_expert,
                valid,
                tokens_per_source=config.tokens_per_rank,
                topk=config.topk,
                implementation=SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU,
                block_sizes=block_sizes,
                mesh=mesh,
            )
        _block_until_ready(output)
        return output

    start = time.perf_counter()
    output = call_dx13_route_buffer()
    first_call_time = time.perf_counter() - start

    for _ in range(warmup):
        output = call_dx13_route_buffer()

    steady_state_times = []
    for _ in range(repeat_runs):
        start = time.perf_counter()
        for _ in range(steps):
            output = call_dx13_route_buffer()
        steady_state_times.append((time.perf_counter() - start) / steps)

    _block_until_ready(output)
    stage_useful_flops, stage_rounded_flops = _backward_stage_flops_per_rank(
        config,
        host_inputs.queue_stats,
        BACKWARD_STAGE_DX13_PUSH,
    )
    dropped_routes = int(jax.device_get(host_inputs.plan.dropped_routes))
    rows = [
        _decomposed_backward_row(
            config,
            queue_stats=host_inputs.queue_stats,
            repeat_run=repeat_run,
            repeat_runs=repeat_runs,
            stage=BACKWARD_STAGE_DX13_PUSH,
            steady_state_time=steady_state_time,
            first_call_time=first_call_time,
            useful_backward_flops=stage_useful_flops,
            rounded_backward_flops=stage_rounded_flops,
            dropped_routes=dropped_routes,
            mode=MODE_BACKWARD_DX13_ROUTE_BUFFER_ONLY,
        )
        for repeat_run, steady_state_time in enumerate(steady_state_times)
    ]
    rows.append(_summary_row(rows))
    for row in rows:
        row["backward_dx13_implementation"] = SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU
        row["backward_dx13_boundary"] = "direct_source_route_buffer_remote_write"
        row["backward_dx13_row_block"] = block_sizes.row_block
        row["backward_dx13_hidden_block"] = block_sizes.hidden_block
        row["backward_dx13_output_block"] = block_sizes.output_block
        row["backward_stop_after_stage"] = BACKWARD_STAGE_DX13_PUSH
    return rows


def _run_source_push_backward_dx13_source_compact_only(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    route_table,
    inputs: dict[str, jax.Array],
    warmup: int,
    steps: int,
    repeat_runs: int,
    block_contiguous: bool = False,
    mode: str = MODE_BACKWARD_DX13_SOURCE_COMPACT_ONLY,
) -> list[dict[str, Any]]:
    """Benchmark expert-owner DX13 plus source-compact contribution return."""

    compact_capacity = route_table.valid_by_expert.shape[-1]
    compact_block_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None))
    compact_meta_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    if block_contiguous:
        block_defaults = SourcePushDx13PallasBlockSizes.get_default()
        block_sizes = SourcePushDx13PallasBlockSizes(
            row_block=config.block_m,
            hidden_block=block_defaults.hidden_block,
            output_block=block_defaults.output_block,
        )
    else:
        block_sizes = source_push_dx13_pallas_resolved_block_sizes(SourcePushDx13PallasBlockSizes.get_default())
    d_activation_blocks = jax.device_put(
        np.ones(
            (config.ep_size, config.experts_per_rank, compact_capacity, config.intermediate_dim),
            dtype=inputs["w13_source"].dtype,
        ),
        compact_block_sharding,
    )
    z_blocks = jax.device_put(
        np.ones(
            (config.ep_size, config.experts_per_rank, compact_capacity, 2 * config.intermediate_dim),
            dtype=inputs["w13_source"].dtype,
        ),
        compact_block_sharding,
    )
    compact_slots = source_push_dx13_compact_assignment_slots_from_fields(
        jax.device_put(route_table.source_rank_by_expert, compact_meta_sharding),
        jax.device_put(route_table.dst_ordinal_by_expert, compact_meta_sharding),
        jax.device_put(route_table.entry_by_expert, compact_meta_sharding),
        jax.device_put(route_table.row_in_entry_by_expert, compact_meta_sharding),
        jax.device_put(route_table.valid_by_expert, compact_meta_sharding),
    )
    queue_shape = tuple(int(dim) for dim in host_inputs.plan.valid_mask.shape)

    def call_dx13_source_compact():
        with jax.set_mesh(mesh):
            if block_contiguous:
                output = source_push_dx13_push_contrib_block_contiguous_pallas_mgpu(
                    d_activation_blocks,
                    z_blocks,
                    inputs["w13_source"],
                    compact_slots,
                    queue_shape=queue_shape,
                    block_sizes=block_sizes,
                    mesh=mesh,
                )
            else:
                push_contrib = (
                    source_push_dx13_push_compact_contrib
                    if mode == MODE_BACKWARD_DX13_PUSH_CONTRIB_ONLY
                    else source_push_dx13_push_contrib
                )
                output = push_contrib(
                    d_activation_blocks,
                    z_blocks,
                    inputs["w13_source"],
                    compact_slots,
                    queue_shape=queue_shape,
                    implementation=SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU,
                    block_sizes=block_sizes,
                    mesh=mesh,
                )
        _block_until_ready(output)
        return output

    start = time.perf_counter()
    output = call_dx13_source_compact()
    first_call_time = time.perf_counter() - start

    for _ in range(warmup):
        output = call_dx13_source_compact()

    steady_state_times = []
    for _ in range(repeat_runs):
        start = time.perf_counter()
        for _ in range(steps):
            output = call_dx13_source_compact()
        steady_state_times.append((time.perf_counter() - start) / steps)

    _block_until_ready(output)
    stage_useful_flops, stage_rounded_flops = _backward_stage_flops_per_rank(
        config,
        host_inputs.queue_stats,
        BACKWARD_STAGE_DX13_PUSH,
    )
    dropped_routes = int(jax.device_get(host_inputs.plan.dropped_routes))
    output_bytes_per_rank = (
        output.dx_contrib.shape[1]
        * output.dx_contrib.shape[2]
        * output.dx_contrib.shape[3]
        * output.dx_contrib.shape[4]
        * output.dx_contrib.dtype.itemsize
    )
    rows = [
        _decomposed_backward_row(
            config,
            queue_stats=host_inputs.queue_stats,
            repeat_run=repeat_run,
            repeat_runs=repeat_runs,
            stage=BACKWARD_STAGE_DX13_PUSH,
            steady_state_time=steady_state_time,
            first_call_time=first_call_time,
            useful_backward_flops=stage_useful_flops,
            rounded_backward_flops=stage_rounded_flops,
            dropped_routes=dropped_routes,
            mode=(MODE_BACKWARD_DX13_SOURCE_COMPACT_BLOCK_ONLY if block_contiguous else mode),
        )
        for repeat_run, steady_state_time in enumerate(steady_state_times)
    ]
    rows.append(_summary_row(rows))
    for row in rows:
        steady = row.get("steady_state_time") or row.get("median_steady_state_time")
        row["backward_dx13_implementation"] = SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU
        row["backward_dx13_boundary"] = (
            "source_compact_queue_block_contiguous_remote_write"
            if block_contiguous
            else (
                "expert_owner_dx13_to_source_compact_contrib_remote_write"
                if mode == MODE_BACKWARD_DX13_PUSH_CONTRIB_ONLY
                else "source_compact_queue_remote_write"
            )
        )
        row["backward_dx13_block_contiguous"] = block_contiguous
        row["backward_dx13_row_block"] = block_sizes.row_block
        row["backward_dx13_hidden_block"] = block_sizes.hidden_block
        row["backward_dx13_output_block"] = block_sizes.output_block
        row["backward_stop_after_stage"] = BACKWARD_STAGE_DX13_PUSH
        row["source_compact_output_bytes_per_rank"] = output_bytes_per_rank
        row["source_compact_output_gbps_per_rank"] = None if steady is None else output_bytes_per_rank / steady / 1e9
    return rows


def _run_source_push_backward_dx13_source_compact_combine_only(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> list[dict[str, Any]]:
    """Benchmark source-local combine from source-compact DX13 contributions."""

    queue_shape = tuple(int(dim) for dim in host_inputs.plan.valid_mask.shape)
    dx_contrib = jax.device_put(
        jnp.ones((*queue_shape, config.hidden_dim), dtype=jnp.bfloat16),
        NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None, None)),
    )
    compact_output = SourcePushDx13SourceCompactOutput(dx_contrib=dx_contrib)
    _block_until_ready(compact_output)

    def call_route_buffer_then_sum():
        with jax.set_mesh(mesh):
            output = source_push_dx13_source_compact_to_route_buffer_reference(compact_output, host_inputs.plan)
        _block_until_ready(output)
        return output

    def call_direct_token_sum():
        with jax.set_mesh(mesh):
            output = source_push_dx13_source_compact_combine_reference(compact_output, host_inputs.plan)
        _block_until_ready(output)
        return output

    variants = (
        ("route_buffer_then_sum", call_route_buffer_then_sum),
        ("direct_token_sum", call_direct_token_sum),
    )
    rows: list[dict[str, Any]] = []
    dropped_routes = int(jax.device_get(host_inputs.plan.dropped_routes))
    input_bytes_per_rank = (
        dx_contrib.shape[1]
        * dx_contrib.shape[2]
        * dx_contrib.shape[3]
        * dx_contrib.shape[4]
        * dx_contrib.dtype.itemsize
    )
    for variant, call_combine in variants:
        start = time.perf_counter()
        output = call_combine()
        first_call_time = time.perf_counter() - start

        for _ in range(warmup):
            output = call_combine()

        steady_state_times = []
        for _ in range(repeat_runs):
            start = time.perf_counter()
            for _ in range(steps):
                output = call_combine()
            steady_state_times.append((time.perf_counter() - start) / steps)

        _block_until_ready(output)
        variant_rows = [
            _decomposed_backward_row(
                config,
                queue_stats=host_inputs.queue_stats,
                repeat_run=repeat_run,
                repeat_runs=repeat_runs,
                stage=BACKWARD_STAGE_DX_COMBINE,
                steady_state_time=steady_state_time,
                first_call_time=first_call_time,
                useful_backward_flops=None,
                rounded_backward_flops=None,
                dropped_routes=dropped_routes,
                mode=MODE_BACKWARD_DX13_SOURCE_COMPACT_COMBINE_ONLY,
            )
            for repeat_run, steady_state_time in enumerate(steady_state_times)
        ]
        variant_rows.append(_summary_row(variant_rows))
        for row in variant_rows:
            steady = row.get("steady_state_time") or row.get("median_steady_state_time")
            row["backward_dx13_boundary"] = "source_compact_queue_combine"
            row["source_compact_combine_variant"] = variant
            row["source_compact_input_bytes_per_rank"] = input_bytes_per_rank
            row["source_compact_input_gbps_per_rank"] = None if steady is None else input_bytes_per_rank / steady / 1e9
        rows.extend(variant_rows)
    return rows


def _run_source_push_backward_dx13_xla_source_compact_direct_only(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    route_table,
    inputs: dict[str, jax.Array],
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> list[dict[str, Any]]:
    """Benchmark XLA DX13 math plus source-compact scatter and direct combine."""

    compact_capacity = route_table.valid_by_expert.shape[-1]
    compact_block_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None))
    compact_meta_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    d_activation_blocks = jax.device_put(
        np.ones(
            (config.ep_size, config.experts_per_rank, compact_capacity, config.intermediate_dim),
            dtype=inputs["w13_source"].dtype,
        ),
        compact_block_sharding,
    )
    z_blocks = jax.device_put(
        np.ones(
            (config.ep_size, config.experts_per_rank, compact_capacity, 2 * config.intermediate_dim),
            dtype=inputs["w13_source"].dtype,
        ),
        compact_block_sharding,
    )
    valid = jax.device_put(route_table.valid_by_expert, compact_meta_sharding)
    source_rank_by_expert = jax.device_put(route_table.source_rank_by_expert, compact_meta_sharding)
    token_id_by_expert = jax.device_put(route_table.token_id_by_expert, compact_meta_sharding)
    route_slot_by_expert = jax.device_put(route_table.route_slot_by_expert, compact_meta_sharding)
    compact_slots = source_push_dx13_compact_assignment_slots_from_fields(
        jax.device_put(route_table.source_rank_by_expert, compact_meta_sharding),
        jax.device_put(route_table.dst_ordinal_by_expert, compact_meta_sharding),
        jax.device_put(route_table.entry_by_expert, compact_meta_sharding),
        jax.device_put(route_table.row_in_entry_by_expert, compact_meta_sharding),
        valid,
    )
    queue_shape = tuple(int(dim) for dim in host_inputs.plan.valid_mask.shape)

    def call_dx13_source_compact_direct():
        with jax.set_mesh(mesh):
            dx13_output = source_push_dx13_push_compact_xla(
                d_activation_blocks,
                z_blocks,
                inputs["w13_source"],
                source_rank_by_expert,
                token_id_by_expert,
                route_slot_by_expert,
                valid,
            )
            source_compact = source_push_dx13_contrib_buffer_from_expert_reference(
                dx13_output.dx_expert_major,
                compact_slots,
                queue_shape=queue_shape,
            )
            dx = source_push_dx13_source_compact_combine_reference(source_compact, host_inputs.plan)
        _block_until_ready(dx)
        return dx

    start = time.perf_counter()
    output = call_dx13_source_compact_direct()
    first_call_time = time.perf_counter() - start

    for _ in range(warmup):
        output = call_dx13_source_compact_direct()

    steady_state_times = []
    for _ in range(repeat_runs):
        start = time.perf_counter()
        for _ in range(steps):
            output = call_dx13_source_compact_direct()
        steady_state_times.append((time.perf_counter() - start) / steps)

    _block_until_ready(output)
    stage_useful_flops, stage_rounded_flops = _backward_stage_flops_per_rank(
        config,
        host_inputs.queue_stats,
        BACKWARD_STAGE_DX13_PUSH,
    )
    dropped_routes = int(jax.device_get(host_inputs.plan.dropped_routes))
    source_compact_bytes_per_rank = (
        queue_shape[1] * queue_shape[2] * queue_shape[3] * config.hidden_dim * jnp.dtype(jnp.bfloat16).itemsize
    )
    dx_bytes_per_rank = config.tokens_per_rank * config.hidden_dim * jnp.dtype(jnp.bfloat16).itemsize
    rows = [
        _decomposed_backward_row(
            config,
            queue_stats=host_inputs.queue_stats,
            repeat_run=repeat_run,
            repeat_runs=repeat_runs,
            stage=BACKWARD_STAGE_DX13_PUSH,
            steady_state_time=steady_state_time,
            first_call_time=first_call_time,
            useful_backward_flops=stage_useful_flops,
            rounded_backward_flops=stage_rounded_flops,
            dropped_routes=dropped_routes,
            mode=MODE_BACKWARD_DX13_XLA_SOURCE_COMPACT_DIRECT_ONLY,
        )
        for repeat_run, steady_state_time in enumerate(steady_state_times)
    ]
    rows.append(_summary_row(rows))
    for row in rows:
        steady = row.get("steady_state_time") or row.get("median_steady_state_time")
        row["backward_dx13_implementation"] = BACKWARD_DX13_IMPLEMENTATION_XLA_EXPERT_MAJOR
        row["backward_dx13_boundary"] = "xla_expert_major_to_source_compact_direct_combine"
        row["backward_stop_after_stage"] = BACKWARD_STAGE_DX13_PUSH
        row["source_compact_output_bytes_per_rank"] = source_compact_bytes_per_rank
        row["dx_output_bytes_per_rank"] = dx_bytes_per_rank
        row["source_compact_output_gbps_per_rank"] = (
            None if steady is None else source_compact_bytes_per_rank / steady / 1e9
        )
    return rows


def _run_source_push_backward_dx13_xla_route_buffer_direct_only(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    route_table,
    inputs: dict[str, jax.Array],
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> list[dict[str, Any]]:
    """Benchmark XLA DX13 math plus direct scatter into source route slots."""

    compact_capacity = route_table.valid_by_expert.shape[-1]
    compact_block_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None))
    compact_meta_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    d_activation_blocks = jax.device_put(
        np.ones(
            (config.ep_size, config.experts_per_rank, compact_capacity, config.intermediate_dim),
            dtype=inputs["w13_source"].dtype,
        ),
        compact_block_sharding,
    )
    z_blocks = jax.device_put(
        np.ones(
            (config.ep_size, config.experts_per_rank, compact_capacity, 2 * config.intermediate_dim),
            dtype=inputs["w13_source"].dtype,
        ),
        compact_block_sharding,
    )
    valid = jax.device_put(route_table.valid_by_expert, compact_meta_sharding)
    source_rank_by_expert = jax.device_put(route_table.source_rank_by_expert, compact_meta_sharding)
    token_id_by_expert = jax.device_put(route_table.token_id_by_expert, compact_meta_sharding)
    route_slot_by_expert = jax.device_put(route_table.route_slot_by_expert, compact_meta_sharding)

    def call_dx13_route_buffer_direct():
        with jax.set_mesh(mesh):
            dx13_output = source_push_dx13_push_compact_xla(
                d_activation_blocks,
                z_blocks,
                inputs["w13_source"],
                source_rank_by_expert,
                token_id_by_expert,
                route_slot_by_expert,
                valid,
            )
            dx_routes = source_push_dx13_source_route_buffer_reference(
                dx13_output.dx_expert_major,
                dx13_output.source_rank_by_expert,
                dx13_output.token_id_by_expert,
                dx13_output.route_slot_by_expert,
                dx13_output.valid_by_expert,
                tokens_per_source=config.tokens_per_rank,
                topk=config.topk,
            )
            dx = jnp.sum(dx_routes, axis=2)
        _block_until_ready(dx)
        return dx

    start = time.perf_counter()
    output = call_dx13_route_buffer_direct()
    first_call_time = time.perf_counter() - start

    for _ in range(warmup):
        output = call_dx13_route_buffer_direct()

    steady_state_times = []
    for _ in range(repeat_runs):
        start = time.perf_counter()
        for _ in range(steps):
            output = call_dx13_route_buffer_direct()
        steady_state_times.append((time.perf_counter() - start) / steps)

    _block_until_ready(output)
    stage_useful_flops, stage_rounded_flops = _backward_stage_flops_per_rank(
        config,
        host_inputs.queue_stats,
        BACKWARD_STAGE_DX13_PUSH,
    )
    dropped_routes = int(jax.device_get(host_inputs.plan.dropped_routes))
    route_buffer_bytes_per_rank = (
        config.tokens_per_rank * config.topk * config.hidden_dim * jnp.dtype(jnp.bfloat16).itemsize
    )
    dx_bytes_per_rank = config.tokens_per_rank * config.hidden_dim * jnp.dtype(jnp.bfloat16).itemsize
    rows = [
        _decomposed_backward_row(
            config,
            queue_stats=host_inputs.queue_stats,
            repeat_run=repeat_run,
            repeat_runs=repeat_runs,
            stage=BACKWARD_STAGE_DX13_PUSH,
            steady_state_time=steady_state_time,
            first_call_time=first_call_time,
            useful_backward_flops=stage_useful_flops,
            rounded_backward_flops=stage_rounded_flops,
            dropped_routes=dropped_routes,
            mode=MODE_BACKWARD_DX13_XLA_ROUTE_BUFFER_DIRECT_ONLY,
        )
        for repeat_run, steady_state_time in enumerate(steady_state_times)
    ]
    rows.append(_summary_row(rows))
    for row in rows:
        steady = row.get("steady_state_time") or row.get("median_steady_state_time")
        row["backward_dx13_implementation"] = BACKWARD_DX13_IMPLEMENTATION_XLA_EXPERT_MAJOR
        row["backward_dx13_boundary"] = "xla_expert_major_to_route_buffer_direct_sum"
        row["backward_stop_after_stage"] = BACKWARD_STAGE_DX13_PUSH
        row["route_buffer_output_bytes_per_rank"] = route_buffer_bytes_per_rank
        row["dx_output_bytes_per_rank"] = dx_bytes_per_rank
        row["route_buffer_output_gbps_per_rank"] = (
            None if steady is None else route_buffer_bytes_per_rank / steady / 1e9
        )
    return rows


def _gather_compact_d_route_weights_only(
    d_route_block: jax.Array,
    route_indices,
) -> jax.Array:
    valid = route_indices.valid
    safe_dst = jnp.where(valid, route_indices.dst, 0)
    safe_expert = jnp.where(valid, route_indices.expert, 0)
    safe_row = jnp.where(valid, route_indices.row, 0)
    d_route_weights = d_route_block.at[safe_dst, safe_expert, safe_row].get(
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None)
    )
    return jnp.where(valid, d_route_weights, jnp.zeros((), dtype=d_route_weights.dtype))


def _run_source_push_backward_dx13_source_grouped_only(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    route_table,
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> list[dict[str, Any]]:
    """Benchmark the source-grouped DX13 copy epilogue in isolation."""

    compact_capacity = route_table.valid_by_expert.shape[-1]
    compact_block_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None))
    compact_meta_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    src_base_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    block_sizes = source_push_dx13_pallas_resolved_block_sizes(SourcePushDx13PallasBlockSizes.get_default())
    dx_expert_major = jax.device_put(
        jnp.ones(
            (config.ep_size, config.experts_per_rank, compact_capacity, config.hidden_dim),
            dtype=jnp.bfloat16,
        ),
        compact_block_sharding,
    )
    valid = jax.device_put(route_table.valid_by_expert, compact_meta_sharding)
    source_rank_by_expert = jax.device_put(route_table.source_rank_by_expert, compact_meta_sharding)
    token_id_by_expert = jax.device_put(route_table.token_id_by_expert, compact_meta_sharding)
    route_slot_by_expert = jax.device_put(route_table.route_slot_by_expert, compact_meta_sharding)
    src_base_by_expert = jax.device_put(
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32), src_base_sharding
    )
    source_rows = _dx13_max_source_group_rows(
        route_table.source_rank_by_expert,
        route_table.valid_by_expert,
        jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
    )
    original_rows = dx_expert_major.shape[2]
    padded_rows = math.ceil(original_rows / block_sizes.row_block) * block_sizes.row_block
    if padded_rows != original_rows:
        row_pad = ((0, 0), (0, 0), (0, padded_rows - original_rows))
        dx_expert_major = jnp.pad(dx_expert_major, (*row_pad, (0, 0)))
        source_rank_by_expert = jnp.pad(source_rank_by_expert, row_pad, constant_values=0)
        valid = jnp.pad(valid, row_pad, constant_values=False)

    def call_dx13_source_grouped():
        with jax.set_mesh(mesh):
            output = _source_push_dx13_source_grouped_sharded_mgpu_kernel(
                mesh,
                dx_expert_major,
                source_rank_by_expert,
                valid,
                src_base_by_expert,
                source_rows=source_rows,
                row_block=block_sizes.row_block,
                hidden_block=block_sizes.hidden_block,
            )
        _block_until_ready(output)
        return output

    start = time.perf_counter()
    output = call_dx13_source_grouped()
    first_call_time = time.perf_counter() - start
    output_bytes_per_rank = (
        output.shape[1] * output.shape[2] * output.shape[3] * output.shape[4] * output.dtype.itemsize
    )
    _ = (token_id_by_expert, route_slot_by_expert)

    for _ in range(warmup):
        output = call_dx13_source_grouped()

    steady_state_times = []
    for _ in range(repeat_runs):
        start = time.perf_counter()
        for _ in range(steps):
            output = call_dx13_source_grouped()
        steady_state_times.append((time.perf_counter() - start) / steps)

    _block_until_ready(output)
    dropped_routes = int(jax.device_get(host_inputs.plan.dropped_routes))
    rows = [
        _decomposed_backward_row(
            config,
            queue_stats=host_inputs.queue_stats,
            repeat_run=repeat_run,
            repeat_runs=repeat_runs,
            stage=BACKWARD_STAGE_DX13_SOURCE_GROUPED,
            steady_state_time=steady_state_time,
            first_call_time=first_call_time,
            useful_backward_flops=None,
            rounded_backward_flops=None,
            dropped_routes=dropped_routes,
            mode=MODE_BACKWARD_DX13_SOURCE_GROUPED_ONLY,
        )
        for repeat_run, steady_state_time in enumerate(steady_state_times)
    ]
    rows.append(_summary_row(rows))
    for row in rows:
        steady = row.get("steady_state_time") or row.get("median_steady_state_time")
        row["backward_dx13_implementation"] = SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU
        row["backward_dx13_boundary"] = "source_grouped_remote_write_copy_only"
        row["backward_dx13_row_block"] = block_sizes.row_block
        row["backward_dx13_hidden_block"] = block_sizes.hidden_block
        row["backward_dx13_output_block"] = block_sizes.output_block
        row["backward_stop_after_stage"] = BACKWARD_STAGE_DX13_SOURCE_GROUPED
        row["source_grouped_output_bytes_per_rank"] = output_bytes_per_rank
        row["source_grouped_output_gbps_per_rank"] = None if steady is None else output_bytes_per_rank / steady / 1e9
    return rows


def _run_source_push_backward_return_only(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    route_table,
    warmup: int,
    steps: int,
    repeat_runs: int,
    backward_return_implementation: str,
) -> list[dict[str, Any]]:
    """Benchmark compact backward return/combine in isolation.

    This diagnostic isolates source-local direct gather from W2/W13 backward
    inputs. It intentionally lives in the benchmark harness, not production
    config, because it is a failure-localization tool.
    """

    return_implementation = _resolve_backward_stage_implementation(
        backward_return_implementation,
        source_push_mlp._source_push_mlp_backward_return_implementation(SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU),
    )
    compact_capacity = route_table.valid_by_expert.shape[-1]
    if return_implementation == "pallas_mgpu":
        output_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None))
        route_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
        dx_expert_major = jax.device_put(
            jnp.ones(
                (config.ep_size, config.experts_per_rank, compact_capacity, config.hidden_dim),
                dtype=jnp.bfloat16,
            ),
            output_sharding,
        )
        d_route_block = jax.device_put(
            jnp.ones((config.ep_size, config.experts_per_rank, compact_capacity), dtype=jnp.float32),
            route_sharding,
        )
    else:
        dx_expert_major = jnp.ones(
            (config.ep_size, config.experts_per_rank, compact_capacity, config.hidden_dim),
            dtype=jnp.bfloat16,
        )
        d_route_block = jnp.ones((config.ep_size, config.experts_per_rank, compact_capacity), dtype=jnp.float32)
    with jax.set_mesh(mesh):
        return_route_indices = source_push_backward_return_route_indices_jax(
            host_inputs.plan,
            src_base_by_expert=host_inputs.src_base_by_expert,
        )
        _block_until_ready((dx_expert_major, d_route_block, return_route_indices))

    route_bounds = _compact_return_route_index_bounds(
        return_route_indices,
        compact_capacity,
        experts_per_rank=config.experts_per_rank,
    )

    def call_return():
        with jax.set_mesh(mesh):
            returned = source_push_backward_return(
                dx_expert_major,
                d_route_block,
                host_inputs.plan,
                src_base_by_expert=host_inputs.src_base_by_expert,
                route_indices=return_route_indices,
                implementation=return_implementation,
                mesh=mesh if return_implementation == "pallas_mgpu" else None,
            )
        _block_until_ready(returned)
        return returned

    start = time.perf_counter()
    output = call_return()
    first_call_time = time.perf_counter() - start

    for _ in range(warmup):
        output = call_return()

    steady_state_times = []
    for _ in range(repeat_runs):
        start = time.perf_counter()
        for _ in range(steps):
            output = call_return()
        steady_state_times.append((time.perf_counter() - start) / steps)

    rows = [
        _decomposed_backward_row(
            config,
            queue_stats=host_inputs.queue_stats,
            repeat_run=repeat_run,
            repeat_runs=repeat_runs,
            stage=BACKWARD_STAGE_DX_COMBINE,
            steady_state_time=steady_state_time,
            first_call_time=first_call_time,
            useful_backward_flops=None,
            rounded_backward_flops=None,
            dropped_routes=int(jax.device_get(host_inputs.plan.dropped_routes)),
            mode=MODE_BACKWARD_RETURN_ONLY,
        )
        for repeat_run, steady_state_time in enumerate(steady_state_times)
    ]
    for row in rows:
        row["backward_return_implementation"] = backward_return_implementation
        row["resolved_backward_return_implementation"] = return_implementation
        row["compact_expert_capacity"] = compact_capacity
        row.update(route_bounds)
    summary = _summary_row(rows)
    summary["backward_return_implementation"] = backward_return_implementation
    summary["resolved_backward_return_implementation"] = return_implementation
    summary["compact_expert_capacity"] = compact_capacity
    summary.update(route_bounds)
    _ = output
    return [*rows, summary]


def _run_source_push_backward_return_components_only(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    route_table,
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> list[dict[str, Any]]:
    """Benchmark the main JAX compact return gathers separately."""

    compact_capacity = route_table.valid_by_expert.shape[-1]
    dx_expert_major = jnp.ones(
        (config.ep_size, config.experts_per_rank, compact_capacity, config.hidden_dim),
        dtype=jnp.bfloat16,
    )
    d_route_block = jnp.ones((config.ep_size, config.experts_per_rank, compact_capacity), dtype=jnp.float32)
    route_dx_for_sum = jax.device_put(
        jnp.ones(
            (config.ep_size, config.tokens_per_rank, config.topk, config.hidden_dim),
            dtype=jnp.bfloat16,
        ),
        NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None)),
    )
    with jax.set_mesh(mesh):
        return_route_indices = source_push_backward_return_route_indices_jax(
            host_inputs.plan,
            src_base_by_expert=host_inputs.src_base_by_expert,
        )
        _block_until_ready((dx_expert_major, d_route_block, return_route_indices, route_dx_for_sum))

    route_bounds = _compact_return_route_index_bounds(
        return_route_indices,
        compact_capacity,
        experts_per_rank=config.experts_per_rank,
    )

    valid = return_route_indices.valid
    safe_dst = jnp.where(valid, return_route_indices.dst, 0)
    safe_expert = jnp.where(valid, return_route_indices.expert, 0)
    safe_row = jnp.where(valid, return_route_indices.row, 0)

    def call_dx_gather_sum():
        with jax.set_mesh(mesh):
            route_dx = dx_expert_major.at[safe_dst, safe_expert, safe_row].get(
                out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None)
            )
            route_dx = jnp.where(valid[..., None], route_dx, jnp.zeros((), dtype=route_dx.dtype))
            dx = jnp.sum(route_dx, axis=2)
        _block_until_ready(dx)
        return dx

    def call_droute_gather():
        with jax.set_mesh(mesh):
            d_route_weights = d_route_block.at[safe_dst, safe_expert, safe_row].get(
                out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None)
            )
            d_route_weights = jnp.where(valid, d_route_weights, jnp.zeros((), dtype=d_route_weights.dtype))
        _block_until_ready(d_route_weights)
        return d_route_weights

    def call_dx_sum_only():
        with jax.set_mesh(mesh):
            dx = jnp.sum(route_dx_for_sum, axis=2)
        _block_until_ready(dx)
        return dx

    variants = (
        ("dx_gather_sum", call_dx_gather_sum),
        ("dx_sum_only", call_dx_sum_only),
        ("droute_gather", call_droute_gather),
    )
    dropped_routes = int(jax.device_get(host_inputs.plan.dropped_routes))
    rows: list[dict[str, Any]] = []
    dx_route_bytes_per_rank = config.tokens_per_rank * config.topk * config.hidden_dim * BYTES_PER_BF16
    droute_bytes_per_rank = config.tokens_per_rank * config.topk * jnp.dtype(jnp.float32).itemsize
    for variant, call_component in variants:
        start = time.perf_counter()
        output = call_component()
        first_call_time = time.perf_counter() - start

        for _ in range(warmup):
            output = call_component()

        steady_state_times = []
        for _ in range(repeat_runs):
            start = time.perf_counter()
            for _ in range(steps):
                output = call_component()
            steady_state_times.append((time.perf_counter() - start) / steps)

        _block_until_ready(output)
        variant_rows = [
            _decomposed_backward_row(
                config,
                queue_stats=host_inputs.queue_stats,
                repeat_run=repeat_run,
                repeat_runs=repeat_runs,
                stage=BACKWARD_STAGE_DX_COMBINE,
                steady_state_time=steady_state_time,
                first_call_time=first_call_time,
                useful_backward_flops=None,
                rounded_backward_flops=None,
                dropped_routes=dropped_routes,
                mode=MODE_BACKWARD_RETURN_COMPONENTS_ONLY,
            )
            for repeat_run, steady_state_time in enumerate(steady_state_times)
        ]
        variant_rows.append(_summary_row(variant_rows))
        component_bytes = droute_bytes_per_rank if variant == "droute_gather" else dx_route_bytes_per_rank
        for row in variant_rows:
            steady = row.get("steady_state_time") or row.get("median_steady_state_time")
            row["backward_return_component"] = variant
            row["component_bytes_per_rank"] = component_bytes
            row["component_gbps_per_rank"] = None if steady is None else component_bytes / steady / 1e9
            row.update(route_bounds)
        rows.extend(variant_rows)
    return rows


def _compact_return_route_index_bounds(
    return_route_indices,
    compact_capacity: int,
    *,
    experts_per_rank: int,
) -> dict[str, Any]:
    valid = np.asarray(jax.device_get(return_route_indices.valid), dtype=np.bool_)
    dst = np.asarray(jax.device_get(return_route_indices.dst), dtype=np.int32)
    expert = np.asarray(jax.device_get(return_route_indices.expert), dtype=np.int32)
    row = np.asarray(jax.device_get(return_route_indices.row), dtype=np.int32)
    live_routes = int(np.sum(valid))
    if live_routes == 0:
        return {
            "return_live_routes": 0,
            "return_max_dst": None,
            "return_max_expert": None,
            "return_max_row": None,
            "return_indices_in_bounds": True,
        }
    live_dst = dst[valid]
    live_expert = expert[valid]
    live_row = row[valid]
    in_bounds = (
        np.all(live_dst >= 0)
        and np.all(live_dst < valid.shape[0])
        and np.all(live_expert >= 0)
        and np.all(live_expert < experts_per_rank)
        and np.all(live_row >= 0)
        and np.all(live_row < compact_capacity)
    )
    expert_bound = int(np.max(live_expert))
    return {
        "return_live_routes": live_routes,
        "return_max_dst": int(np.max(live_dst)),
        "return_max_expert": expert_bound,
        "return_max_row": int(np.max(live_row)),
        "return_indices_in_bounds": bool(in_bounds),
    }


def _time_source_push_backward_decomposed(
    route_table,
    expert_base: jax.Array,
    x: jax.Array,
    expert_route_weights: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    h_flat: jax.Array,
    dy: jax.Array,
    *,
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> BackwardDecomposedTiming:
    def call_backward(*, record_stage_times: bool = False):
        stage_times = {stage: 0.0 for stage in BACKWARD_STAGES}
        gradients = source_push_mlp._source_push_mlp_zero_gradients(
            route_table,
            x,
            expert_route_weights,
            w13,
            w2,
        )

        for expert in range(route_table.experts_per_rank):
            if not record_stage_times:
                h_block = source_push_mlp._source_push_mlp_h_flat_for_expert(route_table, expert_base, h_flat, expert)
                backward = source_push_mlp._source_push_mlp_backward_for_expert(
                    route_table,
                    x,
                    expert_route_weights,
                    w13,
                    w2,
                    dy,
                    h_block,
                    expert,
                )
                gradients = source_push_mlp._source_push_mlp_accumulate_expert_backward_outputs(
                    gradients,
                    expert,
                    backward,
                )
                continue

            route_indices = source_push_mlp._source_push_mlp_expert_route_indices(
                route_table,
                expert,
            )

            stage_start = time.perf_counter()
            h_block = source_push_mlp._source_push_mlp_h_flat_for_expert(route_table, expert_base, h_flat, expert)
            h_block = h_block.astype(jnp.float32) * route_indices.valid_f[..., None]
            weights = source_push_mlp._source_push_mlp_route_weights_to_expert_major(
                expert_route_weights,
                expert,
                route_indices.valid_f,
            )
            _block_until_ready((h_block, weights))
            if record_stage_times:
                stage_times[BACKWARD_STAGE_H_WEIGHT_GATHER] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            dy_block = source_push_mlp._source_push_mlp_dy_to_expert_major(
                dy,
                route_indices.safe_src,
                route_indices.safe_token,
                route_indices.valid_f,
            )
            _block_until_ready(dy_block)
            if record_stage_times:
                stage_times[BACKWARD_STAGE_DY_ROUTE] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            gate, up, silu_gate, activation = source_push_mlp._source_push_mlp_activation_from_h(h_block)
            weighted_activation = source_push_mlp._source_push_mlp_weight_activation(activation, weights)
            _block_until_ready((gate, up, silu_gate, activation, weighted_activation))
            if record_stage_times:
                stage_times[BACKWARD_STAGE_ACTIVATION] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            w2_block = w2[:, expert].astype(jnp.float32)
            d_weighted_activation, d_route_block, dw2_block = source_push_mlp._source_push_mlp_w2_backward_for_expert(
                dy_block,
                activation,
                weighted_activation,
                w2_block,
                route_indices.valid_f,
            )
            _block_until_ready((d_weighted_activation, d_route_block, dw2_block))
            if record_stage_times:
                stage_times[BACKWARD_STAGE_W2] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            d_h_block = source_push_mlp._source_push_mlp_swiglu_backward_from_h(
                d_weighted_activation,
                weights,
                gate,
                up,
                silu_gate,
            )
            _block_until_ready((d_route_block, d_h_block))
            if record_stage_times:
                stage_times[BACKWARD_STAGE_SWIGLU] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            x_block = source_push_mlp._source_push_mlp_x_to_expert_major(
                x,
                route_indices.safe_src,
                route_indices.safe_token,
                route_indices.valid_f,
            )
            _block_until_ready(x_block)
            if record_stage_times:
                stage_times[BACKWARD_STAGE_X_REMAT] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            w13_block = w13[:, expert].astype(jnp.float32)
            x_w13 = source_push_mlp._source_push_mlp_w13_backward_for_expert(
                x_block,
                d_h_block,
                w13_block,
            )
            dx_block = x_w13.dx_block
            dw13_block = x_w13.dw13_block
            _block_until_ready((dx_block, dw13_block))
            if record_stage_times:
                stage_times[BACKWARD_STAGE_W13] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            backward = source_push_mlp._SourcePushMlpExpertBackwardOutput(
                route_indices=route_indices,
                dx_block=dx_block,
                d_route_block=d_route_block,
                dw13_block=dw13_block,
                dw2_block=dw2_block,
            )
            gradients = source_push_mlp._source_push_mlp_accumulate_expert_backward_outputs(
                gradients,
                expert,
                backward,
            )
            _block_until_ready(gradients)
            if record_stage_times:
                stage_times[BACKWARD_STAGE_DX_COMBINE] += time.perf_counter() - stage_start

        gradients = source_push_mlp._source_push_mlp_cast_gradients(
            gradients,
            x,
            expert_route_weights,
            w13,
            w2,
        )
        output = (gradients.dx, gradients.d_route_weights, gradients.dw13, gradients.dw2)
        _block_until_ready(output)
        return output, stage_times

    start = time.perf_counter()
    output, _ = call_backward(record_stage_times=False)
    first_call_time = time.perf_counter() - start

    for _ in range(warmup):
        output, _ = call_backward(record_stage_times=False)

    steady_state_times = []
    stage_steady_state_times: dict[str, list[float]] = {stage: [] for stage in BACKWARD_STAGES}
    for _ in range(repeat_runs):
        total_elapsed = 0.0
        stage_elapsed = {stage: 0.0 for stage in BACKWARD_STAGES}
        for _ in range(steps):
            start = time.perf_counter()
            output, step_stage_times = call_backward(record_stage_times=True)
            total_elapsed += time.perf_counter() - start
            for stage in BACKWARD_STAGES:
                stage_elapsed[stage] += step_stage_times[stage]
        steady_state_times.append(total_elapsed / steps)
        for stage in BACKWARD_STAGES:
            stage_steady_state_times[stage].append(stage_elapsed[stage] / steps)

    return BackwardDecomposedTiming(
        first_call_time=first_call_time,
        steady_state_times=steady_state_times,
        output=output,
        stage_steady_state_times=stage_steady_state_times,
    )


def _time_source_push_backward_staged_flat(
    config: PushInboxConfig,
    host_inputs,
    route_table,
    expert_base: jax.Array,
    x: jax.Array,
    h_route_weights: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    h_flat: jax.Array,
    dy: jax.Array,
    *,
    return_route_indices,
    mesh: Mesh,
    warmup: int,
    steps: int,
    repeat_runs: int,
    backward_dy_route_implementation: str,
    backward_w2_implementation: str,
    backward_w2_split_timing: bool,
    backward_w13_implementation: str,
    backward_return_implementation: str,
    backward_stop_after_stage: str,
) -> BackwardDecomposedTiming:
    dy_route_implementation = _resolve_backward_stage_implementation(
        backward_dy_route_implementation,
        source_push_mlp._source_push_mlp_backward_dy_route_implementation(SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU),
    )
    w13_implementation = _resolve_backward_stage_implementation(
        backward_w13_implementation,
        source_push_mlp._source_push_mlp_backward_w13_implementation(SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU),
    )
    w13_diagnostic_only = source_push_w13_backward_is_diagnostic_only(w13_implementation)
    if w13_diagnostic_only and backward_stop_after_stage != BACKWARD_STAGE_W13:
        raise ValueError(
            "W13 dx-only/dw13-only diagnostics produce partial gradients and require "
            f"--backward-stop-after-stage {BACKWARD_STAGE_W13!r}"
        )
    w2_implementation, w2_matmul_implementation, w2_swiglu_implementation = _resolve_w2_backward_implementations(
        backward_w2_implementation,
        source_push_mlp._source_push_mlp_backward_w2_implementation(SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU),
    )
    if backward_w2_implementation == BACKWARD_IMPLEMENTATION_DEFAULT:
        w2_matmul_implementation = source_push_mlp._source_push_mlp_backward_w2_matmul_implementation(
            SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU
        )
        w2_swiglu_implementation = source_push_mlp._source_push_mlp_backward_w2_swiglu_implementation(
            SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU
        )
    return_implementation = _resolve_backward_stage_implementation(
        backward_return_implementation,
        source_push_mlp._source_push_mlp_backward_return_implementation(SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU),
    )
    valid = _source_push_w2_valid_blocks_sharded(route_table.valid_by_expert)
    compact_meta_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    source_rank_by_expert = jax.device_put(route_table.source_rank_by_expert, compact_meta_sharding)
    token_id_by_expert = jax.device_put(route_table.token_id_by_expert, compact_meta_sharding)
    flat_stages = _staged_flat_backward_stages(backward_stop_after_stage)
    timed_stages = BACKWARD_STAGES
    if backward_w2_split_timing and BACKWARD_STAGE_W2 in flat_stages:
        timed_stages = (*BACKWARD_STAGES, *BACKWARD_W2_SPLIT_STAGES)

    def call_w2_backward(
        expert_base_arg: jax.Array,
        h_flat_arg: jax.Array,
        h_route_weights_arg: jax.Array,
        dy_flat_arg: jax.Array,
        w2_arg: jax.Array,
        valid_by_expert_arg: jax.Array,
    ) -> _SourcePushW2BackwardOutput:
        return _source_push_w2_backward_from_flat_h(
            expert_base_arg,
            h_flat_arg,
            h_route_weights_arg,
            dy_flat_arg,
            w2_arg,
            valid_by_expert_arg,
            implementation=w2_implementation,
            matmul_implementation=w2_matmul_implementation,
            swiglu_implementation=w2_swiglu_implementation,
            mesh=mesh,
            contiguous_expert_gather=not host_inputs.use_exact_expert_major,
        )

    jitted_w2_backward = jax.jit(call_w2_backward)

    def call_backward(*, record_stage_times: bool = False):
        stage_times = {stage: 0.0 for stage in timed_stages}

        stage_start = time.perf_counter()
        dy_flat = _source_push_backward_dy_to_h_rows(
            config,
            host_inputs,
            dy,
            implementation=dy_route_implementation,
            mesh=mesh,
        )
        _block_until_ready(dy_flat)
        if record_stage_times:
            stage_times[BACKWARD_STAGE_DY_ROUTE] = time.perf_counter() - stage_start
        if backward_stop_after_stage == BACKWARD_STAGE_DY_ROUTE:
            return dy_flat, stage_times

        stage_start = time.perf_counter()
        if backward_w2_split_timing:
            w2_grads, w2_stage_times = _source_push_w2_backward_from_flat_h_split_timing(
                expert_base,
                h_flat,
                h_route_weights,
                dy_flat,
                w2,
                route_table.valid_by_expert,
                implementation=w2_implementation,
                matmul_implementation=w2_matmul_implementation,
                swiglu_implementation=w2_swiglu_implementation,
                mesh=mesh,
                record_stage_times=record_stage_times,
                contiguous_expert_gather=not host_inputs.use_exact_expert_major,
            )
        else:
            w2_stage_times = {}
            w2_grads = jitted_w2_backward(
                expert_base,
                h_flat,
                h_route_weights,
                dy_flat,
                w2,
                route_table.valid_by_expert,
            )
        _block_until_ready(w2_grads)
        if record_stage_times:
            stage_times[BACKWARD_STAGE_W2] = time.perf_counter() - stage_start
            stage_times.update(w2_stage_times)
        if backward_stop_after_stage == BACKWARD_STAGE_W2:
            return w2_grads, stage_times

        stage_start = time.perf_counter()
        if w13_implementation in (
            SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_PALLAS_MGPU_COMPACT,
            SOURCE_PUSH_W13_BACKWARD_EXPERIMENT_COMPACT_DX_SOURCE_GATHER_DW13,
            SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_COMPACT_DX_ONLY,
            SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_COMPACT_DW13_ONLY,
            SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_EXACT_FLAT_DW13_ONLY,
            SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_PERSISTENT_DW13_ONLY,
            SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_XLA_COMPACT_DW13_ONLY,
            SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_XLA_SOURCE_PADDED_DW13_ONLY,
            SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_SOURCE_PADDED_PARTIALS_DW13_ONLY,
            SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_SOURCE_GATHER_DW13_ONLY,
        ):
            d_h_blocks = _gather_flat_rows_by_expert_slice(w2_grads.d_h, expert_base, valid.shape[-1])
            if w13_implementation == SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_PALLAS_MGPU_COMPACT:
                w13_grads = _source_push_w13_backward_expert_blocks_pallas_mgpu(
                    x,
                    d_h_blocks,
                    w13,
                    source_rank_by_expert,
                    token_id_by_expert,
                    valid,
                    mesh=mesh,
                )
            elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_EXPERIMENT_COMPACT_DX_SOURCE_GATHER_DW13:
                w13_grads = _source_push_w13_backward_expert_blocks_compact_dx_source_gather_dw13(
                    x,
                    d_h_blocks,
                    w13,
                    source_rank_by_expert,
                    token_id_by_expert,
                    valid,
                    mesh=mesh,
                )
            elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_COMPACT_DX_ONLY:
                w13_grads = _source_push_w13_backward_expert_blocks_dx_only_pallas_mgpu(
                    d_h_blocks,
                    w13,
                    valid,
                    mesh=mesh,
                )
            elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_COMPACT_DW13_ONLY:
                w13_grads = _source_push_w13_backward_expert_blocks_dw13_only_pallas_mgpu(
                    x,
                    d_h_blocks,
                    w13,
                    source_rank_by_expert,
                    token_id_by_expert,
                    valid,
                    mesh=mesh,
                )
            elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_EXACT_FLAT_DW13_ONLY:
                w13_grads = _source_push_w13_backward_expert_blocks_dw13_only_exact_flat_pallas_mgpu(
                    x,
                    d_h_blocks,
                    w13,
                    source_rank_by_expert,
                    token_id_by_expert,
                    valid,
                    mesh=mesh,
                )
            elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_XLA_COMPACT_DW13_ONLY:
                w13_grads = source_push_w13_backward_expert_blocks_dw13_only_xla(
                    x,
                    d_h_blocks,
                    w13,
                    source_rank_by_expert,
                    token_id_by_expert,
                    valid,
                )
            elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_XLA_SOURCE_PADDED_DW13_ONLY:
                w13_grads = source_push_w13_backward_expert_blocks_source_padded_dw13_only_xla(
                    x,
                    d_h_blocks,
                    source_rank_by_expert,
                    token_id_by_expert,
                    valid,
                    host_inputs.src_base_by_expert,
                )
            elif (
                w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_SOURCE_PADDED_PARTIALS_DW13_ONLY
            ):
                partials = _source_push_w13_dw13_source_padded_partials_pallas_mgpu(
                    x,
                    d_h_blocks,
                    source_rank_by_expert,
                    token_id_by_expert,
                    valid,
                    host_inputs.src_base_by_expert,
                    mesh=mesh,
                )
                w13_grads = SourcePushW13CompactBackwardOutput(
                    x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
                    dx_expert_major=jnp.zeros((0,), dtype=jnp.float32),
                    dw13=jnp.sum(partials, axis=0),
                )
            elif w13_implementation in (
                BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_LOCAL_SWIGLU_DX13_DW13,
                BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_DX13_XLA_LOCAL_SWIGLU_DW13,
                BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_DX13_SPLIT_LOCAL_SWIGLU_DW13,
                BACKWARD_W13_IMPLEMENTATION_XLA_DX13_SOURCE_GATHER_DW13,
                BACKWARD_W13_IMPLEMENTATION_XLA_DX13_PALLAS_X_REMAT_XLA_LOCAL_SWIGLU_DW13,
            ):
                raise ValueError(
                    f"{w13_implementation!r} requires compact-H "
                    f"`d_activation` and saved W13 preactivation; use {MODE_BACKWARD_STAGED_BLOCKS!r}."
                )
            elif w13_implementation in (
                SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_PERSISTENT_DW13_ONLY,
                SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_SPLIT_DW13_ONLY,
            ):
                raise ValueError(
                    f"{w13_implementation!r} "
                    f"requires compact-H `d_activation` and saved W13 preactivation; use {MODE_BACKWARD_STAGED_BLOCKS!r} "
                    f"or {MODE_BACKWARD_W13_ONLY!r}."
                )
            else:
                w13_grads = source_push_w13_backward_expert_blocks_source_gather_dw13_only(
                    x,
                    d_h_blocks,
                    source_rank_by_expert,
                    token_id_by_expert,
                    valid,
                )
        else:
            w13_grads = source_push_w13_backward(
                x,
                w2_grads.d_h,
                w13,
                host_inputs.plan,
                host_inputs.send_meta,
                expert_base,
                host_inputs.src_base_by_expert,
                use_exact_expert_major=host_inputs.use_exact_expert_major,
                implementation=w13_implementation,
                mesh=mesh,
            )
        _block_until_ready(w13_grads)
        if record_stage_times:
            stage_times[BACKWARD_STAGE_W13] = time.perf_counter() - stage_start
        if backward_stop_after_stage == BACKWARD_STAGE_W13:
            return w13_grads, stage_times

        stage_start = time.perf_counter()
        if w13_implementation in (
            SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_PALLAS_MGPU_COMPACT,
            SOURCE_PUSH_W13_BACKWARD_EXPERIMENT_COMPACT_DX_SOURCE_GATHER_DW13,
            SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_COMPACT_DX_ONLY,
        ):
            d_route_weight_blocks = _gather_flat_rows_by_expert_slice(
                w2_grads.d_route_weight,
                expert_base,
                valid.shape[-1],
            )
            returned = source_push_backward_return(
                w13_grads.dx_expert_major,
                d_route_weight_blocks,
                host_inputs.plan,
                src_base_by_expert=host_inputs.src_base_by_expert,
                route_indices=return_route_indices,
                implementation=return_implementation,
                mesh=mesh,
            )
        else:
            returned = source_push_backward_return_flat(
                w13_grads.dx_expert_major,
                w2_grads.d_route_weight,
                host_inputs.plan,
                expert_base=expert_base,
                src_base_by_expert=host_inputs.src_base_by_expert,
                route_indices=return_route_indices,
                implementation=return_implementation,
                mesh=mesh,
            )
        output = (
            returned.dx.astype(x.dtype),
            returned.d_route_weights.astype(h_route_weights.dtype),
            w13_grads.dw13.astype(w13.dtype),
            w2_grads.dw2.astype(w2.dtype),
        )
        _block_until_ready(output)
        if record_stage_times:
            stage_times[BACKWARD_STAGE_DX_COMBINE] = time.perf_counter() - stage_start

        return output, stage_times

    start = time.perf_counter()
    output, _ = call_backward(record_stage_times=False)
    first_call_time = time.perf_counter() - start

    for _ in range(warmup):
        output, _ = call_backward(record_stage_times=False)

    steady_state_times = []
    stage_steady_state_times: dict[str, list[float]] = {stage: [] for stage in timed_stages}
    for _ in range(repeat_runs):
        total_elapsed = 0.0
        stage_elapsed = {stage: 0.0 for stage in timed_stages}
        for _ in range(steps):
            start = time.perf_counter()
            output, step_stage_times = call_backward(record_stage_times=True)
            total_elapsed += time.perf_counter() - start
            for stage in timed_stages:
                stage_elapsed[stage] += step_stage_times[stage]
        steady_state_times.append(total_elapsed / steps)
        for stage in timed_stages:
            stage_steady_state_times[stage].append(stage_elapsed[stage] / steps)

    return BackwardDecomposedTiming(
        first_call_time=first_call_time,
        steady_state_times=steady_state_times,
        output=output,
        stage_steady_state_times=stage_steady_state_times,
    )


def _time_source_push_backward_staged_blocks(
    config: PushInboxConfig,
    host_inputs,
    route_table,
    expert_base: jax.Array,
    x: jax.Array,
    expert_route_weights: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    h_blocks: jax.Array,
    dy: jax.Array,
    *,
    mesh: Mesh,
    warmup: int,
    steps: int,
    repeat_runs: int,
    backward_dy_route_implementation: str,
    backward_w2_implementation: str,
    backward_w13_implementation: str,
    backward_return_implementation: str,
    backward_stop_after_stage: str,
    return_route_indices,
) -> BackwardDecomposedTiming:
    w2_implementation, w2_matmul_implementation, w2_swiglu_implementation = _resolve_w2_backward_implementations(
        backward_w2_implementation,
        source_push_mlp._source_push_mlp_backward_w2_implementation(SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU),
    )
    if backward_w2_implementation == BACKWARD_IMPLEMENTATION_DEFAULT:
        w2_matmul_implementation = source_push_mlp._source_push_mlp_backward_w2_matmul_implementation(
            SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU
        )
        w2_swiglu_implementation = source_push_mlp._source_push_mlp_backward_w2_swiglu_implementation(
            SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU
        )
    w13_implementation = _resolve_backward_stage_implementation(
        backward_w13_implementation,
        source_push_mlp._source_push_mlp_backward_w13_implementation(SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU),
    )
    return_implementation = _resolve_backward_stage_implementation(
        backward_return_implementation,
        source_push_mlp._source_push_mlp_backward_return_implementation(SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU),
    )
    dy_route_implementation = _resolve_backward_stage_implementation(
        backward_dy_route_implementation,
        source_push_mlp._source_push_mlp_backward_dy_route_implementation(SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU),
    )

    def call_w2_backward(
        h_blocks_arg: jax.Array,
        expert_route_weights_arg: jax.Array,
        dy_blocks_arg: jax.Array,
        w2_arg: jax.Array,
        valid_arg: jax.Array,
    ) -> _SourcePushW2BackwardOutput:
        return _source_push_w2_backward_expert_blocks(
            h_blocks_arg,
            expert_route_weights_arg,
            dy_blocks_arg,
            w2_arg,
            valid_arg,
            implementation=w2_implementation,
            matmul_implementation=w2_matmul_implementation,
            swiglu_implementation=w2_swiglu_implementation,
            mesh=mesh,
        )

    jitted_w2_backward = jax.jit(call_w2_backward)
    valid = _source_push_w2_valid_blocks_sharded(route_table.valid_by_expert)
    compact_meta_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    compact_block_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None))
    source_rank_by_expert = jax.device_put(route_table.source_rank_by_expert, compact_meta_sharding)
    token_id_by_expert = jax.device_put(route_table.token_id_by_expert, compact_meta_sharding)
    route_slot_by_expert = jax.device_put(route_table.route_slot_by_expert, compact_meta_sharding)
    stages = _backward_staged_block_timed_stages(backward_stop_after_stage, w13_implementation)

    def rematerialize_x_expert_major() -> jax.Array:
        safe_src = jnp.where(valid, source_rank_by_expert, 0)
        safe_token = jnp.where(valid, token_id_by_expert, 0)
        x_expert_major = x.at[safe_src, safe_token].get(out_sharding=compact_block_sharding)
        return jnp.where(valid[..., None], x_expert_major, jnp.zeros_like(x_expert_major))

    def rematerialize_x_expert_major_pallas_flat() -> jax.Array:
        rows_per_expert = valid.shape[-1]
        expected_rows = config.experts_per_rank * rows_per_expert
        if config.hidden_rows_per_rank != expected_rows:
            raise ValueError(
                "Pallas flat x-remat diagnostic requires flat W13 rows to reshape to compact expert blocks; "
                f"got hidden_rows_per_rank={config.hidden_rows_per_rank}, "
                f"experts_per_rank={config.experts_per_rank}, rows_per_expert={rows_per_expert}"
            )
        x_rows = source_push_x_to_w13_rows(
            x,
            host_inputs.plan,
            host_inputs.send_meta,
            expert_base,
            host_inputs.src_base_by_expert,
            hidden_rows_per_rank=config.hidden_rows_per_rank,
            use_exact_expert_major=host_inputs.use_exact_expert_major,
            implementation=SOURCE_PUSH_X_TO_W13_ROWS_IMPLEMENTATION_PALLAS_MGPU,
            block_sizes=SourcePushXToW13RowsPallasBlockSizes(hidden_block=128),
            mesh=mesh,
        )
        x_expert_major = x_rows.reshape((config.ep_size, config.experts_per_rank, rows_per_expert, config.hidden_dim))
        return jax.device_put(x_expert_major, compact_block_sharding)

    def call_backward(*, record_stage_times: bool = False):
        stage_times = {stage: 0.0 for stage in stages}
        x_remat_stage_time = 0.0
        dx_direct = None

        def timed_rematerialize_x_expert_major() -> jax.Array:
            nonlocal x_remat_stage_time
            remat_start = time.perf_counter()
            x_expert_major = rematerialize_x_expert_major()
            if record_stage_times:
                _block_until_ready(x_expert_major)
                x_remat_stage_time += time.perf_counter() - remat_start
                if BACKWARD_STAGE_X_REMAT in stage_times:
                    stage_times[BACKWARD_STAGE_X_REMAT] = x_remat_stage_time
            return x_expert_major

        def timed_rematerialize_x_expert_major_pallas_flat() -> jax.Array:
            nonlocal x_remat_stage_time
            remat_start = time.perf_counter()
            x_expert_major = rematerialize_x_expert_major_pallas_flat()
            if record_stage_times:
                _block_until_ready(x_expert_major)
                x_remat_stage_time += time.perf_counter() - remat_start
                if BACKWARD_STAGE_X_REMAT in stage_times:
                    stage_times[BACKWARD_STAGE_X_REMAT] = x_remat_stage_time
            return x_expert_major

        stage_start = time.perf_counter()
        if dy_route_implementation == SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_SOURCE_PUSH_PALLAS_MGPU:
            dy_blocks = _source_push_backward_dy_to_expert_major_source_push_pallas_call(
                mesh,
                dy,
                host_inputs.plan.token_ids,
                host_inputs.send_meta,
                ep_size=config.ep_size,
                entries_per_rank=config.entries_per_rank,
                block_m=config.block_m,
                experts_per_rank=config.experts_per_rank,
                expert_capacity=route_table.valid_by_expert.shape[-1],
                row_block=config.block_m,
                hidden_block=128,
            )
        elif dy_route_implementation == SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_SOURCE_PUSH_JAX:
            dy_blocks = _source_push_backward_dy_to_expert_major_from_plan_source_push_jax(
                dy,
                host_inputs.plan,
                host_inputs.send_meta,
                host_inputs.expert_base,
                host_inputs.src_base_by_expert,
                experts_per_rank=config.experts_per_rank,
                expert_capacity=route_table.valid_by_expert.shape[-1],
                use_exact_expert_major=host_inputs.use_exact_expert_major,
            )
        else:
            dy_blocks = _source_push_backward_dy_to_expert_major(
                dy,
                source_rank_by_expert,
                token_id_by_expert,
                valid,
                implementation=dy_route_implementation,
                mesh=mesh,
            )
        _block_until_ready(dy_blocks)
        if record_stage_times:
            stage_times[BACKWARD_STAGE_DY_ROUTE] = time.perf_counter() - stage_start
        if backward_stop_after_stage == BACKWARD_STAGE_DY_ROUTE:
            return dy_blocks, stage_times

        stage_start = time.perf_counter()
        w2_grads = jitted_w2_backward(h_blocks, expert_route_weights, dy_blocks, w2, valid)
        _block_until_ready(w2_grads)
        if record_stage_times:
            stage_times[BACKWARD_STAGE_W2] = time.perf_counter() - stage_start
        if backward_stop_after_stage == BACKWARD_STAGE_W2:
            return w2_grads, stage_times

        stage_start = time.perf_counter()
        if w13_implementation == SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_TILED:
            w13_grads = source_push_w13_backward_expert_blocks_tiled_reference(
                x,
                w2_grads.d_h,
                w13,
                source_rank_by_expert,
                token_id_by_expert,
                valid,
            )
        elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_PALLAS_MGPU_COMPACT:
            w13_grads = _source_push_w13_backward_expert_blocks_pallas_mgpu(
                x,
                w2_grads.d_h,
                w13,
                source_rank_by_expert,
                token_id_by_expert,
                valid,
                mesh=mesh,
            )
        elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_EXPERIMENT_COMPACT_DX_SOURCE_GATHER_DW13:
            w13_grads = _source_push_w13_backward_expert_blocks_compact_dx_source_gather_dw13(
                x,
                w2_grads.d_h,
                w13,
                source_rank_by_expert,
                token_id_by_expert,
                valid,
                mesh=mesh,
            )
        elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_COMPACT_DX_ONLY:
            w13_grads = _source_push_w13_backward_expert_blocks_dx_only_pallas_mgpu(
                w2_grads.d_h,
                w13,
                valid,
                mesh=mesh,
            )
        elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_COMPACT_DW13_ONLY:
            w13_grads = _source_push_w13_backward_expert_blocks_dw13_only_pallas_mgpu(
                x,
                w2_grads.d_h,
                w13,
                source_rank_by_expert,
                token_id_by_expert,
                valid,
                mesh=mesh,
            )
        elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_PREFILLED_X_DW13_ONLY:
            raise ValueError(
                f"{SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_PREFILLED_X_DW13_ONLY!r} is a W13-only "
                "diagnostic because it pre-materializes x outside the timed W13 call."
            )
        elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_EXACT_FLAT_DW13_ONLY:
            w13_grads = _source_push_w13_backward_expert_blocks_dw13_only_exact_flat_pallas_mgpu(
                x,
                w2_grads.d_h,
                w13,
                source_rank_by_expert,
                token_id_by_expert,
                valid,
                mesh=mesh,
            )
        elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_DW13_ONLY:
            if backward_stop_after_stage != BACKWARD_STAGE_W13:
                raise ValueError(
                    f"{SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_DW13_ONLY!r} only produces "
                    f"`dw13`; use {BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_LOCAL_SWIGLU_DX13_DW13!r} for a "
                    "full staged-block backward."
                )
            w13_grads = _source_push_w13_backward_expert_blocks_local_swiglu_dw13_only_pallas_mgpu(
                timed_rematerialize_x_expert_major(),
                w2_grads.d_activation,
                h_blocks,
                valid,
                mesh=mesh,
            )
        elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_PERSISTENT_DW13_ONLY:
            if backward_stop_after_stage != BACKWARD_STAGE_W13:
                raise ValueError(
                    f"{SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_PERSISTENT_DW13_ONLY!r} only "
                    "produces `dw13`."
                )
            w13_grads = _source_push_w13_backward_expert_blocks_local_swiglu_persistent_dw13_only_pallas_mgpu(
                timed_rematerialize_x_expert_major(),
                w2_grads.d_activation,
                h_blocks,
                valid,
                mesh=mesh,
            )
        elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_SPLIT_DW13_ONLY:
            if backward_stop_after_stage != BACKWARD_STAGE_W13:
                raise ValueError(
                    f"{SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_SPLIT_DW13_ONLY!r} only "
                    "produces `dw13`."
                )
            w13_grads = _source_push_w13_backward_expert_blocks_local_swiglu_split_dw13_only_pallas_mgpu(
                timed_rematerialize_x_expert_major(),
                w2_grads.d_activation,
                h_blocks,
                valid,
                mesh=mesh,
            )
        elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_LINEAR_DW13_ONLY:
            raise ValueError(
                f"{SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_LINEAR_DW13_ONLY!r} is a W13-only "
                "diagnostic for isolating dSwiGLU pointwise cost; it is not a correct staged backward."
            )
        elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_XLA_LOCAL_SWIGLU_DW13_ONLY:
            if backward_stop_after_stage != BACKWARD_STAGE_W13:
                raise ValueError(
                    f"{SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_XLA_LOCAL_SWIGLU_DW13_ONLY!r} only produces `dw13`."
                )
            w13_grads = source_push_w13_backward_expert_blocks_local_swiglu_dw13_only_xla(
                timed_rematerialize_x_expert_major(),
                w2_grads.d_activation,
                h_blocks,
                valid,
            )
        elif w13_implementation in (
            SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_GATE_DW13_ONLY,
            SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_UP_DW13_ONLY,
        ):
            raise ValueError(f"{w13_implementation!r} is a W13-only half-DW13 diagnostic.")
        elif w13_implementation == BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_LOCAL_SWIGLU_DX13_DW13:
            dx13_output = _source_push_w13_backward_expert_blocks_dx_only_pallas_mgpu(
                w2_grads.d_h,
                w13,
                valid,
                mesh=mesh,
            )
            dw13_output = _source_push_w13_backward_expert_blocks_local_swiglu_dw13_only_pallas_mgpu(
                timed_rematerialize_x_expert_major(),
                w2_grads.d_activation,
                h_blocks,
                valid,
                mesh=mesh,
            )
            w13_grads = SourcePushW13CompactBackwardOutput(
                x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
                dx_expert_major=dx13_output.dx_expert_major,
                dw13=dw13_output.dw13,
            )
        elif w13_implementation == BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_DX13_XLA_LOCAL_SWIGLU_DW13:
            dx13_output = _source_push_w13_backward_expert_blocks_dx_only_pallas_mgpu(
                w2_grads.d_h,
                w13,
                valid,
                mesh=mesh,
            )
            dw13_output = source_push_w13_backward_expert_blocks_local_swiglu_dw13_only_xla(
                timed_rematerialize_x_expert_major(),
                w2_grads.d_activation,
                h_blocks,
                valid,
            )
            w13_grads = SourcePushW13CompactBackwardOutput(
                x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
                dx_expert_major=dx13_output.dx_expert_major,
                dw13=dw13_output.dw13,
            )
        elif w13_implementation == BACKWARD_W13_IMPLEMENTATION_XLA_DX13_XLA_LOCAL_SWIGLU_DW13:
            dx13_output = source_push_dx13_push_compact_xla(
                w2_grads.d_activation,
                h_blocks,
                w13,
                source_rank_by_expert,
                token_id_by_expert,
                route_slot_by_expert,
                valid,
            )
            dw13_output = source_push_w13_backward_expert_blocks_local_swiglu_dw13_only_xla(
                timed_rematerialize_x_expert_major(),
                w2_grads.d_activation,
                h_blocks,
                valid,
            )
            w13_grads = SourcePushW13CompactBackwardOutput(
                x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
                dx_expert_major=dx13_output.dx_expert_major,
                dw13=dw13_output.dw13,
            )
        elif w13_implementation == BACKWARD_W13_IMPLEMENTATION_XLA_DX13_PALLAS_X_REMAT_XLA_LOCAL_SWIGLU_DW13:
            dx13_output = source_push_dx13_push_compact_xla(
                w2_grads.d_activation,
                h_blocks,
                w13,
                source_rank_by_expert,
                token_id_by_expert,
                route_slot_by_expert,
                valid,
            )
            dw13_output = source_push_w13_backward_expert_blocks_local_swiglu_dw13_only_xla(
                timed_rematerialize_x_expert_major_pallas_flat(),
                w2_grads.d_activation,
                h_blocks,
                valid,
            )
            w13_grads = SourcePushW13CompactBackwardOutput(
                x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
                dx_expert_major=dx13_output.dx_expert_major,
                dw13=dw13_output.dw13,
            )
        elif w13_implementation == BACKWARD_W13_IMPLEMENTATION_XLA_DX13_ROUTE_BUFFER_XLA_LOCAL_SWIGLU_DW13:
            dx13_output = source_push_dx13_push_compact_xla(
                w2_grads.d_activation,
                h_blocks,
                w13,
                source_rank_by_expert,
                token_id_by_expert,
                route_slot_by_expert,
                valid,
            )
            dx_routes = source_push_dx13_source_route_buffer_reference(
                dx13_output.dx_expert_major,
                dx13_output.source_rank_by_expert,
                dx13_output.token_id_by_expert,
                dx13_output.route_slot_by_expert,
                dx13_output.valid_by_expert,
                tokens_per_source=config.tokens_per_rank,
                topk=config.topk,
            )
            dx_direct = jnp.sum(dx_routes, axis=2)
            dw13_output = source_push_w13_backward_expert_blocks_local_swiglu_dw13_only_xla(
                timed_rematerialize_x_expert_major(),
                w2_grads.d_activation,
                h_blocks,
                valid,
            )
            w13_grads = SourcePushW13CompactBackwardOutput(
                x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
                dx_expert_major=jnp.zeros((0,), dtype=jnp.float32),
                dw13=dw13_output.dw13,
            )
        elif w13_implementation == BACKWARD_W13_IMPLEMENTATION_XLA_DX13_SOURCE_GATHER_DW13:
            dx13_output = source_push_dx13_push_compact_xla(
                w2_grads.d_activation,
                h_blocks,
                w13,
                source_rank_by_expert,
                token_id_by_expert,
                route_slot_by_expert,
                valid,
            )
            dw13_output = source_push_w13_backward_expert_blocks_source_gather_dw13_only(
                x,
                w2_grads.d_h,
                source_rank_by_expert,
                token_id_by_expert,
                valid,
            )
            w13_grads = SourcePushW13CompactBackwardOutput(
                x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
                dx_expert_major=dx13_output.dx_expert_major,
                dw13=dw13_output.dw13,
            )
        elif w13_implementation == BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_DX13_SPLIT_LOCAL_SWIGLU_DW13:
            dx13_output = _source_push_w13_backward_expert_blocks_dx_only_pallas_mgpu(
                w2_grads.d_h,
                w13,
                valid,
                mesh=mesh,
            )
            dw13_output = _source_push_w13_backward_expert_blocks_local_swiglu_split_dw13_only_pallas_mgpu(
                timed_rematerialize_x_expert_major(),
                w2_grads.d_activation,
                h_blocks,
                valid,
                mesh=mesh,
            )
            w13_grads = SourcePushW13CompactBackwardOutput(
                x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
                dx_expert_major=dx13_output.dx_expert_major,
                dw13=dw13_output.dw13,
            )
        elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_XLA_COMPACT_DW13_ONLY:
            w13_grads = source_push_w13_backward_expert_blocks_dw13_only_xla(
                x,
                w2_grads.d_h,
                w13,
                source_rank_by_expert,
                token_id_by_expert,
                valid,
            )
        elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_XLA_SOURCE_PADDED_DW13_ONLY:
            w13_grads = source_push_w13_backward_expert_blocks_source_padded_dw13_only_xla(
                x,
                w2_grads.d_h,
                source_rank_by_expert,
                token_id_by_expert,
                valid,
                host_inputs.src_base_by_expert,
            )
        elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_SOURCE_PADDED_PARTIALS_DW13_ONLY:
            partials = _source_push_w13_dw13_source_padded_partials_pallas_mgpu(
                x,
                w2_grads.d_h,
                source_rank_by_expert,
                token_id_by_expert,
                valid,
                host_inputs.src_base_by_expert,
                mesh=mesh,
            )
            w13_grads = SourcePushW13CompactBackwardOutput(
                x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
                dx_expert_major=jnp.zeros((0,), dtype=jnp.float32),
                dw13=jnp.sum(partials, axis=0),
            )
        elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_SOURCE_GATHER_DW13_ONLY:
            w13_grads = source_push_w13_backward_expert_blocks_source_gather_dw13_only(
                x,
                w2_grads.d_h,
                source_rank_by_expert,
                token_id_by_expert,
                valid,
            )
        else:
            d_h_flat = _flatten_expert_blocks_to_flat_rows(
                expert_base,
                w2_grads.d_h,
                flat_rows_per_rank=config.hidden_rows_per_rank,
                valid=valid,
            )
            w13_grads = source_push_w13_backward(
                x,
                d_h_flat,
                w13,
                host_inputs.plan,
                host_inputs.send_meta,
                expert_base,
                host_inputs.src_base_by_expert,
                use_exact_expert_major=host_inputs.use_exact_expert_major,
                implementation=w13_implementation,
                mesh=mesh,
            )
        _block_until_ready((w13_grads, dx_direct) if dx_direct is not None else w13_grads)
        if record_stage_times:
            stage_times[BACKWARD_STAGE_W13] = time.perf_counter() - stage_start - x_remat_stage_time
        if backward_stop_after_stage == BACKWARD_STAGE_W13:
            return w13_grads, stage_times

        stage_start = time.perf_counter()
        if w13_implementation == BACKWARD_W13_IMPLEMENTATION_XLA_DX13_ROUTE_BUFFER_XLA_LOCAL_SWIGLU_DW13:
            d_route_weights = _gather_compact_d_route_weights_only(w2_grads.d_route_weight, return_route_indices)
            output = (
                dx_direct.astype(x.dtype),
                d_route_weights.astype(expert_route_weights.dtype),
                w13_grads.dw13.astype(w13.dtype),
                w2_grads.dw2.astype(w2.dtype),
            )
            _block_until_ready(output)
            if record_stage_times:
                stage_times[BACKWARD_STAGE_DX_COMBINE] = time.perf_counter() - stage_start
            return output, stage_times
        if w13_implementation in (
            SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_TILED,
            SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_PALLAS_MGPU_COMPACT,
            SOURCE_PUSH_W13_BACKWARD_EXPERIMENT_COMPACT_DX_SOURCE_GATHER_DW13,
            SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_COMPACT_DX_ONLY,
            BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_LOCAL_SWIGLU_DX13_DW13,
            BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_DX13_XLA_LOCAL_SWIGLU_DW13,
            BACKWARD_W13_IMPLEMENTATION_PALLAS_MGPU_DX13_SPLIT_LOCAL_SWIGLU_DW13,
            BACKWARD_W13_IMPLEMENTATION_XLA_DX13_XLA_LOCAL_SWIGLU_DW13,
            BACKWARD_W13_IMPLEMENTATION_XLA_DX13_PALLAS_X_REMAT_XLA_LOCAL_SWIGLU_DW13,
            BACKWARD_W13_IMPLEMENTATION_XLA_DX13_SOURCE_GATHER_DW13,
        ):
            returned = source_push_backward_return(
                w13_grads.dx_expert_major,
                w2_grads.d_route_weight,
                host_inputs.plan,
                src_base_by_expert=host_inputs.src_base_by_expert,
                route_indices=return_route_indices,
                implementation=return_implementation,
                mesh=mesh,
            )
        else:
            d_route_flat = _flatten_expert_blocks_to_flat_rows(
                expert_base,
                w2_grads.d_route_weight,
                flat_rows_per_rank=config.hidden_rows_per_rank,
                valid=valid,
            )
            returned = source_push_backward_return_flat(
                w13_grads.dx_expert_major,
                d_route_flat,
                host_inputs.plan,
                expert_base=expert_base,
                src_base_by_expert=host_inputs.src_base_by_expert,
                route_indices=return_route_indices,
                implementation=return_implementation,
                mesh=mesh,
            )
        output = (
            returned.dx.astype(x.dtype),
            returned.d_route_weights.astype(expert_route_weights.dtype),
            w13_grads.dw13.astype(w13.dtype),
            w2_grads.dw2.astype(w2.dtype),
        )
        _block_until_ready(output)
        if record_stage_times:
            stage_times[BACKWARD_STAGE_DX_COMBINE] = time.perf_counter() - stage_start

        return output, stage_times

    start = time.perf_counter()
    output, _ = call_backward(record_stage_times=False)
    first_call_time = time.perf_counter() - start

    for _ in range(warmup):
        output, _ = call_backward(record_stage_times=False)

    steady_state_times = []
    stage_steady_state_times: dict[str, list[float]] = {stage: [] for stage in stages}
    for _ in range(repeat_runs):
        total_elapsed = 0.0
        stage_elapsed = {stage: 0.0 for stage in stages}
        for _ in range(steps):
            start = time.perf_counter()
            output, step_stage_times = call_backward(record_stage_times=True)
            total_elapsed += time.perf_counter() - start
            for stage in stages:
                stage_elapsed[stage] += step_stage_times[stage]
        steady_state_times.append(total_elapsed / steps)
        for stage in stages:
            stage_steady_state_times[stage].append(stage_elapsed[stage] / steps)

    return BackwardDecomposedTiming(
        first_call_time=first_call_time,
        steady_state_times=steady_state_times,
        output=output,
        stage_steady_state_times=stage_steady_state_times,
    )


def _flatten_expert_blocks_to_flat_rows(
    expert_base: jax.Array,
    blocks: jax.Array,
    *,
    flat_rows_per_rank: int,
    valid: jax.Array,
) -> jax.Array:
    rows_per_expert = valid.shape[-1]
    flat_rows = _expert_flat_rows(expert_base, rows_per_expert)
    dst_index = _dst_indices(expert_base.shape[0], expert_base.shape[1], rows_per_expert)
    out_shape = (expert_base.shape[0], flat_rows_per_rank, *blocks.shape[3:])
    mask = valid.astype(blocks.dtype)
    while mask.ndim < blocks.ndim:
        mask = mask[..., None]
    out = jnp.zeros(out_shape, dtype=blocks.dtype)
    return out.at[dst_index, flat_rows].add(
        blocks * mask,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, *(None for _ in range(blocks.ndim - 2))),
    )


def _source_push_w2_backward_from_flat_h_split_timing(
    expert_base: jax.Array,
    h: jax.Array,
    route_weight: jax.Array,
    dy: jax.Array,
    w2: jax.Array,
    valid: jax.Array,
    *,
    implementation: str,
    matmul_implementation: str | None,
    swiglu_implementation: str | None,
    mesh: Mesh,
    record_stage_times: bool,
    contiguous_expert_gather: bool,
) -> tuple[_SourcePushW2BackwardOutput, dict[str, float]]:
    """Benchmark-only decomposition of staged-flat W2 backward.

    This mirrors ``_source_push_w2_backward_from_flat_h`` but exposes the major
    materialization boundaries. Each substage is explicitly synchronized when
    timing is requested, so use this to locate taxes rather than as a production
    timing path.
    """

    stage_times = {stage: 0.0 for stage in BACKWARD_W2_SPLIT_STAGES}
    if implementation == "reference_matmul_pallas_mgpu_swiglu":
        implementation = "reference"
        matmul_implementation = "reference"
        swiglu_implementation = "pallas_mgpu"
    if implementation != "reference":
        raise ValueError(
            "--backward-w2-split-timing currently supports reference/staged W2 implementations only, "
            f"got {implementation!r}"
        )
    matmul_implementation = matmul_implementation or "reference"
    swiglu_implementation = swiglu_implementation or "reference"

    stage_start = time.perf_counter()
    if contiguous_expert_gather:
        h_blocks = _gather_flat_rows_by_expert_slice(h, expert_base, valid.shape[-1])
        route_weight_blocks = _gather_flat_rows_by_expert_slice(route_weight, expert_base, valid.shape[-1])
        dy_blocks = _gather_flat_rows_by_expert_slice(dy, expert_base, valid.shape[-1])
    else:
        flat_rows = _expert_flat_rows(expert_base, valid.shape[-1])
        h_blocks = _gather_flat_rows(h, flat_rows, fill_value=0)
        route_weight_blocks = _gather_flat_rows(route_weight, flat_rows, fill_value=0)
        dy_blocks = _gather_flat_rows(dy, flat_rows, fill_value=0)
    valid_blocks = _source_push_w2_valid_blocks_sharded(valid)
    if record_stage_times:
        _block_until_ready((h_blocks, route_weight_blocks, dy_blocks, valid_blocks))
        stage_times[BACKWARD_STAGE_W2_GATHER] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    _activation, weighted_activation = _source_push_w2_activation_and_weighted_activation_reference(
        h_blocks,
        route_weight_blocks,
        valid_blocks,
    )
    if record_stage_times:
        _block_until_ready(weighted_activation)
        stage_times[BACKWARD_STAGE_W2_ACTIVATION] = time.perf_counter() - stage_start

    matmul_output, matmul_stage_times = _source_push_w2_matmul_backward_split_timing(
        weighted_activation,
        dy_blocks,
        w2,
        valid_blocks,
        implementation=matmul_implementation,
        mesh=mesh,
        record_stage_times=record_stage_times,
    )
    if record_stage_times:
        _block_until_ready(matmul_output)
        stage_times[BACKWARD_STAGE_W2_MATMUL] = (
            matmul_stage_times[BACKWARD_STAGE_W2_D_WEIGHTED_ACTIVATION] + matmul_stage_times[BACKWARD_STAGE_W2_DW2]
        )
        stage_times.update(matmul_stage_times)

    stage_start = time.perf_counter()
    swiglu_output = _source_push_w2_swiglu_backward(
        h_blocks,
        route_weight_blocks,
        matmul_output.d_weighted_activation,
        valid_blocks,
        implementation=swiglu_implementation,
        mesh=mesh,
    )
    if record_stage_times:
        _block_until_ready(swiglu_output)
        stage_times[BACKWARD_STAGE_W2_SWIGLU] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    valid_f = valid_blocks.astype(swiglu_output.d_h.dtype)
    flat_rows = _expert_flat_rows(expert_base, valid.shape[-1])
    dst_index = _dst_indices(expert_base.shape[0], expert_base.shape[1], valid.shape[-1])
    d_h = jnp.zeros(h.shape, dtype=swiglu_output.d_h.dtype)
    d_route_weight = jnp.zeros(route_weight.shape, dtype=swiglu_output.d_route_weight.dtype)
    d_h = d_h.at[dst_index, flat_rows].add(
        swiglu_output.d_h * valid_f[..., None],
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None),
    )
    d_route_weight = d_route_weight.at[dst_index, flat_rows].add(
        swiglu_output.d_route_weight * valid_f,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None),
    )
    output = _SourcePushW2BackwardOutput(
        d_h=d_h,
        d_route_weight=d_route_weight,
        dw2=matmul_output.dw2,
        d_activation=swiglu_output.d_activation,
    )
    if record_stage_times:
        _block_until_ready(output)
        stage_times[BACKWARD_STAGE_W2_SCATTER] = time.perf_counter() - stage_start
    return output, stage_times


def _source_push_w2_matmul_backward_split_timing(
    weighted_activation: jax.Array,
    dy: jax.Array,
    w2: jax.Array,
    valid: jax.Array,
    *,
    implementation: str,
    mesh: Mesh,
    record_stage_times: bool,
) -> tuple[_SourcePushW2MatmulBackwardOutput, dict[str, float]]:
    stage_times = {
        BACKWARD_STAGE_W2_D_WEIGHTED_ACTIVATION: 0.0,
        BACKWARD_STAGE_W2_DW2: 0.0,
    }
    if implementation == "reference":
        valid_f = valid.astype(jnp.float32)
        weighted_activation = weighted_activation.astype(jnp.float32) * valid_f[..., None]
        dy = dy.astype(jnp.float32) * valid_f[..., None]
        w2 = w2.astype(jnp.float32)

        stage_start = time.perf_counter()
        d_weighted_activation = jnp.einsum("dech,deih->deci", dy, w2)
        if record_stage_times:
            _block_until_ready(d_weighted_activation)
            stage_times[BACKWARD_STAGE_W2_D_WEIGHTED_ACTIVATION] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        dw2 = jnp.einsum("deci,dech->deih", weighted_activation, dy)
        if record_stage_times:
            _block_until_ready(dw2)
            stage_times[BACKWARD_STAGE_W2_DW2] = time.perf_counter() - stage_start
        return _SourcePushW2MatmulBackwardOutput(d_weighted_activation=d_weighted_activation, dw2=dw2), stage_times

    if implementation != "pallas_mgpu":
        output = _source_push_w2_matmul_backward(
            weighted_activation,
            dy,
            w2,
            valid,
            implementation=implementation,
            mesh=mesh,
        )
        return output, stage_times

    original_rows = weighted_activation.shape[2]
    weighted_activation_padded, dy_padded, valid_padded = _pad_w2_matmul_rows_for_pallas(
        weighted_activation,
        dy,
        valid,
        row_multiple=MIN_SOURCE_PUSH_W2_MATMUL_ROW_BLOCK,
    )
    block_sizes = _source_push_w2_matmul_backward_inferred_block_sizes(weighted_activation_padded, dy_padded, w2)
    dy_for_wgmma = dy_padded.astype(w2.dtype)
    weighted_activation_for_wgmma = weighted_activation_padded.astype(w2.dtype)

    stage_start = time.perf_counter()
    d_weighted_activation = _source_push_w2_d_weighted_activation_pallas_call(
        dy_for_wgmma,
        w2,
        valid_padded,
        row_block=block_sizes.row_block,
        intermediate_block=block_sizes.intermediate_block,
        hidden_block=block_sizes.hidden_block,
        interpret=False,
        mesh=mesh,
    )
    d_weighted_activation = d_weighted_activation[:, :, :original_rows, :]
    if record_stage_times:
        _block_until_ready(d_weighted_activation)
        stage_times[BACKWARD_STAGE_W2_D_WEIGHTED_ACTIVATION] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    dw2 = _source_push_w2_dw2_pallas_call(
        weighted_activation_for_wgmma,
        dy_for_wgmma,
        valid_padded,
        row_block=block_sizes.row_block,
        intermediate_block=block_sizes.intermediate_block,
        hidden_block=block_sizes.hidden_block,
        interpret=False,
        mesh=mesh,
    )
    if record_stage_times:
        _block_until_ready(dw2)
        stage_times[BACKWARD_STAGE_W2_DW2] = time.perf_counter() - stage_start
    return _SourcePushW2MatmulBackwardOutput(d_weighted_activation=d_weighted_activation, dw2=dw2), stage_times


def _resolve_backward_stage_implementation(requested: str, default_implementation: str) -> str:
    if requested == BACKWARD_IMPLEMENTATION_DEFAULT:
        return default_implementation
    return requested


def _resolve_w2_backward_implementations(
    requested: str,
    default_implementation: str,
) -> tuple[str, str | None, str | None]:
    if requested == "pallas_mgpu_matmul_reference_swiglu":
        return "reference", "pallas_mgpu", "reference"
    if requested == "pallas_mgpu_matmul_pallas_mgpu_swiglu":
        return "reference", "pallas_mgpu", "pallas_mgpu"
    return _resolve_backward_stage_implementation(requested, default_implementation), None, None


def _staged_flat_backward_stages(stop_after_stage: str) -> tuple[str, ...]:
    stages = (
        BACKWARD_STAGE_DY_ROUTE,
        BACKWARD_STAGE_W2,
        BACKWARD_STAGE_W13,
        BACKWARD_STAGE_DX_COMBINE,
    )
    if stop_after_stage == BACKWARD_STOP_AFTER_NONE:
        return stages
    if stop_after_stage not in stages:
        raise ValueError(
            f"backward_stop_after_stage must be one of {BACKWARD_STOP_AFTER_STAGES}, got {stop_after_stage!r}"
        )
    return stages[: stages.index(stop_after_stage) + 1]


def _time_source_push_input_pack(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    x: jax.Array,
    route_weights: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> InputPackTiming:
    def pack_inputs() -> SourcePushForwardDeviceInputs:
        with jax.set_mesh(mesh):
            packed = device_source_push_forward_inputs_from_plan(config, host_inputs, x, route_weights, w13, w2)
            packed = _shard_source_push_forward_inputs(mesh, packed)
        return _block_source_push_forward_device_inputs(packed)

    def pack_inputs_with_stage_times() -> tuple[SourcePushForwardDeviceInputs, dict[str, float]]:
        stage_times = {}
        with jax.set_mesh(mesh):
            stage_start = time.perf_counter()
            packed_x = pack_source_push_tokens_jax(x, host_inputs.plan).astype(jnp.bfloat16)
            _block_until_ready(packed_x)
            stage_times[FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_TOKEN_PACK] = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            h_route_weights = source_push_h_row_route_weights_jax(
                route_weights,
                host_inputs.plan,
                host_inputs.send_meta,
                host_inputs.expert_base,
                host_inputs.src_base_by_expert,
                hidden_rows_per_rank=config.hidden_rows_per_rank,
                use_exact_expert_major=host_inputs.use_exact_expert_major,
            ).astype(jnp.bfloat16)
            _block_until_ready(h_route_weights)
            stage_times[FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_H_ROUTE_WEIGHTS] = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            packed = SourcePushForwardDeviceInputs(
                x=packed_x,
                send_meta=jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
                recv_meta=jnp.asarray(host_inputs.recv_meta, dtype=jnp.int32),
                expert_base=jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
                src_base_by_expert=jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
                w_gate_up=jnp.asarray(w13, dtype=jnp.bfloat16),
                w_down=jnp.asarray(w2, dtype=jnp.bfloat16),
                queue_dst_ord=jnp.asarray(host_inputs.queue_dst_ord, dtype=jnp.int32),
                queue_entry=jnp.asarray(host_inputs.queue_entry, dtype=jnp.int32),
                queue_row=jnp.asarray(host_inputs.queue_row, dtype=jnp.int32),
                h_route_weights=h_route_weights,
                route_combine_weights=jnp.asarray(host_inputs.route_combine_weights, dtype=jnp.bfloat16),
                route_valid_mask=jnp.asarray(host_inputs.route_valid_mask, dtype=jnp.bool_),
                queue_stats=host_inputs.queue_stats,
                use_exact_expert_major=host_inputs.use_exact_expert_major,
            )
            packed = _shard_source_push_forward_inputs(mesh, packed)
        packed = _block_source_push_forward_device_inputs(packed)
        stage_times[FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_STATIC_SHARD] = time.perf_counter() - stage_start
        return packed, stage_times

    start = time.perf_counter()
    output = pack_inputs()
    first_call_time = time.perf_counter() - start

    for _ in range(warmup):
        output = pack_inputs()

    steady_state_times = []
    stage_steady_state_times = {
        FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_TOKEN_PACK: [],
        FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_H_ROUTE_WEIGHTS: [],
        FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_STATIC_SHARD: [],
    }
    for _ in range(repeat_runs):
        start = time.perf_counter()
        for _ in range(steps):
            output = pack_inputs()
        steady_state_times.append((time.perf_counter() - start) / steps)

        stage_elapsed = {stage: 0.0 for stage in stage_steady_state_times}
        for _ in range(steps):
            _, step_stage_times = pack_inputs_with_stage_times()
            for stage, elapsed in step_stage_times.items():
                stage_elapsed[stage] += elapsed
        for stage in stage_steady_state_times:
            stage_steady_state_times[stage].append(stage_elapsed[stage] / steps)

    return InputPackTiming(
        first_call_time=first_call_time,
        steady_state_times=steady_state_times,
        output=output,
        stage_steady_state_times=stage_steady_state_times,
    )


def _time_source_push_compact_h_input_pack(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    route_table,
    x: jax.Array,
    route_weights: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    compact_expert_capacity: int,
    use_pallas_token_pack: bool,
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> InputPackTiming:
    def pack_inputs() -> tuple[SourcePushForwardDeviceInputs, jax.Array]:
        return _pack_compact_h_probe_total(
            config,
            mesh,
            host_inputs,
            route_table,
            x,
            route_weights,
            w13,
            w2,
            compact_expert_capacity=compact_expert_capacity,
            use_pallas_token_pack=use_pallas_token_pack,
        )

    def pack_inputs_with_stage_times() -> tuple[tuple[SourcePushForwardDeviceInputs, jax.Array], dict[str, float]]:
        stage_times = {}
        with jax.set_mesh(mesh):
            stage_start = time.perf_counter()
            if use_pallas_token_pack:
                packed_x = source_push_pack_tokens_pallas_mgpu(x, host_inputs.plan, mesh=mesh)
            else:
                packed_x = pack_source_push_tokens_jax(x, host_inputs.plan).astype(jnp.bfloat16)
            _block_until_ready(packed_x)
            stage_times[FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_TOKEN_PACK] = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            expert_route_weights = source_push_mlp._source_push_mlp_route_weights_to_all_expert_major(
                route_table,
                route_weights,
            )
            h_route_weights = source_push_mlp._source_push_mlp_pad_expert_route_weights(
                expert_route_weights,
                compact_expert_capacity,
            ).astype(jnp.bfloat16)
            _block_until_ready(h_route_weights)
            stage_times[FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_COMPACT_H_ROUTE_WEIGHTS] = (
                time.perf_counter() - stage_start
            )

            stage_start = time.perf_counter()
            packed = _pack_compact_h_static_inputs(mesh, host_inputs, packed_x, h_route_weights, w13, w2, config)
        output = _block_compact_h_forward_inputs(packed, h_route_weights)
        stage_times[FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_STATIC_SHARD] = time.perf_counter() - stage_start
        return output, stage_times

    start = time.perf_counter()
    output = pack_inputs()
    first_call_time = time.perf_counter() - start

    for _ in range(warmup):
        output = pack_inputs()

    steady_state_times = []
    stage_steady_state_times = {
        FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_TOKEN_PACK: [],
        FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_COMPACT_H_ROUTE_WEIGHTS: [],
        FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_STATIC_SHARD: [],
    }
    for _ in range(repeat_runs):
        start = time.perf_counter()
        for _ in range(steps):
            output = pack_inputs()
        steady_state_times.append((time.perf_counter() - start) / steps)

        stage_elapsed = {stage: 0.0 for stage in stage_steady_state_times}
        for _ in range(steps):
            _, step_stage_times = pack_inputs_with_stage_times()
            for stage, elapsed in step_stage_times.items():
                stage_elapsed[stage] += elapsed
        for stage in stage_steady_state_times:
            stage_steady_state_times[stage].append(stage_elapsed[stage] / steps)

    return InputPackTiming(
        first_call_time=first_call_time,
        steady_state_times=steady_state_times,
        output=output,
        stage_steady_state_times=stage_steady_state_times,
    )


def _time_source_push_raw_token_input_prepare(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    x: jax.Array,
    route_weights: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> InputPackTiming:
    def prepare_inputs() -> RawTokenForwardInputs:
        with jax.set_mesh(mesh):
            h_route_weights = source_push_h_row_route_weights_jax(
                route_weights,
                host_inputs.plan,
                host_inputs.send_meta,
                host_inputs.expert_base,
                host_inputs.src_base_by_expert,
                hidden_rows_per_rank=config.hidden_rows_per_rank,
                use_exact_expert_major=host_inputs.use_exact_expert_major,
            )
            prepared = RawTokenForwardInputs(
                x=x.astype(jnp.bfloat16),
                token_ids=jnp.asarray(host_inputs.plan.token_ids, dtype=jnp.int32),
                send_meta=jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
                recv_meta=jnp.asarray(host_inputs.recv_meta, dtype=jnp.int32),
                expert_base=jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
                src_base_by_expert=jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
                w_gate_up=jnp.asarray(w13, dtype=jnp.bfloat16),
                h_route_weights=h_route_weights.astype(jnp.bfloat16),
                w_down=jnp.asarray(w2, dtype=jnp.bfloat16),
                queue_dst_ord=jnp.asarray(host_inputs.queue_dst_ord, dtype=jnp.int32),
                queue_entry=jnp.asarray(host_inputs.queue_entry, dtype=jnp.int32),
                queue_row=jnp.asarray(host_inputs.queue_row, dtype=jnp.int32),
                route_combine_weights=jnp.asarray(host_inputs.route_combine_weights, dtype=jnp.bfloat16),
                route_valid_mask=jnp.asarray(host_inputs.route_valid_mask, dtype=jnp.bool_),
                use_exact_expert_major=host_inputs.use_exact_expert_major,
            )
            prepared = _shard_raw_token_forward_inputs(mesh, prepared)
        return _block_raw_token_forward_inputs(prepared)

    start = time.perf_counter()
    output = prepare_inputs()
    first_call_time = time.perf_counter() - start

    for _ in range(warmup):
        output = prepare_inputs()

    steady_state_times = []
    for _ in range(repeat_runs):
        start = time.perf_counter()
        for _ in range(steps):
            output = prepare_inputs()
        steady_state_times.append((time.perf_counter() - start) / steps)

    return InputPackTiming(
        first_call_time=first_call_time,
        steady_state_times=steady_state_times,
        output=output,
    )


def _time_source_push_raw_token_compact_h_input_prepare(
    *,
    mesh: Mesh,
    host_inputs,
    route_table,
    x: jax.Array,
    route_weights: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    compact_expert_capacity: int,
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> InputPackTiming:
    def prepare_inputs() -> RawTokenCompactHForwardInputs:
        with jax.set_mesh(mesh):
            expert_route_weights = source_push_mlp._source_push_mlp_route_weights_to_all_expert_major(
                route_table,
                route_weights,
            )
            h_route_weights = source_push_mlp._source_push_mlp_pad_expert_route_weights(
                expert_route_weights,
                compact_expert_capacity,
            ).astype(jnp.bfloat16)
            prepared = RawTokenCompactHForwardInputs(
                x=x.astype(jnp.bfloat16),
                token_ids=jnp.asarray(host_inputs.plan.token_ids, dtype=jnp.int32),
                send_meta=jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
                recv_meta=jnp.asarray(host_inputs.recv_meta, dtype=jnp.int32),
                expert_base=jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
                src_base_by_expert=jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
                w_gate_up=jnp.asarray(w13, dtype=jnp.bfloat16),
                h_route_weights=h_route_weights,
                w_down=jnp.asarray(w2, dtype=jnp.bfloat16),
                queue_dst_ord=jnp.asarray(host_inputs.queue_dst_ord, dtype=jnp.int32),
                queue_entry=jnp.asarray(host_inputs.queue_entry, dtype=jnp.int32),
                queue_row=jnp.asarray(host_inputs.queue_row, dtype=jnp.int32),
                route_combine_weights=jnp.asarray(host_inputs.route_combine_weights, dtype=jnp.bfloat16),
                route_valid_mask=jnp.asarray(host_inputs.route_valid_mask, dtype=jnp.bool_),
                use_exact_expert_major=host_inputs.use_exact_expert_major,
            )
            prepared = _shard_raw_token_compact_h_forward_inputs(mesh, prepared)
        return _block_raw_token_compact_h_forward_inputs(prepared)

    def prepare_inputs_with_stage_times() -> tuple[RawTokenCompactHForwardInputs, dict[str, float]]:
        stage_times = {}
        with jax.set_mesh(mesh):
            stage_start = time.perf_counter()
            expert_route_weights = source_push_mlp._source_push_mlp_route_weights_to_all_expert_major(
                route_table,
                route_weights,
            )
            h_route_weights = source_push_mlp._source_push_mlp_pad_expert_route_weights(
                expert_route_weights,
                compact_expert_capacity,
            ).astype(jnp.bfloat16)
            _block_until_ready(h_route_weights)
            stage_times[FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_COMPACT_H_ROUTE_WEIGHTS] = (
                time.perf_counter() - stage_start
            )

            stage_start = time.perf_counter()
            prepared = RawTokenCompactHForwardInputs(
                x=x.astype(jnp.bfloat16),
                token_ids=jnp.asarray(host_inputs.plan.token_ids, dtype=jnp.int32),
                send_meta=jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
                recv_meta=jnp.asarray(host_inputs.recv_meta, dtype=jnp.int32),
                expert_base=jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
                src_base_by_expert=jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
                w_gate_up=jnp.asarray(w13, dtype=jnp.bfloat16),
                h_route_weights=h_route_weights,
                w_down=jnp.asarray(w2, dtype=jnp.bfloat16),
                queue_dst_ord=jnp.asarray(host_inputs.queue_dst_ord, dtype=jnp.int32),
                queue_entry=jnp.asarray(host_inputs.queue_entry, dtype=jnp.int32),
                queue_row=jnp.asarray(host_inputs.queue_row, dtype=jnp.int32),
                route_combine_weights=jnp.asarray(host_inputs.route_combine_weights, dtype=jnp.bfloat16),
                route_valid_mask=jnp.asarray(host_inputs.route_valid_mask, dtype=jnp.bool_),
                use_exact_expert_major=host_inputs.use_exact_expert_major,
            )
            prepared = _shard_raw_token_compact_h_forward_inputs(mesh, prepared)
        prepared = _block_raw_token_compact_h_forward_inputs(prepared)
        stage_times[FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_STATIC_SHARD] = time.perf_counter() - stage_start
        return prepared, stage_times

    start = time.perf_counter()
    output = prepare_inputs()
    first_call_time = time.perf_counter() - start

    for _ in range(warmup):
        output = prepare_inputs()

    steady_state_times = []
    stage_steady_state_times = {
        FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_COMPACT_H_ROUTE_WEIGHTS: [],
        FORWARD_DECOMPOSED_STAGE_PACK_INPUTS_STATIC_SHARD: [],
    }
    for _ in range(repeat_runs):
        start = time.perf_counter()
        for _ in range(steps):
            output = prepare_inputs()
        steady_state_times.append((time.perf_counter() - start) / steps)

        stage_elapsed = {stage: 0.0 for stage in stage_steady_state_times}
        for _ in range(steps):
            _, step_stage_times = prepare_inputs_with_stage_times()
            for stage, elapsed in step_stage_times.items():
                stage_elapsed[stage] += elapsed
        for stage in stage_steady_state_times:
            stage_steady_state_times[stage].append(stage_elapsed[stage] / steps)

    return InputPackTiming(
        first_call_time=first_call_time,
        steady_state_times=steady_state_times,
        output=output,
        stage_steady_state_times=stage_steady_state_times,
    )


def _shard_raw_token_forward_inputs(mesh: Mesh, inputs: RawTokenForwardInputs) -> RawTokenForwardInputs:
    return RawTokenForwardInputs(
        x=jax.device_put(inputs.x, NamedSharding(mesh, P(AXIS, None, None))),
        token_ids=jax.device_put(inputs.token_ids, NamedSharding(mesh, P(AXIS, None, None, None))),
        send_meta=jax.device_put(inputs.send_meta, NamedSharding(mesh, P(AXIS, None, None, None))),
        recv_meta=jax.device_put(inputs.recv_meta, NamedSharding(mesh, P(AXIS, None, None, None))),
        expert_base=jax.device_put(inputs.expert_base, NamedSharding(mesh, P(AXIS, None))),
        src_base_by_expert=jax.device_put(inputs.src_base_by_expert, NamedSharding(mesh, P(AXIS, None, None))),
        w_gate_up=jax.device_put(inputs.w_gate_up, NamedSharding(mesh, P(AXIS, None, None, None))),
        h_route_weights=jax.device_put(inputs.h_route_weights, NamedSharding(mesh, P(AXIS, None))),
        w_down=jax.device_put(inputs.w_down, NamedSharding(mesh, P(AXIS, None, None, None))),
        queue_dst_ord=jax.device_put(inputs.queue_dst_ord, NamedSharding(mesh, P(AXIS, None, None))),
        queue_entry=jax.device_put(inputs.queue_entry, NamedSharding(mesh, P(AXIS, None, None))),
        queue_row=jax.device_put(inputs.queue_row, NamedSharding(mesh, P(AXIS, None, None))),
        route_combine_weights=jax.device_put(inputs.route_combine_weights, NamedSharding(mesh, P(AXIS, None, None))),
        route_valid_mask=jax.device_put(inputs.route_valid_mask, NamedSharding(mesh, P(AXIS, None, None))),
        use_exact_expert_major=inputs.use_exact_expert_major,
    )


def _shard_raw_token_compact_h_forward_inputs(
    mesh: Mesh, inputs: RawTokenCompactHForwardInputs
) -> RawTokenCompactHForwardInputs:
    return RawTokenCompactHForwardInputs(
        x=jax.device_put(inputs.x, NamedSharding(mesh, P(AXIS, None, None))),
        token_ids=jax.device_put(inputs.token_ids, NamedSharding(mesh, P(AXIS, None, None, None))),
        send_meta=jax.device_put(inputs.send_meta, NamedSharding(mesh, P(AXIS, None, None, None))),
        recv_meta=jax.device_put(inputs.recv_meta, NamedSharding(mesh, P(AXIS, None, None, None))),
        expert_base=jax.device_put(inputs.expert_base, NamedSharding(mesh, P(AXIS, None))),
        src_base_by_expert=jax.device_put(inputs.src_base_by_expert, NamedSharding(mesh, P(AXIS, None, None))),
        w_gate_up=jax.device_put(inputs.w_gate_up, NamedSharding(mesh, P(AXIS, None, None, None))),
        h_route_weights=jax.device_put(inputs.h_route_weights, NamedSharding(mesh, P(AXIS, None, None))),
        w_down=jax.device_put(inputs.w_down, NamedSharding(mesh, P(AXIS, None, None, None))),
        queue_dst_ord=jax.device_put(inputs.queue_dst_ord, NamedSharding(mesh, P(AXIS, None, None))),
        queue_entry=jax.device_put(inputs.queue_entry, NamedSharding(mesh, P(AXIS, None, None))),
        queue_row=jax.device_put(inputs.queue_row, NamedSharding(mesh, P(AXIS, None, None))),
        route_combine_weights=jax.device_put(inputs.route_combine_weights, NamedSharding(mesh, P(AXIS, None, None))),
        route_valid_mask=jax.device_put(inputs.route_valid_mask, NamedSharding(mesh, P(AXIS, None, None))),
        use_exact_expert_major=inputs.use_exact_expert_major,
    )


def _block_raw_token_forward_inputs(inputs: RawTokenForwardInputs) -> RawTokenForwardInputs:
    _block_until_ready(tuple(value for value in inputs[:-1]))
    return inputs


def _block_raw_token_compact_h_forward_inputs(
    inputs: RawTokenCompactHForwardInputs,
) -> RawTokenCompactHForwardInputs:
    _block_until_ready(tuple(value for value in inputs[:-1]))
    return inputs


def _time_source_push_w13_direct_compact(
    mesh: Mesh,
    config: PushInboxConfig,
    inputs: SourcePushForwardDeviceInputs,
    *,
    compact_expert_capacity: int,
    warmup: int,
    steps: int,
    repeat_runs: int,
    diagnostic_variant: str = DIAGNOSTIC_VARIANT_FULL,
) -> MlpTiming:
    w13_h_fn = jax.jit(
        _sharded_w13_h_compact_kernel(
            mesh,
            config,
            compact_expert_capacity=compact_expert_capacity,
            use_exact_expert_major=inputs.use_exact_expert_major,
            diagnostic_variant=diagnostic_variant,
        )
    )

    def call_w13_compact():
        _, h = w13_h_fn(
            inputs.x,
            inputs.send_meta,
            inputs.recv_meta,
            inputs.expert_base,
            inputs.src_base_by_expert,
            inputs.w_gate_up,
        )
        return h

    start = time.perf_counter()
    output = call_w13_compact()
    _block_until_ready(output)
    compile_time = time.perf_counter() - start

    for _ in range(warmup):
        _block_until_ready(call_w13_compact())

    steady_state_times = []
    for _ in range(repeat_runs):
        start = time.perf_counter()
        for _ in range(steps):
            _block_until_ready(call_w13_compact())
        steady_state_times.append((time.perf_counter() - start) / steps)

    return MlpTiming(
        compile_time=compile_time,
        lower_compile_time=None,
        first_run_time=None,
        first_call_time=compile_time,
        steady_state_times=steady_state_times,
        output=output,
    )


def _time_staged_source_push_forward_raw_tokens(
    mesh: Mesh,
    config: PushInboxConfig,
    inputs: RawTokenForwardInputs,
    *,
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> RawTokenForwardTiming:
    w13_h_fn = jax.jit(
        _sharded_raw_token_w13_h_kernel(
            mesh,
            config,
            use_exact_expert_major=inputs.use_exact_expert_major,
        )
    )
    w2_from_h_return_fn = jax.jit(
        _sharded_w2_from_h_return_direct_to_source_kernel(
            mesh,
            config,
            use_exact_expert_major=inputs.use_exact_expert_major,
        )
    )
    combine_fn = jax.jit(_sharded_source_combine_kernel(mesh, config))

    def call_stages(*, record_stage_times: bool = False):
        stage_times: dict[str, float] = {}

        stage_start = time.perf_counter()
        _, h = w13_h_fn(
            inputs.x,
            inputs.token_ids,
            inputs.send_meta,
            inputs.recv_meta,
            inputs.expert_base,
            inputs.src_base_by_expert,
            inputs.w_gate_up,
        )
        _block_until_ready(h)
        if record_stage_times:
            stage_times[FORWARD_STAGE_W13] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        source_return = w2_from_h_return_fn(
            h,
            inputs.h_route_weights,
            inputs.recv_meta,
            inputs.expert_base,
            inputs.src_base_by_expert,
            inputs.w_down,
        )
        _block_until_ready(source_return)
        if record_stage_times:
            stage_times[FORWARD_STAGE_W2_RETURN] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        out = combine_fn(
            source_return,
            inputs.queue_dst_ord,
            inputs.queue_entry,
            inputs.queue_row,
            jnp.ones_like(inputs.route_combine_weights),
            inputs.route_valid_mask,
        )
        _block_until_ready(out)
        if record_stage_times:
            stage_times[FORWARD_STAGE_COMBINE] = time.perf_counter() - stage_start
        return out, stage_times

    start = time.perf_counter()
    out, stage_compile_times = call_stages(record_stage_times=True)
    compile_time = time.perf_counter() - start

    for _ in range(warmup):
        out, _ = call_stages()

    steady_state_times = []
    stage_steady_state_times: dict[str, list[float]] = {stage: [] for stage in FORWARD_STAGES}
    for _ in range(repeat_runs):
        start = time.perf_counter()
        stage_elapsed = {stage: 0.0 for stage in FORWARD_STAGES}
        for _ in range(steps):
            out, step_stage_times = call_stages(record_stage_times=True)
            for stage in FORWARD_STAGES:
                stage_elapsed[stage] += step_stage_times[stage]
        steady_state_times.append((time.perf_counter() - start) / steps)
        for stage in FORWARD_STAGES:
            stage_steady_state_times[stage].append(stage_elapsed[stage] / steps)

    return RawTokenForwardTiming(
        compile_time=compile_time,
        steady_state_times=steady_state_times,
        output=out,
        stage_steady_state_times=stage_steady_state_times,
        stage_compile_times=stage_compile_times,
    )


def _time_staged_source_push_forward_raw_tokens_compact_h(
    mesh: Mesh,
    config: PushInboxConfig,
    inputs: RawTokenCompactHForwardInputs,
    *,
    compact_expert_capacity: int,
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> RawTokenForwardTiming:
    w13_h_fn = jax.jit(
        _sharded_raw_token_w13_h_compact_kernel(
            mesh,
            config,
            compact_expert_capacity=compact_expert_capacity,
            use_exact_expert_major=inputs.use_exact_expert_major,
        )
    )
    w2_from_h_return_fn = jax.jit(
        _sharded_w2_from_compact_h_return_direct_to_source_kernel(
            mesh,
            config,
            use_exact_expert_major=inputs.use_exact_expert_major,
        )
    )
    combine_fn = jax.jit(_sharded_source_combine_kernel(mesh, config))

    def call_stages(*, record_stage_times: bool = False):
        stage_times: dict[str, float] = {}

        stage_start = time.perf_counter()
        _, h = w13_h_fn(
            inputs.x,
            inputs.token_ids,
            inputs.send_meta,
            inputs.recv_meta,
            inputs.expert_base,
            inputs.src_base_by_expert,
            inputs.w_gate_up,
        )
        _block_until_ready(h)
        if record_stage_times:
            stage_times[FORWARD_STAGE_W13] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        source_return = w2_from_h_return_fn(
            h,
            inputs.h_route_weights,
            inputs.recv_meta,
            inputs.expert_base,
            inputs.src_base_by_expert,
            inputs.w_down,
        )
        _block_until_ready(source_return)
        if record_stage_times:
            stage_times[FORWARD_STAGE_W2_RETURN] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        out = combine_fn(
            source_return,
            inputs.queue_dst_ord,
            inputs.queue_entry,
            inputs.queue_row,
            jnp.ones_like(inputs.route_combine_weights),
            inputs.route_valid_mask,
        )
        _block_until_ready(out)
        if record_stage_times:
            stage_times[FORWARD_STAGE_COMBINE] = time.perf_counter() - stage_start

        return out, stage_times

    start = time.perf_counter()
    out, stage_compile_times = call_stages(record_stage_times=True)
    compile_time = time.perf_counter() - start

    for _ in range(warmup):
        out, _ = call_stages()

    steady_state_times = []
    stage_steady_state_times: dict[str, list[float]] = {stage: [] for stage in FORWARD_STAGES}
    for _ in range(repeat_runs):
        start = time.perf_counter()
        stage_elapsed = {stage: 0.0 for stage in FORWARD_STAGES}
        for _ in range(steps):
            out, step_stage_times = call_stages(record_stage_times=True)
            for stage in FORWARD_STAGES:
                stage_elapsed[stage] += step_stage_times[stage]
        steady_state_times.append((time.perf_counter() - start) / steps)
        for stage in FORWARD_STAGES:
            stage_steady_state_times[stage].append(stage_elapsed[stage] / steps)

    return RawTokenForwardTiming(
        compile_time=compile_time,
        steady_state_times=steady_state_times,
        output=out,
        stage_steady_state_times=stage_steady_state_times,
        stage_compile_times=stage_compile_times,
    )


def _time_staged_source_push_forward_compact_h(
    mesh: Mesh,
    config: PushInboxConfig,
    inputs: SourcePushForwardDeviceInputs,
    h_route_weights: jax.Array,
    *,
    compact_expert_capacity: int,
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> RawTokenForwardTiming:
    w13_h_fn = jax.jit(
        _sharded_w13_h_compact_kernel(
            mesh,
            config,
            compact_expert_capacity=compact_expert_capacity,
            use_exact_expert_major=inputs.use_exact_expert_major,
        )
    )
    w2_from_h_return_fn = jax.jit(
        _sharded_w2_from_compact_h_return_direct_to_source_kernel(
            mesh,
            config,
            use_exact_expert_major=inputs.use_exact_expert_major,
        )
    )
    combine_fn = jax.jit(_sharded_source_combine_kernel(mesh, config))

    def call_stages(*, record_stage_times: bool = False):
        stage_times: dict[str, float] = {}

        stage_start = time.perf_counter()
        _, h = w13_h_fn(
            inputs.x,
            inputs.send_meta,
            inputs.recv_meta,
            inputs.expert_base,
            inputs.src_base_by_expert,
            inputs.w_gate_up,
        )
        _block_until_ready(h)
        if record_stage_times:
            stage_times[FORWARD_STAGE_W13] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        source_return = w2_from_h_return_fn(
            h,
            h_route_weights,
            inputs.recv_meta,
            inputs.expert_base,
            inputs.src_base_by_expert,
            inputs.w_down,
        )
        _block_until_ready(source_return)
        if record_stage_times:
            stage_times[FORWARD_STAGE_W2_RETURN] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        out = combine_fn(
            source_return,
            inputs.queue_dst_ord,
            inputs.queue_entry,
            inputs.queue_row,
            jnp.ones_like(inputs.route_combine_weights),
            inputs.route_valid_mask,
        )
        _block_until_ready(out)
        if record_stage_times:
            stage_times[FORWARD_STAGE_COMBINE] = time.perf_counter() - stage_start
        return out, stage_times

    start = time.perf_counter()
    out, stage_compile_times = call_stages(record_stage_times=True)
    compile_time = time.perf_counter() - start

    for _ in range(warmup):
        out, _ = call_stages()

    steady_state_times = []
    stage_steady_state_times: dict[str, list[float]] = {stage: [] for stage in FORWARD_STAGES}
    for _ in range(repeat_runs):
        start = time.perf_counter()
        stage_elapsed = {stage: 0.0 for stage in FORWARD_STAGES}
        for _ in range(steps):
            out, step_stage_times = call_stages(record_stage_times=True)
            for stage in FORWARD_STAGES:
                stage_elapsed[stage] += step_stage_times[stage]
        steady_state_times.append((time.perf_counter() - start) / steps)
        for stage in FORWARD_STAGES:
            stage_steady_state_times[stage].append(stage_elapsed[stage] / steps)

    return RawTokenForwardTiming(
        compile_time=compile_time,
        steady_state_times=steady_state_times,
        output=out,
        stage_steady_state_times=stage_steady_state_times,
        stage_compile_times=stage_compile_times,
    )


def _block_source_push_forward_device_inputs(inputs: SourcePushForwardDeviceInputs) -> SourcePushForwardDeviceInputs:
    _block_until_ready(
        (
            inputs.x,
            inputs.send_meta,
            inputs.recv_meta,
            inputs.expert_base,
            inputs.src_base_by_expert,
            inputs.w_gate_up,
            inputs.w_down,
            inputs.queue_dst_ord,
            inputs.queue_entry,
            inputs.queue_row,
            inputs.h_route_weights,
            inputs.route_combine_weights,
            inputs.route_valid_mask,
        )
    )
    return inputs


def _decomposed_forward_rows(
    config: PushInboxConfig,
    *,
    pack_timing: InputPackTiming,
    staged_timing,
    queue_stats: dict[str, Any],
    repeat_runs: int,
    mode: str,
    input_stage: str,
) -> list[dict[str, Any]]:
    if staged_timing.stage_steady_state_times is None:
        raise ValueError("decomposed forward requires staged source-push timing")

    useful_forward_flops, rounded_forward_flops = _forward_flops_per_rank(config, queue_stats)
    bytes_per_rank = _forward_bytes_per_rank(config, queue_stats)
    dropped_routes = int(jax.device_get(queue_stats["dropped_routes"]))
    rows = []

    for repeat_run, pack_time in enumerate(pack_timing.steady_state_times):
        staged_total_time = staged_timing.steady_state_times[repeat_run]
        total_time = pack_time + staged_total_time
        rows.append(
            _decomposed_forward_row(
                config,
                queue_stats=queue_stats,
                repeat_run=repeat_run,
                repeat_runs=repeat_runs,
                stage=FORWARD_STAGE_TOTAL,
                steady_state_time=total_time,
                first_call_time=pack_timing.first_call_time + staged_timing.compile_time,
                compile_time=staged_timing.compile_time,
                bytes_per_rank=bytes_per_rank,
                useful_forward_flops=useful_forward_flops,
                rounded_forward_flops=rounded_forward_flops,
                dropped_routes=dropped_routes,
                mode=mode,
            )
        )
        rows.append(
            _decomposed_forward_row(
                config,
                queue_stats=queue_stats,
                repeat_run=repeat_run,
                repeat_runs=repeat_runs,
                stage=input_stage,
                steady_state_time=pack_time,
                first_call_time=pack_timing.first_call_time,
                compile_time=None,
                bytes_per_rank=bytes_per_rank,
                useful_forward_flops=None,
                rounded_forward_flops=None,
                dropped_routes=dropped_routes,
                mode=mode,
            )
        )
        if pack_timing.stage_steady_state_times is not None:
            for stage, stage_times in pack_timing.stage_steady_state_times.items():
                rows.append(
                    _decomposed_forward_row(
                        config,
                        queue_stats=queue_stats,
                        repeat_run=repeat_run,
                        repeat_runs=repeat_runs,
                        stage=stage,
                        steady_state_time=stage_times[repeat_run],
                        first_call_time=None,
                        compile_time=None,
                        bytes_per_rank=None,
                        useful_forward_flops=None,
                        rounded_forward_flops=None,
                        dropped_routes=dropped_routes,
                        mode=mode,
                    )
                )
        for stage in FORWARD_STAGES:
            stage_time = staged_timing.stage_steady_state_times[stage][repeat_run]
            rows.append(
                _decomposed_forward_row(
                    config,
                    queue_stats=queue_stats,
                    repeat_run=repeat_run,
                    repeat_runs=repeat_runs,
                    stage=stage,
                    steady_state_time=stage_time,
                    first_call_time=None,
                    compile_time=(
                        None if staged_timing.stage_compile_times is None else staged_timing.stage_compile_times[stage]
                    ),
                    bytes_per_rank=_stage_bytes_per_rank(config, queue_stats, stage),
                    useful_forward_flops=_stage_useful_flops_per_rank(config, queue_stats, stage),
                    rounded_forward_flops=_stage_rounded_flops_per_rank(config, queue_stats, stage),
                    dropped_routes=dropped_routes,
                    mode=mode,
                )
            )

    grouped_rows = []
    pack_substages = tuple(pack_timing.stage_steady_state_times or ())
    for stage in (FORWARD_STAGE_TOTAL, input_stage, *pack_substages, *FORWARD_STAGES):
        stage_rows = [row for row in rows if row["stage"] == stage]
        grouped_rows.extend(stage_rows)
        grouped_rows.append(_summary_row(stage_rows))
    return grouped_rows


def _decomposed_forward_row(
    config: PushInboxConfig,
    *,
    queue_stats: dict[str, Any],
    repeat_run: int,
    repeat_runs: int,
    stage: str,
    steady_state_time: float,
    first_call_time: float | None,
    compile_time: float | None,
    bytes_per_rank: float | None,
    useful_forward_flops: float | None,
    rounded_forward_flops: float | None,
    dropped_routes: int,
    mode: str,
) -> dict[str, Any]:
    has_positive_time = steady_state_time > 0.0
    useful_tflops = (
        None
        if useful_forward_flops is None or not has_positive_time
        else useful_forward_flops / steady_state_time / 1e12
    )
    rounded_tflops = (
        None
        if rounded_forward_flops is None or not has_positive_time
        else rounded_forward_flops / steady_state_time / 1e12
    )
    is_w13_stage = stage == FORWARD_STAGE_W13
    h_layout = (
        "direct_padded_expert_major"
        if mode
        in (
            MODE_FORWARD_COMPACT_H_DECOMPOSED,
            MODE_FORWARD_COMPACT_H_DECOMPOSED_WITH_PREP,
            MODE_FORWARD_COMPACT_H_RAW_TOKENS_DECOMPOSED,
        )
        else "flat_expert_major"
    )
    return {
        "kernel": KERNEL_NAME,
        "implementation": (
            BACKEND_SOURCE_PUSH_PALLAS if stage == FORWARD_STAGE_TOTAL else f"{BACKEND_SOURCE_PUSH_PALLAS}_{stage}"
        ),
        "backend": BACKEND_SOURCE_PUSH_PALLAS,
        "mode": mode,
        "stage": stage,
        "row_type": "repeat",
        "config": asdict(config),
        "queue_stats": queue_stats,
        **queue_stats,
        "outer_jit": False,
        "compile_time": compile_time,
        "lower_compile_time": None,
        "first_run_time": None,
        "first_call_time": first_call_time,
        "repeat_run": repeat_run,
        "repeat_runs": repeat_runs,
        "steady_state_time": steady_state_time,
        "bytes_per_rank": bytes_per_rank,
        "forward_gbps_per_rank": (
            None if bytes_per_rank is None or not has_positive_time else bytes_per_rank / steady_state_time / 1e9
        ),
        "useful_forward_tflops_per_rank": useful_tflops,
        "rounded_forward_tflops_per_rank": rounded_tflops,
        "useful_fwd_bwd_tflops_per_rank": None,
        "rounded_fwd_bwd_tflops_per_rank": None,
        "useful_backward_tflops_per_rank": None,
        "rounded_backward_tflops_per_rank": None,
        "useful_tflops_per_rank": useful_tflops,
        "rounded_tflops_per_rank": rounded_tflops,
        "w13_baseline_tflops_per_rank": (SOURCE_PUSH_W13_STABLE_BASELINE_TFLOPS_PER_RANK if is_w13_stage else None),
        "passes_w13_216_949_gate": (
            None
            if not is_w13_stage or useful_tflops is None
            else useful_tflops >= SOURCE_PUSH_W13_STABLE_BASELINE_TFLOPS_PER_RANK
        ),
        "h_layout": h_layout if is_w13_stage else None,
        "dropped_routes": dropped_routes,
        "error": None,
        "error_type": None,
        "error_message": None,
    }


def _w13_direct_compact_rows(
    config: PushInboxConfig,
    *,
    timing: MlpTiming,
    queue_stats: dict[str, Any],
    compact_expert_capacity: int,
    repeat_runs: int,
    mode: str = MODE_FORWARD_W13_DIRECT_COMPACT,
    diagnostic_variant: str = DIAGNOSTIC_VARIANT_FULL,
) -> list[dict[str, Any]]:
    useful_flops = _stage_useful_flops_per_rank(config, queue_stats, FORWARD_STAGE_W13)
    rounded_flops = _stage_rounded_flops_per_rank(config, queue_stats, FORWARD_STAGE_W13)
    bytes_per_rank = _stage_bytes_per_rank(config, queue_stats, FORWARD_STAGE_W13)
    if useful_flops is None or rounded_flops is None:
        raise ValueError("W13 direct compact rows require W13 flop accounting")

    dropped_routes = int(jax.device_get(queue_stats["dropped_routes"]))
    byte_breakdown = _w13_direct_compact_bytes_per_rank(config, queue_stats)
    rows = []
    for repeat_run, steady_state_time in enumerate(timing.steady_state_times):
        useful_tflops = useful_flops / steady_state_time / 1e12
        rounded_tflops = rounded_flops / steady_state_time / 1e12
        total_estimated_gbps = byte_breakdown["w13_estimated_total_bytes_per_rank"] / steady_state_time / 1e9
        rows.append(
            {
                "kernel": KERNEL_NAME,
                "implementation": f"{BACKEND_SOURCE_PUSH_PALLAS}_w13_direct_compact",
                "backend": BACKEND_SOURCE_PUSH_PALLAS,
                "mode": mode,
                "stage": FORWARD_STAGE_W13,
                "row_type": "repeat",
                "diagnostic_variant": diagnostic_variant,
                "diagnostic": diagnostic_variant != DIAGNOSTIC_VARIANT_FULL,
                "config": asdict(config),
                "queue_stats": queue_stats,
                **queue_stats,
                "outer_jit": False,
                "compile_time": timing.compile_time,
                "lower_compile_time": timing.lower_compile_time,
                "first_run_time": timing.first_run_time,
                "first_call_time": timing.first_call_time,
                "repeat_run": repeat_run,
                "repeat_runs": repeat_runs,
                "steady_state_time": steady_state_time,
                "bytes_per_rank": bytes_per_rank,
                "forward_gbps_per_rank": None if bytes_per_rank is None else bytes_per_rank / steady_state_time / 1e9,
                **byte_breakdown,
                "w13_estimated_total_gbps_per_rank": total_estimated_gbps,
                "useful_forward_tflops_per_rank": useful_tflops,
                "rounded_forward_tflops_per_rank": rounded_tflops,
                "useful_fwd_bwd_tflops_per_rank": None,
                "rounded_fwd_bwd_tflops_per_rank": None,
                "useful_backward_tflops_per_rank": None,
                "rounded_backward_tflops_per_rank": None,
                "useful_tflops_per_rank": useful_tflops,
                "rounded_tflops_per_rank": rounded_tflops,
                "w13_baseline_tflops_per_rank": SOURCE_PUSH_W13_STABLE_BASELINE_TFLOPS_PER_RANK,
                "passes_w13_216_949_gate": useful_tflops >= SOURCE_PUSH_W13_STABLE_BASELINE_TFLOPS_PER_RANK,
                "h_layout": "direct_padded_expert_major",
                "compact_h_layout": "direct_padded_expert_major",
                "compact_expert_capacity": compact_expert_capacity,
                "dropped_routes": dropped_routes,
                "error": None,
                "error_type": None,
                "error_message": None,
            }
        )
    return [*rows, _summary_row(rows)]


def _w13_direct_compact_bytes_per_rank(config: PushInboxConfig, queue_stats: dict[str, Any]) -> dict[str, float]:
    """Approximate W13 direct-compact GMEM traffic per rank."""

    rounded_rows = float(queue_stats["rounded_rows_per_rank_mean"])
    send_rounded_rows = float(queue_stats["send_rounded_rows_per_rank_mean"])
    live_entries = float(queue_stats["live_entries_per_rank_mean"])
    n_tiles = config.intermediate_dim // config.block_n
    n_work_groups = n_tiles // config.n_group

    payload_send_bytes = send_rounded_rows * config.hidden_dim * BYTES_PER_BF16
    lhs_compute_read_bytes = rounded_rows * config.hidden_dim * BYTES_PER_BF16 * n_work_groups
    weight_read_bytes = live_entries * 2 * config.hidden_dim * config.intermediate_dim * BYTES_PER_BF16
    compact_h_store_bytes = rounded_rows * 2 * config.intermediate_dim * BYTES_PER_BF16
    estimated_total_bytes = payload_send_bytes + lhs_compute_read_bytes + weight_read_bytes + compact_h_store_bytes
    return {
        "w13_payload_send_bytes_per_rank": float(payload_send_bytes),
        "w13_lhs_compute_read_bytes_per_rank": float(lhs_compute_read_bytes),
        "w13_weight_read_bytes_per_rank": float(weight_read_bytes),
        "w13_compact_h_store_bytes_per_rank": float(compact_h_store_bytes),
        "w13_estimated_total_bytes_per_rank": float(estimated_total_bytes),
    }


def _decomposed_backward_rows(
    config: PushInboxConfig,
    *,
    timing: BackwardDecomposedTiming,
    queue_stats: dict[str, Any],
    repeat_runs: int,
    dropped_routes: int,
    forward_h_first_call_time: float,
    forward_h_times: list[float],
    mode: str = MODE_BACKWARD_DECOMPOSED,
    stages: Sequence[str] = BACKWARD_STAGES,
    w13_backward_component: str | None = None,
) -> list[dict[str, Any]]:
    useful_backward_flops, rounded_backward_flops = _backward_flops_per_rank(config, queue_stats)
    rows = []
    for repeat_run, backward_time in enumerate(timing.steady_state_times):
        rows.append(
            _decomposed_backward_row(
                config,
                queue_stats=queue_stats,
                repeat_run=repeat_run,
                repeat_runs=repeat_runs,
                stage=BACKWARD_STAGE_TOTAL,
                steady_state_time=backward_time,
                first_call_time=timing.first_call_time,
                useful_backward_flops=useful_backward_flops,
                rounded_backward_flops=rounded_backward_flops,
                dropped_routes=dropped_routes,
                mode=mode,
            )
        )
        rows.append(
            _decomposed_backward_row(
                config,
                queue_stats=queue_stats,
                repeat_run=repeat_run,
                repeat_runs=repeat_runs,
                stage=BACKWARD_STAGE_FORWARD_H,
                steady_state_time=forward_h_times[repeat_run],
                first_call_time=forward_h_first_call_time,
                useful_backward_flops=None,
                rounded_backward_flops=None,
                dropped_routes=dropped_routes,
                mode=mode,
            )
        )
        for stage in stages:
            stage_time = timing.stage_steady_state_times[stage][repeat_run]
            stage_useful_flops, stage_rounded_flops = _backward_stage_flops_per_rank(
                config,
                queue_stats,
                stage,
                w13_backward_component=w13_backward_component,
            )
            rows.append(
                _decomposed_backward_row(
                    config,
                    queue_stats=queue_stats,
                    repeat_run=repeat_run,
                    repeat_runs=repeat_runs,
                    stage=stage,
                    steady_state_time=stage_time,
                    first_call_time=None,
                    useful_backward_flops=stage_useful_flops,
                    rounded_backward_flops=stage_rounded_flops,
                    dropped_routes=dropped_routes,
                    mode=mode,
                )
            )

    grouped_rows = []
    for stage in (
        BACKWARD_STAGE_FORWARD_H,
        BACKWARD_STAGE_TOTAL,
        *stages,
    ):
        stage_rows = [row for row in rows if row["stage"] == stage]
        grouped_rows.extend(stage_rows)
        grouped_rows.append(_summary_row(stage_rows))
    return grouped_rows


def _w13_backward_component(w13_implementation: str) -> str | None:
    return source_push_w13_backward_diagnostic_component(w13_implementation)


def _row_w13_backward_component(row: Mapping[str, Any], w13_implementation: str) -> str | None:
    if row["stage"] != BACKWARD_STAGE_W13:
        return None
    return _w13_backward_component(w13_implementation)


def _decomposed_backward_row(
    config: PushInboxConfig,
    *,
    queue_stats: dict[str, Any],
    repeat_run: int,
    repeat_runs: int,
    stage: str,
    steady_state_time: float,
    first_call_time: float | None,
    useful_backward_flops: float | None,
    rounded_backward_flops: float | None,
    dropped_routes: int,
    mode: str,
) -> dict[str, Any]:
    useful_tflops = None if useful_backward_flops is None else useful_backward_flops / steady_state_time / 1e12
    rounded_tflops = None if rounded_backward_flops is None else rounded_backward_flops / steady_state_time / 1e12
    return {
        "kernel": KERNEL_NAME,
        "implementation": (
            BACKEND_SOURCE_PUSH_PALLAS if stage == BACKWARD_STAGE_TOTAL else f"{BACKEND_SOURCE_PUSH_PALLAS}_{stage}"
        ),
        "backend": BACKEND_SOURCE_PUSH_PALLAS,
        "mode": mode,
        "stage": stage,
        "row_type": "repeat",
        "config": asdict(config),
        "queue_stats": queue_stats,
        **queue_stats,
        "outer_jit": False,
        "compile_time": None,
        "lower_compile_time": None,
        "first_run_time": None,
        "first_call_time": first_call_time,
        "repeat_run": repeat_run,
        "repeat_runs": repeat_runs,
        "steady_state_time": steady_state_time,
        "bytes_per_rank": None,
        "forward_gbps_per_rank": None,
        "useful_forward_tflops_per_rank": None,
        "rounded_forward_tflops_per_rank": None,
        "useful_fwd_bwd_tflops_per_rank": None,
        "rounded_fwd_bwd_tflops_per_rank": None,
        "useful_backward_tflops_per_rank": useful_tflops,
        "rounded_backward_tflops_per_rank": rounded_tflops,
        "useful_tflops_per_rank": useful_tflops,
        "rounded_tflops_per_rank": rounded_tflops,
        "dropped_routes": dropped_routes,
        "error": None,
        "error_type": None,
        "error_message": None,
    }


def _make_benchmark_callable(
    config: PushInboxConfig,
    *,
    backend: str,
    mode: str,
    mesh: Mesh,
    host_inputs,
    route_table,
    inputs: dict[str, jax.Array],
    backward_dy_route_implementation: str = BACKWARD_IMPLEMENTATION_DEFAULT,
    backward_w2_implementation: str = BACKWARD_IMPLEMENTATION_DEFAULT,
    backward_w13_implementation: str = BACKWARD_IMPLEMENTATION_DEFAULT,
    backward_return_implementation: str = BACKWARD_IMPLEMENTATION_DEFAULT,
) -> tuple[Callable[..., Any], tuple[jax.Array, ...]]:
    if backend in PUBLIC_BACKEND_TO_IMPLEMENTATION:
        implementation = PUBLIC_BACKEND_TO_IMPLEMENTATION[backend]
        if mode == MODE_FORWARD:
            return (
                lambda x, selected, combine, w13, w2: _public_moe_forward(
                    config, mesh, implementation, x, selected, combine, w13, w2
                ),
                (
                    inputs["x_public"],
                    inputs["selected_public"],
                    inputs["combine_public"],
                    inputs["w13_public"],
                    inputs["w2_public"],
                ),
            )
        if mode == MODE_FORWARD_BACKWARD:
            return (
                jax.value_and_grad(
                    lambda x, selected, combine, w13, w2: _public_moe_loss_aux(
                        config, mesh, implementation, x, selected, combine, w13, w2
                    ),
                    argnums=(0, 2, 3, 4),
                    has_aux=True,
                ),
                (
                    inputs["x_public"],
                    inputs["selected_public"],
                    inputs["combine_public"],
                    inputs["w13_public"],
                    inputs["w2_public"],
                ),
            )
    if backend in SOURCE_PUSH_BACKEND_TO_IMPLEMENTATION:
        implementation = SOURCE_PUSH_BACKEND_TO_IMPLEMENTATION[backend]
        if mode == MODE_FORWARD:
            return (
                lambda x, combine, w13, w2: _preplanned_source_push_forward(
                    config,
                    mesh,
                    host_inputs,
                    route_table,
                    implementation,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    x,
                    combine,
                    w13,
                    w2,
                ),
                (inputs["x_source"], inputs["combine_source"], inputs["w13_source"], inputs["w2_source"]),
            )
        if mode in (
            MODE_FORWARD_BACKWARD,
            MODE_FORWARD_BACKWARD_REDUCED,
            *FORWARD_BACKWARD_GRAD_CHECKSUM_ARGNUM,
        ):
            resolved_dy_route_implementation = _resolve_backward_stage_implementation(
                backward_dy_route_implementation,
                source_push_mlp._source_push_mlp_backward_dy_route_implementation(implementation),
            )
            (
                resolved_w2_implementation,
                resolved_w2_matmul_implementation,
                resolved_w2_swiglu_implementation,
            ) = _resolve_w2_backward_implementations(
                backward_w2_implementation,
                source_push_mlp._source_push_mlp_backward_w2_implementation(implementation),
            )
            if backward_w2_implementation == BACKWARD_IMPLEMENTATION_DEFAULT:
                resolved_w2_matmul_implementation = source_push_mlp._source_push_mlp_backward_w2_matmul_implementation(
                    implementation
                )
                resolved_w2_swiglu_implementation = source_push_mlp._source_push_mlp_backward_w2_swiglu_implementation(
                    implementation
                )
            resolved_w13_implementation = _resolve_backward_stage_implementation(
                backward_w13_implementation,
                source_push_mlp._source_push_mlp_backward_w13_implementation(implementation),
            )
            resolved_return_implementation = _resolve_backward_stage_implementation(
                backward_return_implementation,
                source_push_mlp._source_push_mlp_backward_return_implementation(implementation),
            )
            return_route_indices = None
            if resolved_return_implementation == "pallas_mgpu":
                with jax.set_mesh(mesh):
                    return_route_indices = source_push_backward_return_route_indices_jax(
                        host_inputs.plan,
                        src_base_by_expert=host_inputs.src_base_by_expert,
                    )
                    _block_until_ready(return_route_indices)
            if mode == MODE_FORWARD_BACKWARD_REDUCED:
                return (
                    lambda x, combine, w13, w2: _preplanned_source_push_fwd_bwd_reduced(
                        config,
                        mesh,
                        host_inputs,
                        route_table,
                        implementation,
                        resolved_dy_route_implementation,
                        resolved_w2_implementation,
                        resolved_w2_matmul_implementation,
                        resolved_w2_swiglu_implementation,
                        resolved_w13_implementation,
                        resolved_return_implementation,
                        return_route_indices,
                        x,
                        combine,
                        w13,
                        w2,
                    ),
                    (inputs["x_source"], inputs["combine_source"], inputs["w13_source"], inputs["w2_source"]),
                )
            if mode in FORWARD_BACKWARD_GRAD_CHECKSUM_ARGNUM:
                return (
                    lambda x, combine, w13, w2: _preplanned_source_push_fwd_bwd_grad_checksum(
                        config,
                        mesh,
                        host_inputs,
                        route_table,
                        implementation,
                        resolved_dy_route_implementation,
                        resolved_w2_implementation,
                        resolved_w2_matmul_implementation,
                        resolved_w2_swiglu_implementation,
                        resolved_w13_implementation,
                        resolved_return_implementation,
                        return_route_indices,
                        FORWARD_BACKWARD_GRAD_CHECKSUM_ARGNUM[mode],
                        x,
                        combine,
                        w13,
                        w2,
                    ),
                    (inputs["x_source"], inputs["combine_source"], inputs["w13_source"], inputs["w2_source"]),
                )
            return (
                jax.value_and_grad(
                    lambda x, combine, w13, w2: _preplanned_source_push_loss_aux(
                        config,
                        mesh,
                        host_inputs,
                        route_table,
                        implementation,
                        resolved_dy_route_implementation,
                        resolved_w2_implementation,
                        resolved_w2_matmul_implementation,
                        resolved_w2_swiglu_implementation,
                        resolved_w13_implementation,
                        resolved_return_implementation,
                        return_route_indices,
                        x,
                        combine,
                        w13,
                        w2,
                    ),
                    argnums=(0, 1, 2, 3),
                    has_aux=True,
                ),
                (inputs["x_source"], inputs["combine_source"], inputs["w13_source"], inputs["w2_source"]),
            )
    raise ValueError(f"unsupported backend={backend!r} mode={mode!r}")


def _public_moe_forward(config: PushInboxConfig, mesh: Mesh, implementation: str, x, selected, combine, w13, w2):
    return moe_mlp(
        x,
        selected,
        combine,
        w13,
        w2,
        activation=ActivationFunctionEnum.silu,
        implementation=implementation,
        mesh=mesh,
        capacity_factor=config.capacity_factor,
        report_capacity_overflow=True,
    )


def _public_moe_loss_aux(config: PushInboxConfig, mesh: Mesh, implementation: str, x, selected, combine, w13, w2):
    out, dropped = _public_moe_forward(config, mesh, implementation, x, selected, combine, w13, w2)
    return jnp.sum(out.astype(jnp.float32)), dropped


def _preplanned_source_push_forward(
    config: PushInboxConfig,
    mesh: Mesh,
    host_inputs,
    route_table,
    implementation: str,
    backward_dy_route_implementation: str | None,
    backward_w2_implementation: str | None,
    backward_w2_matmul_implementation: str | None,
    backward_w2_swiglu_implementation: str | None,
    backward_w13_implementation: str | None,
    backward_return_implementation: str | None,
    return_route_indices,
    x,
    combine,
    w13,
    w2,
):
    return source_push_moe_mlp_from_plan(
        config,
        host_inputs,
        route_table,
        x,
        combine,
        w13,
        w2,
        implementation=implementation,
        execution_mode=FORWARD_EXECUTION_STAGED_HOST_SYNC,
        backward_dy_route_implementation=backward_dy_route_implementation,
        backward_w2_implementation=backward_w2_implementation,
        backward_w2_matmul_implementation=backward_w2_matmul_implementation,
        backward_w2_swiglu_implementation=backward_w2_swiglu_implementation,
        backward_w13_implementation=backward_w13_implementation,
        backward_return_implementation=backward_return_implementation,
        return_route_indices=return_route_indices,
        mesh=mesh,
    )


def _preplanned_source_push_loss_aux(
    config: PushInboxConfig,
    mesh: Mesh,
    host_inputs,
    route_table,
    implementation: str,
    backward_dy_route_implementation: str | None,
    backward_w2_implementation: str | None,
    backward_w2_matmul_implementation: str | None,
    backward_w2_swiglu_implementation: str | None,
    backward_w13_implementation: str | None,
    backward_return_implementation: str | None,
    return_route_indices,
    x,
    combine,
    w13,
    w2,
):
    out, dropped = _preplanned_source_push_forward(
        config,
        mesh,
        host_inputs,
        route_table,
        implementation,
        backward_dy_route_implementation,
        backward_w2_implementation,
        backward_w2_matmul_implementation,
        backward_w2_swiglu_implementation,
        backward_w13_implementation,
        backward_return_implementation,
        return_route_indices,
        x,
        combine,
        w13,
        w2,
    )
    return jnp.sum(out.astype(jnp.float32)), dropped


def _preplanned_source_push_fwd_bwd_reduced(
    config: PushInboxConfig,
    mesh: Mesh,
    host_inputs,
    route_table,
    implementation: str,
    backward_dy_route_implementation: str | None,
    backward_w2_implementation: str | None,
    backward_w2_matmul_implementation: str | None,
    backward_w2_swiglu_implementation: str | None,
    backward_w13_implementation: str | None,
    backward_return_implementation: str | None,
    return_route_indices,
    x,
    combine,
    w13,
    w2,
):
    """Run source-push fwd+bwd but return scalar gradient checksums only."""

    (loss, dropped), grads = jax.value_and_grad(
        lambda x_arg, combine_arg, w13_arg, w2_arg: _preplanned_source_push_loss_aux(
            config,
            mesh,
            host_inputs,
            route_table,
            implementation,
            backward_dy_route_implementation,
            backward_w2_implementation,
            backward_w2_matmul_implementation,
            backward_w2_swiglu_implementation,
            backward_w13_implementation,
            backward_return_implementation,
            return_route_indices,
            x_arg,
            combine_arg,
            w13_arg,
            w2_arg,
        ),
        argnums=(0, 1, 2, 3),
        has_aux=True,
    )(x, combine, w13, w2)
    # Cast after the reduction. Casting the full gradient leaves to fp32 here
    # recreates the target-shape materialization pressure this mode is meant to
    # avoid.
    grad_checksum = sum(jnp.sum(grad).astype(jnp.float32) for grad in jax.tree.leaves(grads))
    return loss, dropped, grad_checksum


def _preplanned_source_push_fwd_bwd_grad_checksum(
    config: PushInboxConfig,
    mesh: Mesh,
    host_inputs,
    route_table,
    implementation: str,
    backward_dy_route_implementation: str | None,
    backward_w2_implementation: str | None,
    backward_w2_matmul_implementation: str | None,
    backward_w2_swiglu_implementation: str | None,
    backward_w13_implementation: str | None,
    backward_return_implementation: str | None,
    return_route_indices,
    grad_argnum: int,
    x,
    combine,
    w13,
    w2,
):
    """Run source-push fwd+bwd for one selected gradient output."""

    (loss, dropped), grad = jax.value_and_grad(
        lambda x_arg, combine_arg, w13_arg, w2_arg: _preplanned_source_push_loss_aux(
            config,
            mesh,
            host_inputs,
            route_table,
            implementation,
            backward_dy_route_implementation,
            backward_w2_implementation,
            backward_w2_matmul_implementation,
            backward_w2_swiglu_implementation,
            backward_w13_implementation,
            backward_return_implementation,
            return_route_indices,
            x_arg,
            combine_arg,
            w13_arg,
            w2_arg,
        ),
        argnums=grad_argnum,
        has_aux=True,
    )(x, combine, w13, w2)
    grad_checksum = jnp.sum(grad).astype(jnp.float32)
    return loss, dropped, grad_checksum


def _time_callable(
    fn: Callable[..., Any],
    *args,
    mesh: Mesh,
    warmup: int,
    steps: int,
    repeat_runs: int,
    use_outer_jit: bool,
    separate_compile: bool,
) -> MlpTiming:
    call = jax.jit(fn) if use_outer_jit else fn
    lower_compile_time = None
    first_run_time = None

    if use_outer_jit and separate_compile:
        with jax.set_mesh(mesh):
            lowered = call.lower(*args)
            start = time.perf_counter()
            compiled = lowered.compile()
            lower_compile_time = time.perf_counter() - start
            start = time.perf_counter()
            output = compiled(*args)
        _block_until_ready(output)
        first_run_time = time.perf_counter() - start
        first_call_time = lower_compile_time + first_run_time
        timed_call = compiled
        compile_time = first_call_time
    else:
        start = time.perf_counter()
        with jax.set_mesh(mesh):
            output = call(*args)
        _block_until_ready(output)
        first_call_time = time.perf_counter() - start
        timed_call = call
        compile_time = first_call_time if use_outer_jit else None
        first_run_time = first_call_time

    for _ in range(warmup):
        with jax.set_mesh(mesh):
            output = timed_call(*args)
        _block_until_ready(output)

    steady_state_times = []
    for _ in range(repeat_runs):
        start = time.perf_counter()
        for _ in range(steps):
            with jax.set_mesh(mesh):
                output = timed_call(*args)
            _block_until_ready(output)
        steady_state_times.append((time.perf_counter() - start) / steps)

    return MlpTiming(
        compile_time=compile_time,
        lower_compile_time=lower_compile_time,
        first_run_time=first_run_time,
        first_call_time=first_call_time,
        steady_state_times=steady_state_times,
        output=output,
    )


def _timing_rows(
    config: PushInboxConfig,
    *,
    backend: str,
    mode: str,
    timing: MlpTiming,
    queue_stats: dict[str, Any],
    repeat_runs: int,
    outer_jit: bool,
) -> list[dict[str, Any]]:
    useful_forward_flops, rounded_forward_flops = _forward_flops_per_rank(config, queue_stats)
    useful_fwd_bwd_flops = useful_forward_flops * 3
    rounded_fwd_bwd_flops = rounded_forward_flops * 3
    mode_is_fwd_bwd = mode in (
        MODE_FORWARD_BACKWARD,
        MODE_FORWARD_BACKWARD_REDUCED,
        *FORWARD_BACKWARD_GRAD_CHECKSUM_ARGNUM,
    )
    useful_mode_flops = useful_fwd_bwd_flops if mode_is_fwd_bwd else useful_forward_flops
    rounded_mode_flops = rounded_fwd_bwd_flops if mode_is_fwd_bwd else rounded_forward_flops
    bytes_per_rank = _forward_bytes_per_rank(config, queue_stats)
    dropped_routes = _dropped_routes_from_output(mode, timing.output)

    rows = []
    for repeat_run, steady_state_time in enumerate(timing.steady_state_times):
        row = {
            "kernel": KERNEL_NAME,
            "implementation": backend,
            "backend": backend,
            "mode": mode,
            "row_type": "repeat",
            "config": asdict(config),
            "queue_stats": queue_stats,
            **queue_stats,
            "outer_jit": outer_jit,
            "compile_time": timing.compile_time,
            "lower_compile_time": timing.lower_compile_time,
            "first_run_time": timing.first_run_time,
            "first_call_time": timing.first_call_time,
            "repeat_run": repeat_run,
            "repeat_runs": repeat_runs,
            "steady_state_time": steady_state_time,
            "bytes_per_rank": bytes_per_rank,
            "forward_gbps_per_rank": bytes_per_rank / steady_state_time / 1e9,
            "useful_forward_tflops_per_rank": useful_forward_flops / steady_state_time / 1e12,
            "rounded_forward_tflops_per_rank": rounded_forward_flops / steady_state_time / 1e12,
            "useful_fwd_bwd_tflops_per_rank": useful_fwd_bwd_flops / steady_state_time / 1e12,
            "rounded_fwd_bwd_tflops_per_rank": rounded_fwd_bwd_flops / steady_state_time / 1e12,
            "useful_backward_tflops_per_rank": None,
            "rounded_backward_tflops_per_rank": None,
            "useful_tflops_per_rank": useful_mode_flops / steady_state_time / 1e12,
            "rounded_tflops_per_rank": rounded_mode_flops / steady_state_time / 1e12,
            "dropped_routes": dropped_routes,
            "error": None,
            "error_type": None,
            "error_message": None,
        }
        rows.append(row)
    return [*rows, _summary_row(rows)]


def _summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    summary = {
        "kernel": KERNEL_NAME,
        "implementation": first["implementation"],
        "backend": first["backend"],
        "mode": first["mode"],
        "row_type": "summary",
        "config": first["config"],
        "queue_stats": first["queue_stats"],
        "outer_jit": first["outer_jit"],
        "repeat_runs": first["repeat_runs"],
        "repeat_rows": len(rows),
        "error": None,
        "error_type": None,
        "error_message": None,
        "min_steady_state_time": min(row["steady_state_time"] for row in rows),
        "max_steady_state_time": max(row["steady_state_time"] for row in rows),
        "p90_steady_state_time": _percentile(rows, "steady_state_time", 0.90),
        "p95_steady_state_time": _percentile(rows, "steady_state_time", 0.95),
    }
    if "stage" in first:
        summary["stage"] = first["stage"]
    if "diagnostic_variant" in first:
        summary["diagnostic_variant"] = first["diagnostic_variant"]
    if "diagnostic" in first:
        summary["diagnostic"] = first["diagnostic"]
    summary.update(first["queue_stats"])
    for metric in SUMMARY_METRICS:
        summary[f"median_{metric}"] = _median(rows, metric)
    if first.get("w13_baseline_tflops_per_rank") is not None:
        summary["w13_baseline_tflops_per_rank"] = first["w13_baseline_tflops_per_rank"]
        summary["h_layout"] = first.get("h_layout") or first.get("compact_h_layout")
        if first.get("compact_h_layout") is not None:
            summary["compact_h_layout"] = first["compact_h_layout"]
        if first.get("compact_expert_capacity") is not None:
            summary["compact_expert_capacity"] = first["compact_expert_capacity"]
        median_useful_tflops = summary["median_useful_forward_tflops_per_rank"]
        summary["passes_w13_216_949_gate"] = (
            False
            if median_useful_tflops is None
            else median_useful_tflops >= SOURCE_PUSH_W13_STABLE_BASELINE_TFLOPS_PER_RANK
        )
    return summary


def _device_benchmark_inputs(config: PushInboxConfig, raw_inputs, mesh: Mesh) -> dict[str, jax.Array]:
    x_source = jnp.asarray(raw_inputs.x, dtype=jnp.bfloat16)
    selected_source = jnp.asarray(raw_inputs.selected_experts, dtype=jnp.int32)
    combine_source = jnp.asarray(raw_inputs.combine_weights, dtype=jnp.bfloat16)
    w13_source = jnp.asarray(raw_inputs.w_gate_up, dtype=jnp.bfloat16)
    w2_source = jnp.asarray(raw_inputs.w_down, dtype=jnp.bfloat16)
    return {
        "x_source": jax.device_put(x_source, NamedSharding(mesh, P(AXIS, None, None))),
        "selected_source": jax.device_put(selected_source, NamedSharding(mesh, P(AXIS, None, None))),
        "combine_source": jax.device_put(combine_source, NamedSharding(mesh, P(AXIS, None, None))),
        "w13_source": jax.device_put(w13_source, NamedSharding(mesh, P(AXIS, None, None, None))),
        "w2_source": jax.device_put(w2_source, NamedSharding(mesh, P(AXIS, None, None, None))),
        "x_public": jax.device_put(
            x_source.reshape(config.ep_size * config.tokens_per_rank, config.hidden_dim),
            NamedSharding(mesh, P(AXIS, None)),
        ),
        "selected_public": jax.device_put(
            selected_source.reshape(config.ep_size * config.tokens_per_rank, config.topk),
            NamedSharding(mesh, P(AXIS, None)),
        ),
        "combine_public": jax.device_put(
            combine_source.reshape(config.ep_size * config.tokens_per_rank, config.topk),
            NamedSharding(mesh, P(AXIS, None)),
        ),
        "w13_public": jax.device_put(
            w13_source.reshape(
                config.ep_size * config.experts_per_rank,
                config.hidden_dim,
                2 * config.intermediate_dim,
            ),
            NamedSharding(mesh, P(AXIS, None, None)),
        ),
        "w2_public": jax.device_put(
            w2_source.reshape(
                config.ep_size * config.experts_per_rank,
                config.intermediate_dim,
                config.hidden_dim,
            ),
            NamedSharding(mesh, P(AXIS, None, None)),
        ),
    }


def _make_public_ep_mesh(ep_size: int) -> Mesh:
    devices = np.asarray(jax.devices()[:ep_size])
    if devices.size < ep_size:
        raise RuntimeError(f"Need {ep_size} visible JAX devices, got {devices.size}")
    return Mesh(devices, (AXIS,), axis_types=(AxisType.Explicit,))


def _resolve_outer_jit(backend: str, outer_jit: str) -> bool:
    if outer_jit == "true":
        return True
    if outer_jit == "false":
        return False
    return backend in (BACKEND_RING, BACKEND_RAGGED_A2A, BACKEND_SOURCE_PUSH_REFERENCE)


def _outer_jit_error_value(backend: str, outer_jit: str) -> bool | str:
    if outer_jit in ("true", "false"):
        return outer_jit == "true"
    if backend not in BACKENDS:
        return "auto"
    return _resolve_outer_jit(backend, outer_jit)


def _forward_flops_per_rank(config: PushInboxConfig, queue_stats: dict[str, Any]) -> tuple[float, float]:
    useful_rows = queue_stats["valid_rows_per_rank_mean"]
    rounded_rows = queue_stats["rounded_rows_per_rank_mean"]
    useful = useful_rows * config.hidden_dim * config.intermediate_dim * 6
    rounded = rounded_rows * config.hidden_dim * config.intermediate_dim * 6
    return float(useful), float(rounded)


def _backward_flops_per_rank(config: PushInboxConfig, queue_stats: dict[str, Any]) -> tuple[float, float]:
    useful_rows = queue_stats["valid_rows_per_rank_mean"]
    rounded_rows = queue_stats["rounded_rows_per_rank_mean"]
    useful = useful_rows * config.hidden_dim * config.intermediate_dim * 12
    rounded = rounded_rows * config.hidden_dim * config.intermediate_dim * 12
    return float(useful), float(rounded)


def _backward_stage_flops_per_rank(
    config: PushInboxConfig,
    queue_stats: dict[str, Any],
    stage: str,
    *,
    w13_backward_component: str | None = None,
) -> tuple[float | None, float | None]:
    useful_rows = queue_stats["valid_rows_per_rank_mean"]
    rounded_rows = queue_stats["rounded_rows_per_rank_mean"]
    if stage == BACKWARD_STAGE_W2:
        useful = useful_rows * config.hidden_dim * config.intermediate_dim * 4
        rounded = rounded_rows * config.hidden_dim * config.intermediate_dim * 4
        return float(useful), float(rounded)
    if stage == BACKWARD_STAGE_W13:
        if w13_backward_component == "dw13_half":
            flop_multiplier = 2
        else:
            flop_multiplier = 4 if w13_backward_component in ("dx", "dw13") else 8
        useful = useful_rows * config.hidden_dim * config.intermediate_dim * flop_multiplier
        rounded = rounded_rows * config.hidden_dim * config.intermediate_dim * flop_multiplier
        return float(useful), float(rounded)
    if stage == BACKWARD_STAGE_DX13_PUSH:
        useful = useful_rows * config.hidden_dim * config.intermediate_dim * 4
        rounded = rounded_rows * config.hidden_dim * config.intermediate_dim * 4
        return float(useful), float(rounded)
    return None, None


def _forward_bytes_per_rank(config: PushInboxConfig, queue_stats: dict[str, Any]) -> float:
    rounded_rows = queue_stats["rounded_rows_per_rank_mean"]
    send_bytes = queue_stats["send_rounded_rows_per_rank_mean"] * config.hidden_dim * BYTES_PER_BF16
    w2_bytes = rounded_rows * (config.intermediate_dim + config.hidden_dim) * BYTES_PER_BF16
    output_bytes = config.tokens_per_rank * config.hidden_dim * BYTES_PER_BF16
    return float(send_bytes + w2_bytes + output_bytes)


def _stage_bytes_per_rank(config: PushInboxConfig, queue_stats: dict[str, Any], stage: str) -> float | None:
    rounded_rows = queue_stats["rounded_rows_per_rank_mean"]
    if stage == FORWARD_STAGE_W13:
        return float(queue_stats["send_rounded_rows_per_rank_mean"] * config.hidden_dim * BYTES_PER_BF16)
    if stage == FORWARD_STAGE_W2_RETURN:
        return float(rounded_rows * (config.intermediate_dim + config.hidden_dim) * BYTES_PER_BF16)
    if stage == FORWARD_STAGE_COMBINE:
        return float(config.tokens_per_rank * config.hidden_dim * BYTES_PER_BF16)
    return None


def _stage_useful_flops_per_rank(config: PushInboxConfig, queue_stats: dict[str, Any], stage: str) -> float | None:
    useful_rows = queue_stats["valid_rows_per_rank_mean"]
    if stage == FORWARD_STAGE_W13:
        return float(useful_rows * config.hidden_dim * config.intermediate_dim * 4)
    if stage == FORWARD_STAGE_W2_RETURN:
        return float(useful_rows * config.hidden_dim * config.intermediate_dim * 2)
    return None


def _stage_rounded_flops_per_rank(config: PushInboxConfig, queue_stats: dict[str, Any], stage: str) -> float | None:
    rounded_rows = queue_stats["rounded_rows_per_rank_mean"]
    if stage == FORWARD_STAGE_W13:
        return float(rounded_rows * config.hidden_dim * config.intermediate_dim * 4)
    if stage == FORWARD_STAGE_W2_RETURN:
        return float(rounded_rows * config.hidden_dim * config.intermediate_dim * 2)
    return None


def _dropped_routes_from_output(mode: str, output: Any) -> int:
    if mode == MODE_FORWARD:
        dropped = output[1]
    elif mode == MODE_FORWARD_BACKWARD_REDUCED or mode in FORWARD_BACKWARD_GRAD_CHECKSUM_ARGNUM:
        dropped = output[1]
    else:
        dropped = output[0][1]
    return int(jax.device_get(dropped))


def _median(rows: list[dict[str, Any]], field: str) -> float | int | None:
    values = [row[field] for row in rows if row.get(field) is not None]
    if not values:
        return None
    return median(values)


def _percentile(rows: list[dict[str, Any]], field: str, percentile: float) -> float | int | None:
    values = sorted(row[field] for row in rows if row.get(field) is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _parse_csv_choices(value: str, choices: Sequence[str], *, flag: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError(f"{flag} must include at least one value")
    invalid = [item for item in parsed if item not in choices]
    if invalid:
        raise ValueError(f"{flag} has unsupported values {invalid}; expected choices from {tuple(choices)}")
    return parsed


if __name__ == "__main__":
    main()
