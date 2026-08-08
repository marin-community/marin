# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower routed Contract/Map/Fold semantics to the SM90 streaming skeleton.

The relation supplies only the sparse fold domain.  This module does not call
an attention interface: it derives compact block lists from ``RelationPlan``
and instantiates the same Shuttle-owned QK/online-state/PV skeleton used by the
dense path.  The physical skeleton performs TMA K/V copies into a circular
shared-memory pipeline before the contractions consume them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    normalize_block_sparse_config,
    to_cute_block_sparse_tensors,
)
from flash_attn.cute.cute_dsl_utils import to_cute_tensor

from tile_lifetime.h100_streaming_lowering import LoweredScoreMap, lower_h100_streaming_program
from tile_lifetime.routed_attention_plan import (
    RoutedAttentionOrientation,
    RoutedStreamingAttentionCompilation,
    query_major_block_index_plan,
)

from .cute_streaming_emitter import _softcap_score_map, _validate_runtime_tensor
from .cute_streaming_sm90 import ShuttleStreamingAttentionSm90


@dataclass
class CompiledH100RoutedStreamingProgram:
    """One relation-specialized query-major executable."""

    compilation: RoutedStreamingAttentionCompilation
    score_map: LoweredScoreMap
    query_shape: tuple[int, ...]
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    block_sparse_tensors: BlockSparseTensorsTorch
    executable: Any

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        log_sum_exp: torch.Tensor,
    ) -> None:
        _validate_runtime_tensor("query", query, self.query_shape)
        _validate_runtime_tensor("key", key, self.key_shape)
        _validate_runtime_tensor("value", value, self.value_shape)
        _validate_runtime_tensor("output", output, (*self.query_shape[:-1], self.value_shape[-1]))
        expected_lse = (self.query_shape[0], self.query_shape[2], self.query_shape[1])
        if tuple(log_sum_exp.shape) != expected_lse or log_sum_exp.dtype is not torch.float32:
            raise ValueError(f"log_sum_exp must be FP32 with shape {expected_lse}")
        sparse_arguments = (
            self.block_sparse_tensors.mask_block_cnt,
            self.block_sparse_tensors.mask_block_idx,
            self.block_sparse_tensors.full_block_cnt,
            self.block_sparse_tensors.full_block_idx,
            self.block_sparse_tensors.cu_total_m_blocks,
            self.block_sparse_tensors.cu_block_idx_offsets,
            self.block_sparse_tensors.dq_write_order,
            self.block_sparse_tensors.dq_write_order_full,
        )
        self.executable(
            query,
            key,
            value,
            output,
            log_sum_exp,
            self.score_map.scale,
            *([None] * 8),
            sparse_arguments,
            None,
        )


def compile_h100_routed_streaming_program(
    compilation: RoutedStreamingAttentionCompilation,
    *,
    orientation: RoutedAttentionOrientation,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    log_sum_exp: torch.Tensor,
) -> CompiledH100RoutedStreamingProgram:
    """Instantiate shared-memory streaming for one generic sparse relation."""
    if orientation is not RoutedAttentionOrientation.QUERY_MAJOR:
        raise ValueError("the first SM90 routed emitter implements only the query-major candidate")
    candidate = next((item for item in compilation.candidates if item.orientation is orientation), None)
    if candidate is None:
        raise ValueError(f"compilation does not contain the {orientation.value} candidate")

    program = compilation.program
    query_shape = program.qk.inputs[0].shape
    key_shape = program.qk.inputs[1].shape
    value_shape = program.pv.inputs[1].shape
    for name, tensor, shape in (
        ("query", query, query_shape),
        ("key", key, key_shape),
        ("value", value, value_shape),
        ("output", output, program.finalize.output.shape),
    ):
        _validate_runtime_tensor(name, tensor, shape)

    lowering = lower_h100_streaming_program(program)
    schedule = lowering.schedule
    if candidate.config.query_block_size != schedule.tile_m:
        raise ValueError("relation query block size does not match the streaming tile")
    if candidate.config.key_value_block_size != schedule.tile_n:
        raise ValueError("relation KV block size does not match the streaming tile")

    index_plan = query_major_block_index_plan(compilation.relation)
    block_count = torch.as_tensor(index_plan.block_count, device=query.device)[None, None]
    block_index = torch.as_tensor(index_plan.block_index, device=query.device)[None, None]
    raw_sparse_tensors = BlockSparseTensorsTorch(
        mask_block_cnt=block_count,
        mask_block_idx=block_index,
        # The generated dynamic branch must carry concrete tensors for both
        # lists even when every selected block uses the masking path.
        full_block_cnt=torch.zeros_like(block_count),
        full_block_idx=torch.zeros_like(block_index),
        block_size=(schedule.tile_m, schedule.tile_n),
    )
    sparse_tensors, _, query_subtile_factor = normalize_block_sparse_config(
        raw_sparse_tensors,
        batch_size=query_shape[0],
        num_head=query_shape[2],
        seqlen_q=query_shape[1],
        seqlen_k=key_shape[1],
        block_size=(schedule.tile_m, schedule.tile_n),
        q_stage=1,
    )
    sparse_cute_tensors = to_cute_block_sparse_tensors(sparse_tensors)
    assert sparse_cute_tensors is not None

    score_map = lowering.score_map
    score_mod = _softcap_score_map(score_map.softcap) if score_map.softcap is not None else None
    skeleton = ShuttleStreamingAttentionSm90(
        cutlass.BFloat16,
        query_shape[-1],
        value_shape[-1],
        lowering.head_group_size,
        is_causal=score_map.causal,
        is_local=False,
        pack_gqa=schedule.pack_gqa,
        tile_m=schedule.tile_m,
        tile_n=schedule.tile_n,
        num_stages=schedule.stages,
        num_threads=schedule.threads,
        Q_in_regs=schedule.q_in_registers,
        score_mod=score_mod,
        mask_mod=None,
        has_aux_tensors=False,
        intra_wg_overlap=schedule.intra_warpgroup_overlap,
        mma_pv_is_rs=schedule.pv_register_source,
        q_subtile_factor=query_subtile_factor,
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    cute_arguments = [to_cute_tensor(tensor) for tensor in (query, key, value, output)]
    cute_arguments.extend((to_cute_tensor(log_sum_exp, assumed_align=4), score_map.scale))
    cute_arguments.extend([None] * 8)
    cute_arguments.extend((sparse_cute_tensors, None, fake_stream))
    executable = cute.compile(skeleton, *cute_arguments, options="--enable-tvm-ffi")
    return CompiledH100RoutedStreamingProgram(
        compilation=compilation,
        score_map=score_map,
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        block_sparse_tensors=sparse_tensors,
        executable=executable,
    )
