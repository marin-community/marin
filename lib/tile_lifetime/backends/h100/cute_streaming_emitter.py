# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Instantiate the SM90 streaming contraction/fold skeleton from tensor semantics.

This optional backend module deliberately lives outside the typed core package.
It depends on CUDA, CuTe DSL, QuACK, and a small set of physical helper
primitives extracted from the pinned FlashAttention CuTe implementation.  It
does not call a FlashAttention interface or select a kernel by attention name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from flash_attn.cute.cute_dsl_utils import to_cute_tensor

from tile_lifetime.h100_streaming_lowering import (
    H100StreamingSchedule,
    LoweredScoreMap,
    lower_h100_streaming_program,
)
from tile_lifetime.streaming_attention import StreamingAttentionProgram

from .cute_streaming_sm90 import ShuttleStreamingAttentionSm90


@dataclass
class CompiledH100StreamingProgram:
    """One CuTe executable and the semantic/runtime contract it implements."""

    program: StreamingAttentionProgram
    score_map: LoweredScoreMap
    schedule: H100StreamingSchedule
    query_shape: tuple[int, ...]
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    head_group_size: int
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
        self.executable(
            query,
            key,
            value,
            output,
            log_sum_exp,
            self.score_map.scale,
            *([None] * 10),
        )


def compile_h100_streaming_program(
    program: StreamingAttentionProgram,
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    log_sum_exp: torch.Tensor,
) -> CompiledH100StreamingProgram:
    """Compile generic QK/Map/Fold/PV semantics into the SM90 skeleton."""
    query_value, key_value = program.qk.inputs
    value_value = program.pv.inputs[1]
    query_shape = query_value.shape
    key_shape = key_value.shape
    value_shape = value_value.shape
    for name, tensor, shape in (
        ("query", query, query_shape),
        ("key", key, key_shape),
        ("value", value, value_shape),
        ("output", output, program.finalize.output.shape),
    ):
        _validate_runtime_tensor(name, tensor, shape)

    lowering = lower_h100_streaming_program(program)
    head_group_size = lowering.head_group_size
    score_map = lowering.score_map
    schedule = lowering.schedule

    score_mod = _softcap_score_map(score_map.softcap) if score_map.softcap is not None else None
    skeleton = ShuttleStreamingAttentionSm90(
        cutlass.BFloat16,
        query_shape[-1],
        value_shape[-1],
        head_group_size,
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
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    cute_arguments = [to_cute_tensor(tensor) for tensor in (query, key, value, output)]
    cute_arguments.extend((to_cute_tensor(log_sum_exp, assumed_align=4), score_map.scale))
    cute_arguments.extend([None] * 10)
    cute_arguments.append(fake_stream)
    executable = cute.compile(skeleton, *cute_arguments, options="--enable-tvm-ffi")
    return CompiledH100StreamingProgram(
        program=program,
        score_map=score_map,
        schedule=schedule,
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        head_group_size=head_group_size,
        executable=executable,
    )


def _softcap_score_map(softcap: float):
    @cute.jit
    def generated_score_map(scores, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        normalized = scores / softcap
        return softcap * cute.math.tanh(normalized, fastmath=True)

    return generated_score_map


def _validate_runtime_tensor(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
    if tensor.dtype is not torch.bfloat16:
        raise ValueError(f"{name} must be BF16")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be CUDA-resident")
