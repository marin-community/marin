# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate and benchmark a deterministic streamed normalized-exp reverse pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax
import torch
import torch.nn.functional as functional
import triton
import triton.language as tl
from h100_generated_streaming_attention import _inputs, _program, emit_streaming_attention, lower_score_map

from tile_lifetime import (
    StreamingAttentionBackwardDomainTraversal,
    StreamingTileSchedule,
    derive_streaming_attention_backward,
    derive_streaming_attention_backward_tile_schedule,
    estimate_streaming_attention_backward_work,
    verify_owner_preparation_fold_attachment,
)
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.stablehlo_import import import_stablehlo
from tile_lifetime.stablehlo_streaming_attention_backward import (
    recover_experimental_whole_pattern_streaming_attention_backward,
)
from tile_lifetime.streaming_attention_backward import (
    StreamingAttentionBackwardMaximumVJP,
    StreamingAttentionBackwardProgram,
    StreamingAttentionBackwardTileSchedule,
    eliminate_normalized_exp_maximum_vjp,
    verify_streaming_attention_backward_score_map_vjp,
)
from tile_lifetime.streaming_attention_backward_reference import (
    STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
    StreamingAttentionBackwardDebugConfig,
    export_debug_streaming_attention_backward,
)
from tile_lifetime.tensor_program import serialize_scalar_expression

LOG2_E = 1.4426950408889634


@triton.jit
def _streaming_dq_kernel(
    query,
    key,
    value,
    output,
    output_cotangent,
    log_sum_exp,
    output_dot,
    query_cotangent,
    sequence_length,
    query_heads,
    key_value_heads,
    scale,
    scale_log2,
    softcap,
    output_scale,
    stride_qb: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_os: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    stride_db: tl.constexpr,
    stride_ds: tl.constexpr,
    stride_dh: tl.constexpr,
    stride_dd: tl.constexpr,
    stride_dqb: tl.constexpr,
    stride_dqs: tl.constexpr,
    stride_dqh: tl.constexpr,
    stride_dqd: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    head_dimension: tl.constexpr,
    head_group_size: tl.constexpr,
    causal: tl.constexpr,
    has_softcap: tl.constexpr,
):
    query_block_index = tl.program_id(0)
    batch_key_value_head = tl.program_id(1)
    batch_index = batch_key_value_head // key_value_heads
    key_value_head = batch_key_value_head % key_value_heads
    query_start = query_block_index * block_m
    packed_query_rows: tl.constexpr = block_m * head_group_size
    packed_rows = tl.arange(0, packed_query_rows)
    query_offsets_in_block = packed_rows // head_group_size
    query_head_offsets = packed_rows % head_group_size
    query_heads_for_rows = key_value_head * head_group_size + query_head_offsets
    query_tokens = query_start + query_offsets_in_block
    key_offsets = tl.arange(0, block_n)
    features = tl.arange(0, head_dimension)
    query_valid = query_tokens < sequence_length
    query_offsets = (
        batch_index * stride_qb
        + query_tokens[:, None] * stride_qs
        + query_heads_for_rows[:, None] * stride_qh
        + features[None, :] * stride_qd
    )
    output_cotangent_offsets = (
        batch_index * stride_db
        + query_tokens[:, None] * stride_ds
        + query_heads_for_rows[:, None] * stride_dh
        + features[None, :] * stride_dd
    )
    output_offsets = (
        batch_index * stride_ob
        + query_tokens[:, None] * stride_os
        + query_heads_for_rows[:, None] * stride_oh
        + features[None, :] * stride_od
    )
    query_cotangent_offsets = (
        batch_index * stride_dqb
        + query_tokens[:, None] * stride_dqs
        + query_heads_for_rows[:, None] * stride_dqh
        + features[None, :] * stride_dqd
    )
    query_tile = tl.load(query + query_offsets, mask=query_valid[:, None], other=0.0)
    output_tile = tl.load(output + output_offsets, mask=query_valid[:, None], other=0.0).to(tl.float32)
    output_cotangent_tile = tl.load(
        output_cotangent + output_cotangent_offsets,
        mask=query_valid[:, None],
        other=0.0,
    )
    row_offset = (batch_index * query_heads + query_heads_for_rows) * sequence_length + query_tokens
    lse = tl.load(log_sum_exp + row_offset, mask=query_valid, other=-float("inf"))
    delta = tl.sum(output_tile * output_cotangent_tile.to(tl.float32), axis=1)
    tl.store(output_dot + row_offset, delta, mask=query_valid)
    query_gradient = tl.zeros((packed_query_rows, head_dimension), tl.float32)
    key_stop = sequence_length
    if causal:
        key_stop = query_start + block_m
    for key_start in tl.range(0, key_stop, block_n):
        key_start = tl.multiple_of(key_start, block_n)
        key_block = tl.make_block_ptr(
            base=key + batch_index * stride_kb + key_value_head * stride_kh,
            shape=(head_dimension, sequence_length),
            strides=(stride_kd, stride_ks),
            offsets=(0, key_start),
            block_shape=(head_dimension, block_n),
            order=(0, 1),
        )
        value_block = tl.make_block_ptr(
            base=value + batch_index * stride_vb + key_value_head * stride_vh,
            shape=(head_dimension, sequence_length),
            strides=(stride_vd, stride_vs),
            offsets=(0, key_start),
            block_shape=(head_dimension, block_n),
            order=(0, 1),
        )
        key_tile = tl.load(key_block, boundary_check=(0, 1), padding_option="zero")
        value_tile = tl.load(value_block, boundary_check=(0, 1), padding_option="zero")
        score = tl.dot(query_tile, key_tile) * scale_log2
        score_slope = tl.full((packed_query_rows, block_n), scale, tl.float32)
        if has_softcap:
            cap_log2 = softcap * 1.4426950408889634
            tanh_score = 2.0 * tl.sigmoid(2.0 * score / cap_log2) - 1.0
            score = cap_log2 * tanh_score
            score_slope *= 1.0 - tanh_score * tanh_score
        key_valid = key_start + key_offsets < sequence_length
        valid = query_valid[:, None] & key_valid[None, :]
        if causal:
            valid &= query_tokens[:, None] >= key_start + key_offsets[None, :]
        probability = tl.where(valid, tl.math.exp2(score - lse[:, None]), 0.0)
        probability_cotangent = tl.dot(output_cotangent_tile, value_tile) * output_scale
        mapped_score_cotangent = probability * (probability_cotangent - delta[:, None])
        raw_score_cotangent = tl.where(valid, mapped_score_cotangent * score_slope, 0.0)
        query_gradient += tl.dot(raw_score_cotangent.to(tl.bfloat16), tl.trans(key_tile))
    tl.store(
        query_cotangent + query_cotangent_offsets,
        query_gradient.to(tl.bfloat16),
        mask=query_valid[:, None],
    )


@triton.jit
def _streaming_dkdv_kernel(
    query,
    key,
    value,
    output_cotangent,
    log_sum_exp,
    output_dot,
    key_cotangent,
    value_cotangent,
    sequence_length,
    query_heads,
    key_value_heads,
    scale,
    scale_log2,
    softcap,
    output_scale,
    stride_qb: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_db: tl.constexpr,
    stride_ds: tl.constexpr,
    stride_dh: tl.constexpr,
    stride_dd: tl.constexpr,
    stride_dkb: tl.constexpr,
    stride_dks: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvs: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvd: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    head_dimension: tl.constexpr,
    head_group_size: tl.constexpr,
    causal: tl.constexpr,
    has_softcap: tl.constexpr,
):
    key_block_index = tl.program_id(0)
    batch_key_value_head = tl.program_id(1)
    batch_index = batch_key_value_head // key_value_heads
    key_value_head = batch_key_value_head % key_value_heads
    key_start = key_block_index * block_n
    key_offsets = key_start + tl.arange(0, block_n)
    key_valid = key_offsets < sequence_length
    key_block = tl.make_block_ptr(
        base=key + batch_index * stride_kb + key_value_head * stride_kh,
        shape=(head_dimension, sequence_length),
        strides=(stride_kd, stride_ks),
        offsets=(0, key_start),
        block_shape=(head_dimension, block_n),
        order=(0, 1),
    )
    value_block = tl.make_block_ptr(
        base=value + batch_index * stride_vb + key_value_head * stride_vh,
        shape=(head_dimension, sequence_length),
        strides=(stride_vd, stride_vs),
        offsets=(0, key_start),
        block_shape=(head_dimension, block_n),
        order=(0, 1),
    )
    key_tile = tl.load(key_block, boundary_check=(0, 1), padding_option="zero")
    value_tile = tl.load(value_block, boundary_check=(0, 1), padding_option="zero")
    key_gradient = tl.zeros((block_n, head_dimension), tl.float32)
    value_gradient = tl.zeros((block_n, head_dimension), tl.float32)
    packed_query_rows: tl.constexpr = block_m * head_group_size
    packed_rows = tl.arange(0, packed_query_rows)
    query_offsets_in_block = packed_rows // head_group_size
    query_head_offsets = packed_rows % head_group_size
    query_heads_for_rows = key_value_head * head_group_size + query_head_offsets
    features = tl.arange(0, head_dimension)
    first_query_start = 0
    if causal:
        first_query_start = (key_start // block_m) * block_m
    for query_start in tl.range(first_query_start, sequence_length, block_m):
        query_start = tl.multiple_of(query_start, block_m)
        query_tokens = query_start + query_offsets_in_block
        query_valid = query_tokens < sequence_length
        query_offsets = (
            batch_index * stride_qb
            + query_tokens[:, None] * stride_qs
            + query_heads_for_rows[:, None] * stride_qh
            + features[None, :] * stride_qd
        )
        output_cotangent_offsets = (
            batch_index * stride_db
            + query_tokens[:, None] * stride_ds
            + query_heads_for_rows[:, None] * stride_dh
            + features[None, :] * stride_dd
        )
        query_tile = tl.load(query + query_offsets, mask=query_valid[:, None], other=0.0)
        output_cotangent_tile = tl.load(
            output_cotangent + output_cotangent_offsets,
            mask=query_valid[:, None],
            other=0.0,
        )
        row_offset = (batch_index * query_heads + query_heads_for_rows) * sequence_length + query_tokens
        lse = tl.load(log_sum_exp + row_offset, mask=query_valid, other=-float("inf"))
        delta = tl.load(output_dot + row_offset, mask=query_valid, other=0.0)
        score = tl.dot(query_tile, key_tile) * scale_log2
        score_slope = tl.full((packed_query_rows, block_n), scale, tl.float32)
        if has_softcap:
            cap_log2 = softcap * 1.4426950408889634
            tanh_score = 2.0 * tl.sigmoid(2.0 * score / cap_log2) - 1.0
            score = cap_log2 * tanh_score
            score_slope *= 1.0 - tanh_score * tanh_score
        valid = query_valid[:, None] & key_valid[None, :]
        if causal:
            valid &= query_tokens[:, None] >= key_offsets[None, :]
        probability = tl.where(valid, tl.math.exp2(score - lse[:, None]), 0.0)
        value_gradient += output_scale * tl.dot(
            tl.trans(probability.to(tl.bfloat16)),
            output_cotangent_tile,
        )
        probability_cotangent = tl.dot(output_cotangent_tile, value_tile) * output_scale
        mapped_score_cotangent = probability * (probability_cotangent - delta[:, None])
        raw_score_cotangent = tl.where(valid, mapped_score_cotangent * score_slope, 0.0)
        key_gradient += tl.dot(tl.trans(raw_score_cotangent.to(tl.bfloat16)), query_tile)
    key_cotangent_offsets = (
        batch_index * stride_dkb
        + key_offsets[:, None] * stride_dks
        + key_value_head * stride_dkh
        + features[None, :] * stride_dkd
    )
    value_cotangent_offsets = (
        batch_index * stride_dvb
        + key_offsets[:, None] * stride_dvs
        + key_value_head * stride_dvh
        + features[None, :] * stride_dvd
    )
    tl.store(key_cotangent + key_cotangent_offsets, key_gradient.to(tl.bfloat16), mask=key_valid[:, None])
    tl.store(value_cotangent + value_cotangent_offsets, value_gradient.to(tl.bfloat16), mask=key_valid[:, None])


@dataclass(frozen=True)
class StreamingAttentionBackwardLaunches:
    """Validated component launches for one generated reverse schedule."""

    query_cotangent: Callable[[], None]
    key_value_cotangents: Callable[[], None]

    def execute(self) -> None:
        """Launch every reverse stage in dependency order."""
        self.query_cotangent()
        self.key_value_cotangents()


def prepare_streaming_attention_backward_launches(
    program: StreamingAttentionBackwardProgram,
    inputs: dict[str, torch.Tensor],
    output: torch.Tensor,
    log_sum_exp: torch.Tensor,
    output_cotangent: torch.Tensor,
    query_cotangent: torch.Tensor,
    key_cotangent: torch.Tensor,
    value_cotangent: torch.Tensor,
    output_dot: torch.Tensor,
    *,
    schedule: StreamingAttentionBackwardTileSchedule,
    num_warps: int,
    num_stages: int,
) -> StreamingAttentionBackwardLaunches:
    """Validate and bind a generic query-major/grouped-key-major reverse schedule."""
    if program.maximum_vjp is not StreamingAttentionBackwardMaximumVJP.NORMALIZED_EXP_INVARIANT:
        raise ValueError(
            "the first reverse emitter requires an explicit legality rewrite to the "
            "normalized-exp maximum-VJP invariant"
        )
    verify_streaming_attention_backward_score_map_vjp(program)
    if len(schedule.query_owner_attachments) != 1:
        raise ValueError("the reverse emitter requires one query-owner Fold attachment")
    output_dot_attachment = schedule.query_owner_attachments[0]
    verify_owner_preparation_fold_attachment(output_dot_attachment)
    if output_dot_attachment.producer != program.output_dot_map or output_dot_attachment.fold != program.output_dot_fold:
        raise ValueError("the query-owner Fold attachment must implement the recovered output-dot program")
    if len(output_dot_attachment.fold.reduction_axes) != 1:
        raise ValueError("the first reverse emitter supports one complete Fold reduction axis")
    complete_feature_axis = output_dot_attachment.fold.reduction_axes[0]
    forward = program.forward
    lowered = lower_score_map(forward)
    block_m = schedule.query_tile_size
    block_n = schedule.key_value_tile_size
    if lowered.bias_name is not None or lowered.mask_name is not None:
        raise ValueError("the first reverse emitter supports domain predicates and scalar score Maps only")
    query = inputs[forward.qk.inputs[0].name]
    key = inputs[forward.qk.inputs[1].name]
    value = inputs[forward.pv.inputs[1].name]
    if any(tensor.dtype is not torch.bfloat16 for tensor in (query, key, value, output, output_cotangent)):
        raise ValueError("the first reverse emitter requires BF16 Q/K/V/output/cotangent")
    if any(tensor.dtype is not torch.bfloat16 for tensor in (query_cotangent, key_cotangent, value_cotangent)):
        raise ValueError("the first reverse emitter writes BF16 input cotangents")
    if query.shape != output.shape or query.shape != output_cotangent.shape or query.shape != query_cotangent.shape:
        raise ValueError("query/output/cotangent shapes must match")
    if key.shape != value.shape or key.shape != key_cotangent.shape or value.shape != value_cotangent.shape:
        raise ValueError("K/V and their cotangent shapes must match")
    if query.shape[1] % block_m or key.shape[1] % block_n:
        raise ValueError("the first reverse emitter requires tile-aligned sequence lengths")
    if query.shape[-1] not in (64, 128) or key.shape[-1] != query.shape[-1]:
        raise ValueError("the first reverse emitter supports head dimensions 64 and 128")
    if complete_feature_axis != program.forward.finalize.output.axes[-1]:
        raise ValueError("the attached Fold reduction axis must be the physical output feature axis")
    if query.shape[-1] != complete_feature_axis.extent:
        raise ValueError("the dQ owner tile must contain the complete attached Fold reduction axis")
    if query.shape[2] % key.shape[2]:
        raise ValueError("query heads must be divisible by key/value heads")
    head_group_size = query.shape[2] // key.shape[2]
    if schedule.query_heads_per_key_value_tile != head_group_size:
        raise ValueError("physical GQA packing must match the Contract input index relation")
    packed_query_rows = block_m * head_group_size
    if packed_query_rows & (packed_query_rows - 1):
        raise ValueError("the Triton grouped-row schedule requires a power-of-two packed row extent")
    if packed_query_rows > 256:
        raise ValueError("the Triton grouped-row schedule supports at most 256 packed query rows")
    expected_domain = (
        StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR
        if lowered.causal
        else StreamingAttentionBackwardDomainTraversal.FULL
    )
    if schedule.domain_traversal is not expected_domain:
        raise ValueError("physical traversal must match the score DomainRestriction")
    if lowered.causal and block_m % block_n:
        raise ValueError("lower-triangular traversal requires query tiles divisible by key tiles")
    expected_state_shape = (query.shape[0], query.shape[2], query.shape[1])
    if log_sum_exp.shape != expected_state_shape or output_dot.shape != expected_state_shape:
        raise ValueError("saved Fold state and output-dot buffer shapes must be [batch, head, query]")
    if log_sum_exp.dtype is not torch.float32 or output_dot.dtype is not torch.float32:
        raise ValueError("saved Fold state and output-dot buffer must be FP32")

    common = (
        query,
        key,
        value,
        output_cotangent,
        log_sum_exp,
        output_dot,
    )
    scalar_arguments = (
        query.shape[1],
        query.shape[2],
        key.shape[2],
        lowered.scale,
        lowered.scale * LOG2_E,
        lowered.softcap or 1.0,
        program.output_scale,
    )
    common_strides = (*query.stride(), *key.stride(), *value.stride(), *output_cotangent.stride())

    def emit_query_cotangent() -> None:
        _streaming_dq_kernel[(triton.cdiv(query.shape[1], block_m), query.shape[0] * key.shape[2])](
            query,
            key,
            value,
            output,
            output_cotangent,
            log_sum_exp,
            output_dot,
            query_cotangent,
            *scalar_arguments,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *output.stride(),
            *output_cotangent.stride(),
            *query_cotangent.stride(),
            block_m,
            block_n,
            complete_feature_axis.extent,
            head_group_size,
            lowered.causal,
            lowered.softcap is not None,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    def emit_key_value_cotangents() -> None:
        _streaming_dkdv_kernel[(triton.cdiv(key.shape[1], block_n), key.shape[0] * key.shape[2])](
            *common,
            key_cotangent,
            value_cotangent,
            *scalar_arguments,
            *common_strides,
            *key_cotangent.stride(),
            *value_cotangent.stride(),
            block_m,
            block_n,
            complete_feature_axis.extent,
            head_group_size,
            lowered.causal,
            lowered.softcap is not None,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return StreamingAttentionBackwardLaunches(
        query_cotangent=emit_query_cotangent,
        key_value_cotangents=emit_key_value_cotangents,
    )


def emit_streaming_attention_backward(
    program: StreamingAttentionBackwardProgram,
    inputs: dict[str, torch.Tensor],
    output: torch.Tensor,
    log_sum_exp: torch.Tensor,
    output_cotangent: torch.Tensor,
    query_cotangent: torch.Tensor,
    key_cotangent: torch.Tensor,
    value_cotangent: torch.Tensor,
    output_dot: torch.Tensor,
    *,
    schedule: StreamingAttentionBackwardTileSchedule,
    num_warps: int,
    num_stages: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute a generic query-major/grouped-key-major reverse schedule."""
    launches = prepare_streaming_attention_backward_launches(
        program,
        inputs,
        output,
        log_sum_exp,
        output_cotangent,
        query_cotangent,
        key_cotangent,
        value_cotangent,
        output_dot,
        schedule=schedule,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    launches.execute()
    return query_cotangent, key_cotangent, value_cotangent


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    return {
        "maximum_absolute_error": difference.max().item(),
        "mean_absolute_error": difference.mean().item(),
    }


def _hash(tensors: tuple[torch.Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _benchmark_variants(
    variants: tuple[tuple[str, Callable[[], None]], ...],
    *,
    warmups: int,
    repeats: int,
    iterations: int,
) -> dict[str, object]:
    for _ in range(warmups):
        for _, function in variants:
            function()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {name: [] for name, _ in variants}
    orders: list[list[str]] = []
    for repeat in range(repeats):
        order = variants if repeat % 2 == 0 else tuple(reversed(variants))
        orders.append([name for name, _ in order])
        for name, function in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                function()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / iterations)
    records = {
        name: {"samples_ms": values, "median_ms": statistics.median(values), "minimum_ms": min(values)}
        for name, values in samples.items()
    }
    result: dict[str, object] = {"variants": records, "execution_order": orders}
    if "oracle" in records:
        result["ratio_generated_to_oracle"] = records["generated"]["median_ms"] / records["oracle"]["median_ms"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument(
        "--semantic-source",
        choices=("jax_vjp_hlo_recovery", "reference_symbolic_vjp"),
        default="jax_vjp_hlo_recovery",
    )
    parser.add_argument("--mutation", choices=("causal", "softcap"), default="causal")
    parser.add_argument("--scale", type=float, default=1.0 / math.sqrt(128))
    parser.add_argument("--block-m", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--block-n", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--num-warps", type=int, choices=(4, 8), default=8)
    parser.add_argument("--num-stages", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--profile-components", action="store_true")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-absolute-error-threshold", type=float, default=0.125)
    parser.add_argument("--mean-absolute-error-threshold", type=float, default=0.01)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--shuttle-revision", required=True)
    args = parser.parse_args()
    if args.repeats % 2:
        raise ValueError("counterbalanced benchmark requires an even repeat count")
    if not torch.cuda.is_available():
        raise RuntimeError("streaming backward benchmark requires CUDA")

    if args.semantic_source == "jax_vjp_hlo_recovery":
        if args.mutation != "causal":
            raise ValueError("natural JAX VJP recovery currently supports the causal domain mutation")
        jax.config.update("jax_platforms", "cpu")
        config = StreamingAttentionBackwardDebugConfig(
            batch=1,
            query_length=args.sequence,
            key_length=args.sequence,
            query_heads=32,
            key_value_heads=8,
            head_dimension=128,
            scale=args.scale,
        )
        graph = import_stablehlo(
            export_debug_streaming_attention_backward(config),
            input_names=STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
        )
        recovered = recover_experimental_whole_pattern_streaming_attention_backward(
            graph,
            schedule=StreamingTileSchedule(
                query_tile_size=args.block_m,
                key_value_tile_size=args.block_n,
                pipeline_depth=args.num_stages,
            ),
        )
        source_backward = recovered.program
        backward = eliminate_normalized_exp_maximum_vjp(
            source_backward,
            numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        )
    else:
        forward = _program(args.sequence, score_scale=args.scale, mutation=args.mutation)
        source_backward = derive_streaming_attention_backward(forward)
        backward = source_backward
        recovered = None
    forward = backward.forward
    semantic_scale = lower_score_map(forward).scale
    backward_schedule = derive_streaming_attention_backward_tile_schedule(
        backward,
        query_tile_size=args.block_m,
        key_value_tile_size=args.block_n,
        domain_traversal=(
            StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR
            if args.mutation == "causal"
            else StreamingAttentionBackwardDomainTraversal.FULL
        ),
    )
    work_estimate = estimate_streaming_attention_backward_work(backward, backward_schedule)
    inputs = _inputs(forward, mutation=args.mutation)
    output = torch.empty_like(inputs["query"])
    log_sum_exp = torch.empty(
        (inputs["query"].shape[0], inputs["query"].shape[2], args.sequence),
        dtype=torch.float32,
        device="cuda",
    )
    emit_streaming_attention(
        forward,
        inputs,
        output,
        log_sum_exp,
        block_m=args.block_m,
        block_n=args.block_n,
        heads_per_program=4,
        num_warps=8,
        num_stages=3,
    )
    generator = torch.Generator(device="cuda").manual_seed(20260809)
    output_cotangent = torch.randn(output.shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    query_cotangent = torch.empty_like(inputs["query"])
    key_cotangent = torch.empty_like(inputs["key"])
    value_cotangent = torch.empty_like(inputs["value"])
    output_dot = torch.empty_like(log_sum_exp)

    generated_launches = prepare_streaming_attention_backward_launches(
        backward,
        inputs,
        output,
        log_sum_exp,
        output_cotangent,
        query_cotangent,
        key_cotangent,
        value_cotangent,
        output_dot,
        schedule=backward_schedule,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    generated_call = generated_launches.execute

    generated_call()
    torch.cuda.synchronize()
    first_hash = _hash((query_cotangent, key_cotangent, value_cotangent))
    generated_call()
    torch.cuda.synchronize()
    second_hash = _hash((query_cotangent, key_cotangent, value_cotangent))
    if first_hash != second_hash:
        raise AssertionError("generated streaming backward is not deterministic")

    oracle_state: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    q_oracle = inputs["query"].transpose(1, 2).detach().requires_grad_(True)
    k_oracle = inputs["key"].transpose(1, 2).detach().requires_grad_(True)
    v_oracle = inputs["value"].transpose(1, 2).detach().requires_grad_(True)
    oracle_output = functional.scaled_dot_product_attention(
        q_oracle,
        k_oracle,
        v_oracle,
        is_causal=args.mutation == "causal",
        scale=semantic_scale,
        enable_gqa=q_oracle.shape[1] != k_oracle.shape[1],
    )
    oracle_cotangent = output_cotangent.transpose(1, 2)

    def oracle_call() -> None:
        oracle_state["grads"] = torch.autograd.grad(
            oracle_output,
            (q_oracle, k_oracle, v_oracle),
            oracle_cotangent,
            retain_graph=True,
        )

    measurements: dict[str, object]
    correctness: dict[str, object]
    if args.mutation == "causal":
        oracle_call()
        oracle_query, oracle_key, oracle_value = oracle_state["grads"]
        correctness = {
            "forward_output": _error(output, oracle_output.transpose(1, 2)),
            "query": _error(query_cotangent, oracle_query.transpose(1, 2)),
            "key": _error(key_cotangent, oracle_key.transpose(1, 2)),
            "value": _error(value_cotangent, oracle_value.transpose(1, 2)),
            "deterministic_hash": first_hash,
        }
        measurements = _benchmark_variants(
            (("generated", generated_call), ("oracle", oracle_call)),
            warmups=args.warmups,
            repeats=args.repeats,
            iterations=args.iterations,
        )
    else:
        correctness = {"deterministic_hash": first_hash, "oracle": "softcap mutation has no matched SDPA oracle"}
        measurements = _benchmark_variants(
            (("generated", generated_call),),
            warmups=args.warmups,
            repeats=args.repeats,
            iterations=args.iterations,
        )

    telemetry = subprocess.check_output(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,power.limit,clocks.current.sm,clocks.current.memory",
            "--format=csv,noheader,nounits",
            "--id=0",
        ),
        text=True,
    ).strip()
    correctness_passes = args.mutation == "causal" and all(
        error["maximum_absolute_error"] <= args.max_absolute_error_threshold
        and error["mean_absolute_error"] <= args.mean_absolute_error_threshold
        for error in correctness.values()
        if isinstance(error, dict) and "maximum_absolute_error" in error
    )
    performance_passes = (
        measurements.get("ratio_generated_to_oracle") is not None and measurements["ratio_generated_to_oracle"] <= 1.2
    )
    component_measurements = None
    if args.profile_components:
        component_measurements = _benchmark_variants(
            (
                ("query_cotangent_with_output_dot", generated_launches.query_cotangent),
                ("key_value_cotangents", generated_launches.key_value_cotangents),
            ),
            warmups=args.warmups,
            repeats=args.repeats,
            iterations=args.iterations,
        )
    result = {
        "schema_version": 1,
        "workload": {
            "sequence": args.sequence,
            "query_heads": inputs["query"].shape[2],
            "key_value_heads": inputs["key"].shape[2],
            "head_dimension": inputs["query"].shape[-1],
            "mutation": args.mutation,
            "requested_score_scale": args.scale,
            "semantic_score_scale": semantic_scale,
            "dtype": "bfloat16",
        },
        "semantic_generation": {
            "provenance": backward.provenance.value,
            "semantic_source": args.semantic_source,
            "accepted_frontend_boundary": (
                "ordinary JAX causal GQA differentiated by jax.vjp and recovered from StableHLO"
                if recovered is not None
                else "local symbolic reverse component oracle"
            ),
            "stages": [stage.value for stage in backward.stages],
            "score_map_vjp": serialize_scalar_expression(backward.score_map_vjp.expression),
            "reassociation": backward.reassociation.value,
            "source_maximum_vjp": source_backward.maximum_vjp.value,
            "physical_maximum_vjp": backward.maximum_vjp.value,
            "maximum_vjp_rewrite": (
                {
                    "property": "normalized exponential is invariant to an additive row constant",
                    "numerical_policy": NumericalPolicy.ALLOW_ROUNDING_REORDER.value,
                    "finite_precision_effect": (
                        "eliminates JAX's explicit equal-split maximum cotangent and changes operation order"
                    ),
                }
                if source_backward.maximum_vjp is not backward.maximum_vjp
                else None
            ),
            "recovered_operation_counts": (
                {
                    "contracts": len(recovered.contract_operation_ids),
                    "normalized_exponential_folds": len(recovered.normalized_exponential_fold_operation_ids),
                    "maximum_vjp_tie_folds": 1,
                    "broadcast_vjp_folds": len(recovered.broadcast_vjp_fold_operation_ids),
                    "domain_restrictions": len(recovered.domain_restriction_operation_ids),
                }
                if recovered is not None
                else None
            ),
            "materialized_values": [value.name for value in backward.materialized_values],
        },
        "schedule": {
            "block_m": args.block_m,
            "block_n": args.block_n,
            "num_warps": args.num_warps,
            "num_stages": args.num_stages,
            "query_gradient_orientation": "query_major_grouped_query_rows",
            "key_value_gradient_orientation": "key_value_major_grouped_query_rows",
            "query_heads_per_key_value_tile": backward_schedule.query_heads_per_key_value_tile,
            "domain_traversal": backward_schedule.domain_traversal.value,
            "key_value_fold_order": backward_schedule.key_value_fold_order.value,
            "atomic_accumulation": False,
            "standalone_output_dot_launch": False,
            "query_owner_fold_attachments": [
                {
                    "producer": attachment.producer.name,
                    "fold": attachment.fold.name,
                    "site": attachment.site.value,
                    "result_disposition": attachment.result_disposition.value,
                    "input_availability": [
                        {
                            "value": available.value.name,
                            "complete_axes": [
                                {"id": axis.id, "extent": axis.extent, "label": axis.label}
                                for axis in available.complete_axes
                            ],
                        }
                        for available in attachment.input_availability
                    ],
                }
                for attachment in backward_schedule.query_owner_attachments
            ],
            "static_work": {
                "logical_query_key_tile_pairs": work_estimate.logical_query_key_tile_pairs,
                "fully_restricted_tile_pairs": work_estimate.fully_restricted_tile_pairs,
                "query_gradient_contract_invocations": work_estimate.query_gradient_contract_invocations,
                "scalar_head_query_gradient_contract_invocations": (
                    work_estimate.scalar_head_query_gradient_contract_invocations
                ),
                "full_domain_query_gradient_contract_invocations": (
                    work_estimate.full_domain_query_gradient_contract_invocations
                ),
                "full_domain_scalar_head_query_gradient_contract_invocations": (
                    work_estimate.full_domain_scalar_head_query_gradient_contract_invocations
                ),
                "query_gradient_contract_invocation_reduction": (
                    work_estimate.query_gradient_contract_invocation_reduction
                ),
                "query_gradient_contract_invocation_reduction_from_full_scalar": (
                    work_estimate.query_gradient_contract_invocation_reduction_from_full_scalar
                ),
                "key_value_gradient_contract_invocations": work_estimate.key_value_gradient_contract_invocations,
                "scalar_head_key_value_contract_invocations": work_estimate.scalar_head_key_value_contract_invocations,
                "full_domain_scalar_head_key_value_contract_invocations": (
                    work_estimate.full_domain_scalar_head_key_value_contract_invocations
                ),
                "key_value_contract_invocation_reduction": work_estimate.key_value_contract_invocation_reduction,
                "key_value_contract_invocation_reduction_from_full_scalar": (
                    work_estimate.key_value_contract_invocation_reduction_from_full_scalar
                ),
                "packed_query_rows": work_estimate.packed_query_rows,
                "peak_score_tile_elements": work_estimate.peak_score_tile_elements,
                "peak_query_tile_elements": work_estimate.peak_query_tile_elements,
                "key_value_gradient_accumulator_elements": work_estimate.key_value_gradient_accumulator_elements,
            },
        },
        "correctness": correctness,
        "measurements": measurements,
        "component_measurements": component_measurements,
        "acceptance": {
            "oracle": "torch SDPA selected backend backward, matched causal GQA semantics",
            "threshold": 1.2,
            "maximum_absolute_error_threshold": args.max_absolute_error_threshold,
            "mean_absolute_error_threshold": args.mean_absolute_error_threshold,
            "ratio": measurements.get("ratio_generated_to_oracle"),
            "correctness_passes": correctness_passes,
            "performance_passes": performance_passes,
            "passes": correctness_passes and performance_passes,
        },
        "benchmark": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "iterations_per_sample": args.iterations,
            "counterbalanced_order": args.mutation == "causal",
        },
        "environment": {
            "jax": jax.__version__,
            "jax_semantic_export_platform": "cpu",
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "telemetry": telemetry,
        },
        "revisions": {"shuttle": args.shuttle_revision},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
