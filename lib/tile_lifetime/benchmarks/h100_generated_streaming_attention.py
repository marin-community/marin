# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Emit and benchmark a compiler-owned H100 streaming-attention skeleton.

The semantic input is a :class:`StreamingAttentionProgram` derived from
Contract/Map/Fold tensor math.  This file deliberately does not import an
attention operator from Torch, FlashAttention, or another kernel package.  Its
Triton kernel is a Q-resident online-softmax skeleton adapted from the public
Triton fused-attention tutorial, with explicit GQA operand indexing and score
map lowering.  Official FA3 is imported only by the optional oracle path.

The first emitter supports BF16 Q/K/V, D=64/128, arbitrary strides, grouped
query heads, causal or dense traversal, scalar scale, optional tanh softcap,
and optional broadcast bias/mask tensors.  It emits one bounded online state
per query tile and materializes only the final output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import triton
import triton.language as tl

from tile_lifetime import (
    AttentionScoreAxis,
    DType,
    ScalarExpression,
    ScalarExpressionKind,
    StreamingAttentionProgram,
    StreamingTileSchedule,
    add_score_bias,
    apply_arbitrary_score_mask,
    apply_causal_score_mask,
    apply_tanh_softcap,
    build_attention_tensor_program,
    derive_streaming_attention,
    scaled_score_map,
)

LOG2_E = 1.4426950408889634
TRITON_TUTORIAL_REVISION = "7c56a5e40f7fd928dfd5c72902d5def0097db73a"
OFFICIAL_FA3_ORACLE_REVISION = "3fa810570e17bb4354155bdb71d826eca6079208"


@dataclass(frozen=True)
class LoweredScoreMap:
    """Physical scalar parameters recovered from the score-map expression."""

    scale: float
    softcap: float | None
    bias_name: str | None
    mask_name: str | None
    query_position_name: str | None
    key_position_name: str | None

    @property
    def causal(self) -> bool:
        return self.query_position_name is not None


def lower_score_map(program: StreamingAttentionProgram) -> LoweredScoreMap:
    """Lower the supported scalar-expression subset without naming attention variants."""
    scale: float | None = None
    softcap: float | None = None
    bias_name: str | None = None
    mask_name: str | None = None
    query_position_name: str | None = None
    key_position_name: str | None = None

    def literal(expression: ScalarExpression) -> float | bool | None:
        if expression.kind is ScalarExpressionKind.CONSTANT:
            return expression.constant
        return None

    def input_name(expression: ScalarExpression) -> str | None:
        if expression.kind is ScalarExpressionKind.INPUT:
            return expression.input_name
        return None

    def visit(expression: ScalarExpression) -> None:
        nonlocal scale, softcap, bias_name, mask_name, query_position_name, key_position_name
        if expression.kind is ScalarExpressionKind.SELECT:
            predicate, selected, rejected = expression.operands
            if literal(rejected) != float("-inf"):
                raise ValueError("the H100 score-map emitter only lowers select-to-negative-infinity masks")
            if predicate.kind is ScalarExpressionKind.LESS_EQUAL:
                key_position_name = input_name(predicate.operands[0])
                query_position_name = input_name(predicate.operands[1])
                if query_position_name is None or key_position_name is None:
                    raise ValueError("causal position comparison must reference input positions")
            else:
                mask_name = input_name(predicate)
                if mask_name is None:
                    raise ValueError("the H100 score-map emitter requires a tensor-valued Boolean mask")
            visit(selected)
            return
        if expression.kind is ScalarExpressionKind.ADD:
            left, right = expression.operands
            left_name = input_name(left)
            right_name = input_name(right)
            if left_name is not None and left_name != program.qk.output.name:
                bias_name = left_name
                visit(right)
                return
            if right_name is not None and right_name != program.qk.output.name:
                bias_name = right_name
                visit(left)
                return
            raise ValueError("the H100 score-map emitter requires one tensor bias operand")
        if expression.kind is ScalarExpressionKind.MULTIPLY:
            left, right = expression.operands
            left_literal = literal(left)
            right_literal = literal(right)
            tanh_expression = right if left_literal is not None else left
            cap_literal = left_literal if left_literal is not None else right_literal
            if tanh_expression.kind is ScalarExpressionKind.TANH and cap_literal is not None:
                divided = tanh_expression.operands[0]
                if divided.kind is not ScalarExpressionKind.DIVIDE or literal(divided.operands[1]) != cap_literal:
                    raise ValueError("softcap must have the form cap * tanh(score / cap)")
                softcap = float(cap_literal)
                visit(divided.operands[0])
                return
            raw_name = input_name(left) or input_name(right)
            scalar = right_literal if input_name(left) is not None else left_literal
            if raw_name != program.qk.output.name or scalar is None:
                raise ValueError("raw QK scores must be multiplied by one scalar")
            scale = float(scalar)
            return
        raise ValueError(f"unsupported H100 score-map expression {expression.kind.value}")

    visit(program.score_map.expression)
    if scale is None:
        raise ValueError("score map does not contain a scalar QK scale")
    return LoweredScoreMap(
        scale=scale,
        softcap=softcap,
        bias_name=bias_name,
        mask_name=mask_name,
        query_position_name=query_position_name,
        key_position_name=key_position_name,
    )


@triton.jit
def _stream_kv_tiles(
    accumulator,
    row_sum,
    row_max,
    query_tile,
    key,
    value,
    bias,
    score_mask,
    query_positions,
    key_positions,
    batch_index,
    key_value_head,
    query_head,
    query_start,
    key_start,
    key_stop,
    sequence_length,
    scale_log2,
    softcap,
    stride_kb: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_bias_b: tl.constexpr,
    stride_bias_h: tl.constexpr,
    stride_bias_q: tl.constexpr,
    stride_bias_k: tl.constexpr,
    stride_mask_b: tl.constexpr,
    stride_mask_h: tl.constexpr,
    stride_mask_q: tl.constexpr,
    stride_mask_k: tl.constexpr,
    stride_query_position: tl.constexpr,
    stride_key_position: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    head_dimension: tl.constexpr,
    causal: tl.constexpr,
    has_bias: tl.constexpr,
    has_mask: tl.constexpr,
    has_softcap: tl.constexpr,
):
    query_offsets = query_start + tl.arange(0, block_m)
    key_offsets = tl.arange(0, block_n)
    for current_key in tl.range(key_start, key_stop, block_n):
        current_key = tl.multiple_of(current_key, block_n)
        key_block = tl.make_block_ptr(
            base=key + batch_index * stride_kb + key_value_head * stride_kh,
            shape=(head_dimension, sequence_length),
            strides=(stride_kd, stride_ks),
            offsets=(0, current_key),
            block_shape=(head_dimension, block_n),
            order=(0, 1),
        )
        value_block = tl.make_block_ptr(
            base=value + batch_index * stride_vb + key_value_head * stride_vh,
            shape=(sequence_length, head_dimension),
            strides=(stride_vs, stride_vd),
            offsets=(current_key, 0),
            block_shape=(block_n, head_dimension),
            order=(1, 0),
        )
        key_tile = tl.load(key_block, boundary_check=(0, 1), padding_option="zero")
        scores = tl.dot(query_tile, key_tile) * scale_log2
        if has_softcap:
            cap_log2 = softcap * 1.4426950408889634
            scores = cap_log2 * (2.0 * tl.sigmoid(2.0 * scores / cap_log2) - 1.0)
        if has_bias:
            bias_offsets = (
                batch_index * stride_bias_b
                + query_head * stride_bias_h
                + query_offsets[:, None] * stride_bias_q
                + (current_key + key_offsets[None, :]) * stride_bias_k
            )
            scores += tl.load(bias + bias_offsets).to(tl.float32) * 1.4426950408889634
        valid = tl.full((block_m, block_n), True, tl.int1)
        if has_mask:
            mask_offsets = (
                batch_index * stride_mask_b
                + query_head * stride_mask_h
                + query_offsets[:, None] * stride_mask_q
                + (current_key + key_offsets[None, :]) * stride_mask_k
            )
            valid &= tl.load(score_mask + mask_offsets).to(tl.int1)
        if causal:
            query_position = tl.load(query_positions + query_offsets * stride_query_position)
            key_position = tl.load(key_positions + (current_key + key_offsets) * stride_key_position)
            valid &= query_position[:, None] >= key_position[None, :]
        scores = tl.where(valid, scores, -float("inf"))
        next_max = tl.maximum(row_max, tl.max(scores, axis=1))
        probabilities = tl.math.exp2(scores - next_max[:, None])
        correction = tl.math.exp2(row_max - next_max)
        row_sum = row_sum * correction + tl.sum(probabilities, axis=1)
        accumulator *= correction[:, None]
        value_tile = tl.load(value_block)
        accumulator = tl.dot(probabilities.to(tl.bfloat16), value_tile, accumulator)
        row_max = next_max
    return accumulator, row_sum, row_max


@triton.jit
def _streaming_attention_forward(
    query,
    key,
    value,
    output,
    log_sum_exp,
    bias,
    score_mask,
    query_positions,
    key_positions,
    sequence_length,
    query_heads,
    key_value_heads,
    scale_log2,
    softcap,
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
    stride_bias_b: tl.constexpr,
    stride_bias_h: tl.constexpr,
    stride_bias_q: tl.constexpr,
    stride_bias_k: tl.constexpr,
    stride_mask_b: tl.constexpr,
    stride_mask_h: tl.constexpr,
    stride_mask_q: tl.constexpr,
    stride_mask_k: tl.constexpr,
    stride_query_position: tl.constexpr,
    stride_key_position: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    head_dimension: tl.constexpr,
    causal: tl.constexpr,
    has_bias: tl.constexpr,
    has_mask: tl.constexpr,
    has_softcap: tl.constexpr,
    natural_log_sum_exp: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_query_head = tl.program_id(1)
    batch_index = batch_query_head // query_heads
    query_head = batch_query_head % query_heads
    head_group_size = query_heads // key_value_heads
    key_value_head = query_head // head_group_size
    query_start = query_tile_index * block_m
    query_block = tl.make_block_ptr(
        base=query + batch_index * stride_qb + query_head * stride_qh,
        shape=(sequence_length, head_dimension),
        strides=(stride_qs, stride_qd),
        offsets=(query_start, 0),
        block_shape=(block_m, head_dimension),
        order=(1, 0),
    )
    output_block = tl.make_block_ptr(
        base=output + batch_index * stride_ob + query_head * stride_oh,
        shape=(sequence_length, head_dimension),
        strides=(stride_os, stride_od),
        offsets=(query_start, 0),
        block_shape=(block_m, head_dimension),
        order=(1, 0),
    )
    query_tile = tl.load(query_block)
    accumulator = tl.zeros((block_m, head_dimension), tl.float32)
    row_sum = tl.full((block_m,), 1.0, tl.float32)
    row_max = tl.full((block_m,), -float("inf"), tl.float32)

    if causal:
        off_diagonal_stop = query_tile_index * block_m
        accumulator, row_sum, row_max = _stream_kv_tiles(
            accumulator,
            row_sum,
            row_max,
            query_tile,
            key,
            value,
            bias,
            score_mask,
            query_positions,
            key_positions,
            batch_index,
            key_value_head,
            query_head,
            query_start,
            0,
            off_diagonal_stop,
            sequence_length,
            scale_log2,
            softcap,
            stride_kb,
            stride_ks,
            stride_kh,
            stride_kd,
            stride_vb,
            stride_vs,
            stride_vh,
            stride_vd,
            stride_bias_b,
            stride_bias_h,
            stride_bias_q,
            stride_bias_k,
            stride_mask_b,
            stride_mask_h,
            stride_mask_q,
            stride_mask_k,
            stride_query_position,
            stride_key_position,
            block_m,
            block_n,
            head_dimension,
            False,
            has_bias,
            has_mask,
            has_softcap,
        )
        accumulator, row_sum, row_max = _stream_kv_tiles(
            accumulator,
            row_sum,
            row_max,
            query_tile,
            key,
            value,
            bias,
            score_mask,
            query_positions,
            key_positions,
            batch_index,
            key_value_head,
            query_head,
            query_start,
            off_diagonal_stop,
            off_diagonal_stop + block_m,
            sequence_length,
            scale_log2,
            softcap,
            stride_kb,
            stride_ks,
            stride_kh,
            stride_kd,
            stride_vb,
            stride_vs,
            stride_vh,
            stride_vd,
            stride_bias_b,
            stride_bias_h,
            stride_bias_q,
            stride_bias_k,
            stride_mask_b,
            stride_mask_h,
            stride_mask_q,
            stride_mask_k,
            stride_query_position,
            stride_key_position,
            block_m,
            block_n,
            head_dimension,
            True,
            has_bias,
            has_mask,
            has_softcap,
        )
    else:
        accumulator, row_sum, row_max = _stream_kv_tiles(
            accumulator,
            row_sum,
            row_max,
            query_tile,
            key,
            value,
            bias,
            score_mask,
            query_positions,
            key_positions,
            batch_index,
            key_value_head,
            query_head,
            query_start,
            0,
            sequence_length,
            sequence_length,
            scale_log2,
            softcap,
            stride_kb,
            stride_ks,
            stride_kh,
            stride_kd,
            stride_vb,
            stride_vs,
            stride_vh,
            stride_vd,
            stride_bias_b,
            stride_bias_h,
            stride_bias_q,
            stride_bias_k,
            stride_mask_b,
            stride_mask_h,
            stride_mask_q,
            stride_mask_k,
            stride_query_position,
            stride_key_position,
            block_m,
            block_n,
            head_dimension,
            False,
            has_bias,
            has_mask,
            has_softcap,
        )

    accumulator /= row_sum[:, None]
    tl.store(output_block, accumulator.to(tl.bfloat16))
    row_offsets = batch_query_head * sequence_length + query_start + tl.arange(0, block_m)
    log_normalizer = row_max + tl.math.log2(row_sum)
    if natural_log_sum_exp:
        log_normalizer /= 1.4426950408889634
    tl.store(log_sum_exp + row_offsets, log_normalizer)


@triton.jit
def _stream_grouped_query_kv_tiles(
    accumulator,
    row_sum,
    row_max,
    query_tile,
    key,
    value,
    bias,
    score_mask,
    query_positions,
    key_positions,
    batch_index,
    key_value_head,
    query_tokens,
    query_head_indices,
    key_start,
    key_stop,
    sequence_length,
    scale_log2,
    softcap,
    stride_kb: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_bias_b: tl.constexpr,
    stride_bias_h: tl.constexpr,
    stride_bias_q: tl.constexpr,
    stride_bias_k: tl.constexpr,
    stride_mask_b: tl.constexpr,
    stride_mask_h: tl.constexpr,
    stride_mask_q: tl.constexpr,
    stride_mask_k: tl.constexpr,
    stride_query_position: tl.constexpr,
    stride_key_position: tl.constexpr,
    tile_rows: tl.constexpr,
    block_n: tl.constexpr,
    head_dimension: tl.constexpr,
    causal: tl.constexpr,
    has_bias: tl.constexpr,
    has_mask: tl.constexpr,
    has_softcap: tl.constexpr,
):
    key_offsets = tl.arange(0, block_n)
    for current_key in tl.range(key_start, key_stop, block_n):
        current_key = tl.multiple_of(current_key, block_n)
        key_block = tl.make_block_ptr(
            base=key + batch_index * stride_kb + key_value_head * stride_kh,
            shape=(head_dimension, sequence_length),
            strides=(stride_kd, stride_ks),
            offsets=(0, current_key),
            block_shape=(head_dimension, block_n),
            order=(0, 1),
        )
        value_block = tl.make_block_ptr(
            base=value + batch_index * stride_vb + key_value_head * stride_vh,
            shape=(sequence_length, head_dimension),
            strides=(stride_vs, stride_vd),
            offsets=(current_key, 0),
            block_shape=(block_n, head_dimension),
            order=(1, 0),
        )
        key_tile = tl.load(key_block, boundary_check=(0, 1), padding_option="zero")
        scores = tl.dot(query_tile, key_tile) * scale_log2
        if has_softcap:
            cap_log2 = softcap * 1.4426950408889634
            scores = cap_log2 * (2.0 * tl.sigmoid(2.0 * scores / cap_log2) - 1.0)
        if has_bias:
            bias_offsets = (
                batch_index * stride_bias_b
                + query_head_indices[:, None] * stride_bias_h
                + query_tokens[:, None] * stride_bias_q
                + (current_key + key_offsets[None, :]) * stride_bias_k
            )
            scores += tl.load(bias + bias_offsets).to(tl.float32) * 1.4426950408889634
        query_valid = query_tokens < sequence_length
        key_valid = current_key + key_offsets < sequence_length
        valid = query_valid[:, None] & key_valid[None, :]
        if has_mask:
            mask_offsets = (
                batch_index * stride_mask_b
                + query_head_indices[:, None] * stride_mask_h
                + query_tokens[:, None] * stride_mask_q
                + (current_key + key_offsets[None, :]) * stride_mask_k
            )
            valid &= tl.load(score_mask + mask_offsets, mask=valid, other=0).to(tl.int1)
        if causal:
            query_position = tl.load(
                query_positions + query_tokens * stride_query_position,
                mask=query_valid,
                other=0,
            )
            key_position = tl.load(
                key_positions + (current_key + key_offsets) * stride_key_position,
                mask=key_valid,
                other=0,
            )
            valid &= query_position[:, None] >= key_position[None, :]
        scores = tl.where(valid, scores, -float("inf"))
        next_max = tl.maximum(row_max, tl.max(scores, axis=1))
        probabilities = tl.where(valid, tl.math.exp2(scores - next_max[:, None]), 0.0)
        correction = tl.math.exp2(row_max - next_max)
        row_sum = row_sum * correction + tl.sum(probabilities, axis=1)
        accumulator *= correction[:, None]
        value_tile = tl.load(value_block, boundary_check=(0, 1), padding_option="zero")
        accumulator = tl.dot(probabilities.to(tl.bfloat16), value_tile, accumulator)
        row_max = next_max
    return accumulator, row_sum, row_max


@triton.jit
def _streaming_grouped_query_forward(
    query,
    key,
    value,
    output,
    log_sum_exp,
    bias,
    score_mask,
    query_positions,
    key_positions,
    sequence_length,
    query_heads,
    key_value_heads,
    scale_log2,
    softcap,
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
    stride_bias_b: tl.constexpr,
    stride_bias_h: tl.constexpr,
    stride_bias_q: tl.constexpr,
    stride_bias_k: tl.constexpr,
    stride_mask_b: tl.constexpr,
    stride_mask_h: tl.constexpr,
    stride_mask_q: tl.constexpr,
    stride_mask_k: tl.constexpr,
    stride_query_position: tl.constexpr,
    stride_key_position: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    head_dimension: tl.constexpr,
    heads_per_program: tl.constexpr,
    causal: tl.constexpr,
    has_bias: tl.constexpr,
    has_mask: tl.constexpr,
    has_softcap: tl.constexpr,
    natural_log_sum_exp: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_head_group = tl.program_id(1)
    head_groups = query_heads // heads_per_program
    batch_index = batch_head_group // head_groups
    query_head_group = batch_head_group % head_groups
    head_group_size = query_heads // key_value_heads
    key_value_head = (query_head_group * heads_per_program) // head_group_size
    tile_rows: tl.constexpr = block_m * heads_per_program
    flattened_rows = tl.arange(0, tile_rows)
    query_tokens = query_tile_index * block_m + flattened_rows // heads_per_program
    query_head_indices = query_head_group * heads_per_program + flattened_rows % heads_per_program
    features = tl.arange(0, head_dimension)
    query_offsets = (
        batch_index * stride_qb
        + query_tokens[:, None] * stride_qs
        + query_head_indices[:, None] * stride_qh
        + features[None, :] * stride_qd
    )
    query_valid = query_tokens < sequence_length
    query_tile = tl.load(query + query_offsets, mask=query_valid[:, None], other=0.0)
    accumulator = tl.zeros((tile_rows, head_dimension), tl.float32)
    row_sum = tl.full((tile_rows,), 1.0, tl.float32)
    row_max = tl.full((tile_rows,), -float("inf"), tl.float32)
    if causal:
        off_diagonal_stop = query_tile_index * block_m
        accumulator, row_sum, row_max = _stream_grouped_query_kv_tiles(
            accumulator,
            row_sum,
            row_max,
            query_tile,
            key,
            value,
            bias,
            score_mask,
            query_positions,
            key_positions,
            batch_index,
            key_value_head,
            query_tokens,
            query_head_indices,
            0,
            off_diagonal_stop,
            sequence_length,
            scale_log2,
            softcap,
            stride_kb,
            stride_ks,
            stride_kh,
            stride_kd,
            stride_vb,
            stride_vs,
            stride_vh,
            stride_vd,
            stride_bias_b,
            stride_bias_h,
            stride_bias_q,
            stride_bias_k,
            stride_mask_b,
            stride_mask_h,
            stride_mask_q,
            stride_mask_k,
            stride_query_position,
            stride_key_position,
            tile_rows,
            block_n,
            head_dimension,
            False,
            has_bias,
            has_mask,
            has_softcap,
        )
        accumulator, row_sum, row_max = _stream_grouped_query_kv_tiles(
            accumulator,
            row_sum,
            row_max,
            query_tile,
            key,
            value,
            bias,
            score_mask,
            query_positions,
            key_positions,
            batch_index,
            key_value_head,
            query_tokens,
            query_head_indices,
            off_diagonal_stop,
            off_diagonal_stop + block_m,
            sequence_length,
            scale_log2,
            softcap,
            stride_kb,
            stride_ks,
            stride_kh,
            stride_kd,
            stride_vb,
            stride_vs,
            stride_vh,
            stride_vd,
            stride_bias_b,
            stride_bias_h,
            stride_bias_q,
            stride_bias_k,
            stride_mask_b,
            stride_mask_h,
            stride_mask_q,
            stride_mask_k,
            stride_query_position,
            stride_key_position,
            tile_rows,
            block_n,
            head_dimension,
            True,
            has_bias,
            has_mask,
            has_softcap,
        )
    else:
        accumulator, row_sum, row_max = _stream_grouped_query_kv_tiles(
            accumulator,
            row_sum,
            row_max,
            query_tile,
            key,
            value,
            bias,
            score_mask,
            query_positions,
            key_positions,
            batch_index,
            key_value_head,
            query_tokens,
            query_head_indices,
            0,
            sequence_length,
            sequence_length,
            scale_log2,
            softcap,
            stride_kb,
            stride_ks,
            stride_kh,
            stride_kd,
            stride_vb,
            stride_vs,
            stride_vh,
            stride_vd,
            stride_bias_b,
            stride_bias_h,
            stride_bias_q,
            stride_bias_k,
            stride_mask_b,
            stride_mask_h,
            stride_mask_q,
            stride_mask_k,
            stride_query_position,
            stride_key_position,
            tile_rows,
            block_n,
            head_dimension,
            False,
            has_bias,
            has_mask,
            has_softcap,
        )
    accumulator /= row_sum[:, None]
    output_offsets = (
        batch_index * stride_ob
        + query_tokens[:, None] * stride_os
        + query_head_indices[:, None] * stride_oh
        + features[None, :] * stride_od
    )
    tl.store(output + output_offsets, accumulator.to(tl.bfloat16), mask=query_valid[:, None])
    lse_offsets = batch_index * query_heads * sequence_length + query_head_indices * sequence_length + query_tokens
    log_normalizer = row_max + tl.math.log2(row_sum)
    if natural_log_sum_exp:
        log_normalizer /= 1.4426950408889634
    tl.store(log_sum_exp + lse_offsets, log_normalizer, mask=query_valid)


def _aligned_score_strides(
    program: StreamingAttentionProgram,
    name: str | None,
    tensor: torch.Tensor,
) -> tuple[int, int, int, int]:
    if name is None:
        return (0, 0, 0, 0)
    value = next(value for value in program.score_map.inputs if value.name == name)
    stride_by_axis = {axis.label: stride for axis, stride in zip(value.axes, tensor.stride(), strict=True)}
    return tuple(
        stride_by_axis.get(axis.value, 0)
        for axis in (
            AttentionScoreAxis.BATCH,
            AttentionScoreAxis.HEAD,
            AttentionScoreAxis.QUERY,
            AttentionScoreAxis.KEY,
        )
    )


def emit_streaming_attention(
    program: StreamingAttentionProgram,
    inputs: dict[str, torch.Tensor],
    output: torch.Tensor,
    log_sum_exp: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    heads_per_program: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    """Execute one generated H100 program without an opaque attention call."""
    query = inputs[program.qk.inputs[0].name]
    key = inputs[program.qk.inputs[1].name]
    value = inputs[program.pv.inputs[1].name]
    if query.dtype is not torch.bfloat16 or key.dtype is not torch.bfloat16 or value.dtype is not torch.bfloat16:
        raise ValueError("the first H100 emitter supports BF16 Q/K/V")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Q/K/V must have [batch, sequence, head, feature] shape")
    if query.shape[0] != key.shape[0] or query.shape[1] != key.shape[1] or key.shape != value.shape:
        raise ValueError("the first H100 emitter requires equal batch/sequence K/V shapes")
    if query.shape[-1] not in (64, 128) or key.shape[-1] != query.shape[-1]:
        raise ValueError("the first H100 emitter supports equal Q/K/V dimensions 64 or 128")
    if query.shape[2] % key.shape[2]:
        raise ValueError("query heads must be an integer multiple of key/value heads")
    key_map = program.qk.index_maps_for_input(1)
    value_map = program.pv.index_maps_for_input(1)
    expected_divisor = query.shape[2] // key.shape[2]
    if len(key_map) != 1 or len(value_map) != 1 or key_map[0].divisor != expected_divisor:
        raise ValueError("the semantic contractions do not expose the physical grouped-head relation")
    if value_map[0].divisor != expected_divisor:
        raise ValueError("QK and PV grouped-head relations disagree")
    if expected_divisor % heads_per_program or query.shape[2] % heads_per_program:
        raise ValueError("heads per program must divide the semantic grouped-head relation")
    sequence_length = query.shape[1]
    if sequence_length % block_m or sequence_length % block_n:
        raise ValueError("the measured H100 path requires sequence lengths aligned to both tile sizes")

    lowered = lower_score_map(program)
    if lowered.causal and block_m % block_n:
        raise ValueError("causal diagonal splitting requires the query tile to be a multiple of the key tile")
    dummy = query
    bias = inputs[lowered.bias_name] if lowered.bias_name is not None else dummy
    score_mask = inputs[lowered.mask_name] if lowered.mask_name is not None else dummy
    query_positions = (
        inputs[lowered.query_position_name]
        if lowered.query_position_name is not None
        else torch.arange(sequence_length, dtype=torch.int32, device=query.device)
    )
    key_positions = inputs[lowered.key_position_name] if lowered.key_position_name is not None else query_positions
    bias_strides = _aligned_score_strides(program, lowered.bias_name, bias)
    mask_strides = _aligned_score_strides(program, lowered.mask_name, score_mask)
    expected_lse_shape = (query.shape[0], query.shape[2], sequence_length)
    if output.shape != query.shape or output.dtype != query.dtype or output.device != query.device:
        raise ValueError("the H100 emitter output must match the query shape, dtype, and device")
    if log_sum_exp.shape != expected_lse_shape or log_sum_exp.dtype is not torch.float32:
        raise ValueError(f"log-sum-exp must be FP32 with shape {expected_lse_shape}")
    kernel = _streaming_grouped_query_forward if heads_per_program > 1 else _streaming_attention_forward
    grid_heads = query.shape[0] * query.shape[2] // heads_per_program
    common_arguments = (
        query,
        key,
        value,
        output,
        log_sum_exp,
        bias,
        score_mask,
        query_positions,
        key_positions,
        sequence_length,
        query.shape[2],
        key.shape[2],
        lowered.scale * LOG2_E,
        lowered.softcap or 1.0,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *output.stride(),
        *bias_strides,
        *mask_strides,
        query_positions.stride(0),
        key_positions.stride(0),
        block_m,
        block_n,
        query.shape[-1],
        lowered.causal,
        lowered.bias_name is not None,
        lowered.mask_name is not None,
        lowered.softcap is not None,
        False,
    )
    if heads_per_program > 1:
        kernel[(triton.cdiv(sequence_length, block_m), grid_heads)](
            *common_arguments[:-5],
            heads_per_program,
            *common_arguments[-5:],
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        kernel[(triton.cdiv(sequence_length, block_m), grid_heads)](
            *common_arguments,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return output


def _program(
    sequence_length: int,
    *,
    score_scale: float,
    mutation: str,
) -> StreamingAttentionProgram:
    score_map = scaled_score_map(score_scale)
    if mutation == "causal":
        score_map = apply_causal_score_mask(score_map)
    elif mutation == "softcap":
        score_map = apply_tanh_softcap(score_map, cap=30.0)
        score_map = apply_causal_score_mask(score_map)
    elif mutation == "bias_mask":
        score_map = add_score_bias(score_map)
        score_map = apply_arbitrary_score_mask(score_map)
    else:
        raise ValueError(f"unknown mutation {mutation!r}")
    source = build_attention_tensor_program(
        batch_size=1,
        query_length=sequence_length,
        key_length=sequence_length,
        query_heads=32,
        key_value_heads=8,
        key_dimension=128,
        value_dimension=128,
        score_map=score_map,
        input_dtype=DType.BF16,
    )
    return derive_streaming_attention(
        source,
        schedule=StreamingTileSchedule(query_tile_size=128, key_value_tile_size=64, pipeline_depth=3),
    )


def _inputs(program: StreamingAttentionProgram, *, mutation: str) -> dict[str, torch.Tensor]:
    torch.manual_seed(17)
    inputs = {
        "query": torch.randn(program.qk.inputs[0].shape, dtype=torch.bfloat16, device="cuda"),
        "key": torch.randn(program.qk.inputs[1].shape, dtype=torch.bfloat16, device="cuda"),
        "value": torch.randn(program.pv.inputs[1].shape, dtype=torch.bfloat16, device="cuda"),
    }
    sequence_length = program.qk.inputs[0].shape[1]
    if mutation in ("causal", "softcap"):
        inputs["query.position"] = torch.arange(sequence_length, dtype=torch.int32, device="cuda")
        inputs["key.position"] = torch.arange(sequence_length, dtype=torch.int32, device="cuda")
    else:
        inputs["score.bias"] = (
            torch.randn((32, sequence_length, sequence_length), dtype=torch.float32, device="cuda") * 0.01
        )
        mask = torch.rand((1, 32, sequence_length, sequence_length), device="cuda") > 0.1
        mask[..., 0] = True
        inputs["score.mask"] = mask
    return inputs


def _timings(call, *, warmups: int, repeats: int, iterations: int) -> list[float]:
    for _ in range(warmups):
        call()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return samples


def _hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _hardware_state() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,clocks.current.sm,clocks.current.memory,power.limit",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()


def _correctness_reference(
    program: StreamingAttentionProgram,
    inputs: dict[str, torch.Tensor],
    *,
    query_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize a small independent score slice for numerical validation."""
    query = inputs["query"]
    key = inputs["key"]
    value = inputs["value"]
    sequence_length = query.shape[1]
    edge_rows = min(query_rows // 2, sequence_length // 2)
    query_indices = torch.cat(
        (
            torch.arange(edge_rows, device=query.device),
            torch.arange(sequence_length - edge_rows, sequence_length, device=query.device),
        )
    )
    ratio = query.shape[2] // key.shape[2]
    expanded_key = torch.repeat_interleave(key.float(), ratio, dim=2)
    expanded_value = torch.repeat_interleave(value.float(), ratio, dim=2)
    selected_query = query[:, query_indices].float()
    scores = torch.einsum("bqhd,bkhd->bhqk", selected_query, expanded_key)
    lowered = lower_score_map(program)
    scores *= lowered.scale
    if lowered.softcap is not None:
        scores = lowered.softcap * torch.tanh(scores / lowered.softcap)
    if lowered.bias_name is not None:
        bias = inputs[lowered.bias_name]
        if bias.ndim == 3:
            scores += bias[:, query_indices, :].unsqueeze(0)
        else:
            scores += bias[:, :, query_indices, :]
    valid = torch.ones_like(scores, dtype=torch.bool)
    if lowered.mask_name is not None:
        valid &= inputs[lowered.mask_name][:, :, query_indices, :]
    if lowered.causal:
        assert lowered.query_position_name is not None
        assert lowered.key_position_name is not None
        query_positions = inputs[lowered.query_position_name][query_indices]
        key_positions = inputs[lowered.key_position_name]
        valid &= query_positions[None, None, :, None] >= key_positions[None, None, None, :]
    scores = torch.where(valid, scores, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    reference = torch.einsum("bhqk,bkhd->bqhd", probabilities, expanded_value)
    return query_indices, reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, choices=(2048, 4096), default=2048)
    parser.add_argument("--mutation", choices=("causal", "softcap", "bias_mask"), default="causal")
    parser.add_argument("--scale", type=float, default=1.0 / math.sqrt(128))
    parser.add_argument("--block-m", type=int, choices=(16, 32, 64, 128), default=32)
    parser.add_argument("--block-n", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--heads-per-program", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--num-warps", type=int, choices=(4, 8), default=8)
    parser.add_argument("--num-stages", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--oracle-fa3", action="store_true")
    parser.add_argument("--correctness-rows", type=int, default=16)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--shuttle-revision", required=True)
    args = parser.parse_args()

    program = _program(args.sequence, score_scale=args.scale, mutation=args.mutation)
    inputs = _inputs(program, mutation=args.mutation)
    output = torch.empty_like(inputs["query"])
    log_sum_exp = torch.empty(
        (inputs["query"].shape[0], inputs["query"].shape[2], args.sequence),
        dtype=torch.float32,
        device="cuda",
    )

    def generated_call() -> torch.Tensor:
        return emit_streaming_attention(
            program,
            inputs,
            output,
            log_sum_exp,
            block_m=args.block_m,
            block_n=args.block_n,
            heads_per_program=args.heads_per_program,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )

    generated_output = generated_call()
    correctness_indices, correctness_reference = _correctness_reference(
        program,
        inputs,
        query_rows=args.correctness_rows,
    )
    correctness_difference = (generated_output[:, correctness_indices].float() - correctness_reference.float()).abs()
    repeated_hashes = [_hash(generated_call()) for _ in range(3)]
    generated_samples = _timings(
        generated_call,
        warmups=args.warmups,
        repeats=args.repeats,
        iterations=args.iterations,
    )

    oracle_samples: list[float] | None = None
    maximum_absolute_error: float | None = None
    mean_absolute_error: float | None = None
    if args.oracle_fa3:
        if args.mutation != "causal":
            raise ValueError("official FA3 oracle comparison is defined only for the causal score program")
        flash_attn_3_gpu = importlib.import_module("flash_attn_interface").flash_attn_3_gpu

        def oracle_call() -> torch.Tensor:
            return flash_attn_3_gpu.fwd(
                inputs["query"],
                inputs["key"],
                inputs["value"],
                softmax_scale=args.scale,
                is_causal=True,
                is_rotary_interleaved=True,
                num_splits=1,
                pack_gqa=True,
            )[0]

        oracle_output = oracle_call()
        difference = (generated_output.float() - oracle_output.float()).abs()
        maximum_absolute_error = float(difference.max().item())
        mean_absolute_error = float(difference.mean().item())
        oracle_samples = _timings(
            oracle_call,
            warmups=args.warmups,
            repeats=args.repeats,
            iterations=args.iterations,
        )

    result = {
        "benchmark": "shuttle_generated_streaming_attention_h100",
        "shuttle_revision": args.shuttle_revision,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "triton_tutorial_revision": TRITON_TUTORIAL_REVISION,
        "official_fa3_oracle_revision": OFFICIAL_FA3_ORACLE_REVISION,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "hardware_state": _hardware_state(),
        "emitter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": vars(args) | {"json_output": str(args.json_output) if args.json_output else None},
        "score_lowering": asdict(lower_score_map(program)),
        "semantic_head_index_divisor": program.qk.index_maps_for_input(1)[0].divisor,
        "generated_samples_ms": generated_samples,
        "generated_median_ms": statistics.median(generated_samples),
        "generated_minimum_ms": min(generated_samples),
        "deterministic_hashes": repeated_hashes,
        "deterministic": len(set(repeated_hashes)) == 1,
        "correctness_query_indices": correctness_indices.tolist(),
        "reference_maximum_absolute_error": float(correctness_difference.max().item()),
        "reference_mean_absolute_error": float(correctness_difference.mean().item()),
        "oracle_fa3_samples_ms": oracle_samples,
        "oracle_fa3_median_ms": statistics.median(oracle_samples) if oracle_samples else None,
        "generated_over_oracle": (
            statistics.median(generated_samples) / statistics.median(oracle_samples) if oracle_samples else None
        ),
        "maximum_absolute_error": maximum_absolute_error,
        "mean_absolute_error": mean_absolute_error,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
