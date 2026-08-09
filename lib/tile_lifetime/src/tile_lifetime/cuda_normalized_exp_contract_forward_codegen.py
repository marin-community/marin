# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a bounded Contract/normalized-exp/indexed-selection CUDA forward."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from tile_lifetime.cuda_map_fold_codegen import CudaMapFoldProgram, CudaScalarFunction, render_cuda_map_fold_include
from tile_lifetime.tensor_program import (
    ScalarExpression,
    scalar_expression_inputs,
    scalar_input,
    serialize_scalar_expression,
)
from tile_lifetime.xla_normalized_exp_contract_forward import NormalizedExpContractForwardHloReplacementPlan

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\]")
_MAX_SHARED_SCORE_BYTES = 48 * 1024


@dataclass(frozen=True)
class GeneratedCudaNormalizedExpContractForwardFfi:
    """Generated source and semantic identity for one bounded forward family."""

    target: str
    handler_symbol: str
    source: str
    semantic_digest: str
    source_digest: str
    rows: int
    reduction: int
    fold_extent: int
    threads: int
    shared_bytes: int


def generate_cuda_normalized_exp_contract_forward_ffi(
    plan: NormalizedExpContractForwardHloReplacementPlan,
    *,
    target: str,
    score_expression: ScalarExpression | None = None,
    threads: int = 256,
) -> GeneratedCudaNormalizedExpContractForwardFfi:
    """Generate one CTA from compact Contract, Maps, Folds, and selection."""
    if threads not in {128, 256, 512}:
        raise ValueError("bounded normalized-exp forward requires 128, 256, or 512 threads")
    rows, reduction, fold_extent = _validated_extents(plan)
    shared_bytes = rows * fold_extent * 2
    if shared_bytes > _MAX_SHARED_SCORE_BYTES:
        raise ValueError(f"score tile requires {shared_bytes} shared bytes, above the bounded skeleton limit")
    expression = score_expression or scalar_input("raw_score")
    if scalar_expression_inputs(expression) != {"raw_score"}:
        raise ValueError("score Map must read exactly one raw_score scalar")
    scalar_source = render_cuda_map_fold_include(
        CudaMapFoldProgram((CudaScalarFunction("generated_score_map", ("raw_score",), expression),))
    )
    semantic_record = {
        "recovered_region": plan.region.semantic_digest,
        "rows": rows,
        "reduction": reduction,
        "fold_extent": fold_extent,
        "score_expression": serialize_scalar_expression(expression),
        "score_contract_boundary": "bf16_rne",
        "fold_order": "source_ordered_fp32",
        "selection": "indexed_row_valid",
    }
    semantic_digest = hashlib.sha256(
        json.dumps(semantic_record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    handler_symbol = _target_symbol(target)
    source = f"""// Generated from generic Contract/Map/Fold forward semantics; do not edit.
#include <atomic>
#include <cmath>
#include <cstdint>
#include <string>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

{scalar_source}

namespace {{
constexpr int kRows = {rows};
constexpr int kReduction = {reduction};
constexpr int kFoldExtent = {fold_extent};
constexpr int kThreads = {threads};
std::atomic<int> call_count{{0}};

__global__ void ShuttleNormalizedExpContractForwardKernel(
    const __nv_bfloat16* __restrict__ lhs,
    const __nv_bfloat16* __restrict__ rhs,
    const std::uint8_t* __restrict__ fold_validity,
    const std::int32_t* __restrict__ selected_indices,
    float* __restrict__ output,
    float* __restrict__ saved_state) {{
  extern __shared__ __nv_bfloat16 raw_scores[];
  for (int linear = threadIdx.x; linear < kRows * kFoldExtent; linear += blockDim.x) {{
    const int row = linear / kFoldExtent;
    const int fold = linear - row * kFoldExtent;
    float accumulator = 0.0f;
    for (int reduction = 0; reduction < kReduction; ++reduction) {{
      accumulator = __fadd_rn(
          accumulator,
          __fmul_rn(
              __bfloat162float(lhs[row * kReduction + reduction]),
              __bfloat162float(rhs[reduction * kFoldExtent + fold])));
    }}
    raw_scores[linear] = __float2bfloat16_rn(accumulator);
  }}
  __syncthreads();

  if (threadIdx.x < kRows) {{
    const int row = threadIdx.x;
    const int selected = selected_indices[row];
    const bool row_valid = selected >= 0 && selected < kFoldExtent && fold_validity[selected] != 0;
    float maximum = -INFINITY;
    for (int fold = 0; fold < kFoldExtent; ++fold) {{
      if (fold_validity[fold] != 0) {{
        maximum = fmaxf(
            maximum,
            generated_score_map(__bfloat162float(raw_scores[row * kFoldExtent + fold])));
      }}
    }}
    float sum_exp = 0.0f;
    for (int fold = 0; fold < kFoldExtent; ++fold) {{
      if (fold_validity[fold] != 0) {{
        const float mapped = generated_score_map(__bfloat162float(raw_scores[row * kFoldExtent + fold]));
        sum_exp = __fadd_rn(sum_exp, expf(__fsub_rn(mapped, maximum)));
      }}
    }}
    if (!row_valid || !isfinite(maximum) || sum_exp == 0.0f) {{
      output[row] = 0.0f;
      saved_state[row] = 0.0f;
      return;
    }}
    const float log_normalizer = __fadd_rn(logf(sum_exp), maximum);
    const float selected_score =
        generated_score_map(__bfloat162float(raw_scores[row * kFoldExtent + selected]));
    output[row] = __fsub_rn(log_normalizer, selected_score);
    saved_state[row] = log_normalizer;
  }}
}}

ffi::Error ShuttleNormalizedExpContractForward(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> lhs_buffer,
    ffi::Buffer<ffi::BF16, 2> rhs_buffer,
    ffi::Buffer<ffi::PRED, 1> fold_validity_buffer,
    ffi::Buffer<ffi::S32, 1> selected_indices_buffer,
    ffi::Result<ffi::Buffer<ffi::F32, 1>> output_buffer,
    ffi::Result<ffi::Buffer<ffi::F32, 1>> saved_state_buffer) {{
  ShuttleNormalizedExpContractForwardKernel<<<1, kThreads, {shared_bytes}, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(lhs_buffer.typed_data()),
      reinterpret_cast<const __nv_bfloat16*>(rhs_buffer.typed_data()),
      reinterpret_cast<const std::uint8_t*>(fold_validity_buffer.typed_data()),
      selected_indices_buffer.typed_data(),
      output_buffer->typed_data(),
      saved_state_buffer->typed_data());
  const cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {{
    return ffi::Error::Internal(
        "normalized-exp Contract forward launch failed: " + std::string(cudaGetErrorString(status)));
  }}
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}}

auto ShuttleNormalizedExpContractForwardBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::PRED, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::F32, 1>>()
      .Ret<ffi::Buffer<ffi::F32, 1>>();
}}
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {handler_symbol},
    ShuttleNormalizedExpContractForward,
    ShuttleNormalizedExpContractForwardBinding());

extern "C" int shuttle_normalized_exp_contract_forward_call_count() {{
  return call_count.load(std::memory_order_relaxed);
}}
"""
    return GeneratedCudaNormalizedExpContractForwardFfi(
        target,
        handler_symbol,
        source,
        semantic_digest,
        hashlib.sha256(source.encode()).hexdigest(),
        rows,
        reduction,
        fold_extent,
        threads,
        shared_bytes,
    )


def _validated_extents(plan: NormalizedExpContractForwardHloReplacementPlan) -> tuple[int, int, int]:
    lhs = _shape(plan.region.compact_score_contract.lhs.shape, "bf16", 2)
    rhs = _shape(plan.region.compact_score_contract.rhs.shape, "bf16", 2)
    rows, reduction = lhs
    rhs_reduction, fold_extent = rhs
    if reduction != rhs_reduction:
        raise ValueError("compact score Contract reduction extents disagree")
    if plan.region.compact_score_contract.dimensions.lhs_contracting != (1,):
        raise ValueError("bounded score Contract requires the trailing lhs reduction axis")
    if plan.region.compact_score_contract.dimensions.rhs_contracting != (0,):
        raise ValueError("bounded score Contract requires the leading rhs reduction axis")
    if plan.region.output.shape != f"f32[{rows}]{{0}}" or plan.region.saved_state.shape != f"f32[{rows}]{{0}}":
        raise ValueError("compact forward outputs do not match its row domain")
    return rows, reduction, fold_extent


def _shape(shape: str, dtype: str, rank: int) -> tuple[int, ...]:
    match = _ARRAY_SHAPE.match(shape)
    if match is None or match.group("dtype") != dtype:
        raise ValueError(f"expected {dtype} physical array shape, found {shape!r}")
    dimensions = tuple(int(value) for value in match.group("dims").split(",") if value)
    if len(dimensions) != rank:
        raise ValueError(f"expected rank {rank}, found {shape!r}")
    return dimensions


def _target_symbol(target: str) -> str:
    symbol = target.replace(".", "_").replace("-", "_")
    if not symbol.isidentifier():
        raise ValueError(f"typed-FFI target {target!r} cannot form a C++ symbol")
    return symbol
