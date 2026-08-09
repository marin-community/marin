# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a bounded Contract/normalized-exp/reverse-Contract CUDA skeleton."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from tile_lifetime.autodiff import differentiate_scalar_expression
from tile_lifetime.cuda_map_fold_codegen import CudaMapFoldProgram, CudaScalarFunction, render_cuda_map_fold_include
from tile_lifetime.ffi_command_buffer import finalize_ffi_handler_source
from tile_lifetime.tensor_program import (
    ScalarExpression,
    scalar_expression_inputs,
    scalar_input,
    serialize_scalar_expression,
)
from tile_lifetime.xla_normalized_exp_contract_reverse import NormalizedExpContractReverseHloReplacementPlan

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\]")
_MAX_SHARED_SCORE_BYTES = 48 * 1024


@dataclass(frozen=True)
class GeneratedCudaNormalizedExpContractReverseFfi:
    """Generated source and semantic identity for one bounded reverse family."""

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
    command_buffer_compatible: bool


def generate_cuda_normalized_exp_contract_reverse_ffi(
    plan: NormalizedExpContractReverseHloReplacementPlan,
    *,
    target: str,
    score_expression: ScalarExpression | None = None,
    threads: int = 256,
    command_buffer_compatible: bool = False,
) -> GeneratedCudaNormalizedExpContractReverseFfi:
    """Generate one CTA from generic Contracts, Maps, and indexed Fold state."""
    if threads not in {128, 256, 512}:
        raise ValueError("bounded normalized-exp reverse requires 128, 256, or 512 threads")
    rows, reduction, fold_extent = _validated_extents(plan)
    shared_bytes = rows * fold_extent * 2
    if shared_bytes > _MAX_SHARED_SCORE_BYTES:
        raise ValueError(f"score-cotangent tile requires {shared_bytes} shared bytes, above the bounded skeleton limit")
    expression = score_expression or scalar_input("raw_score")
    if scalar_expression_inputs(expression) != {"raw_score"}:
        raise ValueError("score Map must read exactly one raw_score scalar")
    derivative = differentiate_scalar_expression(expression, "raw_score")
    derivative_arguments = ("raw_score",) if scalar_expression_inputs(derivative) else ()
    derivative_call = "generated_score_derivative(raw_score)" if derivative_arguments else "generated_score_derivative()"
    scalar_program = CudaMapFoldProgram(
        (
            CudaScalarFunction("generated_score_map", ("raw_score",), expression),
            CudaScalarFunction("generated_score_derivative", derivative_arguments, derivative),
        )
    )
    scalar_source = render_cuda_map_fold_include(scalar_program)
    semantic_record = {
        "recovered_region": plan.region.semantic_digest,
        "rows": rows,
        "reduction": reduction,
        "fold_extent": fold_extent,
        "score_expression": serialize_scalar_expression(expression),
        "score_derivative": serialize_scalar_expression(derivative),
        "score_contract_boundary": "bf16_rne",
        "score_cotangent_boundary": "bf16_rne",
        "accumulation": "ordered_fp32",
        "selection": "indexed_row_valid",
    }
    semantic_digest = hashlib.sha256(
        json.dumps(semantic_record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    handler_symbol = _target_symbol(target)
    launch_status_check = (
        ""
        if command_buffer_compatible
        else """
  const cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return ffi::Error::Internal(
        "normalized-exp Contract reverse launch failed: " + std::string(cudaGetErrorString(status)));
  }
"""
    )
    source_template = f"""// Generated from generic Contract/Map/Fold reverse semantics; do not edit.
#include <atomic>
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

__global__ void ShuttleNormalizedExpContractReverseKernel(
    const __nv_bfloat16* __restrict__ lhs,
    const __nv_bfloat16* __restrict__ rhs,
    const float* __restrict__ saved_state,
    const std::uint8_t* __restrict__ fold_validity,
    const float* __restrict__ row_cotangent,
    const std::int32_t* __restrict__ selected_indices,
    const std::uint8_t* __restrict__ row_validity,
    __nv_bfloat16* __restrict__ input_cotangent,
    __nv_bfloat16* __restrict__ operand_cotangent) {{
  extern __shared__ __nv_bfloat16 score_cotangent[];

  for (int linear = threadIdx.x; linear < kRows * kFoldExtent; linear += blockDim.x) {{
    const int row = linear / kFoldExtent;
    const int fold = linear - row * kFoldExtent;
    float score_accumulator = 0.0f;
    for (int reduction = 0; reduction < kReduction; ++reduction) {{
      score_accumulator = __fadd_rn(
          score_accumulator,
          __fmul_rn(
              __bfloat162float(lhs[row * kReduction + reduction]),
              __bfloat162float(rhs[reduction * kFoldExtent + fold])));
    }}
    const float raw_score = __bfloat162float(__float2bfloat16_rn(score_accumulator));
    float probability = 0.0f;
    if (fold_validity[fold] != 0) {{
      probability = expf(__fsub_rn(generated_score_map(raw_score), saved_state[row]));
    }}
    const float selected =
        row_validity[row] != 0 && selected_indices[row] == fold ? row_cotangent[row] : 0.0f;
    const float base_cotangent = __fsub_rn(__fmul_rn(probability, row_cotangent[row]), selected);
    const float mapped_cotangent = __fmul_rn(base_cotangent, {derivative_call});
    score_cotangent[linear] = __float2bfloat16_rn(mapped_cotangent);
  }}
  __syncthreads();

  for (int linear = threadIdx.x; linear < kRows * kReduction; linear += blockDim.x) {{
    const int row = linear / kReduction;
    const int reduction = linear - row * kReduction;
    float accumulator = 0.0f;
    for (int fold = 0; fold < kFoldExtent; ++fold) {{
      accumulator = __fadd_rn(
          accumulator,
          __fmul_rn(
              __bfloat162float(score_cotangent[row * kFoldExtent + fold]),
              __bfloat162float(rhs[reduction * kFoldExtent + fold])));
    }}
    input_cotangent[linear] = __float2bfloat16_rn(accumulator);
  }}

  for (int linear = threadIdx.x; linear < kReduction * kFoldExtent; linear += blockDim.x) {{
    const int reduction = linear / kFoldExtent;
    const int fold = linear - reduction * kFoldExtent;
    float accumulator = 0.0f;
    for (int row = 0; row < kRows; ++row) {{
      accumulator = __fadd_rn(
          accumulator,
          __fmul_rn(
              __bfloat162float(lhs[row * kReduction + reduction]),
              __bfloat162float(score_cotangent[row * kFoldExtent + fold])));
    }}
    operand_cotangent[linear] = __float2bfloat16_rn(accumulator);
  }}
}}

ffi::Error ShuttleNormalizedExpContractReverse(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> lhs_buffer,
    ffi::Buffer<ffi::BF16, 2> rhs_buffer,
    ffi::Buffer<ffi::F32, 1> saved_state_buffer,
    ffi::Buffer<ffi::PRED, 1> fold_validity_buffer,
    ffi::Buffer<ffi::F32, 1> row_cotangent_buffer,
    ffi::Buffer<ffi::S32, 1> selected_indices_buffer,
    ffi::Buffer<ffi::PRED, 1> row_validity_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> input_cotangent_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> operand_cotangent_buffer) {{
  ShuttleNormalizedExpContractReverseKernel<<<1, kThreads, {shared_bytes}, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(lhs_buffer.typed_data()),
      reinterpret_cast<const __nv_bfloat16*>(rhs_buffer.typed_data()),
      saved_state_buffer.typed_data(),
      reinterpret_cast<const std::uint8_t*>(fold_validity_buffer.typed_data()),
      row_cotangent_buffer.typed_data(),
      selected_indices_buffer.typed_data(),
      reinterpret_cast<const std::uint8_t*>(row_validity_buffer.typed_data()),
      reinterpret_cast<__nv_bfloat16*>(input_cotangent_buffer->typed_data()),
      reinterpret_cast<__nv_bfloat16*>(operand_cotangent_buffer->typed_data()));
{launch_status_check}
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}}

auto ShuttleNormalizedExpContractReverseBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::F32, 1>>()
      .Arg<ffi::Buffer<ffi::PRED, 1>>()
      .Arg<ffi::Buffer<ffi::F32, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Arg<ffi::Buffer<ffi::PRED, 1>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>();
}}
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {handler_symbol},
    ShuttleNormalizedExpContractReverse,
    ShuttleNormalizedExpContractReverseBinding()__SHUTTLE_FFI_HANDLER_TRAITS__);

extern "C" int shuttle_normalized_exp_contract_reverse_call_count() {{
  return call_count.load(std::memory_order_relaxed);
}}
"""
    source = finalize_ffi_handler_source(
        source_template,
        command_buffer_compatible=command_buffer_compatible,
    )
    return GeneratedCudaNormalizedExpContractReverseFfi(
        target=target,
        handler_symbol=handler_symbol,
        source=source,
        semantic_digest=semantic_digest,
        source_digest=hashlib.sha256(source.encode()).hexdigest(),
        rows=rows,
        reduction=reduction,
        fold_extent=fold_extent,
        threads=threads,
        shared_bytes=shared_bytes,
        command_buffer_compatible=command_buffer_compatible,
    )


def _validated_extents(plan: NormalizedExpContractReverseHloReplacementPlan) -> tuple[int, int, int]:
    lhs = _shape(plan.region.score_contract.lhs.shape, dtype="bf16", rank=2)
    rhs = _shape(plan.region.score_contract.rhs.shape, dtype="bf16", rank=2)
    score = _shape(plan.region.score_contract.output_shape, dtype="bf16", rank=2)
    input_reverse = _shape(plan.region.input_reverse_contract.output_shape, dtype="bf16", rank=2)
    operand_reverse = _shape(plan.region.operand_reverse_contract.output_shape, dtype="bf16", rank=2)
    rows, reduction = lhs
    rhs_reduction, fold_extent = rhs
    if rhs_reduction != reduction or score != (rows, fold_extent):
        raise ValueError("score Contract shapes do not form [row,reduction] @ [reduction,fold]")
    if input_reverse != lhs or operand_reverse != rhs:
        raise ValueError("reverse Contract outputs do not match the primal operand shapes")
    if plan.region.score_contract.dimensions.lhs_contracting != (1,):
        raise ValueError("bounded score Contract requires the trailing lhs reduction axis")
    if plan.region.score_contract.dimensions.rhs_contracting != (0,):
        raise ValueError("bounded score Contract requires the leading rhs reduction axis")
    return rows, reduction, fold_extent


def _shape(shape: str, *, dtype: str, rank: int) -> tuple[int, ...]:
    match = _ARRAY_SHAPE.match(shape)
    if match is None or match.group("dtype") != dtype:
        raise ValueError(f"expected {dtype} physical array shape, found {shape!r}")
    dimensions = tuple(int(value) for value in match.group("dims").split(",") if value)
    if len(dimensions) != rank:
        raise ValueError(f"expected rank {rank}, found shape {shape!r}")
    return dimensions


def _target_symbol(target: str) -> str:
    symbol = target.replace(".", "_").replace("-", "_")
    if not symbol.isidentifier():
        raise ValueError(f"typed-FFI target {target!r} cannot form a C++ symbol")
    return symbol
