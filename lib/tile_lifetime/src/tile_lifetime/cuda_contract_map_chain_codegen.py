# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate bounded CUDA for generic two-Contract scalar-Map training chains."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum

from tile_lifetime.cast_scalar_program import generate_cuda_scalar_body
from tile_lifetime.contract_map_chain import (
    BoundCastScalarMap,
    ContractMapChainPhysicalAbi,
    TwoContractMapTrainingProgram,
    contract_map_chain_physical_abi,
)
from tile_lifetime.ffi_command_buffer import (
    audit_ffi_command_buffer_eligibility,
    finalize_ffi_handler_source,
)

_MAX_SHARED_BYTES = 48 * 1024


class ContractMapChainFfiPhysicalCandidate(StrEnum):
    """Host-dispatch policy for one generated Contract/Map physical family."""

    LAUNCH_CHECKED = "launch_checked"
    COMMAND_BUFFER_CAPTURE_SAFE = "command_buffer_capture_safe"

    @property
    def command_buffer_compatible(self) -> bool:
        """Whether the handler can be replayed from an XLA command buffer."""
        return self is ContractMapChainFfiPhysicalCandidate.COMMAND_BUFFER_CAPTURE_SAFE


@dataclass(frozen=True)
class GeneratedCudaContractMapChainFfi:
    """Two generated one-CTA handlers and their semantic provenance."""

    forward_target: str
    reverse_target: str
    forward_handler_symbol: str
    reverse_handler_symbol: str
    source: str
    semantic_digest: str
    source_digest: str
    physical_candidate: ContractMapChainFfiPhysicalCandidate
    rows: int
    input_features: int
    rank: int
    threads: int
    forward_shared_bytes: int
    reverse_shared_bytes: int
    first_weight_adjoint_minor_to_major: tuple[int, int]
    second_weight_adjoint_minor_to_major: tuple[int, int]
    physical_abi: ContractMapChainPhysicalAbi
    kernel_count: int = 2
    external_dependencies: tuple[str, ...] = ("CUDA BF16/runtime primitives", "XLA typed FFI")

    @property
    def command_buffer_compatible(self) -> bool:
        """Whether both generated handlers declare capture-safe replay."""
        return self.physical_candidate.command_buffer_compatible


@dataclass(frozen=True)
class ContractMapChainSourceAudit:
    """Machine-readable clean-boundary evidence for generated source."""

    kernel_count: int
    has_explicit_bf16_contract_boundaries: bool
    has_generated_forward_maps: bool
    has_generated_reverse_maps: bool
    has_handler_counters: bool
    has_command_buffer_traits: bool
    has_launch_status_query: bool
    command_buffer_eligible: bool
    forbidden_command_buffer_operations: tuple[str, ...]
    has_atomics: bool
    opaque_semantic_dependencies: tuple[str, ...]


def generate_cuda_contract_map_chain_ffi(
    program: TwoContractMapTrainingProgram,
    *,
    forward_target: str,
    reverse_target: str,
    threads: int = 256,
    physical_candidate: ContractMapChainFfiPhysicalCandidate = ContractMapChainFfiPhysicalCandidate.LAUNCH_CHECKED,
) -> GeneratedCudaContractMapChainFfi:
    """Generate source-ordered forward and JAX-owned reverse physical bodies."""
    if threads not in {128, 256, 512}:
        raise ValueError("bounded Contract/Map chains require 128, 256, or 512 threads")
    first = program.first_contract
    physical_abi = contract_map_chain_physical_abi(program)
    rows = first.rows
    input_features = first.reduction
    rank = first.features
    forward_shared_bytes = 2 * rows * rank * 2
    reverse_shared_bytes = (rows * input_features + rows * rank) * 2
    if max(forward_shared_bytes, reverse_shared_bytes) > _MAX_SHARED_BYTES:
        raise ValueError("Contract/Map chain exceeds the bounded one-CTA shared-memory limit")

    hidden_source, hidden_call = _scalar_map_source_and_call(program.hidden_map, symbol="generated_hidden_map")
    output_source, output_call = _scalar_map_source_and_call(program.output_map, symbol="generated_output_map")
    second_vjp_source, second_vjp_call = _scalar_map_source_and_call(
        program.second_output_vjp_map,
        symbol="generated_second_output_vjp_map",
    )
    hidden_vjp_source, hidden_vjp_call = _scalar_map_source_and_call(
        program.hidden_vjp_map,
        symbol="generated_hidden_vjp_map",
    )
    input_vjp_source, input_vjp_call = _scalar_map_source_and_call(
        program.input_vjp_map,
        symbol="generated_input_vjp_map",
    )
    semantic_record = {
        "first_contract": first.__dict__,
        "second_contract": program.second_contract.__dict__,
        "maps": {
            "hidden": program.hidden_map.program.serialized,
            "output": program.output_map.program.serialized,
            "second_output_vjp": program.second_output_vjp_map.program.serialized,
            "hidden_vjp": program.hidden_vjp_map.program.serialized,
            "input_vjp": program.input_vjp_map.program.serialized,
        },
        "map_bindings": {
            "hidden": tuple(program.hidden_map.inputs),
            "output": tuple(program.output_map.inputs),
            "second_output_vjp": tuple(program.second_output_vjp_map.inputs),
            "hidden_vjp": tuple(program.hidden_vjp_map.inputs),
            "input_vjp": tuple(program.input_vjp_map.inputs),
        },
        "numerical_policy": program.numerical_policy,
        "weight_adjoint_layouts": {
            "first": program.first_weight_adjoint_minor_to_major,
            "second": program.second_weight_adjoint_minor_to_major,
        },
        "physical_family": "bounded_one_cta_two_contract_map_training",
        "threads": threads,
    }
    semantic_digest = hashlib.sha256(
        json.dumps(semantic_record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    first_weight_adjoint_dimension_zero_minor = json.dumps(physical_abi.reverse_outputs[1].minor_to_major == (0, 1))
    second_weight_adjoint_dimension_zero_minor = json.dumps(physical_abi.reverse_outputs[2].minor_to_major == (0, 1))
    forward_symbol = _target_symbol(forward_target)
    reverse_symbol = _target_symbol(reverse_target)
    forward_launch_status_check = _launch_status_check(physical_candidate, operation="forward")
    reverse_launch_status_check = _launch_status_check(physical_candidate, operation="reverse")
    source_template = f"""// Generated from generic Contract/Map forward and JAX-owned reverse semantics; do not edit.
#include <cstdint>
#include <string>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "xla/ffi/api/ffi.h"

{hidden_source}
{output_source}
{second_vjp_source}
{hidden_vjp_source}
{input_vjp_source}

namespace ffi = xla::ffi;

namespace {{
constexpr int kRows = {rows};
constexpr int kInputFeatures = {input_features};
constexpr int kRank = {rank};
constexpr int kThreads = {threads};
constexpr bool kFirstWeightAdjointDimensionZeroMinor = {first_weight_adjoint_dimension_zero_minor};
constexpr bool kSecondWeightAdjointDimensionZeroMinor = {second_weight_adjoint_dimension_zero_minor};
std::uint64_t forward_call_count = 0;
std::uint64_t reverse_call_count = 0;

__global__ void ShuttleContractMapChainForwardKernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ first_weight,
    const __nv_bfloat16* __restrict__ second_weight,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ saved_first_contract,
    __nv_bfloat16* __restrict__ saved_hidden,
    __nv_bfloat16* __restrict__ saved_second_contract) {{
  __shared__ __nv_bfloat16 first_contract[kRows * kRank];
  __shared__ __nv_bfloat16 hidden[kRows * kRank];

  for (int linear = threadIdx.x; linear < kRows * kRank; linear += blockDim.x) {{
    const int row = linear / kRank;
    const int feature = linear - row * kRank;
    float accumulator = 0.0f;
    for (int reduction = 0; reduction < kInputFeatures; ++reduction) {{
      accumulator = __fadd_rn(
          accumulator,
          __fmul_rn(
              __bfloat162float(input[row * kInputFeatures + reduction]),
              __bfloat162float(first_weight[reduction * kRank + feature])));
    }}
    const __nv_bfloat16 contract_value = __float2bfloat16_rn(accumulator);
    first_contract[linear] = contract_value;
    saved_first_contract[linear] = contract_value;
    const float first_contract_output_value = __bfloat162float(contract_value);
    const __nv_bfloat16 hidden_value = __float2bfloat16_rn({hidden_call});
    hidden[linear] = hidden_value;
    saved_hidden[linear] = hidden_value;
  }}
  __syncthreads();

  for (int linear = threadIdx.x; linear < kRows * kInputFeatures; linear += blockDim.x) {{
    const int row = linear / kInputFeatures;
    const int feature = linear - row * kInputFeatures;
    float accumulator = 0.0f;
    for (int reduction = 0; reduction < kRank; ++reduction) {{
      accumulator = __fadd_rn(
          accumulator,
          __fmul_rn(
              __bfloat162float(hidden[row * kRank + reduction]),
              __bfloat162float(second_weight[reduction * kInputFeatures + feature])));
    }}
    const __nv_bfloat16 contract_value = __float2bfloat16_rn(accumulator);
    saved_second_contract[linear] = contract_value;
    const float input_value = __bfloat162float(input[linear]);
    const float second_contract_output_value = __bfloat162float(contract_value);
    output[linear] = __float2bfloat16_rn({output_call});
  }}
}}

__global__ void ShuttleContractMapChainReverseKernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ first_weight,
    const __nv_bfloat16* __restrict__ second_weight,
    const __nv_bfloat16* __restrict__ saved_first_contract,
    const __nv_bfloat16* __restrict__ saved_hidden,
    const __nv_bfloat16* __restrict__ saved_second_contract,
    const __nv_bfloat16* __restrict__ output_cotangent,
    __nv_bfloat16* __restrict__ input_adjoint,
    __nv_bfloat16* __restrict__ first_weight_adjoint,
    __nv_bfloat16* __restrict__ second_weight_adjoint) {{
  __shared__ __nv_bfloat16 second_output_adjoint[kRows * kInputFeatures];
  __shared__ __nv_bfloat16 rank_adjoint[kRows * kRank];

  for (int linear = threadIdx.x; linear < kRows * kInputFeatures; linear += blockDim.x) {{
    const float input_value = __bfloat162float(input[linear]);
    const float second_contract_output_value = __bfloat162float(saved_second_contract[linear]);
    const float output_cotangent_value = __bfloat162float(output_cotangent[linear]);
    second_output_adjoint[linear] = __float2bfloat16_rn({second_vjp_call});
  }}
  __syncthreads();

  for (int linear = threadIdx.x; linear < kRows * kRank; linear += blockDim.x) {{
    const int row = linear / kRank;
    const int feature = linear - row * kRank;
    float accumulator = 0.0f;
    for (int reduction = 0; reduction < kInputFeatures; ++reduction) {{
      accumulator = __fadd_rn(
          accumulator,
          __fmul_rn(
              __bfloat162float(second_output_adjoint[row * kInputFeatures + reduction]),
              __bfloat162float(second_weight[feature * kInputFeatures + reduction])));
    }}
    rank_adjoint[linear] = __float2bfloat16_rn(accumulator);
  }}
  __syncthreads();

  for (int linear = threadIdx.x; linear < kRows * kRank; linear += blockDim.x) {{
    const float second_contract_input_adjoint_value = __bfloat162float(rank_adjoint[linear]);
    const float first_contract_output_value = __bfloat162float(saved_first_contract[linear]);
    rank_adjoint[linear] = __float2bfloat16_rn({hidden_vjp_call});
  }}
  __syncthreads();

  for (int linear = threadIdx.x; linear < kRows * kInputFeatures; linear += blockDim.x) {{
    const int row = linear / kInputFeatures;
    const int feature = linear - row * kInputFeatures;
    float accumulator = 0.0f;
    for (int reduction = 0; reduction < kRank; ++reduction) {{
      accumulator = __fadd_rn(
          accumulator,
          __fmul_rn(
              __bfloat162float(rank_adjoint[row * kRank + reduction]),
              __bfloat162float(first_weight[feature * kRank + reduction])));
    }}
    const float first_contract_input_adjoint_value = __bfloat162float(__float2bfloat16_rn(accumulator));
    const float second_contract_output_value = __bfloat162float(saved_second_contract[linear]);
    const float output_cotangent_value = __bfloat162float(output_cotangent[linear]);
    input_adjoint[linear] = __float2bfloat16_rn({input_vjp_call});
  }}

  for (int linear = threadIdx.x; linear < kInputFeatures * kRank; linear += blockDim.x) {{
    const int input_feature = linear / kRank;
    const int rank_feature = linear - input_feature * kRank;
    float accumulator = 0.0f;
    for (int row = 0; row < kRows; ++row) {{
      accumulator = __fadd_rn(
          accumulator,
          __fmul_rn(
              __bfloat162float(input[row * kInputFeatures + input_feature]),
              __bfloat162float(rank_adjoint[row * kRank + rank_feature])));
    }}
    const int output_offset = kFirstWeightAdjointDimensionZeroMinor
        ? rank_feature * kInputFeatures + input_feature
        : input_feature * kRank + rank_feature;
    first_weight_adjoint[output_offset] = __float2bfloat16_rn(accumulator);
  }}

  for (int linear = threadIdx.x; linear < kRank * kInputFeatures; linear += blockDim.x) {{
    const int rank_feature = linear / kInputFeatures;
    const int input_feature = linear - rank_feature * kInputFeatures;
    float accumulator = 0.0f;
    for (int row = 0; row < kRows; ++row) {{
      accumulator = __fadd_rn(
          accumulator,
          __fmul_rn(
              __bfloat162float(saved_hidden[row * kRank + rank_feature]),
              __bfloat162float(second_output_adjoint[row * kInputFeatures + input_feature])));
    }}
    const int output_offset = kSecondWeightAdjointDimensionZeroMinor
        ? input_feature * kRank + rank_feature
        : rank_feature * kInputFeatures + input_feature;
    second_weight_adjoint[output_offset] = __float2bfloat16_rn(accumulator);
  }}
}}

ffi::Error ShuttleContractMapChainForward(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> input_buffer,
    ffi::Buffer<ffi::BF16, 2> first_weight_buffer,
    ffi::Buffer<ffi::BF16, 2> second_weight_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> output_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> first_contract_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> hidden_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> second_contract_buffer) {{
  ShuttleContractMapChainForwardKernel<<<1, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input_buffer.typed_data()),
      reinterpret_cast<const __nv_bfloat16*>(first_weight_buffer.typed_data()),
      reinterpret_cast<const __nv_bfloat16*>(second_weight_buffer.typed_data()),
      reinterpret_cast<__nv_bfloat16*>(output_buffer->typed_data()),
      reinterpret_cast<__nv_bfloat16*>(first_contract_buffer->typed_data()),
      reinterpret_cast<__nv_bfloat16*>(hidden_buffer->typed_data()),
      reinterpret_cast<__nv_bfloat16*>(second_contract_buffer->typed_data()));
{forward_launch_status_check}
  ++forward_call_count;
  return ffi::Error::Success();
}}

ffi::Error ShuttleContractMapChainReverse(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> input_buffer,
    ffi::Buffer<ffi::BF16, 2> first_weight_buffer,
    ffi::Buffer<ffi::BF16, 2> second_weight_buffer,
    ffi::Buffer<ffi::BF16, 2> first_contract_buffer,
    ffi::Buffer<ffi::BF16, 2> hidden_buffer,
    ffi::Buffer<ffi::BF16, 2> second_contract_buffer,
    ffi::Buffer<ffi::BF16, 2> output_cotangent_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> input_adjoint_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> first_weight_adjoint_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> second_weight_adjoint_buffer) {{
  ShuttleContractMapChainReverseKernel<<<1, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input_buffer.typed_data()),
      reinterpret_cast<const __nv_bfloat16*>(first_weight_buffer.typed_data()),
      reinterpret_cast<const __nv_bfloat16*>(second_weight_buffer.typed_data()),
      reinterpret_cast<const __nv_bfloat16*>(first_contract_buffer.typed_data()),
      reinterpret_cast<const __nv_bfloat16*>(hidden_buffer.typed_data()),
      reinterpret_cast<const __nv_bfloat16*>(second_contract_buffer.typed_data()),
      reinterpret_cast<const __nv_bfloat16*>(output_cotangent_buffer.typed_data()),
      reinterpret_cast<__nv_bfloat16*>(input_adjoint_buffer->typed_data()),
      reinterpret_cast<__nv_bfloat16*>(first_weight_adjoint_buffer->typed_data()),
      reinterpret_cast<__nv_bfloat16*>(second_weight_adjoint_buffer->typed_data()));
{reverse_launch_status_check}
  ++reverse_call_count;
  return ffi::Error::Success();
}}

auto ShuttleContractMapChainForwardBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>();
}}

auto ShuttleContractMapChainReverseBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>();
}}
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {forward_symbol},
    ShuttleContractMapChainForward,
    ShuttleContractMapChainForwardBinding()__SHUTTLE_FFI_HANDLER_TRAITS__);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {reverse_symbol},
    ShuttleContractMapChainReverse,
    ShuttleContractMapChainReverseBinding()__SHUTTLE_FFI_HANDLER_TRAITS__);

extern "C" std::uint64_t shuttle_contract_map_chain_forward_call_count() {{
  return forward_call_count;
}}

extern "C" std::uint64_t shuttle_contract_map_chain_reverse_call_count() {{
  return reverse_call_count;
}}
"""
    source = finalize_ffi_handler_source(
        source_template,
        command_buffer_compatible=physical_candidate.command_buffer_compatible,
        expected_handler_count=2,
    )
    return GeneratedCudaContractMapChainFfi(
        forward_target=forward_target,
        reverse_target=reverse_target,
        forward_handler_symbol=forward_symbol,
        reverse_handler_symbol=reverse_symbol,
        source=source,
        semantic_digest=semantic_digest,
        source_digest=hashlib.sha256(source.encode()).hexdigest(),
        physical_candidate=physical_candidate,
        rows=rows,
        input_features=input_features,
        rank=rank,
        threads=threads,
        forward_shared_bytes=forward_shared_bytes,
        reverse_shared_bytes=reverse_shared_bytes,
        first_weight_adjoint_minor_to_major=program.first_weight_adjoint_minor_to_major,
        second_weight_adjoint_minor_to_major=program.second_weight_adjoint_minor_to_major,
        physical_abi=physical_abi,
    )


def audit_cuda_contract_map_chain_source(
    generated: GeneratedCudaContractMapChainFfi,
) -> ContractMapChainSourceAudit:
    """Report physical ownership without relying on workload names."""
    lowered = generated.source.lower()
    command_buffer_audit = audit_ffi_command_buffer_eligibility(generated.source)
    opaque_tokens = (
        "flash_attention",
        "mok_forward",
        "gdn_chunk",
        "sparse_attention_forward",
        "deep_ep_moe_combine",
        "cublas",
    )
    return ContractMapChainSourceAudit(
        kernel_count=generated.source.count("__global__ void "),
        has_explicit_bf16_contract_boundaries=generated.source.count("__float2bfloat16_rn(accumulator)") >= 6,
        has_generated_forward_maps="generated_hidden_map" in generated.source
        and "generated_output_map" in generated.source,
        has_generated_reverse_maps=all(
            name in generated.source
            for name in ("generated_second_output_vjp_map", "generated_hidden_vjp_map", "generated_input_vjp_map")
        ),
        has_handler_counters=(
            "shuttle_contract_map_chain_forward_call_count" in generated.source
            and "shuttle_contract_map_chain_reverse_call_count" in generated.source
        ),
        has_command_buffer_traits=(
            generated.source.count("{ffi::Traits::kCmdBufferCompatible}") == generated.kernel_count
        ),
        has_launch_status_query=("cudaPeekAtLastError(" in generated.source or "cudaGetLastError(" in generated.source),
        command_buffer_eligible=command_buffer_audit.eligible,
        forbidden_command_buffer_operations=command_buffer_audit.forbidden_operations,
        has_atomics="atomic" in lowered,
        opaque_semantic_dependencies=tuple(token for token in opaque_tokens if token in lowered),
    )


def _launch_status_check(physical_candidate: ContractMapChainFfiPhysicalCandidate, *, operation: str) -> str:
    if physical_candidate is ContractMapChainFfiPhysicalCandidate.COMMAND_BUFFER_CAPTURE_SAFE:
        return ""
    if physical_candidate is ContractMapChainFfiPhysicalCandidate.LAUNCH_CHECKED:
        return f"""  const cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {{
    return ffi::Error::Internal(\"Contract/Map {operation} launch failed: \" + std::string(cudaGetErrorString(status)));
  }}"""
    raise ValueError(f"unsupported Contract/Map FFI physical candidate: {physical_candidate}")


def _scalar_map_source_and_call(scalar_map: BoundCastScalarMap, *, symbol: str) -> tuple[str, str]:
    generated = generate_cuda_scalar_body(scalar_map.program, symbol=symbol)
    role_names = tuple(f"{role.value}_value" for role in scalar_map.inputs)
    return generated.source, f"{symbol}({', '.join(role_names)})"


def _target_symbol(target: str) -> str:
    symbol = re.sub(r"[^A-Za-z0-9_]", "_", target)
    if not symbol or symbol[0].isdigit():
        raise ValueError(f"typed-FFI target cannot form a C++ symbol: {target!r}")
    return symbol
