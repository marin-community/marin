# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate multi-CTA CUDA FFI backends from anonymous Contract/Map algebra."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from tile_lifetime.contract_map_backend import ContractMapBackendProgram, ContractMapNumericalPolicy
from tile_lifetime.cuda_map_fold_codegen import (
    CudaArithmeticMode,
    CudaMapFoldProgram,
    CudaScalarFunction,
    render_cuda_scalar_program_include,
)
from tile_lifetime.ffi_command_buffer import (
    DirectLaunchFfiPhysicalCandidate,
    audit_ffi_command_buffer_eligibility,
    direct_launch_status_check,
    finalize_ffi_handler_source,
)
from tile_lifetime.tensor_program import MapPrimitive, ScalarExpression, ScalarExpressionKind, scalar_input


@dataclass(frozen=True)
class ContractMapBackendBuffer:
    """One dense row-major BF16 buffer at the typed-FFI boundary."""

    role: str
    shape: tuple[int, int]
    minor_to_major: tuple[int, int] = (1, 0)

    def __post_init__(self) -> None:
        if min(self.shape) <= 0:
            raise ValueError("Contract/Map backend buffers require positive rank-two shapes")
        if self.minor_to_major not in {(1, 0), (0, 1)}:
            raise ValueError("Contract/Map backend buffer layout must be a rank-two permutation")


@dataclass(frozen=True)
class ContractMapBackendPhysicalAbi:
    """Forward, reverse, and explicit scratch buffers owned by XLA."""

    forward_inputs: tuple[ContractMapBackendBuffer, ...]
    forward_outputs: tuple[ContractMapBackendBuffer, ...]
    reverse_inputs: tuple[ContractMapBackendBuffer, ...]
    reverse_outputs: tuple[ContractMapBackendBuffer, ...]
    reverse_scratch_outputs: tuple[ContractMapBackendBuffer, ...]


@dataclass(frozen=True)
class GeneratedCudaContractMapBackendFfi:
    """Generated forward/reverse handlers and immutable physical metadata."""

    policy: ContractMapNumericalPolicy
    forward_target: str
    reverse_target: str
    forward_handler_symbol: str
    reverse_handler_symbol: str
    source: str
    semantic_fingerprint: str
    source_sha256: str
    threads: int
    physical_candidate: DirectLaunchFfiPhysicalCandidate
    physical_abi: ContractMapBackendPhysicalAbi
    kernel_names: tuple[str, ...]
    forward_launch_count: int
    reverse_launch_count: int
    dynamic_shared_bytes: int = 0

    @property
    def command_buffer_compatible(self) -> bool:
        return self.physical_candidate.command_buffer_compatible


@dataclass(frozen=True)
class ContractMapBackendSourceAudit:
    """Structured source facts used by offline backend acceptance tests."""

    kernel_names: tuple[str, ...]
    launch_count: int
    global_intermediates: bool
    whole_matrix_shared_memory: bool
    source_ordered_reductions: bool
    fixed_tree_reductions: bool
    device_atomics: bool
    dense_linear_indexing: bool
    command_buffer_eligible: bool
    forbidden_command_buffer_operations: tuple[str, ...]
    opaque_semantic_dependencies: tuple[str, ...]


def contract_map_backend_physical_abi(program: ContractMapBackendProgram) -> ContractMapBackendPhysicalAbi:
    """Return the exact dense buffers indexed by generated CUDA."""
    rows, reduction, features = program.rows, program.reduction, program.features
    activation = (rows, reduction)
    first_weight = (reduction, features)
    second_weight = (features, reduction)
    feature_value = (rows, features)

    def buffer(role: str, shape: tuple[int, int]) -> ContractMapBackendBuffer:
        return ContractMapBackendBuffer(role=role, shape=shape)

    forward_inputs = (
        buffer("activation", activation),
        buffer("first_weight", first_weight),
        buffer("second_weight", second_weight),
    )
    forward_outputs = (
        buffer("output", activation),
        buffer("preactivation", feature_value),
        buffer("hidden", feature_value),
    )
    reverse_inputs = (
        *forward_inputs,
        forward_outputs[1],
        forward_outputs[2],
        buffer("output_cotangent", activation),
    )
    reverse_outputs = (
        buffer("input_adjoint", activation),
        buffer("first_weight_adjoint", first_weight),
        buffer("second_weight_adjoint", second_weight),
    )
    return ContractMapBackendPhysicalAbi(
        forward_inputs=forward_inputs,
        forward_outputs=forward_outputs,
        reverse_inputs=reverse_inputs,
        reverse_outputs=reverse_outputs,
        reverse_scratch_outputs=(buffer("preactivation_adjoint_scratch", feature_value),),
    )


def generate_cuda_contract_map_backend_ffi(
    program: ContractMapBackendProgram,
    *,
    target_prefix: str = "shuttle.generic.contract_map",
    threads: int = 256,
    physical_candidate: DirectLaunchFfiPhysicalCandidate = DirectLaunchFfiPhysicalCandidate.LAUNCH_CHECKED,
) -> GeneratedCudaContractMapBackendFfi:
    """Emit one policy-specific multi-CTA forward and derived reverse."""
    if threads not in {128, 256, 512}:
        raise ValueError("Contract/Map backends require 128, 256, or 512 threads")
    if not target_prefix or any(character.isspace() for character in target_prefix):
        raise ValueError("typed-FFI target prefix must be nonempty and contain no whitespace")
    rows, reduction, features = program.rows, program.reduction, program.features
    suffix = program.semantic_fingerprint[:16]
    forward_target = f"{target_prefix}.{program.numerical_policy.value}.{suffix}.forward"
    reverse_target = f"{target_prefix}.{program.numerical_policy.value}.{suffix}.reverse"
    forward_symbol = _target_symbol(forward_target)
    reverse_symbol = _target_symbol(reverse_target)
    scalar_include = _scalar_include(program)
    kernels = (
        "ShuttleContractMapFirstForwardKernel",
        "ShuttleContractMapSecondForwardKernel",
        "ShuttleContractMapAdjointMapKernel",
        "ShuttleContractMapInputAdjointKernel",
        "ShuttleContractMapFirstWeightAdjointKernel",
        "ShuttleContractMapSecondWeightAdjointKernel",
    )
    policy_record = {
        "semantic_fingerprint": program.semantic_fingerprint,
        "policy": program.numerical_policy.value,
        "threads": threads,
        "physical_family": "multi_cta_global_intermediate_contract_map",
        "kernel_names": kernels,
    }
    backend_fingerprint = hashlib.sha256(
        json.dumps(policy_record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    forward_feature_grid = _grid_expression("kRows * kFeatures", program.numerical_policy)
    forward_output_grid = _grid_expression("kRows * kReduction", program.numerical_policy)
    reverse_feature_grid = _grid_expression("kRows * kFeatures", program.numerical_policy)
    reverse_input_grid = _grid_expression("kRows * kReduction", program.numerical_policy)
    first_weight_grid = _grid_expression("kReduction * kFeatures", program.numerical_policy)
    second_weight_grid = _grid_expression("kFeatures * kReduction", program.numerical_policy)
    forward_status = direct_launch_status_check(physical_candidate, operation="Contract/Map forward")
    reverse_status = direct_launch_status_check(physical_candidate, operation="Contract/Map reverse")
    source_template = f"""// Generated from anonymous Contract/Map algebra; do not edit.
#include <atomic>
#include <cstdint>
#include <string>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "xla/ffi/api/ffi.h"

{scalar_include}

namespace ffi = xla::ffi;

namespace {{
constexpr int kRows = {rows};
constexpr int kReduction = {reduction};
constexpr int kFeatures = {features};
constexpr int kThreads = {threads};
constexpr int kWarpsPerBlock = kThreads / 32;
std::atomic<std::uint64_t> forward_call_count{{0}};
std::atomic<std::uint64_t> reverse_call_count{{0}};

{_first_forward_kernel(program.numerical_policy)}

{_second_forward_kernel(program.numerical_policy)}

{_adjoint_map_kernel(program.numerical_policy)}

{_input_adjoint_kernel(program.numerical_policy)}

{_first_weight_adjoint_kernel(program.numerical_policy)}

{_second_weight_adjoint_kernel(program.numerical_policy)}

ffi::Error ShuttleContractMapForward(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> activation_buffer,
    ffi::Buffer<ffi::BF16, 2> first_weight_buffer,
    ffi::Buffer<ffi::BF16, 2> second_weight_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> output_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> preactivation_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> hidden_buffer) {{
  const __nv_bfloat16* activation = reinterpret_cast<const __nv_bfloat16*>(activation_buffer.typed_data());
  const __nv_bfloat16* first_weight = reinterpret_cast<const __nv_bfloat16*>(first_weight_buffer.typed_data());
  const __nv_bfloat16* second_weight = reinterpret_cast<const __nv_bfloat16*>(second_weight_buffer.typed_data());
  __nv_bfloat16* output = reinterpret_cast<__nv_bfloat16*>(output_buffer->typed_data());
  __nv_bfloat16* preactivation = reinterpret_cast<__nv_bfloat16*>(preactivation_buffer->typed_data());
  __nv_bfloat16* hidden = reinterpret_cast<__nv_bfloat16*>(hidden_buffer->typed_data());
  ShuttleContractMapFirstForwardKernel<<<{forward_feature_grid}, kThreads, 0, stream>>>(
      activation, first_weight, preactivation, hidden);
  ShuttleContractMapSecondForwardKernel<<<{forward_output_grid}, kThreads, 0, stream>>>(
      hidden, second_weight, output);
{forward_status}
  forward_call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}}

ffi::Error ShuttleContractMapReverse(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> activation_buffer,
    ffi::Buffer<ffi::BF16, 2> first_weight_buffer,
    ffi::Buffer<ffi::BF16, 2> second_weight_buffer,
    ffi::Buffer<ffi::BF16, 2> preactivation_buffer,
    ffi::Buffer<ffi::BF16, 2> hidden_buffer,
    ffi::Buffer<ffi::BF16, 2> output_cotangent_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> input_adjoint_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> first_weight_adjoint_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> second_weight_adjoint_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> preactivation_adjoint_buffer) {{
  const __nv_bfloat16* activation = reinterpret_cast<const __nv_bfloat16*>(activation_buffer.typed_data());
  const __nv_bfloat16* first_weight = reinterpret_cast<const __nv_bfloat16*>(first_weight_buffer.typed_data());
  const __nv_bfloat16* second_weight = reinterpret_cast<const __nv_bfloat16*>(second_weight_buffer.typed_data());
  const __nv_bfloat16* preactivation = reinterpret_cast<const __nv_bfloat16*>(preactivation_buffer.typed_data());
  const __nv_bfloat16* hidden = reinterpret_cast<const __nv_bfloat16*>(hidden_buffer.typed_data());
  const __nv_bfloat16* output_cotangent = reinterpret_cast<const __nv_bfloat16*>(output_cotangent_buffer.typed_data());
  __nv_bfloat16* input_adjoint = reinterpret_cast<__nv_bfloat16*>(input_adjoint_buffer->typed_data());
  __nv_bfloat16* first_weight_adjoint = reinterpret_cast<__nv_bfloat16*>(first_weight_adjoint_buffer->typed_data());
  __nv_bfloat16* second_weight_adjoint = reinterpret_cast<__nv_bfloat16*>(second_weight_adjoint_buffer->typed_data());
  __nv_bfloat16* preactivation_adjoint = reinterpret_cast<__nv_bfloat16*>(preactivation_adjoint_buffer->typed_data());
  ShuttleContractMapAdjointMapKernel<<<{reverse_feature_grid}, kThreads, 0, stream>>>(
      preactivation, second_weight, output_cotangent, preactivation_adjoint);
  ShuttleContractMapInputAdjointKernel<<<{reverse_input_grid}, kThreads, 0, stream>>>(
      preactivation_adjoint, first_weight, input_adjoint);
  ShuttleContractMapFirstWeightAdjointKernel<<<{first_weight_grid}, kThreads, 0, stream>>>(
      activation, preactivation_adjoint, first_weight_adjoint);
  ShuttleContractMapSecondWeightAdjointKernel<<<{second_weight_grid}, kThreads, 0, stream>>>(
      hidden, output_cotangent, second_weight_adjoint);
{reverse_status}
  reverse_call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}}

auto ShuttleContractMapForwardBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>();
}}

auto ShuttleContractMapReverseBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>();
}}
}}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {forward_symbol}, ShuttleContractMapForward,
    ShuttleContractMapForwardBinding()__SHUTTLE_FFI_HANDLER_TRAITS__);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {reverse_symbol}, ShuttleContractMapReverse,
    ShuttleContractMapReverseBinding()__SHUTTLE_FFI_HANDLER_TRAITS__);

extern "C" std::uint64_t shuttle_contract_map_backend_forward_call_count() {{
  return forward_call_count.load(std::memory_order_relaxed);
}}

extern "C" std::uint64_t shuttle_contract_map_backend_reverse_call_count() {{
  return reverse_call_count.load(std::memory_order_relaxed);
}}

extern "C" const char* shuttle_contract_map_backend_fingerprint() {{
  return "{backend_fingerprint}";
}}
"""
    source = finalize_ffi_handler_source(
        source_template,
        command_buffer_compatible=physical_candidate.command_buffer_compatible,
        expected_handler_count=2,
    )
    return GeneratedCudaContractMapBackendFfi(
        policy=program.numerical_policy,
        forward_target=forward_target,
        reverse_target=reverse_target,
        forward_handler_symbol=forward_symbol,
        reverse_handler_symbol=reverse_symbol,
        source=source,
        semantic_fingerprint=program.semantic_fingerprint,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        threads=threads,
        physical_candidate=physical_candidate,
        physical_abi=contract_map_backend_physical_abi(program),
        kernel_names=kernels,
        forward_launch_count=2,
        reverse_launch_count=4,
    )


def audit_cuda_contract_map_backend_source(
    generated: GeneratedCudaContractMapBackendFfi,
) -> ContractMapBackendSourceAudit:
    """Audit the externally meaningful physical distinctions in generated source."""
    command_buffer = audit_ffi_command_buffer_eligibility(generated.source)
    opaque_tokens = ("flash_attention", "mok_", "gdn_", "cublas", "cudnn", "workload", "fixture")
    lowered = generated.source.lower()
    return ContractMapBackendSourceAudit(
        kernel_names=tuple(
            match.group(1) for match in re.finditer(r"__global__ void ([A-Za-z_][A-Za-z0-9_]*)", generated.source)
        ),
        launch_count=generated.source.count("<<<"),
        global_intermediates="preactivation_adjoint_buffer" in generated.source,
        whole_matrix_shared_memory="__shared__" in generated.source,
        source_ordered_reductions="for (int reduction = 0; reduction <" in generated.source,
        fixed_tree_reductions="__shfl_down_sync" in generated.source,
        device_atomics="atomicAdd(" in generated.source or "atomicCAS(" in generated.source,
        dense_linear_indexing=re.search(r"\[[^\]\n]*,[^\]\n]*\]", generated.source) is None,
        command_buffer_eligible=command_buffer.eligible,
        forbidden_command_buffer_operations=command_buffer.forbidden_operations,
        opaque_semantic_dependencies=tuple(token for token in opaque_tokens if token in lowered),
    )


def _scalar_include(program: ContractMapBackendProgram) -> str:
    scalar_map = program.source.operations[1]
    assert isinstance(scalar_map, MapPrimitive)
    source_name = scalar_map.inputs[0].name
    primal = _rename_expression(scalar_map.expression, {source_name: "z"})
    reverse_maps = tuple(
        operation
        for operation in program.differentiated.program.operations[len(program.source.operations) :]
        if isinstance(operation, MapPrimitive)
    )
    if len(reverse_maps) != 1 or len(reverse_maps[0].inputs) != 2:
        raise ValueError("mechanical reverse must contain one binary scalar Map")
    reverse = reverse_maps[0]
    reverse_expression = _rename_expression(
        reverse.expression,
        {reverse.inputs[0].name: "z", reverse.inputs[1].name: "cotangent"},
    )
    arithmetic = (
        CudaArithmeticMode.EXPLICIT_RN
        if program.numerical_policy is ContractMapNumericalPolicy.SOURCE_ORDERED
        else CudaArithmeticMode.CUDA_EXPRESSION
    )
    scalar_program = CudaMapFoldProgram(
        functions=(
            CudaScalarFunction("generated_phi", ("z",), primal, arithmetic),
            CudaScalarFunction("generated_phi_vjp", ("z", "cotangent"), reverse_expression, arithmetic),
        )
    )
    return render_cuda_scalar_program_include(
        scalar_program,
        fingerprint_macro="SHUTTLE_CONTRACT_MAP_SCALAR_SHA256",
        generated_by="tile_lifetime.cuda_contract_map_backend_codegen",
    )


def _first_forward_kernel(policy: ContractMapNumericalPolicy) -> str:
    return _contract_kernel(
        name="ShuttleContractMapFirstForwardKernel",
        parameters=(
            "const __nv_bfloat16* activation",
            "const __nv_bfloat16* first_weight",
            "__nv_bfloat16* preactivation",
            "__nv_bfloat16* hidden",
        ),
        output_count="kRows * kFeatures",
        coordinates="const int row = linear / kFeatures;\n  const int feature = linear - row * kFeatures;",
        reduction_count="kReduction",
        product=(
            "__bfloat162float(activation[row * kReduction + reduction])",
            "__bfloat162float(first_weight[reduction * kFeatures + feature])",
        ),
        store=(
            "const __nv_bfloat16 z = __float2bfloat16_rn(accumulator);",
            "preactivation[linear] = z;",
            "hidden[linear] = __float2bfloat16_rn(generated_phi(__bfloat162float(z)));",
        ),
        policy=policy,
    )


def _second_forward_kernel(policy: ContractMapNumericalPolicy) -> str:
    return _contract_kernel(
        name="ShuttleContractMapSecondForwardKernel",
        parameters=("const __nv_bfloat16* hidden", "const __nv_bfloat16* second_weight", "__nv_bfloat16* output"),
        output_count="kRows * kReduction",
        coordinates="const int row = linear / kReduction;\n  const int column = linear - row * kReduction;",
        reduction_count="kFeatures",
        product=(
            "__bfloat162float(hidden[row * kFeatures + reduction])",
            "__bfloat162float(second_weight[reduction * kReduction + column])",
        ),
        store=("output[linear] = __float2bfloat16_rn(accumulator);",),
        policy=policy,
    )


def _adjoint_map_kernel(policy: ContractMapNumericalPolicy) -> str:
    return _contract_kernel(
        name="ShuttleContractMapAdjointMapKernel",
        parameters=(
            "const __nv_bfloat16* preactivation",
            "const __nv_bfloat16* second_weight",
            "const __nv_bfloat16* output_cotangent",
            "__nv_bfloat16* preactivation_adjoint",
        ),
        output_count="kRows * kFeatures",
        coordinates="const int row = linear / kFeatures;\n  const int feature = linear - row * kFeatures;",
        reduction_count="kReduction",
        product=(
            "__bfloat162float(output_cotangent[row * kReduction + reduction])",
            "__bfloat162float(second_weight[feature * kReduction + reduction])",
        ),
        store=(
            "const float z = __bfloat162float(preactivation[linear]);",
            "preactivation_adjoint[linear] = __float2bfloat16_rn(generated_phi_vjp(z, accumulator));",
        ),
        policy=policy,
    )


def _input_adjoint_kernel(policy: ContractMapNumericalPolicy) -> str:
    return _contract_kernel(
        name="ShuttleContractMapInputAdjointKernel",
        parameters=(
            "const __nv_bfloat16* preactivation_adjoint",
            "const __nv_bfloat16* first_weight",
            "__nv_bfloat16* input_adjoint",
        ),
        output_count="kRows * kReduction",
        coordinates="const int row = linear / kReduction;\n  const int column = linear - row * kReduction;",
        reduction_count="kFeatures",
        product=(
            "__bfloat162float(preactivation_adjoint[row * kFeatures + reduction])",
            "__bfloat162float(first_weight[column * kFeatures + reduction])",
        ),
        store=("input_adjoint[linear] = __float2bfloat16_rn(accumulator);",),
        policy=policy,
    )


def _first_weight_adjoint_kernel(policy: ContractMapNumericalPolicy) -> str:
    return _contract_kernel(
        name="ShuttleContractMapFirstWeightAdjointKernel",
        parameters=(
            "const __nv_bfloat16* activation",
            "const __nv_bfloat16* preactivation_adjoint",
            "__nv_bfloat16* first_weight_adjoint",
        ),
        output_count="kReduction * kFeatures",
        coordinates=(
            "const int input_feature = linear / kFeatures;\n  const int feature = linear - input_feature * kFeatures;"
        ),
        reduction_count="kRows",
        product=(
            "__bfloat162float(activation[reduction * kReduction + input_feature])",
            "__bfloat162float(preactivation_adjoint[reduction * kFeatures + feature])",
        ),
        store=("first_weight_adjoint[linear] = __float2bfloat16_rn(accumulator);",),
        policy=policy,
    )


def _second_weight_adjoint_kernel(policy: ContractMapNumericalPolicy) -> str:
    return _contract_kernel(
        name="ShuttleContractMapSecondWeightAdjointKernel",
        parameters=(
            "const __nv_bfloat16* hidden",
            "const __nv_bfloat16* output_cotangent",
            "__nv_bfloat16* second_weight_adjoint",
        ),
        output_count="kFeatures * kReduction",
        coordinates=(
            "const int feature = linear / kReduction;\n  const int output_feature = linear - feature * kReduction;"
        ),
        reduction_count="kRows",
        product=(
            "__bfloat162float(hidden[reduction * kFeatures + feature])",
            "__bfloat162float(output_cotangent[reduction * kReduction + output_feature])",
        ),
        store=("second_weight_adjoint[linear] = __float2bfloat16_rn(accumulator);",),
        policy=policy,
    )


def _contract_kernel(
    *,
    name: str,
    parameters: tuple[str, ...],
    output_count: str,
    coordinates: str,
    reduction_count: str,
    product: tuple[str, str],
    store: tuple[str, ...],
    policy: ContractMapNumericalPolicy,
) -> str:
    parameter_list = ",\n    ".join(parameters)
    stores = "\n  ".join(store)
    if policy is ContractMapNumericalPolicy.SOURCE_ORDERED:
        return f"""extern "C" __global__ void {name}(
    {parameter_list}) {{
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= {output_count}) return;
  {coordinates}
  float accumulator = 0.0f;
  for (int reduction = 0; reduction < {reduction_count}; ++reduction) {{
    const float product = __fmul_rn({product[0]}, {product[1]});
    accumulator = __fadd_rn(accumulator, product);
  }}
  {stores}
}}"""
    return f"""extern "C" __global__ void {name}(
    {parameter_list}) {{
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int linear = blockIdx.x * kWarpsPerBlock + warp;
  if (linear >= {output_count}) return;
  {coordinates}
  float accumulator = 0.0f;
  for (int reduction = lane; reduction < {reduction_count}; reduction += 32) {{
    accumulator += {product[0]} * {product[1]};
  }}
  accumulator += __shfl_down_sync(0xffffffffu, accumulator, 16);
  accumulator += __shfl_down_sync(0xffffffffu, accumulator, 8);
  accumulator += __shfl_down_sync(0xffffffffu, accumulator, 4);
  accumulator += __shfl_down_sync(0xffffffffu, accumulator, 2);
  accumulator += __shfl_down_sync(0xffffffffu, accumulator, 1);
  if (lane != 0) return;
  {stores}
}}"""


def _grid_expression(output_count: str, policy: ContractMapNumericalPolicy) -> str:
    if policy is ContractMapNumericalPolicy.SOURCE_ORDERED:
        return f"({output_count} + kThreads - 1) / kThreads"
    return f"({output_count} + kWarpsPerBlock - 1) / kWarpsPerBlock"


def _rename_expression(expression: ScalarExpression, names: dict[str, str]) -> ScalarExpression:
    if expression.kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        try:
            return scalar_input(names[expression.input_name])
        except KeyError as error:
            raise ValueError(f"scalar expression references unavailable input {expression.input_name!r}") from error
    return ScalarExpression(
        kind=expression.kind,
        operands=tuple(_rename_expression(operand, names) for operand in expression.operands),
        constant=expression.constant,
    )


def _target_symbol(target: str) -> str:
    symbol = re.sub(r"[^A-Za-z0-9_]", "_", target)
    if not symbol or symbol[0].isdigit():
        raise ValueError(f"typed-FFI target cannot form a C++ symbol: {target!r}")
    return symbol
