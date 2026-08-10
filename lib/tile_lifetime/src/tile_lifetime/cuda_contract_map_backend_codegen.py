# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate multi-CTA CUDA FFI backends from anonymous Contract/Map algebra."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from tile_lifetime.contract_map_backend import (
    ContractMapBackendProgram,
    ContractMapNumericalPolicy,
    ContractMapReverseLoweringPlan,
    contract_map_reverse_lowering_plan,
)
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
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    MapPrimitive,
    ProgramValue,
    ScalarExpression,
    ScalarExpressionKind,
    TensorAxis,
    scalar_input,
)

CONTRACT_MAP_INT32_MAX = 2_147_483_647
CONTRACT_MAP_BF16_BYTES = 2
CONTRACT_MAP_GRID_X_MAX = 2_147_483_647


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
    forward_call_count_symbol: str
    reverse_call_count_symbol: str
    backend_fingerprint_symbol: str
    source: str
    semantic_fingerprint: str
    physical_digest: str
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


@dataclass(frozen=True)
class ContractMapBufferSize:
    """Checked flattened size for one ABI buffer occurrence."""

    boundary: str
    role: str
    elements: int
    bytes: int


@dataclass(frozen=True)
class ContractMapLaunchSize:
    """Checked work and grid dimensions for one generated kernel."""

    kernel_name: str
    work_items: int
    work_items_per_block: int
    grid_numerator: int
    block_count: int


@dataclass(frozen=True)
class ContractMapBackendSizeAudit:
    """All int32 ABI and grid bounds checked before source generation."""

    buffers: tuple[ContractMapBufferSize, ...]
    launches: tuple[ContractMapLaunchSize, ...]


@dataclass(frozen=True)
class _RenderedReverseKernel:
    name: str
    source: str
    launch_arguments: tuple[str, ...]
    output_axes: tuple[TensorAxis, ...]


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
    if type(physical_candidate) is not DirectLaunchFfiPhysicalCandidate:
        raise TypeError("physical_candidate must be a DirectLaunchFfiPhysicalCandidate")
    if not target_prefix or any(character.isspace() for character in target_prefix):
        raise ValueError("typed-FFI target prefix must be nonempty and contain no whitespace")
    rows, reduction, features = program.rows, program.reduction, program.features
    reverse_plan = contract_map_reverse_lowering_plan(program)
    physical_abi = contract_map_backend_physical_abi(program)
    reverse_kernels = _render_reverse_kernels(program, reverse_plan)
    kernel_names = (
        "ShuttleContractMapFirstForwardKernel",
        "ShuttleContractMapSecondForwardKernel",
        *(kernel.name for kernel in reverse_kernels),
    )
    _validate_contract_map_backend_sizes(
        program,
        threads=threads,
        physical_abi=physical_abi,
        reverse_kernels=reverse_kernels,
    )
    physical_digest = _physical_codegen_digest(
        program,
        reverse_plan,
        physical_abi=physical_abi,
        kernel_names=kernel_names,
        threads=threads,
        physical_candidate=physical_candidate,
        target_prefix=target_prefix,
    )
    suffix = physical_digest
    forward_target = f"{target_prefix}.{program.numerical_policy.value}.{suffix}.forward"
    reverse_target = f"{target_prefix}.{program.numerical_policy.value}.{suffix}.reverse"
    forward_symbol = _target_symbol(forward_target)
    reverse_symbol = _target_symbol(reverse_target)
    forward_call_count_symbol = f"shuttle_contract_map_backend_forward_call_count_{suffix}"
    reverse_call_count_symbol = f"shuttle_contract_map_backend_reverse_call_count_{suffix}"
    backend_fingerprint_symbol = f"shuttle_contract_map_backend_fingerprint_{suffix}"
    scalar_include = _scalar_include(program, reverse_plan)
    forward_feature_grid = _grid_expression("kRows * kFeatures", program.numerical_policy)
    forward_output_grid = _grid_expression("kRows * kReduction", program.numerical_policy)
    forward_status = direct_launch_status_check(physical_candidate, operation="Contract/Map forward")
    reverse_status = direct_launch_status_check(physical_candidate, operation="Contract/Map reverse")
    reverse_kernel_source = "\n\n".join(kernel.source for kernel in reverse_kernels)
    reverse_launch_source = "\n".join(
        _render_kernel_launch(kernel, program, policy=program.numerical_policy) for kernel in reverse_kernels
    )
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

{reverse_kernel_source}

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
{reverse_launch_source}
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

extern "C" std::uint64_t {forward_call_count_symbol}() {{
  return forward_call_count.load(std::memory_order_relaxed);
}}

extern "C" std::uint64_t {reverse_call_count_symbol}() {{
  return reverse_call_count.load(std::memory_order_relaxed);
}}

extern "C" const char* {backend_fingerprint_symbol}() {{
  return "{physical_digest}";
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
        forward_call_count_symbol=forward_call_count_symbol,
        reverse_call_count_symbol=reverse_call_count_symbol,
        backend_fingerprint_symbol=backend_fingerprint_symbol,
        source=source,
        semantic_fingerprint=program.semantic_fingerprint,
        physical_digest=physical_digest,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        threads=threads,
        physical_candidate=physical_candidate,
        physical_abi=physical_abi,
        kernel_names=kernel_names,
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


def _scalar_include(
    program: ContractMapBackendProgram,
    reverse_plan: ContractMapReverseLoweringPlan,
) -> str:
    scalar_map = program.source.operations[1]
    assert isinstance(scalar_map, MapPrimitive)
    source_name = scalar_map.inputs[0].name
    primal = _rename_expression(scalar_map.expression, {source_name: "z"})
    reverse = reverse_plan.pointwise_adjoint
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


def _render_reverse_kernels(
    program: ContractMapBackendProgram,
    plan: ContractMapReverseLoweringPlan,
) -> tuple[_RenderedReverseKernel, ...]:
    roles = _reverse_value_roles(program, plan)
    return (
        _render_reverse_contract_kernel(
            "ShuttleContractMapAdjointMapKernel",
            plan.hidden_adjoint_contract,
            program,
            roles,
            fused_map=plan.pointwise_adjoint,
        ),
        _render_reverse_contract_kernel(
            "ShuttleContractMapSecondWeightAdjointKernel",
            plan.second_weight_adjoint_contract,
            program,
            roles,
        ),
        _render_reverse_contract_kernel(
            "ShuttleContractMapInputAdjointKernel",
            plan.input_adjoint_contract,
            program,
            roles,
        ),
        _render_reverse_contract_kernel(
            "ShuttleContractMapFirstWeightAdjointKernel",
            plan.first_weight_adjoint_contract,
            program,
            roles,
        ),
    )


def _render_reverse_contract_kernel(
    name: str,
    operation: ContractPrimitive,
    program: ContractMapBackendProgram,
    roles: dict[ProgramValue, str],
    *,
    fused_map: MapPrimitive | None = None,
) -> _RenderedReverseKernel:
    if len(operation.inputs) != 2 or len(operation.reduction_axes) != 1 or len(operation.output.axes) != 2:
        raise ValueError("reverse Contract lowering requires two operands, one reduction axis, and rank-two output")
    output = fused_map.output if fused_map is not None else operation.output
    if output.axes != operation.output.axes:
        raise ValueError("hidden-adjoint Map fusion requires identical Contract and Map output axes")
    external_map_inputs = (
        () if fused_map is None else tuple(value for value in fused_map.inputs if value != operation.output)
    )
    input_values = tuple(dict.fromkeys((*external_map_inputs, *operation.inputs)))
    output_role = _required_role(roles, output)
    parameters = (
        *(f"const __nv_bfloat16* {_required_role(roles, value)}" for value in input_values),
        f"__nv_bfloat16* {output_role}",
    )
    reduction_axis = operation.reduction_axes[0]
    indices = tuple(_flattened_index(value, output.axes, reduction_axis, program) for value in operation.inputs)
    product = tuple(
        f"__bfloat162float({_required_role(roles, value)}[{index}])"
        for value, index in zip(operation.inputs, indices, strict=True)
    )
    if fused_map is None:
        store = (f"{output_role}[linear] = __float2bfloat16_rn(accumulator);",)
    else:
        if external_map_inputs != (program.preactivation,):
            raise ValueError("hidden-adjoint fusion requires preactivation as the only external Map input")
        preactivation_role = _required_role(roles, external_map_inputs[0])
        store = (
            f"const float z = __bfloat162float({preactivation_role}[linear]);",
            f"{output_role}[linear] = __float2bfloat16_rn(generated_phi_vjp(z, accumulator));",
        )
    source = _contract_kernel(
        name=name,
        parameters=parameters,
        output_count=_axes_product_expression(output.axes, program),
        coordinates=_output_coordinates(output.axes, program),
        reduction_count=_axis_extent_expression(reduction_axis, program),
        product=(product[0], product[1]),
        store=store,
        policy=program.numerical_policy,
    )
    return _RenderedReverseKernel(
        name=name,
        source=source,
        launch_arguments=(*(_required_role(roles, value) for value in input_values), output_role),
        output_axes=output.axes,
    )


def _reverse_value_roles(
    program: ContractMapBackendProgram,
    plan: ContractMapReverseLoweringPlan,
) -> dict[ProgramValue, str]:
    return {
        program.activation: "activation",
        program.first_weight: "first_weight",
        program.second_weight: "second_weight",
        program.preactivation: "preactivation",
        program.hidden: "hidden",
        program.output_cotangent: "output_cotangent",
        plan.hidden_adjoint_contract.output: "hidden_adjoint",
        plan.pointwise_adjoint.output: "preactivation_adjoint",
        plan.input_adjoint_contract.output: "input_adjoint",
        plan.first_weight_adjoint_contract.output: "first_weight_adjoint",
        plan.second_weight_adjoint_contract.output: "second_weight_adjoint",
    }


def _required_role(roles: dict[ProgramValue, str], value: ProgramValue) -> str:
    try:
        return roles[value]
    except KeyError as error:
        raise ValueError(f"reverse lowering has no physical role for value {value.name!r}") from error


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


def contract_map_backend_size_audit(
    program: ContractMapBackendProgram,
    *,
    threads: int = 256,
) -> ContractMapBackendSizeAudit:
    """Fail closed when the current flattened int32 BF16 ABI cannot address a buffer or launch."""
    reverse_plan = contract_map_reverse_lowering_plan(program)
    return _validate_contract_map_backend_sizes(
        program,
        threads=threads,
        physical_abi=contract_map_backend_physical_abi(program),
        reverse_kernels=_render_reverse_kernels(program, reverse_plan),
    )


def _validate_contract_map_backend_sizes(
    program: ContractMapBackendProgram,
    *,
    threads: int,
    physical_abi: ContractMapBackendPhysicalAbi,
    reverse_kernels: tuple[_RenderedReverseKernel, ...],
) -> ContractMapBackendSizeAudit:
    if threads not in {128, 256, 512}:
        raise ValueError("Contract/Map backends require 128, 256, or 512 threads")
    buffers: list[ContractMapBufferSize] = []
    for boundary, group in (
        ("forward_input", physical_abi.forward_inputs),
        ("forward_output", physical_abi.forward_outputs),
        ("reverse_input", physical_abi.reverse_inputs),
        ("reverse_output", physical_abi.reverse_outputs),
        ("reverse_scratch_output", physical_abi.reverse_scratch_outputs),
    ):
        for buffer in group:
            elements = _checked_int32_product(buffer.shape, context=f"{boundary}.{buffer.role} flattened elements")
            if elements > CONTRACT_MAP_INT32_MAX // CONTRACT_MAP_BF16_BYTES:
                raise ValueError(
                    f"{boundary}.{buffer.role} byte size exceeds the signed-int32 ABI bound {CONTRACT_MAP_INT32_MAX}"
                )
            buffers.append(
                ContractMapBufferSize(
                    boundary=boundary,
                    role=buffer.role,
                    elements=elements,
                    bytes=elements * CONTRACT_MAP_BF16_BYTES,
                )
            )
    launch_axes = (
        ("ShuttleContractMapFirstForwardKernel", program.preactivation.axes),
        ("ShuttleContractMapSecondForwardKernel", program.output.axes),
        *((kernel.name, kernel.output_axes) for kernel in reverse_kernels),
    )
    work_items_per_block = (
        threads if program.numerical_policy is ContractMapNumericalPolicy.SOURCE_ORDERED else threads // 32
    )
    launches: list[ContractMapLaunchSize] = []
    for kernel_name, axes in launch_axes:
        work_items = _checked_int32_product(
            tuple(axis.extent for axis in axes),
            context=f"{kernel_name} flattened work items",
        )
        increment = work_items_per_block - 1
        if work_items > CONTRACT_MAP_INT32_MAX - increment:
            raise ValueError(f"{kernel_name} grid numerator exceeds the signed-int32 ABI bound")
        grid_numerator = work_items + increment
        block_count = grid_numerator // work_items_per_block
        if block_count > CONTRACT_MAP_GRID_X_MAX:
            raise ValueError(f"{kernel_name} block count exceeds the device grid.x bound")
        launches.append(
            ContractMapLaunchSize(
                kernel_name=kernel_name,
                work_items=work_items,
                work_items_per_block=work_items_per_block,
                grid_numerator=grid_numerator,
                block_count=block_count,
            )
        )
    return ContractMapBackendSizeAudit(buffers=tuple(buffers), launches=tuple(launches))


def _checked_int32_product(dimensions: tuple[int, ...], *, context: str) -> int:
    product = 1
    for dimension in dimensions:
        if type(dimension) is not int or dimension <= 0:
            raise ValueError(f"{context} requires positive integer dimensions")
        if product > CONTRACT_MAP_INT32_MAX // dimension:
            raise ValueError(f"{context} exceeds the signed-int32 ABI bound {CONTRACT_MAP_INT32_MAX}")
        product *= dimension
    return product


def _render_kernel_launch(
    kernel: _RenderedReverseKernel,
    program: ContractMapBackendProgram,
    *,
    policy: ContractMapNumericalPolicy,
) -> str:
    output_count = _axes_product_expression(kernel.output_axes, program)
    grid = _grid_expression(output_count, policy)
    arguments = ", ".join(kernel.launch_arguments)
    return f"  {kernel.name}<<<{grid}, kThreads, 0, stream>>>(\n      {arguments});"


def _output_coordinates(axes: tuple[TensorAxis, ...], program: ContractMapBackendProgram) -> str:
    if len(axes) != 2:
        raise ValueError("Contract CUDA lowering requires rank-two output axes")
    inner_extent = _axis_extent_expression(axes[1], program)
    return (
        f"const int coordinate_0 = linear / {inner_extent};\n"
        f"  const int coordinate_1 = linear - coordinate_0 * {inner_extent};"
    )


def _flattened_index(
    value: ProgramValue,
    output_axes: tuple[TensorAxis, ...],
    reduction_axis: TensorAxis,
    program: ContractMapBackendProgram,
) -> str:
    if len(value.axes) != 2 or len(output_axes) != 2:
        raise ValueError("Contract CUDA lowering requires rank-two values")
    indices: list[str] = []
    for axis in value.axes:
        if axis == reduction_axis:
            indices.append("reduction")
            continue
        try:
            indices.append(f"coordinate_{output_axes.index(axis)}")
        except ValueError as error:
            raise ValueError(f"Contract operand {value.name!r} has an axis outside output and reduction axes") from error
    inner_extent = _axis_extent_expression(value.axes[1], program)
    return f"{indices[0]} * {inner_extent} + {indices[1]}"


def _axes_product_expression(axes: tuple[TensorAxis, ...], program: ContractMapBackendProgram) -> str:
    if len(axes) != 2:
        raise ValueError("Contract CUDA lowering requires rank-two output axes")
    return " * ".join(_axis_extent_expression(axis, program) for axis in axes)


def _axis_extent_expression(axis: TensorAxis, program: ContractMapBackendProgram) -> str:
    row_axis, reduction_axis = program.activation.axes
    feature_axis = program.first_weight.axes[1]
    if axis == row_axis:
        return "kRows"
    if axis == reduction_axis:
        return "kReduction"
    if axis == feature_axis:
        return "kFeatures"
    raise ValueError("Contract CUDA lowering encountered an unanchored logical axis")


def _grid_expression(output_count: str, policy: ContractMapNumericalPolicy) -> str:
    if policy is ContractMapNumericalPolicy.SOURCE_ORDERED:
        return f"({output_count} + kThreads - 1) / kThreads"
    return f"({output_count} + kWarpsPerBlock - 1) / kWarpsPerBlock"


def _physical_codegen_digest(
    program: ContractMapBackendProgram,
    reverse_plan: ContractMapReverseLoweringPlan,
    *,
    physical_abi: ContractMapBackendPhysicalAbi,
    kernel_names: tuple[str, ...],
    threads: int,
    physical_candidate: DirectLaunchFfiPhysicalCandidate,
    target_prefix: str,
) -> str:
    roles = _reverse_value_roles(program, reverse_plan)
    axis_roles = {
        program.activation.axes[0]: "row",
        program.activation.axes[1]: "reduction",
        program.first_weight.axes[1]: "feature",
    }

    def axes(value: ProgramValue) -> tuple[str, ...]:
        try:
            return tuple(axis_roles[axis] for axis in value.axes)
        except KeyError as error:
            raise ValueError(f"physical digest encountered an unanchored axis on {value.name!r}") from error

    operations: list[dict[str, object]] = []
    for operation in reverse_plan.operations:
        if isinstance(operation, ContractPrimitive):
            operations.append(
                {
                    "kind": "contract",
                    "inputs": tuple(_required_role(roles, value) for value in operation.inputs),
                    "output": _required_role(roles, operation.output),
                    "output_axes": axes(operation.output),
                    "reduction_axes": tuple(axis_roles[axis] for axis in operation.reduction_axes),
                    "accumulation_dtype": operation.accumulation_dtype.value,
                }
            )
            continue
        operations.append(
            {
                "kind": "map",
                "inputs": tuple(_required_role(roles, value) for value in operation.inputs),
                "output": _required_role(roles, operation.output),
                "output_axes": axes(operation.output),
                "expression": _expression_digest_record(operation.expression, roles),
            }
        )
    record = {
        "semantic_digest": program.semantic_fingerprint,
        "policy": program.numerical_policy.value,
        "threads": threads,
        "target_prefix": target_prefix,
        "candidate": physical_candidate.value,
        "command_buffer_trait": physical_candidate.command_buffer_compatible,
        "abi": {
            "forward_inputs": _buffer_digest_records(physical_abi.forward_inputs),
            "forward_outputs": _buffer_digest_records(physical_abi.forward_outputs),
            "reverse_inputs": _buffer_digest_records(physical_abi.reverse_inputs),
            "reverse_outputs": _buffer_digest_records(physical_abi.reverse_outputs),
            "reverse_scratch_outputs": _buffer_digest_records(physical_abi.reverse_scratch_outputs),
        },
        "topology": {
            "family": "multi_cta_global_intermediate_contract_map",
            "kernel_names": kernel_names,
            "forward_launch_count": 2,
            "reverse_launch_count": 4,
            "source_map_expression": _expression_digest_record(program.scalar_expression, roles),
            "reverse_operations": operations,
            "fusion": ("hidden_adjoint_contract", "pointwise_adjoint"),
        },
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _buffer_digest_records(buffers: tuple[ContractMapBackendBuffer, ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "role": buffer.role,
            "shape": buffer.shape,
            "minor_to_major": buffer.minor_to_major,
        }
        for buffer in buffers
    )


def _expression_digest_record(
    expression: ScalarExpression,
    roles: dict[ProgramValue, str],
) -> dict[str, object]:
    names = {value.name: role for value, role in roles.items()}
    record: dict[str, object] = {"kind": expression.kind.value}
    if expression.input_name is not None:
        try:
            record["input"] = names[expression.input_name]
        except KeyError as error:
            raise ValueError(f"physical digest has no role for scalar input {expression.input_name!r}") from error
    if expression.constant is not None:
        record["constant"] = expression.constant
    if expression.operands:
        record["operands"] = tuple(_expression_digest_record(operand, roles) for operand in expression.operands)
    return record


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
