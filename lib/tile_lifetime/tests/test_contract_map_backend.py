# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

import tile_lifetime.cuda_contract_map_backend_codegen as contract_map_codegen
from lib.tile_lifetime.benchmarks.h100_contract_map_backend_training import (
    generated_contract_map_candidates,
    natural_jax_training_step,
)
from tile_lifetime.contract_map_backend import (
    ContractMapBackendProgram,
    ContractMapNumericalPolicy,
    build_contract_map_backend_program,
    cubic_mix_expression,
    execute_contract_map_source_ordered_forward,
    execute_contract_map_source_ordered_reverse,
    form_contract_map_backend_program,
    sigmoid_product_expression,
    tanh_product_expression,
)
from tile_lifetime.contract_map_backend_resources import (
    contract_map_compile_plan,
    expected_contract_map_logical_boundary,
    parse_ptxas_kernel_resources,
)
from tile_lifetime.cuda_contract_map_backend_codegen import (
    CONTRACT_MAP_BF16_BYTES,
    CONTRACT_MAP_GRID_X_MAX,
    CONTRACT_MAP_INT32_MAX,
    audit_cuda_contract_map_backend_source,
    contract_map_backend_size_audit,
    generate_cuda_contract_map_backend_ffi,
)
from tile_lifetime.ffi_command_buffer import DirectLaunchFfiPhysicalCandidate
from tile_lifetime.h100_contract_map_benchmark import MeasurementBoundary, ScalarMapFamily
from tile_lifetime.jax_contract_map_backend_ffi import (
    call_cuda_contract_map_backend_forward_ffi,
    call_cuda_contract_map_backend_reverse_ffi,
)
from tile_lifetime.tensor_program import (
    AxisIndexMap,
    ContractPrimitive,
    MapPrimitive,
    ProgramValue,
    ScalarExpression,
    ScalarExpressionKind,
    TensorProgram,
    scalar_binary,
    scalar_input,
)

_MAP_CASES = (
    (ScalarMapFamily.SIGMOID_PRODUCT, sigmoid_product_expression),
    (ScalarMapFamily.TANH_PRODUCT, tanh_product_expression),
    (ScalarMapFamily.CUBIC_MIX, cubic_mix_expression),
)


@pytest.mark.parametrize(("family", "expression"), _MAP_CASES)
def test_contract_map_source_ordered_forward_and_reverse_match_jax_vjp(
    family: ScalarMapFamily,
    expression: Callable[[], ScalarExpression],
) -> None:
    program = build_contract_map_backend_program(
        rows=3,
        reduction=5,
        features=4,
        scalar_expression=expression(),
        numerical_policy=ContractMapNumericalPolicy.SOURCE_ORDERED,
    )
    rng = np.random.default_rng(20260810)
    activation = rng.normal(scale=0.15, size=(3, 5)).astype(np.float32)
    first_weight = rng.normal(scale=0.15, size=(5, 4)).astype(np.float32)
    second_weight = rng.normal(scale=0.15, size=(4, 5)).astype(np.float32)
    output_cotangent = rng.normal(scale=0.15, size=(3, 5)).astype(np.float32)
    forward = execute_contract_map_source_ordered_forward(program, activation, first_weight, second_weight)
    reverse = execute_contract_map_source_ordered_reverse(
        program,
        activation,
        first_weight,
        second_weight,
        forward,
        output_cotangent,
    )
    expected = natural_jax_training_step(
        family,
        jnp.asarray(activation, dtype=jnp.bfloat16),
        jnp.asarray(first_weight, dtype=jnp.bfloat16),
        jnp.asarray(second_weight, dtype=jnp.bfloat16),
        jnp.asarray(output_cotangent, dtype=jnp.bfloat16),
    )
    actual = (
        forward.output,
        reverse.input_adjoint,
        reverse.first_weight_adjoint,
        reverse.second_weight_adjoint,
    )
    errors = tuple(
        np.abs(np.asarray(actual_value, dtype=np.float32) - np.asarray(expected_value, dtype=np.float32))
        for actual_value, expected_value in zip(actual, expected, strict=True)
    )
    assert max(float(error.max(initial=0.0)) for error in errors) <= 0.0078125
    assert max(float(error.mean()) for error in errors) <= 0.0005


def test_contract_map_generator_covers_all_reviewed_shapes_with_distinct_policies() -> None:
    candidates = generated_contract_map_candidates()

    assert len(candidates) == 8
    grouped = {candidate.case.case_id: [] for candidate in candidates}
    for candidate in candidates:
        grouped[candidate.case.case_id].append(candidate)
        audit = audit_cuda_contract_map_backend_source(candidate.generated)
        assert candidate.case.rows % 2 == 1
        assert audit.kernel_names == candidate.generated.kernel_names
        assert audit.launch_count == 6
        assert audit.global_intermediates
        assert not audit.whole_matrix_shared_memory
        assert not audit.device_atomics
        assert audit.dense_linear_indexing
        assert not audit.opaque_semantic_dependencies
        assert candidate.generated.dynamic_shared_bytes == 0
        assert tuple(boundary for boundary, _ in candidate.boundaries) == tuple(MeasurementBoundary)
    for pair in grouped.values():
        assert len(pair) == 2
        source_ordered = next(
            candidate
            for candidate in pair
            if candidate.program.numerical_policy is ContractMapNumericalPolicy.SOURCE_ORDERED
        )
        fast = next(
            candidate for candidate in pair if candidate.program.numerical_policy is ContractMapNumericalPolicy.FAST
        )
        source_audit = audit_cuda_contract_map_backend_source(source_ordered.generated)
        fast_audit = audit_cuda_contract_map_backend_source(fast.generated)
        assert source_audit.source_ordered_reductions and not source_audit.fixed_tree_reductions
        assert fast_audit.fixed_tree_reductions and not fast_audit.source_ordered_reductions
        assert source_ordered.generated.source_sha256 != fast.generated.source_sha256
        assert source_ordered.program.semantic_fingerprint != fast.program.semantic_fingerprint


def test_contract_map_scalar_mutation_changes_forward_and_derived_reverse() -> None:
    baseline = build_contract_map_backend_program(
        rows=5,
        reduction=8,
        features=8,
        scalar_expression=sigmoid_product_expression(),
        numerical_policy=ContractMapNumericalPolicy.SOURCE_ORDERED,
    )
    mutated = build_contract_map_backend_program(
        rows=5,
        reduction=8,
        features=8,
        scalar_expression=tanh_product_expression(),
        numerical_policy=ContractMapNumericalPolicy.SOURCE_ORDERED,
    )
    baseline_source = generate_cuda_contract_map_backend_ffi(baseline)
    mutated_source = generate_cuda_contract_map_backend_ffi(mutated)

    assert baseline.semantic_fingerprint != mutated.semantic_fingerprint
    assert baseline_source.source_sha256 != mutated_source.source_sha256
    assert "expf(" in baseline_source.source
    assert "tanhf(" in mutated_source.source
    baseline_reverse = baseline_source.source.split("generated_phi_vjp", maxsplit=1)[1]
    mutated_reverse = mutated_source.source.split("generated_phi_vjp", maxsplit=1)[1]
    assert baseline_reverse != mutated_reverse


@pytest.mark.parametrize("reverse_index", (0, 1, 3, 4))
def test_contract_map_reverse_contract_operands_are_authoritative(reverse_index: int) -> None:
    baseline = build_contract_map_backend_program(
        rows=5,
        reduction=8,
        features=4,
        scalar_expression=tanh_product_expression(),
        numerical_policy=ContractMapNumericalPolicy.SOURCE_ORDERED,
    )
    reverse = list(baseline.differentiated.program.operations[len(baseline.source.operations) :])
    operation = reverse[reverse_index]
    assert isinstance(operation, ContractPrimitive)
    reverse[reverse_index] = replace(operation, inputs=tuple(reversed(operation.inputs)))
    mutated = _replace_reverse_operations(baseline, tuple(reverse))

    with pytest.raises(ValueError, match="adjoint Contract operands"):
        generate_cuda_contract_map_backend_ffi(mutated)


def test_contract_map_reverse_pointwise_expression_drives_code_and_physical_digest() -> None:
    baseline = build_contract_map_backend_program(
        rows=5,
        reduction=8,
        features=4,
        scalar_expression=tanh_product_expression(),
        numerical_policy=ContractMapNumericalPolicy.SOURCE_ORDERED,
    )
    reverse = list(baseline.differentiated.program.operations[len(baseline.source.operations) :])
    pointwise = reverse[2]
    assert isinstance(pointwise, MapPrimitive)
    mutated_expression = scalar_binary(
        ScalarExpressionKind.ADD,
        scalar_input(pointwise.inputs[0].name),
        scalar_input(pointwise.inputs[1].name),
    )
    reverse[2] = replace(pointwise, expression=mutated_expression)
    mutated = _replace_reverse_operations(baseline, tuple(reverse))

    baseline_generated = generate_cuda_contract_map_backend_ffi(baseline)
    mutated_generated = generate_cuda_contract_map_backend_ffi(mutated)
    assert baseline_generated.source_sha256 != mutated_generated.source_sha256
    assert baseline_generated.physical_digest != mutated_generated.physical_digest
    assert baseline_generated.forward_target != mutated_generated.forward_target


def test_contract_map_reverse_operation_removal_and_reordering_fail_closed() -> None:
    baseline = build_contract_map_backend_program(
        rows=5,
        reduction=8,
        features=4,
        scalar_expression=tanh_product_expression(),
        numerical_policy=ContractMapNumericalPolicy.FAST,
    )
    reverse = baseline.differentiated.program.operations[len(baseline.source.operations) :]
    reordered = _replace_reverse_operations(baseline, (reverse[1], reverse[0], *reverse[2:]))
    with pytest.raises(ValueError, match="hidden-adjoint Contract operands"):
        generate_cuda_contract_map_backend_ffi(reordered)

    shortened_program = replace(
        baseline.differentiated.program,
        operations=(*baseline.source.operations, reverse[0], *reverse[2:]),
        outputs=(baseline.input_adjoint, baseline.first_weight_adjoint),
    )
    shortened_differentiated = replace(
        baseline.differentiated,
        program=shortened_program,
        input_gradients=(baseline.input_adjoint, baseline.first_weight_adjoint),
    )
    shortened = replace(baseline, differentiated=shortened_differentiated)
    with pytest.raises(ValueError, match="gradients must preserve source-input order"):
        generate_cuda_contract_map_backend_ffi(shortened)


def test_contract_map_reverse_reduction_axes_are_authoritative() -> None:
    baseline = build_contract_map_backend_program(
        rows=5,
        reduction=8,
        features=4,
        scalar_expression=tanh_product_expression(),
        numerical_policy=ContractMapNumericalPolicy.FAST,
    )
    reverse = list(baseline.differentiated.program.operations[len(baseline.source.operations) :])
    hidden_adjoint = reverse[0]
    assert isinstance(hidden_adjoint, ContractPrimitive)
    reverse[0] = replace(hidden_adjoint, reduction_axes=())
    mutated = _replace_reverse_operations(baseline, tuple(reverse))

    with pytest.raises(ValueError, match="hidden-adjoint Contract has incompatible reduction axes"):
        generate_cuda_contract_map_backend_ffi(mutated)


def test_contract_map_reverse_output_axes_are_authoritative() -> None:
    baseline = build_contract_map_backend_program(
        rows=5,
        reduction=8,
        features=4,
        scalar_expression=tanh_product_expression(),
        numerical_policy=ContractMapNumericalPolicy.FAST,
    )
    reverse = list(baseline.differentiated.program.operations[len(baseline.source.operations) :])
    second_weight_adjoint = reverse[1]
    assert isinstance(second_weight_adjoint, ContractPrimitive)
    wrong_output = ProgramValue(
        second_weight_adjoint.output.name,
        tuple(reversed(second_weight_adjoint.output.axes)),
        second_weight_adjoint.output.dtype,
    )
    reverse[1] = replace(second_weight_adjoint, output=wrong_output)
    differentiated_program = replace(
        baseline.differentiated.program,
        operations=(*baseline.source.operations, *reverse),
        outputs=(baseline.input_adjoint, baseline.first_weight_adjoint, wrong_output),
    )
    differentiated = replace(
        baseline.differentiated,
        program=differentiated_program,
        input_gradients=(baseline.input_adjoint, baseline.first_weight_adjoint, wrong_output),
    )
    mutated = replace(
        baseline,
        differentiated=differentiated,
        second_weight_adjoint=wrong_output,
    )

    with pytest.raises(ValueError, match="second-weight-adjoint Contract has incompatible output axes"):
        generate_cuda_contract_map_backend_ffi(mutated)


def test_contract_map_capture_safe_handlers_preserve_multi_launch_topology() -> None:
    program = build_contract_map_backend_program(
        rows=43,
        reduction=104,
        features=72,
        scalar_expression=tanh_product_expression(),
        numerical_policy=ContractMapNumericalPolicy.FAST,
    )
    generated = generate_cuda_contract_map_backend_ffi(
        program,
        physical_candidate=DirectLaunchFfiPhysicalCandidate.COMMAND_BUFFER_CAPTURE_SAFE,
    )
    audit = audit_cuda_contract_map_backend_source(generated)

    assert generated.command_buffer_compatible
    assert audit.command_buffer_eligible
    assert not audit.forbidden_command_buffer_operations
    assert audit.launch_count == generated.forward_launch_count + generated.reverse_launch_count


@pytest.mark.parametrize("policy", tuple(ContractMapNumericalPolicy))
def test_contract_map_int32_abi_accepts_byte_size_boundary(policy: ContractMapNumericalPolicy) -> None:
    maximum_bf16_elements = CONTRACT_MAP_INT32_MAX // CONTRACT_MAP_BF16_BYTES
    program = build_contract_map_backend_program(
        rows=1,
        reduction=1,
        features=maximum_bf16_elements,
        scalar_expression=tanh_product_expression(),
        numerical_policy=policy,
    )

    generated = generate_cuda_contract_map_backend_ffi(program)
    size_audit = contract_map_backend_size_audit(program)
    assert max(buffer.bytes for buffer in size_audit.buffers) == maximum_bf16_elements * CONTRACT_MAP_BF16_BYTES
    assert len(size_audit.buffers) == 16
    assert len(size_audit.launches) == 6
    assert all(launch.grid_numerator <= CONTRACT_MAP_INT32_MAX for launch in size_audit.launches)
    assert all(launch.block_count <= CONTRACT_MAP_GRID_X_MAX for launch in size_audit.launches)
    assert generated.physical_abi.forward_outputs[1].shape == (1, maximum_bf16_elements)


@pytest.mark.parametrize("policy", tuple(ContractMapNumericalPolicy))
def test_contract_map_int32_abi_rejects_one_bf16_element_past_byte_size_boundary(
    policy: ContractMapNumericalPolicy,
) -> None:
    program = build_contract_map_backend_program(
        rows=1,
        reduction=1,
        features=CONTRACT_MAP_INT32_MAX // CONTRACT_MAP_BF16_BYTES + 1,
        scalar_expression=tanh_product_expression(),
        numerical_policy=policy,
    )

    with pytest.raises(ValueError, match="byte size exceeds the signed-int32 ABI bound"):
        generate_cuda_contract_map_backend_ffi(program)


def test_contract_map_physical_digest_separates_codegen_variants_and_artifact_stems(tmp_path) -> None:
    program = build_contract_map_backend_program(
        rows=43,
        reduction=104,
        features=72,
        scalar_expression=tanh_product_expression(),
        numerical_policy=ContractMapNumericalPolicy.FAST,
    )
    baseline = generate_cuda_contract_map_backend_ffi(program, threads=256)
    identical = generate_cuda_contract_map_backend_ffi(program, threads=256)
    different_threads = generate_cuda_contract_map_backend_ffi(program, threads=128)
    capture_safe = generate_cuda_contract_map_backend_ffi(
        program,
        threads=256,
        physical_candidate=DirectLaunchFfiPhysicalCandidate.COMMAND_BUFFER_CAPTURE_SAFE,
    )
    different_prefix = generate_cuda_contract_map_backend_ffi(program, target_prefix="shuttle.alternate.contract_map")
    variants = (baseline, different_threads, capture_safe, different_prefix)

    assert identical == baseline
    assert len({variant.physical_digest for variant in variants}) == len(variants)
    assert len({variant.forward_target for variant in variants}) == len(variants)
    assert len({variant.reverse_target for variant in variants}) == len(variants)
    assert len({variant.forward_handler_symbol for variant in variants}) == len(variants)
    assert len({variant.reverse_handler_symbol for variant in variants}) == len(variants)
    assert len({variant.forward_implementation_symbol for variant in variants}) == len(variants)
    assert len({variant.reverse_implementation_symbol for variant in variants}) == len(variants)
    assert len({variant.forward_binding_symbol for variant in variants}) == len(variants)
    assert len({variant.reverse_binding_symbol for variant in variants}) == len(variants)
    assert len({variant.forward_call_count_symbol for variant in variants}) == len(variants)
    assert len({variant.reverse_call_count_symbol for variant in variants}) == len(variants)
    assert len({variant.backend_fingerprint_symbol for variant in variants}) == len(variants)
    assert len({variant.source_sha256 for variant in variants}) == len(variants)
    for variant in variants:
        suffix = variant.physical_digest
        assert suffix in variant.forward_target
        assert suffix in variant.reverse_target
        assert suffix in variant.forward_handler_symbol
        assert suffix in variant.reverse_handler_symbol
        assert suffix in variant.forward_implementation_symbol
        assert suffix in variant.reverse_implementation_symbol
        assert suffix in variant.forward_binding_symbol
        assert suffix in variant.reverse_binding_symbol
        assert suffix in variant.forward_call_count_symbol
        assert suffix in variant.reverse_call_count_symbol
        assert suffix in variant.backend_fingerprint_symbol
        assert all(suffix in kernel_name for kernel_name in variant.kernel_names)
        assert "SHUTTLE_CONTRACT_MAP_PHYSICAL_IDENTITY" not in variant.source
        assert f'return "{variant.physical_digest}";' in variant.source
        compile_plan = contract_map_compile_plan(
            variant,
            artifact_directory=tmp_path,
            nvcc=tmp_path / "nvcc",
            include_directory=tmp_path / "include",
        )
        assert suffix in compile_plan.source_path.stem
        assert suffix in compile_plan.shared_library_path.stem
        assert suffix in compile_plan.ptx_path.stem
        assert suffix in compile_plan.cubin_path.stem
        assert suffix in compile_plan.sass_path.stem


def test_contract_map_emitted_kernel_body_changes_every_physical_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    program = build_contract_map_backend_program(
        rows=43,
        reduction=104,
        features=72,
        scalar_expression=tanh_product_expression(),
        numerical_policy=ContractMapNumericalPolicy.FAST,
    )
    baseline = generate_cuda_contract_map_backend_ffi(program)
    original_renderer = contract_map_codegen._first_forward_kernel

    def render_changed_kernel(policy: ContractMapNumericalPolicy, *, name: str) -> str:
        source = original_renderer(policy, name=name)
        old = "float accumulator = 0.0f;"
        assert old in source
        return source.replace(old, "float accumulator = __fadd_rn(0.0f, 0.0f);", 1)

    monkeypatch.setattr(contract_map_codegen, "_first_forward_kernel", render_changed_kernel)
    changed = generate_cuda_contract_map_backend_ffi(program)
    repeated_changed = generate_cuda_contract_map_backend_ffi(program)

    assert repeated_changed == changed
    assert changed.physical_digest != baseline.physical_digest
    assert changed.source_sha256 != baseline.source_sha256
    assert changed.forward_target != baseline.forward_target
    assert changed.reverse_target != baseline.reverse_target
    assert changed.physical_digest in changed.forward_target
    assert changed.physical_digest in changed.reverse_target
    host_symbols = (
        "forward_handler_symbol",
        "reverse_handler_symbol",
        "forward_implementation_symbol",
        "reverse_implementation_symbol",
        "forward_binding_symbol",
        "reverse_binding_symbol",
        "forward_call_count_symbol",
        "reverse_call_count_symbol",
        "backend_fingerprint_symbol",
    )
    for field in host_symbols:
        assert getattr(changed, field) != getattr(baseline, field)
        assert changed.physical_digest in getattr(changed, field)
        assert getattr(changed, field) in changed.source
    assert len(baseline.kernel_names) == len(changed.kernel_names) == 6
    for old_name, new_name in zip(baseline.kernel_names, changed.kernel_names, strict=True):
        assert old_name != new_name
        assert changed.physical_digest in new_name
        assert new_name in changed.source
    assert f'return "{changed.physical_digest}";' in changed.source

    baseline_plan = contract_map_compile_plan(
        baseline,
        artifact_directory=tmp_path,
        nvcc=tmp_path / "nvcc",
        include_directory=tmp_path / "include",
    )
    changed_plan = contract_map_compile_plan(
        changed,
        artifact_directory=tmp_path,
        nvcc=tmp_path / "nvcc",
        include_directory=tmp_path / "include",
    )
    for field in ("source_path", "shared_library_path", "ptx_path", "cubin_path", "sass_path"):
        baseline_stem = getattr(baseline_plan, field).stem
        changed_stem = getattr(changed_plan, field).stem
        assert changed_stem != baseline_stem
        assert changed.physical_digest in changed_stem


def test_contract_map_semantic_fingerprint_ignores_value_and_operation_names() -> None:
    baseline = build_contract_map_backend_program(
        rows=5,
        reduction=8,
        features=4,
        scalar_expression=cubic_mix_expression(),
        numerical_policy=ContractMapNumericalPolicy.FAST,
    )
    renamed_source = _rename_program(baseline.source)
    renamed = form_contract_map_backend_program(renamed_source, numerical_policy=ContractMapNumericalPolicy.FAST)

    assert renamed.semantic_fingerprint == baseline.semantic_fingerprint
    assert (
        generate_cuda_contract_map_backend_ffi(renamed).source_sha256
        == generate_cuda_contract_map_backend_ffi(baseline).source_sha256
    )


def test_contract_map_backend_rejects_return_rewiring() -> None:
    baseline = build_contract_map_backend_program(
        rows=5,
        reduction=8,
        features=4,
        scalar_expression=cubic_mix_expression(),
        numerical_policy=ContractMapNumericalPolicy.FAST,
    )
    rewired = TensorProgram(
        inputs=baseline.source.inputs,
        operations=baseline.source.operations,
        outputs=(baseline.preactivation,),
    )

    with pytest.raises(ValueError, match="return must be the second Contract"):
        form_contract_map_backend_program(rewired, numerical_policy=ContractMapNumericalPolicy.FAST)


def test_contract_map_backend_rejects_unlowered_contract_index_maps() -> None:
    baseline = build_contract_map_backend_program(
        rows=5,
        reduction=8,
        features=4,
        scalar_expression=cubic_mix_expression(),
        numerical_policy=ContractMapNumericalPolicy.FAST,
    )
    first_contract = baseline.source.operations[0]
    assert isinstance(first_contract, ContractPrimitive)
    row_axis = baseline.activation.axes[0]
    indexed_contract = replace(
        first_contract,
        input_index_maps=((AxisIndexMap(domain_axis=row_axis, operand_axis=row_axis),), ()),
    )
    indexed = replace(
        baseline.source,
        operations=(indexed_contract, *baseline.source.operations[1:]),
    )

    with pytest.raises(ValueError, match="does not support Contract index maps"):
        form_contract_map_backend_program(indexed, numerical_policy=ContractMapNumericalPolicy.FAST)


def test_contract_map_logical_boundary_serializes_to_reviewed_closed_shape() -> None:
    program = build_contract_map_backend_program(
        rows=43,
        reduction=104,
        features=72,
        scalar_expression=sigmoid_product_expression(),
        numerical_policy=ContractMapNumericalPolicy.SOURCE_ORDERED,
    )
    generated = generate_cuda_contract_map_backend_ffi(program)
    logical = expected_contract_map_logical_boundary(generated, kernel_only=False).to_evidence()
    round_tripped = json.loads(json.dumps(logical))

    assert round_tripped["saved_state_names_and_bytes"] == {
        "preactivation": 43 * 72 * 2,
        "hidden": 43 * 72 * 2,
    }
    assert round_tripped["layout_adapters"] == []
    assert round_tripped["materialized_copies"] == []
    assert len(round_tripped["input_layouts"]) == 3
    assert len(round_tripped["output_layouts"]) == 4


def test_contract_map_ptxas_parser_requires_resources_for_every_generated_kernel() -> None:
    names = ("KernelA", "KernelB")
    output = """
ptxas info    : Function properties for KernelA
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, 0 bytes smem
ptxas info    : Function properties for KernelB
    16 bytes stack frame, 8 bytes spill stores, 4 bytes spill loads
ptxas info    : Used 64 registers, 128 bytes smem
"""
    resources = parse_ptxas_kernel_resources(output, expected_kernel_names=names)

    assert tuple(resource.kernel_name for resource in resources) == names
    assert resources[0].registers_per_thread == 40
    assert resources[1].spill_load_bytes == 4
    assert resources[1].spill_store_bytes == 8
    assert resources[1].static_shared_bytes == 128
    with pytest.raises(ValueError, match="omits generated kernels"):
        parse_ptxas_kernel_resources(output, expected_kernel_names=(*names, "KernelC"))


def test_contract_map_ffi_rejects_wrong_logical_buffers_before_dispatch() -> None:
    program = build_contract_map_backend_program(
        rows=5,
        reduction=8,
        features=4,
        scalar_expression=tanh_product_expression(),
        numerical_policy=ContractMapNumericalPolicy.FAST,
    )
    generated = generate_cuda_contract_map_backend_ffi(program)
    activation = jnp.zeros((5, 8), dtype=jnp.bfloat16)
    first_weight = jnp.zeros((8, 4), dtype=jnp.bfloat16)
    second_weight = jnp.zeros((4, 8), dtype=jnp.bfloat16)

    with pytest.raises(ValueError, match=r"activation.*shape"):
        call_cuda_contract_map_backend_forward_ffi(
            generated,
            jnp.zeros((4, 8), dtype=jnp.bfloat16),
            first_weight,
            second_weight,
        )
    with pytest.raises(ValueError, match=r"preactivation.*dtype"):
        call_cuda_contract_map_backend_reverse_ffi(
            generated,
            activation,
            first_weight,
            second_weight,
            jnp.zeros((5, 4), dtype=jnp.float32),
            jnp.zeros((5, 4), dtype=jnp.bfloat16),
            jnp.zeros((5, 8), dtype=jnp.bfloat16),
        )


def _replace_reverse_operations(
    program: ContractMapBackendProgram,
    reverse_operations: tuple[ContractPrimitive | MapPrimitive, ...],
) -> ContractMapBackendProgram:
    differentiated_program = replace(
        program.differentiated.program,
        operations=(*program.source.operations, *reverse_operations),
    )
    return replace(
        program,
        differentiated=replace(program.differentiated, program=differentiated_program),
    )


def _rename_program(source: TensorProgram) -> TensorProgram:
    values: dict[str, ProgramValue] = {}

    def renamed(value: ProgramValue) -> ProgramValue:
        if value.name not in values:
            values[value.name] = ProgramValue(f"renamed_{len(values)}", value.axes, value.dtype)
        return values[value.name]

    inputs = tuple(renamed(value) for value in source.inputs)
    operations = []
    for index, operation in enumerate(source.operations):
        if isinstance(operation, ContractPrimitive):
            operations.append(
                ContractPrimitive(
                    name=f"renamed_contract_{index}",
                    inputs=tuple(renamed(value) for value in operation.inputs),
                    output=renamed(operation.output),
                    reduction_axes=operation.reduction_axes,
                    accumulation_dtype=operation.accumulation_dtype,
                )
            )
        else:
            assert isinstance(operation, MapPrimitive)
            names = {value.name: renamed(value).name for value in operation.inputs}
            operations.append(
                MapPrimitive(
                    name=f"renamed_map_{index}",
                    inputs=tuple(renamed(value) for value in operation.inputs),
                    output=renamed(operation.output),
                    expression=_rename_expression(operation.expression, names),
                )
            )
    return TensorProgram(
        inputs=inputs, operations=tuple(operations), outputs=tuple(renamed(value) for value in source.outputs)
    )


def _rename_expression(expression: ScalarExpression, names: dict[str, str]) -> ScalarExpression:
    if expression.kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        return scalar_input(names[expression.input_name])
    return ScalarExpression(
        kind=expression.kind,
        operands=tuple(_rename_expression(operand, names) for operand in expression.operands),
        constant=expression.constant,
    )
