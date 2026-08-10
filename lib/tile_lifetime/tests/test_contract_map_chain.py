# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tile_lifetime.contract_map_chain import (
    BoundCastScalarMap,
    ContractMapChainValue,
    contract_map_chain_physical_abi,
    execute_two_contract_map_forward,
    execute_two_contract_map_reverse,
    form_two_contract_map_training_program,
)
from tile_lifetime.cuda_contract_map_chain_codegen import (
    ContractMapChainFfiPhysicalCandidate,
    audit_cuda_contract_map_chain_source,
    generate_cuda_contract_map_chain_ffi,
)
from tile_lifetime.xla_hlo_recovery import inline_elementwise_fusions, parse_hlo_module_text
from tile_lifetime.xla_low_rank_gated_product import recover_low_rank_gated_product_training
from tile_lifetime.xla_normalized_exp_contract_forward import (
    plan_normalized_exp_contract_forward_hlo_replacement,
    replace_normalized_exp_contract_forward_hlo_region_with_custom_call,
)
from tile_lifetime.xla_normalized_exp_contract_reverse import (
    plan_normalized_exp_contract_reverse_hlo_replacement,
    replace_normalized_exp_contract_reverse_hlo_region_with_custom_call,
)
from tile_lifetime.xla_scalar_map_import import import_hlo_scalar_map

_ARTIFACT = (
    Path(__file__).parents[1] / "benchmarks/artifacts/xla_grug_shared_map_h100_narrowed_unaccepted_da49b94c_v0/"
    "transformed-gpu-pre-scheduler-hlo.txt.gz"
)


def _program():
    hlo = gzip.decompress(_ARTIFACT.read_bytes()).decode()
    forward_replacement = plan_normalized_exp_contract_forward_hlo_replacement(hlo)
    hlo = replace_normalized_exp_contract_forward_hlo_region_with_custom_call(
        hlo,
        forward_replacement,
        target="shuttle.test.normalized_exp.forward",
    )
    reverse_replacement = plan_normalized_exp_contract_reverse_hlo_replacement(hlo)
    hlo = replace_normalized_exp_contract_reverse_hlo_region_with_custom_call(
        hlo,
        reverse_replacement,
        target="shuttle.test.normalized_exp.reverse",
    )
    report = recover_low_rank_gated_product_training(hlo)
    reverse = report.reverse_families[0]
    return form_two_contract_map_training_program(reverse.primal, reverse)


def _natural_jax(activation, first_weight, second_weight):
    one = jnp.asarray(1.0, dtype=jnp.bfloat16)
    first = jnp.matmul(activation, first_weight, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
    hidden = (first * (one / (one + jnp.exp(-first)))).astype(jnp.bfloat16)
    second = jnp.matmul(hidden, second_weight, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
    return (activation * (one / (one + jnp.exp(-second)))).astype(jnp.bfloat16)


def _natural_jax_tanh_hidden(activation, first_weight, second_weight):
    one = jnp.asarray(1.0, dtype=jnp.bfloat16)
    first = jnp.matmul(activation, first_weight, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
    hidden = jnp.tanh(first)
    second = jnp.matmul(hidden, second_weight, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
    return (activation * (one / (one + jnp.exp(-second)))).astype(jnp.bfloat16)


def _jax_tanh_maps():
    shape = jax.ShapeDtypeStruct((8, 128), jnp.bfloat16)
    forward_hlo = jax.jit(jnp.tanh).lower(shape).compiler_ir("hlo").as_hlo_text()
    forward_module = parse_hlo_module_text(forward_hlo)
    forward_entry = forward_module.computation(forward_module.entry)
    forward_graph = inline_elementwise_fusions(forward_module)
    forward_parameter = next(value for value in forward_entry.instructions if value.opcode == "parameter")
    forward = import_hlo_scalar_map(
        forward_graph,
        source_nodes=(forward_graph.entry_value(forward_parameter.name),),
        target_node=forward_graph.entry_value(forward_entry.root.name),
    )

    def reverse_function(value, cotangent):
        return jax.vjp(jnp.tanh, value)[1](cotangent)[0]

    reverse_hlo = jax.jit(reverse_function).lower(shape, shape).compiler_ir("hlo").as_hlo_text()
    reverse_module = parse_hlo_module_text(reverse_hlo)
    reverse_entry = reverse_module.computation(reverse_module.entry)
    reverse_graph = inline_elementwise_fusions(reverse_module)
    parameter_by_number = {
        int(value.attributes.removeprefix("parameter(").removesuffix(")")): value
        for value in reverse_entry.instructions
        if value.opcode == "parameter"
    }
    reverse = import_hlo_scalar_map(
        reverse_graph,
        source_nodes=(
            reverse_graph.entry_value(parameter_by_number[1].name),
            reverse_graph.entry_value(parameter_by_number[0].name),
        ),
        target_node=reverse_graph.entry_value(reverse_entry.root.name),
    )
    return forward, reverse


def test_contract_map_chain_reference_matches_natural_jax_forward_and_vjp() -> None:
    program = _program()
    rng = np.random.default_rng(0)
    activation = rng.normal(scale=0.2, size=(8, 32)).astype(np.float32)
    first_weight = rng.normal(scale=0.2, size=(32, 128)).astype(np.float32)
    second_weight = rng.normal(scale=0.2, size=(128, 32)).astype(np.float32)
    output_cotangent = rng.normal(scale=0.2, size=(8, 32)).astype(np.float32)

    forward = execute_two_contract_map_forward(program, activation, first_weight, second_weight)
    activation_jax = jnp.asarray(activation, dtype=jnp.bfloat16)
    first_weight_jax = jnp.asarray(first_weight, dtype=jnp.bfloat16)
    second_weight_jax = jnp.asarray(second_weight, dtype=jnp.bfloat16)
    expected_output, pullback = jax.vjp(_natural_jax, activation_jax, first_weight_jax, second_weight_jax)
    expected_reverse = pullback(jnp.asarray(output_cotangent, dtype=jnp.bfloat16))
    reverse = execute_two_contract_map_reverse(
        program,
        activation,
        first_weight,
        second_weight,
        forward,
        output_cotangent,
    )

    np.testing.assert_array_equal(forward.output, np.asarray(expected_output, dtype=np.float32))
    actual_reverse = (reverse.input_adjoint, reverse.first_weight_adjoint, reverse.second_weight_adjoint)
    max_errors = tuple(
        float(np.max(np.abs(actual - np.asarray(expected, dtype=np.float32))))
        for actual, expected in zip(actual_reverse, expected_reverse, strict=True)
    )
    mean_errors = tuple(
        float(np.mean(np.abs(actual - np.asarray(expected, dtype=np.float32))))
        for actual, expected in zip(actual_reverse, expected_reverse, strict=True)
    )
    assert max(max_errors) <= 0.001
    assert max(mean_errors) <= 0.0001


def test_contract_map_chain_source_owns_generic_maps_and_ordered_bf16_boundaries() -> None:
    program = _program()
    generated = generate_cuda_contract_map_chain_ffi(
        program,
        forward_target="shuttle.generic.contract_map_chain.forward",
        reverse_target="shuttle.generic.contract_map_chain.reverse",
    )
    capture_safe = generate_cuda_contract_map_chain_ffi(
        program,
        forward_target="shuttle.generic.contract_map_chain.forward",
        reverse_target="shuttle.generic.contract_map_chain.reverse",
        physical_candidate=ContractMapChainFfiPhysicalCandidate.COMMAND_BUFFER_CAPTURE_SAFE,
    )
    audit = audit_cuda_contract_map_chain_source(generated)
    capture_safe_audit = audit_cuda_contract_map_chain_source(capture_safe)

    assert (generated.rows, generated.input_features, generated.rank) == (8, 32, 128)
    assert generated.kernel_count == 2
    assert generated.forward_shared_bytes == 4096
    assert generated.reverse_shared_bytes == 2560
    assert generated.source.count("WeightAdjointDimensionZeroMinor = true") == 2
    assert audit.kernel_count == 2
    assert audit.has_explicit_bf16_contract_boundaries
    assert audit.has_generated_forward_maps
    assert audit.has_generated_reverse_maps
    assert audit.has_handler_counters
    assert audit.has_launch_status_query
    assert not audit.has_command_buffer_traits
    assert not audit.command_buffer_eligible
    assert audit.forbidden_command_buffer_operations == ("runtime launch-status query",)
    assert capture_safe.semantic_digest == generated.semantic_digest
    assert capture_safe.source_digest != generated.source_digest
    assert capture_safe.command_buffer_compatible
    assert capture_safe_audit.command_buffer_eligible
    assert capture_safe_audit.has_command_buffer_traits
    assert not capture_safe_audit.has_launch_status_query
    assert not capture_safe_audit.forbidden_command_buffer_operations
    assert not audit.has_atomics
    assert not audit.opaque_semantic_dependencies
    assert generated.external_dependencies == ("CUDA BF16/runtime primitives", "XLA typed FFI")


def test_contract_map_chain_physical_abi_matches_cuda_indexing() -> None:
    program = _program()
    physical_abi = contract_map_chain_physical_abi(program)
    generated = generate_cuda_contract_map_chain_ffi(
        program,
        forward_target="shuttle.generic.contract_map_chain.forward",
        reverse_target="shuttle.generic.contract_map_chain.reverse",
    )

    assert generated.physical_abi == physical_abi
    assert tuple(value.hlo_shape for value in physical_abi.forward_inputs) == (
        "bf16[8,32]{1,0}",
        "bf16[32,128]{1,0}",
        "bf16[128,32]{1,0}",
    )
    assert tuple(value.hlo_shape for value in physical_abi.forward_outputs) == (
        "bf16[8,32]{1,0}",
        "bf16[8,128]{1,0}",
        "bf16[8,128]{1,0}",
        "bf16[8,32]{1,0}",
    )
    assert tuple(value.hlo_shape for value in physical_abi.reverse_outputs) == (
        "bf16[8,32]{1,0}",
        "bf16[32,128]{0,1}",
        "bf16[128,32]{0,1}",
    )


def test_hidden_map_mutation_regenerates_the_same_physical_family() -> None:
    program = _program()
    tanh_forward, tanh_reverse = _jax_tanh_maps()
    tanh_map = BoundCastScalarMap(
        tanh_forward,
        (ContractMapChainValue.FIRST_CONTRACT_OUTPUT,),
    )
    tanh_vjp_map = BoundCastScalarMap(
        tanh_reverse,
        (
            ContractMapChainValue.SECOND_CONTRACT_INPUT_ADJOINT,
            ContractMapChainValue.FIRST_CONTRACT_OUTPUT,
        ),
    )
    mutated = replace(program, hidden_map=tanh_map, hidden_vjp_map=tanh_vjp_map)
    baseline_source = generate_cuda_contract_map_chain_ffi(
        program,
        forward_target="shuttle.generic.contract_map_chain.forward",
        reverse_target="shuttle.generic.contract_map_chain.reverse",
        physical_candidate=ContractMapChainFfiPhysicalCandidate.COMMAND_BUFFER_CAPTURE_SAFE,
    )
    mutated_source = generate_cuda_contract_map_chain_ffi(
        mutated,
        forward_target="shuttle.generic.contract_map_chain.forward",
        reverse_target="shuttle.generic.contract_map_chain.reverse",
        physical_candidate=ContractMapChainFfiPhysicalCandidate.COMMAND_BUFFER_CAPTURE_SAFE,
    )

    baseline_audit = audit_cuda_contract_map_chain_source(baseline_source)
    mutated_audit = audit_cuda_contract_map_chain_source(mutated_source)
    assert baseline_source.kernel_count == mutated_source.kernel_count
    assert baseline_source.forward_shared_bytes == mutated_source.forward_shared_bytes
    assert baseline_source.reverse_shared_bytes == mutated_source.reverse_shared_bytes
    assert baseline_source.forward_handler_symbol == mutated_source.forward_handler_symbol
    assert baseline_source.reverse_handler_symbol == mutated_source.reverse_handler_symbol
    assert baseline_source.semantic_digest != mutated_source.semantic_digest
    assert baseline_source.source_digest != mutated_source.source_digest
    assert baseline_source.command_buffer_compatible
    assert mutated_source.command_buffer_compatible
    assert baseline_audit.command_buffer_eligible
    assert mutated_audit.command_buffer_eligible
    assert baseline_audit.has_command_buffer_traits
    assert mutated_audit.has_command_buffer_traits
    assert not baseline_audit.has_launch_status_query
    assert not mutated_audit.has_launch_status_query
    assert "tanhf" in mutated_source.source

    rng = np.random.default_rng(1)
    activation = rng.normal(scale=0.2, size=(8, 32)).astype(np.float32)
    first_weight = rng.normal(scale=0.2, size=(32, 128)).astype(np.float32)
    second_weight = rng.normal(scale=0.2, size=(128, 32)).astype(np.float32)
    output_cotangent = rng.normal(scale=0.2, size=(8, 32)).astype(np.float32)
    forward = execute_two_contract_map_forward(mutated, activation, first_weight, second_weight)
    reverse = execute_two_contract_map_reverse(
        mutated,
        activation,
        first_weight,
        second_weight,
        forward,
        output_cotangent,
    )
    expected_output, pullback = jax.vjp(
        _natural_jax_tanh_hidden,
        jnp.asarray(activation, dtype=jnp.bfloat16),
        jnp.asarray(first_weight, dtype=jnp.bfloat16),
        jnp.asarray(second_weight, dtype=jnp.bfloat16),
    )
    expected_reverse = pullback(jnp.asarray(output_cotangent, dtype=jnp.bfloat16))
    np.testing.assert_array_equal(forward.output, np.asarray(expected_output, dtype=np.float32))
    for actual, expected in zip(
        (reverse.input_adjoint, reverse.first_weight_adjoint, reverse.second_weight_adjoint),
        expected_reverse,
        strict=True,
    ):
        np.testing.assert_allclose(actual, np.asarray(expected, dtype=np.float32), rtol=0.015625, atol=0.0078125)
