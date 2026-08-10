# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import math
import re
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from tile_lifetime.cast_scalar_program import (
    CastScalarDType,
    CastScalarExpression,
    CastScalarKind,
    CastScalarProgram,
    ScalarIndexRelation,
    evaluate_cast_scalar_program,
)
from tile_lifetime.cuda_partitioned_gemm_codegen import (
    audit_cuda_partitioned_gemm_source,
    generate_cuda_partitioned_gemm_ffi,
)
from tile_lifetime.jax_partitioned_gemm_ffi import (
    evaluate_partitioned_gemm_jax,
    partitioned_gemm_cuda_compile_plan,
    partitioned_gemm_jax_ffi_spec,
)
from tile_lifetime.partitioned_gemm_program import (
    AccumulatorPartition,
    PartitionedGemmProgram,
    PassthroughPartitionFinalization,
    ScalarPartitionFinalization,
)
from tile_lifetime.partitioned_gemm_reference import evaluate_partitioned_gemm_reference
from tile_lifetime.quack_partitioned_gemm_adapter import (
    QUACK_PARTITION_ADAPTER_BASE_REVISION,
    QuackPartitionFinalizationKind,
    plan_quack_partitioned_gemm_adapter,
)
from tile_lifetime.quack_partitioned_mainloop import (
    QUACK_0_5_0_WHEEL_SHA256,
    QUACK_PARTITIONED_SM90_PATCH_SHA256,
    audit_quack_partitioned_extension_patch,
    generate_quack_partitioned_mainloop,
    plan_quack_partitioned_mainloop,
)
from tile_lifetime.xla_low_rank_gated_product_ffi import (
    plan_generated_low_rank_contract_map_training,
    replace_generated_low_rank_contract_map_training,
)
from tile_lifetime.xla_partitioned_contract_map import (
    audit_attached_partitioned_contract_maps,
    plan_attached_partitioned_contract_maps,
    replace_attached_partitioned_contract_maps,
)

_TARGET_PREFIX = "shuttle.generic.partitioned_contract_map.test"


def _post_gated_product_grug_hlo() -> str:
    artifact = (
        Path(__file__).parents[1]
        / "benchmarks/artifacts/xla_grug_shared_map_h100_fused_reverses_unaccepted_e3411679_v0/"
        "transformed-gpu-pre-scheduler-hlo.txt.gz"
    )
    hlo = gzip.decompress(artifact.read_bytes()).decode()
    gated = plan_generated_low_rank_contract_map_training(
        hlo,
        forward_target_prefix="shuttle.generic.low_rank_contract_map.generated.forward.test",
        reverse_target_prefix="shuttle.generic.low_rank_contract_map.generated.reverse.test",
    )
    return replace_generated_low_rank_contract_map_training(hlo, gated)


def _tanh_gate_mutation(hlo: str) -> str:
    removed = {"neg.17", "exp.24", "add.45", "div.28"}
    lines: list[str] = []
    for line in hlo.splitlines(keepends=True):
        if any(re.match(rf"^\s*%{re.escape(name)} = ", line) for name in removed):
            continue
        if re.match(r"^\s*%mul\.52 = ", line):
            line = re.sub(r"multiply\([^)]*\)", "tanh(%reshape.1023)", line, count=1)
        lines.append(line)
    return "".join(lines)


def _kinds(expression) -> frozenset[CastScalarKind]:
    return frozenset({expression.kind, *(kind for operand in expression.operands for kind in _kinds(operand))})


def _bf16(value: float) -> float:
    return float(ml_dtypes.bfloat16(value))


def _tiny_partitioned_program() -> PartitionedGemmProgram:
    index = ScalarIndexRelation(row_offset=0, feature_offset=0)
    left = CastScalarExpression(
        CastScalarKind.INPUT,
        CastScalarDType.BF16,
        input_name="left",
        input_index=index,
    )
    right = CastScalarExpression(
        CastScalarKind.INPUT,
        CastScalarDType.BF16,
        input_name="right",
        input_index=index,
    )
    return PartitionedGemmProgram(
        shape=(2, 5, 3),
        partitioned_operand=1,
        operand_shapes=(
            "bf16[2,3]{1,0}",
            "bf16[2,3]{0,1}",
            "bf16[2,3]{0,1}",
            "bf16[1,3]{0,1}",
        ),
        partitions=(
            AccumulatorPartition(0, 2, "bf16[2,2]{1,0}"),
            AccumulatorPartition(2, 4, "bf16[2,2]{1,0}"),
            AccumulatorPartition(4, 5, "bf16[2,1]{1,0}"),
        ),
        scalar_finalizations=(
            ScalarPartitionFinalization(
                source_partitions=(0, 1),
                program=CastScalarProgram(
                    CastScalarExpression(
                        CastScalarKind.MULTIPLY,
                        CastScalarDType.BF16,
                        operands=(left, right),
                    )
                ),
                output_shape="bf16[2,2]{1,0}",
            ),
        ),
        passthrough_finalizations=(PassthroughPartitionFinalization(2, "bf16[2,1]{1,0}"),),
        input_dtype="bf16",
        accumulation_dtype="f32",
        partition_dtype="bf16",
        output_dtype="bf16",
        output_rounding="round_to_nearest_even",
    )


def test_natural_grug_recovers_one_exclusive_partition_map_program() -> None:
    plan = plan_attached_partitioned_contract_maps(_post_gated_product_grug_hlo(), target_prefix=_TARGET_PREFIX)

    assert len(plan.families) == 1
    assert len(plan.calls) == 1
    family = plan.families[0]
    call = plan.calls[0]
    assert family.generated.template == "partitioned_gemm_scalar_finalization"
    assert family.call_names == (call.call_name,)
    assert family.program.shape == (8, 68, 32)
    assert tuple((partition.start, partition.limit) for partition in family.program.partitions) == (
        (0, 32),
        (32, 64),
        (64, 68),
    )
    assert family.program.scalar_finalizations[0].source_partitions == (0, 1)
    assert family.program.passthrough_finalizations[0].source_partition == 2
    assert family.program.output_shapes == ("bf16[8,32]{1,0}", "bf16[2,4,4]{2,1,0}")
    assert family.program.accumulation_dtype == "f32"
    assert family.program.partition_dtype == "bf16"
    assert family.program.output_rounding == "round_to_nearest_even"
    assert call.base.recovered.entry_instruction == "dot.88"
    assert call.map_output.instruction == "mul.781"
    assert tuple(output.instruction for output in call.passthrough_outputs) == ("slice.72",)
    scalar = family.program.scalar_finalizations[0].program
    assert _kinds(scalar.expression) >= {
        CastScalarKind.EXP,
        CastScalarKind.NEGATE,
        CastScalarKind.DIVIDE,
        CastScalarKind.MULTIPLY,
    }
    assert "expf" in family.generated.scalar_bodies[0].source
    assert "swiglu" not in family.generated.scalar_bodies[0].source.lower()


def test_recovered_scalar_map_preserves_source_ordered_bf16_operations() -> None:
    plan = plan_attached_partitioned_contract_maps(_post_gated_product_grug_hlo(), target_prefix=_TARGET_PREFIX)
    scalar = plan.families[0].program.scalar_finalizations[0].program
    gate = _bf16(0.73)
    up = _bf16(-1.17)
    expected = _bf16(_bf16(gate * _bf16(_bf16(1.0) / _bf16(_bf16(1.0) + _bf16(math.exp(_bf16(-gate)))))) * up)

    actual = evaluate_cast_scalar_program(scalar, {"input0_r0_f0": gate, "input1_r0_f0": up})

    assert actual == expected


def test_attached_replacement_removes_real_intermediates_and_preserves_boundaries() -> None:
    hlo = _post_gated_product_grug_hlo()
    plan = plan_attached_partitioned_contract_maps(hlo, target_prefix=_TARGET_PREFIX)
    transformed = replace_attached_partitioned_contract_maps(hlo, plan)
    audit = audit_attached_partitioned_contract_maps(hlo, transformed, plan)

    assert audit.removed_contract_count == 1
    assert audit.removed_contract_flops == 34_816
    assert audit.target_occurrences == ((plan.families[0].target, 1),)
    assert audit.output_users == (
        ("mul.781", ("dot_general.182",)),
        ("slice.72", ("reshape.1025",)),
    )
    assert len(audit.collective_instructions) == 10
    assert audit.copy_count == (0, 0)
    assert audit.transpose_count == (43, 43)
    assert "%slice.70 =" not in transformed
    assert "%slice.71 =" not in transformed
    assert "%mul.781 = bf16[8,32]{1,0} get-tuple-element" in transformed
    assert "%slice.72 = bf16[2,4,4]{2,1,0} get-tuple-element" in transformed


def test_activation_mutation_regenerates_through_the_same_partitioned_family() -> None:
    hlo = _post_gated_product_grug_hlo()
    mutated_hlo = _tanh_gate_mutation(hlo)
    original = plan_attached_partitioned_contract_maps(hlo, target_prefix=_TARGET_PREFIX)
    mutated = plan_attached_partitioned_contract_maps(mutated_hlo, target_prefix=_TARGET_PREFIX)

    original_family = original.families[0]
    mutated_family = mutated.families[0]
    assert original_family.generated.template == mutated_family.generated.template
    assert original_family.program.shape == mutated_family.program.shape
    assert original_family.program.partitions == mutated_family.program.partitions
    assert original_family.program.output_shapes == mutated_family.program.output_shapes
    assert original_family.generated.semantic_digest != mutated_family.generated.semantic_digest
    assert original_family.target != mutated_family.target
    assert CastScalarKind.TANH in _kinds(mutated_family.program.scalar_finalizations[0].program.expression)
    assert CastScalarKind.EXP not in _kinds(mutated_family.program.scalar_finalizations[0].program.expression)
    assert "tanhf" in mutated_family.generated.scalar_bodies[0].source
    assert "expf" not in mutated_family.generated.scalar_bodies[0].source

    transformed = replace_attached_partitioned_contract_maps(mutated_hlo, mutated)
    audit = audit_attached_partitioned_contract_maps(mutated_hlo, transformed, mutated)
    assert audit.removed_contract_count == 1
    assert audit.output_users == (
        ("mul.781", ("dot_general.182",)),
        ("slice.72", ("reshape.1025",)),
    )


def test_quack_adapter_requires_one_segmented_rhs_mainloop_and_direct_partition_stores() -> None:
    family = plan_attached_partitioned_contract_maps(
        _post_gated_product_grug_hlo(), target_prefix=_TARGET_PREFIX
    ).families[0]

    adapter = plan_quack_partitioned_gemm_adapter(family.program)

    assert adapter.base_revision == QUACK_PARTITION_ADAPTER_BASE_REVISION
    assert adapter.unpartitioned_operand_index == 0
    assert tuple(
        (source.operand_index, source.n_start, source.n_limit, source.shape) for source in adapter.segmented_rhs_sources
    ) == (
        (1, 0, 32, "bf16[32,32]{0,1}"),
        (2, 32, 64, "bf16[32,32]{0,1}"),
        (3, 64, 68, "bf16[4,32]{0,1}"),
    )
    assert tuple((view.n_start, view.n_limit) for view in adapter.accumulator_views) == (
        (0, 32),
        (32, 64),
        (64, 68),
    )
    assert all(view.boundary_dtype == "bf16" for view in adapter.accumulator_views)
    assert all(view.boundary_rounding == "round_to_nearest_even" for view in adapter.accumulator_views)
    assert tuple((store.kind, store.source_partitions, store.output_shape) for store in adapter.stores) == (
        (QuackPartitionFinalizationKind.SCALAR_MAP, (0, 1), "bf16[8,32]{1,0}"),
        (QuackPartitionFinalizationKind.PASSTHROUGH, (2,), "bf16[2,4,4]{2,1,0}"),
    )
    scalar_body = adapter.stores[0].scalar_body
    assert scalar_body is not None
    assert "expf" in scalar_body
    assert adapter.stores[1].scalar_body is None
    assert adapter.requires_composed_rhs_tma
    assert adapter.requires_physical_proof
    assert adapter.implementation_sites[-1].startswith("quack/gemm_base.py")


def test_quack_adapter_mutation_reuses_physical_partition_structure() -> None:
    hlo = _post_gated_product_grug_hlo()
    original = plan_attached_partitioned_contract_maps(hlo, target_prefix=_TARGET_PREFIX).families[0]
    mutated = plan_attached_partitioned_contract_maps(_tanh_gate_mutation(hlo), target_prefix=_TARGET_PREFIX).families[0]

    original_adapter = plan_quack_partitioned_gemm_adapter(original.program)
    mutated_adapter = plan_quack_partitioned_gemm_adapter(mutated.program)

    assert original_adapter.segmented_rhs_sources == mutated_adapter.segmented_rhs_sources
    assert original_adapter.accumulator_views == mutated_adapter.accumulator_views
    original_stores = tuple(
        (store.kind, store.source_partitions, store.output_shape) for store in original_adapter.stores
    )
    mutated_stores = tuple((store.kind, store.source_partitions, store.output_shape) for store in mutated_adapter.stores)
    assert original_stores == mutated_stores
    assert original_adapter.semantic_digest != mutated_adapter.semantic_digest
    original_body = original_adapter.stores[0].scalar_body
    mutated_body = mutated_adapter.stores[0].scalar_body
    assert original_body is not None
    assert mutated_body is not None
    assert "expf" in original_body
    assert "tanhf" in mutated_body


def test_quack_adapter_rejects_a_partition_source_with_the_wrong_physical_extent() -> None:
    family = plan_attached_partitioned_contract_maps(
        _post_gated_product_grug_hlo(), target_prefix=_TARGET_PREFIX
    ).families[0]
    invalid = replace(
        family.program,
        operand_shapes=(
            family.program.operand_shapes[0],
            "bf16[31,32]{0,1}",
            *family.program.operand_shapes[2:],
        ),
    )

    with pytest.raises(ValueError, match="physical shape"):
        plan_quack_partitioned_gemm_adapter(invalid)


def test_quack_partitioned_mainloop_reuses_one_a_stage_and_one_k_loop() -> None:
    family = plan_attached_partitioned_contract_maps(
        _post_gated_product_grug_hlo(), target_prefix=_TARGET_PREFIX
    ).families[0]

    physical = plan_quack_partitioned_mainloop(family.program)

    assert physical.base_revision == QUACK_PARTITION_ADAPTER_BASE_REVISION
    assert physical.inspected_wheel_sha256 == QUACK_0_5_0_WHEEL_SHA256
    assert physical.one_kernel
    assert physical.one_k_loop
    assert physical.shared_a_stage
    assert tuple(
        (group.logical_n_start, group.logical_n_limit, group.mma_n, group.valid_n) for group in physical.rhs_groups
    ) == ((0, 32, 32, 32), (32, 64, 32, 32), (64, 68, 8, 4))
    assert tuple(
        (group.logical_n_start, group.logical_n_limit, group.mma_n, group.valid_n)
        for group in physical.accumulator_groups
    ) == ((0, 32, 32, 32), (32, 64, 32, 32), (64, 68, 8, 4))
    assert all(group.boundary_dtype == "bf16" for group in physical.accumulator_groups)
    assert all(group.boundary_rounding == "round_to_nearest_even" for group in physical.accumulator_groups)
    assert tuple((store.kind, store.source_groups, store.valid_n) for store in physical.stores) == (
        (QuackPartitionFinalizationKind.SCALAR_MAP, (0, 1), 32),
        (QuackPartitionFinalizationKind.PASSTHROUGH, (2,), 4),
    )
    assert physical.stores[0].scalar_body is not None
    assert "expf" in physical.stores[0].scalar_body
    assert physical.stores[1].scalar_body is None
    assert physical.requires_external_quack_extension


def test_quack_partitioned_mainloop_activation_mutation_preserves_tiled_structure() -> None:
    hlo = _post_gated_product_grug_hlo()
    original = plan_attached_partitioned_contract_maps(hlo, target_prefix=_TARGET_PREFIX).families[0]
    mutated = plan_attached_partitioned_contract_maps(_tanh_gate_mutation(hlo), target_prefix=_TARGET_PREFIX).families[0]

    original_physical = plan_quack_partitioned_mainloop(original.program)
    mutated_physical = plan_quack_partitioned_mainloop(mutated.program)

    assert original_physical.rhs_groups == mutated_physical.rhs_groups
    assert original_physical.accumulator_groups == mutated_physical.accumulator_groups
    assert tuple(
        (store.kind, store.source_groups, store.valid_n, store.output_shape) for store in original_physical.stores
    ) == tuple((store.kind, store.source_groups, store.valid_n, store.output_shape) for store in mutated_physical.stores)
    assert original_physical.physical_digest != mutated_physical.physical_digest
    assert original_physical.stores[0].scalar_body is not None
    assert mutated_physical.stores[0].scalar_body is not None
    assert "expf" in original_physical.stores[0].scalar_body
    assert "tanhf" in mutated_physical.stores[0].scalar_body


def test_quack_partitioned_extension_patch_is_pinned_and_workload_independent() -> None:
    patch = Path(__file__).parents[1] / "backends/h100/quack_partitioned_sm90.patch"

    audit = audit_quack_partitioned_extension_patch(patch)

    assert audit.sha256 == QUACK_PARTITIONED_SM90_PATCH_SHA256
    assert audit.required_symbols == (
        "validate_rhs_segments",
        "partition_accumulator_groups",
        "gemm_groups_w_idx",
        "round_group_to_bf16_rne",
        "PartitionedGemmSm90",
    )
    assert audit.missing_symbols == ()
    assert audit.forbidden_tokens == ()
    assert audit.creates_one_module
    assert audit.syntax_compiles
    assert audit.clean


def test_quack_partitioned_authoring_source_uses_one_generic_executor() -> None:
    family = plan_attached_partitioned_contract_maps(
        _post_gated_product_grug_hlo(), target_prefix=_TARGET_PREFIX
    ).families[0]

    generated = generate_quack_partitioned_mainloop(family.program)

    compile(generated.source, generated.module_name, "exec")
    assert generated.rhs_mma_ns == (32, 32, 8)
    assert generated.output_count == 2
    assert generated.source.count("PartitionedGemmSm90(") == 1
    assert "_executor(lhs, (rhs0, rhs1, rhs2), (output0, output1), stream)" in generated.source
    assert "cute.math.exp" in generated.source
    assert "fastmath=False" in generated.source
    assert "boundaries[2][local_m, feature]" in generated.source
    assert "swiglu" not in generated.source.lower()
    assert "router" not in generated.source.lower()


def test_quack_partitioned_authoring_mutation_preserves_executor_abi() -> None:
    hlo = _post_gated_product_grug_hlo()
    original = plan_attached_partitioned_contract_maps(hlo, target_prefix=_TARGET_PREFIX).families[0]
    mutated = plan_attached_partitioned_contract_maps(_tanh_gate_mutation(hlo), target_prefix=_TARGET_PREFIX).families[0]

    original_generated = generate_quack_partitioned_mainloop(original.program)
    mutated_generated = generate_quack_partitioned_mainloop(mutated.program)

    assert original_generated.rhs_mma_ns == mutated_generated.rhs_mma_ns
    assert original_generated.output_count == mutated_generated.output_count
    assert original_generated.source_digest != mutated_generated.source_digest
    assert "cute.math.exp" in original_generated.source
    assert "cute.math.tanh" in mutated_generated.source
    assert "cute.math.exp" not in mutated_generated.source


def test_quack_partitioned_natural_program_passes_cpu_and_jax_reference_gate() -> None:
    family = plan_attached_partitioned_contract_maps(
        _post_gated_product_grug_hlo(), target_prefix=_TARGET_PREFIX
    ).families[0]
    rng = np.random.default_rng(194)
    operands = (
        np.asarray(rng.normal(size=(2, 4, 32)), dtype=ml_dtypes.bfloat16),
        np.asarray(rng.normal(size=(32, 32)), dtype=ml_dtypes.bfloat16),
        np.asarray(rng.normal(size=(32, 32)), dtype=ml_dtypes.bfloat16),
        np.asarray(rng.normal(size=(4, 32)), dtype=ml_dtypes.bfloat16),
    )

    expected = evaluate_partitioned_gemm_reference(family.program, operands)
    actual = evaluate_partitioned_gemm_jax(family.program, tuple(jnp.asarray(operand) for operand in operands))

    for jax_output, reference_output in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(jax_output), reference_output)


def test_partitioned_contract_reference_preserves_ordered_bf16_partition_boundaries() -> None:
    program = _tiny_partitioned_program()
    operands = (
        np.asarray([[0.5, -1.0, 0.25], [1.5, 0.75, -0.5]], dtype=ml_dtypes.bfloat16),
        np.asarray([[1.0, 0.5, -0.25], [-0.5, 1.25, 0.75]], dtype=ml_dtypes.bfloat16),
        np.asarray([[0.25, -0.75, 1.0], [1.5, 0.5, -1.25]], dtype=ml_dtypes.bfloat16),
        np.asarray([[0.5, 0.25, 2.0]], dtype=ml_dtypes.bfloat16),
    )

    mapped, passthrough = evaluate_partitioned_gemm_reference(program, operands)

    lhs = operands[0].astype(np.float32)
    partitions = tuple(np.asarray(lhs @ rhs.astype(np.float32).T, dtype=ml_dtypes.bfloat16) for rhs in operands[1:])
    expected_mapped = np.asarray(
        partitions[0].astype(np.float32) * partitions[1].astype(np.float32),
        dtype=ml_dtypes.bfloat16,
    )
    np.testing.assert_array_equal(mapped, expected_mapped)
    np.testing.assert_array_equal(passthrough, partitions[2])

    jax_outputs = evaluate_partitioned_gemm_jax(
        program,
        tuple(jnp.asarray(operand) for operand in operands),
    )
    for actual, expected in zip(jax_outputs, (mapped, passthrough), strict=True):
        np.testing.assert_array_equal(np.asarray(actual), expected)


def test_partitioned_contract_cuda_generation_owns_one_generic_mainloop_and_direct_stores() -> None:
    family = plan_attached_partitioned_contract_maps(
        _post_gated_product_grug_hlo(), target_prefix=_TARGET_PREFIX
    ).families[0]
    generated = generate_cuda_partitioned_gemm_ffi(
        family.program,
        target="shuttle.generic.partitioned_contract_map.physical.test",
    )
    audit = audit_cuda_partitioned_gemm_source(generated)

    assert audit.kernel_count == 1
    assert audit.segmented_rhs_count == 3
    assert audit.direct_output_count == 2
    assert audit.has_ordered_fp32_mainloop
    assert audit.has_bf16_rne_partition_boundary
    assert audit.has_command_buffer_trait
    assert audit.has_handler_counter
    assert audit.command_buffer_eligible
    assert audit.forbidden_command_buffer_operations == ()
    assert not audit.has_atomics
    assert audit.opaque_semantic_dependencies == ()
    assert generated.shared_bytes == 8 * 68 * 2
    assert "ffi::Buffer<ffi::BF16, 3> operand0_buffer" in generated.source
    assert generated.source.count("ffi::Buffer<ffi::BF16, 2> operand") == 3
    assert "ffi::Result<ffi::Buffer<ffi::BF16, 2>> output0_buffer" in generated.source
    assert "ffi::Result<ffi::Buffer<ffi::BF16, 3>> output1_buffer" in generated.source
    assert "ffi::ScratchAllocator" not in generated.source
    assert "concatenated_output" not in generated.source
    assert f'extern "C" std::uint64_t {generated.call_count_symbol}()' in generated.source


def test_partitioned_contract_jax_ffi_spec_converts_layout_conventions_exactly() -> None:
    family = plan_attached_partitioned_contract_maps(
        _post_gated_product_grug_hlo(), target_prefix=_TARGET_PREFIX
    ).families[0]
    generated = generate_cuda_partitioned_gemm_ffi(family.program, target="shuttle.partition.layout.test")

    spec = partitioned_gemm_jax_ffi_spec(generated)

    assert spec.input_shapes == ((2, 4, 32), (32, 32), (32, 32), (4, 32))
    assert spec.output_shapes == ((8, 32), (2, 4, 4))
    assert spec.input_layouts == ((0, 1, 2), (1, 0), (1, 0), (1, 0))
    assert spec.output_layouts == ((0, 1), (0, 1, 2))


def test_partitioned_contract_activation_mutation_regenerates_the_same_physical_family() -> None:
    hlo = _post_gated_product_grug_hlo()
    original = plan_attached_partitioned_contract_maps(hlo, target_prefix=_TARGET_PREFIX).families[0]
    mutated = plan_attached_partitioned_contract_maps(_tanh_gate_mutation(hlo), target_prefix=_TARGET_PREFIX).families[0]
    original_generated = generate_cuda_partitioned_gemm_ffi(
        original.program, target="shuttle.partition.mutation.original"
    )
    mutated_generated = generate_cuda_partitioned_gemm_ffi(mutated.program, target="shuttle.partition.mutation.changed")

    assert original_generated.abi == mutated_generated.abi
    assert original_generated.threads == mutated_generated.threads
    assert original_generated.shared_bytes == mutated_generated.shared_bytes
    assert original_generated.semantic_digest != mutated_generated.semantic_digest
    assert original_generated.source_digest != mutated_generated.source_digest
    assert "expf" in original_generated.source
    assert "tanhf" in mutated_generated.source


def test_partitioned_contract_compile_plan_has_no_torch_or_opaque_compute_dependency(tmp_path: Path) -> None:
    toolkit = tmp_path / "cuda"
    nvcc = toolkit / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    library_directory = toolkit / "lib64"
    library_directory.mkdir()
    (library_directory / "libcudart.so").touch()
    include_directory = tmp_path / "jaxlib-include"
    include_directory.mkdir()
    generated = generate_cuda_partitioned_gemm_ffi(_tiny_partitioned_program(), target="shuttle.partition.compile.test")

    plan = partitioned_gemm_cuda_compile_plan(
        generated,
        directory=tmp_path / "build",
        nvcc=nvcc,
        architecture="sm_90a",
        jaxlib_include=include_directory,
    )

    assert "-arch=sm_90a" in plan.argv
    assert str(library_directory / "libcudart.so") in plan.argv
    assert all("torch" not in argument.lower() for argument in plan.argv)
    assert all("cublas" not in argument.lower() for argument in plan.argv)
