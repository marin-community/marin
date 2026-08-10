# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import math
import re
from dataclasses import replace
from pathlib import Path

import ml_dtypes
import pytest

from tile_lifetime.cast_scalar_program import CastScalarKind, evaluate_cast_scalar_program
from tile_lifetime.quack_partitioned_gemm_adapter import (
    QUACK_PARTITION_ADAPTER_BASE_REVISION,
    QuackPartitionFinalizationKind,
    plan_quack_partitioned_gemm_adapter,
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
