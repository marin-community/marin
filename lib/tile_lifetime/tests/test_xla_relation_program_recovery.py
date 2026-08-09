# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import re
from pathlib import Path

import jax.numpy as jnp
import pytest

from tile_lifetime.cast_scalar_program import (
    CastScalarExpression,
    CastScalarKind,
    evaluate_cast_scalar_program,
)
from tile_lifetime.xla_relation_program_recovery import ContractChainRole, recover_relation_programs

_ARTIFACT = (
    Path(__file__).parents[1]
    / "benchmarks/artifacts/grug_moe_train_step_pre_scheduler_jax011_v0/pre-scheduler-hlo.txt.gz"
)


def _frozen_hlo() -> str:
    return gzip.decompress(_ARTIFACT.read_bytes()).decode()


def test_grug_hlo_recovers_generic_routed_forward_and_backward_program() -> None:
    report = recover_relation_programs(_frozen_hlo())

    assert len(report.relation_plans) == 2
    assert {
        (plan.token_count, plan.slots_per_token, plan.edge_count, plan.destination_count)
        for plan in report.relation_plans
    } == {(8, 2, 16, 4)}
    assert all(plan.destination_sort != plan.selection for plan in report.relation_plans)
    assert all(plan.destination_counts != plan.destination_offsets for plan in report.relation_plans)

    assert len(report.contract_chains) == 3
    chains_by_role = {chain.role: chain for chain in report.contract_chains}
    assert set(chains_by_role) == {
        ContractChainRole.FORWARD,
        ContractChainRole.FORWARD_RECOMPUTE,
        ContractChainRole.INPUT_GRADIENT,
    }
    forward = chains_by_role[ContractChainRole.FORWARD]
    assert forward.first.output_shape == "f32[512,64]{1,0}"
    assert forward.second.output_shape == "f32[512,32]{1,0}"
    assert "slice" in forward.map.opcodes
    assert "multiply" in forward.map.opcodes
    assert any(source.startswith("f32") and result.startswith("bf16") for source, result in forward.map.cast_shapes)
    assert forward.map.scalar_program is not None
    assert forward.map.generated_cuda is not None
    assert tuple(
        (value.input_index.row_offset, value.input_index.feature_offset)
        for value in forward.map.scalar_program.inputs
        if value.input_index is not None
    ) == ((0, 0), (0, 32))
    assert _count_kind(forward.map.scalar_program.expression, CastScalarKind.CONVERT) == 30
    assert "__float2bfloat16_rn" in forward.map.generated_cuda.source
    assert "MoE" not in forward.map.generated_cuda.source
    recompute = chains_by_role[ContractChainRole.FORWARD_RECOMPUTE]
    assert recompute.map.scalar_program is not None
    assert recompute.map.scalar_program.digest == forward.map.scalar_program.digest

    input_gradient = chains_by_role[ContractChainRole.INPUT_GRADIENT]
    assert "concatenate" in input_gradient.map.opcodes
    assert "multiply" in input_gradient.map.opcodes
    assert input_gradient.first.output_shape == input_gradient.second.output_shape == "f32[512,32]{1,0}"
    assert input_gradient.map.scalar_program is None
    assert input_gradient.map.generated_cuda is None

    assert len(report.folds) == 2
    assert {fold.reducer for fold in report.folds} == {"add"}
    assert {fold.output_shape for fold in report.folds} == {"f32[8,32]{1,0}"}
    assert all(fold.reducer_opcodes == ("add", "convert", "convert") for fold in report.folds)
    assert any("multiply" in fold.contribution_opcodes for fold in report.folds)

    assert {gradient.contract.output_shape for gradient in report.weight_gradients} == {
        "f32[4,32,64]{2,1,0}",
        "f32[4,32,32]{2,1,0}",
    }
    assert all(gradient.external_collectives for gradient in report.weight_gradients)
    assert report.external_collectives


def test_grug_hlo_relation_recovery_ignores_frontend_metadata() -> None:
    hlo = _frozen_hlo()
    renamed_metadata = re.sub(r'op_name="[^"]*"', 'op_name="unrelated_diagnostic_label"', hlo)

    baseline = recover_relation_programs(hlo).to_dict()
    renamed = recover_relation_programs(renamed_metadata).to_dict()

    assert renamed == baseline


def test_grug_hlo_forward_map_ast_preserves_source_bf16_boundaries() -> None:
    forward = next(
        chain
        for chain in recover_relation_programs(_frozen_hlo()).contract_chains
        if chain.role is ContractChainRole.FORWARD
    )
    assert forward.map.scalar_program is not None
    left = 0.78125
    right = -1.3125

    def bf16(value):
        return jnp.asarray(value, dtype=jnp.bfloat16).astype(jnp.float32)

    left_source = bf16(bf16(bf16(jnp.float32(left))))
    right_source = bf16(bf16(bf16(jnp.float32(right))))
    denominator = bf16(jnp.exp(bf16(-left_source)))
    denominator = bf16(denominator + jnp.float32(1.0))
    sigmoid = bf16(jnp.float32(1.0) / denominator)
    activated = bf16(left_source * sigmoid)
    expected = bf16(activated * right_source)

    observed = evaluate_cast_scalar_program(
        forward.map.scalar_program,
        {"input_r0_f0": left, "input_r0_f32": right},
    )

    assert float(observed) == pytest.approx(float(expected), abs=0.0)


def test_grug_hlo_forward_map_mutation_regenerates_scalar_cuda() -> None:
    baseline = next(
        chain
        for chain in recover_relation_programs(_frozen_hlo()).contract_chains
        if chain.role is ContractChainRole.FORWARD
    )
    mutated_hlo = _frozen_hlo().replace(
        "multiply(%convert.3758, %convert.3756)",
        "add(%convert.3758, %convert.3756)",
        1,
    )
    mutated = next(
        chain
        for chain in recover_relation_programs(mutated_hlo).contract_chains
        if chain.role is ContractChainRole.FORWARD
    )
    assert baseline.map.scalar_program is not None and mutated.map.scalar_program is not None
    assert baseline.map.generated_cuda is not None and mutated.map.generated_cuda is not None

    assert mutated.map.scalar_program.digest != baseline.map.scalar_program.digest
    assert mutated.map.generated_cuda.source_digest != baseline.map.generated_cuda.source_digest
    assert _count_kind(mutated.map.scalar_program.expression, CastScalarKind.ADD) == (
        _count_kind(baseline.map.scalar_program.expression, CastScalarKind.ADD) + 1
    )
    assert _count_kind(mutated.map.scalar_program.expression, CastScalarKind.MULTIPLY) == (
        _count_kind(baseline.map.scalar_program.expression, CastScalarKind.MULTIPLY) - 1
    )


def _count_kind(expression: CastScalarExpression, kind: CastScalarKind) -> int:
    return int(expression.kind is kind) + sum(_count_kind(operand, kind) for operand in expression.operands)
