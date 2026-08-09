# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import re
from pathlib import Path

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

    input_gradient = chains_by_role[ContractChainRole.INPUT_GRADIENT]
    assert "concatenate" in input_gradient.map.opcodes
    assert "multiply" in input_gradient.map.opcodes
    assert input_gradient.first.output_shape == input_gradient.second.output_shape == "f32[512,32]{1,0}"

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
