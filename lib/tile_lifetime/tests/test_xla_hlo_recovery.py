# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
from pathlib import Path

from lib.tile_lifetime.benchmarks.xla_pair_map_custom_call_smoke import (
    MultiOutputFixedShapeProgram,
    generate_cuda_multi_output_ffi_handler,
    pair_map_recovery_diagnostic,
)
from tile_lifetime.xla_hlo_recovery import (
    form_pair_map_entry_region,
    inline_elementwise_fusions,
    parse_hlo_module_text,
    recover_pair_map_regions,
)

_SYNTHETIC_PAIR_MAP = """\
HloModule synthetic

%left_contract (activation: f32[8,32], weight: f32[32,64]) -> f32[8,64] {
  %activation = f32[8,32]{1,0} parameter(0)
  %weight = f32[32,64]{1,0} parameter(1)
  ROOT %result = f32[8,64]{1,0} dot(%activation, %weight), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

%right_contract (activation: f32[8,32], weight: f32[32,64]) -> f32[8,64] {
  %activation = f32[8,32]{1,0} parameter(0)
  %weight = f32[32,64]{1,0} parameter(1)
  ROOT %result = f32[8,64]{1,0} dot(%activation, %weight), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

%scalar_body (left: f32[8,64], right: f32[8,64]) -> bf16[8,64] {
  %left = f32[8,64]{1,0} parameter(0)
  %right = f32[8,64]{1,0} parameter(1)
  %low = bf16[8,64]{1,0} convert(%left)
  %wide = f32[8,64]{1,0} convert(%low)
  %nonlinear = f32[8,64]{1,0} tanh(%wide)
  %product = f32[8,64]{1,0} multiply(%nonlinear, %right)
  ROOT %result = bf16[8,64]{1,0} convert(%product)
}

ENTRY %main (arg: f32[8,32], lhs_weight: f32[32,64], rhs_weight: f32[32,64], down_weight: f32[64,32]) -> f32[8,32] {
  %arg = f32[8,32]{1,0} parameter(0)
  %lhs_weight = f32[32,64]{1,0} parameter(1)
  %rhs_weight = f32[32,64]{1,0} parameter(2)
  %down_weight = f32[64,32]{1,0} parameter(3)
  %lhs = f32[8,64]{1,0} fusion(%arg, %lhs_weight), kind=kCustom, calls=%left_contract
  %rhs = f32[8,64]{1,0} fusion(%arg, %rhs_weight), kind=kCustom, calls=%right_contract
  %mapped = bf16[8,64]{1,0} fusion(%lhs, %rhs), kind=kLoop, calls=%scalar_body
  %mapped_wide = f32[8,64]{1,0} convert(%mapped)
  ROOT %down = f32[8,32]{1,0} dot(%mapped_wide, %down_weight), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
"""


def test_fusion_inlining_exposes_contract_and_pointwise_operations() -> None:
    module = parse_hlo_module_text(_SYNTHETIC_PAIR_MAP)
    graph = inline_elementwise_fusions(module)

    assert module.entry == "main"
    assert sum(node.opcode == "dot" for node in graph.nodes) == 3
    assert all(node.opcode != "fusion" for node in graph.nodes)
    mapped = graph.node(graph.entry_value("mapped"))
    assert mapped.opcode == "convert"
    assert mapped.dtype == "bf16"
    assert mapped.attributes.startswith("convert(")


def test_pair_map_recovery_uses_structure_and_preserves_cast_boundaries() -> None:
    report = recover_pair_map_regions(_SYNTHETIC_PAIR_MAP)

    assert report.contract_count == 3
    assert len(report.regions) == 1
    region = report.regions[0]
    assert region.left_contract.output_shape == "f32[8,64]{1,0}"
    assert region.right_contract.output_shape == "f32[8,64]{1,0}"
    assert region.map_opcodes == ("convert", "convert", "tanh", "multiply", "convert", "convert")
    assert tuple(boundary.changes_dtype for boundary in region.map_cast_boundaries) == (True, True, True, True)
    assert len(region.consumer_contracts) == 1


def test_pair_map_recovery_tracks_scalar_body_mutation_without_a_named_pattern() -> None:
    mutated_hlo = _SYNTHETIC_PAIR_MAP.replace("tanh(%wide)", "exponential(%wide)")

    region = recover_pair_map_regions(mutated_hlo).regions[0]

    assert "exponential" in region.map_opcodes
    assert "tanh" not in region.map_opcodes


def test_frozen_grug_hlo_recovers_pair_map_without_source_names() -> None:
    artifact = (
        Path(__file__).parents[1]
        / "benchmarks/artifacts/grug_moe_train_step_pre_scheduler_jax011_v0/pre-scheduler-hlo.txt.gz"
    )
    hlo_text = gzip.decompress(artifact.read_bytes()).decode()
    report = recover_pair_map_regions(hlo_text)

    assert report.contract_count == 82
    assert len(report.regions) == 2
    for region in report.regions:
        assert region.left_contract.output_shape == "f32[8,32]{1,0}"
        assert region.right_contract.output_shape == "f32[8,32]{1,0}"
        assert "exponential" in region.map_opcodes
        assert "divide" in region.map_opcodes
        assert "multiply" in region.map_opcodes
        assert any(boundary.changes_dtype for boundary in region.map_cast_boundaries)
        assert len(region.consumer_contracts) == 1


def test_frozen_grug_backward_pair_map_forms_generic_multi_output_boundary() -> None:
    artifact = (
        Path(__file__).parents[1]
        / "benchmarks/artifacts/grug_moe_train_step_pre_scheduler_jax011_v0/pre-scheduler-hlo.txt.gz"
    )
    hlo_text = gzip.decompress(artifact.read_bytes()).decode()
    regions = recover_pair_map_regions(hlo_text).regions
    boundaries = tuple(form_pair_map_entry_region(hlo_text, region) for region in regions)
    multi_output = tuple(boundary for boundary in boundaries if len(boundary.outputs) > 1)

    assert len(multi_output) == 1
    boundary = multi_output[0]
    assert len(boundary.inputs) == 4
    assert tuple(value.shape for value in boundary.inputs) == (
        "f32[8,32]{1,0}",
        "f32[32,32]{1,0}",
        "f32[32,32]{1,0}",
        "bf16[8,32]{1,0}",
    )
    assert len(boundary.internal_instructions) == 9
    assert tuple(value.shape for value in boundary.outputs) == ("f32[8,32]{1,0}",) * 3
    assert tuple(len(users) for _, users in boundary.external_users) == (2, 2, 1)
    assert not boundary.has_explicit_sharding
    assert not boundary.has_side_effect


def test_cuda_multi_output_ffi_uses_one_generic_thread_local_scalar_body() -> None:
    program = MultiOutputFixedShapeProgram(
        rows=8,
        reduction=32,
        features=32,
        scalar_expressions=(
            "shuttle_round_bf16(projection0[row * kFeatures + feature] * "
            "shuttle_bf16_to_f32(cotangent[row * kFeatures + feature]))",
            "projection1[row * kFeatures + feature]",
        ),
    )

    source = generate_cuda_multi_output_ffi_handler(program)

    assert ".Ctx<ffi::PlatformStream<cudaStream_t>>()" in source
    assert source.count("cublasGemmEx(") == 1
    assert "CUBLAS_COMPUTE_32F_PEDANTIC" in source
    assert "left * shuttle_bf16_to_f32(cotangent[index])" in source
    assert "output1[index] = right;" in source
    assert "projection0[kRows" not in source
    assert "projection1[kRows" not in source
    assert "for (int reduction" not in source


def test_pair_map_recovery_diagnostic_records_candidates_before_selection() -> None:
    diagnostic = pair_map_recovery_diagnostic(_SYNTHETIC_PAIR_MAP)

    assert diagnostic["contract_count"] == 3
    assert diagnostic["pair_map_region_count"] == 1
    candidate = diagnostic["candidates"][0]
    assert candidate["index"] == 0
    assert candidate["output_shapes"] == ("f32[8,64]{1,0}",)
    assert candidate["external_user_counts"] == (1,)
