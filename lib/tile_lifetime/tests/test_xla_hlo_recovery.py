# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
from pathlib import Path

from lib.tile_lifetime.benchmarks.xla_grug_backward_multi_output_gpu_custom_call_smoke import (
    _recover_gpu_region_rewrite,
)
from lib.tile_lifetime.benchmarks.xla_pair_map_custom_call_smoke import (
    _TARGET_NAME,
    ContractMapRegionRewrite,
    MultiOutputFixedShapeProgram,
    contract_map_recovery_diagnostic,
    generate_cuda_contract_map_ffi_handler,
    generate_cuda_multi_output_ffi_handler,
    pair_map_recovery_diagnostic,
    recover_contract_map_region_rewrite,
    replace_multi_output_region_with_custom_call,
)
from tile_lifetime.xla_hlo_recovery import (
    form_pair_map_entry_region,
    inline_elementwise_fusions,
    parse_hlo_module_text,
    recover_aligned_demand_sliced_contracts,
    recover_multi_output_contract_map_regions,
    recover_pair_map_regions,
)
from tile_lifetime.xla_low_rank_gated_product_ffi import (
    plan_generated_low_rank_contract_map_training,
    replace_generated_low_rank_contract_map_training,
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

_SYNTHETIC_REVERSE_CONTRACT_MAP = """\
HloModule synthetic_reverse

ENTRY %main (cotangent: f32[8,32], \
weight: f32[32,64], saved_left: f32[8,64], \
saved_right: f32[8,64], left_weight: f32[64,16], \
right_weight: f32[64,16]) -> (f32[8,16], f32[8,16]) {
  %cotangent = f32[8,32]{1,0} parameter(0)
  %weight = f32[32,64]{1,0} parameter(1)
  %saved_left = f32[8,64]{1,0} parameter(2)
  %saved_right = f32[8,64]{1,0} parameter(3)
  %left_weight = f32[64,16]{1,0} parameter(4)
  %right_weight = f32[64,16]{1,0} parameter(5)
  %projected = f32[8,64]{1,0} dot(%cotangent, %weight), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %left_map = f32[8,64]{1,0} multiply(%projected, %saved_left)
  %right_map = f32[8,64]{1,0} multiply(%projected, %saved_right)
  %left_result = f32[8,16]{1,0} dot(%left_map, %left_weight), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %right_result = f32[8,16]{1,0} dot(%right_map, %right_weight), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT %result = (f32[8,16]{1,0}, f32[8,16]{1,0}) tuple(%left_result, %right_result)
}
"""

_SYNTHETIC_DEMAND_SLICED_CONTRACT = """\
HloModule synthetic_demand_sliced

ENTRY %main (activation: bf16[2,3], left: bf16[2,3], middle: bf16[1,3], right: bf16[3,3]) \
-> (bf16[2,2], bf16[2,1], bf16[2,3]) {
  %activation = bf16[2,3]{1,0} parameter(0)
  %left = bf16[2,3]{0,1} parameter(1)
  %middle = bf16[1,3]{0,1} parameter(2)
  %right = bf16[3,3]{0,1} parameter(3)
  %weights = bf16[6,3]{0,1} concatenate(%left, %middle, %right), dimensions={0}
  %projection = bf16[2,6]{1,0} dot(%activation, %weights), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  %left_result = bf16[2,2]{1,0} slice(%projection), slice={[0:2], [0:2]}
  %middle_result = bf16[2,1]{1,0} slice(%projection), slice={[0:2], [2:3]}
  %right_result = bf16[2,3]{1,0} slice(%projection), slice={[0:2], [3:6]}
  ROOT %result = (bf16[2,2]{1,0}, bf16[2,1]{1,0}, bf16[2,3]{1,0}) tuple(%left_result, %middle_result, %right_result)
}
"""


def _post_gated_product_grug_hlo() -> str:
    artifact = (
        Path(__file__).parents[1]
        / "benchmarks/artifacts/xla_grug_shared_map_h100_fused_reverses_unaccepted_e3411679_v0/"
        "transformed-gpu-pre-scheduler-hlo.txt.gz"
    )
    hlo = gzip.decompress(artifact.read_bytes()).decode()
    plan = plan_generated_low_rank_contract_map_training(
        hlo,
        forward_target_prefix="shuttle.generic.low_rank_contract_map.generated.forward.test",
        reverse_target_prefix="shuttle.generic.low_rank_contract_map.generated.reverse.test",
    )
    return replace_generated_low_rank_contract_map_training(hlo, plan)


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


def test_demand_sliced_contract_recovery_aligns_concat_parts_without_named_semantics() -> None:
    report = recover_aligned_demand_sliced_contracts(_SYNTHETIC_DEMAND_SLICED_CONTRACT)

    assert report.live_contract_count == 1
    assert report.live_contract_flops == 72
    assert report.covered_contract_count == 1
    assert report.covered_contract_flops == 72
    region = report.regions[0]
    assert region.concatenated_operand == 1
    assert region.concatenation_axis == 0
    assert region.output_partition_axis == 1
    assert region.reduction_order_unchanged
    assert tuple((part.source_start, part.source_limit) for part in region.partitions) == ((0, 2), (2, 3), (3, 6))
    assert tuple((part.output_start, part.output_limit) for part in region.partitions) == ((0, 2), (2, 3), (3, 6))
    assert tuple(part.external_users for part in region.partitions) == (("result",),) * 3


def test_demand_sliced_contract_recovery_tracks_partition_mutation_through_same_plan() -> None:
    mutated = (
        _SYNTHETIC_DEMAND_SLICED_CONTRACT.replace("left: bf16[2,3]", "left: bf16[1,3]")
        .replace("middle: bf16[1,3]", "middle: bf16[2,3]")
        .replace("%left = bf16[2,3]{0,1}", "%left = bf16[1,3]{0,1}")
        .replace("%middle = bf16[1,3]{0,1}", "%middle = bf16[2,3]{0,1}")
        .replace(
            "%left_result = bf16[2,2]{1,0} slice(%projection), slice={[0:2], [0:2]}",
            "%left_result = bf16[2,1]{1,0} slice(%projection), slice={[0:2], [0:1]}",
        )
        .replace(
            "%middle_result = bf16[2,1]{1,0} slice(%projection), slice={[0:2], [2:3]}",
            "%middle_result = bf16[2,2]{1,0} slice(%projection), slice={[0:2], [1:3]}",
        )
        .replace(
            "(bf16[2,2]{1,0}, bf16[2,1]{1,0}, bf16[2,3]{1,0}) tuple",
            "(bf16[2,1]{1,0}, bf16[2,2]{1,0}, bf16[2,3]{1,0}) tuple",
        )
    )

    original = recover_aligned_demand_sliced_contracts(_SYNTHETIC_DEMAND_SLICED_CONTRACT).regions[0]
    changed = recover_aligned_demand_sliced_contracts(mutated).regions[0]

    assert changed.contract == original.contract
    assert changed.flops == original.flops
    assert tuple((part.source_start, part.source_limit) for part in changed.partitions) == ((0, 1), (1, 3), (3, 6))
    assert tuple((part.output_start, part.output_limit) for part in changed.partitions) == ((0, 1), (1, 3), (3, 6))


def test_demand_sliced_contract_recovery_rejects_non_aligned_output_demand() -> None:
    non_aligned = _SYNTHETIC_DEMAND_SLICED_CONTRACT.replace(
        "%left_result = bf16[2,2]{1,0} slice(%projection), slice={[0:2], [0:2]}",
        "%left_result = bf16[2,1]{1,0} slice(%projection), slice={[0:2], [0:1]}",
    )

    assert not recover_aligned_demand_sliced_contracts(non_aligned).regions


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


def test_post_gated_product_grug_hlo_recovers_aligned_demand_sliced_contracts() -> None:
    report = recover_aligned_demand_sliced_contracts(_post_gated_product_grug_hlo())

    assert report.live_contract_count == 24
    assert report.live_contract_flops == 397_312
    assert report.covered_contract_count == 6
    assert report.covered_contract_flops == 205_824
    assert all(region.concatenated_operand == 1 for region in report.regions)
    assert all(region.concatenation_axis == 0 for region in report.regions)
    assert tuple(region.output_partition_axis for region in report.regions) == (2, 2, 2, 2, 1, 1)
    assert tuple(len(region.partitions) for region in report.regions) == (4, 3, 4, 3, 4, 3)
    assert all(region.reduction_order_unchanged for region in report.regions)
    assert tuple(region.contract.output_shape for region in report.regions[:4]) == (
        "bf16[2,4,66]{2,1,0}",
        "bf16[2,4,68]{2,1,0}",
        "bf16[2,4,66]{2,1,0}",
        "bf16[2,4,68]{2,1,0}",
    )
    assert tuple(region.contract.output_shape for region in report.regions[4:]) == (
        "bf16[32,66]{1,0}",
        "bf16[32,68]{1,0}",
    )


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


def test_reverse_contract_map_recovery_exposes_two_generated_map_results() -> None:
    report = recover_multi_output_contract_map_regions(_SYNTHETIC_REVERSE_CONTRACT_MAP)

    assert report.contract_count == 3
    assert len(report.regions) == 1
    region = report.regions[0]
    assert region.contract.output_shape == "f32[8,64]{1,0}"
    assert region.map_opcodes == ("multiply", "multiply")
    assert tuple(value.shape for value in region.boundary.outputs) == ("f32[8,64]{1,0}",) * 2
    assert len(region.consumer_contracts) == 2


def test_reverse_contract_map_rewrite_generates_from_structure_and_tracks_mutation() -> None:
    rewrite = recover_contract_map_region_rewrite(_SYNTHETIC_REVERSE_CONTRACT_MAP, 0)

    assert rewrite.program.rows == 8
    assert rewrite.program.reduction == 32
    assert rewrite.program.features == 64
    assert rewrite.program.contract_lhs_input == 0
    assert rewrite.program.contract_rhs_input == 1
    assert rewrite.program.rhs_contracting_dimension == 0
    assert rewrite.program.scalar_expressions == (
        "((projection_value) * (input2[index]))",
        "((projection_value) * (input3[index]))",
    )

    mutated = _SYNTHETIC_REVERSE_CONTRACT_MAP.replace(
        "%right_map = f32[8,64]{1,0} multiply(%projected, %saved_right)",
        "%right_map = f32[8,64]{1,0} add(%projected, %saved_right)",
    )
    mutated_rewrite = recover_contract_map_region_rewrite(mutated, 0)
    assert mutated_rewrite.program.scalar_expressions[0] == rewrite.program.scalar_expressions[0]
    assert mutated_rewrite.program.scalar_expressions[1] == "((projection_value) + (input3[index]))"


def test_reverse_contract_map_rewrite_rewires_both_live_values() -> None:
    rewrite = recover_contract_map_region_rewrite(_SYNTHETIC_REVERSE_CONTRACT_MAP, 0)

    transformed = replace_multi_output_region_with_custom_call(
        _SYNTHETIC_REVERSE_CONTRACT_MAP,
        rewrite,
        _TARGET_NAME,
        typed_ffi=True,
    )

    assert transformed.count(_TARGET_NAME) == 1
    assert transformed.count("get-tuple-element(%shuttle_generated_multi_output_region)") == 2
    assert "custom-call(%cotangent, %weight, %saved_left, %saved_right)" in transformed


def test_gpu_grug_hlo_recovers_generic_reverse_contract_map_boundary() -> None:
    artifact = (
        Path(__file__).parents[1] / "benchmarks/artifacts/xla_grug_backward_multi_output_gpu_sm100_diagnostic_v0/"
        "original-gpu-pre-scheduler-hlo.txt.gz"
    )
    hlo_text = gzip.decompress(artifact.read_bytes()).decode()
    report = recover_multi_output_contract_map_regions(hlo_text)

    assert report.contract_count == 68
    assert len(report.regions) == 1
    region = report.regions[0]
    assert region.contract.output_shape == "bf16[8,32]{1,0}"
    assert region.map_opcodes == ("multiply",) * 5 + ("add",)
    assert tuple(value.shape for value in region.boundary.outputs) == ("bf16[8,32]{1,0}",) * 2
    assert len(region.boundary.inputs) == 7
    assert len(region.consumer_contracts) == 2


def test_gpu_grug_contract_map_rewrite_preserves_bf16_map_rounding() -> None:
    artifact = (
        Path(__file__).parents[1] / "benchmarks/artifacts/xla_grug_backward_multi_output_gpu_sm100_diagnostic_v0/"
        "original-gpu-pre-scheduler-hlo.txt.gz"
    )
    hlo_text = gzip.decompress(artifact.read_bytes()).decode()

    rewrite = recover_contract_map_region_rewrite(hlo_text, 0)
    selected = _recover_gpu_region_rewrite(hlo_text)
    assert isinstance(selected, ContractMapRegionRewrite)
    source = generate_cuda_contract_map_ffi_handler(selected.program)
    diagnostic = contract_map_recovery_diagnostic(hlo_text)

    assert rewrite.program.input_dtypes == ("bf16",) * 7
    assert tuple(value.instruction for value in selected.boundary.inputs) == (
        "mul.73",
        "reshape.227",
        "remat2.66",
    )
    assert tuple(value.instruction for value in selected.boundary.outputs) == ("dot_general.226", "mul.963")
    assert selected.program.scalar_expressions == (
        "projection_value",
        "shuttle_round_bf16(((shuttle_bf16_to_f32(input0[index])) * (projection_value)))",
    )
    assert selected.program.contract_lhs_input == 1
    assert selected.program.contract_rhs_input == 2
    assert rewrite.program.output_dtype == "bf16"
    assert rewrite.program.contract_lhs_input == 3
    assert rewrite.program.contract_rhs_input == 4
    assert rewrite.program.rhs_contracting_dimension == 1
    assert len(rewrite.program.scalar_expressions) == 2
    assert rewrite.program.scalar_expressions[0].startswith("shuttle_round_bf16")
    assert rewrite.program.scalar_expressions[1].count("shuttle_round_bf16") == 6
    assert "CUBLAS_OP_T" in source
    assert source.count("cublasGemmEx(") == 1
    assert "CUDA_R_16BF" in source
    assert "CUBLAS_COMPUTE_32F_PEDANTIC" in source
    assert "shuttle_f32_to_bf16" in source
    assert diagnostic["contract_map_region_count"] == 1
    assert "lowering_error" not in diagnostic["candidates"][0]


def test_gpu_grug_nonconvex_region_replacement_preserves_contract_for_later_maps() -> None:
    artifact = (
        Path(__file__).parents[1] / "benchmarks/artifacts/xla_grug_backward_multi_output_gpu_sm100_diagnostic_v0/"
        "original-gpu-pre-scheduler-hlo.txt.gz"
    )
    hlo_text = gzip.decompress(artifact.read_bytes()).decode()
    selected = _recover_gpu_region_rewrite(hlo_text)
    assert isinstance(selected, ContractMapRegionRewrite)

    transformed = replace_multi_output_region_with_custom_call(
        hlo_text,
        selected,
        _TARGET_NAME,
        typed_ffi=True,
    )

    call_position = transformed.index("%shuttle_generated_multi_output_region =")
    assert transformed.index("%remat2.66 =") < call_position
    assert call_position < transformed.index("%dot_general.226 =")
    assert transformed.count(_TARGET_NAME) == 1
    assert transformed.count("get-tuple-element(%shuttle_generated_multi_output_region)") == 2
    assert "%mul.965 = bf16[8,32]{1,0} multiply(%dot_general.226, %reshape.1030)" in transformed
