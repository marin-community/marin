# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import functools
import gzip
from pathlib import Path

import pytest

from shuttle.experimental.stablehlo_import import import_stablehlo
from tile_lifetime.jax_streaming_attention_backward_ffi import (
    StreamingAttentionBackwardFfiBufferLayout,
    StreamingAttentionBackwardStatePolicy,
    generate_streaming_attention_backward_ffi,
)
from tile_lifetime.jax_streaming_attention_forward_ffi import generate_streaming_attention_forward_ffi
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.stablehlo_streaming_attention_backward import (
    recover_experimental_whole_pattern_streaming_attention_backward,
)
from tile_lifetime.streaming_attention import StreamingTileSchedule
from tile_lifetime.streaming_attention_backward import (
    StreamingAttentionBackwardDomainTraversal,
    StreamingAttentionBackwardProvenance,
    derive_streaming_attention_backward_tile_schedule,
    eliminate_normalized_exp_maximum_vjp,
)
from tile_lifetime.streaming_attention_backward_reference import (
    STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
    StreamingAttentionBackwardDebugConfig,
    export_debug_streaming_attention_backward,
)
from tile_lifetime.xla_hlo_recovery import parse_hlo_module_text
from tile_lifetime.xla_streaming_attention_training_regions import (
    audit_streaming_attention_training_region_replacement,
    plan_streaming_attention_training_regions,
    replace_streaming_attention_training_regions_with_custom_calls,
)

_GRUG_GPU_HLO = (
    Path(__file__).parents[1]
    / "benchmarks/artifacts/xla_grug_routed_combined_gpu_gb200_v0/original-gpu-pre-scheduler-hlo.txt.gz"
)


@functools.cache
def _generated_pair(scale: float = 0.32421875):
    config = StreamingAttentionBackwardDebugConfig(
        batch=2,
        query_length=4,
        key_length=4,
        query_heads=2,
        key_value_heads=1,
        head_dimension=16,
        scale=scale,
    )
    graph = import_stablehlo(
        export_debug_streaming_attention_backward(config),
        input_names=STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
    )
    recovered = recover_experimental_whole_pattern_streaming_attention_backward(
        graph,
        schedule=StreamingTileSchedule(query_tile_size=4, key_value_tile_size=4, pipeline_depth=2),
    )
    program = eliminate_normalized_exp_maximum_vjp(
        recovered.program,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    schedule = derive_streaming_attention_backward_tile_schedule(
        program,
        query_tile_size=4,
        key_value_tile_size=4,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    forward = generate_streaming_attention_forward_ffi(
        program,
        schedule,
        target_name="shuttle.streaming_forward.grug_split_test",
    )
    reverse = generate_streaming_attention_backward_ffi(
        program,
        schedule,
        target_name="shuttle.streaming_saved_reverse.grug_split_test",
        state_policy=StreamingAttentionBackwardStatePolicy.SAVED_OUTPUT_AND_LOG_SUM_EXP,
    )
    return recovered, program, forward, reverse


def test_natural_jax_vjp_drives_two_generic_generated_families() -> None:
    recovered, _, forward, reverse = _generated_pair()

    assert recovered.program.provenance is StreamingAttentionBackwardProvenance.JAX_VJP_HLO_RECOVERY
    assert tuple(value.name for value in forward.outputs) == ("output", "log_sum_exp")
    assert tuple(value.name for value in reverse.inputs) == (
        "query",
        "key",
        "value",
        "output",
        "log_sum_exp",
        "output_cotangent",
    )
    assert tuple(kernel.kernel_name for kernel in reverse.aot_kernels) == (
        "_streaming_dq_kernel",
        "_streaming_dkdv_kernel",
    )


def test_grug_post_spmd_hlo_proves_early_forward_remat_elision_and_later_reverse() -> None:
    _, program, forward, reverse = _generated_pair()
    hlo = gzip.decompress(_GRUG_GPU_HLO.read_bytes()).decode()

    plan = plan_streaming_attention_training_regions(hlo, program, forward, reverse)

    assert plan.forward.provenance.score_contract == "dot.16"
    assert plan.forward.provenance.maximum_fold == "reduce_max.56"
    assert plan.forward.provenance.value_contract == "dot.17"
    assert plan.forward.output.instruction == "transpose.47"
    assert plan.rematerialized_forward.provenance.score_contract == "dot.31"
    assert plan.rematerialized_forward.output.instruction == "transpose.54"
    assert plan.reverse.provenance.score_contract == "dot.31"
    assert plan.saved_state_policy is StreamingAttentionBackwardStatePolicy.SAVED_OUTPUT_AND_LOG_SUM_EXP
    assert plan.collectives_inside_regions == ()
    assert plan.forward.saved_state.ffi_shape == "f32[2,2,4]{2,1,0}"


def test_grug_split_rewrite_links_saved_state_and_leaves_collectives_external() -> None:
    _, program, forward, reverse = _generated_pair()
    hlo = gzip.decompress(_GRUG_GPU_HLO.read_bytes()).decode()
    plan = plan_streaming_attention_training_regions(hlo, program, forward, reverse)

    transformed = replace_streaming_attention_training_regions_with_custom_calls(hlo, plan)
    audit = audit_streaming_attention_training_region_replacement(hlo, transformed, plan)
    entry = parse_hlo_module_text(transformed).computation(parse_hlo_module_text(transformed).entry)
    source_order = {instruction.name: index for index, instruction in enumerate(entry.instructions)}
    reverse_call = next(instruction for instruction in entry.instructions if instruction.name == audit.reverse_call)

    assert source_order[audit.forward_call] < source_order[audit.reverse_call]
    assert audit.saved_state_producer == "shuttle.streaming_forward.log_sum_exp.ffi"
    assert audit.reverse_saved_state_operands == (
        "shuttle.streaming_forward.output.ffi",
        "shuttle.streaming_forward.log_sum_exp.ffi",
    )
    assert all(value in reverse_call.operands for value in audit.reverse_saved_state_operands)
    assert {"dot.16", "reduce_max.56", "dot.17"} <= set(audit.dead_forward_closure)
    assert {"dot.31", "reduce_max.64", "dot.32"} <= set(audit.dead_rematerialized_forward_closure)
    assert audit.external_collectives


def test_pair_rejects_saved_state_layout_mismatch_before_hlo_rewrite() -> None:
    _, program, _, reverse = _generated_pair()
    schedule = derive_streaming_attention_backward_tile_schedule(
        program,
        query_tile_size=4,
        key_value_tile_size=4,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    forward = generate_streaming_attention_forward_ffi(
        program,
        schedule,
        target_name="shuttle.streaming_forward.layout_mismatch",
        output_layouts=(
            StreamingAttentionBackwardFfiBufferLayout("output", (1, 3, 2, 0)),
            StreamingAttentionBackwardFfiBufferLayout("log_sum_exp", (2, 1, 0)),
        ),
    )
    hlo = gzip.decompress(_GRUG_GPU_HLO.read_bytes()).decode()

    with pytest.raises(ValueError, match="state ABI differs for output"):
        plan_streaming_attention_training_regions(hlo, program, forward, reverse)
