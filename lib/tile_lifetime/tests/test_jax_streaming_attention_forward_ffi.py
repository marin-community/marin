# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import functools

import jax.numpy as jnp
import pytest

from tile_lifetime.jax_streaming_attention_forward_ffi import (
    call_streaming_attention_forward_ffi,
    generate_streaming_attention_forward_ffi,
)
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.stablehlo_import import import_stablehlo
from tile_lifetime.stablehlo_streaming_attention_backward import recover_stablehlo_streaming_attention_backward
from tile_lifetime.streaming_attention import StreamingTileSchedule
from tile_lifetime.streaming_attention_backward import (
    StreamingAttentionBackwardDomainTraversal,
    derive_streaming_attention_backward_tile_schedule,
    eliminate_normalized_exp_maximum_vjp,
)
from tile_lifetime.streaming_attention_backward_reference import (
    STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
    StreamingAttentionBackwardDebugConfig,
    export_debug_streaming_attention_backward,
)


@functools.cache
def _program_and_schedule(scale: float):
    config = StreamingAttentionBackwardDebugConfig(
        batch=1,
        query_length=64,
        key_length=64,
        query_heads=4,
        key_value_heads=2,
        head_dimension=64,
        scale=scale,
    )
    graph = import_stablehlo(
        export_debug_streaming_attention_backward(config),
        input_names=STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
    )
    recovered = recover_stablehlo_streaming_attention_backward(
        graph,
        schedule=StreamingTileSchedule(query_tile_size=32, key_value_tile_size=32, pipeline_depth=3),
    )
    program = eliminate_normalized_exp_maximum_vjp(
        recovered.program,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    schedule = derive_streaming_attention_backward_tile_schedule(
        program,
        query_tile_size=32,
        key_value_tile_size=32,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    return program, schedule


def test_forward_family_exposes_only_qkv_output_and_minimal_fold_state() -> None:
    program, schedule = _program_and_schedule(0.5)

    generated = generate_streaming_attention_forward_ffi(
        program,
        schedule,
        target_name="shuttle.streaming_forward.static_test",
    )

    assert tuple(value.name for value in generated.inputs) == ("query", "key", "value")
    assert tuple((value.name, value.shape, value.dtype.value) for value in generated.outputs) == (
        ("output", (1, 64, 4, 64), "bf16"),
        ("log_sum_exp", (1, 4, 64), "fp32"),
    )
    assert generated.aot_kernel.kernel_name == "_streaming_grouped_query_forward"
    assert "torch" not in generated.handler_template.lower()
    assert "triton" not in generated.handler_template.lower()
    assert "log_sum_exp_pointer" in generated.handler_template


def test_scale_mutation_regenerates_same_forward_family() -> None:
    baseline_program, baseline_schedule = _program_and_schedule(0.5)
    mutated_program, mutated_schedule = _program_and_schedule(0.375)

    baseline = generate_streaming_attention_forward_ffi(
        baseline_program,
        baseline_schedule,
        target_name="shuttle.streaming_forward.scale_baseline",
    )
    mutated = generate_streaming_attention_forward_ffi(
        mutated_program,
        mutated_schedule,
        target_name="shuttle.streaming_forward.scale_mutation",
    )

    assert baseline.inputs == mutated.inputs
    assert baseline.outputs == mutated.outputs
    assert baseline.aot_kernel.kernel_name == mutated.aot_kernel.kernel_name
    assert baseline.semantic_fingerprint != mutated.semantic_fingerprint
    assert baseline.aot_kernel.signature != mutated.aot_kernel.signature


def test_forward_call_rejects_wrong_shape_before_dispatch() -> None:
    program, schedule = _program_and_schedule(0.5)
    generated = generate_streaming_attention_forward_ffi(
        program,
        schedule,
        target_name="shuttle.streaming_forward.shape_validation",
    )

    with pytest.raises(ValueError, match=r"query.*shape"):
        call_streaming_attention_forward_ffi(
            generated,
            query=jnp.zeros((1, 32, 4, 64), dtype=jnp.bfloat16),
            key=jnp.zeros(generated.inputs[1].shape, dtype=jnp.bfloat16),
            value=jnp.zeros(generated.inputs[2].shape, dtype=jnp.bfloat16),
        )
