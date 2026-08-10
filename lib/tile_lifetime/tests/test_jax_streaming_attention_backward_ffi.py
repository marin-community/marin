# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import ast
import functools
import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

from tile_lifetime.jax_streaming_attention_backward_ffi import (
    StreamingAttentionBackwardResultPolicy,
    StreamingAttentionBackwardStatePolicy,
    StreamingAttentionLogSumExpEncoding,
    _run_triton_aot_compile,
    call_streaming_attention_backward_ffi,
    call_streaming_attention_training_ffi,
    compile_streaming_attention_backward_ffi,
    generate_streaming_attention_backward_ffi,
)
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.stablehlo_import import import_stablehlo
from tile_lifetime.stablehlo_streaming_attention_backward import (
    recover_experimental_whole_pattern_streaming_attention_backward,
)
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

REPOSITORY = Path(__file__).resolve().parents[3]


@functools.cache
def _program_and_schedule(scale: float = 0.5):
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
    recovered = recover_experimental_whole_pattern_streaming_attention_backward(
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


@functools.cache
def _short_program_and_schedule(head_dimension: int):
    config = StreamingAttentionBackwardDebugConfig(
        batch=2,
        query_length=4,
        key_length=4,
        query_heads=2,
        key_value_heads=1,
        head_dimension=head_dimension,
        scale=0.32421875,
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
    return program, schedule


def _function_argument_count(path: Path, function_name: str) -> int:
    module = ast.parse(path.read_text())
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    return len(function.args.args)


def test_recompute_plan_preserves_natural_vjp_signature_and_emits_three_aot_kernels() -> None:
    program, schedule = _program_and_schedule()

    generated = generate_streaming_attention_backward_ffi(
        program,
        schedule,
        target_name="shuttle.streaming_reverse.recompute_test_v1",
    )

    assert generated.state_policy is StreamingAttentionBackwardStatePolicy.RECOMPUTE
    assert generated.saved_state_encoding is None
    assert generated.result_policy is StreamingAttentionBackwardResultPolicy.GRADIENTS_ONLY
    assert tuple(value.name for value in generated.inputs) == ("query", "key", "value", "output_cotangent")
    assert tuple(value.name for value in generated.outputs) == (
        "query_cotangent",
        "key_cotangent",
        "value_cotangent",
    )
    assert tuple(kernel.kernel_name for kernel in generated.aot_kernels) == (
        "_streaming_grouped_query_forward",
        "_streaming_dq_kernel",
        "_streaming_dkdv_kernel",
    )
    assert all(kernel.signature[-1] == "0" for kernel in generated.aot_kernels)
    for kernel in generated.aot_kernels:
        assert len(kernel.signature) == _function_argument_count(REPOSITORY / kernel.source, kernel.kernel_name)
    assert "torch" not in generated.handler_template.lower()
    assert "triton" not in generated.handler_template.lower()
    assert "ScratchAllocator" in generated.handler_template
    assert "{forward_launcher}" in generated.handler_template


def test_training_plan_returns_recomputed_forward_output_without_output_scratch() -> None:
    program, schedule = _program_and_schedule()

    generated = generate_streaming_attention_backward_ffi(
        program,
        schedule,
        target_name="shuttle.streaming_training.recompute_test_v1",
        result_policy=StreamingAttentionBackwardResultPolicy.FORWARD_OUTPUT_AND_GRADIENTS,
    )

    assert generated.state_policy is StreamingAttentionBackwardStatePolicy.RECOMPUTE
    assert generated.result_policy is StreamingAttentionBackwardResultPolicy.FORWARD_OUTPUT_AND_GRADIENTS
    assert tuple(value.name for value in generated.outputs) == (
        "forward_output",
        "query_cotangent",
        "key_cotangent",
        "value_cotangent",
    )
    assert tuple(output.layout for output in generated.outputs) == ((3, 2, 1, 0),) * 4
    assert tuple(output.strides for output in generated.outputs) == (
        (16384, 256, 64, 1),
        (16384, 256, 64, 1),
        (8192, 128, 64, 1),
        (8192, 128, 64, 1),
    )
    forward, dq, dkdv = generated.aot_kernels
    assert forward.signature[26:30] == tuple(str(stride) for stride in generated.outputs[0].strides)
    assert dq.signature[27:31] == tuple(str(stride) for stride in generated.outputs[0].strides)
    assert dq.signature[35:39] == tuple(str(stride) for stride in generated.outputs[1].strides)
    assert dkdv.signature[31:35] == tuple(str(stride) for stride in generated.outputs[2].strides)
    assert dkdv.signature[35:39] == tuple(str(stride) for stride in generated.outputs[3].strides)
    assert tuple(kernel.kernel_name for kernel in generated.aot_kernels) == (
        "_streaming_grouped_query_forward",
        "_streaming_dq_kernel",
        "_streaming_dkdv_kernel",
    )


def test_training_and_reverse_calls_require_their_explicit_result_policy() -> None:
    program, schedule = _program_and_schedule()
    reverse = generate_streaming_attention_backward_ffi(
        program,
        schedule,
        target_name="shuttle.streaming_reverse.result_policy_test_v1",
    )
    training = generate_streaming_attention_backward_ffi(
        program,
        schedule,
        target_name="shuttle.streaming_training.result_policy_test_v1",
        result_policy=StreamingAttentionBackwardResultPolicy.FORWARD_OUTPUT_AND_GRADIENTS,
    )
    arguments = {
        "query": jnp.zeros(reverse.inputs[0].shape, dtype=jnp.bfloat16),
        "key": jnp.zeros(reverse.inputs[1].shape, dtype=jnp.bfloat16),
        "value": jnp.zeros(reverse.inputs[2].shape, dtype=jnp.bfloat16),
        "output_cotangent": jnp.zeros(reverse.inputs[3].shape, dtype=jnp.bfloat16),
    }

    with pytest.raises(ValueError, match="training call requires"):
        call_streaming_attention_training_ffi(reverse, **arguments)
    with pytest.raises(ValueError, match="reverse-only call requires"):
        call_streaming_attention_backward_ffi(training, **arguments)


def test_training_result_rejects_external_saved_state_boundary() -> None:
    program, schedule = _program_and_schedule()

    with pytest.raises(ValueError, match="recomputed forward state"):
        generate_streaming_attention_backward_ffi(
            program,
            schedule,
            target_name="shuttle.streaming_training.saved_state_rejected_v1",
            state_policy=StreamingAttentionBackwardStatePolicy.SAVED_OUTPUT_AND_LOG_SUM_EXP,
            result_policy=StreamingAttentionBackwardResultPolicy.FORWARD_OUTPUT_AND_GRADIENTS,
        )


def test_short_d16_plan_legalizes_only_physical_aot_tiles() -> None:
    program, schedule = _short_program_and_schedule(16)

    generated = generate_streaming_attention_backward_ffi(
        program,
        schedule,
        target_name="shuttle.streaming_reverse.short_d16_test_v1",
    )

    forward, query_cotangent, key_value_cotangents = generated.aot_kernels
    assert (schedule.query_tile_size, schedule.key_value_tile_size) == (4, 4)
    assert forward.signature[-9:-5] == ("16", "16", "16", "2")
    assert query_cotangent.signature[-7:-3] == ("16", "16", "16", "2")
    assert key_value_cotangents.signature[-7:-3] == ("16", "16", "16", "2")
    assert tuple(kernel.grid for kernel in generated.aot_kernels) == ((1, 2, 1), (1, 2, 1), (1, 2, 1))


def test_short_unsupported_head_dimension_fails_closed() -> None:
    program, schedule = _short_program_and_schedule(32)

    with pytest.raises(ValueError, match="equal head dimensions 16, 64, and 128"):
        generate_streaming_attention_backward_ffi(
            program,
            schedule,
            target_name="shuttle.streaming_reverse.short_d32_test_v1",
        )


def test_saved_state_is_an_explicit_alternative_not_a_hidden_recompute_input() -> None:
    program, schedule = _program_and_schedule()

    generated = generate_streaming_attention_backward_ffi(
        program,
        schedule,
        target_name="shuttle.streaming_reverse.saved_test_v1",
        state_policy=StreamingAttentionBackwardStatePolicy.SAVED_OUTPUT_AND_LOG_SUM_EXP,
    )

    assert generated.saved_state_encoding is StreamingAttentionLogSumExpEncoding.NATURAL_LOG
    assert all(kernel.signature[-1] == "1" for kernel in generated.aot_kernels)
    assert tuple(value.name for value in generated.inputs) == (
        "query",
        "key",
        "value",
        "output",
        "log_sum_exp",
        "output_cotangent",
    )
    assert tuple(kernel.kernel_name for kernel in generated.aot_kernels) == (
        "_streaming_dq_kernel",
        "_streaming_dkdv_kernel",
    )
    assert "{forward_launcher}" not in generated.handler_template
    shape = generated.inputs[0].shape
    query = jnp.zeros(shape, dtype=jnp.bfloat16)
    key = jnp.zeros(generated.inputs[1].shape, dtype=jnp.bfloat16)
    value = jnp.zeros(generated.inputs[2].shape, dtype=jnp.bfloat16)
    output_cotangent = jnp.zeros(shape, dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="requires output and log_sum_exp"):
        call_streaming_attention_backward_ffi(
            generated,
            query=query,
            key=key,
            value=value,
            output_cotangent=output_cotangent,
        )


def test_score_scale_mutation_changes_semantic_digest_and_aot_specialization() -> None:
    first_program, first_schedule = _program_and_schedule(0.5)
    second_program, second_schedule = _program_and_schedule(0.375)

    first = generate_streaming_attention_backward_ffi(
        first_program,
        first_schedule,
        target_name="shuttle.streaming_reverse.scale_first_v1",
        result_policy=StreamingAttentionBackwardResultPolicy.FORWARD_OUTPUT_AND_GRADIENTS,
    )
    second = generate_streaming_attention_backward_ffi(
        second_program,
        second_schedule,
        target_name="shuttle.streaming_reverse.scale_second_v1",
        result_policy=StreamingAttentionBackwardResultPolicy.FORWARD_OUTPUT_AND_GRADIENTS,
    )

    assert first.semantic_fingerprint != second.semantic_fingerprint
    assert first.aot_kernels[1].signature != second.aot_kernels[1].signature
    assert tuple(kernel.kernel_name for kernel in first.aot_kernels) == tuple(
        kernel.kernel_name for kernel in second.aot_kernels
    )


def test_aot_compile_plan_is_build_time_triton_but_runtime_handler_is_self_contained(tmp_path: Path) -> None:
    program, schedule = _program_and_schedule()
    generated = generate_streaming_attention_backward_ffi(
        program,
        schedule,
        target_name="shuttle.streaming_reverse.compile_audit_v1",
    )

    commands = tuple(
        kernel.compile_argv(
            repository=REPOSITORY,
            output_directory=tmp_path,
            target=None,
            python=Path(sys.executable),
        )
        for kernel in generated.aot_kernels
    )

    assert all(command[1:3] == ("-m", "triton.tools.compile") for command in commands)
    assert all("--target" not in command for command in commands)
    assert all("torch" not in " ".join(command).lower() for command in commands)
    assert "#include <cuda.h>" in generated.handler_template
    assert "xla/ffi/api/ffi.h" in generated.handler_template
    assert "triton" not in generated.handler_template.lower()
    assert "torch" not in generated.handler_template.lower()


def test_triton_36_cross_target_mode_fails_closed_before_build(tmp_path: Path) -> None:
    program, schedule = _program_and_schedule()
    generated = generate_streaming_attention_backward_ffi(
        program,
        schedule,
        target_name="shuttle.streaming_reverse.cross_target_rejected_v1",
    )

    with pytest.raises(ValueError, match="numeric target fields as strings"):
        compile_streaming_attention_backward_ffi(
            generated,
            repository=REPOSITORY,
            directory=tmp_path,
            nvcc=tmp_path / "nvcc",
            architecture="sm_90a",
            triton_target="cuda:90:32",
        )


def test_triton_aot_subprocess_owns_its_cache_directory(tmp_path: Path) -> None:
    cache_directory = tmp_path / "build" / ".triton-cache"
    captured_environment = tmp_path / "child-triton-cache.txt"
    script = "import os, pathlib, sys; pathlib.Path(sys.argv[1]).write_text(os.environ['TRITON_CACHE_DIR'])"
    command = (sys.executable, "-c", script, str(captured_environment))
    _run_triton_aot_compile(command, repository=REPOSITORY, cache_directory=cache_directory)

    assert cache_directory.is_dir()
    assert captured_environment.read_text() == str(cache_directory.resolve())


def test_call_validates_shape_before_ffi_dispatch() -> None:
    program, schedule = _program_and_schedule()
    generated = generate_streaming_attention_backward_ffi(
        program,
        schedule,
        target_name="shuttle.streaming_reverse.shape_test_v1",
    )
    query_shape = generated.inputs[0].shape

    with pytest.raises(ValueError, match=r"query.*shape"):
        call_streaming_attention_backward_ffi(
            generated,
            query=jnp.zeros((query_shape[0], query_shape[1] // 2, *query_shape[2:]), dtype=jnp.bfloat16),
            key=jnp.zeros(generated.inputs[1].shape, dtype=jnp.bfloat16),
            value=jnp.zeros(generated.inputs[2].shape, dtype=jnp.bfloat16),
            output_cotangent=jnp.zeros(generated.inputs[3].shape, dtype=jnp.bfloat16),
        )
