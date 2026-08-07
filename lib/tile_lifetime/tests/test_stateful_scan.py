# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from tile_lifetime import (
    AffineStateTransform,
    ChunkSummaryRepresentation,
    DType,
    LogicalAxis,
    StatefulScanExecutionForm,
    StateTransitionStructure,
    TensorExpressionKind,
    apply_affine_transform,
    apply_factored_affine_chunk,
    binary_expression,
    chunkwise_gated_delta_reference,
    chunkwise_kimi_delta_reference,
    compile_affine_scan_candidates,
    compile_gated_delta_scan,
    compile_kimi_delta_scan,
    compose_affine_transforms,
    execute_recurrent_factored_affine,
    explain_stateful_scan,
    input_expression,
    recover_affine_state_update,
    recurrent_gated_delta_reference,
    recurrent_kimi_delta_reference,
    summarize_factored_affine_chunk,
)
from tile_lifetime.delta_rule_reference import delta_rule_update_expression


def _inputs(
    *,
    batch: int = 2,
    length: int = 11,
    heads: int = 3,
    key_dimension: int = 5,
    value_dimension: int = 7,
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(123)
    query = rng.normal(size=(batch, length, heads, key_dimension)).astype(np.float32)
    key = rng.normal(size=(batch, length, heads, key_dimension)).astype(np.float32)
    value = rng.normal(size=(batch, length, heads, value_dimension)).astype(np.float32)
    log_decay = -rng.uniform(0.01, 0.4, size=(batch, length, heads)).astype(np.float32)
    beta = rng.uniform(0.05, 0.95, size=(batch, length, heads)).astype(np.float32)
    return query, key, value, log_decay, beta


def _direct_recurrence(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    log_decay: np.ndarray,
    beta: np.ndarray,
    initial_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    epsilon = np.float32(1e-6)
    query = query / np.sqrt(np.sum(query * query, axis=-1, keepdims=True) + epsilon)
    key = key / np.sqrt(np.sum(key * key, axis=-1, keepdims=True) + epsilon)
    query = query * np.float32(query.shape[-1] ** -0.5)

    state = initial_state.copy()
    outputs = []
    for position in range(query.shape[1]):
        state *= np.exp(log_decay[:, position, :])[..., None, None]
        predicted_value = np.sum(state * key[:, position, :, :, None], axis=-2)
        delta = beta[:, position, :, None] * (value[:, position, :, :] - predicted_value)
        state += key[:, position, :, :, None] * delta[:, :, None, :]
        outputs.append(np.sum(state * query[:, position, :, :, None], axis=-2))
    return np.stack(outputs, axis=1), state


def _direct_kimi_recurrence(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    log_decay: np.ndarray,
    beta: np.ndarray,
    initial_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    query = query * np.float32(query.shape[-1] ** -0.5)

    state = initial_state.copy()
    outputs = []
    for position in range(query.shape[1]):
        state *= np.exp(log_decay[:, position, :, :])[..., None]
        predicted_value = np.sum(state * key[:, position, :, :, None], axis=-2)
        residual = beta[:, position, :, None] * (value[:, position, :, :] - predicted_value)
        state += key[:, position, :, :, None] * residual[:, :, None, :]
        outputs.append(np.sum(state * query[:, position, :, :, None], axis=-2))
    return np.stack(outputs, axis=1), state


def test_affine_transform_composition_matches_sequential_application():
    rng = np.random.default_rng(0)
    shape = (2, 3, 4, 4)
    bias_shape = (2, 3, 4, 5)
    earlier = AffineStateTransform(
        transition=rng.normal(size=shape).astype(np.float32),
        bias=rng.normal(size=bias_shape).astype(np.float32),
    )
    later = AffineStateTransform(
        transition=rng.normal(size=shape).astype(np.float32),
        bias=rng.normal(size=bias_shape).astype(np.float32),
    )
    state = rng.normal(size=bias_shape).astype(np.float32)

    sequential = apply_affine_transform(later, apply_affine_transform(earlier, state))
    composed = apply_affine_transform(compose_affine_transforms(earlier, later), state)

    np.testing.assert_allclose(composed, sequential, rtol=2e-6, atol=2e-5)


def test_gated_delta_scan_recovers_generic_update_and_bounded_candidates():
    compilation = compile_gated_delta_scan(
        batch_size=1,
        sequence_length=256,
        heads=32,
        key_dimension=128,
        value_dimension=128,
        chunk_sizes=(32, 64),
    )

    assert compilation.program.state_input == "state_prev"
    assert compilation.program.state_output == "state_next"
    assert tuple(candidate.execution_form for candidate in compilation.candidates) == (
        StatefulScanExecutionForm.RECURRENT,
        StatefulScanExecutionForm.CHUNKWISE,
        StatefulScanExecutionForm.CHUNKWISE,
    )
    assert tuple(candidate.chunk_size for candidate in compilation.candidates) == (1, 32, 64)
    assert compilation.candidates[0].summary_representation is ChunkSummaryRepresentation.NONE
    assert compilation.candidates[1].summary_representation is ChunkSummaryRepresentation.FACTORED_AFFINE
    assert {candidate.backend for candidate in compilation.candidates} == {
        "shuttle_affine_scan_recurrent_template",
        "shuttle_factored_affine_scan_template",
    }
    assert {candidate.transition_structure for candidate in compilation.candidates} == {
        StateTransitionStructure.DIAGONAL_PLUS_LOW_RANK
    }
    assert {candidate.maximum_update_rank for candidate in compilation.candidates} == {1}
    dump = explain_stateful_scan(compilation.program)
    assert "state_decayed + correction" in dump
    assert "state_next^T @ query" in dump
    assert "chunk_transition, chunk_bias" in dump


def test_recurrent_gated_delta_matches_direct_source_order_reference():
    inputs = _inputs()
    initial_state = np.random.default_rng(7).normal(size=(2, 3, 5, 7)).astype(np.float32) * 0.1

    expected_output, expected_state = _direct_recurrence(*inputs, initial_state)
    output, state = recurrent_gated_delta_reference(*inputs, initial_state=initial_state)

    np.testing.assert_allclose(output, expected_output, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(state, expected_state, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("chunk_size", [1, 2, 5, 8, 16])
def test_chunkwise_affine_summary_matches_recurrent_with_tail_chunk(chunk_size: int):
    inputs = _inputs(length=13)
    initial_state = np.random.default_rng(9).normal(size=(2, 3, 5, 7)).astype(np.float32) * 0.05

    recurrent_output, recurrent_state = recurrent_gated_delta_reference(*inputs, initial_state=initial_state)
    chunk_output, chunk_state = chunkwise_gated_delta_reference(
        *inputs,
        chunk_size=chunk_size,
        initial_state=initial_state,
    )

    np.testing.assert_allclose(chunk_output, recurrent_output, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(chunk_state, recurrent_state, rtol=2e-5, atol=2e-5)


def test_gated_delta_persistent_state_continuation_matches_one_pass():
    inputs = _inputs(length=17)
    split = 6
    initial_state = np.random.default_rng(11).normal(size=(2, 3, 5, 7)).astype(np.float32) * 0.05

    full_output, full_state = recurrent_gated_delta_reference(*inputs, initial_state=initial_state)
    prefix_inputs = tuple(value[:, :split] for value in inputs)
    suffix_inputs = tuple(value[:, split:] for value in inputs)
    prefix_output, middle_state = recurrent_gated_delta_reference(*prefix_inputs, initial_state=initial_state)
    suffix_output, final_state = recurrent_gated_delta_reference(*suffix_inputs, initial_state=middle_state)

    np.testing.assert_array_equal(prefix_output, full_output[:, :split])
    np.testing.assert_array_equal(suffix_output, full_output[:, split:])
    np.testing.assert_array_equal(final_state, full_state)


def test_gated_delta_rejects_incompatible_state_and_precision():
    with pytest.raises(ValueError, match="requires FP32 persistent state"):
        compile_gated_delta_scan(
            batch_size=1,
            sequence_length=8,
            heads=2,
            key_dimension=4,
            value_dimension=6,
            state_dtype=DType.BF16,
        )

    inputs = _inputs(batch=1, heads=2, key_dimension=4, value_dimension=6)
    with pytest.raises(ValueError, match="initial state shape"):
        recurrent_gated_delta_reference(*inputs, initial_state=np.zeros((1, 2, 4, 5), dtype=np.float32))


def test_kimi_delta_uses_the_same_stateful_scan_abstraction():
    compilation = compile_kimi_delta_scan(
        batch_size=1,
        sequence_length=64,
        heads=4,
        key_dimension=128,
        value_dimension=128,
    )

    assert compilation.program.state_input == "state_prev"
    assert compilation.program.state_output == "state_next"
    assert tuple(candidate.execution_form for candidate in compilation.candidates) == (
        StatefulScanExecutionForm.RECURRENT,
        StatefulScanExecutionForm.CHUNKWISE,
        StatefulScanExecutionForm.CHUNKWISE,
    )
    assert compilation.program.chunk_algebra is not None
    assert not compilation.program.chunk_algebra.bounded_representation_closed_under_compose
    assert "diag(alpha)" in compilation.program.chunk_algebra.summarize[0].equation
    assert {candidate.backend for candidate in compilation.candidates} == {
        "shuttle_affine_scan_recurrent_template",
        "shuttle_factored_affine_scan_template",
    }
    assert {candidate.transition_structure for candidate in compilation.candidates} == {
        StateTransitionStructure.DIAGONAL_PLUS_LOW_RANK
    }


@pytest.mark.parametrize("decay_axes", ["scalar", "key"])
@pytest.mark.parametrize("gate_operation", ["exp", "sigmoid", "clamped_softplus"])
@pytest.mark.parametrize("update_rank", [1, 2, 4])
def test_affine_recovery_reuses_one_factor_family_across_nearby_recurrences(
    decay_axes: str,
    gate_operation: str,
    update_rank: int,
):
    fixture = delta_rule_update_expression(
        batch_size=1,
        heads=3,
        key_dimension=16,
        value_dimension=24,
        decay_axes=decay_axes,
        gate_operation=gate_operation,
        update_rank=update_rank,
    )

    recovered = recover_affine_state_update(fixture.update, fixture.state_name)

    assert recovered.transition_structure is StateTransitionStructure.DIAGONAL_PLUS_LOW_RANK
    assert recovered.maximum_low_rank == update_rank
    assert tuple(axis.label for axis in recovered.diagonal_scale_axes) == (
        ("batch", "head") if decay_axes == "scalar" else ("batch", "head", "key")
    )
    assert recovered.term_signatures[0] == ("multiply",)
    assert recovered.term_signatures[1][0:3] == ("multiply", "contract[key]", "multiply")

    candidates = compile_affine_scan_candidates(
        recovered,
        ordered_axis="position",
        length=256,
        state="state",
        state_shape=(1, 3, 16, 24),
        state_dtype=DType.FP32,
        output="output",
        state_layout="batch_head_key_value",
        chunk_sizes=(32, 64),
    )
    assert tuple(candidate.name for candidate in candidates) == (
        "affine_scan_recurrent",
        "affine_scan_chunk_32",
        "affine_scan_chunk_64",
    )
    assert {candidate.maximum_update_rank for candidate in candidates} == {update_rank}


def test_affine_recovery_rejects_nonlinear_prior_state_dependency():
    batch = LogicalAxis(id=0, extent=1, label="batch")
    key = LogicalAxis(id=1, extent=8, label="key")
    value = LogicalAxis(id=2, extent=8, label="value")
    state = input_expression("state", (batch, key, value))
    nonlinear = binary_expression(TensorExpressionKind.MULTIPLY, state, state, state.axes)

    with pytest.raises(ValueError, match="nonlinear"):
        recover_affine_state_update(nonlinear, "state")


def test_affine_recovery_preserves_factor_family_when_diagonal_moves_after_update():
    fixture = delta_rule_update_expression(
        batch_size=1,
        heads=2,
        key_dimension=8,
        value_dimension=12,
        decay_axes="key",
        update_rank=2,
    )
    post_scale = input_expression("post_scale", fixture.update.axes[:-1])
    mutated = binary_expression(
        TensorExpressionKind.MULTIPLY,
        post_scale,
        fixture.update,
        fixture.update.axes,
    )

    recovered = recover_affine_state_update(mutated, fixture.state_name)

    assert recovered.transition_structure is StateTransitionStructure.DIAGONAL_PLUS_LOW_RANK
    assert recovered.maximum_low_rank == 2
    assert recovered.term_signatures[0] == ("multiply", "multiply")
    assert recovered.term_signatures[1][-1] == "multiply"


@pytest.mark.parametrize("update_rank", [1, 3])
@pytest.mark.parametrize("diagonal_mode", ["scalar", "key"])
def test_generic_factored_executor_handles_diagonal_and_rank_mutations(
    update_rank: int,
    diagonal_mode: str,
):
    rng = np.random.default_rng(31)
    batch, length, heads, key_dimension, value_dimension = 2, 5, 3, 7, 9
    read = rng.normal(size=(batch, length, heads, key_dimension)).astype(np.float32)
    key = rng.normal(size=(batch, length, heads, update_rank, key_dimension)).astype(np.float32)
    value = rng.normal(size=(batch, length, heads, update_rank, value_dimension)).astype(np.float32)
    beta = rng.uniform(0.05, 0.8, size=(batch, length, heads, update_rank)).astype(np.float32)
    if diagonal_mode == "scalar":
        scalar = rng.uniform(0.5, 1.0, size=(batch, length, heads, 1)).astype(np.float32)
        diagonal = np.broadcast_to(scalar, (batch, length, heads, key_dimension)).copy()
    else:
        diagonal = rng.uniform(0.5, 1.0, size=(batch, length, heads, key_dimension)).astype(np.float32)
    initial_state = rng.normal(size=(batch, heads, key_dimension, value_dimension)).astype(np.float32) * 0.05

    left = key
    right = key
    additive = value
    output, final_state = execute_recurrent_factored_affine(
        read,
        diagonal,
        left,
        right,
        additive,
        beta,
        initial_state,
    )

    expected_state = initial_state.copy()
    expected_outputs = []
    for position in range(length):
        expected_state *= diagonal[:, position, :, :, None]
        prediction = np.einsum("bhkv,bhrk->bhrv", expected_state, key[:, position], optimize=False)
        residual = beta[:, position, :, :, None] * (value[:, position] - prediction)
        expected_state += np.einsum("bhrk,bhrv->bhkv", key[:, position], residual, optimize=False)
        expected_outputs.append(np.einsum("bhkv,bhk->bhv", expected_state, read[:, position], optimize=False))

    np.testing.assert_allclose(output, np.stack(expected_outputs, axis=1), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(final_state, expected_state, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("update_rank", [1, 3])
@pytest.mark.parametrize("diagonal_mode", ["scalar", "key"])
@pytest.mark.parametrize("chunk_size", [1, 2, 5])
def test_generic_factored_chunk_summary_handles_nearby_recurrences(
    update_rank: int,
    diagonal_mode: str,
    chunk_size: int,
):
    rng = np.random.default_rng(37)
    batch, length, heads, key_dimension, value_dimension = 2, 7, 3, 5, 6
    read = rng.normal(size=(batch, length, heads, key_dimension)).astype(np.float32)
    left = rng.normal(size=(batch, length, heads, update_rank, key_dimension)).astype(np.float32) * 0.2
    right = rng.normal(size=(batch, length, heads, update_rank, key_dimension)).astype(np.float32) * 0.2
    additive = rng.normal(size=(batch, length, heads, update_rank, value_dimension)).astype(np.float32)
    scale = rng.uniform(0.05, 0.8, size=(batch, length, heads, update_rank)).astype(np.float32)
    if diagonal_mode == "scalar":
        scalar = rng.uniform(0.7, 1.0, size=(batch, length, heads, 1)).astype(np.float32)
        diagonal = np.broadcast_to(scalar, read.shape).copy()
    else:
        diagonal = rng.uniform(0.7, 1.0, size=read.shape).astype(np.float32)
    initial_state = rng.normal(size=(batch, heads, key_dimension, value_dimension)).astype(np.float32) * 0.05

    expected_output, expected_state = execute_recurrent_factored_affine(
        read,
        diagonal,
        left,
        right,
        additive,
        scale,
        initial_state,
    )
    state = initial_state
    outputs = []
    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        summary = summarize_factored_affine_chunk(
            read[:, start:stop],
            diagonal[:, start:stop],
            left[:, start:stop],
            right[:, start:stop],
            additive[:, start:stop],
            scale[:, start:stop],
        )
        chunk_output, state = apply_factored_affine_chunk(summary, state)
        outputs.append(chunk_output)

    np.testing.assert_allclose(np.concatenate(outputs, axis=1), expected_output, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(state, expected_state, rtol=3e-5, atol=3e-5)


def test_kimi_delta_recurrent_matches_direct_per_channel_decay_reference():
    query, key, value, _, beta = _inputs(length=9)
    rng = np.random.default_rng(17)
    log_decay = -rng.uniform(0.01, 0.3, size=query.shape).astype(np.float32)
    initial_state = rng.normal(size=(2, 3, 5, 7)).astype(np.float32) * 0.05

    expected_output, expected_state = _direct_kimi_recurrence(
        query,
        key,
        value,
        log_decay,
        beta,
        initial_state,
    )
    output, state = recurrent_kimi_delta_reference(
        query,
        key,
        value,
        log_decay,
        beta,
        initial_state=initial_state,
    )

    np.testing.assert_allclose(output, expected_output, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(state, expected_state, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("chunk_size", [1, 3, 8])
def test_kimi_delta_affine_chunks_match_recurrent(chunk_size: int):
    query, key, value, _, beta = _inputs(length=11)
    rng = np.random.default_rng(19)
    log_decay = -rng.uniform(0.01, 0.3, size=query.shape).astype(np.float32)
    initial_state = rng.normal(size=(2, 3, 5, 7)).astype(np.float32) * 0.05

    recurrent_output, recurrent_state = recurrent_kimi_delta_reference(
        query,
        key,
        value,
        log_decay,
        beta,
        initial_state=initial_state,
    )
    chunk_output, chunk_state = chunkwise_kimi_delta_reference(
        query,
        key,
        value,
        log_decay,
        beta,
        chunk_size=chunk_size,
        initial_state=initial_state,
    )

    np.testing.assert_allclose(chunk_output, recurrent_output, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(chunk_state, recurrent_state, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("log_decay_value", [0.0, -0.1, -1.0, -8.0])
def test_gated_delta_chunk_algebra_handles_distinct_decay_regimes(log_decay_value: float):
    query, key, value, _, beta = _inputs(batch=1, length=7, heads=2, key_dimension=4, value_dimension=6)
    log_decay = np.full(query.shape[:3], log_decay_value, dtype=np.float32)
    initial_state = np.random.default_rng(23).normal(size=(1, 2, 4, 6)).astype(np.float32) * 0.05

    recurrent_output, recurrent_state = recurrent_gated_delta_reference(
        query,
        key,
        value,
        log_decay,
        beta,
        initial_state=initial_state,
    )
    chunk_output, chunk_state = chunkwise_gated_delta_reference(
        query,
        key,
        value,
        log_decay,
        beta,
        chunk_size=4,
        initial_state=initial_state,
    )

    assert np.isfinite(chunk_output).all()
    assert np.isfinite(chunk_state).all()
    np.testing.assert_allclose(chunk_output, recurrent_output, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(chunk_state, recurrent_state, rtol=3e-5, atol=3e-5)
