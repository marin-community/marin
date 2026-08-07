# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kimi Delta Attention core represented as a generic ``StatefulScan``."""

from dataclasses import dataclass

import numpy as np

from tile_lifetime.delta_rule_reference import delta_rule_update_expression, prepare_delta_rule_inputs
from tile_lifetime.ir import DType
from tile_lifetime.plan import (
    ScanNumericalContract,
    StatefulScanSkeleton,
)
from tile_lifetime.stateful_scan import (
    AffineChunkSummary,
    ChunkAlgebra,
    LogicalAxis,
    ScanPrimitive,
    ScanPrimitiveKind,
    ScanValue,
    ScanValueRole,
    StatefulScan,
    apply_affine_transform,
    summarize_affine_sequence,
)
from tile_lifetime.stateful_scan_planner import compile_affine_scan_candidates
from tile_lifetime.stateful_scan_recovery import recover_affine_state_update


@dataclass(frozen=True)
class KimiDeltaScanCompilation:
    """Recovered per-channel-decay scan and its bounded candidates."""

    program: StatefulScan
    candidates: tuple[StatefulScanSkeleton, ...]


def compile_kimi_delta_scan(
    *,
    batch_size: int,
    sequence_length: int,
    heads: int,
    key_dimension: int,
    value_dimension: int,
    input_dtype: DType = DType.BF16,
    state_dtype: DType = DType.FP32,
    chunk_sizes: tuple[int, ...] = (32, 64),
) -> KimiDeltaScanCompilation:
    """Recover the KDA matrix-state core without a KDA-specific semantic node."""
    dimensions = (batch_size, sequence_length, heads, key_dimension, value_dimension)
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError("all Kimi Delta scan dimensions must be positive")
    if state_dtype is not DType.FP32:
        raise ValueError("the initial Kimi Delta prototype requires FP32 persistent state")
    if not chunk_sizes or any(chunk_size <= 0 for chunk_size in chunk_sizes):
        raise ValueError("chunk sizes must be a non-empty tuple of positive integers")

    batch = LogicalAxis(id=0, extent=batch_size, label="batch")
    position = LogicalAxis(id=1, extent=sequence_length, label="position")
    head = LogicalAxis(id=2, extent=heads, label="head")
    key = LogicalAxis(id=3, extent=key_dimension, label="key")
    value = LogicalAxis(id=4, extent=value_dimension, label="value")
    key_out = LogicalAxis(id=5, extent=key_dimension, label="key_out")

    values = (
        ScanValue("query", (batch, position, head, key), input_dtype, ScanValueRole.INPUT),
        ScanValue("key", (batch, position, head, key), input_dtype, ScanValueRole.INPUT),
        ScanValue("value", (batch, position, head, value), input_dtype, ScanValueRole.INPUT),
        ScanValue("log_decay", (batch, position, head, key), DType.FP32, ScanValueRole.INPUT),
        ScanValue("beta", (batch, position, head), DType.FP32, ScanValueRole.INPUT),
        ScanValue("state_prev", (batch, head, key, value), state_dtype, ScanValueRole.STATE),
        ScanValue("alpha", (batch, head, key), DType.FP32, ScanValueRole.TEMPORARY),
        ScanValue("state_decayed", (batch, head, key, value), state_dtype, ScanValueRole.TEMPORARY),
        ScanValue("prediction", (batch, head, value), DType.FP32, ScanValueRole.TEMPORARY),
        ScanValue("delta", (batch, head, value), DType.FP32, ScanValueRole.TEMPORARY),
        ScanValue("correction", (batch, head, key, value), DType.FP32, ScanValueRole.TEMPORARY),
        ScanValue("state_next", (batch, head, key, value), state_dtype, ScanValueRole.STATE),
        ScanValue("output", (batch, position, head, value), input_dtype, ScanValueRole.OUTPUT),
        ScanValue("chunk_transition", (batch, head, key_out, key), DType.FP32, ScanValueRole.SUMMARY),
        ScanValue("chunk_bias", (batch, head, key, value), DType.FP32, ScanValueRole.SUMMARY),
    )
    update = (
        ScanPrimitive("channel_decay", ScanPrimitiveKind.MAP, ("log_decay",), "alpha", "exp(log_decay)"),
        ScanPrimitive(
            "decay_state_rows",
            ScanPrimitiveKind.MAP,
            ("alpha", "state_prev"),
            "state_decayed",
            "alpha[key] * state_prev[key, value]",
        ),
        ScanPrimitive(
            "predict_value",
            ScanPrimitiveKind.CONTRACT,
            ("key", "state_decayed"),
            "prediction",
            "state_decayed^T @ key",
            ("key",),
        ),
        ScanPrimitive(
            "delta_residual",
            ScanPrimitiveKind.MAP,
            ("beta", "value", "prediction"),
            "delta",
            "beta * (value - prediction)",
        ),
        ScanPrimitive(
            "rank_one_update",
            ScanPrimitiveKind.CONTRACT,
            ("key", "delta"),
            "correction",
            "outer(key, delta)",
        ),
        ScanPrimitive(
            "write_state",
            ScanPrimitiveKind.MAP,
            ("state_decayed", "correction"),
            "state_next",
            "state_decayed + correction",
        ),
    )
    read = (
        ScanPrimitive(
            "read_state",
            ScanPrimitiveKind.CONTRACT,
            ("query", "state_next"),
            "output",
            "state_next^T @ query",
            ("key",),
        ),
    )
    chunk_algebra = ChunkAlgebra(
        summary_values=("chunk_transition", "chunk_bias"),
        summarize=(
            ScanPrimitive(
                "summarize_dplr_transition",
                ScanPrimitiveKind.FOLD,
                ("alpha", "beta", "key"),
                "chunk_transition",
                "ordered_compose((I - beta * outer(key, key)) @ diag(alpha))",
                ("position",),
            ),
            ScanPrimitive(
                "summarize_affine_bias",
                ScanPrimitiveKind.FOLD,
                ("beta", "key", "value"),
                "chunk_bias",
                "ordered_affine_bias(beta * outer(key, value))",
                ("position",),
            ),
        ),
        apply=(
            ScanPrimitive(
                "apply_factored_chunk",
                ScanPrimitiveKind.CONTRACT,
                ("chunk_transition", "state_prev", "chunk_bias"),
                "state_next",
                "chunk_transition @ state_prev + chunk_bias",
                ("key",),
            ),
        ),
        emit=(
            ScanPrimitive(
                "emit_factored_chunk",
                ScanPrimitiveKind.CONTRACT,
                ("query", "chunk_transition", "state_prev", "chunk_bias"),
                "output",
                "(prefix_transition @ state_prev + prefix_bias)^T @ query",
                ("key",),
            ),
        ),
        compose=(
            ScanPrimitive(
                "compose_exact_transition",
                ScanPrimitiveKind.CONTRACT,
                ("chunk_transition",),
                "chunk_transition",
                "later.transition @ earlier.transition",
                ("key",),
            ),
            ScanPrimitive(
                "compose_exact_bias",
                ScanPrimitiveKind.MAP,
                ("chunk_transition", "chunk_bias"),
                "chunk_bias",
                "later.transition @ earlier.bias + later.bias",
            ),
        ),
        bounded_representation_closed_under_compose=False,
    )
    program = StatefulScan(
        name="diagonal_delta_core",
        ordered_axis=position,
        values=values,
        state_input="state_prev",
        state_output="state_next",
        scan_inputs=("query", "key", "value", "log_decay", "beta"),
        scan_outputs=("output",),
        update=update,
        read=read,
        numerical_contract=ScanNumericalContract.SOURCE_ORDERED,
        chunk_algebra=chunk_algebra,
    )

    fixture = delta_rule_update_expression(
        batch_size=batch_size,
        heads=heads,
        key_dimension=key_dimension,
        value_dimension=value_dimension,
        decay_axes="key",
    )
    recovered_update = recover_affine_state_update(fixture.update, fixture.state_name)

    candidates = compile_affine_scan_candidates(
        recovered_update,
        ordered_axis="position",
        length=sequence_length,
        state="state",
        state_shape=(batch_size, heads, key_dimension, value_dimension),
        state_dtype=state_dtype,
        output="output",
        state_layout="batch_head_key_value",
        chunk_sizes=chunk_sizes,
    )
    return KimiDeltaScanCompilation(program=program, candidates=candidates)


def recurrent_kimi_delta_reference(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    log_decay: np.ndarray,
    beta: np.ndarray,
    *,
    initial_state: np.ndarray | None = None,
    normalize_query_key: bool = False,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute the per-key-channel KDA recurrence in source order."""
    q, k, v, b, state = prepare_delta_rule_inputs(
        query,
        key,
        value,
        beta,
        initial_state=initial_state,
        normalize_query_key=normalize_query_key,
        epsilon=epsilon,
    )
    g = np.asarray(log_decay, dtype=np.float32)
    if g.shape != q.shape:
        raise ValueError("per-channel log decay must match the query/key shape")
    output = np.empty((q.shape[0], q.shape[1], q.shape[2], v.shape[-1]), dtype=np.float32)
    for position in range(q.shape[1]):
        alpha = np.exp(g[:, position, :, :], dtype=np.float32)
        state = state * alpha[..., None]
        prediction = np.einsum("bhkv,bhk->bhv", state, k[:, position, :, :], optimize=False)
        delta = b[:, position, :, None] * (v[:, position, :, :] - prediction)
        state = state + np.einsum("bhk,bhv->bhkv", k[:, position, :, :], delta, optimize=False)
        output[:, position, :, :] = np.einsum("bhkv,bhk->bhv", state, q[:, position, :, :], optimize=False)
    return output, state


def summarize_kimi_delta_chunk(
    key: np.ndarray,
    value: np.ndarray,
    log_decay: np.ndarray,
    beta: np.ndarray,
) -> AffineChunkSummary:
    """Build exact full-affine prefixes for one prepared KDA chunk."""
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError("chunk key and value must have shape [batch, position, head, dimension]")
    if key.shape[:3] != value.shape[:3] or log_decay.shape != key.shape or beta.shape != key.shape[:3]:
        raise ValueError("chunk key, value, per-channel decay, and beta domains must match")

    batch, length, heads, key_dimension = key.shape
    identity = np.broadcast_to(
        np.eye(key_dimension, dtype=np.float32),
        (batch, heads, key_dimension, key_dimension),
    ).copy()
    key_bhpk = np.moveaxis(key.astype(np.float32, copy=False), 1, 2)
    value_bhpv = np.moveaxis(value.astype(np.float32, copy=False), 1, 2)
    beta_bhp = np.moveaxis(beta.astype(np.float32, copy=False), 1, 2)
    alpha_bhpk = np.exp(np.moveaxis(log_decay.astype(np.float32, copy=False), 1, 2), dtype=np.float32)
    outer_kk = np.einsum("bhpk,bhpj->bhpkj", key_bhpk, key_bhpk, optimize=False)
    delta_transition = identity[:, :, None] - beta_bhp[..., None, None] * outer_kk
    transitions = delta_transition * alpha_bhpk[..., None, :]
    biases = beta_bhp[..., None, None] * np.einsum("bhpk,bhpv->bhpkv", key_bhpk, value_bhpv, optimize=False)
    if transitions.shape[2] != length:
        raise AssertionError("affine transition sequence lost its ordered extent")
    return summarize_affine_sequence(transitions, biases)


def chunkwise_kimi_delta_reference(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    log_decay: np.ndarray,
    beta: np.ndarray,
    *,
    chunk_size: int,
    initial_state: np.ndarray | None = None,
    normalize_query_key: bool = False,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute KDA through exact affine summaries and an ordered chunk scan."""
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    q, k, v, b, state = prepare_delta_rule_inputs(
        query,
        key,
        value,
        beta,
        initial_state=initial_state,
        normalize_query_key=normalize_query_key,
        epsilon=epsilon,
    )
    g = np.asarray(log_decay, dtype=np.float32)
    if g.shape != q.shape:
        raise ValueError("per-channel log decay must match the query/key shape")
    output = np.empty((q.shape[0], q.shape[1], q.shape[2], v.shape[-1]), dtype=np.float32)
    for start in range(0, q.shape[1], chunk_size):
        stop = min(start + chunk_size, q.shape[1])
        summary = summarize_kimi_delta_chunk(k[:, start:stop], v[:, start:stop], g[:, start:stop], b[:, start:stop])
        incoming = state
        for offset, prefix in enumerate(summary.prefixes):
            prefix_state = apply_affine_transform(prefix, incoming)
            output[:, start + offset, :, :] = np.einsum(
                "bhkv,bhk->bhv", prefix_state, q[:, start + offset, :, :], optimize=False
            )
        state = apply_affine_transform(summary.final, incoming)
    return output, state
