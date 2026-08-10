# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference-only delta-rule fixtures and NumPy input preparation.

``delta_rule_update_expression`` is intentionally limited to recovery unit tests.
Accepted compiler entrypoints and performance harnesses must recover expressions
from a structured StableHLO ``while`` instead.
"""

from dataclasses import dataclass

import numpy as np

from tile_lifetime.stateful_scan import (
    LogicalAxis,
    TensorExpression,
    TensorExpressionKind,
    binary_expression,
    contract_expression,
    input_expression,
    unary_expression,
)


@dataclass(frozen=True)
class DeltaRuleExpressionFixture:
    """One named-workload fixture expressed only through generic tensor algebra."""

    update: TensorExpression
    state_name: str
    axes: tuple[LogicalAxis, ...]


def delta_rule_update_expression(
    *,
    batch_size: int,
    heads: int,
    key_dimension: int,
    value_dimension: int,
    decay_axes: str,
    gate_operation: str = "exp",
    update_rank: int = 1,
) -> DeltaRuleExpressionFixture:
    """Build a hand-authored expression for isolated recovery unit tests only."""
    if any(value <= 0 for value in (batch_size, heads, key_dimension, value_dimension, update_rank)):
        raise ValueError("delta-rule expression dimensions must be positive")
    if decay_axes not in ("scalar", "key"):
        raise ValueError("decay_axes must be 'scalar' or 'key'")

    batch = LogicalAxis(id=0, extent=batch_size, label="batch")
    head = LogicalAxis(id=2, extent=heads, label="head")
    key_axis = LogicalAxis(id=3, extent=key_dimension, label="key")
    value_axis = LogicalAxis(id=4, extent=value_dimension, label="value")
    rank_axis = LogicalAxis(id=6, extent=update_rank, label="update_rank")
    state_axes = (batch, head, key_axis, value_axis)
    vector_prefix = (batch, head) if update_rank == 1 else (batch, head, rank_axis)
    key_axes = (*vector_prefix, key_axis)
    value_axes = (*vector_prefix, value_axis)

    state = input_expression("state", state_axes)
    key = input_expression("key", key_axes)
    value = input_expression("value", value_axes)
    beta = input_expression("beta", vector_prefix)
    log_decay_axes = (batch, head) if decay_axes == "scalar" else (batch, head, key_axis)
    log_decay = input_expression("log_decay", log_decay_axes)
    decay = unary_expression(gate_operation, log_decay)
    decayed_state = binary_expression(TensorExpressionKind.MULTIPLY, decay, state, state_axes)

    prediction = contract_expression(
        decayed_state,
        key,
        axes=value_axes,
        reduction_axes=(key_axis,),
    )
    residual = binary_expression(TensorExpressionKind.SUBTRACT, value, prediction, value_axes)
    scaled_residual = binary_expression(TensorExpressionKind.MULTIPLY, beta, residual, value_axes)
    correction = contract_expression(
        key,
        scaled_residual,
        axes=state_axes,
        reduction_axes=() if update_rank == 1 else (rank_axis,),
    )
    update = binary_expression(TensorExpressionKind.ADD, decayed_state, correction, state_axes)
    axes = (batch, head, key_axis, value_axis) if update_rank == 1 else (batch, head, key_axis, value_axis, rank_axis)
    return DeltaRuleExpressionFixture(update=update, state_name="state", axes=axes)


def prepare_delta_rule_inputs(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    beta: np.ndarray,
    *,
    initial_state: np.ndarray | None,
    normalize_query_key: bool,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize Q/K/V, beta, and a matrix-valued recurrent state."""
    q = np.asarray(query, dtype=np.float32)
    k = np.asarray(key, dtype=np.float32)
    v = np.asarray(value, dtype=np.float32)
    b = np.asarray(beta, dtype=np.float32)
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("query/key must match [batch, position, head, key], and value must be rank four")
    if v.shape[:3] != q.shape[:3] or b.shape != q.shape[:3]:
        raise ValueError("query, value, and beta domains must match")
    if epsilon <= 0:
        raise ValueError("normalization epsilon must be positive")

    if normalize_query_key:
        q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True, dtype=np.float32) + np.float32(epsilon))
        k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True, dtype=np.float32) + np.float32(epsilon))
    q = q * np.float32(q.shape[-1] ** -0.5)

    state_shape = (q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    if initial_state is None:
        state = np.zeros(state_shape, dtype=np.float32)
    else:
        state = np.asarray(initial_state, dtype=np.float32)
        if state.shape != state_shape:
            raise ValueError(f"initial state shape {state.shape} does not match {state_shape}")
        state = state.copy()
    return q, k, v, b, state
