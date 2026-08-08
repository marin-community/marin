# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model-independent affine-state analysis for ordered tensor programs."""

from dataclasses import dataclass, replace
from functools import reduce
from operator import mul

import numpy as np

from tile_lifetime.plan import StateTransitionStructure
from tile_lifetime.stateful_scan import LogicalAxis, TensorExpression, TensorExpressionKind


@dataclass(frozen=True)
class AppliedLinearMap:
    """One tensor operation applied to a state-dependent value."""

    kind: TensorExpressionKind
    operation: str | None
    independent_inputs: tuple[TensorExpression, ...]
    dependent_input_index: int
    input_axes: tuple[LogicalAxis, ...]
    output_axes: tuple[LogicalAxis, ...]
    reduction_axes: tuple[LogicalAxis, ...]


@dataclass(frozen=True)
class StateLinearTerm:
    """One signed linear path from prior state to an update contribution."""

    coefficient: int
    maps: tuple[AppliedLinearMap, ...]


@dataclass(frozen=True)
class AffineTensorExpression:
    """A tensor expression decomposed into bias and state-linear paths."""

    bias: TensorExpression | None
    state_terms: tuple[StateLinearTerm, ...]


@dataclass(frozen=True)
class RecoveredAffineStateUpdate:
    """Factorization facts used to select a generic scan skeleton."""

    state_name: str
    state_axes: tuple[LogicalAxis, ...]
    affine: AffineTensorExpression
    transition_structure: StateTransitionStructure
    diagonal_scale_axes: tuple[LogicalAxis, ...]
    maximum_low_rank: int
    term_signatures: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class FactoredAffineChunkSummary:
    """Exact bounded-chunk summary for a diagonal-plus-low-rank scan.

    The represented affine map is ``D * state + U @ (V^T @ state + Z)``.
    Rank grows with the number of updates inside one chunk, so this is used for
    bounded chunks followed by an ordered inter-chunk scan.
    """

    diagonal: np.ndarray
    low_rank_left: np.ndarray
    low_rank_right: np.ndarray
    additive_coefficients: np.ndarray
    transformed_read: np.ndarray
    local_output: np.ndarray


def execute_recurrent_factored_affine(
    read: np.ndarray,
    diagonal: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    additive: np.ndarray,
    residual_scale: np.ndarray,
    initial_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute the generic bounded-rank recurrent skeleton in FP32."""
    q = np.asarray(read, dtype=np.float32)
    d = np.asarray(diagonal, dtype=np.float32)
    u = np.asarray(left, dtype=np.float32)
    v = np.asarray(right, dtype=np.float32)
    c = np.asarray(additive, dtype=np.float32)
    scale = np.asarray(residual_scale, dtype=np.float32)
    state = np.asarray(initial_state, dtype=np.float32).copy()
    if q.ndim != 4 or d.shape != q.shape:
        raise ValueError("read and diagonal factors must have shape [B,T,H,K]")
    batch, length, heads, key_dimension = q.shape
    if state.ndim != 4 or state.shape[:3] != (batch, heads, key_dimension):
        raise ValueError("initial state must have shape [B,H,K,V]")
    value_dimension = state.shape[-1]
    if u.ndim != 5 or u.shape[:3] != (batch, length, heads) or u.shape[-1] != key_dimension:
        raise ValueError("left factors must have shape [B,T,H,R,K]")
    if v.shape != u.shape:
        raise ValueError("right factors must match left factors")
    if c.shape != (*u.shape[:-1], value_dimension):
        raise ValueError("additive factors must have shape [B,T,H,R,V]")
    if scale.shape != u.shape[:-1]:
        raise ValueError("residual scale must have shape [B,T,H,R]")

    output = np.empty((batch, length, heads, value_dimension), dtype=np.float32)
    for position in range(length):
        state *= d[:, position, :, :, None]
        prediction = np.einsum("bhkv,bhrk->bhrv", state, v[:, position], optimize=False)
        residual = scale[:, position, :, :, None] * (c[:, position] - prediction)
        state += np.einsum("bhrk,bhrv->bhkv", u[:, position], residual, optimize=False)
        output[:, position] = np.einsum("bhkv,bhk->bhv", state, q[:, position], optimize=False)
    return output, state


def summarize_factored_affine_chunk(
    read: np.ndarray,
    diagonal: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    additive: np.ndarray,
    residual_scale: np.ndarray,
) -> FactoredAffineChunkSummary:
    """Derive an exact factored summary from generic affine update factors."""
    q, d, u, v, c, scale = _validate_factored_inputs(read, diagonal, left, right, additive, residual_scale)
    batch, length, heads, key_dimension = q.shape
    value_dimension = c.shape[-1]
    diagonal_summary = np.ones((batch, heads, key_dimension), dtype=np.float32)
    left_summary = np.empty((batch, heads, key_dimension, 0), dtype=np.float32)
    right_summary = np.empty((batch, heads, key_dimension, 0), dtype=np.float32)
    additive_summary = np.empty((batch, heads, 0, value_dimension), dtype=np.float32)
    transformed_read = np.empty_like(q)
    local_output = np.empty((batch, length, heads, value_dimension), dtype=np.float32)

    for position in range(length):
        token_diagonal = d[:, position]
        token_left = np.moveaxis(u[:, position], -2, -1)
        token_right = np.moveaxis(v[:, position], -2, -1)
        token_scale = scale[:, position]
        token_transition_right = -token_diagonal[..., :, None] * token_right * token_scale[..., None, :]
        token_additive = c[:, position] * token_scale[..., :, None]

        scaled_prior_left = token_diagonal[..., :, None] * left_summary
        interaction = np.einsum("bhkl,bhkr->bhlr", left_summary, token_transition_right, optimize=False)
        appended_right = diagonal_summary[..., :, None] * token_transition_right
        appended_right += np.einsum("bhkl,bhlr->bhkr", right_summary, interaction, optimize=False)
        appended_additive = token_additive
        appended_additive += np.einsum("bhlr,bhlv->bhrv", interaction, additive_summary, optimize=False)
        left_summary = np.concatenate((scaled_prior_left, token_left), axis=-1)
        right_summary = np.concatenate((right_summary, appended_right), axis=-1)
        additive_summary = np.concatenate((additive_summary, appended_additive), axis=-2)
        diagonal_summary *= token_diagonal

        token_read = q[:, position]
        read_projection = np.einsum("bhkl,bhk->bhl", left_summary, token_read, optimize=False)
        transformed_read[:, position] = diagonal_summary * token_read
        transformed_read[:, position] += np.einsum("bhkl,bhl->bhk", right_summary, read_projection, optimize=False)
        local_output[:, position] = np.einsum("bhl,bhlv->bhv", read_projection, additive_summary, optimize=False)

    return FactoredAffineChunkSummary(
        diagonal=diagonal_summary,
        low_rank_left=left_summary,
        low_rank_right=right_summary,
        additive_coefficients=additive_summary,
        transformed_read=transformed_read,
        local_output=local_output,
    )


def solve_factored_affine_chunk(
    read: np.ndarray,
    diagonal: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    additive: np.ndarray,
    residual_scale: np.ndarray,
) -> FactoredAffineChunkSummary:
    """Derive one chunk summary with a masked triangular factor solve.

    This is the contraction-oriented form used by the GPU chunk backend. It
    expresses the same ordered update as :func:`summarize_factored_affine_chunk`
    but batches all within-chunk interactions into matrix products and unit
    lower-triangular solves. Updates at the same position remain simultaneous;
    only factors from earlier positions appear below the triangular diagonal.

    The solve reassociates FP32 operations within the bounded chunk, so callers
    must select a ``bounded_reassociation`` numerical contract.
    """
    q, d, u, v, c, scale = _validate_factored_inputs(read, diagonal, left, right, additive, residual_scale)
    batch, length, heads, key_dimension = q.shape
    update_rank = u.shape[-2]
    value_dimension = c.shape[-1]
    summary_rank = length * update_rank

    prefix_diagonal = np.cumprod(d, axis=1, dtype=np.float32)
    expanded_prefix = prefix_diagonal[:, :, :, None, :]
    transformed_left = u / expanded_prefix
    transformed_right = v * expanded_prefix * scale[..., None]
    transformed_additive = c * scale[..., None]

    left_matrix = transformed_left.transpose(0, 2, 1, 3, 4).reshape(batch, heads, summary_rank, key_dimension)
    left_matrix = left_matrix.transpose(0, 1, 3, 2)
    right_matrix = transformed_right.transpose(0, 2, 1, 3, 4).reshape(batch, heads, summary_rank, key_dimension)
    right_matrix = right_matrix.transpose(0, 1, 3, 2)
    additive_matrix = transformed_additive.transpose(0, 2, 1, 3, 4).reshape(batch, heads, summary_rank, value_dimension)

    interactions = np.einsum("bhki,bhkj->bhij", right_matrix, left_matrix, optimize=False)
    update_positions = np.arange(summary_rank) // update_rank
    prior_position = update_positions[:, None] > update_positions[None, :]
    triangular = np.eye(summary_rank, dtype=np.float32) + interactions * prior_position
    solved_additive = np.linalg.solve(triangular, additive_matrix)
    solved_right = -np.linalg.solve(triangular, right_matrix.transpose(0, 1, 3, 2))
    solved_right = solved_right.transpose(0, 1, 3, 2)

    final_diagonal = prefix_diagonal[:, -1]
    summary_left = final_diagonal[..., :, None] * left_matrix
    scaled_read = q * prefix_diagonal
    read_left = np.einsum("bthk,bhkl->bhtl", scaled_read, left_matrix, optimize=False)
    visible_update = np.arange(length)[:, None] >= update_positions[None, :]
    read_left *= visible_update
    read_solve = np.linalg.solve(
        triangular.transpose(0, 1, 3, 2),
        read_left.transpose(0, 1, 3, 2),
    ).transpose(0, 1, 3, 2)
    transformed_read = scaled_read.transpose(0, 2, 1, 3)
    transformed_read -= np.einsum(
        "bhtl,bhlk->bhtk",
        read_solve,
        right_matrix.transpose(0, 1, 3, 2),
        optimize=False,
    )
    transformed_read = transformed_read.transpose(0, 2, 1, 3)
    local_output = np.einsum("bhtl,bhlv->bthv", read_left, solved_additive, optimize=False)

    return FactoredAffineChunkSummary(
        diagonal=final_diagonal,
        low_rank_left=summary_left,
        low_rank_right=solved_right,
        additive_coefficients=solved_additive,
        transformed_read=transformed_read,
        local_output=local_output,
    )


def apply_factored_affine_chunk(
    summary: FactoredAffineChunkSummary,
    initial_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Emit chunk outputs and advance state from one factored summary."""
    state = np.asarray(initial_state, dtype=np.float32)
    if state.ndim != 4 or state.shape[:3] != summary.diagonal.shape:
        raise ValueError("initial state must match the summary batch, head, and key domains")
    outputs = summary.local_output + np.einsum("bthk,bhkv->bthv", summary.transformed_read, state, optimize=False)
    projected_state = np.einsum("bhkl,bhkv->bhlv", summary.low_rank_right, state, optimize=False)
    projected_state += summary.additive_coefficients
    final_state = summary.diagonal[..., :, None] * state
    final_state += np.einsum("bhkl,bhlv->bhkv", summary.low_rank_left, projected_state, optimize=False)
    return outputs, final_state


def _validate_factored_inputs(
    read: np.ndarray,
    diagonal: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    additive: np.ndarray,
    residual_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(read, dtype=np.float32)
    d = np.asarray(diagonal, dtype=np.float32)
    u = np.asarray(left, dtype=np.float32)
    v = np.asarray(right, dtype=np.float32)
    c = np.asarray(additive, dtype=np.float32)
    scale = np.asarray(residual_scale, dtype=np.float32)
    if q.ndim != 4 or d.shape != q.shape:
        raise ValueError("read and diagonal factors must have shape [B,T,H,K]")
    if u.ndim != 5 or u.shape[:3] != q.shape[:3] or u.shape[-1] != q.shape[-1]:
        raise ValueError("left factors must have shape [B,T,H,R,K]")
    if v.shape != u.shape:
        raise ValueError("right factors must match left factors")
    if c.ndim != 5 or c.shape[:-1] != u.shape[:-1]:
        raise ValueError("additive factors must have shape [B,T,H,R,V]")
    if scale.shape != u.shape[:-1]:
        raise ValueError("residual scale must have shape [B,T,H,R]")
    return q, d, u, v, c, scale


def recover_affine_state_update(update: TensorExpression, state_name: str) -> RecoveredAffineStateUpdate:
    """Recover state-affine structure without matching a named recurrence."""
    state_inputs = _named_inputs(update, state_name)
    if not state_inputs:
        raise ValueError(f"state update does not reference {state_name!r}")
    state_axes = state_inputs[0].axes
    if any(value.axes != state_axes for value in state_inputs[1:]):
        raise ValueError("all references to the prior state must have identical axes")
    if update.axes != state_axes:
        raise ValueError("state update output axes must match prior-state axes")

    affine = _linearize(update, state_name)
    if not affine.state_terms:
        raise ValueError("state update must depend linearly on prior state")

    term_classes = tuple(_term_structure(term, state_axes) for term in affine.state_terms)
    has_low_rank = any(kind is StateTransitionStructure.DIAGONAL_PLUS_LOW_RANK for kind, _ in term_classes)
    has_general = any(kind is StateTransitionStructure.GENERAL_AFFINE for kind, _ in term_classes)
    if has_general:
        transition_structure = StateTransitionStructure.GENERAL_AFFINE
    elif has_low_rank:
        transition_structure = StateTransitionStructure.DIAGONAL_PLUS_LOW_RANK
    else:
        transition_structure = StateTransitionStructure.DIAGONAL

    diagonal_axes = _diagonal_scale_axes(affine.state_terms, state_axes)
    maximum_low_rank = max((rank for _, rank in term_classes), default=0)
    signatures = tuple(
        tuple(_map_signature(linear_map) for linear_map in term.maps) or ("identity",) for term in affine.state_terms
    )
    return RecoveredAffineStateUpdate(
        state_name=state_name,
        state_axes=state_axes,
        affine=affine,
        transition_structure=transition_structure,
        diagonal_scale_axes=diagonal_axes,
        maximum_low_rank=maximum_low_rank,
        term_signatures=signatures,
    )


def _linearize(expression: TensorExpression, state_name: str) -> AffineTensorExpression:
    if expression.kind is TensorExpressionKind.INPUT:
        if expression.name == state_name:
            return AffineTensorExpression(bias=None, state_terms=(StateLinearTerm(coefficient=1, maps=()),))
        return AffineTensorExpression(bias=expression, state_terms=())

    inputs = tuple(_linearize(value, state_name) for value in expression.inputs)
    if expression.kind in (TensorExpressionKind.ADD, TensorExpressionKind.SUBTRACT):
        left, right = inputs
        sign = -1 if expression.kind is TensorExpressionKind.SUBTRACT else 1
        terms = (*left.state_terms, *(_scale_term(term, sign) for term in right.state_terms))
        bias = _combine_bias(expression, left.bias, right.bias)
        return AffineTensorExpression(bias=bias, state_terms=terms)

    dependent_indices = tuple(index for index, value in enumerate(inputs) if value.state_terms)
    if expression.kind is TensorExpressionKind.UNARY:
        if dependent_indices and expression.operation not in ("convert", "negate"):
            raise ValueError(f"nonlinear unary operation {expression.operation!r} depends on prior state")
    elif expression.kind is TensorExpressionKind.MULTIPLY:
        if len(dependent_indices) > 1:
            raise ValueError("multiplication of two state-dependent values is nonlinear")
    elif expression.kind is TensorExpressionKind.CONTRACT and len(dependent_indices) > 1:
        raise ValueError("contraction of multiple state-dependent operands is nonlinear")

    if not dependent_indices:
        return AffineTensorExpression(bias=expression, state_terms=())
    dependent_index = dependent_indices[0]
    dependent = inputs[dependent_index]
    independent = tuple(value for index, value in enumerate(expression.inputs) if index != dependent_index)
    linear_map = AppliedLinearMap(
        kind=expression.kind,
        operation=expression.operation,
        independent_inputs=independent,
        dependent_input_index=dependent_index,
        input_axes=expression.inputs[dependent_index].axes,
        output_axes=expression.axes,
        reduction_axes=expression.reduction_axes,
    )
    terms = tuple(replace(term, maps=(*term.maps, linear_map)) for term in dependent.state_terms)
    bias = _map_bias(expression, dependent_index, dependent.bias)
    return AffineTensorExpression(bias=bias, state_terms=terms)


def _combine_bias(
    expression: TensorExpression,
    left: TensorExpression | None,
    right: TensorExpression | None,
) -> TensorExpression | None:
    if left is None:
        if right is None:
            return None
        if expression.kind is TensorExpressionKind.ADD:
            return right
        return TensorExpression(
            kind=TensorExpressionKind.UNARY,
            axes=right.axes,
            inputs=(right,),
            operation="negate",
        )
    if right is None:
        return left
    return replace(expression, inputs=(left, right))


def _map_bias(
    expression: TensorExpression,
    dependent_index: int,
    dependent_bias: TensorExpression | None,
) -> TensorExpression | None:
    if dependent_bias is None:
        return None
    inputs = list(expression.inputs)
    inputs[dependent_index] = dependent_bias
    return replace(expression, inputs=tuple(inputs))


def _scale_term(term: StateLinearTerm, coefficient: int) -> StateLinearTerm:
    return replace(term, coefficient=term.coefficient * coefficient)


def _term_structure(
    term: StateLinearTerm,
    state_axes: tuple[LogicalAxis, ...],
) -> tuple[StateTransitionStructure, int]:
    if all(linear_map.kind in (TensorExpressionKind.MULTIPLY, TensorExpressionKind.UNARY) for linear_map in term.maps):
        return StateTransitionStructure.DIAGONAL, 0

    contractions = tuple(linear_map for linear_map in term.maps if linear_map.kind is TensorExpressionKind.CONTRACT)
    if len(contractions) < 2 or term.maps[-1].output_axes != state_axes:
        return StateTransitionStructure.GENERAL_AFFINE, 0
    bottleneck = min(contractions, key=lambda linear_map: len(set(linear_map.output_axes) & set(state_axes)))
    missing_state_axes = set(state_axes) - set(bottleneck.output_axes)
    if not missing_state_axes:
        return StateTransitionStructure.GENERAL_AFFINE, 0
    restored = all(axis in contractions[-1].output_axes for axis in missing_state_axes)
    if not restored:
        return StateTransitionStructure.GENERAL_AFFINE, 0
    rank_axes = tuple(axis for axis in bottleneck.output_axes if axis not in state_axes)
    rank = reduce(mul, (axis.extent for axis in rank_axes), 1)
    return StateTransitionStructure.DIAGONAL_PLUS_LOW_RANK, rank


def _diagonal_scale_axes(
    terms: tuple[StateLinearTerm, ...],
    state_axes: tuple[LogicalAxis, ...],
) -> tuple[LogicalAxis, ...]:
    axes: list[LogicalAxis] = []
    for term in terms:
        kind, _ = _term_structure(term, state_axes)
        if kind is not StateTransitionStructure.DIAGONAL:
            continue
        for linear_map in term.maps:
            for value in linear_map.independent_inputs:
                for axis in value.axes:
                    if axis in state_axes and axis not in axes:
                        axes.append(axis)
    return tuple(axes)


def _map_signature(linear_map: AppliedLinearMap) -> str:
    if linear_map.kind is TensorExpressionKind.UNARY:
        return f"unary:{linear_map.operation}"
    if linear_map.kind is TensorExpressionKind.CONTRACT:
        reduced = ",".join(axis.label or str(axis.id) for axis in linear_map.reduction_axes)
        return f"contract[{reduced}]"
    return linear_map.kind.value


def _named_inputs(expression: TensorExpression, name: str) -> tuple[TensorExpression, ...]:
    matches: list[TensorExpression] = []
    if expression.kind is TensorExpressionKind.INPUT and expression.name == name:
        matches.append(expression)
    for value in expression.inputs:
        matches.extend(_named_inputs(value, name))
    return tuple(matches)
