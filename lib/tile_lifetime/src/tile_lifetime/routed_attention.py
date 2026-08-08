# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact computation over a selected binary relation and normalized Fold state."""

from dataclasses import dataclass

import numpy as np

from tile_lifetime.attention import (
    AttentionPartial,
    finalize_attention_partial,
    merge_attention_partials,
    summarize_attention_partial,
)
from tile_lifetime.relation import RelationPlan, build_relation_plan


@dataclass(frozen=True)
class IndexDomainRestriction:
    """One generic affine predicate limiting an index domain."""

    left_axis: str
    right_axis: str
    predicate: str

    def __post_init__(self) -> None:
        if self.predicate not in {"left_greater_equal_right", "all"}:
            raise ValueError(f"unsupported index-domain predicate {self.predicate!r}")


@dataclass(frozen=True)
class RelationSelectionProgram:
    """Generic Contract, domain restriction, and top-k relation construction."""

    left_input: str
    right_input: str
    left_count: int
    right_count: int
    feature_count: int
    selected_count: int
    restriction: IndexDomainRestriction
    accumulation_dtype: str = "fp32"

    def __post_init__(self) -> None:
        if (
            min(
                self.left_count,
                self.right_count,
                self.feature_count,
                self.selected_count,
            )
            <= 0
        ):
            raise ValueError("relation selection dimensions must be positive")
        if self.selected_count > self.right_count:
            raise ValueError("relation selection count exceeds the right domain")


def execute_relation_selection(
    program: RelationSelectionProgram,
    inputs: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Execute generic relation selection with deterministic descending top-k."""
    try:
        left = np.asarray(inputs[program.left_input], dtype=np.float32)
        right = np.asarray(inputs[program.right_input], dtype=np.float32)
    except KeyError as error:
        raise ValueError(f"missing relation-selection input {error.args[0]!r}") from error
    if left.shape != (program.left_count, program.feature_count):
        raise ValueError(
            f"left relation metadata has shape {left.shape}, expected " f"{(program.left_count, program.feature_count)}"
        )
    if right.shape != (program.right_count, program.feature_count):
        raise ValueError(
            f"right relation metadata has shape {right.shape}, expected "
            f"{(program.right_count, program.feature_count)}"
        )
    score = left @ right.T
    validity = np.ones(score.shape, dtype=np.bool_)
    if program.restriction.predicate == "left_greater_equal_right":
        validity &= np.arange(program.right_count)[None, :] <= np.arange(program.left_count)[:, None]
    score = np.where(validity, score, -np.inf)
    # Stable descending sort makes tie behavior deterministic without atomics.
    selected = np.argsort(-score, axis=1, kind="stable")[:, : program.selected_count].astype(np.int32)
    selected_valid = np.take_along_axis(validity, selected, axis=1)
    if np.any(~selected_valid):
        # Early rows in a causal relation can contain fewer legal right items.
        # Invalid slots are explicit Relation edges rather than padded duplicates.
        selected = np.where(selected_valid, selected, -1)
    return selected, selected_valid


def make_causal_block_relation(
    *, sequence_length: int, block_size: int, selected_blocks: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build evenly spaced historical selections including the first and current blocks."""
    if sequence_length <= 0 or block_size <= 0 or selected_blocks <= 0:
        raise ValueError("sequence length, block size, and selected-block count must be positive")
    block_count = (sequence_length + block_size - 1) // block_size
    selected = np.full((block_count, selected_blocks), -1, dtype=np.int32)
    edge_valid = np.zeros(selected.shape, dtype=np.bool_)
    for query_block in range(block_count):
        degree = min(selected_blocks, query_block + 1)
        chosen = np.unique(np.rint(np.linspace(0, query_block, degree)).astype(np.int32))
        selected[query_block, :degree] = chosen
        edge_valid[query_block, :degree] = True
    return selected, edge_valid


def build_routed_attention_relation(
    selected_kv_blocks: np.ndarray,
    *,
    edge_valid: np.ndarray | None = None,
    kv_rank_by_block: np.ndarray | None = None,
    kv_local_block_by_block: np.ndarray | None = None,
    padding_quantum: int = 1,
) -> RelationPlan:
    """Map query blocks and selection slots onto the generic relation index plane."""
    selected_kv_blocks = np.asarray(selected_kv_blocks)
    if selected_kv_blocks.ndim != 2:
        raise ValueError("selected KV blocks must have shape [query_block, selected_slot]")
    resolved_validity = _resolve_edge_validity(selected_kv_blocks, edge_valid)
    valid_blocks = selected_kv_blocks[resolved_validity]
    inferred_block_count = int(np.max(valid_blocks)) + 1 if valid_blocks.size else 0
    if kv_rank_by_block is None:
        kv_rank_by_block = np.zeros(inferred_block_count, dtype=np.int32)
    else:
        kv_rank_by_block = np.asarray(kv_rank_by_block)
    if kv_local_block_by_block is None:
        kv_local_block_by_block = np.arange(kv_rank_by_block.shape[0], dtype=np.int32)
    else:
        kv_local_block_by_block = np.asarray(kv_local_block_by_block)
    return build_relation_plan(
        selected_kv_blocks,
        np.ones(selected_kv_blocks.shape, dtype=np.float32),
        edge_valid=resolved_validity,
        destination_rank_by_item=kv_rank_by_block,
        destination_local_item_by_item=kv_local_block_by_block,
        padding_quantum=padding_quantum,
    )


def execute_query_major_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    selected_kv_blocks: np.ndarray,
    *,
    edge_valid: np.ndarray | None = None,
    scale: float,
    causal: bool,
    sequence_length: int | None = None,
) -> np.ndarray:
    """Keep each query block's online state while visiting selected KV blocks."""
    query, key, value, selected_kv_blocks, resolved_validity = _validate_routed_inputs(
        query, key, value, selected_kv_blocks, edge_valid=edge_valid, scale=scale
    )
    query_positions, key_positions, query_valid, key_valid = _blocked_positions(
        query, key, sequence_length=sequence_length
    )
    output = np.empty((*query.shape[:3], value.shape[-1]), dtype=np.float32)
    for query_block in range(query.shape[0]):
        state = _empty_attention_partial(query.shape[1], query.shape[2], value.shape[-1])
        for selected_slot in range(selected_kv_blocks.shape[1]):
            if not resolved_validity[query_block, selected_slot]:
                continue
            kv_block = int(selected_kv_blocks[query_block, selected_slot])
            partial = summarize_attention_partial(
                query[query_block],
                key[kv_block],
                value[kv_block],
                scale=scale,
                causal=causal,
                query_positions=query_positions[query_block],
                key_positions=key_positions[kv_block],
                query_valid=query_valid[query_block],
                key_valid=key_valid[kv_block],
            )
            state = merge_attention_partials(state, partial)
        output[query_block] = _finalize_valid_query_rows(state, query_valid[query_block])
    return output


def execute_kv_major_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    selected_kv_blocks: np.ndarray,
    *,
    edge_valid: np.ndarray | None = None,
    scale: float,
    causal: bool,
    padding_quantum: int,
    sequence_length: int | None = None,
) -> np.ndarray:
    """Group relation edges by KV block, then restore and merge state by query block."""
    query, key, value, selected_kv_blocks, resolved_validity = _validate_routed_inputs(
        query, key, value, selected_kv_blocks, edge_valid=edge_valid, scale=scale
    )
    relation = build_routed_attention_relation(
        selected_kv_blocks,
        edge_valid=resolved_validity,
        kv_rank_by_block=np.zeros(key.shape[0], dtype=np.int32),
        kv_local_block_by_block=np.arange(key.shape[0], dtype=np.int32),
        padding_quantum=padding_quantum,
    )
    query_positions, key_positions, query_valid, key_valid = _blocked_positions(
        query, key, sequence_length=sequence_length
    )
    row_max = np.full(
        (relation.destination_row_count, query.shape[1], query.shape[2]),
        -np.inf,
        dtype=np.float32,
    )
    row_sum_exp = np.zeros_like(row_max)
    weighted_value = np.zeros((*row_max.shape, value.shape[-1]), dtype=np.float32)
    for destination_row in np.flatnonzero(relation.row_valid):
        query_block = int(relation.row_source_item[destination_row])
        kv_block = int(relation.row_destination_item[destination_row])
        partial = summarize_attention_partial(
            query[query_block],
            key[kv_block],
            value[kv_block],
            scale=scale,
            causal=causal,
            query_positions=query_positions[query_block],
            key_positions=key_positions[kv_block],
            query_valid=query_valid[query_block],
            key_valid=key_valid[kv_block],
        )
        row_max[destination_row] = partial.row_max
        row_sum_exp[destination_row] = partial.row_sum_exp
        weighted_value[destination_row] = partial.weighted_value_accumulator

    restored_max = relation.inverse_dispatch(row_max, fill_value=-np.inf)
    restored_sum = relation.inverse_dispatch(row_sum_exp)
    restored_value = relation.inverse_dispatch(weighted_value)
    output = np.empty((*query.shape[:3], value.shape[-1]), dtype=np.float32)
    for query_block in range(query.shape[0]):
        state = _empty_attention_partial(query.shape[1], query.shape[2], value.shape[-1])
        for selected_slot in range(selected_kv_blocks.shape[1]):
            if not resolved_validity[query_block, selected_slot]:
                continue
            partial = AttentionPartial(
                row_max=restored_max[query_block, selected_slot],
                row_sum_exp=restored_sum[query_block, selected_slot],
                weighted_value_accumulator=restored_value[query_block, selected_slot],
            )
            state = merge_attention_partials(state, partial)
        output[query_block] = _finalize_valid_query_rows(state, query_valid[query_block])
    return output


def routed_attention_reference(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    selected_kv_blocks: np.ndarray,
    *,
    edge_valid: np.ndarray | None = None,
    scale: float,
    causal: bool,
    sequence_length: int | None = None,
) -> np.ndarray:
    """Compute selected-block attention directly, materializing each small reference score matrix."""
    query, key, value, selected_kv_blocks, resolved_validity = _validate_routed_inputs(
        query, key, value, selected_kv_blocks, edge_valid=edge_valid, scale=scale
    )
    query_positions, key_positions, query_valid, key_valid = _blocked_positions(
        query, key, sequence_length=sequence_length
    )
    head_map = _query_to_kv_head(query.shape[2], key.shape[2])
    output = np.empty((*query.shape[:3], value.shape[-1]), dtype=np.float32)
    for query_block in range(query.shape[0]):
        selected = selected_kv_blocks[query_block, resolved_validity[query_block]]
        if selected.size == 0:
            raise ValueError(f"attention rows have no valid selected keys in query block {query_block}")
        selected_key = np.concatenate([key[int(block)] for block in selected], axis=0).astype(np.float32)
        selected_value = np.concatenate([value[int(block)] for block in selected], axis=0).astype(np.float32)
        selected_positions = np.concatenate([key_positions[int(block)] for block in selected])
        selected_key_valid = np.concatenate([key_valid[int(block)] for block in selected])
        expanded_key = selected_key[:, head_map, :]
        scores = np.einsum("qhd,khd->qhk", query[query_block].astype(np.float32), expanded_key) * np.float32(scale)
        score_valid = query_valid[query_block, :, None] & selected_key_valid[None, :]
        if causal:
            score_valid &= query_positions[query_block, :, None] >= selected_positions[None, :]
        scores = np.where(score_valid[:, None, :], scores, -np.inf)
        maximum = np.max(scores, axis=-1)
        if np.any(~np.isfinite(maximum[query_valid[query_block]])):
            raise ValueError(f"attention rows have no valid selected keys in query block {query_block}")
        centered = np.full(scores.shape, -np.inf, dtype=np.float32)
        valid_rows = query_valid[query_block]
        centered[valid_rows] = scores[valid_rows] - maximum[valid_rows, :, None]
        probabilities = np.exp(centered)
        denominator = np.sum(probabilities, axis=-1, keepdims=True)
        probabilities[valid_rows] /= denominator[valid_rows]
        expanded_value = selected_value[:, head_map, :]
        output[query_block] = np.einsum("qhk,khv->qhv", probabilities, expanded_value)
    return output


def _empty_attention_partial(query_count: int, query_heads: int, value_dimension: int) -> AttentionPartial:
    return AttentionPartial(
        row_max=np.full((query_count, query_heads), -np.inf, dtype=np.float32),
        row_sum_exp=np.zeros((query_count, query_heads), dtype=np.float32),
        weighted_value_accumulator=np.zeros((query_count, query_heads, value_dimension), dtype=np.float32),
    )


def _validate_attention_block(
    query: np.ndarray, key: np.ndarray, value: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = np.asarray(query)
    key = np.asarray(key)
    value = np.asarray(value)
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("attention blocks must have shapes [token, head, feature]")
    if key.shape[:2] != value.shape[:2]:
        raise ValueError("key and value blocks must have matching token and head dimensions")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key feature dimensions must match")
    _query_to_kv_head(query.shape[1], key.shape[1])
    if not np.isfinite(scale):
        raise ValueError("attention scale must be finite")
    return query, key, value


def _validate_routed_inputs(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    selected_kv_blocks: np.ndarray,
    *,
    edge_valid: np.ndarray | None,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    query = np.asarray(query)
    key = np.asarray(key)
    value = np.asarray(value)
    selected_kv_blocks = np.asarray(selected_kv_blocks)
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("routed Q/K/V must have shapes [block, token, head, feature]")
    _validate_attention_block(query[0], key[0], value[0], scale)
    if key.shape[:3] != value.shape[:3]:
        raise ValueError("routed key and value tensors must have matching block, token, and head dimensions")
    if selected_kv_blocks.ndim != 2 or selected_kv_blocks.shape[0] != query.shape[0]:
        raise ValueError("selected KV blocks must have shape [query_block, selected_slot]")
    resolved_validity = _resolve_edge_validity(selected_kv_blocks, edge_valid)
    selected_valid = selected_kv_blocks[resolved_validity]
    if np.any(selected_valid < 0) or np.any(selected_valid >= key.shape[0]):
        raise ValueError("selected KV block is outside the available key/value blocks")
    for query_block in range(query.shape[0]):
        blocks = selected_kv_blocks[query_block, resolved_validity[query_block]]
        if np.unique(blocks).shape[0] != blocks.shape[0]:
            raise ValueError(f"query block {query_block} has duplicate selected KV blocks")
    return query, key, value, selected_kv_blocks, resolved_validity


def _resolve_edge_validity(selected_kv_blocks: np.ndarray, edge_valid: np.ndarray | None) -> np.ndarray:
    if edge_valid is None:
        return selected_kv_blocks >= 0
    resolved = np.asarray(edge_valid, dtype=np.bool_)
    if resolved.shape != selected_kv_blocks.shape:
        raise ValueError("edge validity must match selected-KV-block shape")
    return resolved


def _blocked_positions(
    query: np.ndarray, key: np.ndarray, *, sequence_length: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    query_positions = np.arange(query.shape[0] * query.shape[1], dtype=np.int64).reshape(query.shape[:2])
    key_positions = np.arange(key.shape[0] * key.shape[1], dtype=np.int64).reshape(key.shape[:2])
    maximum_length = min(query_positions.size, key_positions.size)
    if sequence_length is None:
        sequence_length = maximum_length
    if sequence_length <= 0 or sequence_length > maximum_length:
        raise ValueError(f"sequence length must be in [1, {maximum_length}], got {sequence_length}")
    return query_positions, key_positions, query_positions < sequence_length, key_positions < sequence_length


def _finalize_valid_query_rows(partial: AttentionPartial, query_valid: np.ndarray) -> np.ndarray:
    output = np.zeros(partial.weighted_value_accumulator.shape, dtype=np.float32)
    if np.any(query_valid):
        valid_partial = AttentionPartial(
            row_max=partial.row_max[query_valid],
            row_sum_exp=partial.row_sum_exp[query_valid],
            weighted_value_accumulator=partial.weighted_value_accumulator[query_valid],
        )
        output[query_valid] = finalize_attention_partial(valid_partial)
    return output


def _query_to_kv_head(query_head_count: int, kv_head_count: int) -> np.ndarray:
    if kv_head_count <= 0 or query_head_count % kv_head_count:
        raise ValueError("query heads must map evenly onto KV heads")
    return np.arange(query_head_count, dtype=np.int32) // (query_head_count // kv_head_count)
