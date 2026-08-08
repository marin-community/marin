# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact computation over a selected binary relation and normalized Fold state."""

from dataclasses import dataclass
from enum import StrEnum

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


class SelectionTieBreak(StrEnum):
    """Deterministic ordering among equal valid selection scores."""

    RIGHT_INDEX_ASCENDING = "right_index_ascending"


class SelectionOutputOrder(StrEnum):
    """Observable order of valid slots emitted by a Selection."""

    SCORE_DESCENDING = "score_descending"
    RIGHT_INDEX_ASCENDING = "right_index_ascending"


class UnderfilledSelectionPolicy(StrEnum):
    """Representation used when fewer than ``selected_count`` items are legal."""

    EXPLICIT_INVALID_SLOTS = "explicit_invalid_slots"


@dataclass(frozen=True)
class SelectionSemantics:
    """Source-visible ordering and underfilled-row contract for top-k Selection."""

    tie_break: SelectionTieBreak = SelectionTieBreak.RIGHT_INDEX_ASCENDING
    output_order: SelectionOutputOrder = SelectionOutputOrder.SCORE_DESCENDING
    underfilled_policy: UnderfilledSelectionPolicy = UnderfilledSelectionPolicy.EXPLICIT_INVALID_SLOTS
    invalid_index: int = -1

    def __post_init__(self) -> None:
        if self.invalid_index >= 0:
            raise ValueError("the explicit invalid Selection index must be negative")

    @property
    def scheduling_key(self) -> str:
        """Stable policy fragment used by diagnostics and physical legalization."""
        return (
            f"tie={self.tie_break.value}:output={self.output_order.value}:"
            f"underfilled={self.underfilled_policy.value}:invalid={self.invalid_index}"
        )


@dataclass(frozen=True)
class SelectionResult:
    """Rectangular Selection output with validity separate from index identity."""

    indices: np.ndarray
    valid: np.ndarray
    semantics: SelectionSemantics

    def __post_init__(self) -> None:
        if self.indices.shape != self.valid.shape:
            raise ValueError("Selection indices and validity must have identical shapes")
        if self.indices.dtype != np.int32:
            raise ValueError("Selection indices must be INT32")
        if self.valid.dtype != np.bool_:
            raise ValueError("Selection validity must be Boolean")
        if np.any(self.indices[self.valid] < 0):
            raise ValueError("valid Selection slots must contain nonnegative right indices")
        if np.any(self.indices[~self.valid] != self.semantics.invalid_index):
            raise ValueError("invalid Selection slots must contain the declared invalid index")

    @property
    def invalid(self) -> np.ndarray:
        """Boolean mask naming padded/invalid rectangular slots explicitly."""
        return ~self.valid


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
    selection_semantics: SelectionSemantics = SelectionSemantics()
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


@dataclass(frozen=True)
class ProjectedBlockSelectionProgram:
    """Generic projected-token Contract, block Fold, and deterministic Selection."""

    source_input: str
    left_weight_input: str
    right_weight_input: str
    source_count: int
    source_feature_count: int
    group_count: int
    relation_feature_count: int
    right_block_size: int
    selected_count: int
    score_scale: float
    token_restriction: IndexDomainRestriction
    force_local_block: bool
    selection_semantics: SelectionSemantics = SelectionSemantics()
    accumulation_dtype: str = "fp32"
    projection_output_dtype: str = "bf16"
    right_source_input: str | None = None
    right_source_feature_count: int | None = None
    right_count: int | None = None
    left_position_offset: int = 0
    right_position_offset: int = 0

    def __post_init__(self) -> None:
        dimensions = (
            self.source_count,
            self.source_feature_count,
            self.group_count,
            self.relation_feature_count,
            self.right_block_size,
            self.selected_count,
        )
        if min(dimensions) <= 0:
            raise ValueError("projected block-selection dimensions must be positive")
        if self.resolved_right_count % self.right_block_size:
            raise ValueError("right token count must be divisible by right block size")
        if self.selected_count > self.right_block_count:
            raise ValueError("selection count exceeds the right-block domain")
        if not np.isfinite(self.score_scale):
            raise ValueError("selection score scale must be finite")
        if self.projection_output_dtype not in {"bf16", "fp32"}:
            raise ValueError("projected selection supports BF16 or FP32 projected values")
        if self.token_restriction.predicate != "left_greater_equal_right":
            raise ValueError("projected block selection requires a causal token-domain restriction")
        left_start = self.left_position_offset
        left_stop = left_start + self.source_count
        right_start = self.right_position_offset
        right_stop = right_start + self.resolved_right_count
        if self.force_local_block and not (right_start <= left_start and left_stop <= right_stop):
            raise ValueError("forced-local selection requires the left position domain to lie inside the right domain")

    @property
    def resolved_right_source_input(self) -> str:
        """Right-token input, defaulting to the symmetric left input."""
        return self.source_input if self.right_source_input is None else self.right_source_input

    @property
    def resolved_right_source_feature_count(self) -> int:
        """Right-token feature count, defaulting to the left feature count."""
        if self.right_source_feature_count is None:
            return self.source_feature_count
        return self.right_source_feature_count

    @property
    def resolved_right_count(self) -> int:
        """Right-token count, defaulting to the symmetric left count."""
        return self.source_count if self.right_count is None else self.right_count

    @property
    def right_block_count(self) -> int:
        """Number of blocks in the projected right-token domain."""
        return self.resolved_right_count // self.right_block_size


def execute_relation_selection(
    program: RelationSelectionProgram,
    inputs: dict[str, np.ndarray],
) -> SelectionResult:
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
    return _execute_top_k_selection(
        score,
        validity,
        selected_count=program.selected_count,
        semantics=program.selection_semantics,
    )


def execute_projected_block_selection(
    program: ProjectedBlockSelectionProgram,
    inputs: dict[str, np.ndarray],
) -> SelectionResult:
    """Execute projected token scores, causal block max, and stable top-k."""
    try:
        source = np.asarray(inputs[program.source_input], dtype=np.float32)
        right_source = np.asarray(inputs[program.resolved_right_source_input], dtype=np.float32)
        left_weight = np.asarray(inputs[program.left_weight_input], dtype=np.float32)
        right_weight = np.asarray(inputs[program.right_weight_input], dtype=np.float32)
    except KeyError as error:
        raise ValueError(f"missing projected-selection input {error.args[0]!r}") from error
    expected_source = (program.source_count, program.source_feature_count)
    expected_right_source = (program.resolved_right_count, program.resolved_right_source_feature_count)
    expected_left_weight = (
        program.source_feature_count,
        program.group_count * program.relation_feature_count,
    )
    expected_right_weight = (program.resolved_right_source_feature_count, program.relation_feature_count)
    if source.shape != expected_source:
        raise ValueError(f"selection source has shape {source.shape}, expected {expected_source}")
    if right_source.shape != expected_right_source:
        raise ValueError(f"right selection source has shape {right_source.shape}, expected {expected_right_source}")
    if left_weight.shape != expected_left_weight:
        raise ValueError(f"left projection weight has shape {left_weight.shape}, expected {expected_left_weight}")
    if right_weight.shape != expected_right_weight:
        raise ValueError(f"right projection weight has shape {right_weight.shape}, expected {expected_right_weight}")

    left = (source @ left_weight).reshape(
        program.source_count,
        program.group_count,
        program.relation_feature_count,
    )
    right = right_source @ right_weight
    if program.projection_output_dtype == "bf16":
        left = _round_float32_to_bfloat16(left)
        right = _round_float32_to_bfloat16(right)
    token_scores = np.einsum("qhd,kd->qhk", left, right, dtype=np.float32) * np.float32(program.score_scale)
    query_position = np.arange(program.source_count, dtype=np.int32) + program.left_position_offset
    key_position = np.arange(program.resolved_right_count, dtype=np.int32) + program.right_position_offset
    token_valid = key_position[None, :] <= query_position[:, None]
    token_scores = np.where(token_valid[:, None, :], token_scores, -np.inf)
    block_scores = np.max(
        token_scores.reshape(
            program.source_count,
            program.group_count,
            program.right_block_count,
            program.right_block_size,
        ),
        axis=-1,
    )
    block_valid = (
        np.arange(program.right_block_count, dtype=np.int32)[None, :] * program.right_block_size
        <= query_position[:, None]
    )
    block_scores = np.where(block_valid[:, None, :], block_scores, -np.inf)
    if program.force_local_block:
        local_block = (query_position - program.right_position_offset) // program.right_block_size
        block_scores[np.arange(program.source_count), :, local_block] = np.inf

    return _execute_top_k_selection(
        block_scores,
        np.broadcast_to(block_valid[:, None, :], block_scores.shape),
        selected_count=program.selected_count,
        semantics=program.selection_semantics,
    )


def _execute_top_k_selection(
    score: np.ndarray,
    validity: np.ndarray,
    *,
    selected_count: int,
    semantics: SelectionSemantics,
) -> SelectionResult:
    """Apply the generic deterministic and underfilled Selection contract."""
    if score.shape != validity.shape:
        raise ValueError("Selection score and validity domains must have identical shapes")
    if semantics.tie_break is not SelectionTieBreak.RIGHT_INDEX_ASCENDING:
        raise ValueError(f"unsupported Selection tie break {semantics.tie_break.value}")
    if semantics.underfilled_policy is not UnderfilledSelectionPolicy.EXPLICIT_INVALID_SLOTS:
        raise ValueError(f"unsupported underfilled Selection policy {semantics.underfilled_policy.value}")

    masked_score = np.where(validity, score, -np.inf)
    # A stable descending sort visits the original right-index domain in
    # ascending order on ties, matching source chlo.top_k behavior.
    selected = np.argsort(-masked_score, axis=-1, kind="stable")[..., :selected_count]
    selected_valid = np.take_along_axis(validity, selected, axis=-1)
    selected = np.where(selected_valid, selected, semantics.invalid_index)
    if semantics.output_order is SelectionOutputOrder.RIGHT_INDEX_ASCENDING:
        sentinel = score.shape[-1]
        selected = np.sort(np.where(selected_valid, selected, sentinel), axis=-1)
        selected_valid = selected < sentinel
        selected = np.where(selected_valid, selected, semantics.invalid_index)
    elif semantics.output_order is not SelectionOutputOrder.SCORE_DESCENDING:
        raise ValueError(f"unsupported Selection output order {semantics.output_order.value}")
    return SelectionResult(
        indices=selected.astype(np.int32, copy=False),
        valid=selected_valid.astype(np.bool_, copy=False),
        semantics=semantics,
    )


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


def build_grouped_routed_attention_relation(
    selected_right_blocks: np.ndarray,
    *,
    edge_valid: np.ndarray | None = None,
    padding_quantum: int = 1,
) -> RelationPlan:
    """Flatten grouped selection slots while preserving one left-item domain.

    The rectangular route slot is ``group * selected_count + selected_slot``.
    This keeps semantic query-token and KV-block domains unchanged while making
    the GQA group recoverable from generic relation metadata.
    """
    selected_right_blocks = np.asarray(selected_right_blocks)
    if selected_right_blocks.ndim != 3:
        raise ValueError("grouped selected blocks must have shape [left, group, selected_slot]")
    resolved_validity = _resolve_edge_validity(selected_right_blocks, edge_valid)
    left_count, group_count, selected_count = selected_right_blocks.shape
    valid_blocks = selected_right_blocks[resolved_validity]
    right_block_count = int(np.max(valid_blocks)) + 1 if valid_blocks.size else 0
    grouped_destination = np.where(
        resolved_validity,
        selected_right_blocks,
        -1,
    ).reshape(left_count, group_count * selected_count)
    grouped_validity = resolved_validity.reshape(grouped_destination.shape)
    return build_relation_plan(
        grouped_destination,
        np.ones(grouped_destination.shape, dtype=np.float32),
        edge_valid=grouped_validity,
        destination_rank_by_item=np.zeros(right_block_count, dtype=np.int32),
        destination_local_item_by_item=np.arange(right_block_count, dtype=np.int32),
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


def _round_float32_to_bfloat16(value: np.ndarray) -> np.ndarray:
    """Round FP32 to BF16 round-to-nearest-even, retaining an FP32 container."""
    contiguous = np.ascontiguousarray(value, dtype=np.float32)
    bits = contiguous.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = (bits + rounding_bias) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)
