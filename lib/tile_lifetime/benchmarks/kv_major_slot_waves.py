# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic launch planning and CPU semantics for KV-major slot waves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tile_lifetime import (
    AttentionPartial,
    RelationPlan,
    finalize_attention_partial,
    merge_attention_partials,
    summarize_attention_partial,
)


@dataclass(frozen=True)
class SlotWave:
    """One selected-slot launch ordered by destination KV block then query block."""

    selected_slot: int
    query_blocks: np.ndarray
    key_value_blocks: np.ndarray
    destination_blocks: np.ndarray
    destination_edge_offsets: np.ndarray

    @property
    def edge_count(self) -> int:
        """Number of valid relation edges in this wave."""
        return int(self.query_blocks.size)


@dataclass(frozen=True)
class SlotWaveSchedule:
    """Bounded deterministic launches for source-ordered online attention updates."""

    query_block_count: int
    key_value_block_count: int
    selected_slot_count: int
    edge_order: str
    waves: tuple[SlotWave, ...]

    @property
    def edge_count(self) -> int:
        """Total valid edges across all waves."""
        return sum(wave.edge_count for wave in self.waves)


def build_slot_wave_schedule(
    relation: RelationPlan, *, require_causal: bool = True, edge_order: str = "kv_major"
) -> SlotWaveSchedule:
    """Orient every selected-slot wave by KV block without changing slot order."""
    if edge_order not in ("kv_major", "source"):
        raise ValueError(f"unsupported slot-wave edge order: {edge_order}")
    if np.any(np.count_nonzero(relation.edge_valid, axis=1) == 0):
        raise ValueError("every query block must select at least one KV block")

    waves = []
    for selected_slot in range(relation.route_slots):
        query_blocks = np.flatnonzero(relation.edge_valid[:, selected_slot]).astype(np.int32, copy=False)
        route_indices = query_blocks * relation.route_slots + selected_slot
        key_value_blocks = relation.destination_item[route_indices].astype(np.int32, copy=False)
        if require_causal and np.any(key_value_blocks > query_blocks):
            raise ValueError(f"selected slot {selected_slot} contains a future KV block")

        order = (
            np.lexsort((query_blocks, key_value_blocks)) if edge_order == "kv_major" else np.arange(query_blocks.size)
        )
        ordered_queries = np.ascontiguousarray(query_blocks[order])
        ordered_key_values = np.ascontiguousarray(key_value_blocks[order])
        if np.unique(ordered_queries).size != ordered_queries.size:
            raise ValueError(f"selected slot {selected_slot} has more than one edge for a query block")

        if ordered_key_values.size:
            run_starts = np.concatenate(
                (
                    np.zeros(1, dtype=np.int32),
                    (np.flatnonzero(ordered_key_values[1:] != ordered_key_values[:-1]) + 1).astype(np.int32),
                )
            )
            destination_blocks = ordered_key_values[run_starts]
            destination_offsets = np.concatenate((run_starts, np.asarray([ordered_key_values.size], dtype=np.int32)))
        else:
            destination_blocks = np.empty(0, dtype=np.int32)
            destination_offsets = np.zeros(1, dtype=np.int32)
        waves.append(
            SlotWave(
                selected_slot=selected_slot,
                query_blocks=ordered_queries,
                key_value_blocks=ordered_key_values,
                destination_blocks=destination_blocks.astype(np.int32, copy=False),
                destination_edge_offsets=destination_offsets,
            )
        )

    schedule = SlotWaveSchedule(
        query_block_count=relation.source_item_count,
        key_value_block_count=relation.destination_count,
        selected_slot_count=relation.route_slots,
        edge_order=edge_order,
        waves=tuple(waves),
    )
    if schedule.edge_count != relation.route_count:
        raise ValueError(f"slot waves lost relation edges: {schedule.edge_count} != {relation.route_count}")
    return schedule


def execute_slot_wave_reference(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    schedule: SlotWaveSchedule,
    *,
    scale: float,
    causal: bool,
    sequence_length: int | None = None,
) -> np.ndarray:
    """Execute the wave schedule with one source-ordered FP32 state per query block."""
    query = np.asarray(query)
    key = np.asarray(key)
    value = np.asarray(value)
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Q/K/V must have shapes [block, token, head, feature]")
    if query.shape[0] != schedule.query_block_count or key.shape[0] != schedule.key_value_block_count:
        raise ValueError("Q/K block counts must match the slot-wave schedule")
    if key.shape[:3] != value.shape[:3] or query.shape[-1] != key.shape[-1]:
        raise ValueError("Q/K/V shapes are incompatible")
    if query.shape[2] % key.shape[2]:
        raise ValueError("query heads must map evenly onto KV heads")

    maximum_length = min(query.shape[0] * query.shape[1], key.shape[0] * key.shape[1])
    if sequence_length is None:
        sequence_length = maximum_length
    if sequence_length <= 0 or sequence_length > maximum_length:
        raise ValueError(f"sequence length must be in [1, {maximum_length}]")

    query_positions = np.arange(query.shape[0] * query.shape[1], dtype=np.int64).reshape(query.shape[:2])
    key_positions = np.arange(key.shape[0] * key.shape[1], dtype=np.int64).reshape(key.shape[:2])
    query_valid = query_positions < sequence_length
    key_valid = key_positions < sequence_length
    states: list[AttentionPartial | None] = [None] * schedule.query_block_count
    for wave in schedule.waves:
        for query_block, key_value_block in zip(wave.query_blocks, wave.key_value_blocks, strict=True):
            partial = summarize_attention_partial(
                query[int(query_block)],
                key[int(key_value_block)],
                value[int(key_value_block)],
                scale=scale,
                causal=causal,
                query_positions=query_positions[int(query_block)],
                key_positions=key_positions[int(key_value_block)],
                query_valid=query_valid[int(query_block)],
                key_valid=key_valid[int(key_value_block)],
            )
            previous = states[int(query_block)]
            states[int(query_block)] = partial if previous is None else merge_attention_partials(previous, partial)

    output = np.zeros((*query.shape[:3], value.shape[-1]), dtype=np.float32)
    for query_block, state in enumerate(states):
        if state is None:
            raise ValueError(f"query block {query_block} has no online state")
        valid_rows = query_valid[query_block]
        if not np.any(valid_rows):
            continue
        valid_state = AttentionPartial(
            row_max=state.row_max[valid_rows],
            row_sum_exp=state.row_sum_exp[valid_rows],
            weighted_value_accumulator=state.weighted_value_accumulator[valid_rows],
        )
        output[query_block, valid_rows] = finalize_attention_partial(valid_state)
    return output


def schedule_record(schedule: SlotWaveSchedule) -> dict[str, object]:
    """Return a JSON-compatible record of the exact bounded launch order."""
    return {
        "query_block_count": schedule.query_block_count,
        "key_value_block_count": schedule.key_value_block_count,
        "selected_slot_count": schedule.selected_slot_count,
        "edge_order": schedule.edge_order,
        "edge_count": schedule.edge_count,
        "waves": [
            {
                "selected_slot": wave.selected_slot,
                "edge_count": wave.edge_count,
                "query_blocks": wave.query_blocks.tolist(),
                "key_value_blocks": wave.key_value_blocks.tolist(),
                "destination_blocks": wave.destination_blocks.tolist(),
                "destination_edge_offsets": wave.destination_edge_offsets.tolist(),
            }
            for wave in schedule.waves
        ],
    }
