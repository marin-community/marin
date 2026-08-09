# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Freeze Stage 1 of the matched-compute StarCoder WSD80 N-D surface grid."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.llama import llama3_tokenizer_vocab_size
from experiments.scaling_law_sweeps.completed_adamh import CompletedAdamHHeuristic

SEQ_LEN = 2048
BATCH_SIZE = 128
TOKENS_PER_STEP = SEQ_LEN * BATCH_SIZE
BASE_STEPS = 3820
STEP_ALIGNMENT = 20
PHASE_0_FRACTION = 0.8
REFERENCE_SEED = 20_260_711
COMPUTE_MULTIPLIER = 3.0
MAX_COMPUTE_MISMATCH = 0.002
DESIGN_VERSION = "2026-07-31-v2"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "starcoder_wsd80_matched_nd_stage1_design_20260731.json"

INCREASE_N_HIDDEN_SIZES = (640, 896, 1280, 1664)
INCREASE_ND_HIDDEN_SIZES = (640, 768, 896, 1024)


@dataclass(frozen=True)
class CoordinateSpec:
    """One common policy coordinate used in every N-D cell."""

    coordinate_id: str
    phase_0_starcoder: float
    phase_1_starcoder: float
    role: str

    @property
    def aggregate_starcoder(self) -> float:
        return PHASE_0_FRACTION * self.phase_0_starcoder + (1.0 - PHASE_0_FRACTION) * self.phase_1_starcoder

    @property
    def phase_contrast(self) -> float:
        return self.phase_1_starcoder - self.phase_0_starcoder


COORDINATES = (
    CoordinateSpec("diag_prop", 0.0364194347695976, 0.0364194347695976, "proportional_tied_control"),
    *(
        CoordinateSpec(f"diag_{index}", weight, weight, "tied_diagonal")
        for index, weight in enumerate((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), start=1)
    ),
    CoordinateSpec("diag_agg018", 0.18, 0.18, "aggregate_matched_tied_control"),
    CoordinateSpec("diag_agg035", 0.35, 0.35, "aggregate_matched_tied_control"),
    CoordinateSpec("diag_agg075", 0.75, 0.75, "aggregate_matched_tied_control"),
    CoordinateSpec("off_low_d040", 0.10, 0.50, "known_low_aggregate_late_code"),
    CoordinateSpec("off_low_d080", 0.02, 0.82, "aggressive_low_aggregate_late_code"),
    CoordinateSpec("off_mid_plus", 0.31, 0.51, "mid_aggregate_late_code"),
    CoordinateSpec("off_mid_minus", 0.39, 0.19, "mid_aggregate_early_code_control"),
    CoordinateSpec("off_high_plus", 0.72, 0.87, "high_aggregate_late_code"),
)


@dataclass(frozen=True)
class CellSpec:
    """One unique architecture and token-count cell shared by one or more tracks."""

    cell_id: str
    rung: int
    track_memberships: tuple[str, ...]
    hidden_size: int
    num_layers: int
    num_heads: int
    total_steps: int
    materialized_tokens: int
    total_parameters: int
    non_embedding_parameters: int
    flops_per_token: float
    compute_flops: float
    target_compute_flops: float
    relative_compute_mismatch: float


def _aligned_steps(raw_steps: float) -> int:
    return max(STEP_ALIGNMENT, round(raw_steps / STEP_ALIGNMENT) * STEP_ALIGNMENT)


def _cell_id(rung: int, track_memberships: tuple[str, ...], hidden_size: int, total_steps: int) -> str:
    track = "shared" if len(track_memberships) > 1 else track_memberships[0]
    return f"r{rung}_{track}_h{hidden_size:04d}_s{total_steps:05d}"


def build_cells() -> tuple[CellSpec, ...]:
    """Build ten unique cells across the N-only, D-only, and joint tracks."""
    heuristic = CompletedAdamHHeuristic()
    hidden_sizes = sorted(set(INCREASE_N_HIDDEN_SIZES + INCREASE_ND_HIDDEN_SIZES))
    models = {hidden: heuristic._build_model_config(hidden, seq_len=SEQ_LEN) for hidden in hidden_sizes}
    flops_per_token = {
        hidden: float(model.flops_per_token(llama3_tokenizer_vocab_size, SEQ_LEN)) for hidden, model in models.items()
    }
    base_tokens = BASE_STEPS * TOKENS_PER_STEP
    target_compute = {
        rung: COMPUTE_MULTIPLIER * flops_per_token[hidden] * base_tokens
        for rung, hidden in enumerate(INCREASE_N_HIDDEN_SIZES)
    }

    requested: list[tuple[int, str, int, int]] = []
    for rung, target_hidden in enumerate(INCREASE_N_HIDDEN_SIZES):
        requested.append((rung, "increase_n", target_hidden, BASE_STEPS))
        fixed_n_steps = _aligned_steps(
            target_compute[rung] / (COMPUTE_MULTIPLIER * flops_per_token[INCREASE_N_HIDDEN_SIZES[0]] * TOKENS_PER_STEP)
        )
        requested.append((rung, "increase_d", INCREASE_N_HIDDEN_SIZES[0], fixed_n_steps))
        joint_hidden = INCREASE_ND_HIDDEN_SIZES[rung]
        joint_steps = _aligned_steps(
            target_compute[rung] / (COMPUTE_MULTIPLIER * flops_per_token[joint_hidden] * TOKENS_PER_STEP)
        )
        requested.append((rung, "increase_nd", joint_hidden, joint_steps))

    grouped: dict[tuple[int, int, int], list[str]] = {}
    for rung, track, hidden, steps in requested:
        grouped.setdefault((rung, hidden, steps), []).append(track)

    cells: list[CellSpec] = []
    for (rung, hidden, steps), tracks in grouped.items():
        model = models[hidden]
        materialized_tokens = steps * TOKENS_PER_STEP
        compute_flops = COMPUTE_MULTIPLIER * flops_per_token[hidden] * materialized_tokens
        mismatch = compute_flops / target_compute[rung] - 1.0
        track_memberships = tuple(sorted(tracks))
        cells.append(
            CellSpec(
                cell_id=_cell_id(rung, track_memberships, hidden, steps),
                rung=rung,
                track_memberships=track_memberships,
                hidden_size=hidden,
                num_layers=model.num_layers,
                num_heads=model.num_heads,
                total_steps=steps,
                materialized_tokens=materialized_tokens,
                total_parameters=model.total_trainable_params(llama3_tokenizer_vocab_size),
                non_embedding_parameters=model.total_trainable_params(0),
                flops_per_token=flops_per_token[hidden],
                compute_flops=compute_flops,
                target_compute_flops=target_compute[rung],
                relative_compute_mismatch=mismatch,
            )
        )
    cells.sort(key=lambda item: (item.rung, item.cell_id))
    return tuple(cells)


def validate_design(cells: tuple[CellSpec, ...]) -> None:
    if len(cells) != 10:
        raise ValueError(f"Expected ten unique N-D cells, got {len(cells)}")
    if len(COORDINATES) != 18 or len({item.coordinate_id for item in COORDINATES}) != 18:
        raise ValueError("Stage 1 must contain eighteen unique common coordinates")
    tied_coordinates = tuple(item for item in COORDINATES if item.phase_0_starcoder == item.phase_1_starcoder)
    if len(tied_coordinates) != 13:
        raise ValueError("Stage 1 must contain thirteen tied coordinates")
    tied_aggregates = {round(item.aggregate_starcoder, 12) for item in tied_coordinates}
    unmatched_off_diagonal = [
        item.coordinate_id
        for item in COORDINATES
        if item.phase_0_starcoder != item.phase_1_starcoder
        and round(item.aggregate_starcoder, 12) not in tied_aggregates
    ]
    if unmatched_off_diagonal:
        raise ValueError(f"Off-diagonal coordinates lack exact tied controls: {unmatched_off_diagonal}")
    for coordinate in COORDINATES:
        if not 0.0 <= coordinate.phase_0_starcoder <= 1.0:
            raise ValueError(f"Invalid phase-0 weight: {coordinate}")
        if not 0.0 <= coordinate.phase_1_starcoder <= 1.0:
            raise ValueError(f"Invalid phase-1 weight: {coordinate}")
    for cell in cells:
        if cell.total_steps % STEP_ALIGNMENT != 0:
            raise ValueError(f"Cell does not have an exact 80/20 boundary: {cell.cell_id}")
        if abs(cell.relative_compute_mismatch) > MAX_COMPUTE_MISMATCH:
            raise ValueError(f"Cell exceeds the compute-matching tolerance: {cell.cell_id}")
    track_counts = {
        track: sum(track in cell.track_memberships for cell in cells)
        for track in ("increase_n", "increase_d", "increase_nd")
    }
    if track_counts != {"increase_n": 4, "increase_d": 4, "increase_nd": 4}:
        raise ValueError(f"Unexpected track coverage: {track_counts}")


def build_manifest() -> dict[str, object]:
    cells = build_cells()
    validate_design(cells)
    coordinate_rows = [
        {
            **asdict(coordinate),
            "aggregate_starcoder": coordinate.aggregate_starcoder,
            "phase_contrast": coordinate.phase_contrast,
        }
        for coordinate in COORDINATES
    ]
    runs = [
        {
            "run_name": f"s1_{cell.cell_id}_{coordinate.coordinate_id}",
            "cell_id": cell.cell_id,
            "coordinate_id": coordinate.coordinate_id,
            "hidden_size": cell.hidden_size,
            "total_steps": cell.total_steps,
            "materialized_tokens": cell.materialized_tokens,
            "phase_0_starcoder": coordinate.phase_0_starcoder,
            "phase_1_starcoder": coordinate.phase_1_starcoder,
            "data_seed": REFERENCE_SEED,
            "simulated_epoch_subset_seed": REFERENCE_SEED,
        }
        for cell in cells
        for coordinate in COORDINATES
    ]
    return {
        "design_version": DESIGN_VERSION,
        "description": "Common Stage-1 surface coverage for matched-compute StarCoder WSD80 N-D tracks.",
        "phase_0_fraction": PHASE_0_FRACTION,
        "sequence_length": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "tokens_per_step": TOKENS_PER_STEP,
        "step_alignment": STEP_ALIGNMENT,
        "cell_count": len(cells),
        "coordinate_count_per_cell": len(COORDINATES),
        "expected_run_count": len(runs),
        "compute_convention": "3 * model_config.flops_per_token(llama3_vocab, seq_len) * materialized_tokens",
        "cells": [asdict(cell) for cell in cells],
        "coordinates": coordinate_rows,
        "runs": runs,
    }


def main() -> None:
    manifest = build_manifest()
    OUTPUT_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output_path": str(OUTPUT_PATH),
                "cell_count": manifest["cell_count"],
                "coordinate_count_per_cell": manifest["coordinate_count_per_cell"],
                "expected_run_count": manifest["expected_run_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
