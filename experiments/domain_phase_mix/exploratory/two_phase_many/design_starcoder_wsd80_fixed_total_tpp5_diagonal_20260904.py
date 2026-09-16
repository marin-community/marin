# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Freeze the fixed-total-TPP StarCoder matched-compute diagonal ladder."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path

from levanter.models.qwen import Qwen3Config

from experiments.llama import llama3_tokenizer_vocab_size
from experiments.scaling_law_sweeps.completed_adamh import CompletedAdamHHeuristic

SEQ_LEN = 2048
BATCH_SIZE = 128
TOKENS_PER_STEP = SEQ_LEN * BATCH_SIZE
STEP_ALIGNMENT = 20
PHASE_0_FRACTION = 0.8
REFERENCE_SEED = 20_260_711
COMPUTE_MULTIPLIER = 3.0
TARGET_TOTAL_PARAMETER_TPP = 5.0
MAX_RELATIVE_TPP_MISMATCH = 0.002
MAX_RELATIVE_COMPUTE_MISMATCH = 0.03
TARGET_TRAINING_TOKENS = 5_729_908_864_777
DESIGN_VERSION = "2026-09-04-v1"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "starcoder_wsd80_fixed_total_tpp5_diagonal_design_20260904.json"
SOURCE_COMPUTE_DESIGN_PATH = (
    Path(__file__).resolve().parents[2] / "starcoder_wsd80_matched_nd_stage1_design_20260731.json"
)
CANARY_RUN_NAME = "tpp5_r3_h0896_l20_s09280_proportional"

# These are the four compute rungs shown in the existing B5 matched-scaling figure.
TARGET_COMPUTE_FLOPS = (
    8.797210250575872e17,
    1.6840365118901453e18,
    3.4458534426968064e18,
    6.507659145385083e18,
)

# Width, depth, and steps are frozen after a discrete search over Qwen configurations.
ARCHITECTURES = (
    (512, 14, 3_620),
    (640, 15, 5_000),
    (768, 17, 6_820),
    (896, 20, 9_280),
)

STARCODER_WEIGHTS = (
    0.0364194347695976,
    0.10,
    0.18,
    0.20,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.70,
    0.75,
    0.80,
    0.90,
)


@dataclass(frozen=True)
class CellSpec:
    """One model-and-token rung in the fixed-TPP ladder."""

    cell_id: str
    rung: int
    hidden_size: int
    num_layers: int
    natural_num_layers: int
    hidden_to_depth_ratio: float
    num_heads: int
    total_steps: int
    materialized_tokens: int
    total_parameters: int
    non_embedding_parameters: int
    total_parameter_tpp: float
    non_embedding_tpp: float
    flops_per_token: float
    compute_flops: float
    target_compute_flops: float
    relative_compute_mismatch: float


def _weight_slug(weight: float) -> str:
    return f"{weight:.4f}".replace(".", "p")


def _cell_id(rung: int, hidden_size: int, num_layers: int, total_steps: int) -> str:
    return f"r{rung}_h{hidden_size:04d}_l{num_layers:02d}_s{total_steps:05d}"


def _model_config(hidden_size: int, num_layers: int) -> Qwen3Config:
    heuristic = CompletedAdamHHeuristic()
    return replace(heuristic._build_model_config(hidden_size, seq_len=SEQ_LEN), num_layers=num_layers)


def source_compute_targets() -> tuple[float, ...]:
    """Read the four B5 rung targets from the experiment they summarize."""
    payload = json.loads(SOURCE_COMPUTE_DESIGN_PATH.read_text())
    targets: list[float] = []
    for rung in range(4):
        rung_cells = (cell for cell in payload["cells"] if int(cell["rung"]) == rung)
        rung_targets = {float(cell["target_compute_flops"]) for cell in rung_cells}
        if len(rung_targets) != 1:
            raise ValueError(f"Source design rung {rung} does not have one compute target: {rung_targets}")
        targets.append(rung_targets.pop())
    return tuple(targets)


def build_cells() -> tuple[CellSpec, ...]:
    """Build the four frozen fixed-TPP cells."""
    cells: list[CellSpec] = []
    for rung, ((hidden_size, num_layers, total_steps), target_compute) in enumerate(
        zip(ARCHITECTURES, TARGET_COMPUTE_FLOPS, strict=True)
    ):
        natural_model = CompletedAdamHHeuristic()._build_model_config(hidden_size, seq_len=SEQ_LEN)
        model = replace(natural_model, num_layers=num_layers)
        materialized_tokens = total_steps * TOKENS_PER_STEP
        total_parameters = model.total_trainable_params(llama3_tokenizer_vocab_size)
        non_embedding_parameters = model.total_trainable_params(0)
        flops_per_token = float(model.flops_per_token(llama3_tokenizer_vocab_size, SEQ_LEN))
        compute_flops = COMPUTE_MULTIPLIER * flops_per_token * materialized_tokens
        cells.append(
            CellSpec(
                cell_id=_cell_id(rung, hidden_size, num_layers, total_steps),
                rung=rung,
                hidden_size=hidden_size,
                num_layers=num_layers,
                natural_num_layers=natural_model.num_layers,
                hidden_to_depth_ratio=hidden_size / num_layers,
                num_heads=model.num_heads,
                total_steps=total_steps,
                materialized_tokens=materialized_tokens,
                total_parameters=total_parameters,
                non_embedding_parameters=non_embedding_parameters,
                total_parameter_tpp=materialized_tokens / total_parameters,
                non_embedding_tpp=materialized_tokens / non_embedding_parameters,
                flops_per_token=flops_per_token,
                compute_flops=compute_flops,
                target_compute_flops=target_compute,
                relative_compute_mismatch=compute_flops / target_compute - 1.0,
            )
        )
    return tuple(cells)


def validate_design(cells: tuple[CellSpec, ...]) -> None:
    """Reject drift from the reviewed ladder geometry."""
    if len(cells) != 4:
        raise ValueError(f"Expected four fixed-TPP cells, got {len(cells)}")
    if len(STARCODER_WEIGHTS) != 15 or len(set(STARCODER_WEIGHTS)) != 15:
        raise ValueError("The diagonal must contain fifteen unique StarCoder weights")
    if tuple(sorted(STARCODER_WEIGHTS)) != STARCODER_WEIGHTS:
        raise ValueError("StarCoder weights must be strictly increasing")
    if any(not 0.0 <= weight <= 1.0 for weight in STARCODER_WEIGHTS):
        raise ValueError("StarCoder weights must remain inside the simplex interval")
    if source_compute_targets() != TARGET_COMPUTE_FLOPS:
        raise ValueError("Target compute rungs drifted from the source B5 design")

    for cell in cells:
        if cell.total_steps % STEP_ALIGNMENT != 0:
            raise ValueError(f"Cell does not realize an exact 80/20 boundary: {cell.cell_id}")
        if abs(cell.total_parameter_tpp / TARGET_TOTAL_PARAMETER_TPP - 1.0) > MAX_RELATIVE_TPP_MISMATCH:
            raise ValueError(f"Cell exceeds the total-parameter TPP tolerance: {cell.cell_id}")
        if abs(cell.relative_compute_mismatch) > MAX_RELATIVE_COMPUTE_MISMATCH:
            raise ValueError(f"Cell exceeds the compute-matching tolerance: {cell.cell_id}")
        if cell.num_layers <= cell.natural_num_layers:
            raise ValueError(f"Cell no longer contains the reviewed depth override: {cell.cell_id}")

    for field in ("hidden_size", "num_layers", "total_parameters", "materialized_tokens", "compute_flops"):
        values = [getattr(cell, field) for cell in cells]
        if any(left >= right for left, right in pairwise(values)):
            raise ValueError(f"Fixed-TPP ladder is not strictly increasing in {field}: {values}")


def build_manifest() -> dict[str, object]:
    """Return the complete immutable launch manifest."""
    cells = build_cells()
    validate_design(cells)
    coordinates = [
        {
            "coordinate_id": "proportional" if index == 0 else f"p{_weight_slug(weight)}",
            "starcoder_weight": weight,
            "role": "proportional_tied_control" if index == 0 else "tied_diagonal",
        }
        for index, weight in enumerate(STARCODER_WEIGHTS)
    ]
    runs = [
        {
            "run_name": f"tpp5_{cell.cell_id}_{coordinate['coordinate_id']}",
            "cell_id": cell.cell_id,
            "coordinate_id": coordinate["coordinate_id"],
            "hidden_size": cell.hidden_size,
            "num_layers": cell.num_layers,
            "total_steps": cell.total_steps,
            "materialized_tokens": cell.materialized_tokens,
            "phase_0_starcoder": coordinate["starcoder_weight"],
            "phase_1_starcoder": coordinate["starcoder_weight"],
            "data_seed": REFERENCE_SEED,
            "simulated_epoch_subset_seed": REFERENCE_SEED,
        }
        for cell in cells
        for coordinate in coordinates
    ]
    if sum(run["run_name"] == CANARY_RUN_NAME for run in runs) != 1:
        raise ValueError("The stability canary must identify exactly one run")
    return {
        "design_version": DESIGN_VERSION,
        "description": "Self-contained matched-compute StarCoder diagonal ladder at fixed total-parameter TPP.",
        "source_compute_design": "starcoder_wsd80_matched_nd_stage1_design_20260731.json",
        "model_family": (
            "Qwen configurations with frozen width and depth chosen jointly to match total-parameter TPP "
            "and B5 compute; "
            "these are not direct extensions of the source design's CompletedAdamH depth heuristic"
        ),
        "phase_0_fraction": PHASE_0_FRACTION,
        "sequence_length": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "tokens_per_step": TOKENS_PER_STEP,
        "step_alignment": STEP_ALIGNMENT,
        "target_training_tokens": TARGET_TRAINING_TOKENS,
        "target_total_parameter_tpp": TARGET_TOTAL_PARAMETER_TPP,
        "max_relative_tpp_mismatch": MAX_RELATIVE_TPP_MISMATCH,
        "max_relative_compute_mismatch": MAX_RELATIVE_COMPUTE_MISMATCH,
        "reference_seed": REFERENCE_SEED,
        "canary_run_name": CANARY_RUN_NAME,
        "tpp_convention": "materialized_tokens / total_trainable_parameters_including_embeddings",
        "compute_convention": "3 * model_config.flops_per_token(llama3_vocab, seq_len) * materialized_tokens",
        "cell_count": len(cells),
        "coordinate_count_per_cell": len(coordinates),
        "expected_run_count": len(runs),
        "cells": [asdict(cell) for cell in cells],
        "coordinates": coordinates,
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
