# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Launch Stage 1 of the matched-compute StarCoder WSD80 N-D surface grid."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep, lower, run
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint

from experiments.datasets.dolma import dolma_datasets
from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.llama import llama3_tokenizer, llama3_tokenizer_vocab_size
from experiments.scaling_law_sweeps.completed_adamh import CompletedAdamHHeuristic

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_wsd80_matched_nd_stage1_20260731"
WANDB_EXPERIMENT_TAG = "starcoder_wsd80_matched_nd_stage1"
PANEL_TAG = "matched_nd_stage1_20260731"
DESIGN_VERSION = "2026-07-31-v2"
DESIGN_PATH = Path(__file__).with_name("starcoder_wsd80_matched_nd_stage1_design_20260731.json")
EXPECTED_RUN_COUNT = 180
EXPECTED_CELL_COUNT = 10
EXPECTED_COORDINATE_COUNT = 18
DEFAULT_MAX_CONCURRENT = 64


@dataclass(frozen=True)
class Cell:
    """One unique model-size and token-count cell."""

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


@dataclass(frozen=True)
class Stage1Run:
    """One policy coordinate within an N-D cell."""

    run_name: str
    cell_id: str
    coordinate_id: str
    hidden_size: int
    total_steps: int
    materialized_tokens: int
    phase_0_starcoder: float
    phase_1_starcoder: float
    data_seed: int
    simulated_epoch_subset_seed: int


def load_design(selected_cells: frozenset[str] | None = None) -> tuple[dict[str, Cell], tuple[Stage1Run, ...]]:
    """Load and audit the immutable Stage-1 design."""
    payload = json.loads(DESIGN_PATH.read_text())
    if payload.get("design_version") != DESIGN_VERSION:
        raise ValueError(f"Unexpected design version in {DESIGN_PATH}")
    if payload.get("cell_count") != EXPECTED_CELL_COUNT:
        raise ValueError("Unexpected N-D cell count")
    if payload.get("coordinate_count_per_cell") != EXPECTED_COORDINATE_COUNT:
        raise ValueError("Unexpected coordinate count per cell")
    if payload.get("expected_run_count") != EXPECTED_RUN_COUNT:
        raise ValueError("Unexpected Stage-1 run count")

    cells = {
        str(row["cell_id"]): Cell(
            **{key: value for key, value in row.items() if key != "track_memberships"},
            track_memberships=tuple(row["track_memberships"]),
        )
        for row in payload["cells"]
    }
    raw_runs = tuple(Stage1Run(**row) for row in payload["runs"])
    if len(cells) != EXPECTED_CELL_COUNT or len(raw_runs) != EXPECTED_RUN_COUNT:
        raise ValueError("Manifest rows do not match declared counts")
    if len({item.run_name for item in raw_runs}) != len(raw_runs):
        raise ValueError("Stage-1 run names are not unique")
    if set(item.cell_id for item in raw_runs) != set(cells):
        raise ValueError("Stage-1 runs do not cover every declared cell")
    for cell_id in cells:
        cell_runs = [item for item in raw_runs if item.cell_id == cell_id]
        if len(cell_runs) != EXPECTED_COORDINATE_COUNT:
            raise ValueError(f"Cell {cell_id} has {len(cell_runs)} coordinates")
        if len({item.coordinate_id for item in cell_runs}) != EXPECTED_COORDINATE_COUNT:
            raise ValueError(f"Cell {cell_id} contains duplicate coordinates")
    for item in raw_runs:
        cell = cells[item.cell_id]
        if item.hidden_size != cell.hidden_size or item.total_steps != cell.total_steps:
            raise ValueError(f"Run/cell architecture mismatch for {item.run_name}")
        if item.materialized_tokens != cell.materialized_tokens:
            raise ValueError(f"Run/cell token mismatch for {item.run_name}")
        if not 0.0 <= item.phase_0_starcoder <= 1.0 or not 0.0 <= item.phase_1_starcoder <= 1.0:
            raise ValueError(f"Invalid mixture coordinate for {item.run_name}")

    if selected_cells is not None:
        unknown = selected_cells - set(cells)
        if unknown:
            raise ValueError(f"Unknown cells: {sorted(unknown)}")
        cells = {key: value for key, value in cells.items() if key in selected_cells}
        raw_runs = tuple(item for item in raw_runs if item.cell_id in selected_cells)
    return cells, raw_runs


def _validate_runtime_model(cell: Cell, heuristic: CompletedAdamHHeuristic) -> None:
    model = heuristic._build_model_config(cell.hidden_size, seq_len=base.SEQ_LEN)
    if model.num_layers != cell.num_layers or model.num_heads != cell.num_heads:
        raise ValueError(f"Generated architecture drifted for {cell.cell_id}")
    if model.total_trainable_params(llama3_tokenizer_vocab_size) != cell.total_parameters:
        raise ValueError(f"Generated parameter count drifted for {cell.cell_id}")
    flops_per_token = float(model.flops_per_token(llama3_tokenizer_vocab_size, base.SEQ_LEN))
    if flops_per_token != cell.flops_per_token:
        raise ValueError(f"Generated FLOPs/token drifted for {cell.cell_id}")
    compute_flops = 3.0 * flops_per_token * cell.materialized_tokens
    if compute_flops != cell.compute_flops:
        raise ValueError(f"Generated compute drifted for {cell.cell_id}")
    schedule = base._schedule_summary(cell.materialized_tokens)
    if schedule["total_steps"] != cell.total_steps:
        raise ValueError(f"Training-step drift for {cell.cell_id}")
    if schedule["realized_phase_0_fraction"] != base.PHASE_BOUNDARY:
        raise ValueError(f"Cell {cell.cell_id} does not realize an exact 80/20 boundary")


def build_training_steps(
    *,
    name_prefix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    selected_cells: frozenset[str] | None = None,
) -> tuple[ArtifactStep[LevanterCheckpoint], ...]:
    """Build resumable training handles for every selected Stage-1 row."""
    cells, requested_runs = load_design(selected_cells)
    heuristic = CompletedAdamHHeuristic()
    for cell in cells.values():
        _validate_runtime_model(cell, heuristic)

    nemotron = nemotron_datasets(tokenizer=llama3_tokenizer)
    starcoder = dolma_datasets(tokenizer=llama3_tokenizer)["dolma/starcoder"]
    training_handles: tuple[ArtifactStep[TokenizedCache], ...] = (
        *tuple(nemotron[split] for split in base.NEMOTRON_TOKEN_COUNTS),
        starcoder,
    )
    validation_handles = (
        *paloma_datasets(tokenizer=llama3_tokenizer).values(),
        *uncheatable_datasets(tokenizer=llama3_tokenizer).values(),
    )
    resources = ResourceConfig.with_tpu(tpu_type, regions=(tpu_region,), zone=tpu_zone)

    model_by_cell = {
        cell_id: heuristic._build_model_config(cell.hidden_size, seq_len=base.SEQ_LEN) for cell_id, cell in cells.items()
    }
    steps: list[ArtifactStep[LevanterCheckpoint]] = []
    for request in requested_runs:
        cell = cells[request.cell_id]
        phase_0_weights = base._phase_leaf_weights(
            request.phase_0_starcoder,
            nemotron=nemotron,
            starcoder=starcoder,
        )
        phase_1_weights = base._phase_leaf_weights(
            request.phase_1_starcoder,
            nemotron=nemotron,
            starcoder=starcoder,
        )
        static_weights = {handle: phase_0_weights[handle.name] for handle in training_handles}
        training = train_lm(
            name=f"checkpoints/{name_prefix}/{request.run_name}",
            version=base.VERSION,
            model=model_by_cell[request.cell_id],
            optimizer=base._optimizer(cell.materialized_tokens),
            datasets=static_weights,
            validation=validation_handles,
            batch_size=base.BATCH_SIZE,
            seq_len=base.SEQ_LEN,
            num_train_steps=cell.total_steps,
            z_loss_weight=None,
            evals=None,
            resources=resources,
            steps_per_eval=1_000,
            wandb_project="marin",
            wandb_group=name_prefix,
            run_id=request.run_name,
            tags=(
                WANDB_EXPERIMENT_TAG,
                request.run_name,
                request.cell_id,
                request.coordinate_id,
                "starcoder",
                "wsd80_20",
                PANEL_TAG,
            ),
            env_vars={"HF_ALLOW_CODE_EVAL": "1"},
        )
        steps.append(
            base._with_varying_mixture(
                training,
                train_datasets=static_weights,
                validation_datasets=validation_handles,
                phase_weights=[
                    (0, phase_0_weights),
                    (int(cell.total_steps * base.PHASE_BOUNDARY), phase_1_weights),
                ],
                data_seed=request.data_seed,
                simulated_epoch_subset_seed=request.simulated_epoch_subset_seed,
                experiment_budget=cell.materialized_tokens,
                target_budget=base.TARGET_BUDGET,
            )
        )
    if len(steps) != len(requested_runs):
        raise ValueError(f"Expected {len(requested_runs)} training handles, got {len(steps)}")
    return tuple(steps)


def _parse_cells(value: str | None) -> frozenset[str] | None:
    if value is None:
        return None
    cells = frozenset(item.strip() for item in value.split(",") if item.strip())
    if not cells:
        raise argparse.ArgumentTypeError("--cells must contain at least one cell ID")
    return cells


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--marin-prefix", default=base.DEFAULT_MARIN_PREFIX)
    parser.add_argument("--tpu-type", default=base.DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--cells", help="Comma-separated cell IDs for an idempotent partial retry")
    parser.add_argument("--audit-manifest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if os.getenv("CI") is not None:
        logger.info("Skipping StarCoder WSD80 matched-N,D Stage 1 in CI")
        return
    if args.marin_prefix != base.DEFAULT_MARIN_PREFIX:
        raise ValueError(f"This StarCoder experiment is central1-local: got {args.marin_prefix!r}")
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError(
            "StarCoder child TPU placement must remain central1-local: "
            f"got region={args.tpu_region!r}, zone={args.tpu_zone!r}"
        )

    selected_cells = _parse_cells(args.cells)
    cells, requested_runs = load_design(selected_cells)
    if not 1 <= args.max_concurrent <= len(requested_runs):
        raise ValueError(f"max_concurrent must be in [1, {len(requested_runs)}]")
    logger.info(
        "Prepared %d Stage-1 runs over %d N-D cells: %s",
        len(requested_runs),
        len(cells),
        {cell_id: sum(item.cell_id == cell_id for item in requested_runs) for cell_id in cells},
    )
    if args.audit_manifest:
        return

    os.environ["MARIN_PREFIX"] = args.marin_prefix
    steps = build_training_steps(
        name_prefix=args.name_prefix,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        selected_cells=selected_cells,
    )
    if args.dry_run:
        for step in steps:
            lower(step)
        logger.info("Dry-run lowering passed for all %d training handles", len(steps))
        return
    run(*steps, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
