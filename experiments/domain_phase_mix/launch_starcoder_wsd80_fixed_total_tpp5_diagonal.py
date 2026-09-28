# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Launch the fixed-total-TPP StarCoder matched-compute diagonal ladder."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path

from fray.types import ResourceConfig
from levanter.models.qwen import Qwen3Config
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

NAME = "pinlin_calvin_xu/data_mixture/starcoder_wsd80_fixed_total_tpp5_diagonal_20260904"
WANDB_EXPERIMENT_TAG = "starcoder_wsd80_fixed_total_tpp5_diagonal"
PANEL_TAG = "b5_fixed_total_tpp5_20260904"
DESIGN_VERSION = "2026-09-04-v1"
DESIGN_PATH = Path(__file__).with_name("starcoder_wsd80_fixed_total_tpp5_diagonal_design_20260904.json")
EXPECTED_RUN_COUNT = 60
EXPECTED_CELL_COUNT = 4
EXPECTED_COORDINATE_COUNT = 15
EXPECTED_TARGET_TOTAL_PARAMETER_TPP = 5.0
MAX_RELATIVE_TPP_MISMATCH = 0.002
MAX_RELATIVE_COMPUTE_MISMATCH = 0.03
CANARY_RUN_NAME = "tpp5_r3_h0896_l20_s09280_proportional"
DEFAULT_MAX_CONCURRENT = EXPECTED_RUN_COUNT


@dataclass(frozen=True)
class Cell:
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


@dataclass(frozen=True)
class RunSpec:
    """One tied StarCoder mixture coordinate within a ladder cell."""

    run_name: str
    cell_id: str
    coordinate_id: str
    hidden_size: int
    num_layers: int
    total_steps: int
    materialized_tokens: int
    phase_0_starcoder: float
    phase_1_starcoder: float
    data_seed: int
    simulated_epoch_subset_seed: int


def _model_for_cell(cell: Cell, heuristic: CompletedAdamHHeuristic) -> Qwen3Config:
    return replace(
        heuristic._build_model_config(cell.hidden_size, seq_len=base.SEQ_LEN),
        num_layers=cell.num_layers,
    )


def audit_runtime_design(cells: dict[str, Cell]) -> None:
    """Check the frozen accounting against the current model implementation."""
    heuristic = CompletedAdamHHeuristic()
    for cell in cells.values():
        natural_model = heuristic._build_model_config(cell.hidden_size, seq_len=base.SEQ_LEN)
        model = _model_for_cell(cell, heuristic)
        if model.num_layers != cell.num_layers or model.num_heads != cell.num_heads:
            raise ValueError(f"Generated architecture drifted for {cell.cell_id}")
        if natural_model.num_layers != cell.natural_num_layers:
            raise ValueError(f"Natural depth accounting drifted for {cell.cell_id}")
        if not math.isclose(cell.hidden_to_depth_ratio, cell.hidden_size / cell.num_layers):
            raise ValueError(f"Aspect-ratio accounting drifted for {cell.cell_id}")
        if model.total_trainable_params(llama3_tokenizer_vocab_size) != cell.total_parameters:
            raise ValueError(f"Generated total parameter count drifted for {cell.cell_id}")
        if model.total_trainable_params(0) != cell.non_embedding_parameters:
            raise ValueError(f"Generated non-embedding parameter count drifted for {cell.cell_id}")
        flops_per_token = float(model.flops_per_token(llama3_tokenizer_vocab_size, base.SEQ_LEN))
        if flops_per_token != cell.flops_per_token:
            raise ValueError(f"Generated FLOPs/token drifted for {cell.cell_id}")
        if 3.0 * flops_per_token * cell.materialized_tokens != cell.compute_flops:
            raise ValueError(f"Generated compute drifted for {cell.cell_id}")
        schedule = base._schedule_summary(cell.materialized_tokens)
        if schedule["total_steps"] != cell.total_steps:
            raise ValueError(f"Training-step drift for {cell.cell_id}")
        if schedule["realized_phase_0_fraction"] != base.PHASE_BOUNDARY:
            raise ValueError(f"Cell {cell.cell_id} does not realize an exact 80/20 boundary")


def load_design(
    selected_cells: frozenset[str] | None = None,
    selected_runs: frozenset[str] | None = None,
) -> tuple[dict[str, Cell], tuple[RunSpec, ...]]:
    """Load and audit the immutable fixed-TPP design."""
    payload = json.loads(DESIGN_PATH.read_text())
    if payload.get("design_version") != DESIGN_VERSION:
        raise ValueError(f"Unexpected design version in {DESIGN_PATH}")
    if payload.get("cell_count") != EXPECTED_CELL_COUNT:
        raise ValueError("Unexpected fixed-TPP cell count")
    if payload.get("coordinate_count_per_cell") != EXPECTED_COORDINATE_COUNT:
        raise ValueError("Unexpected coordinate count")
    if payload.get("expected_run_count") != EXPECTED_RUN_COUNT:
        raise ValueError("Unexpected fixed-TPP run count")
    if payload.get("sequence_length") != base.SEQ_LEN:
        raise ValueError("Sequence length drifted from the StarCoder base experiment")
    if payload.get("batch_size") != base.BATCH_SIZE:
        raise ValueError("Batch size drifted from the StarCoder base experiment")
    if payload.get("tokens_per_step") != base.SEQ_LEN * base.BATCH_SIZE:
        raise ValueError("Tokens per step drifted from the StarCoder base experiment")
    if payload.get("phase_0_fraction") != base.PHASE_BOUNDARY:
        raise ValueError("Phase boundary drifted from the StarCoder base experiment")
    if payload.get("reference_seed") != base.DEFAULT_DATA_SEED:
        raise ValueError("Reference seed drifted from the StarCoder base experiment")
    if payload.get("target_training_tokens") != base.TARGET_BUDGET:
        raise ValueError("Simulated-epoching target budget drifted")
    if payload.get("target_total_parameter_tpp") != EXPECTED_TARGET_TOTAL_PARAMETER_TPP:
        raise ValueError("Unexpected total-parameter TPP target")
    if payload.get("max_relative_tpp_mismatch") != MAX_RELATIVE_TPP_MISMATCH:
        raise ValueError("Unexpected total-parameter TPP tolerance")
    if payload.get("max_relative_compute_mismatch") != MAX_RELATIVE_COMPUTE_MISMATCH:
        raise ValueError("Unexpected compute-matching tolerance")
    if payload.get("canary_run_name") != CANARY_RUN_NAME:
        raise ValueError("Unexpected stability canary")

    cells = {str(row["cell_id"]): Cell(**row) for row in payload["cells"]}
    all_runs = tuple(RunSpec(**row) for row in payload["runs"])
    if len(cells) != EXPECTED_CELL_COUNT or len(all_runs) != EXPECTED_RUN_COUNT:
        raise ValueError("Manifest rows do not match declared counts")
    if len({item.run_name for item in all_runs}) != EXPECTED_RUN_COUNT:
        raise ValueError("Run names are not unique")
    if set(item.cell_id for item in all_runs) != set(cells):
        raise ValueError("Runs do not cover every declared cell")

    for cell_id, cell in cells.items():
        realized_total_tpp = cell.materialized_tokens / cell.total_parameters
        realized_non_embedding_tpp = cell.materialized_tokens / cell.non_embedding_parameters
        realized_compute_mismatch = cell.compute_flops / cell.target_compute_flops - 1.0
        if not math.isclose(cell.total_parameter_tpp, realized_total_tpp):
            raise ValueError(f"Total-parameter TPP accounting drifted for {cell_id}")
        if not math.isclose(cell.non_embedding_tpp, realized_non_embedding_tpp):
            raise ValueError(f"Non-embedding TPP accounting drifted for {cell_id}")
        if abs(realized_total_tpp / EXPECTED_TARGET_TOTAL_PARAMETER_TPP - 1.0) > MAX_RELATIVE_TPP_MISMATCH:
            raise ValueError(f"Cell exceeds the total-parameter TPP tolerance: {cell_id}")
        if not math.isclose(cell.relative_compute_mismatch, realized_compute_mismatch):
            raise ValueError(f"Compute-mismatch accounting drifted for {cell_id}")
        if abs(realized_compute_mismatch) > MAX_RELATIVE_COMPUTE_MISMATCH:
            raise ValueError(f"Cell exceeds the compute-matching tolerance: {cell_id}")
        cell_runs = tuple(item for item in all_runs if item.cell_id == cell_id)
        if len(cell_runs) != EXPECTED_COORDINATE_COUNT:
            raise ValueError(f"Cell {cell_id} has {len(cell_runs)} coordinates")
        if len({item.coordinate_id for item in cell_runs}) != EXPECTED_COORDINATE_COUNT:
            raise ValueError(f"Cell {cell_id} contains duplicate coordinates")
        for item in cell_runs:
            if (
                item.hidden_size != cell.hidden_size
                or item.num_layers != cell.num_layers
                or item.total_steps != cell.total_steps
                or item.materialized_tokens != cell.materialized_tokens
            ):
                raise ValueError(f"Run/cell geometry mismatch for {item.run_name}")
            if item.phase_0_starcoder != item.phase_1_starcoder:
                raise ValueError(f"Run is not on the tied diagonal: {item.run_name}")
            if not 0.0 <= item.phase_0_starcoder <= 1.0:
                raise ValueError(f"Invalid mixture coordinate for {item.run_name}")

    audit_runtime_design(cells)

    if selected_cells is not None:
        unknown_cells = selected_cells - set(cells)
        if unknown_cells:
            raise ValueError(f"Unknown cells: {sorted(unknown_cells)}")
        cells = {cell_id: cell for cell_id, cell in cells.items() if cell_id in selected_cells}
        all_runs = tuple(item for item in all_runs if item.cell_id in selected_cells)
    if selected_runs is not None:
        unknown_runs = selected_runs - {item.run_name for item in all_runs}
        if unknown_runs:
            raise ValueError(f"Unknown runs: {sorted(unknown_runs)}")
        all_runs = tuple(item for item in all_runs if item.run_name in selected_runs)
        selected_cell_ids = {item.cell_id for item in all_runs}
        cells = {cell_id: cell for cell_id, cell in cells.items() if cell_id in selected_cell_ids}
    return cells, all_runs


def build_training_steps(
    *,
    name_prefix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    selected_cells: frozenset[str] | None = None,
    selected_runs: frozenset[str] | None = None,
) -> tuple[ArtifactStep[LevanterCheckpoint], ...]:
    """Build resumable training handles for the selected design rows."""
    cells, requested_runs = load_design(selected_cells, selected_runs)
    heuristic = CompletedAdamHHeuristic()
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
    model_by_cell = {cell_id: _model_for_cell(cell, heuristic) for cell_id, cell in cells.items()}

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
                "fixed_total_tpp5",
                "matched_compute_joint_nd",
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
                    (int(base._schedule_summary(cell.materialized_tokens)["boundary_step"]), phase_1_weights),
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


def _parse_selection(value: str | None) -> frozenset[str] | None:
    if value is None:
        return None
    selection = frozenset(item.strip() for item in value.split(",") if item.strip())
    if not selection:
        raise argparse.ArgumentTypeError("Selection must contain at least one ID")
    return selection


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--marin-prefix", default=base.DEFAULT_MARIN_PREFIX)
    parser.add_argument("--tpu-type", default=base.DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--cells", help="Comma-separated cell IDs for a partial retry")
    selection.add_argument("--runs", help="Comma-separated run names for a partial retry")
    parser.add_argument(
        "--launch-mode",
        choices=("staged", "fanout"),
        default="staged",
        help="Run the largest proportional stability canary before fan-out, or launch all selected rows immediately",
    )
    parser.add_argument("--audit-manifest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if os.getenv("CI") is not None:
        logger.info("Skipping the StarCoder fixed-TPP diagonal ladder in CI")
        return
    if args.marin_prefix != base.DEFAULT_MARIN_PREFIX:
        raise ValueError(f"This StarCoder experiment is central1-local: got {args.marin_prefix!r}")
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError(
            "StarCoder child TPU placement must remain central1-local: "
            f"got region={args.tpu_region!r}, zone={args.tpu_zone!r}"
        )

    selected_cells = _parse_selection(args.cells)
    selected_runs = _parse_selection(args.runs)
    cells, requested_runs = load_design(selected_cells, selected_runs)
    if args.max_concurrent < 1:
        raise ValueError("max_concurrent must be positive")
    max_concurrent = min(args.max_concurrent, len(requested_runs))
    logger.info(
        "Prepared %d fixed-total-TPP runs over %d cells: %s",
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
        selected_runs=selected_runs,
    )
    if args.dry_run:
        for step in steps:
            lower(step)
        logger.info("Dry-run lowering passed for all %d training handles", len(steps))
        return

    canary_candidates = (
        step for request, step in zip(requested_runs, steps, strict=True) if request.run_name == CANARY_RUN_NAME
    )
    canary = next(canary_candidates, None)
    if args.launch_mode == "staged" and canary is not None:
        logger.info("Running stability canary before fan-out: %s", CANARY_RUN_NAME)
        run(canary, max_concurrent=1)
        remaining = tuple(
            step for request, step in zip(requested_runs, steps, strict=True) if request.run_name != CANARY_RUN_NAME
        )
        if remaining:
            run(*remaining, max_concurrent=min(max_concurrent, len(remaining)))
        return
    run(*steps, max_concurrent=max_concurrent)


if __name__ == "__main__":
    main()
