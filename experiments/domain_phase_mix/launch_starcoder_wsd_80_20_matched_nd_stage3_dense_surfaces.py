# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the frozen Stage-3 dense-surface expansion for matched N-D WSD80."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep, lower, run
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint

from experiments.datasets.dolma import dolma_datasets
from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_matched_nd_stage1 as stage1
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_starcoder_wsd80_matched_nd_stage3_dense_surfaces_20260802 as frozen_designer,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    starcoder_wsd80_training_identity as stream_identity,
)
from experiments.llama import llama3_tokenizer
from experiments.scaling_law_sweeps.completed_adamh import CompletedAdamHHeuristic

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_wsd80_matched_nd_stage3_dense_20260802"
WANDB_EXPERIMENT_TAG = "starcoder_wsd80_matched_nd_stage3"
PANEL_TAG = "matched_nd_stage3_dense_20260802"
DESIGN_VERSION = "2026-08-02"
DESIGN_PATH = Path(__file__).with_name("starcoder_wsd80_matched_nd_stage3_dense_surface_design_20260802.json")
EXPECTED_UNTIED_RUN_COUNT = 480
EXPECTED_TIED_ANCHOR_COUNT = 4
EXPECTED_RUN_COUNT = EXPECTED_UNTIED_RUN_COUNT + EXPECTED_TIED_ANCHOR_COUNT
EXPECTED_CELL_COUNT = 10
UNTIED_RUNS_PER_CELL = 48
DEFAULT_MAX_CONCURRENT = 64


@dataclass(frozen=True)
class Stage3Run:
    """One frozen systematic surface coordinate within an N-D cell."""

    run_name: str
    cell_id: str
    acquisition_kind: str
    phase_0_starcoder: float
    phase_1_starcoder: float
    boundary_step: int
    data_seed: int
    simulated_epoch_subset_seed: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_design_payload() -> dict[str, Any]:
    payload = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    if payload.get("design_version") != DESIGN_VERSION:
        raise ValueError(f"Unexpected Stage-3 design version in {DESIGN_PATH}")
    expected_environment = {
        "tpu_type": base.DEFAULT_TPU_TYPE,
        "tpu_region": base.DEFAULT_TPU_REGION,
        "tpu_zone": base.DEFAULT_TPU_ZONE,
        "marin_prefix": base.DEFAULT_MARIN_PREFIX,
    }
    environment = payload.get("training_environment", {})
    if any(environment.get(key) != value for key, value in expected_environment.items()):
        raise ValueError("Frozen Stage-3 environment does not match the historical WSD80 environment")
    hashes = payload.get("data_use", {}).get("source_sha256", {})
    if not hashes:
        raise ValueError("Frozen Stage-3 design has no source hashes")
    repo_root = Path(__file__).resolve().parents[2]
    for relative_path, expected in hashes.items():
        actual = _sha256(repo_root / relative_path)
        if actual != expected:
            raise ValueError(f"Frozen Stage-3 source changed: {relative_path}; {actual} != {expected}")
    raw_rows = payload.get("runs")
    if not isinstance(raw_rows, list) or len(raw_rows) != EXPECTED_RUN_COUNT:
        raise ValueError(f"Expected {EXPECTED_RUN_COUNT} frozen Stage-3 runs")
    expected_manifest_hash = payload.get("design", {}).get("launch_manifest_sha256")
    actual_manifest_hash = stream_identity.canonical_sha256(frozen_designer.launch_manifest(raw_rows))
    if actual_manifest_hash != expected_manifest_hash:
        raise ValueError("Frozen Stage-3 launch-manifest hash is invalid")
    return payload


def load_design(
    selected_cells: frozenset[str] | None = None,
    selected_runs: frozenset[str] | None = None,
) -> tuple[dict[str, stage1.Cell], tuple[Stage3Run, ...]]:
    """Load and audit the immutable Stage-3 design."""
    payload = _load_design_payload()
    if payload.get("expected_run_count") != EXPECTED_RUN_COUNT:
        raise ValueError("Unexpected Stage-3 run count")
    if payload.get("expected_untied_run_count") != EXPECTED_UNTIED_RUN_COUNT:
        raise ValueError("Unexpected Stage-3 untied run count")
    if payload.get("cell_count") != EXPECTED_CELL_COUNT:
        raise ValueError("Unexpected Stage-3 cell count")
    cells = {
        str(row["cell_id"]): stage1.Cell(
            **{key: value for key, value in row.items() if key != "track_memberships"},
            track_memberships=tuple(row["track_memberships"]),
        )
        for row in payload["source_cells"]
    }
    requested_runs = tuple(
        Stage3Run(
            run_name=str(row["run_name"]),
            cell_id=str(row["cell_id"]),
            acquisition_kind=str(row["acquisition_kind"]),
            phase_0_starcoder=float(row["phase_0_starcoder"]),
            phase_1_starcoder=float(row["phase_1_starcoder"]),
            boundary_step=int(row["boundary_step"]),
            data_seed=int(row["data_seed"]),
            simulated_epoch_subset_seed=int(row["simulated_epoch_subset_seed"]),
        )
        for row in payload["runs"]
    )
    if len(cells) != EXPECTED_CELL_COUNT or len(requested_runs) != EXPECTED_RUN_COUNT:
        raise ValueError("Stage-3 manifest does not match its declared dimensions")
    if len({item.run_name for item in requested_runs}) != EXPECTED_RUN_COUNT:
        raise ValueError("Stage-3 run names are not unique")
    for cell_id in cells:
        cell_runs = tuple(item for item in requested_runs if item.cell_id == cell_id)
        untied_count = sum(abs(item.phase_0_starcoder - item.phase_1_starcoder) > 1e-12 for item in cell_runs)
        tied_count = len(cell_runs) - untied_count
        if untied_count != UNTIED_RUNS_PER_CELL or tied_count not in (0, 1, 2):
            raise ValueError(f"Cell {cell_id} has invalid untied/tied Stage-3 counts: {untied_count}/{tied_count}")
    for item in requested_runs:
        if item.cell_id not in cells:
            raise ValueError(f"Unknown cell for {item.run_name}: {item.cell_id}")
        if not 0.01 <= item.phase_0_starcoder <= 0.99 or not 0.01 <= item.phase_1_starcoder <= 0.99:
            raise ValueError(f"Invalid mixture coordinate for {item.run_name}")
        expected_boundary = int(cells[item.cell_id].total_steps * base.PHASE_BOUNDARY)
        if item.boundary_step != expected_boundary:
            raise ValueError(f"Frozen phase boundary drifted for {item.run_name}")
        if item.data_seed != frozen_designer.REFERENCE_SEED:
            raise ValueError(f"Stage-3 row does not use the reference data seed: {item.run_name}")
        if item.simulated_epoch_subset_seed != frozen_designer.REFERENCE_SEED:
            raise ValueError(f"Stage-3 row does not use the reference subset seed: {item.run_name}")

    if selected_cells is not None:
        unknown = selected_cells - set(cells)
        if unknown:
            raise ValueError(f"Unknown cells: {sorted(unknown)}")
        cells = {key: value for key, value in cells.items() if key in selected_cells}
        requested_runs = tuple(item for item in requested_runs if item.cell_id in selected_cells)
    if selected_runs is not None:
        available = {item.run_name for item in requested_runs}
        unknown = selected_runs - available
        if unknown:
            raise ValueError(f"Unknown runs: {sorted(unknown)}")
        requested_runs = tuple(item for item in requested_runs if item.run_name in selected_runs)
        selected_run_cells = {item.cell_id for item in requested_runs}
        cells = {key: value for key, value in cells.items() if key in selected_run_cells}
    return cells, requested_runs


def _validate_training_streams(
    steps: tuple[ArtifactStep[LevanterCheckpoint], ...],
    requested_runs: tuple[Stage3Run, ...],
) -> None:
    rows = {item.run_name: item for item in requested_runs}
    digests_by_cell: dict[str, set[str]] = {}
    observed_names = set()
    for step in steps:
        step_spec = lower(step)
        matches = [run_name for run_name in rows if step_spec.name.endswith(f"/{run_name}")]
        if len(matches) != 1:
            raise ValueError(f"Could not map lowered step {step_spec.name!r} to one frozen run")
        run_name = matches[0]
        observed_names.add(run_name)
        row = rows[run_name]
        train_config = stream_identity.lowered_step_training_config(step_spec)
        expected_policy = [
            {"boundary_step": 0, "starcoder_weight": row.phase_0_starcoder},
            {"boundary_step": row.boundary_step, "starcoder_weight": row.phase_1_starcoder},
        ]
        differences = stream_identity.identity_differences(
            stream_identity.policy_coordinates(train_config), expected_policy
        )
        if differences:
            raise ValueError(f"Lowered policy does not match frozen row {run_name}: {differences}")
        digest = stream_identity.canonical_sha256(stream_identity.lowered_step_stream_identity(step_spec))
        digests_by_cell.setdefault(row.cell_id, set()).add(digest)
    if observed_names != set(rows):
        raise ValueError("Lowered Stage-3 handles do not cover the selected rows exactly")
    inconsistent = {cell_id: digests for cell_id, digests in digests_by_cell.items() if len(digests) != 1}
    if inconsistent:
        raise ValueError(f"Stage-3 policies within a cell do not share one policy-free stream: {inconsistent}")


def build_training_steps(
    *,
    name_prefix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    selected_cells: frozenset[str] | None = None,
    selected_runs: frozenset[str] | None = None,
) -> tuple[ArtifactStep[LevanterCheckpoint], ...]:
    """Build resumable training handles for the selected Stage-3 rows."""
    cells, requested_runs = load_design(selected_cells, selected_runs)
    heuristic = CompletedAdamHHeuristic()
    for cell in cells.values():
        stage1._validate_runtime_model(cell, heuristic)

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
        phase_0_weights = base._phase_leaf_weights(request.phase_0_starcoder, nemotron=nemotron, starcoder=starcoder)
        phase_1_weights = base._phase_leaf_weights(request.phase_1_starcoder, nemotron=nemotron, starcoder=starcoder)
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
                request.cell_id,
                request.acquisition_kind,
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
                    (request.boundary_step, phase_1_weights),
                ],
                data_seed=request.data_seed,
                simulated_epoch_subset_seed=request.simulated_epoch_subset_seed,
                experiment_budget=cell.materialized_tokens,
                target_budget=base.TARGET_BUDGET,
            )
        )
    result = tuple(steps)
    if len(result) != len(requested_runs):
        raise ValueError(f"Expected {len(requested_runs)} Stage-3 handles, got {len(result)}")
    _validate_training_streams(result, requested_runs)
    return result


def _parse_values(value: str | None) -> frozenset[str] | None:
    if value is None:
        return None
    values = frozenset(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("A selection must contain at least one value")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--marin-prefix", default=base.DEFAULT_MARIN_PREFIX)
    parser.add_argument("--tpu-type", default=base.DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--cells", help="Comma-separated cell IDs for an idempotent partial retry")
    parser.add_argument("--runs", help="Comma-separated run names for an idempotent partial retry")
    parser.add_argument("--audit-manifest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if os.getenv("CI") is not None:
        logger.info("Skipping StarCoder WSD80 matched-N,D Stage 3 in CI")
        return
    if args.marin_prefix != base.DEFAULT_MARIN_PREFIX:
        raise ValueError(f"This StarCoder experiment is central1-local: got {args.marin_prefix!r}")
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError(
            "StarCoder child TPU placement must remain central1-local: "
            f"got region={args.tpu_region!r}, zone={args.tpu_zone!r}"
        )
    if args.tpu_type != base.DEFAULT_TPU_TYPE:
        raise ValueError(f"StarCoder Stage 3 must use historical accelerator {base.DEFAULT_TPU_TYPE}: {args.tpu_type!r}")
    if args.cells is not None and args.runs is not None:
        raise ValueError("Specify at most one of --cells and --runs")
    selected_cells = _parse_values(args.cells)
    selected_runs = _parse_values(args.runs)
    cells, requested_runs = load_design(selected_cells, selected_runs)
    if not 1 <= args.max_concurrent <= len(requested_runs):
        raise ValueError(f"max_concurrent must be in [1, {len(requested_runs)}]")
    logger.info(
        "Prepared %d Stage-3 runs over %d N-D cells: %s",
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
        logger.info("Dry-run lowering passed for all %d Stage-3 handles", len(steps))
        return
    run(*steps, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
