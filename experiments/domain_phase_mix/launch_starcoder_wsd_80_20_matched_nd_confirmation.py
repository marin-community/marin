# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch frozen fresh-seed confirmations for promoted matched-N,D policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
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
    design_starcoder_wsd80_matched_nd_confirmation_20260801 as frozen_designer,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    starcoder_wsd80_training_identity as stream_identity,
)
from experiments.llama import llama3_tokenizer
from experiments.scaling_law_sweeps.completed_adamh import CompletedAdamHHeuristic

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_wsd80_matched_nd_confirmation_20260801"
WANDB_EXPERIMENT_TAG = "starcoder_wsd80_matched_nd_confirmation"
PANEL_TAG = "matched_nd_confirmation_20260801"
DESIGN_VERSION = "2026-08-01"
DESIGN_PATH = Path(__file__).with_name("starcoder_wsd80_matched_nd_confirmation_design_20260801.json")
DEFAULT_MAX_CONCURRENT = 64


@dataclass(frozen=True)
class ConfirmationRun:
    """One arm of a fresh-seed candidate/comparator pair."""

    run_name: str
    cell_id: str
    role: str
    pair_seed: int
    phase_0_starcoder: float
    phase_1_starcoder: float
    total_steps: int
    boundary_step: int
    data_seed: int
    simulated_epoch_subset_seed: int
    pair_stream_identity_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_design_payload() -> dict[str, Any]:
    payload = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    if payload.get("design_version") != DESIGN_VERSION:
        raise ValueError(f"Unexpected confirmation design version in {DESIGN_PATH}")
    environment = payload.get("training_environment", {})
    expected_environment = {
        "tpu_type": base.DEFAULT_TPU_TYPE,
        "tpu_region": base.DEFAULT_TPU_REGION,
        "tpu_zone": base.DEFAULT_TPU_ZONE,
        "marin_prefix": base.DEFAULT_MARIN_PREFIX,
    }
    if any(environment.get(key) != value for key, value in expected_environment.items()):
        raise ValueError("Frozen confirmation environment does not match historical WSD80 training")
    hashes = payload.get("data_use", {}).get("source_sha256", {})
    if not hashes:
        raise ValueError("Frozen confirmation has no source hashes")
    repo_root = Path(__file__).resolve().parents[2]
    for relative_path, expected in hashes.items():
        actual = _sha256(repo_root / relative_path)
        if actual != expected:
            raise ValueError(f"Frozen confirmation source changed: {relative_path}; {actual} != {expected}")
    rows = payload.get("runs")
    expected_count = int(payload.get("expected_run_count", 0))
    if not isinstance(rows, list) or not rows or len(rows) != expected_count:
        raise ValueError("Frozen confirmation contains an invalid run count")
    expected_hash = payload.get("design", {}).get("launch_manifest_sha256")
    if stream_identity.canonical_sha256(frozen_designer.launch_manifest(rows)) != expected_hash:
        raise ValueError("Frozen confirmation launch-manifest hash is invalid")
    regenerated_rows = frozen_designer.regenerate_rows()
    if regenerated_rows != rows:
        raise ValueError("Frozen confirmation rows do not regenerate from the pinned discovery inputs")
    return payload


def load_design(
    selected_runs: frozenset[str] | None = None,
) -> tuple[dict[str, stage1.Cell], tuple[ConfirmationRun, ...]]:
    """Load and audit the immutable paired-confirmation design."""
    payload = _load_design_payload()
    cells = {
        str(row["cell_id"]): stage1.Cell(
            **{key: value for key, value in row.items() if key != "track_memberships"},
            track_memberships=tuple(row["track_memberships"]),
        )
        for row in payload["source_cells"]
    }
    requests = tuple(
        ConfirmationRun(
            run_name=str(row["run_name"]),
            cell_id=str(row["cell_id"]),
            role=str(row["role"]),
            pair_seed=int(row["pair_seed"]),
            phase_0_starcoder=float(row["phase_0_starcoder"]),
            phase_1_starcoder=float(row["phase_1_starcoder"]),
            total_steps=int(row["total_steps"]),
            boundary_step=int(row["boundary_step"]),
            data_seed=int(row["data_seed"]),
            simulated_epoch_subset_seed=int(row["simulated_epoch_subset_seed"]),
            pair_stream_identity_sha256=str(row["pair_stream_identity_sha256"]),
        )
        for row in payload["runs"]
    )
    if len({request.run_name for request in requests}) != len(requests):
        raise ValueError("Confirmation run names are not unique")
    expected_roles = set(frozen_designer.ROLES)
    for cell_id, cell in cells.items():
        cell_requests = tuple(request for request in requests if request.cell_id == cell_id)
        seeds = {request.pair_seed for request in cell_requests}
        if len(seeds) != 8 or len(cell_requests) != 16:
            raise ValueError(f"{cell_id}: expected eight candidate/comparator pairs")
        for seed in seeds:
            pair = tuple(request for request in cell_requests if request.pair_seed == seed)
            if len(pair) != 2 or {request.role for request in pair} != expected_roles:
                raise ValueError(f"{cell_id}, seed {seed}: incomplete candidate/comparator pair")
        for request in cell_requests:
            if request.data_seed != request.pair_seed or request.simulated_epoch_subset_seed != request.pair_seed:
                raise ValueError(f"{request.run_name}: pair seed does not control both training seeds")
            if request.total_steps != cell.total_steps:
                raise ValueError(f"{request.run_name}: total training steps drifted")
            if request.boundary_step != int(cell.total_steps * base.PHASE_BOUNDARY):
                raise ValueError(f"{request.run_name}: phase boundary drifted")
            expected_pair_identity = frozen_designer.pair_stream_identity(asdict(cell), request.pair_seed)
            if request.pair_stream_identity_sha256 != expected_pair_identity:
                raise ValueError(f"{request.run_name}: frozen policy-free pair identity is invalid")
            tags = (WANDB_EXPERIMENT_TAG, request.cell_id, request.role, "starcoder", "wsd80_20", PANEL_TAG)
            if any(len(tag) > 64 for tag in tags):
                raise ValueError(f"{request.run_name}: W&B tag exceeds 64 characters: {tags}")
    if {request.cell_id for request in requests} != set(cells):
        raise ValueError("Confirmation rows do not cover the declared promoted cells")

    if selected_runs is not None:
        available = {request.run_name for request in requests}
        unknown = selected_runs - available
        if unknown:
            raise ValueError(f"Unknown confirmation runs: {sorted(unknown)}")
        requests = tuple(request for request in requests if request.run_name in selected_runs)
        selected_cells = {request.cell_id for request in requests}
        cells = {cell_id: cell for cell_id, cell in cells.items() if cell_id in selected_cells}
    return cells, requests


def _validate_training_streams(
    steps: tuple[ArtifactStep[LevanterCheckpoint], ...],
    requests: tuple[ConfirmationRun, ...],
) -> None:
    rows = {request.run_name: request for request in requests}
    digests_by_pair: dict[tuple[str, int], set[str]] = {}
    observed_names = set()
    for step in steps:
        step_spec = lower(step)
        matches = [run_name for run_name in rows if step_spec.name.endswith(f"/{run_name}")]
        if len(matches) != 1:
            raise ValueError(f"Could not map lowered step {step_spec.name!r} to one confirmation run")
        request = rows[matches[0]]
        observed_names.add(request.run_name)
        config = stream_identity.lowered_step_training_config(step_spec)
        expected_policy = [
            {"boundary_step": 0, "starcoder_weight": request.phase_0_starcoder},
            {"boundary_step": request.boundary_step, "starcoder_weight": request.phase_1_starcoder},
        ]
        differences = stream_identity.identity_differences(stream_identity.policy_coordinates(config), expected_policy)
        if differences:
            raise ValueError(f"{request.run_name}: lowered policy differs from the frozen design: {differences}")
        digest = stream_identity.canonical_sha256(stream_identity.lowered_step_stream_identity(step_spec))
        digests_by_pair.setdefault((request.cell_id, request.pair_seed), set()).add(digest)
    if observed_names != set(rows):
        raise ValueError("Lowered handles do not cover the selected confirmation rows exactly")
    inconsistent = {pair: digests for pair, digests in digests_by_pair.items() if len(digests) != 1}
    if inconsistent:
        raise ValueError(f"Candidate/comparator arms do not share one policy-free stream: {inconsistent}")


def build_training_steps(
    *,
    name_prefix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    selected_runs: frozenset[str] | None = None,
) -> tuple[ArtifactStep[LevanterCheckpoint], ...]:
    """Build idempotent training handles for selected confirmation arms."""
    cells, requests = load_design(selected_runs)
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
    for request in requests:
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
                request.role,
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
                phase_weights=[(0, phase_0_weights), (request.boundary_step, phase_1_weights)],
                data_seed=request.data_seed,
                simulated_epoch_subset_seed=request.simulated_epoch_subset_seed,
                experiment_budget=cell.materialized_tokens,
                target_budget=base.TARGET_BUDGET,
            )
        )
    result = tuple(steps)
    if len(result) != len(requests):
        raise ValueError(f"Expected {len(requests)} confirmation handles, got {len(result)}")
    _validate_training_streams(result, requests)
    return result


def _parse_runs(value: str | None) -> frozenset[str] | None:
    if value is None:
        return None
    runs = frozenset(item.strip() for item in value.split(",") if item.strip())
    if not runs:
        raise argparse.ArgumentTypeError("--runs must contain at least one exact run name")
    return runs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--marin-prefix", default=base.DEFAULT_MARIN_PREFIX)
    parser.add_argument("--tpu-type", default=base.DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--runs", help="Comma-separated exact run names for an idempotent partial retry")
    parser.add_argument("--audit-manifest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if os.getenv("CI") is not None:
        logger.info("Skipping StarCoder WSD80 matched-N,D confirmation in CI")
        return
    if args.marin_prefix != base.DEFAULT_MARIN_PREFIX:
        raise ValueError(f"This StarCoder experiment is central1-local: got {args.marin_prefix!r}")
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError(
            "StarCoder child TPU placement must remain central1-local: "
            f"got region={args.tpu_region!r}, zone={args.tpu_zone!r}"
        )
    if args.tpu_type != base.DEFAULT_TPU_TYPE:
        raise ValueError(f"Confirmation must use historical accelerator {base.DEFAULT_TPU_TYPE}: {args.tpu_type!r}")
    if args.name_prefix != NAME:
        raise ValueError(f"Confirmation checkpoint identity is frozen: {args.name_prefix!r} != {NAME!r}")
    selected_runs = _parse_runs(args.runs)
    cells, requests = load_design(selected_runs)
    max_concurrent = min(DEFAULT_MAX_CONCURRENT, len(requests)) if args.max_concurrent is None else args.max_concurrent
    if not 1 <= max_concurrent <= len(requests):
        raise ValueError(f"max_concurrent must be in [1, {len(requests)}]")
    logger.info("Prepared %d paired-confirmation arms over %d cells", len(requests), len(cells))
    if args.audit_manifest:
        return

    os.environ["MARIN_PREFIX"] = args.marin_prefix
    steps = build_training_steps(
        name_prefix=args.name_prefix,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        selected_runs=selected_runs,
    )
    if args.dry_run:
        for step in steps:
            lower(step)
        logger.info("Dry-run lowering passed for all %d confirmation handles", len(steps))
        return
    run(*steps, max_concurrent=max_concurrent)


if __name__ == "__main__":
    main()
