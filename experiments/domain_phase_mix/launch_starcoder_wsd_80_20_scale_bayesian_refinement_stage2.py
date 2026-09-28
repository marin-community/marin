# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the frozen second-stage Bayesian refinement of WSD80 scale optima."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from marin.execution.lazy import ArtifactStep, lower, run
from marin.training.training import LevanterCheckpoint

from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_fixed_model_token_scaling as scaling
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_scale_bayesian_refinement as stage1
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_starcoder_wsd80_scale_bayesian_refinement_stage2_20260801 as frozen_designer,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    starcoder_wsd80_training_identity as stream_identity,
)

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_wsd80_scale_bo_stage2_20260801"
WANDB_EXPERIMENT_TAG = "starcoder_wsd80_scale_bo_stage2"
PANEL_TAG = "scale_bo_stage2_20260801"
DESIGN_PATH = Path(__file__).with_name("starcoder_wsd80_scale_bayesian_refinement_stage2_design_20260801.json")
EXPECTED_RUN_COUNT = 26
DEFAULT_MAX_CONCURRENT = 26


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_design_payload() -> dict[str, Any]:
    payload = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    if payload.get("design_version") != "2026-08-01":
        raise ValueError(f"Unexpected Stage-2 design version in {DESIGN_PATH}")
    if payload.get("objective_metric") != scaling.OBJECTIVE_METRIC:
        raise ValueError("Frozen Stage-2 design targets an unexpected objective")
    expected_environment = {
        "tpu_type": base.DEFAULT_TPU_TYPE,
        "tpu_region": base.DEFAULT_TPU_REGION,
        "tpu_zone": base.DEFAULT_TPU_ZONE,
        "marin_prefix": base.DEFAULT_MARIN_PREFIX,
    }
    environment = payload.get("training_environment", {})
    if any(environment.get(key) != value for key, value in expected_environment.items()):
        raise ValueError("Frozen Stage-2 training environment does not match the historical WSD80 environment")
    hashes = payload.get("data_use", {}).get("source_sha256", {})
    if not hashes:
        raise ValueError("Frozen Stage-2 design has no source hashes")
    repo_root = Path(__file__).resolve().parents[2]
    for relative_path, expected in hashes.items():
        actual = _sha256(repo_root / relative_path)
        if actual != expected:
            raise ValueError(f"Frozen Stage-2 source changed: {relative_path}")
    raw_rows = payload.get("runs")
    if not isinstance(raw_rows, list) or len(raw_rows) != EXPECTED_RUN_COUNT:
        raise ValueError(f"Expected {EXPECTED_RUN_COUNT} frozen Stage-2 runs")
    regenerated_rows, _, _, _ = frozen_designer.build_rows()
    frozen_manifest = frozen_designer.launch_manifest(raw_rows)
    regenerated_manifest = frozen_designer.launch_manifest(regenerated_rows)
    if frozen_manifest != regenerated_manifest:
        raise ValueError("Current design code does not reproduce the frozen Stage-2 launch manifest")
    expected_manifest_hash = payload.get("design", {}).get("launch_manifest_sha256")
    if stream_identity.canonical_sha256(frozen_manifest) != expected_manifest_hash:
        raise ValueError("Frozen Stage-2 launch-manifest hash is invalid")
    return payload


def load_design() -> tuple[stage1.RefinementRun, ...]:
    """Load and validate the immutable source-controlled Stage-2 design."""
    raw_rows = _load_design_payload()["runs"]
    runs = tuple(
        stage1.RefinementRun(
            token_budget=int(row["token_budget_requested"]),
            phase_0_starcoder=float(row["phase_0_starcoder"]),
            phase_1_starcoder=float(row["phase_1_starcoder"]),
            run_name=str(row["run_name"]),
            trainer_data_seed=int(row["trainer_data_seed"]),
            simulated_epoch_subset_seed=int(row["simulated_epoch_subset_seed"]),
        )
        for row in raw_rows
    )
    if len({item.run_name for item in runs}) != len(runs):
        raise ValueError("Frozen Stage-2 run names are not unique")
    if {item.token_budget for item in runs} != {2_000_000_000, 4_000_000_000, 8_000_000_000}:
        raise ValueError("Frozen Stage-2 design must cover the 2B, 4B, and 8B rungs")
    for item in runs:
        if not 0.0 <= item.phase_0_starcoder <= 1.0 or not 0.0 <= item.phase_1_starcoder <= 1.0:
            raise ValueError(f"Invalid frozen coordinate: {item}")
    return runs


def _validate_training_streams(
    steps: tuple[ArtifactStep[LevanterCheckpoint], ...],
    payload: dict[str, Any],
) -> None:
    rows = {str(row["run_name"]): row for row in payload["runs"]}
    lowered_by_name = {}
    for step in steps:
        step_spec = lower(step)
        matches = [run_name for run_name in rows if step_spec.name.endswith(f"/{run_name}")]
        if len(matches) != 1:
            raise ValueError(f"Could not map lowered step {step_spec.name!r} to one frozen run")
        lowered_by_name[matches[0]] = step_spec

    historical_audit = payload.get("confirmation", {}).get("historical_pairing_audit", {})
    audit_by_seed = {int(row["seed"]): row for row in historical_audit.get("rows", [])}
    expected_existing_seeds = set(frozen_designer.EXISTING_CONFIRMATION_SEEDS)
    if set(audit_by_seed) != expected_existing_seeds:
        raise ValueError("Frozen historical-pairing audit does not cover every reused incumbent")
    for seed, audit in audit_by_seed.items():
        digests = {
            str(audit["historical_stream_identity_sha256"]),
            str(audit["current_incumbent_stream_identity_sha256"]),
            str(audit["current_candidate_stream_identity_sha256"]),
        }
        if audit.get("identity_match") is not True or len(digests) != 1:
            raise ValueError(f"Frozen historical-pairing audit failed for seed {seed}")

    confirmation_digests: dict[int, list[str]] = {}
    for run_name, row in rows.items():
        step_spec = lowered_by_name[run_name]
        train_config = stream_identity.lowered_step_training_config(step_spec)
        policy = stream_identity.policy_coordinates(train_config)
        expected_policy = [
            {"boundary_step": 0, "starcoder_weight": row["phase_0_starcoder"]},
            {"boundary_step": int(row["boundary_step"]), "starcoder_weight": row["phase_1_starcoder"]},
        ]
        differences = stream_identity.identity_differences(policy, expected_policy)
        if differences:
            raise ValueError(f"Lowered policy does not match frozen row {run_name}: {differences}")
        if row["run_kind"] == "acquisition":
            continue
        digest = stream_identity.canonical_sha256(stream_identity.lowered_step_stream_identity(step_spec))
        seed = int(row["pair_seed"])
        confirmation_digests.setdefault(seed, []).append(digest)
        if seed in audit_by_seed and digest != audit_by_seed[seed]["historical_stream_identity_sha256"]:
            raise ValueError(f"Current candidate no longer matches the historical incumbent stream for seed {seed}")

    for seed in frozen_designer.EXISTING_CONFIRMATION_SEEDS + frozen_designer.NEW_CONFIRMATION_SEEDS:
        digests = confirmation_digests.get(seed, [])
        if len(digests) != 2 or len(set(digests)) != 1:
            raise ValueError(f"Candidate/incumbent pair does not share one training stream for seed {seed}")
    logger.info("Validated all frozen policies and policy-free stream identity for all eight confirmation pairs")


def build_training_steps(
    *, name_prefix: str, tpu_type: str, tpu_region: str, tpu_zone: str
) -> tuple[ArtifactStep[LevanterCheckpoint], ...]:
    """Build Stage-2 training handles grouped by token budget."""
    requested_runs = load_design()
    steps: list[ArtifactStep[LevanterCheckpoint]] = []
    rank = 1
    for token_budget in scaling.TOKEN_BUDGETS:
        budget_runs = [item for item in requested_runs if item.token_budget == token_budget]
        specs = tuple(item.surface_spec(rank + index) for index, item in enumerate(budget_runs))
        rank += len(specs)
        steps.extend(
            base.build_training_steps(
                name_prefix=name_prefix,
                tpu_type=tpu_type,
                tpu_region=tpu_region,
                tpu_zone=tpu_zone,
                data_seed=scaling.REFERENCE_SEED,
                run_specs=specs,
                wandb_experiment_tag=WANDB_EXPERIMENT_TAG,
                panel_tag=f"{PANEL_TAG}_{token_budget // 1_000_000_000}b",
                experiment_budget=token_budget,
                target_budget=base.TARGET_BUDGET,
            )
        )
    if len(steps) != len(requested_runs):
        raise ValueError(f"Expected {len(requested_runs)} Stage-2 handles, got {len(steps)}")
    result = tuple(steps)
    _validate_training_streams(result, _load_design_payload())
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--marin-prefix", default=base.DEFAULT_MARIN_PREFIX)
    parser.add_argument("--tpu-type", default=base.DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if os.getenv("CI") is not None:
        logger.info("Skipping StarCoder WSD80 Bayesian-refinement Stage 2 in CI")
        return
    if args.marin_prefix != base.DEFAULT_MARIN_PREFIX:
        raise ValueError(f"This historical StarCoder experiment is central1-local: got {args.marin_prefix!r}")
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError(
            "StarCoder child TPU placement must remain central1-local: "
            f"got region={args.tpu_region!r}, zone={args.tpu_zone!r}"
        )
    if args.tpu_type != base.DEFAULT_TPU_TYPE:
        raise ValueError(f"StarCoder Stage 2 must use historical accelerator {base.DEFAULT_TPU_TYPE}: {args.tpu_type!r}")
    requested_runs = load_design()
    if not 1 <= args.max_concurrent <= len(requested_runs):
        raise ValueError(f"max_concurrent must be in [1, {len(requested_runs)}], got {args.max_concurrent}")
    os.environ["MARIN_PREFIX"] = args.marin_prefix
    steps = build_training_steps(
        name_prefix=args.name_prefix,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    logger.info("Prepared %d Stage-2 handles", len(steps))
    if args.dry_run:
        for step in steps:
            lower(step)
        logger.info("Dry-run lowering passed")
        return
    run(*steps, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
