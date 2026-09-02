# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the frozen StarCoder WSD80 physical-full-pool intervention."""

from __future__ import annotations

import argparse
import json
import logging
import os
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
from experiments.domain_phase_mix import launch_starcoder_wsd80_batch_repetition_intervention as shared
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_starcoder_wsd80_full_pool_intervention_20260804 as frozen_design,
)
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/starcoder_wsd80_full_pool_intervention_20260804"
WANDB_EXPERIMENT_TAG = "starcoder_wsd80_full_pool_20260804"
PANEL_TAG = "wsd80_full_pool"
DESIGN_VERSION = frozen_design.DESIGN_VERSION
DESIGN_PATH = Path(__file__).with_name("starcoder_wsd80_full_pool_design_20260804.json")
EXPECTED_RUN_COUNT = frozen_design.EXPECTED_RUN_COUNT
DEFAULT_MAX_CONCURRENT = EXPECTED_RUN_COUNT


def _load_payload() -> dict[str, Any]:
    payload = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    if payload.get("design_version") != DESIGN_VERSION:
        raise ValueError(f"Unexpected design version in {DESIGN_PATH}")
    claimed_hash = payload.pop("design_sha256", None)
    observed_hash = frozen_design.parent_design.canonical_sha256(payload)
    if claimed_hash != observed_hash:
        raise ValueError(f"Design self-hash mismatch: {observed_hash} != {claimed_hash}")
    payload["design_sha256"] = claimed_hash
    if payload != frozen_design.build_payload():
        raise ValueError("Frozen full-pool design does not regenerate exactly")
    if payload.get("expected_run_count") != EXPECTED_RUN_COUNT:
        raise ValueError("Unexpected full-pool run count")
    expected_environment = {
        "tpu_type": base.DEFAULT_TPU_TYPE,
        "tpu_region": base.DEFAULT_TPU_REGION,
        "tpu_zone": base.DEFAULT_TPU_ZONE,
        "marin_prefix": base.DEFAULT_MARIN_PREFIX,
    }
    if payload.get("training_environment") != expected_environment:
        raise ValueError("Frozen training environment drifted from historical central1 WSD80")
    return payload


def load_design(selected_runs: frozenset[str] | None = None) -> tuple[shared.InterventionRun, ...]:
    """Load and audit the immutable policy-by-seed manifest."""
    payload = _load_payload()
    requests = tuple(
        shared.InterventionRun(**{key: row[key] for key in shared.InterventionRun.__dataclass_fields__})
        for row in payload["runs"]
    )
    if len(requests) != EXPECTED_RUN_COUNT or len({request.run_name for request in requests}) != EXPECTED_RUN_COUNT:
        raise ValueError("Full-pool rows are missing or duplicated")

    expected_policies = {policy.policy_id for policy in frozen_design.parent_design.POLICIES}
    expected_seeds = set(frozen_design.parent_design.PAIR_SEEDS)
    if {request.policy_id for request in requests} != expected_policies:
        raise ValueError("Full-pool policy block is incomplete")
    for seed in expected_seeds:
        seed_rows = tuple(request for request in requests if request.pair_seed == seed)
        if len(seed_rows) != len(expected_policies) or {request.policy_id for request in seed_rows} != expected_policies:
            raise ValueError(f"Seed {seed}: incomplete paired triplet")

    rows_by_name = {str(row["run_name"]): row for row in payload["runs"]}
    for request in requests:
        row = rows_by_name[request.run_name]
        if request.condition_id != frozen_design.CONDITION_ID:
            raise ValueError(f"{request.run_name}: unexpected condition")
        if request.data_seed != request.pair_seed or request.simulated_epoch_subset_seed != request.pair_seed:
            raise ValueError(f"{request.run_name}: paired seed does not control training and subset selection")
        if request.total_steps * request.batch_size * base.SEQ_LEN != request.materialized_tokens:
            raise ValueError(f"{request.run_name}: token accounting drifted")
        if request.target_budget != request.materialized_tokens:
            raise ValueError(f"{request.run_name}: full-pool target must equal the experiment budget")
        if request.boundary_step * 5 != request.total_steps * 4:
            raise ValueError(f"{request.run_name}: phase boundary is not exactly 80/20")
        if (request.boundary_step * request.batch_size) % base.MIXTURE_BLOCK_SIZE != 0:
            raise ValueError(f"{request.run_name}: phase boundary is not mixture-block aligned")
        if request.eval_interval_steps * 10 != request.total_steps:
            raise ValueError(f"{request.run_name}: evaluation cadence drifted")
        if float(row["starcoder_physical_pool_fraction"]) >= 1.0:
            raise ValueError(f"{request.run_name}: StarCoder source would wrap")
        if float(row["nemotron_physical_pool_fraction"]) >= 1.0:
            raise ValueError(f"{request.run_name}: Nemotron source would wrap")
        tags = (WANDB_EXPERIMENT_TAG, PANEL_TAG, request.policy_id, "starcoder", "wsd80_20")
        if any(len(tag) > 64 for tag in tags):
            raise ValueError(f"{request.run_name}: W&B tag exceeds 64 characters")

    if selected_runs is not None:
        available = {request.run_name for request in requests}
        unknown = selected_runs - available
        if unknown:
            raise ValueError(f"Unknown full-pool runs: {sorted(unknown)}")
        requests = tuple(request for request in requests if request.run_name in selected_runs)
    return requests


def build_training_steps(
    *,
    name_prefix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    selected_runs: frozenset[str] | None = None,
) -> tuple[ArtifactStep[LevanterCheckpoint], ...]:
    """Build resumable training handles for all selected full-pool rows."""
    requests = load_design(selected_runs)
    model = shared._validate_model()
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

    steps: list[ArtifactStep[LevanterCheckpoint]] = []
    for request in requests:
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
            model=model,
            optimizer=shared._optimizer(request),
            datasets=static_weights,
            validation=validation_handles,
            batch_size=request.batch_size,
            seq_len=base.SEQ_LEN,
            num_train_steps=request.total_steps,
            z_loss_weight=None,
            evals=None,
            resources=resources,
            steps_per_eval=request.eval_interval_steps,
            wandb_project="marin",
            wandb_group=name_prefix,
            run_id=request.run_name,
            tags=(WANDB_EXPERIMENT_TAG, PANEL_TAG, request.policy_id, "starcoder", "wsd80_20"),
            env_vars={"HF_ALLOW_CODE_EVAL": "1"},
        )
        steps.append(
            shared._with_intervention_data(
                training,
                train_datasets=static_weights,
                validation_datasets=validation_handles,
                phase_weights=[(0, phase_0_weights), (request.boundary_step, phase_1_weights)],
                request=request,
            )
        )
    if len(steps) != len(requests):
        raise ValueError(f"Expected {len(requests)} training handles, got {len(steps)}")
    return tuple(steps)


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
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--runs", help="Comma-separated exact run names for an idempotent partial retry")
    parser.add_argument("--audit-manifest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if os.getenv("CI") is not None:
        logger.info("Skipping StarCoder full-pool intervention in CI")
        return
    if args.name_prefix != NAME:
        raise ValueError(f"Full-pool checkpoint identity is frozen: {args.name_prefix!r} != {NAME!r}")
    if args.marin_prefix != base.DEFAULT_MARIN_PREFIX:
        raise ValueError(f"Historical StarCoder work must remain central1-local: {args.marin_prefix!r}")
    if args.tpu_type != base.DEFAULT_TPU_TYPE:
        raise ValueError(f"Historical StarCoder accelerator is frozen: {args.tpu_type!r}")
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError(
            "StarCoder child placement must remain central1-local: "
            f"region={args.tpu_region!r}, zone={args.tpu_zone!r}"
        )

    selected_runs = _parse_runs(args.runs)
    requests = load_design(selected_runs)
    if not 1 <= args.max_concurrent <= EXPECTED_RUN_COUNT:
        raise ValueError(f"max_concurrent must be in [1, {EXPECTED_RUN_COUNT}]")
    max_concurrent = min(args.max_concurrent, len(requests))
    logger.info("Prepared %d physical-full-pool intervention runs", len(requests))
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
        logger.info("Dry-run lowering passed for all %d training handles", len(steps))
        return
    run(*steps, max_concurrent=max_concurrent)


if __name__ == "__main__":
    main()
