# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate a runtime-exact Delphi prefix KL path on common seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import fsspec
import pandas as pd
from fray.cluster import ResourceConfig
from levanter.data.text.datasets import DatasetComponent
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from rigging.filesystem import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_3e18_phase0_prefix_replay as replay
from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_3e18_phase0_prefix_candidates_20260824"
DEFAULT_CANDIDATE_DIR = base.REFERENCE_OUTPUT_DIR / "delphi_phase0_prefix_candidates_20260824"
DEFAULT_CANDIDATE_WEIGHTS = DEFAULT_CANDIDATE_DIR / "candidate_weights.csv"
LOCAL_ARTIFACT_DIR = DEFAULT_CANDIDATE_DIR / "launch_dry_run"
CANDIDATE_IDS = (
    "shared_shape_kl0p05",
    "shared_shape_kl0p1",
    "shared_shape_kl0p2",
    "shared_shape_kl0p5",
    "observed_cap10_best",
    "proportional_control",
)
REPEAT_SEEDS = (0, 1, 2)
RUN_ID_BASE = 930_000
DATA_SEED_BASE = 930_000
MAX_CONCURRENT = len(CANDIDATE_IDS) * len(REPEAT_SEEDS)


@dataclass(frozen=True)
class SaveCandidateManifestConfig:
    output_path: str
    candidate_weights_path: str
    candidate_weights_sha256: str
    run_specs_json: str


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_candidate_weights(path: Path, expected_sha256: str) -> tuple[tuple[str, ...], dict[str, dict[str, float]]]:
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Candidate weights changed: {actual_sha256} != {expected_sha256}")
    frame = pd.read_csv(path)
    required = {"candidate_id", "bucket", "phase_0_weight", "phase_0_count", "phase_0_materialized_epochs"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Candidate weights are missing columns: {sorted(required - set(frame.columns))}")
    if tuple(frame.candidate_id.drop_duplicates()) != CANDIDATE_IDS:
        raise ValueError("Candidate order or identities changed")
    buckets = tuple(frame.loc[frame.candidate_id.eq(CANDIDATE_IDS[0]), "bucket"])
    candidates = {}
    for candidate_id in CANDIDATE_IDS:
        rows = frame[frame.candidate_id.eq(candidate_id)]
        if tuple(rows.bucket) != buckets:
            raise ValueError(f"Bucket order changed for {candidate_id}")
        counts = rows.phase_0_count.to_numpy(dtype=int)
        weights = rows.phase_0_weight.to_numpy(dtype=float)
        if counts.sum() != replay.base.MIXTURE_BLOCK_SIZE or not (counts >= 0).all():
            raise ValueError(f"Invalid runtime counts for {candidate_id}")
        if not (weights == counts / replay.base.MIXTURE_BLOCK_SIZE).all():
            raise ValueError(f"Weights are not exact runtime counts for {candidate_id}")
        if float(rows.phase_0_materialized_epochs.max()) > 10.0 + 1e-12:
            raise ValueError(f"Prefix epoch cap violated by {candidate_id}")
        candidates[candidate_id] = dict(zip(buckets, weights, strict=True))
    return buckets, candidates


def candidate_specs(
    *,
    candidate_weights_path: Path,
    expected_sha256: str,
    analysis_output_path: str,
    tpu_region: str,
    tpu_zone: str,
) -> list[base.DelphiSwarmRunSpec]:
    buckets, candidates = load_candidate_weights(candidate_weights_path, expected_sha256)
    source_specs, _audit = replay.load_replay_specs(
        source_panel=base.DEFAULT_SOURCE_PANEL,
        analysis_output_path=analysis_output_path,
        tpu_region=tpu_region,
        tpu_zone=tpu_zone,
    )
    template = source_specs[0]
    if tuple(template.phase_weights["phase_0"]) != buckets:
        raise ValueError("Candidate buckets do not match the Delphi runtime bucket order")

    specs = []
    for candidate_position, candidate_id in enumerate(CANDIDATE_IDS):
        weights = candidates[candidate_id]
        phase_weights = {"phase_0": weights, "phase_1": weights}
        max_epoch, q95_epoch, phase_tv = base._weight_diagnostics(phase_weights)
        for repeat in REPEAT_SEEDS:
            run_order = candidate_position * len(REPEAT_SEEDS) + repeat
            run_name = f"prefix_{candidate_id}_seed{repeat}"
            specs.append(
                replace(
                    template,
                    run_order=run_order,
                    run_id=RUN_ID_BASE + run_order,
                    run_name=run_name,
                    source_run_name=run_name,
                    source_experiment=EXPERIMENT_NAME,
                    panel_source="sequential_prefix_candidate",
                    data_seed=DATA_SEED_BASE + repeat,
                    trainer_seed=repeat,
                    max_simulated_epoch=max_epoch,
                    q95_simulated_epoch=q95_epoch,
                    mean_phase_tv_to_proportional=phase_tv,
                    phase_weights=phase_weights,
                )
            )
    return specs


def save_candidate_manifest(config: SaveCandidateManifestConfig) -> None:
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "candidate_weights_path": config.candidate_weights_path,
        "candidate_weights_sha256": config.candidate_weights_sha256,
        "candidate_ids": list(CANDIDATE_IDS),
        "repeat_seeds": list(REPEAT_SEEDS),
        "selection_target": "exact-boundary Uncheatable BPB",
        "secondary_gate": "exact-boundary GitHub C++ BPB",
        "prefix_train_steps": replay.EXPECTED_PREFIX_TRAIN_STEPS,
        "optimizer_schedule_num_train_steps": replay.EXPECTED_FULL_TRAIN_STEPS,
        "mixture_block_size": base.MIXTURE_BLOCK_SIZE,
        "run_specs": json.loads(config.run_specs_json),
    }
    with fs.open(os.path.join(config.output_path, "manifest.json"), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def build_steps(
    *,
    run_specs: list[base.DelphiSwarmRunSpec],
    analysis_output_path: str,
    validation_configs: dict[str, DatasetComponent],
    candidate_weights_path: Path,
    candidate_weights_sha256: str,
) -> list[ExecutorStep]:
    steps = []
    for run_spec in run_specs:
        resources = ResourceConfig.with_tpu(
            run_spec.tpu_type,
            regions=[run_spec.tpu_region],
            zone=run_spec.tpu_zone,
        )
        steps.append(
            ExecutorStep(
                name=f"{EXPERIMENT_NAME}/{run_spec.run_name}",
                fn=remote(
                    replay.run_phase_0_prefix,
                    resources=resources,
                    env_vars={base.HF_HUB_DISABLE_XET_ENV_VAR: "1"},
                ),
                resources=resources,
                config=replay.PrefixTrainingConfig(
                    analysis_output_path=analysis_output_path,
                    output_path=this_output_path(),
                    run_spec=run_spec,
                    validation_configs=validation_configs,
                    prefix_train_steps=replay.EXPECTED_PREFIX_TRAIN_STEPS,
                    optimizer_schedule_num_train_steps=replay.EXPECTED_FULL_TRAIN_STEPS,
                    wandb_tags=("sequential-prefix-candidate",),
                ),
            )
        )
    steps.append(
        ExecutorStep(
            name=f"{EXPERIMENT_NAME}/manifest",
            fn=save_candidate_manifest,
            config=SaveCandidateManifestConfig(
                output_path=this_output_path(),
                candidate_weights_path=str(candidate_weights_path),
                candidate_weights_sha256=candidate_weights_sha256,
                run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            ),
        )
    )
    return steps


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-weights", type=Path, default=DEFAULT_CANDIDATE_WEIGHTS)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--analysis-output-path", default=base.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = parse_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError(f"This launcher is pinned to {base.DEFAULT_TPU_REGION}/{base.DEFAULT_TPU_ZONE}")
    if not 1 <= args.max_concurrent <= MAX_CONCURRENT:
        raise ValueError(f"--max-concurrent must be in [1, {MAX_CONCURRENT}]")
    expected_prefix = marin_prefix_for_region(args.tpu_region)
    if os.environ.get("MARIN_PREFIX", expected_prefix) != expected_prefix:
        raise ValueError(f"MARIN_PREFIX must be {expected_prefix}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    run_specs = candidate_specs(
        candidate_weights_path=args.candidate_weights,
        expected_sha256=args.expected_candidate_sha256,
        analysis_output_path=args.analysis_output_path,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    if args.dry_run:
        LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        save_candidate_manifest(
            SaveCandidateManifestConfig(
                output_path=str(LOCAL_ARTIFACT_DIR),
                candidate_weights_path=str(args.candidate_weights),
                candidate_weights_sha256=args.expected_candidate_sha256,
                run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            )
        )
        logger.info("Wrote %d candidate validation specs under %s", len(run_specs), LOCAL_ARTIFACT_DIR)
        return

    validation_steps = base._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    with executor_context():
        steps = build_steps(
            run_specs=run_specs,
            analysis_output_path=args.analysis_output_path,
            validation_configs=validation_configs,
            candidate_weights_path=args.candidate_weights,
            candidate_weights_sha256=args.expected_candidate_sha256,
        )
    if os.getenv("CI") is not None:
        logger.info("Built %d prefix candidate validation steps; skipping launch in CI", len(steps))
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=steps,
        description=(f"{EXPERIMENT_NAME}: common-seed exact-boundary validation of the reserve-aware phase-0 KL path"),
    )


if __name__ == "__main__":
    main()
