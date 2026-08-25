# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the runtime-exact Delphi cap-4/cap-6 prefix candidates."""

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
import jax
import pandas as pd
from fray.cluster import ResourceConfig
from levanter.data.text.datasets import DatasetComponent
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main, get_git_commit
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from rigging.filesystem import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_3e18_phase0_prefix_replay as replay
from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_3e18_phase0_harsh_cap_candidates_v6e_r2_20260825"
DEFAULT_CANDIDATE_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "delphi_phase0_harsh_cap_candidates_20260825"
)
DEFAULT_CANDIDATE_WEIGHTS = DEFAULT_CANDIDATE_DIR / "training_candidate_weights.csv"
DEFAULT_ALIAS_PATH = DEFAULT_CANDIDATE_DIR / "candidate_aliases.csv"
LOCAL_ARTIFACT_DIR = DEFAULT_CANDIDATE_DIR / "launch_dry_run_v6e_only"
CANDIDATE_IDS = (
    "cap4_shared_bounded_ensemble_kl0",
    "cap4_shared_bounded_ensemble_kl0p05",
    "cap4_shared_bounded_ensemble_kl0p2",
    "cap4_shared_bounded_ensemble_kl0p5",
    "cap4_observed_cap4_best",
    "cap4_proportional_control",
    "cap6_shared_bounded_ensemble_kl0",
    "cap6_shared_bounded_ensemble_kl0p05",
    "cap6_shared_bounded_ensemble_kl0p2",
    "cap6_shared_bounded_ensemble_kl0p5",
)
REPEAT_SEEDS = (0, 1, 2)
RUN_ID_BASE = 965_000
DATA_SEED_BASE = 965_000
MAX_CONCURRENT = len(CANDIDATE_IDS) * len(REPEAT_SEEDS)
CANDIDATE_PROVENANCE_FILENAME = "prefix_provenance.json"
TPU_TYPE = "v6e-8"
TPU_REGION = "us-east5"
TPU_ZONE = "us-east5-b"
PARENT_ZONE = "us-east5-a"
EXPECTED_GLOBAL_DEVICE_COUNT = 8
TENSOR_PARALLEL_SIZE = 1
PANEL_HARDWARE_STATUS = "v6e_only"


@dataclass(frozen=True)
class HarshCandidatePrefixTrainingConfig:
    prefix_config: replay.PrefixTrainingConfig
    candidate_id: str
    candidate_weights_sha256: str
    candidate_aliases_sha256: str


@dataclass(frozen=True)
class SaveHarshCandidateManifestConfig:
    output_path: str
    candidate_weights_path: str
    candidate_weights_sha256: str
    candidate_aliases_path: str
    candidate_aliases_sha256: str
    replay_code_commit: str
    launch_audit_json: str
    run_specs_json: str


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_candidate_weights(
    path: Path,
    expected_sha256: str,
) -> tuple[tuple[str, ...], dict[str, dict[str, float]], dict[str, tuple[float, float]]]:
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Candidate weights changed: {actual_sha256} != {expected_sha256}")
    frame = pd.read_csv(path)
    required = {
        "candidate_id",
        "bucket",
        "phase_0_weight",
        "phase_0_count",
        "phase_0_materialized_epochs",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Candidate weights are missing columns: {sorted(required - set(frame.columns))}")
    if tuple(frame.candidate_id.drop_duplicates()) != CANDIDATE_IDS:
        raise ValueError("Candidate order or identities changed")

    buckets = tuple(frame.loc[frame.candidate_id.eq(CANDIDATE_IDS[0]), "bucket"])
    candidates = {}
    exposure_diagnostics = {}
    for candidate_id in CANDIDATE_IDS:
        rows = frame[frame.candidate_id.eq(candidate_id)]
        if tuple(rows.bucket) != buckets:
            raise ValueError(f"Bucket order changed for {candidate_id}")
        counts = rows.phase_0_count.to_numpy(dtype=int)
        weights = rows.phase_0_weight.to_numpy(dtype=float)
        if counts.sum() != replay.MIXTURE_BLOCK_SIZE or not (counts >= 0).all():
            raise ValueError(f"Invalid runtime counts for {candidate_id}")
        if not (weights == counts / replay.MIXTURE_BLOCK_SIZE).all():
            raise ValueError(f"Weights are not exact runtime counts for {candidate_id}")
        cap = 4.0 if candidate_id.startswith("cap4_") else 6.0
        if float(rows.phase_0_materialized_epochs.max()) > cap + 1e-12:
            raise ValueError(f"Prefix epoch cap {cap:g} violated by {candidate_id}")
        candidates[candidate_id] = dict(zip(buckets, weights, strict=True))
        exposures = rows.phase_0_materialized_epochs.to_numpy(dtype=float)
        exposure_diagnostics[candidate_id] = (float(exposures.max()), float(pd.Series(exposures).quantile(0.95)))
    return buckets, candidates, exposure_diagnostics


def phase_weights_sha256(phase_weights: dict[str, dict[str, float]]) -> str:
    return hashlib.sha256(json.dumps(phase_weights, sort_keys=True).encode()).hexdigest()


def candidate_id_for_spec(run_spec: base.DelphiSwarmRunSpec) -> str:
    suffix = f"_seed{run_spec.trainer_seed}"
    return run_spec.run_name.removeprefix("prefix_").removesuffix(suffix)


def candidate_specs(
    *,
    candidate_weights_path: Path,
    expected_sha256: str,
    analysis_output_path: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
) -> tuple[list[base.DelphiSwarmRunSpec], dict[str, object]]:
    buckets, candidates, exposure_diagnostics = load_candidate_weights(candidate_weights_path, expected_sha256)
    source_specs, source_launch_audit = replay.load_replay_specs(
        source_panel=base.DEFAULT_SOURCE_PANEL,
        analysis_output_path=analysis_output_path,
        tpu_region=tpu_region,
        tpu_zone=tpu_zone,
    )
    template = source_specs[0]
    runtime_buckets = tuple(template.phase_weights["phase_0"])
    if set(runtime_buckets) != set(buckets):
        raise ValueError("Candidate buckets do not match the Delphi runtime bucket set")

    specs = []
    for candidate_position, candidate_id in enumerate(CANDIDATE_IDS):
        weights = {bucket: candidates[candidate_id][bucket] for bucket in runtime_buckets}
        phase_weights = {"phase_0": weights, "phase_1": weights}
        max_epoch, q95_epoch = exposure_diagnostics[candidate_id]
        for repeat_position, repeat in enumerate(REPEAT_SEEDS):
            run_order = candidate_position * len(REPEAT_SEEDS) + repeat_position
            run_name = f"prefix_{candidate_id}_seed{repeat}"
            specs.append(
                replace(
                    template,
                    run_order=run_order,
                    run_id=RUN_ID_BASE + run_order,
                    run_name=run_name,
                    source_run_name=run_name,
                    source_experiment=EXPERIMENT_NAME,
                    panel_source="sequential_prefix_harsh_cap_candidate",
                    tpu_type=tpu_type,
                    tpu_region=tpu_region,
                    tpu_zone=tpu_zone,
                    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
                    data_seed=DATA_SEED_BASE + repeat,
                    trainer_seed=repeat,
                    max_simulated_epoch=max_epoch,
                    q95_simulated_epoch=q95_epoch,
                    mean_phase_tv_to_proportional=0.0,
                    expected_checkpoint_step=replay.EXPECTED_PREFIX_HF_STEP,
                    phase_weights=phase_weights,
                )
            )
    return specs, source_launch_audit


def run_harsh_candidate_prefix(config: HarshCandidatePrefixTrainingConfig) -> None:
    """Train one prefix and bind the permanent checkpoint to frozen inputs."""
    replay.run_phase_0_prefix(config.prefix_config)
    devices = jax.devices()
    if (
        jax.default_backend() != "tpu"
        or jax.device_count() != EXPECTED_GLOBAL_DEVICE_COUNT
        or jax.local_device_count() != EXPECTED_GLOBAL_DEVICE_COUNT
        or any("v6" not in device.device_kind.lower() for device in devices)
    ):
        raise ValueError(
            "Harsh-cap prefix validation requires exactly eight local/global v6 TPU devices: "
            f"backend={jax.default_backend()}, global={jax.device_count()}, local={jax.local_device_count()}, "
            f"kinds={[device.device_kind for device in devices]}"
        )
    run_spec = config.prefix_config.run_spec
    checkpoint_uri = os.path.join(
        config.prefix_config.output_path,
        "checkpoints",
        f"step-{replay.EXPECTED_PREFIX_HF_STEP}",
    )
    fs, checkpoint_path = fsspec.core.url_to_fs(checkpoint_uri)
    metadata_path = os.path.join(checkpoint_path, "metadata.json")
    if not fs.exists(metadata_path):
        raise FileNotFoundError(f"Candidate checkpoint metadata is missing: {checkpoint_uri}")
    with fs.open(metadata_path) as handle:
        metadata = json.load(handle)
    if metadata.get("step") != replay.EXPECTED_PREFIX_HF_STEP or metadata.get("is_temporary") is not False:
        raise ValueError(f"Candidate checkpoint is not the permanent boundary state: {metadata}")

    provenance = {
        "experiment_name": EXPERIMENT_NAME,
        "candidate_id": config.candidate_id,
        "candidate_weights_sha256": config.candidate_weights_sha256,
        "candidate_aliases_sha256": config.candidate_aliases_sha256,
        "phase_weights_sha256": phase_weights_sha256(run_spec.phase_weights),
        "replay_code_commit": config.prefix_config.replay_code_commit,
        "run_name": run_spec.run_name,
        "run_order": run_spec.run_order,
        "run_id": run_spec.run_id,
        "data_seed": run_spec.data_seed,
        "trainer_seed": run_spec.trainer_seed,
        "checkpoint_uri": checkpoint_uri,
        "checkpoint_step": replay.EXPECTED_PREFIX_HF_STEP,
        "trainer_state_step": replay.EXPECTED_PREFIX_TRAIN_STEPS,
        "tpu_type": TPU_TYPE,
        "tpu_region": TPU_REGION,
        "tpu_zone": TPU_ZONE,
        "observed_device_kinds": sorted({device.device_kind for device in devices}),
        "observed_global_device_count": jax.device_count(),
        "observed_local_device_count": jax.local_device_count(),
        "panel_hardware_status": PANEL_HARDWARE_STATUS,
    }
    output_fs, output_path = fsspec.core.url_to_fs(config.prefix_config.output_path)
    provenance_path = os.path.join(output_path, CANDIDATE_PROVENANCE_FILENAME)
    payload = (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode()
    if output_fs.exists(provenance_path):
        with output_fs.open(provenance_path, "rb") as handle:
            if handle.read() != payload:
                raise ValueError(f"Refusing to replace different candidate provenance: {provenance_path}")
        return
    with output_fs.open(provenance_path, "wb") as handle:
        handle.write(payload)


def save_harsh_candidate_manifest(config: SaveHarshCandidateManifestConfig) -> None:
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "candidate_weights_path": config.candidate_weights_path,
        "candidate_weights_sha256": config.candidate_weights_sha256,
        "candidate_aliases_path": config.candidate_aliases_path,
        "candidate_aliases_sha256": config.candidate_aliases_sha256,
        "canonical_candidate_ids": list(CANDIDATE_IDS),
        "repeat_seed_pairs": [
            {"repeat": seed, "data_seed": DATA_SEED_BASE + seed, "trainer_seed": seed} for seed in REPEAT_SEEDS
        ],
        "selection_target": "mean exact-boundary Uncheatable BPB over paired seeds 0, 1, and 2",
        "selection_rule": (
            "Within each cap, select the eligible KL candidate with the lowest three-seed mean boundary "
            "Uncheatable BPB; break an exact tie by the lower KL penalty. Controls are diagnostic only."
        ),
        "prefix_train_steps": replay.EXPECTED_PREFIX_TRAIN_STEPS,
        "optimizer_schedule_num_train_steps": replay.EXPECTED_FULL_TRAIN_STEPS,
        "mixture_block_size": replay.MIXTURE_BLOCK_SIZE,
        "replay_code_commit": config.replay_code_commit,
        "launch_audit": json.loads(config.launch_audit_json),
        "run_specs": json.loads(config.run_specs_json),
    }
    manifest_path = os.path.join(config.output_path, "manifest.json")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if fs.exists(manifest_path):
        with fs.open(manifest_path, "rb") as handle:
            if handle.read() != encoded:
                raise ValueError(f"Refusing to replace different frozen manifest: {manifest_path}")
        return
    with fs.open(manifest_path, "wb") as handle:
        handle.write(encoded)


def build_steps(
    *,
    run_specs: list[base.DelphiSwarmRunSpec],
    analysis_output_path: str,
    validation_configs: dict[str, DatasetComponent],
    candidate_weights_path: Path,
    candidate_weights_sha256: str,
    candidate_aliases_path: Path,
    candidate_aliases_sha256: str,
    replay_code_commit: str,
    launch_audit: dict[str, object],
) -> list[ExecutorStep]:
    steps = []
    for run_spec in run_specs:
        resources = ResourceConfig.with_tpu(
            run_spec.tpu_type,
            regions=[run_spec.tpu_region],
            zone=run_spec.tpu_zone,
        )
        candidate_id = candidate_id_for_spec(run_spec)
        steps.append(
            ExecutorStep(
                name=f"{EXPERIMENT_NAME}/{run_spec.run_name}",
                fn=remote(
                    run_harsh_candidate_prefix,
                    resources=resources,
                    env_vars={base.HF_HUB_DISABLE_XET_ENV_VAR: "1"},
                ),
                resources=resources,
                config=HarshCandidatePrefixTrainingConfig(
                    prefix_config=replay.PrefixTrainingConfig(
                        analysis_output_path=analysis_output_path,
                        output_path=this_output_path(),
                        run_spec=run_spec,
                        validation_configs=validation_configs,
                        prefix_train_steps=replay.EXPECTED_PREFIX_TRAIN_STEPS,
                        optimizer_schedule_num_train_steps=replay.EXPECTED_FULL_TRAIN_STEPS,
                        replay_code_commit=replay_code_commit,
                        tracker_tags=(
                            "issue-6611",
                            "delphi-3e18-phase0-harsh-cap-validation",
                            f"prefix_candidate={candidate_id}",
                            f"replay_code_commit={replay_code_commit}",
                            f"data_seed={run_spec.data_seed}",
                            f"trainer_seed={run_spec.trainer_seed}",
                            f"prefix_tpu={TPU_TYPE}",
                            f"prefix_zone={TPU_ZONE}",
                            "selection_target=uncheatable",
                        ),
                    ),
                    candidate_id=candidate_id,
                    candidate_weights_sha256=candidate_weights_sha256,
                    candidate_aliases_sha256=candidate_aliases_sha256,
                ),
            )
        )
    steps.append(
        ExecutorStep(
            name=f"{EXPERIMENT_NAME}/manifest",
            fn=save_harsh_candidate_manifest,
            config=SaveHarshCandidateManifestConfig(
                output_path=this_output_path(),
                candidate_weights_path=str(candidate_weights_path),
                candidate_weights_sha256=candidate_weights_sha256,
                candidate_aliases_path=str(candidate_aliases_path),
                candidate_aliases_sha256=candidate_aliases_sha256,
                replay_code_commit=replay_code_commit,
                launch_audit_json=json.dumps(launch_audit, sort_keys=True),
                run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            ),
        )
    )
    return steps


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-weights", type=Path, default=DEFAULT_CANDIDATE_WEIGHTS)
    parser.add_argument("--candidate-aliases", type=Path, default=DEFAULT_ALIAS_PATH)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--expected-aliases-sha256", required=True)
    parser.add_argument("--analysis-output-path", default=base.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--tpu-region", default=TPU_REGION)
    parser.add_argument("--tpu-zone", default=TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--replay-code-commit", required=True)
    parser.add_argument("--run-order", action="append", type=int, dest="run_orders")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = parse_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.tpu_region != TPU_REGION or args.tpu_zone != TPU_ZONE:
        raise ValueError(f"This launcher is pinned to {TPU_REGION}/{TPU_ZONE}")
    if not 1 <= args.max_concurrent <= MAX_CONCURRENT:
        raise ValueError(f"--max-concurrent must be in [1, {MAX_CONCURRENT}]")
    expected_prefix = marin_prefix_for_region(args.tpu_region)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match required prefix {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    replay_code_commit = replay.validate_replay_code_commit(args.replay_code_commit, get_git_commit())
    run_specs, source_launch_audit = candidate_specs(
        candidate_weights_path=args.candidate_weights,
        expected_sha256=args.expected_candidate_sha256,
        analysis_output_path=args.analysis_output_path,
        tpu_type=TPU_TYPE,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    if args.run_orders is not None:
        selected_orders = tuple(dict.fromkeys(args.run_orders))
        unknown_orders = sorted(set(selected_orders) - {spec.run_order for spec in run_specs})
        if unknown_orders:
            raise ValueError(f"Unknown --run-order values: {unknown_orders}")
        run_specs = [spec for spec in run_specs if spec.run_order in selected_orders]
    aliases_sha256 = file_sha256(args.candidate_aliases)
    if aliases_sha256 != args.expected_aliases_sha256:
        raise ValueError(f"Candidate aliases changed: {aliases_sha256} != {args.expected_aliases_sha256}")
    launch_audit = {
        "experiment_name": EXPERIMENT_NAME,
        "source_panel": source_launch_audit["source_panel"],
        "source_panel_sha256": source_launch_audit["source_panel_sha256"],
        "source_coordinate_hash": source_launch_audit["source_coordinate_hash"],
        "source_panel_run_count": source_launch_audit["run_count"],
        "selected_run_count": len(run_specs),
        "selected_run_orders": [spec.run_order for spec in run_specs],
        "candidate_weights_sha256": args.expected_candidate_sha256,
        "candidate_aliases_sha256": aliases_sha256,
        "replay_code_commit": replay_code_commit,
        "prefix_completed_updates": replay.EXPECTED_PREFIX_TRAIN_STEPS,
        "prefix_checkpoint_step": replay.EXPECTED_PREFIX_HF_STEP,
        "optimizer_schedule_num_train_steps": replay.EXPECTED_FULL_TRAIN_STEPS,
        "smooth_uncheatable_boundary_eval_scheduled": True,
        "native_table9_boundary_eval_scheduled": False,
        "full_trainer_state_retained": True,
        "max_concurrent": args.max_concurrent,
        "parent_zone": PARENT_ZONE,
        "prefix_hardware": {"tpu_type": TPU_TYPE, "region": TPU_REGION, "zone": TPU_ZONE},
        "panel_hardware_status": PANEL_HARDWARE_STATUS,
        "identity_contract": {
            "preserved": [
                "source panel bucket set and data-loader implementation",
                "model architecture and optimizer configuration",
                "3007-update optimizer schedule horizon",
                "batch size, sequence length, precision, and logical mesh",
            ],
            "deliberate_changes": [
                "phase-0 weights follow the frozen cap-4/cap-6 KL ladder",
                "data and trainer seeds follow the frozen paired three-seed validation design",
                "execution stops after update 2400",
                "candidate-specific output paths and W&B tags replace fit-panel tags",
                "prefix hardware is v6e-8 in east5b",
                "a v6e-specific experiment namespace isolates this panel from prior executor state",
            ],
        },
    }
    if args.dry_run:
        dry_run_output_path = LOCAL_ARTIFACT_DIR / replay_code_commit[:12]
        save_harsh_candidate_manifest(
            SaveHarshCandidateManifestConfig(
                output_path=str(dry_run_output_path),
                candidate_weights_path=str(args.candidate_weights),
                candidate_weights_sha256=args.expected_candidate_sha256,
                candidate_aliases_path=str(args.candidate_aliases),
                candidate_aliases_sha256=aliases_sha256,
                replay_code_commit=replay_code_commit,
                launch_audit_json=json.dumps(launch_audit, sort_keys=True),
                run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            )
        )
        logger.info("Wrote %d candidate validation specs under %s", len(run_specs), dry_run_output_path)
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
            candidate_aliases_path=args.candidate_aliases,
            candidate_aliases_sha256=aliases_sha256,
            replay_code_commit=replay_code_commit,
            launch_audit=launch_audit,
        )
    if os.getenv("CI") is not None:
        logger.info("Built %d harsh-cap candidate-validation steps; skipping launch in CI", len(steps))
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=steps,
        description=f"{EXPERIMENT_NAME}: paired-seed exact-boundary validation of cap-4/cap-6 KL paths",
    )


if __name__ == "__main__":
    main()
