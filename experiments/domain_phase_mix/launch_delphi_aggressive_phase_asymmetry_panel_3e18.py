# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the frozen aggressive frontier WSD annealing-content panel at Delphi 3e18."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import fsspec
import numpy as np
from fray.cluster import ResourceConfig
from marin.evaluation.olmo_base_eval.run import olmo_base_eval_step
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.rl.placement import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as augmented
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUT_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many" / "reference_outputs"
LOCAL_ARTIFACT_DIR = REFERENCE_OUTPUT_DIR / "delphi_3e18_aggressive_phase_asymmetry_20260722"
DEFAULT_SOURCE_PANEL = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_3e18_aggressive_phase_asymmetry_20260722/"
    "source/launcher_source_panel-a4afd81ebbd166e2.csv"
)
SOURCE_PANEL_SHA256 = "a4afd81ebbd166e2c2604d8a129129465334e5223585fcca4f498841fda71c87"
EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_3e18_aggressive_phase_asymmetry_20260722"
EXPECTED_RUNS = 290
EXPECTED_ANCHOR_COUNTS = {"uncheatable_frontier": 145, "table9_frontier": 145}
EXPECTED_FAMILY_COUNTS = {
    "balanced_partition": 192,
    "center_control": 32,
    "dolmino_late_continuum": 18,
    "handcrafted_late_quality": 48,
}
EXPECTED_SIGN_COUNTS = {"center": 32, "minus": 96, "plus": 162}
EXPECTED_RANDOM_DIRECTIONS = 16
EXPECTED_CENTER_CONTROLS = 16
EXPECTED_REPLICATED_TREATMENT_POLICIES = 6
EXPECTED_TREATMENT_REPLICATES = 3
EXPECTED_TV_LEVELS = {0.10, 0.25, 0.50}
RUN_ID_BASE = 7_223_000
DEFAULT_MAX_CONCURRENT = 56
AGGREGATE_TOLERANCE = 2e-12
GEOMETRY_TOLERANCE = 2e-10


@dataclass(frozen=True)
class AggressivePhaseMetadata:
    candidate_id: str
    policy_sha256: str
    anchor_id: str
    anchor_run_name: str
    anchor_source_run_name: str
    contrast_family: str
    direction_id: str
    direction_label: str
    sign: str
    replicate_index: int
    seed_block: int
    target_phase_tv: float
    phase_tv: float
    phase_information_kl: float
    aggregate_max_abs_error: float
    recipient_domain_count: int
    recipient_aggregate_share: float
    phase_0_dolmino_share: float
    phase_1_dolmino_share: float
    phase_0_broad_share: float
    phase_1_broad_share: float
    min_fit_policy_tv: float
    min_prior_policy_tv: float
    recipient_domains_json: str


@dataclass(frozen=True)
class SavePanelManifestConfig:
    output_path: str
    source_panel: str
    source_panel_sha256: str
    analysis_output_path: str
    run_specs_json: str
    metadata_json: str


@dataclass(frozen=True)
class LaunchArtifacts:
    manifest_step: ExecutorStep
    training_steps: list[ExecutorStep]
    eval_steps: list[ExecutorStep]

    @property
    def steps(self) -> list[ExecutorStep]:
        return [self.manifest_step, *self.training_steps, *self.eval_steps]


def _realized_phase_fractions(candidate) -> tuple[float, float]:
    phase_1_start_step = augmented.PHASE_SCHEDULE.phases[1].get_start_step_aligned(
        candidate.train_steps,
        augmented.TARGET_BATCH_SIZE,
        augmented.MIXTURE_BLOCK_SIZE,
    )
    alpha0 = phase_1_start_step / candidate.train_steps
    return alpha0, 1.0 - alpha0


def _weights_array(weights: dict[str, dict[str, float]]) -> np.ndarray:
    return np.asarray([[weights[phase][domain] for domain in augmented.DOMAIN_NAMES] for phase in augmented.PHASE_NAMES])


def _validate_panel_geometry(
    rows: list[dict[str, str]],
    phase_weights: list[dict[str, dict[str, float]]],
    alpha0: float,
    alpha1: float,
) -> None:
    row_indices = {row["candidate_id"]: index for index, row in enumerate(rows)}
    recipient_sets: dict[tuple[str, str], str] = {}
    for anchor_id in EXPECTED_ANCHOR_COUNTS:
        anchor_rows = [row for row in rows if row["anchor_id"] == anchor_id]
        centers = [row for row in anchor_rows if row["contrast_family"] == "center_control"]
        if len(centers) != EXPECTED_CENTER_CONTROLS:
            raise ValueError(f"{anchor_id} has {len(centers)} tied controls")
        center_arrays = [_weights_array(phase_weights[row_indices[row["candidate_id"]]]) for row in centers]
        anchor = center_arrays[0][0]
        for center in center_arrays:
            if np.max(np.abs(center - np.stack([anchor, anchor]))) > GEOMETRY_TOLERANCE:
                raise ValueError(f"{anchor_id} controls are not one shared tied coordinate")

        for row in anchor_rows:
            array = _weights_array(phase_weights[row_indices[row["candidate_id"]]])
            if float(array.min()) < -GEOMETRY_TOLERANCE:
                raise ValueError(f"{row['candidate_id']} has a negative phase weight")
            if np.max(np.abs(array.sum(axis=1) - 1.0)) > GEOMETRY_TOLERANCE:
                raise ValueError(f"{row['candidate_id']} is not on both phase simplexes")
            aggregate = alpha0 * array[0] + alpha1 * array[1]
            if np.max(np.abs(aggregate - anchor)) > AGGREGATE_TOLERANCE:
                raise ValueError(f"{row['candidate_id']} does not preserve its anchor aggregate")
            phase_tv = float(0.5 * np.abs(array[1] - array[0]).sum())
            if abs(phase_tv - float(row["phase_tv"])) > GEOMETRY_TOLERANCE:
                raise ValueError(f"{row['candidate_id']} phase TV does not match source metadata")
            if float(row["phase_information_kl"]) < -GEOMETRY_TOLERANCE:
                raise ValueError(f"{row['candidate_id']} has negative phase information")
            recipient_sets[(anchor_id, row["direction_id"])] = row["recipient_domains_json"]

        random_rows = [row for row in anchor_rows if row["contrast_family"] == "balanced_partition"]
        direction_groups: dict[str, list[dict[str, str]]] = {}
        for row in random_rows:
            direction_groups.setdefault(row["direction_id"], []).append(row)
        if len(direction_groups) != EXPECTED_RANDOM_DIRECTIONS:
            raise ValueError(f"{anchor_id} has {len(direction_groups)} balanced partitions")
        for direction_id, direction_rows in direction_groups.items():
            if len(direction_rows) != 2 * len(EXPECTED_TV_LEVELS):
                raise ValueError(f"{anchor_id}/{direction_id} has the wrong number of rows")
            if {float(row["target_phase_tv"]) for row in direction_rows} != EXPECTED_TV_LEVELS:
                raise ValueError(f"{anchor_id}/{direction_id} changed TV levels")
            for phase_tv in EXPECTED_TV_LEVELS:
                pair_rows = [row for row in direction_rows if float(row["target_phase_tv"]) == phase_tv]
                if {row["sign"] for row in pair_rows} != {"plus", "minus"}:
                    raise ValueError(f"{anchor_id}/{direction_id}/TV={phase_tv} is not antithetic")
                pair = np.asarray([_weights_array(phase_weights[row_indices[row["candidate_id"]]]) for row in pair_rows])
                if np.max(np.abs(pair.mean(axis=0) - np.stack([anchor, anchor]))) > GEOMETRY_TOLERANCE:
                    raise ValueError(f"{anchor_id}/{direction_id}/TV={phase_tv} is not centered")

    for direction_index in range(EXPECTED_RANDOM_DIRECTIONS):
        direction_id = f"partition_{direction_index:02d}"
        left = recipient_sets[("uncheatable_frontier", direction_id)]
        right = recipient_sets[("table9_frontier", direction_id)]
        if left != right:
            raise ValueError(f"{direction_id} does not use the same partition across anchors")


def load_source_panel(
    *,
    source_panel: str,
    analysis_output_path: str,
    tpu_region: str,
    tpu_zone: str,
) -> tuple[list[augmented.DelphiSwarmRunSpec], list[AggressivePhaseMetadata]]:
    """Load and strictly validate the content-addressed aggressive panel."""
    with fsspec.open(source_panel, "rb") as handle:
        source_bytes = handle.read()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != SOURCE_PANEL_SHA256:
        raise ValueError(f"Source panel SHA-256 changed: {source_sha256} != {SOURCE_PANEL_SHA256}")
    rows = list(csv.DictReader(io.StringIO(source_bytes.decode("utf-8"))))
    if len(rows) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} rows, found {len(rows)}")
    candidate_ids = [row["candidate_id"] for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Aggressive panel has duplicate candidate IDs")
    run_ids = [int(row["run_id"]) for row in rows]
    if run_ids != list(range(RUN_ID_BASE, RUN_ID_BASE + EXPECTED_RUNS)):
        raise ValueError("Aggressive-panel run IDs are not the frozen contiguous range")
    if dict(Counter(row["anchor_id"] for row in rows)) != EXPECTED_ANCHOR_COUNTS:
        raise ValueError("Anchor counts changed")
    if dict(Counter(row["contrast_family"] for row in rows)) != EXPECTED_FAMILY_COUNTS:
        raise ValueError("Contrast-family counts changed")
    if dict(Counter(row["sign"] for row in rows)) != EXPECTED_SIGN_COUNTS:
        raise ValueError("Sign counts changed")
    treatment_rows = [row for row in rows if row["contrast_family"] != "center_control"]
    treatment_policy_counts = Counter(row["policy_sha256"] for row in treatment_rows)
    repeated_treatments = {policy_sha256: count for policy_sha256, count in treatment_policy_counts.items() if count > 1}
    if len(repeated_treatments) != EXPECTED_REPLICATED_TREATMENT_POLICIES:
        raise ValueError(f"Repeated treatment-policy count changed: {repeated_treatments}")
    if set(repeated_treatments.values()) != {EXPECTED_TREATMENT_REPLICATES}:
        raise ValueError(f"Treatment replicate counts changed: {repeated_treatments}")
    for policy_sha256 in repeated_treatments:
        replicate_rows = [row for row in treatment_rows if row["policy_sha256"] == policy_sha256]
        if {row["contrast_family"] for row in replicate_rows} != {"dolmino_late_continuum"}:
            raise ValueError(f"Repeated treatment {policy_sha256} is not a Dolmino continuum schedule")
        if {int(row["replicate_index"]) for row in replicate_rows} != set(range(EXPECTED_TREATMENT_REPLICATES)):
            raise ValueError(f"Repeated treatment {policy_sha256} has incorrect replicate indices")
        if len({int(row["data_seed"]) for row in replicate_rows}) != EXPECTED_TREATMENT_REPLICATES:
            raise ValueError(f"Repeated treatment {policy_sha256} does not use independent data seeds")

    scaling_fits = augmented._read_scaling_fits(analysis_output_path)
    candidate = augmented._candidate_for_budget(scaling_fits=scaling_fits)
    realized_train_tokens = candidate.train_steps * augmented.TARGET_BATCH_SIZE * augmented.SEQ_LEN_DELPHI
    if realized_train_tokens > augmented.SIMULATED_EPOCH_TARGET_BUDGET:
        raise ValueError("Resolved Delphi token budget exceeds the fixed simulated-epoch budget")
    alpha0, alpha1 = _realized_phase_fractions(candidate)
    non_embedding_params = int(candidate.model_config.total_trainable_params(0))
    total_params = int(candidate.model_config.total_trainable_params(augmented.completed_adamh_heuristic.vocab_size))
    tensor_parallel_size = augmented._tensor_parallel_size(candidate.model_config.hidden_dim, augmented.TARGET_TPU_TYPE)

    all_phase_weights = [augmented._phase_weights_from_row(row, source_run_name=row["candidate_id"]) for row in rows]
    _validate_panel_geometry(rows, all_phase_weights, alpha0, alpha1)
    run_specs = []
    metadata = []
    for run_order, (row, weights) in enumerate(zip(rows, all_phase_weights, strict=True)):
        max_epoch, q95_epoch, phase_tv_to_proportional = augmented._weight_diagnostics(weights)
        run_specs.append(
            augmented.DelphiSwarmRunSpec(
                run_order=run_order,
                run_id=int(row["run_id"]),
                run_name=f"agphase_{run_order:03d}_{row['candidate_id']}",
                source_run_name=row["candidate_id"],
                source_experiment=EXPERIMENT_NAME,
                panel_source="aggressive_frontier_phase_asymmetry",
                target_flops=augmented.TARGET_FLOPS,
                tpu_type=augmented.TARGET_TPU_TYPE,
                tpu_region=tpu_region,
                tpu_zone=tpu_zone,
                batch_size=augmented.TARGET_BATCH_SIZE,
                train_steps=candidate.train_steps,
                realized_train_tokens=realized_train_tokens,
                expected_checkpoint_step=candidate.train_steps - 1,
                model_hidden_dim=int(candidate.model_config.hidden_dim),
                model_layers=int(candidate.model_config.num_layers),
                non_embedding_params=non_embedding_params,
                total_trainable_params=total_params,
                tensor_parallel_size=tensor_parallel_size,
                data_seed=int(row["data_seed"]),
                trainer_seed=int(row["trainer_seed"]),
                phase_boundary=augmented.PHASE_BOUNDARIES[0],
                phase_0_fraction=alpha0,
                phase_1_fraction=alpha1,
                simulated_epoch_target_budget=augmented.SIMULATED_EPOCH_TARGET_BUDGET,
                available_top_level_tokens=augmented.TOP_LEVEL_TOTAL_AVAILABLE_TOKENS,
                max_simulated_epoch=max_epoch,
                q95_simulated_epoch=q95_epoch,
                mean_phase_tv_to_proportional=phase_tv_to_proportional,
                phase_weights=weights,
            )
        )
        metadata.append(
            AggressivePhaseMetadata(
                candidate_id=row["candidate_id"],
                policy_sha256=row["policy_sha256"],
                anchor_id=row["anchor_id"],
                anchor_run_name=row["anchor_run_name"],
                anchor_source_run_name=row["anchor_source_run_name"],
                contrast_family=row["contrast_family"],
                direction_id=row["direction_id"],
                direction_label=row["direction_label"],
                sign=row["sign"],
                replicate_index=int(row["replicate_index"]),
                seed_block=int(row["seed_block"]),
                target_phase_tv=float(row["target_phase_tv"]),
                phase_tv=float(row["phase_tv"]),
                phase_information_kl=float(row["phase_information_kl"]),
                aggregate_max_abs_error=float(row["aggregate_max_abs_error"]),
                recipient_domain_count=int(row["recipient_domain_count"]),
                recipient_aggregate_share=float(row["recipient_aggregate_share"]),
                phase_0_dolmino_share=float(row["phase_0_dolmino_share"]),
                phase_1_dolmino_share=float(row["phase_1_dolmino_share"]),
                phase_0_broad_share=float(row["phase_0_broad_share"]),
                phase_1_broad_share=float(row["phase_1_broad_share"]),
                min_fit_policy_tv=float(row["min_fit_policy_tv"]),
                min_prior_policy_tv=float(row["min_prior_policy_tv"]),
                recipient_domains_json=row["recipient_domains_json"],
            )
        )
    return run_specs, metadata


def save_panel_manifest(config: SavePanelManifestConfig) -> None:
    """Persist source provenance and fully resolved training configurations."""
    run_specs = [augmented.DelphiSwarmRunSpec(**item) for item in json.loads(config.run_specs_json)]
    metadata = [AggressivePhaseMetadata(**item) for item in json.loads(config.metadata_json)]
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(os.path.join(config.output_path, "run_specs.json"), "w") as handle:
        json.dump([asdict(spec) for spec in run_specs], handle, indent=2, sort_keys=True)
    with fs.open(os.path.join(config.output_path, "panel_metadata.json"), "w") as handle:
        json.dump([asdict(item) for item in metadata], handle, indent=2, sort_keys=True)
    with (
        fsspec.open(config.source_panel, "r") as source,
        fs.open(os.path.join(config.output_path, "source_panel.csv"), "w") as destination,
    ):
        destination.write(source.read())
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "source_panel": config.source_panel,
        "source_panel_sha256": config.source_panel_sha256,
        "analysis_output_path": config.analysis_output_path,
        "n_runs": len(run_specs),
        "anchor_counts": dict(Counter(item.anchor_id for item in metadata)),
        "contrast_family_counts": dict(Counter(item.contrast_family for item in metadata)),
        "target_phase_tv_counts": dict(Counter(item.target_phase_tv for item in metadata)),
        "target_flops": augmented.TARGET_FLOPS,
        "native_table9_scheduled": True,
        "realized_phase_fractions": {
            "phase_0": run_specs[0].phase_0_fraction,
            "phase_1": run_specs[0].phase_1_fraction,
        },
    }
    with fs.open(os.path.join(config.output_path, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def build_launch_artifacts(
    *,
    run_specs: list[augmented.DelphiSwarmRunSpec],
    metadata: list[AggressivePhaseMetadata],
    analysis_output_path: str,
    source_panel: str,
    validation_configs,
) -> LaunchArtifacts:
    """Build the 290-train plus 290-native-Table-9 graph."""
    training_steps = []
    eval_steps = []
    for run_spec, row_metadata in zip(run_specs, metadata, strict=True):
        resources = ResourceConfig.with_tpu(
            run_spec.tpu_type,
            regions=[run_spec.tpu_region],
            zone=run_spec.tpu_zone,
        )
        training_step = ExecutorStep(
            name=f"{EXPERIMENT_NAME}/{run_spec.run_name}",
            fn=remote(
                augmented.run_delphi_swarm_training,
                resources=resources,
                env_vars={augmented.HF_HUB_DISABLE_XET_ENV_VAR: "1"},
            ),
            resources=resources,
            config=augmented.DelphiSwarmTrainingConfig(
                analysis_output_path=analysis_output_path,
                output_path=this_output_path(),
                run_spec=run_spec,
                validation_configs=validation_configs,
                wandb_tags=(
                    "delphi-3e18-aggressive-phase-asymmetry",
                    "boundary-aligned-wsd-anneal-content",
                    "token-aggregate-matched-phase-contrast",
                    f"anchor={row_metadata.anchor_id}",
                    f"contrast_family={row_metadata.contrast_family}",
                    f"direction={row_metadata.direction_id}",
                    f"phase_tv={row_metadata.phase_tv:g}",
                    f"replicate={row_metadata.replicate_index}",
                    f"seed_block={row_metadata.seed_block}",
                ),
            ),
        )
        training_steps.append(training_step)
        eval_steps.append(
            olmo_base_eval_step(
                name=f"t9_{run_spec.run_name}",
                checkpoint=training_step / f"hf/step-{run_spec.expected_checkpoint_step}",
                request_set_dir=augmented.TABLE9_REQUEST_SET_DIR,
                resource_config=augmented.TABLE9_EVAL_RESOURCES,
                wandb_group="olmo_base_eval_table9_delphi_3e18_aggressive_phase_asymmetry_20260722",
                provenance={
                    "evaluator": "marin-native-table9-bpb",
                    "panel": "delphi_3e18_aggressive_phase_asymmetry_20260722",
                    "scale": "3e18",
                    **asdict(row_metadata),
                },
            )
        )
    manifest_step = ExecutorStep(
        name=f"{EXPERIMENT_NAME}/manifest",
        fn=save_panel_manifest,
        config=SavePanelManifestConfig(
            output_path=this_output_path(),
            source_panel=source_panel,
            source_panel_sha256=SOURCE_PANEL_SHA256,
            analysis_output_path=analysis_output_path,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            metadata_json=json.dumps([asdict(item) for item in metadata], sort_keys=True),
        ),
    )
    return LaunchArtifacts(manifest_step, training_steps, eval_steps)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", default=DEFAULT_SOURCE_PANEL)
    parser.add_argument("--analysis-output-path", default=augmented.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--tpu-region", default=augmented.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=augmented.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = parse_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.tpu_region != augmented.DEFAULT_TPU_REGION or args.tpu_zone != augmented.DEFAULT_TPU_ZONE:
        raise ValueError(f"This launcher is pinned to {augmented.DEFAULT_TPU_REGION}/{augmented.DEFAULT_TPU_ZONE}")
    if args.max_concurrent < 1 or args.max_concurrent > DEFAULT_MAX_CONCURRENT:
        raise ValueError(f"--max-concurrent must be in [1, {DEFAULT_MAX_CONCURRENT}]")
    if not args.dry_run and not args.source_panel.startswith("gs://marin-us-east5/"):
        raise ValueError("The production source panel must be stored in marin-us-east5")
    expected_prefix = marin_prefix_for_region(args.tpu_region)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    run_specs, metadata = load_source_panel(
        source_panel=args.source_panel,
        analysis_output_path=args.analysis_output_path,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    if args.dry_run:
        LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        save_panel_manifest(
            SavePanelManifestConfig(
                output_path=str(LOCAL_ARTIFACT_DIR / "launch_dry_run"),
                source_panel=args.source_panel,
                source_panel_sha256=SOURCE_PANEL_SHA256,
                analysis_output_path=args.analysis_output_path,
                run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
                metadata_json=json.dumps([asdict(item) for item in metadata], sort_keys=True),
            )
        )
        logger.info("Validated %d frozen aggressive phase-asymmetry rows", len(run_specs))
        return

    validation_steps = augmented._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    with executor_context():
        artifacts = build_launch_artifacts(
            run_specs=run_specs,
            metadata=metadata,
            analysis_output_path=args.analysis_output_path,
            source_panel=args.source_panel,
            validation_configs=validation_configs,
        )
    if os.getenv("CI") is not None:
        logger.info(
            "Built aggressive panel graph with %d training and %d Table-9 steps; skipping launch in CI",
            len(artifacts.training_steps),
            len(artifacts.eval_steps),
        )
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=(
            f"{EXPERIMENT_NAME}: 290-row token-aggregate-matched WSD annealing-content panel at Delphi 3e18 "
            "with native Table-9 evaluation"
        ),
    )


if __name__ == "__main__":
    main()
