# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the frozen two-anchor fixed-aggregate phase-order panel at 60M/1.2B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, replace

import fsspec
from fray.cluster import ResourceConfig
from levanter.data.text.datasets import LMMixtureDatasetConfig
from levanter.main.train_lm import TrainLmConfig
from marin.evaluation.olmo_base_eval.run import olmo_base_eval_step
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.execution.types import ExecutorStep, InputName, this_output_path
from marin.rl.placement import marin_prefix_for_region
from marin.training.training import TrainLmOnPodConfig

from experiments.datasets.uncheatable import UNCHEATABLE_SUBSETS
from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.proxy_sweep import regmix_60m_proxy
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    BATCH_SIZE,
    DOMAIN_NAMES,
    EXPERIMENT_BUDGET,
    NUM_TRAIN_STEPS,
    PHASE_NAMES,
    SEQ_LEN,
    TARGET_BUDGET,
    create_two_phase_dolma3_dolmino_top_level_experiment,
    resolve_two_phase_wsd_boundary_schedule,
)

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/dm60po_20260725"
PANEL_TAG = "60m-fixed-aggregate-phase-order-20260725"
EXPECTED_RUNS = 140
EXPECTED_CHECKPOINT_STEP = NUM_TRAIN_STEPS - 1
DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_MAX_CONCURRENT = 56
TABLE9_REQUEST_SET_DIR = InputName.hardcoded("raw/eval-datasets/olmo_base_eval_table9/v2")
TABLE9_RESOURCES = ResourceConfig.with_tpu("v6e-8", regions=["us-east5"], zone="us-east5-b", disk="80g")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SKIP_EVAL_HARNESS_ENV_VAR = "LEVANTER_SKIP_EVAL_HARNESS"
PALOMA_COMPONENT_PREFIX = "paloma/"
UNCHEATABLE_COMPONENT_PREFIX = "uncheatable_eval/"
UNCHEATABLE_CACHE_VERSION = "2026.06.28"
FAMILY_ABBREVIATIONS = {
    "mechanistic_curvature": "mc",
    "mechanistic_primary": "mp",
    "sentinel_repeat": "sr",
    "spanning_tangent": "st",
    "tied_control": "ct",
}
REALIZED_SCHEDULE = resolve_two_phase_wsd_boundary_schedule(
    experiment_budget=EXPERIMENT_BUDGET,
    batch_size=BATCH_SIZE,
    seq_len=SEQ_LEN,
)
ALPHA_0 = REALIZED_SCHEDULE.boundary_step / REALIZED_SCHEDULE.total_steps
ALPHA_1 = 1.0 - ALPHA_0


@dataclass(frozen=True)
class SavePanelManifestConfig:
    """Configuration for persisting the exact submitted panel."""

    output_path: str
    source_panel: str
    source_panel_sha256: str
    rows_json: str


@dataclass(frozen=True)
class LaunchArtifacts:
    """Resolved manifest, training, and native Table-9 graph."""

    manifest_step: ExecutorStep
    training_steps: list[ExecutorStep]
    eval_steps: list[ExecutorStep]

    @property
    def steps(self) -> list[ExecutorStep]:
        return [self.manifest_step, *self.training_steps, *self.eval_steps]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", required=True)
    parser.add_argument("--source-panel-sha256", required=True)
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def _load_rows(source_panel: str, source_panel_sha256: str) -> list[dict[str, str]]:
    with fsspec.open(source_panel, "rb") as handle:
        source_bytes = handle.read()
    actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha256 != source_panel_sha256:
        raise ValueError(f"Source panel SHA-256 changed: {actual_sha256} != {source_panel_sha256}")
    rows = list(csv.DictReader(io.StringIO(source_bytes.decode())))
    if len(rows) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} panel rows, found {len(rows)}")
    if len({row["candidate_id"] for row in rows}) != EXPECTED_RUNS:
        raise ValueError("Panel candidate IDs are not unique")
    if len({int(row["run_id"]) for row in rows}) != EXPECTED_RUNS:
        raise ValueError("Panel run IDs are not unique")
    if {row["anchor_id"] for row in rows} != {"uncheatable_frontier", "proportional"}:
        raise ValueError("Panel must use exactly the Uncheatable-frontier and proportional anchors")
    if any("table9" in row["anchor_id"].lower() for row in rows):
        raise ValueError("Table-9 anchor rows are intentionally deferred")
    for row in rows:
        if abs(float(row["realized_phase_0_fraction"]) - ALPHA_0) > 2e-12:
            raise ValueError(f"{row['candidate_id']} has the wrong realized phase-0 fraction")
        if abs(float(row["realized_phase_1_fraction"]) - ALPHA_1) > 2e-12:
            raise ValueError(f"{row['candidate_id']} has the wrong realized phase-1 fraction")
    return rows


def _phase_weights(row: dict[str, str]) -> dict[str, dict[str, float]]:
    phase_weights: dict[str, dict[str, float]] = {}
    for phase in PHASE_NAMES:
        weights = {domain: float(row[f"{phase}_{domain}"]) for domain in DOMAIN_NAMES}
        if any(value < 0 for value in weights.values()):
            raise ValueError(f"{row['candidate_id']} has a negative {phase} weight")
        if abs(sum(weights.values()) - 1.0) > 2e-12:
            raise ValueError(f"{row['candidate_id']} {phase} weights do not sum to one")
        phase_weights[phase] = weights
    aggregate_error = max(
        abs(
            ALPHA_0 * phase_weights["phase_0"][domain]
            + ALPHA_1 * phase_weights["phase_1"][domain]
            - float(row[f"aggregate_{domain}"])
        )
        for domain in DOMAIN_NAMES
    )
    if aggregate_error > 2e-12:
        raise ValueError(f"{row['candidate_id']} does not preserve its aggregate: {aggregate_error}")
    return phase_weights


def save_panel_manifest(config: SavePanelManifestConfig) -> None:
    """Persist source provenance and the exact submitted rows."""
    rows = json.loads(config.rows_json)
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(os.path.join(config.output_path, "panel_manifest.json"), "w") as handle:
        json.dump(
            {
                "experiment_name": EXPERIMENT_NAME,
                "source_panel": config.source_panel,
                "source_panel_sha256": config.source_panel_sha256,
                "row_count": len(rows),
                "smooth_eval_outcomes": ["uncheatable_eval_bpb", "olmo_base_eval_table9_macro_bpb"],
                "excluded_validation_families": ["paloma"],
                "skip_lm_eval_harness": True,
                "rows": rows,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    with fsspec.open(config.source_panel, "rb") as source:
        source_bytes = source.read()
    actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha256 != config.source_panel_sha256:
        raise ValueError(f"Source panel SHA-256 changed: {actual_sha256} != {config.source_panel_sha256}")
    with fs.open(os.path.join(config.output_path, "candidate_manifest.csv"), "wb") as destination:
        destination.write(source_bytes)


def _validate_uncheatable_caches(tpu_region: str) -> None:
    prefix = marin_prefix_for_region(tpu_region)
    cache_stats = [
        f"{prefix}/uncheatable_eval/{subset}-llama3/{UNCHEATABLE_CACHE_VERSION}/validation/.stats.json"
        for subset in UNCHEATABLE_SUBSETS
    ]
    missing = [path for path in cache_stats if not fsspec.open(path).fs.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing Uncheatable validation caches: {missing}")


def _configure_training_step(training_step: ExecutorStep, *, tpu_region: str) -> ExecutorStep:
    config = training_step.config
    if not isinstance(config, TrainLmOnPodConfig):
        raise TypeError(f"Expected TrainLmOnPodConfig for {training_step.name!r}, got {type(config)!r}")
    train_config = config.train_config
    if not isinstance(train_config, TrainLmConfig):
        raise TypeError(f"Expected TrainLmConfig for {training_step.name!r}, got {type(train_config)!r}")
    data = train_config.data
    if not isinstance(data, LMMixtureDatasetConfig):
        raise TypeError(f"Expected LMMixtureDatasetConfig for {training_step.name!r}, got {type(data)!r}")
    if data.num_validation_sequences is not None:
        raise ValueError(f"{training_step.name} unexpectedly uses num_validation_sequences")

    paloma_components = {name for name in data.components if name.startswith(PALOMA_COMPONENT_PREFIX)}
    uncheatable_components = {name for name in data.components if name.startswith(UNCHEATABLE_COMPONENT_PREFIX)}
    if not paloma_components:
        raise ValueError(f"{training_step.name} has no Paloma validation components to remove")
    if not uncheatable_components:
        raise ValueError(f"{training_step.name} has no Uncheatable validation components")

    retained_components = {
        name: component for name, component in data.components.items() if name not in paloma_components
    }
    if isinstance(data.train_weights, dict):
        retained_weights: dict[str, float] | list[tuple[int, dict[str, float]]] = {
            name: weight for name, weight in data.train_weights.items() if name not in paloma_components
        }
    elif isinstance(data.train_weights, list):
        retained_weights = [
            (step, {name: weight for name, weight in weights.items() if name not in paloma_components})
            for step, weights in data.train_weights
        ]
    else:
        raise TypeError(f"Unexpected train_weights type for {training_step.name!r}: {type(data.train_weights)!r}")

    data = replace(data, components=retained_components, train_weights=retained_weights)
    train_config = replace(train_config, data=data)
    env_vars = dict(config.env_vars or {})
    env_vars["MARIN_PREFIX"] = marin_prefix_for_region(tpu_region)
    env_vars[SKIP_EVAL_HARNESS_ENV_VAR] = "1"
    return replace(training_step, config=replace(config, train_config=train_config, env_vars=env_vars))


def build_launch_artifacts(
    *,
    rows: list[dict[str, str]],
    source_panel: str,
    source_panel_sha256: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
) -> LaunchArtifacts:
    """Build 140 training steps and their native Table-9 dependencies."""
    resources = ResourceConfig.with_tpu(tpu_type, regions=[tpu_region], zone=tpu_zone)
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=EXPERIMENT_NAME,
        experiment_budget=EXPERIMENT_BUDGET,
        target_budget=TARGET_BUDGET,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        model_config=regmix_60m_proxy,
        resources=resources,
        eval_harness_tasks=(),
        runtime_cache_region=tpu_region,
    )
    training_steps: list[ExecutorStep] = []
    eval_steps: list[ExecutorStep] = []
    for index, row in enumerate(rows):
        anchor_code = {"uncheatable_frontier": "u", "proportional": "p"}[row["anchor_id"]]
        family_code = FAMILY_ABBREVIATIONS[row["direction_family"]]
        sign_code = {"plus": "p", "minus": "m", "control": "c"}[row["sign"]]
        run_name = f"p{index:03d}_{anchor_code}_{family_code}_{sign_code}r{row['replicate_index']}"
        training_step = experiment.create_training_step(
            weight_config=WeightConfig(run_id=int(row["run_id"]), phase_weights=_phase_weights(row)),
            name_prefix=EXPERIMENT_NAME,
            run_name=run_name,
            data_seed=int(row["data_seed"]),
            trainer_seed=int(row["trainer_seed"]),
            simulated_epoch_subset_seed=int(row["data_seed"]),
        )
        training_step = _configure_training_step(training_step, tpu_region=tpu_region)
        training_steps.append(training_step)
        eval_steps.append(
            olmo_base_eval_step(
                name=f"t9_{run_name}",
                checkpoint=training_step / f"hf/step-{EXPECTED_CHECKPOINT_STEP}",
                request_set_dir=TABLE9_REQUEST_SET_DIR,
                resource_config=TABLE9_RESOURCES,
                wandb_group="olmo_base_eval_table9_60m_fixed_aggregate_phase_order_20260725",
                provenance={
                    "evaluator": "marin-native-table9-bpb",
                    "panel": PANEL_TAG,
                    "scale": "60m_1p2b",
                    "candidate_id": row["candidate_id"],
                    "anchor_id": row["anchor_id"],
                    "direction_family": row["direction_family"],
                    "direction_id": row["direction_id"],
                    "sign": row["sign"],
                    "replicate_index": row["replicate_index"],
                    "data_seed": row["data_seed"],
                },
            )
        )

    manifest_step = ExecutorStep(
        name=f"{EXPERIMENT_NAME}/manifest",
        fn=save_panel_manifest,
        config=SavePanelManifestConfig(
            output_path=this_output_path(),
            source_panel=source_panel,
            source_panel_sha256=source_panel_sha256,
            rows_json=json.dumps(rows, sort_keys=True),
        ),
    )
    return LaunchArtifacts(manifest_step=manifest_step, training_steps=training_steps, eval_steps=eval_steps)


def _validate_graph(artifacts: LaunchArtifacts, *, tpu_region: str) -> None:
    if len(artifacts.training_steps) != EXPECTED_RUNS or len(artifacts.eval_steps) != EXPECTED_RUNS:
        raise ValueError("Launch graph does not contain one training and Table-9 step per panel row")
    for training_step in artifacts.training_steps:
        config = training_step.config
        if not isinstance(config, TrainLmOnPodConfig):
            raise TypeError(f"Expected TrainLmOnPodConfig for {training_step.name!r}, got {type(config)!r}")
        if int(config.train_config.trainer.num_train_steps) != NUM_TRAIN_STEPS:
            raise ValueError(f"{training_step.name} does not preserve the 60M train-step count")
        env_vars = dict(config.env_vars or {})
        if env_vars.get("MARIN_PREFIX") != marin_prefix_for_region(tpu_region):
            raise ValueError(f"{training_step.name} has an invalid MARIN_PREFIX")
        if env_vars.get(SKIP_EVAL_HARNESS_ENV_VAR) != "1":
            raise ValueError(f"{training_step.name} does not disable the redundant lm-eval harness")
        train_config = config.train_config
        if not isinstance(train_config, TrainLmConfig):
            raise TypeError(f"Expected TrainLmConfig for {training_step.name!r}, got {type(train_config)!r}")
        data = train_config.data
        if not isinstance(data, LMMixtureDatasetConfig):
            raise TypeError(f"Expected LMMixtureDatasetConfig for {training_step.name!r}, got {type(data)!r}")
        if any(name.startswith(PALOMA_COMPONENT_PREFIX) for name in data.components):
            raise ValueError(f"{training_step.name} still contains Paloma validation components")
        uncheatable_components = {name for name in data.components if name.startswith(UNCHEATABLE_COMPONENT_PREFIX)}
        if not uncheatable_components:
            raise ValueError(f"{training_step.name} is missing Uncheatable validation components")
        weight_stages = [data.train_weights] if isinstance(data.train_weights, dict) else data.train_weights
        if weight_stages is None:
            raise ValueError(f"{training_step.name} is missing train weights")
        for stage in weight_stages:
            weights = stage if isinstance(stage, dict) else stage[1]
            if any(weights.get(name, 0.0) != 0.0 for name in uncheatable_components):
                raise ValueError(f"{training_step.name} assigns nonzero weight to an Uncheatable validation component")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = parse_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.tpu_region != DEFAULT_TPU_REGION or args.tpu_zone != DEFAULT_TPU_ZONE:
        raise ValueError(f"This launcher is pinned to {DEFAULT_TPU_REGION}/{DEFAULT_TPU_ZONE}")
    if args.tpu_type != DEFAULT_TPU_TYPE:
        raise ValueError(f"This launcher preserves the original 60M swarm TPU type {DEFAULT_TPU_TYPE}")
    if args.max_concurrent < 1 or args.max_concurrent > DEFAULT_MAX_CONCURRENT:
        raise ValueError(f"--max-concurrent must be in [1, {DEFAULT_MAX_CONCURRENT}]")
    if SHA256_PATTERN.fullmatch(args.source_panel_sha256) is None:
        raise ValueError("--source-panel-sha256 must be a lowercase SHA-256 digest")
    if not args.dry_run and not args.source_panel.startswith("gs://marin-us-east5/"):
        raise ValueError("Production source panel must be stored in marin-us-east5")
    expected_prefix = marin_prefix_for_region(args.tpu_region)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix
    _validate_uncheatable_caches(args.tpu_region)

    rows = _load_rows(args.source_panel, args.source_panel_sha256)
    with executor_context():
        artifacts = build_launch_artifacts(
            rows=rows,
            source_panel=args.source_panel,
            source_panel_sha256=args.source_panel_sha256,
            tpu_type=args.tpu_type,
            tpu_region=args.tpu_region,
            tpu_zone=args.tpu_zone,
        )
    _validate_graph(artifacts, tpu_region=args.tpu_region)
    logger.info(
        "Validated %d 60M training rows, %d native Table-9 steps, and no Table-9-derived anchor.",
        len(artifacts.training_steps),
        len(artifacts.eval_steps),
    )
    if args.dry_run or os.getenv("CI") is not None:
        return

    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=(
            "60M/1.2B fixed-aggregate phase-order DOE around the Uncheatable-frontier and proportional anchors; "
            "each training row receives Uncheatable validation and Marin-native Table-9 evaluation."
        ),
    )


if __name__ == "__main__":
    main()
