# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "numpy", "pandas", "torch"]
# ///
"""Launch a one-row GRP no-L2 single-phase exposure-average validation run."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

import fsspec
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main, this_output_path
from marin.rl.placement import marin_prefix_for_region
from marin.training.training import TrainLmOnPodConfig
import numpy as np
import pandas as pd

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.determinism_analysis import (
    FIT_DATASET_CSV,
    FIT_DATASET_SUMMARY_JSON,
    RESULTS_CSV,
    RUN_MANIFEST_FILE,
    create_fit_dataset_export_step,
    create_manifest_results_step,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_study import (
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from experiments.domain_phase_mix.proxy_sweep import regmix_60m_proxy
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    BATCH_SIZE,
    DOMAIN_NAMES,
    EXPERIMENT_BUDGET,
    NUM_TRAIN_STEPS,
    PHASE_NAMES,
    REALIZED_EXPERIMENT_BUDGET,
    SEQ_LEN,
    TARGET_BUDGET,
    create_two_phase_dolma3_dolmino_top_level_experiment,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_penalty_raw_optima_baselines import (
    GENERICFAMILY_PENALTY_RAW_OPTIMA_SOURCE_EXPERIMENT,
    genericfamily_penalty_raw_optimum_summary,
)

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_single_phase_exposure_average_grp_no_l2_60m_1p2b"
MODEL_FAMILY = "regmix_60m_proxy"
COHORT = "single_phase_exposure_average_validation"
SOURCE_VARIANT = "power_family_penalty_no_l2"
SINGLE_PHASE_STRATEGY = "exposure_average_80_20"
RUN_NAME = "singleavg_baseline_genericfamily_power_family_penalty_no_l2_raw_optimum"
DEFAULT_LOCAL_ARTIFACT_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "single_phase_exposure_average_grp_no_l2_60m_1p2b"
)
LOCAL_MANIFEST_CSV = "single_phase_grp_no_l2_manifest.csv"
LOCAL_SUMMARY_JSON = "single_phase_grp_no_l2_summary.json"
LOCAL_RUN_SPEC_JSON = "single_phase_grp_no_l2_run_spec.json"
OBJECTIVE_METRIC = "eval/uncheatable_eval/bpb"
TARGET_BUDGET_MULTIPLIER = 1.0
PHASE_FRACTIONS = {"phase_0": 0.8, "phase_1": 0.2}
DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_MAX_CONCURRENT = 1
SKIP_EVAL_HARNESS_ENV_VAR = "LEVANTER_SKIP_EVAL_HARNESS"


@dataclass(frozen=True)
class SinglePhaseGrpNoL2RunSpec:
    """Manifest entry for the single-phase GRP no-L2 validation run."""

    run_id: int
    run_name: str
    cohort: str
    model_family: str
    trainer_seed: int | None
    data_seed: int
    simulated_epoch_subset_seed: int | None
    source_run_id: int
    source_run_name: str
    source_two_phase_experiment: str
    candidate_run_id: int
    candidate_run_name: str
    candidate_source_experiment: str
    source_variant: str
    single_phase_strategy: str
    phase_tv: float
    experiment_budget: int
    realized_experiment_budget: int
    target_budget: int
    target_budget_multiplier: float
    num_train_steps: int
    target_final_checkpoint_step: int
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class SaveSinglePhaseGrpNoL2ManifestConfig:
    """Config for writing the single-row validation manifest."""

    output_path: str
    experiment_name: str
    run_spec_json: str


@dataclass(frozen=True)
class SinglePhaseGrpNoL2LaunchArtifacts:
    """Resolved executor steps for the single-row validation."""

    run_spec: SinglePhaseGrpNoL2RunSpec
    run_manifest_step: ExecutorStep
    training_step: ExecutorStep
    results_step: ExecutorStep
    fit_dataset_step: ExecutorStep

    @property
    def steps(self) -> list[object]:
        return [self.run_manifest_step, self.training_step, self.results_step, self.fit_dataset_step]


def _phase_column(phase_name: str, domain_name: str) -> str:
    return f"{phase_name}_{domain_name}"


def _domain_vector(phase_weights: dict[str, float]) -> np.ndarray:
    values = np.asarray([float(phase_weights[domain_name]) for domain_name in DOMAIN_NAMES])
    if np.any(values < 0):
        raise ValueError("GRP no-L2 source weights contain negative values")
    total = float(values.sum())
    if total <= 0:
        raise ValueError(f"GRP no-L2 source weights have non-positive sum {total}")
    return values / total


def _phase_weights_from_vector(weights: np.ndarray) -> dict[str, dict[str, float]]:
    domain_weights = {domain_name: float(weight) for domain_name, weight in zip(DOMAIN_NAMES, weights, strict=True)}
    return {"phase_0": dict(domain_weights), "phase_1": dict(domain_weights)}


def build_run_spec() -> SinglePhaseGrpNoL2RunSpec:
    """Build the GRP no-L2 single-phase exposure-average validation spec."""
    summary = genericfamily_penalty_raw_optimum_summary(SOURCE_VARIANT)
    phase_0 = _domain_vector(summary.phase_weights["phase_0"])
    phase_1 = _domain_vector(summary.phase_weights["phase_1"])
    single = PHASE_FRACTIONS["phase_0"] * phase_0 + PHASE_FRACTIONS["phase_1"] * phase_1
    single = single / single.sum()
    phase_tv = float(0.5 * np.abs(phase_1 - phase_0).sum())
    spec = SinglePhaseGrpNoL2RunSpec(
        run_id=summary.run_id,
        run_name=RUN_NAME,
        cohort=COHORT,
        model_family=MODEL_FAMILY,
        trainer_seed=None,
        data_seed=0,
        simulated_epoch_subset_seed=None,
        source_run_id=summary.run_id,
        source_run_name=summary.run_name,
        source_two_phase_experiment=GENERICFAMILY_PENALTY_RAW_OPTIMA_SOURCE_EXPERIMENT,
        candidate_run_id=summary.run_id,
        candidate_run_name=summary.run_name,
        candidate_source_experiment=GENERICFAMILY_PENALTY_RAW_OPTIMA_SOURCE_EXPERIMENT,
        source_variant=SOURCE_VARIANT,
        single_phase_strategy=SINGLE_PHASE_STRATEGY,
        phase_tv=phase_tv,
        experiment_budget=EXPERIMENT_BUDGET,
        realized_experiment_budget=REALIZED_EXPERIMENT_BUDGET,
        target_budget=TARGET_BUDGET,
        target_budget_multiplier=TARGET_BUDGET_MULTIPLIER,
        num_train_steps=NUM_TRAIN_STEPS,
        target_final_checkpoint_step=NUM_TRAIN_STEPS - 1,
        phase_weights=_phase_weights_from_vector(single),
    )
    validate_run_spec(spec)
    return spec


def validate_run_spec(spec: SinglePhaseGrpNoL2RunSpec) -> None:
    """Validate the single-row manifest invariants before launch."""
    if set(spec.phase_weights) != set(PHASE_NAMES):
        raise ValueError(f"{spec.run_name} phase names do not match expected {PHASE_NAMES}")
    phase_0 = spec.phase_weights["phase_0"]
    phase_1 = spec.phase_weights["phase_1"]
    if set(phase_0) != set(DOMAIN_NAMES) or set(phase_1) != set(DOMAIN_NAMES):
        raise ValueError(f"{spec.run_name} domain names do not match expected top-level domains")
    for phase_name, weights in spec.phase_weights.items():
        values = np.asarray(list(weights.values()), dtype=float)
        if np.any(values < 0):
            raise ValueError(f"{spec.run_name} has negative {phase_name} weights")
        if not np.isclose(values.sum(), 1.0, atol=1e-12):
            raise ValueError(f"{spec.run_name} {phase_name} weights sum to {values.sum()}, not 1")
    max_abs_delta = max(abs(phase_0[domain] - phase_1[domain]) for domain in DOMAIN_NAMES)
    if max_abs_delta > 1e-15:
        raise ValueError(f"{spec.run_name} is not single-phase: max phase delta {max_abs_delta}")


def _flat_manifest_row(spec: SinglePhaseGrpNoL2RunSpec) -> dict[str, Any]:
    row = asdict(spec)
    phase_weights = row.pop("phase_weights")
    for phase_name, weights in phase_weights.items():
        for domain_name, value in weights.items():
            row[_phase_column(phase_name, domain_name)] = value
    return row


def write_local_manifest(spec: SinglePhaseGrpNoL2RunSpec, output_dir: Path) -> None:
    """Write local audit artifacts for the validation run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records([_flat_manifest_row(spec)]).to_csv(output_dir / LOCAL_MANIFEST_CSV, index=False)
    (output_dir / LOCAL_RUN_SPEC_JSON).write_text(
        json.dumps(
            {
                "experiment_name": NAME,
                "single_phase_strategy": SINGLE_PHASE_STRATEGY,
                "run": asdict(spec),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    summary = {
        "experiment_name": NAME,
        "source_variant": SOURCE_VARIANT,
        "source_run_name": spec.source_run_name,
        "run_name": spec.run_name,
        "single_phase_strategy": SINGLE_PHASE_STRATEGY,
        "phase_fractions": PHASE_FRACTIONS,
        "phase_tv": spec.phase_tv,
        "model_family": MODEL_FAMILY,
        "experiment_budget": spec.experiment_budget,
        "realized_experiment_budget": spec.realized_experiment_budget,
        "target_budget": spec.target_budget,
        "target_budget_multiplier": spec.target_budget_multiplier,
        "num_train_steps": spec.num_train_steps,
        "target_final_checkpoint_step": spec.target_final_checkpoint_step,
    }
    (output_dir / LOCAL_SUMMARY_JSON).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def save_run_manifest(config: SaveSinglePhaseGrpNoL2ManifestConfig) -> None:
    """Persist the validation manifest for downstream collection and fitting."""
    run_spec = json.loads(config.run_spec_json)
    payload = {
        "experiment_name": config.experiment_name,
        "n_runs": 1,
        "single_phase_strategy": SINGLE_PHASE_STRATEGY,
        "phase_fractions": PHASE_FRACTIONS,
        "runs": [run_spec],
    }
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fsspec.open(os.path.join(config.output_path, RUN_MANIFEST_FILE), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def create_run_manifest_step(*, name_prefix: str, run_spec: SinglePhaseGrpNoL2RunSpec) -> ExecutorStep:
    """Create the manifest writer step for the validation run."""
    return ExecutorStep(
        name=f"{name_prefix}/run_manifest",
        description="Save GRP no-L2 single-phase exposure-average validation manifest",
        fn=save_run_manifest,
        config=SaveSinglePhaseGrpNoL2ManifestConfig(
            output_path=this_output_path(),
            experiment_name=name_prefix,
            run_spec_json=json.dumps(asdict(run_spec), sort_keys=True),
        ),
    )


def _configure_training_env_for_step(
    training_step: ExecutorStep,
    *,
    tpu_region: str,
    include_eval_harness: bool,
    child_job_name: str,
) -> ExecutorStep:
    config = training_step.config
    if not isinstance(config, TrainLmOnPodConfig):
        raise TypeError(f"Expected TrainLmOnPodConfig for {training_step.name!r}, got {type(config)!r}")
    env_vars = dict(config.env_vars or {})
    env_vars["MARIN_PREFIX"] = marin_prefix_for_region(tpu_region)
    if not include_eval_harness:
        env_vars[SKIP_EVAL_HARNESS_ENV_VAR] = "1"
    return replace(training_step, config=replace(config, env_vars=env_vars, job_name=child_job_name))


def build_launch_artifacts(
    *,
    name_prefix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    include_eval_harness: bool,
) -> SinglePhaseGrpNoL2LaunchArtifacts:
    """Resolve the validation launch graph without submitting it."""
    if tpu_region != DEFAULT_TPU_REGION or tpu_zone != DEFAULT_TPU_ZONE:
        raise ValueError(
            f"This launcher is intentionally pinned to {DEFAULT_TPU_REGION}/{DEFAULT_TPU_ZONE}; "
            f"got {tpu_region}/{tpu_zone}"
        )
    run_spec = build_run_spec()
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=name_prefix,
        experiment_budget=EXPERIMENT_BUDGET,
        target_budget=TARGET_BUDGET,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        model_config=regmix_60m_proxy,
        resources=ResourceConfig.with_tpu(tpu_type, regions=[tpu_region], zone=tpu_zone),
        runtime_cache_region=tpu_region,
    )
    run_manifest_step = create_run_manifest_step(name_prefix=name_prefix, run_spec=run_spec)
    training_step = experiment.create_training_step(
        weight_config=WeightConfig(run_id=run_spec.run_id, phase_weights=run_spec.phase_weights),
        name_prefix=name_prefix,
        run_name=run_spec.run_name,
        data_seed=run_spec.data_seed,
        simulated_epoch_subset_seed=run_spec.simulated_epoch_subset_seed,
    )
    training_step = _configure_training_env_for_step(
        training_step,
        tpu_region=tpu_region,
        include_eval_harness=include_eval_harness,
        child_job_name=f"train_lm_{run_spec.run_name}",
    )
    results_step = create_manifest_results_step(
        name_prefix=name_prefix,
        run_manifest_step=run_manifest_step,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        depends_on=[training_step],
    )
    fit_dataset_step = create_fit_dataset_export_step(
        name_prefix=name_prefix,
        run_manifest_step=run_manifest_step,
        analysis_step=results_step,
    )
    return SinglePhaseGrpNoL2LaunchArtifacts(
        run_spec=run_spec,
        run_manifest_step=run_manifest_step,
        training_step=training_step,
        results_step=results_step,
        fit_dataset_step=fit_dataset_step,
    )


def _has_iris_context() -> bool:
    try:
        from iris.client.client import get_iris_ctx
    except ImportError:
        return False
    return get_iris_ctx() is not None


def _executor_prefix(executor_prefix: str | None, default_tpu_region: str) -> str | None:
    if executor_prefix is None:
        return None
    if executor_prefix.startswith("gs://"):
        return executor_prefix
    if executor_prefix.startswith("/"):
        raise ValueError(f"Executor prefix must be a GCS path or relative key, got {executor_prefix!r}")
    return os.path.join(marin_prefix_for_region(default_tpu_region), executor_prefix)


def _validate_training_graph(artifacts: SinglePhaseGrpNoL2LaunchArtifacts, *, include_eval_harness: bool) -> None:
    config = artifacts.training_step.config
    if not isinstance(config, TrainLmOnPodConfig):
        raise TypeError(f"Expected TrainLmOnPodConfig for {artifacts.training_step.name!r}, got {type(config)!r}")
    actual_num_train_steps = int(config.train_config.trainer.num_train_steps)
    if actual_num_train_steps != NUM_TRAIN_STEPS:
        raise ValueError(f"{artifacts.training_step.name} has num_train_steps={actual_num_train_steps}")
    env_vars = dict(config.env_vars or {})
    has_skip = env_vars.get(SKIP_EVAL_HARNESS_ENV_VAR) == "1"
    if env_vars.get("MARIN_PREFIX") != marin_prefix_for_region(DEFAULT_TPU_REGION):
        raise ValueError(f"{artifacts.training_step.name} has invalid MARIN_PREFIX={env_vars.get('MARIN_PREFIX')!r}")
    if config.job_name == "train_lm":
        raise ValueError(f"{artifacts.training_step.name} has non-unique child job name {config.job_name!r}")
    if include_eval_harness and has_skip:
        raise ValueError(f"{artifacts.training_step.name} unexpectedly skips eval harness")
    if not include_eval_harness and not has_skip:
        raise ValueError(f"{artifacts.training_step.name} is missing {SKIP_EVAL_HARNESS_ENV_VAR}=1")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--executor-prefix")
    parser.add_argument("--local-artifact-dir", default=str(DEFAULT_LOCAL_ARTIFACT_DIR))
    parser.add_argument(
        "--include-eval-harness",
        action="store_true",
        help="Run Levanter lm-eval harness during training. Default is perplexity/checkpoint only.",
    )
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    os.environ.setdefault("MARIN_PREFIX", marin_prefix_for_region(args.tpu_region))
    if not args.dry_run and not args.allow_local and os.getenv("CI") is None and not _has_iris_context():
        raise ValueError("Non-dry-run launches must run inside Iris, e.g. via 'uv run iris job run'.")

    artifacts = build_launch_artifacts(
        name_prefix=args.name_prefix,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        include_eval_harness=args.include_eval_harness,
    )
    _validate_training_graph(artifacts, include_eval_harness=args.include_eval_harness)
    write_local_manifest(artifacts.run_spec, Path(args.local_artifact_dir))
    logger.info("Wrote local manifest to %s/%s", args.local_artifact_dir, LOCAL_MANIFEST_CSV)
    logger.info("Prepared GRP no-L2 single-phase validation run: %s", artifacts.run_spec.run_name)
    logger.info(
        "Launch config: tpu=%s region=%s zone=%s max_concurrent=%d eval_harness=%s",
        args.tpu_type,
        args.tpu_region,
        args.tpu_zone,
        args.max_concurrent,
        "enabled" if args.include_eval_harness else "skipped",
    )
    if args.dry_run or os.getenv("CI") is not None:
        return

    executor_prefix = _executor_prefix(args.executor_prefix, args.tpu_region)
    executor_main(
        ExecutorMainConfig(prefix=executor_prefix, max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=(
            f"{args.name_prefix}: GRP no-L2 60M/1.2B single-phase exposure-average validation. "
            f"Outputs include {RUN_MANIFEST_FILE}, {RESULTS_CSV}, {FIT_DATASET_CSV}, and "
            f"{FIT_DATASET_SUMMARY_JSON}."
        ),
    )


if __name__ == "__main__":
    main()
