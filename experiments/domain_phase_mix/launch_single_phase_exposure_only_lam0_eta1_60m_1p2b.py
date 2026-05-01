# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the one-phase GRP no-L2 `exposure_only_lam0_eta1` raw optimum validation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import sys

import fsspec
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main, this_output_path
from marin.rl.placement import marin_prefix_for_region
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
from experiments.domain_phase_mix.launch_single_phase_average_grp_no_l2_60m_1p2b import (
    DEFAULT_LOCAL_ARTIFACT_DIR as PREVIOUS_LOCAL_ARTIFACT_DIR,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_TPU_REGION,
    DEFAULT_TPU_TYPE,
    DEFAULT_TPU_ZONE,
    MODEL_FAMILY,
    NUM_TRAIN_STEPS,
    OBJECTIVE_METRIC,
    REALIZED_EXPERIMENT_BUDGET,
    SEQ_LEN,
    TARGET_BUDGET,
    TARGET_BUDGET_MULTIPLIER,
    _configure_training_env_for_step,
    _executor_prefix,
    _has_iris_context,
    _validate_training_graph,
    validate_run_spec,
    SinglePhaseGrpNoL2LaunchArtifacts,
    SinglePhaseGrpNoL2RunSpec,
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
    create_two_phase_dolma3_dolmino_top_level_experiment,
)

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/ngd3dm2_single_phase_exposure_only_lam0_eta1_60m_1p2b"
RUN_ID = 910074
RUN_NAME = "singleavg_exposure_only_lam0_eta1_raw_optimum"
COHORT = "single_phase_grp_no_l2_ablation_validation"
SOURCE_VARIANT = "exposure_only_lam0_eta1"
SINGLE_PHASE_STRATEGY = "raw_optimum_exposure_only_lam0_eta1"
SOURCE_EXPERIMENT = "local_single_phase_grp_no_l2_ablation_fit"
WEIGHTS_CSV = (
    PREVIOUS_LOCAL_ARTIFACT_DIR.parent
    / "single_phase_exposure_average_60m_1p2b"
    / "analysis"
    / "grp_no_l2_one_phase_ablations"
    / SOURCE_VARIANT
    / "raw_single_phase_optimum_weights.csv"
)
DEFAULT_LOCAL_ARTIFACT_DIR = PREVIOUS_LOCAL_ARTIFACT_DIR.parent / "single_phase_exposure_only_lam0_eta1_60m_1p2b"
LOCAL_MANIFEST_CSV = "single_phase_grp_no_l2_manifest.csv"
LOCAL_SUMMARY_JSON = "single_phase_grp_no_l2_summary.json"
LOCAL_RUN_SPEC_JSON = "single_phase_grp_no_l2_run_spec.json"
EMBEDDED_WEIGHTS_BY_DOMAIN = {
    "dolma3_cc/art_and_design_high": 4.117270693909919e-06,
    "dolma3_cc/art_and_design_low": 8.9741667454898853e-07,
    "dolma3_cc/crime_and_law_high": 0.00019950267892160001,
    "dolma3_cc/crime_and_law_low": 1.7344979906259669e-06,
    "dolma3_cc/education_and_jobs_high": 1.1482331043604431e-06,
    "dolma3_cc/education_and_jobs_low": 9.049824133390322e-07,
    "dolma3_cc/electronics_and_hardware_high": 1.3181938815756261e-06,
    "dolma3_cc/electronics_and_hardware_low": 6.7311158177489045e-07,
    "dolma3_cc/entertainment_high": 2.6985565743488532e-06,
    "dolma3_cc/entertainment_low": 1.2024916257780888e-06,
    "dolma3_cc/finance_and_business_high": 3.5761991664725092e-06,
    "dolma3_cc/finance_and_business_low": 9.8443920845022891e-07,
    "dolma3_cc/food_and_dining_high": 9.2281312999456317e-07,
    "dolma3_cc/food_and_dining_low": 5.1685308168549914e-07,
    "dolma3_cc/games_high": 6.3802660619638885e-07,
    "dolma3_cc/games_low": 6.4345807088639405e-07,
    "dolma3_cc/health_high": 5.5301113870786268e-07,
    "dolma3_cc/health_low": 7.256264446361681e-07,
    "dolma3_cc/history_and_geography_high": 0.0018962698202468001,
    "dolma3_cc/history_and_geography_low": 2.3996045079172789e-06,
    "dolma3_cc/industrial_high": 0.012368855736323601,
    "dolma3_cc/industrial_low": 7.7486166553294767e-07,
    "dolma3_cc/literature_high": 0.039818480296280302,
    "dolma3_cc/literature_low": 1.8791599415321184e-06,
    "dolma3_cc/science_math_and_technology_high": 0.25223053745781271,
    "dolma3_cc/science_math_and_technology_low": 7.227265894235012e-07,
    "dolma3_stack_edu": 0.13879106406780631,
    "dolma3_arxiv": 0.023001875203239999,
    "dolma3_finemath_3plus": 1.2088558590893504e-05,
    "dolma3_wikipedia": 0.0078353599549677005,
    "dolmino_common_crawl_hq": 0.1399034953204083,
    "dolmino_olmocr_pdfs_hq": 0.18250442391227961,
    "dolmino_stack_edu_fim": 0.1210386382289594,
    "dolmino_stem_heavy_crawl": 0.0094287122438177998,
    "dolmino_synth_code": 0.0228952780848856,
    "dolmino_synth_instruction": 0.0125953012220515,
    "dolmino_synth_math": 0.016889846373055298,
    "dolmino_synth_qa": 0.0044035104889299003,
    "dolmino_synth_thinking": 0.0141577288173302,
}


def _load_single_phase_weights(weights_csv: Path) -> np.ndarray:
    if not weights_csv.exists():
        weights = np.asarray([EMBEDDED_WEIGHTS_BY_DOMAIN[domain_name] for domain_name in DOMAIN_NAMES], dtype=float)
        return weights / float(weights.sum())

    frame = pd.read_csv(weights_csv)
    required_columns = {"domain_name", "single_phase_weight"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{weights_csv} missing required columns: {missing_columns}")
    if frame["domain_name"].duplicated().any():
        duplicates = sorted(frame.loc[frame["domain_name"].duplicated(), "domain_name"].astype(str).unique())
        raise ValueError(f"{weights_csv} has duplicate domains: {duplicates}")

    weights_by_domain = dict(zip(frame["domain_name"], frame["single_phase_weight"], strict=True))
    missing_domains = sorted(set(DOMAIN_NAMES) - set(weights_by_domain))
    extra_domains = sorted(set(weights_by_domain) - set(DOMAIN_NAMES))
    if missing_domains or extra_domains:
        raise ValueError(f"{weights_csv} domain mismatch: missing={missing_domains[:5]} extra={extra_domains[:5]}")

    weights = np.asarray([float(weights_by_domain[domain_name]) for domain_name in DOMAIN_NAMES], dtype=float)
    if np.any(weights < 0.0):
        raise ValueError(f"{weights_csv} contains negative weights")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError(f"{weights_csv} has non-positive weight sum {total}")
    return weights / total


def _phase_weights_from_vector(weights: np.ndarray) -> dict[str, dict[str, float]]:
    domain_weights = {domain_name: float(weight) for domain_name, weight in zip(DOMAIN_NAMES, weights, strict=True)}
    return {"phase_0": dict(domain_weights), "phase_1": dict(domain_weights)}


def _phase_column(phase_name: str, domain_name: str) -> str:
    return f"{phase_name}_{domain_name}"


def build_run_spec(weights_csv: Path = WEIGHTS_CSV) -> SinglePhaseGrpNoL2RunSpec:
    """Build the single-row validation spec for the ablated one-phase raw optimum."""
    weights = _load_single_phase_weights(weights_csv)
    spec = SinglePhaseGrpNoL2RunSpec(
        run_id=RUN_ID,
        run_name=RUN_NAME,
        cohort=COHORT,
        model_family=MODEL_FAMILY,
        trainer_seed=None,
        data_seed=0,
        simulated_epoch_subset_seed=None,
        source_run_id=RUN_ID,
        source_run_name=RUN_NAME,
        source_two_phase_experiment=SOURCE_EXPERIMENT,
        candidate_run_id=RUN_ID,
        candidate_run_name=RUN_NAME,
        candidate_source_experiment=SOURCE_EXPERIMENT,
        source_variant=SOURCE_VARIANT,
        single_phase_strategy=SINGLE_PHASE_STRATEGY,
        phase_tv=0.0,
        experiment_budget=EXPERIMENT_BUDGET,
        realized_experiment_budget=REALIZED_EXPERIMENT_BUDGET,
        target_budget=TARGET_BUDGET,
        target_budget_multiplier=TARGET_BUDGET_MULTIPLIER,
        num_train_steps=NUM_TRAIN_STEPS,
        target_final_checkpoint_step=NUM_TRAIN_STEPS - 1,
        phase_weights=_phase_weights_from_vector(weights),
    )
    validate_run_spec(spec)
    return spec


def _flat_manifest_row(spec: SinglePhaseGrpNoL2RunSpec) -> dict[str, object]:
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


def save_run_manifest(config: dict[str, str]) -> None:
    """Persist the validation manifest for downstream collection and fitting."""
    run_spec = json.loads(config["run_spec_json"])
    payload = {
        "experiment_name": config["experiment_name"],
        "n_runs": 1,
        "single_phase_strategy": SINGLE_PHASE_STRATEGY,
        "runs": [run_spec],
    }
    fs, _, _ = fsspec.get_fs_token_paths(config["output_path"])
    fs.makedirs(config["output_path"], exist_ok=True)
    with fsspec.open(os.path.join(config["output_path"], RUN_MANIFEST_FILE), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def create_run_manifest_step(*, name_prefix: str, run_spec: SinglePhaseGrpNoL2RunSpec) -> ExecutorStep:
    """Create the manifest writer step for the validation run."""
    return ExecutorStep(
        name=f"{name_prefix}/run_manifest",
        description="Save exposure_only_lam0_eta1 single-phase validation manifest",
        fn=save_run_manifest,
        config={
            "output_path": this_output_path(),
            "experiment_name": name_prefix,
            "run_spec_json": json.dumps(asdict(run_spec), sort_keys=True),
        },
    )


def build_launch_artifacts(
    *,
    name_prefix: str,
    weights_csv: Path,
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
    run_spec = build_run_spec(weights_csv)
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


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--weights-csv", default=str(WEIGHTS_CSV))
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
        weights_csv=Path(args.weights_csv),
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        include_eval_harness=args.include_eval_harness,
    )
    _validate_training_graph(artifacts, include_eval_harness=args.include_eval_harness)
    write_local_manifest(artifacts.run_spec, Path(args.local_artifact_dir))
    logger.info("Wrote local manifest to %s", args.local_artifact_dir)
    logger.info("Prepared %s validation run: %s", SOURCE_VARIANT, artifacts.run_spec.run_name)
    logger.info(
        "Launch config: tpu=%s region=%s zone=%s max_concurrent=%d eval_harness=%s",
        args.tpu_type,
        args.tpu_region,
        args.tpu_zone,
        args.max_concurrent,
        "enabled" if args.include_eval_harness else "skipped",
    )
    if args.dry_run or os.getenv("CI") is not None:
        print(json.dumps({"run": asdict(artifacts.run_spec), "strategy": SINGLE_PHASE_STRATEGY}, indent=2))
        return

    executor_prefix = _executor_prefix(args.executor_prefix, args.tpu_region)
    executor_main(
        ExecutorMainConfig(prefix=executor_prefix, max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=(
            f"{args.name_prefix}: {SOURCE_VARIANT} 60M/1.2B single-phase raw-optimum validation. "
            f"Outputs include {RUN_MANIFEST_FILE}, {RESULTS_CSV}, {FIT_DATASET_CSV}, and "
            f"{FIT_DATASET_SUMMARY_JSON}. Objective metric: {OBJECTIVE_METRIC}."
        ),
    )


if __name__ == "__main__":
    main()
