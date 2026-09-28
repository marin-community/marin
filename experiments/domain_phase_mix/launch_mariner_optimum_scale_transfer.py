# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Train the four methods' Uncheatable optima at the Llama 160M/1.2B and 200M/6B proxy settings.

The paper's scale-transfer figure compares the same 280 single-phase designs across Llama 160M/1.2B, Qwen3 360M/1.6B
and Llama 200M/6B, but no mixture optimized at one setting had been trained at the others, so the figure carries
no deployed markers. This launcher trains the Uncheatable proposals of MARINER (``lwspu_u_snc_cap06``, the mixture
of the Qwen3 3e18 validation and ladder runs), matched Olmix (``olmixq_u_kl0p05_cap04``, the ladder's policy),
tuned RegMix (``cmp_u_lgbm_cap08``) and released RegMix (``rgref_u_endpoint_cap64``), the four trained proposals
of the paper's Table 5, at both Llama settings with the same simulated epoching as the swarms (subset = pool x D_r /
6.33T), one run per mixture and setting, through the qsplit240 replay experiment that produced the
proportional-perturbation panels. The runtime mixture CSVs live under the exploratory directory, so submissions
must add them with ``--working-dir-include``.

usage:
  uv run python -m experiments.domain_phase_mix.launch_mariner_optimum_scale_transfer --dry-run
  (submission: see .agents/projects/mariner_optimum_scale_transfer_20260924/launch.sh)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pandas as pd
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main
from rigging.filesystem import marin_prefix_for_region

import experiments.domain_phase_mix.launch_proportional_perturbation_scale_transfer as transfer
from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.determinism_analysis import (
    create_fit_dataset_export_step,
    create_manifest_results_step,
)
from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b import QSPLIT240_300M_EVAL_TASKS
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_study import WANDB_ENTITY, WANDB_PROJECT
from experiments.domain_phase_mix.qsplit240_replay import (
    SKIP_EVAL_HARNESS_ENV_VAR,
    add_eval_cache_dependency_to_training_step,
    create_cache_eval_datasets_step,
    create_qsplit240_replay_experiment,
    normalize_tpu_regions,
    resolve_qsplit240_eval_cache_path_for_regions,
    skip_eval_harness_for_training_step,
)
from experiments.domain_phase_mix.scaling_study_recipes import ScalingStudyScale, resolve_scale_spec

logger = logging.getLogger(__name__)

# Short: W&B truncates run names beyond about 60 characters by dropping the middle, which can collapse two
# scales or two mixtures onto one checkpoint path; with this prefix no name needs truncation.
BASE_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/mopt"
COHORT = "mariner_optimum_scale_transfer"
RUN_ID_BASE = 795_000
REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_OUTPUTS = REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs"
# (runtime table under reference_outputs, candidate id, run name, intervention id)
MIXTURES = (
    (
        "delphi_corrected_screen_20260908/materialized_flat15_nocap/runtime_materialization/candidate_weights.csv",
        "lwspu_u_snc_cap06",
        "mariner_u",
        "mariner_uncheatable_optimum",
    ),
    (
        "delphi_matched_olmix_3e18_20260908/candidate_weights.csv",
        "olmixq_u_kl0p05_cap04",
        "olmix_u",
        "olmix_uncheatable_kl0p05_cap4",
    ),
    (
        "delphi_comparator_proposals_3e18_20260909/candidate_weights.csv",
        "cmp_u_lgbm_cap08",
        "regmix_tun_u",
        "regmix_tuned_uncheatable_proposal",
    ),
    (
        "regmix_official_rerun_20260913/candidate_weights.csv",
        "rgref_u_endpoint_cap64",
        "regmix_rel_u",
        "regmix_released_uncheatable_proposal",
    ),
)
SCALES = transfer.SCALES
DEFAULT_LOCAL_ARTIFACT_DIR = (
    REPO_ROOT
    / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs"
    / "mariner_optimum_scale_transfer_20260924"
)


def runtime_weights(table_path: str, candidate_id: str) -> dict[str, float]:
    """The runtime (1/2048-grid) bucket weights of one trained proposal."""
    path = REFERENCE_OUTPUTS / table_path
    table = pd.read_csv(path)
    rows = table[table["candidate_id"].eq(candidate_id)]
    if rows.empty:
        raise ValueError(f"{candidate_id}: not in {path}")
    if len(rows) != 39:
        raise ValueError(f"{candidate_id}: {len(rows)} rows in {path}, expected 39")
    weights = {str(domain): float(weight) for domain, weight in zip(rows["domain"], rows["weight"], strict=True)}
    transfer._validate_domain_weights(weights, label=candidate_id)
    return weights


def build_interventions() -> list[transfer.InterventionSpec]:
    base = transfer._base_proportional_weights()
    specs = []
    for index, (table_path, candidate_id, run_name, intervention_id) in enumerate(MIXTURES):
        weights = runtime_weights(table_path, candidate_id)
        specs.append(
            transfer.InterventionSpec(
                intervention_index=index,
                intervention_id=intervention_id,
                run_id=RUN_ID_BASE + index,
                run_name=run_name,
                intervention_type="trained_proposal",
                target_unit="mixture",
                target_domain=None,
                target_family=None,
                quality_high_domain=None,
                quality_low_domain=None,
                bump_epsilon=None,
                quality_swap_fraction=None,
                quality_swap_mass=None,
                renormalizer="none",
                donor_pool="none",
                phase_mode="both_phases",
                base_run_name=transfer.BASE_RUN_NAME,
                base_run_id=transfer.BASE_RUN_ID,
                base_source_experiment=transfer.ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT,
                tv_distance=transfer._tv_distance(base, weights),
                target_mass_before=0.0,
                target_mass_after=0.0,
                donor_mass_before=0.0,
                donor_mass_after=0.0,
                phase_weights=transfer._constant_phase_weights(weights),
            )
        )
    return specs


def build_run_specs(interventions: list[transfer.InterventionSpec], scale: ScalingStudyScale) -> list:
    return [
        replace(
            transfer._run_spec_for_scale(spec, scale),
            cohort=COHORT,
            candidate_source_experiment=BASE_NAME_PREFIX,
        )
        for spec in interventions
    ]


def configure_training_step(
    training_step, *, tpu_region: str, include_eval_harness: bool, checkpoint_minutes: int | None
):
    """Pin the child's data prefix to the region; skip the lm-eval harness unless asked (Uncheatable is in-training).

    ``checkpoint_minutes`` shortens the temporary-checkpoint interval (default 10 min) so an attempt on a churning
    preemptible pool banks progress; it does not enter the output path, so a resubmission resumes the same run.
    """
    config = training_step.config
    if checkpoint_minutes is not None:
        trainer = config.train_config.trainer
        checkpointer = replace(trainer.checkpointer, save_interval=timedelta(minutes=checkpoint_minutes))
        config = replace(
            config,
            train_config=replace(config.train_config, trainer=replace(trainer, checkpointer=checkpointer)),
        )
    env_vars = dict(config.env_vars or {})
    env_vars["MARIN_PREFIX"] = marin_prefix_for_region(tpu_region)
    if not include_eval_harness:
        env_vars[SKIP_EVAL_HARNESS_ENV_VAR] = "1"
    return replace(training_step, config=replace(config, env_vars=env_vars))


def build_scale_artifacts(
    *,
    interventions: list[transfer.InterventionSpec],
    scale: ScalingStudyScale,
    tpu_type: str,
    tpu_regions: tuple[str, ...],
    tpu_zone: str,
    eval_datasets_cache_path: str,
    include_eval_harness: bool,
    checkpoint_minutes: int | None = None,
) -> transfer.ScaleLaunchArtifacts:
    """The perturbation launcher's per-scale graph, for these run specs."""
    scale_spec = resolve_scale_spec(scale)
    name_prefix = transfer._scale_name_prefix(BASE_NAME_PREFIX, scale)
    run_specs = build_run_specs(interventions, scale)
    experiment = create_qsplit240_replay_experiment(
        name=name_prefix,
        experiment_budget=run_specs[0].experiment_budget,
        target_budget=run_specs[0].target_budget,
        batch_size=scale_spec.batch_size,
        seq_len=scale_spec.seq_len,
        model_config=scale_spec.model_config,
        optimizer_config=scale_spec.optimizer_config,
        tpu_type=tpu_type,
        tpu_regions=tpu_regions,
        tpu_zone=tpu_zone,
        eval_tasks=QSPLIT240_300M_EVAL_TASKS,
        eval_datasets_cache_path=eval_datasets_cache_path,
    )
    resolved_eval_cache_path = resolve_qsplit240_eval_cache_path_for_regions(tpu_regions, eval_datasets_cache_path)
    run_manifest_step = transfer._run_manifest_step(
        execution_name_prefix=name_prefix, experiment_name=name_prefix, run_specs=run_specs
    )
    cache_eval_datasets_step = create_cache_eval_datasets_step(
        eval_tasks=QSPLIT240_300M_EVAL_TASKS, gcs_path=resolved_eval_cache_path, name_prefix=name_prefix
    )
    training_steps = []
    for run_spec in run_specs:
        training_step = experiment.create_training_step(
            weight_config=WeightConfig(run_id=run_spec.run_id, phase_weights=run_spec.phase_weights),
            name_prefix=name_prefix,
            run_name=run_spec.run_name,
            data_seed=run_spec.data_seed,
            simulated_epoch_subset_seed=run_spec.simulated_epoch_subset_seed,
        )
        training_step = add_eval_cache_dependency_to_training_step(training_step, cache_eval_datasets_step)
        training_step = configure_training_step(
            training_step,
            tpu_region=tpu_regions[0],
            include_eval_harness=include_eval_harness,
            checkpoint_minutes=checkpoint_minutes,
        )
        if not include_eval_harness:
            training_step = skip_eval_harness_for_training_step(training_step)
        training_steps.append(training_step)
    results_step = create_manifest_results_step(
        name_prefix=name_prefix,
        run_manifest_step=run_manifest_step,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        depends_on=training_steps,
    )
    fit_dataset_step = create_fit_dataset_export_step(
        name_prefix=name_prefix, run_manifest_step=run_manifest_step, analysis_step=results_step
    )
    return transfer.ScaleLaunchArtifacts(
        scale=scale,
        name_prefix=name_prefix,
        run_specs=run_specs,
        run_manifest_step=run_manifest_step,
        cache_eval_datasets_step=cache_eval_datasets_step,
        training_steps=training_steps,
        results_step=results_step,
        fit_dataset_step=fit_dataset_step,
    )


def validate(artifacts: transfer.PairedLaunchArtifacts) -> None:
    output_roots = [str(step.override_output_path or step.name) for step in artifacts.training_steps]
    if len(set(output_roots)) != len(output_roots):
        raise ValueError("Duplicate training output paths")
    for artifact in artifacts.scale_artifacts:
        expected = resolve_scale_spec(artifact.scale).num_train_steps_for_multiplier(transfer.TARGET_BUDGET_MULTIPLIER)
        for run_spec, step in zip(artifact.run_specs, artifact.training_steps, strict=True):
            actual = int(step.config.train_config.trainer.num_train_steps)
            if run_spec.num_train_steps != expected or actual != expected:
                raise ValueError(
                    f"{step.name}: num_train_steps {actual} / spec {run_spec.num_train_steps} != {expected}"
                )
            if step.config.env_vars.get("MARIN_PREFIX") != marin_prefix_for_region(transfer.DEFAULT_TPU_REGION):
                raise ValueError(f"{step.name}: MARIN_PREFIX is not {transfer.DEFAULT_TPU_REGION}'s")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--tpu-type", default=transfer.DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=transfer.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=transfer.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--executor-prefix")
    parser.add_argument("--eval-datasets-cache-path", default=transfer.DEFAULT_EVAL_DATASETS_CACHE_PATH)
    parser.add_argument("--local-artifact-dir", default=str(DEFAULT_LOCAL_ARTIFACT_DIR))
    parser.add_argument("--include-eval-harness", action="store_true")
    parser.add_argument("--checkpoint-minutes", type=int, default=None, help="temporary-checkpoint interval override")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    tpu_regions = normalize_tpu_regions(args.tpu_region)
    if tpu_regions != (transfer.DEFAULT_TPU_REGION,) or args.tpu_zone != transfer.DEFAULT_TPU_ZONE:
        raise ValueError(f"The 39-bucket pools live in {transfer.DEFAULT_TPU_REGION}; got {tpu_regions}/{args.tpu_zone}")
    os.environ.setdefault("MARIN_PREFIX", marin_prefix_for_region(transfer.DEFAULT_TPU_REGION))
    if not args.dry_run and not args.allow_local and os.getenv("CI") is None and not transfer._has_iris_context():
        raise ValueError("Non-dry-run launches must run inside Iris, e.g. via 'uv run iris --cluster=marin job run'.")

    # Steps are built inside the executor context: under MARIN_EXECUTOR_STRICT a step built outside it raises.
    with executor_context():
        interventions = build_interventions()
        artifacts = transfer.PairedLaunchArtifacts(
            interventions=interventions,
            scale_artifacts=[
                build_scale_artifacts(
                    interventions=interventions,
                    scale=scale,
                    tpu_type=args.tpu_type,
                    tpu_regions=tpu_regions,
                    tpu_zone=args.tpu_zone,
                    eval_datasets_cache_path=args.eval_datasets_cache_path,
                    include_eval_harness=args.include_eval_harness,
                    checkpoint_minutes=args.checkpoint_minutes,
                )
                for scale in SCALES
            ],
        )
        validate(artifacts)
    transfer.write_local_manifests(artifacts, Path(args.local_artifact_dir))
    for artifact in artifacts.scale_artifacts:
        for run_spec in artifact.run_specs:
            logger.info(
                "%s: %s run %d, %d steps, data seed %d, TV %.3f from proportional",
                artifact.scale.value,
                run_spec.run_name,
                run_spec.run_id,
                run_spec.num_train_steps,
                run_spec.data_seed,
                run_spec.tv_distance,
            )
    if args.dry_run or os.getenv("CI") is not None:
        return
    executor_main(
        ExecutorMainConfig(
            prefix=transfer._executor_prefix(args.executor_prefix, transfer.DEFAULT_TPU_REGION),
            max_concurrent=args.max_concurrent,
        ),
        steps=artifacts.steps,
        description=f"{BASE_NAME_PREFIX}: four Uncheatable optima at 60M/1.2B and 100M/6B (Figure 28 markers).",
    )


if __name__ == "__main__":
    main()
