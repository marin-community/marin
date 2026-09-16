# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scale the matched Olmix primary policies at the frozen procedure's rungs and seeds.

The Olmix policies fitted on the frozen Qwen3 3e18 swarm and validated there (`launch_delphi_matched_olmix_3e18.py`:
KL 0.05 for Uncheatable, KL 0.005 for the suite, epoch cap four) are trained at 2e19, 3e20 and 1e21 on the
ladder's model sizes, token budgets, batches and data seeds, the same as MARINER's scaling runs
(`launch_delphi_frozen_procedure_scaling.py`), so every rung pairs with MARINER's run at the same seed. The 3e18
measurements are not rerun. Training children carry the failure budget that lets a multi-host run survive a
sibling preemption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, replace
from typing import cast

import fsspec
from fray.cluster import ResourceConfig
from levanter.data.text.datasets import DatasetComponent
from marin.evaluation.olmo_base_eval.run import olmo_base_eval_step
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from rigging.filesystem import marin_prefix_for_region

from experiments.defaults import default_validation_sets
from experiments.domain_phase_mix import launch_delphi_frozen_procedure_scaling as frozen
from experiments.domain_phase_mix import launch_delphi_matched_olmix_3e18 as matched
from experiments.domain_phase_mix import launch_delphi_one_phase_wspu_scaling as ladder
from experiments.domain_phase_mix import launch_delphi_uncheatable_optimized_mixtures as base
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_matched_olmix_scaling_v6e_20260910"
LOCAL_ARTIFACT_DIR = matched.REFERENCE_OUTPUTS / "delphi_matched_olmix_scaling_v6e_20260910/launch_dry_run"
METHOD = "olmix_loglinear_d001_qwen3e18"
WANDB_SERIES_TAG = "delphi-matched-olmix-scaling"
RUN_ID_BASE = 7_450_000
TARGET_BUDGETS = frozen.TARGET_BUDGETS
POLICY_CANDIDATES = {"uncheatable": matched.UNCHEATABLE_KL0P05, "table9": matched.TABLE9_KL0P005}
KL_COEFFICIENTS = {"uncheatable": 0.05, "table9": 0.005}
EPOCH_CAP = 4
MAX_CONCURRENT = len(TARGET_BUDGETS) * len(POLICY_CANDIDATES)

FrozenPolicy = frozen.FrozenPolicy
ScalingRow = frozen.ScalingRow
ScalingTrainingConfig = frozen.ScalingTrainingConfig
SaveManifestConfig = frozen.SaveManifestConfig


def selected_policies() -> tuple[FrozenPolicy, ...]:
    """Read the SHA-pinned matched-Olmix table used for the eleven-run 3e18 validation."""
    candidates = {candidate.candidate_id: candidate for _, group in matched.selected_candidates() for candidate in group}
    policies = []
    for target, candidate_id in POLICY_CANDIDATES.items():
        candidate = candidates[candidate_id]
        if candidate.epoch_cap != EPOCH_CAP or candidate.max_materialized_epoch > EPOCH_CAP:
            raise ValueError(f"{candidate_id} is not a cap-{EPOCH_CAP} policy")
        weights_csv = ladder._weight_csv(candidate)
        policies.append(
            FrozenPolicy(
                candidate_id=candidate_id,
                target=target,
                target_metric=base.TABLE9_TARGET_METRIC if target == "table9" else ladder.UNCHEATABLE_TARGET_METRIC,
                max_materialized_epoch=candidate.max_materialized_epoch,
                weights_csv=weights_csv,
                weights_sha256=hashlib.sha256(weights_csv.encode()).hexdigest(),
            )
        )
    return tuple(policies)


def register_policy(policy: FrozenPolicy, data_seed: int) -> base.DelphiValidationMixture:
    """Install embedded weights in each process, including remote training workers."""
    if hashlib.sha256(policy.weights_csv.encode()).hexdigest() != policy.weights_sha256:
        raise ValueError(f"Embedded weights changed for {policy.candidate_id}")
    key = cast(base.DelphiValidationMixture, ladder.ScalingMixtureKey(policy.candidate_id))
    base.MIXTURE_SOURCES[key] = base.MixtureSource(
        key=key,
        display_name=f"Matched Olmix {policy.target}, KL {KL_COEFFICIENTS[policy.target]}, cap {EPOCH_CAP}",
        source_csv=f"embedded://{policy.weights_sha256}/{policy.candidate_id}.csv",
        github_issue=6611 if policy.target == "table9" else 6609,
        target_metric=policy.target_metric,
        method=METHOD,
        wandb_series_tag=WANDB_SERIES_TAG,
        expected_max_simulated_epoch=policy.max_materialized_epoch,
        data_seed_override=data_seed,
    )
    base._EMBEDDED_MIXTURE_WEIGHT_CSVS[key] = policy.weights_csv
    return key


def planned_rows(analysis_output_path: str) -> list[ScalingRow]:
    """Resolve the six rungs without refitting or reoptimizing a mixture."""
    fits = base._read_scaling_fits(analysis_output_path)
    policies = selected_policies()
    rows = []
    for budget, (tpu_type, batch_size) in TARGET_BUDGETS.items():
        if batch_size != ladder.scaling.TARGET_BUDGETS[budget][1]:
            raise ValueError(f"Batch size changed from the OLMix ladder at {budget}")
        for policy in policies:
            seed = ladder.OLMIX_COMPARATOR_SEEDS[policy.target][budget]
            key = register_policy(policy, seed)
            run = base._predict_run_spec(
                scaling_fits=fits,
                mixture=key,
                target_flops=budget,
                tpu_type=tpu_type,
                tpu_region=ladder.TRAIN_TPU_REGION,
                tpu_zone=ladder.TRAIN_TPU_ZONE,
                batch_size=batch_size,
                run_order=len(rows),
            )
            devices = ladder.V6E_DEVICE_COUNTS[tpu_type]
            if run.batch_size % devices or run.model_hidden_dim % devices:
                raise ValueError(f"Unmodified data-axis sharding does not fit {tpu_type}: {run}")
            rows.append(
                ScalingRow(
                    policy=policy,
                    run=replace(run, run_id=RUN_ID_BASE + len(rows), run_name=f"{run.run_name}_seed{seed}"),
                )
            )
    return rows


def run_scaling_training(config: ScalingTrainingConfig) -> None:
    """Use the existing resumable trainer with the embedded, validated policy."""
    key = register_policy(config.policy, config.training.data_seed)
    if key != config.training.mixture:
        raise ValueError("Training mixture does not match the embedded policy")
    base.EXPERIMENT_NAME = EXPERIMENT_NAME
    base.run_delphi_optimized_training(config.training)


def save_manifest(config: SaveManifestConfig) -> None:
    """Persist weights, source hashes and resolved run configurations."""
    rows = json.loads(config.rows_json)
    manifest = {
        "experiment_name": EXPERIMENT_NAME,
        "analysis_output_path": config.analysis_output_path,
        "candidate_weights_path": str(matched.DEFAULT_CANDIDATE_WEIGHTS),
        "candidate_weights_sha256": matched.EXPECTED_CANDIDATE_WEIGHTS_SHA256,
        "method": METHOD,
        "optimization_epoch_cap": EPOCH_CAP,
        "kl_coefficients": KL_COEFFICIENTS,
        "max_concurrent": MAX_CONCURRENT,
        "olmix_comparator_seeds": ladder.OLMIX_COMPARATOR_SEEDS,
        "paired_with": frozen.EXPERIMENT_NAME,
        "accelerator_generation_matched_to_olmix": False,
        "inline_uncheatable_and_native_table9_for_every_run": True,
        "source_validation": "delphi_matched_olmix_3e18_20260908/measured_results.csv",
        "runs": rows,
    }
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(os.path.join(config.output_path, "run_manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def build_launch_artifacts(
    rows: list[ScalingRow],
    analysis_output_path: str,
    validation_configs: dict[str, DatasetComponent],
    *,
    max_retries_failure: int = 0,
    max_task_failures: int = 0,
) -> ladder.LaunchArtifacts:
    """Pair each independent training with its final-checkpoint native Table-9 eval."""
    manifest_step = ExecutorStep(
        name=f"{EXPERIMENT_NAME}/manifest",
        fn=save_manifest,
        config=SaveManifestConfig(this_output_path(), analysis_output_path, json.dumps([asdict(row) for row in rows])),
    )
    training_steps = []
    eval_steps = []
    for row in rows:
        run = row.run
        resources = ResourceConfig.with_tpu(run.tpu_type, regions=[run.tpu_region], zone=run.tpu_zone)
        training_step = ExecutorStep(
            name=f"{EXPERIMENT_NAME}/{run.run_name}",
            fn=remote(
                run_scaling_training,
                resources=resources,
                env_vars={base.HF_HUB_DISABLE_XET_ENV_VAR: "1"},
                max_retries_failure=max_retries_failure,
                max_task_failures=max_task_failures,
            ),
            resources=resources,
            config=ScalingTrainingConfig(
                policy=row.policy,
                training=base.DelphiOptimizedTrainingConfig(
                    analysis_output_path=analysis_output_path,
                    target_flops=run.target_flops,
                    tpu_type=run.tpu_type,
                    tpu_region=run.tpu_region,
                    tpu_zone=run.tpu_zone,
                    batch_size=run.batch_size,
                    mixture=cast(base.DelphiValidationMixture, ladder.ScalingMixtureKey(run.mixture)),
                    label=base.LABEL,
                    output_path=this_output_path(),
                    run_id=run.run_id,
                    run_name=run.run_name,
                    data_seed=run.data_seed,
                    train_tokens_override=None,
                    trainer_seed=run.trainer_seed,
                    validation_configs=validation_configs,
                ),
            ),
        )
        training_steps.append(training_step)
        eval_steps.append(
            olmo_base_eval_step(
                name=f"t9_mo{row.policy.target[0]}_{base._slug(run.target_flops)}_s{run.data_seed}",
                checkpoint=training_step / f"hf/step-{run.expected_checkpoint_step}",
                request_set_dir=base.TABLE9_REQUEST_SET_DIR,
                resource_config=base.TABLE9_EVAL_RESOURCES,
                wandb_group="olmo_base_eval_table9_delphi_matched_olmix_scaling",
                provenance={
                    "evaluator": "marin-native-table9-bpb",
                    "panel": "delphi_matched_olmix_scaling",
                    "scale": base._slug(run.target_flops),
                    "source_run_name": run.run_name,
                    "mixture": run.mixture,
                    "method": METHOD,
                    "data_seed": str(run.data_seed),
                    "weights_sha256": row.policy.weights_sha256,
                },
            )
        )
    return ladder.LaunchArtifacts(manifest_step, training_steps, eval_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-output-path", default=base.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--run-only",
        action="append",
        default=None,
        help="Regex on executor step names; only matching steps and their dependencies run.",
    )
    parser.add_argument(
        "--max-retries-failure",
        type=int,
        default=0,
        help="Fray retries of a training child after a failure (a sibling-host preemption ends a multi-host run "
        "with SIGSEGV, which counts as a failure); the retry resumes from the last checkpoint.",
    )
    parser.add_argument(
        "--max-task-failures",
        type=int,
        default=0,
        help="Cumulative failed task attempts a training child tolerates before it fails (0 fails on the first).",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    logging.basicConfig(level=logging.INFO)
    prefix = marin_prefix_for_region(ladder.TRAIN_TPU_REGION)
    if os.environ.get("MARIN_PREFIX", prefix) != prefix:
        raise ValueError(f"MARIN_PREFIX must be {prefix}")
    os.environ["MARIN_PREFIX"] = prefix
    rows = planned_rows(args.analysis_output_path)
    if args.dry_run:
        save_manifest(
            SaveManifestConfig(
                str(LOCAL_ARTIFACT_DIR), args.analysis_output_path, json.dumps([asdict(row) for row in rows])
            )
        )
        logger.info("Wrote %d matched-Olmix scaling rows to %s", len(rows), LOCAL_ARTIFACT_DIR)
        return
    with executor_context():
        validation_configs = {
            name: step_to_lm_mixture_component(step, include_raw_paths=False)
            for name, step in default_validation_sets(tokenizer=llama3_tokenizer).items()
        }
        artifacts = build_launch_artifacts(
            rows,
            args.analysis_output_path,
            validation_configs,
            max_retries_failure=args.max_retries_failure,
            max_task_failures=args.max_task_failures,
        )
    if os.getenv("CI") is not None:
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=MAX_CONCURRENT, run_only=args.run_only),
        steps=artifacts.steps,
        description="Matched Olmix scaling at MARINER's rungs and seeds, v6e east5b",
    )


if __name__ == "__main__":
    main()
