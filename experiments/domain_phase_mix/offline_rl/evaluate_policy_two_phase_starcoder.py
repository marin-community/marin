# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect or evaluate a trained offline policy on chained 2-phase StarCoder runs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.utils import create_cache_tokenizer_step

from experiments.domain_phase_mix.offline_rl import evaluate_policy_three_phase_starcoder as shared
from experiments.domain_phase_mix.offline_rl.contracts import (
    DEFAULT_OBJECTIVE_METRIC,
    DEFAULT_TWO_PHASE_STARCODER_FAMILY,
)
from experiments.domain_phase_mix.offline_rl.policy_artifact import PolicyArtifactV2
from experiments.domain_phase_mix.two_phase_starcoder_experiment import (
    EVAL_DATASETS_CACHE_PATH,
    EVAL_TASKS,
    TOKENIZER_CACHE_BASE,
    TOKENIZER_NAME,
    create_two_phase_experiment,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluateTwoPhaseConfig:
    """Config for two-phase policy inspection or rollout."""

    policy_artifact_path: str
    output_dir: str
    inspect_only: bool = False
    n_replicates: int = 1
    wandb_entity: str = "marin-community"
    wandb_project: str = "marin"
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC
    run_name_prefix: str = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_policy_eval"
    phase_end_steps: tuple[int, ...] = field(default_factory=lambda: _default_two_phase_phase_end_steps())
    total_steps: int = DEFAULT_TWO_PHASE_STARCODER_FAMILY.total_steps
    dry_run: bool = False
    marin_prefix: str | None = None
    allow_local_fallback: bool = True
    max_concurrent: int = 1
    include_cache_steps: bool = False
    tpu_type: str = "v5p-8"


def build_phase_train_plan(
    config: EvaluateTwoPhaseConfig,
    phase_index: int,
    checkpoint_path: str | None = None,
) -> shared.PhaseTrainPlan:
    """Build the cumulative training plan for one 2-phase rollout stage."""
    cumulative_steps = [config.phase_end_steps[0], config.total_steps][phase_index]
    return shared.PhaseTrainPlan(
        phase_index=phase_index,
        cumulative_steps=cumulative_steps,
        initialize_from_checkpoint_path=checkpoint_path,
        reset_data_loader_on_init=False if checkpoint_path else True,
    )


def _default_two_phase_phase_end_steps() -> tuple[int, ...]:
    experiment = create_two_phase_experiment(name="offline_rl_two_phase_bounds")
    return shared._aligned_phase_end_steps(experiment)


def _build_native_rollout_experiment(
    *,
    name: str,
    tpu_type: str,
    eval_datasets_cache_path: str,
):
    experiment = create_two_phase_experiment(name=name)
    experiment.resources = shared.ResourceConfig.with_tpu(tpu_type)
    experiment.eval_datasets_cache_path = eval_datasets_cache_path
    return experiment


def _build_training_step(
    *,
    run_namespace: str,
    phase_plan: shared.PhaseTrainPlan,
    actions_so_far: list[float],
    override_output_path: str,
    run_id: int,
    data_seed: int,
    global_total_steps: int,
    tpu_type: str,
    eval_datasets_cache_path: str,
):
    experiment = _build_native_rollout_experiment(
        name=run_namespace,
        tpu_type=tpu_type,
        eval_datasets_cache_path=eval_datasets_cache_path,
    )
    optimizer_config = replace(
        experiment.optimizer_config,
        lr_schedule=shared.GlobalCosineLrSchedule(total_steps=global_total_steps),
    )
    weight_config = shared._build_rollout_weight_config(
        run_id=run_id,
        actions_so_far=actions_so_far,
        total_phases=experiment.phase_schedule.n_phases,
    )
    return experiment.create_training_step(
        weight_config=weight_config,
        name_prefix=run_namespace,
        run_name=f"phase_{phase_plan.phase_index}",
        num_train_steps=phase_plan.cumulative_steps,
        steps_per_export=phase_plan.cumulative_steps,
        steps_per_eval=experiment.steps_per_eval,
        data_seed=data_seed,
        optimizer_config=optimizer_config,
        experiment_budget_override=experiment.experiment_budget,
        initialize_from_checkpoint_path=phase_plan.initialize_from_checkpoint_path,
        reset_data_loader_on_init=phase_plan.reset_data_loader_on_init,
    ).with_output_path(override_output_path)


def _hypothetical_state_for_decision(
    *,
    artifact_path: str,
    config: EvaluateTwoPhaseConfig,
    decision_index: int,
    prior_actions: list[float],
) -> dict[str, float]:
    artifact = shared._load_policy_artifact_cached(artifact_path)
    state = shared._artifact_state_defaults(artifact)
    if isinstance(artifact, PolicyArtifactV2):
        defaults = shared._decision_state_default(
            artifact_path,
            decision_index=decision_index,
            run_family="two_phase_starcoder",
        )
        state.update({key: float(value) for key, value in defaults.items() if key in artifact.state_keys})
        phase_lengths = shared._phase_lengths(config.total_steps, config.phase_end_steps)
        prev_action, cumulative_exposure, delta_prev_action = shared._compute_exposure(
            prior_actions,
            phase_lengths,
            decision_index,
        )
        decision_step = shared._decision_steps(config.phase_end_steps)[decision_index]
        budget_frac_consumed = float(decision_step) / float(config.total_steps)
        state.update(
            {
                "decision_index": float(decision_index),
                "num_phases_total": 2.0,
                "remaining_decisions": float(max(0, 1 - decision_index)),
                "budget_frac_consumed": budget_frac_consumed,
                "budget_frac_remaining": 1.0 - budget_frac_consumed,
                "global_step": float(decision_step),
                "prev_action_starcoder": float(prev_action),
                "cumulative_starcoder_exposure": float(cumulative_exposure),
                "delta_prev_action": float(delta_prev_action),
            }
        )
        return {key: float(state[key]) for key in artifact.state_keys}
    raise TypeError(f"Two-phase inspection expects a v2 policy artifact, found {type(artifact)!r}.")


def inspect_policy(config: EvaluateTwoPhaseConfig) -> pd.DataFrame:
    """Inspect offline actions for two-phase StarCoder without launching training."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_0_state = shared._initial_policy_state(
        config.policy_artifact_path,
        num_phases_total=2,
        total_steps=config.total_steps,
        phase_end_steps=config.phase_end_steps,
        run_family="two_phase_starcoder",
    )
    phase_0_action = shared._policy_predict_action(config.policy_artifact_path, phase_0_state)

    phase_1_state = _hypothetical_state_for_decision(
        artifact_path=config.policy_artifact_path,
        config=config,
        decision_index=1,
        prior_actions=[phase_0_action],
    )
    phase_1_action = shared._policy_predict_action(config.policy_artifact_path, phase_1_state)

    rows = [
        {
            "phase_0_starcoder": phase_0_action,
            "phase_1_starcoder": phase_1_action,
            "state_source": "two_phase_family_medians",
        }
    ]
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "policy_inspection_results.csv", index=False)
    summary = {
        "policy_artifact_path": config.policy_artifact_path,
        "run_family": "two_phase_starcoder",
        "phase_end_steps": list(config.phase_end_steps),
        "total_steps": config.total_steps,
        "phase_0_action_starcoder": float(phase_0_action),
        "phase_1_action_starcoder": float(phase_1_action),
        "phase_0_state": phase_0_state,
        "phase_1_hypothetical_state": phase_1_state,
    }
    with (output_dir / "policy_inspection_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return results


def evaluate_policy(config: EvaluateTwoPhaseConfig) -> pd.DataFrame:
    """Run chained 2-phase training for each replicate and return rollout results."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = shared._resolve_prefix(config.marin_prefix)
    os.environ["MARIN_PREFIX"] = prefix
    eval_datasets_cache_path = shared._rebase_to_prefix(EVAL_DATASETS_CACHE_PATH, prefix)
    tokenizer_cache_base = shared._rebase_to_prefix(TOKENIZER_CACHE_BASE, prefix)
    job_token = uuid.uuid4().hex[:8]
    objective_metric = config.objective_metric
    force_dry_run = config.dry_run or shared._should_force_local_dry_run(config)

    rows: list[dict[str, object]] = []
    for replicate_idx in range(config.n_replicates):
        base_slug = re.sub(r"[^a-zA-Z0-9]+", "", config.run_name_prefix.rsplit("/", maxsplit=1)[-1].lower())
        base_slug = (base_slug[:12] or "tpseval").strip("_")
        run_namespace = f"pinlin_calvin_xu/data_mixture/{base_slug}_{job_token}_r{replicate_idx:02d}"
        replicate_data_seed = shared._replicate_data_seed(replicate_idx)
        state = shared._initial_policy_state(
            config.policy_artifact_path,
            num_phases_total=2,
            total_steps=config.total_steps,
            phase_end_steps=config.phase_end_steps,
            run_family="two_phase_starcoder",
        )
        actions: list[float] = []
        run_ids: list[str] = []
        checkpoint_path: str | None = None
        final_metric: float | None = None
        replicate_force_dry = force_dry_run

        for phase_idx in range(2):
            action = shared._policy_predict_action(config.policy_artifact_path, state)
            actions.append(action)

            phase_plan = build_phase_train_plan(config, phase_idx, checkpoint_path=checkpoint_path)
            phase_output_basename = f"phase_{phase_idx}_{job_token}_r{replicate_idx:02d}"
            relative_output_path = (
                "domain_phase_mix/offline_rl/policy_eval_two_phase/"
                f"{job_token}/rep_{replicate_idx:02d}/{phase_output_basename}"
            )
            training_step = _build_training_step(
                run_namespace=run_namespace,
                phase_plan=phase_plan,
                actions_so_far=actions,
                override_output_path=relative_output_path,
                run_id=shared._phase_run_id(replicate_idx, phase_idx),
                data_seed=replicate_data_seed,
                global_total_steps=config.total_steps,
                tpu_type=config.tpu_type,
                eval_datasets_cache_path=eval_datasets_cache_path,
            )

            if not replicate_force_dry:
                try:
                    steps = [training_step]
                    if config.include_cache_steps:
                        os.environ["MARIN_TOKENIZER_CACHE_PATH"] = tokenizer_cache_base
                        cache_tokenizer_step = create_cache_tokenizer_step(
                            tokenizer_name=TOKENIZER_NAME,
                            gcs_path=os.path.join(tokenizer_cache_base, TOKENIZER_NAME.replace("/", "--")),
                            name_prefix=run_namespace,
                        )
                        cache_eval_step = create_cache_eval_datasets_step(
                            eval_tasks=EVAL_TASKS,
                            gcs_path=eval_datasets_cache_path,
                            name_prefix=run_namespace,
                        )
                        steps = [cache_tokenizer_step, cache_eval_step, training_step]

                    executor_main(
                        ExecutorMainConfig(max_concurrent=config.max_concurrent),
                        steps=steps,
                        description=f"Two-phase policy eval replicate {replicate_idx} phase {phase_idx}",
                    )
                except Exception as exc:
                    if config.allow_local_fallback and shared._is_local_region_error(exc):
                        logger.warning(
                            "Falling back to dry-run for replicate %d after phase %d infrastructure error: %s",
                            replicate_idx,
                            phase_idx,
                            exc,
                        )
                        replicate_force_dry = True
                    else:
                        raise

                if not replicate_force_dry:
                    output_path = os.path.join(prefix, relative_output_path)
                    checkpoint_path = shared._discover_latest_checkpoint(output_path)

                    display_name = f"{run_namespace}/phase_{phase_idx}"
                    wb_run = shared._fetch_wandb_run_by_display_name(
                        config.wandb_entity,
                        config.wandb_project,
                        display_name=display_name,
                    )
                    run_ids.append(wb_run.id)

                    if phase_idx < 1:
                        state = shared._state_from_completed_run(
                            wb_run,
                            phase_index=phase_idx + 1,
                            decision_step=config.phase_end_steps[0],
                            prev_action=action,
                            objective_metric=objective_metric,
                            total_steps=config.total_steps,
                            artifact_path=config.policy_artifact_path,
                            prior_actions=actions,
                            phase_end_steps=config.phase_end_steps,
                            num_phases_total=2,
                            run_family="two_phase_starcoder",
                        )
                    else:
                        summary_value = wb_run.summary.get(objective_metric)
                        final_metric = float(summary_value) if isinstance(summary_value, int | float) else None

            if replicate_force_dry:
                run_ids.append(f"dry_run_rep{replicate_idx}_phase{phase_idx}")
                checkpoint_path = f"dry://rep{replicate_idx}/phase{phase_idx}/checkpoint"
                if phase_idx < 1:
                    state = _hypothetical_state_for_decision(
                        artifact_path=config.policy_artifact_path,
                        config=config,
                        decision_index=phase_idx + 1,
                        prior_actions=actions,
                    )
                else:
                    final_metric = 0.0

        rows.append(
            {
                "replicate": replicate_idx,
                "phase_0_starcoder": actions[0],
                "phase_1_starcoder": actions[1],
                "final_objective": final_metric,
                "phase_0_run_id": run_ids[0],
                "phase_1_run_id": run_ids[1],
            }
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "policy_rollout_results.csv", index=False)
    summary = {
        "objective_metric": objective_metric,
        "n_replicates": len(results_df),
        "mean_final_objective": float(results_df["final_objective"].mean()),
        "std_final_objective": float(results_df["final_objective"].std(ddof=0)),
    }
    with (output_dir / "policy_rollout_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return results_df


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Inspect or evaluate a trained offline policy on chained 2-phase runs.")
    parser.add_argument("--policy-artifact-path", type=str, required=True, help="Path to policy_artifact.json.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/two_phase_starcoder/policy_eval",
        help="Directory to write policy inspection or rollout results.",
    )
    parser.add_argument(
        "--inspect-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only inspect the offline phase-0 and hypothetical phase-1 actions without submitting training.",
    )
    parser.add_argument("--n-replicates", type=int, default=1, help="Number of full chained replicates.")
    parser.add_argument("--wandb-entity", type=str, default="marin-community", help="W&B entity.")
    parser.add_argument("--wandb-project", type=str, default="marin", help="W&B project.")
    parser.add_argument("--objective-metric", type=str, default=DEFAULT_OBJECTIVE_METRIC, help="Objective metric.")
    parser.add_argument(
        "--run-name-prefix",
        type=str,
        default="pinlin_calvin_xu/data_mixture/two_phase_starcoder_policy_eval",
        help="Run prefix used for chained phase executions.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip training execution and emit synthetic outputs.")
    parser.add_argument(
        "--marin-prefix",
        type=str,
        default=None,
        help="Storage prefix for executor outputs (defaults to $MARIN_PREFIX or gs://marin-us-central1).",
    )
    parser.add_argument(
        "--allow-local-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fallback to dry-run mode when TPU/GCP VM metadata is unavailable.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Maximum number of executor steps to run concurrently.",
    )
    parser.add_argument(
        "--include-cache-steps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include cache tokenizer/eval steps in each phase submission.",
    )
    parser.add_argument(
        "--tpu-type",
        type=str,
        default="v5p-8",
        help="TPU type requested by each chained training phase (for example, v6e-8).",
    )
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    config = EvaluateTwoPhaseConfig(
        policy_artifact_path=args.policy_artifact_path,
        output_dir=args.output_dir,
        inspect_only=args.inspect_only,
        n_replicates=args.n_replicates,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        objective_metric=args.objective_metric,
        run_name_prefix=args.run_name_prefix,
        dry_run=args.dry_run,
        marin_prefix=args.marin_prefix,
        allow_local_fallback=args.allow_local_fallback,
        max_concurrent=args.max_concurrent,
        include_cache_steps=args.include_cache_steps,
        tpu_type=args.tpu_type,
    )
    if config.inspect_only:
        inspect_policy(config)
        return
    evaluate_policy(config)


if __name__ == "__main__":
    main()
