# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a trained offline policy with chained 3-phase training runs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import fsspec
import optax
import pandas as pd
from fray.cluster import ResourceConfig
from levanter.optim.config import LrSchedule, LrScheduleContext

from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.utils import create_cache_tokenizer_step
from marin.utilities.gcs_utils import get_vm_region

from experiments.domain_phase_mix.config import PhaseSchedule, WeightConfig
from experiments.domain_phase_mix.experiment import MixtureExperiment
from experiments.domain_phase_mix.offline_rl.build_transitions import _feature_defaults, extract_decision_state
from experiments.domain_phase_mix.offline_rl.collect_three_phase_starcoder_dataset import (
    build_wide_history,
    collect_history_long_rows,
    dedupe_history_rows,
    DEFAULT_HISTORY_SAMPLES,
)
from experiments.domain_phase_mix.offline_rl.contracts import (
    DEFAULT_OBJECTIVE_METRIC,
    DEFAULT_PHASE_END_STEPS,
    DEFAULT_TOTAL_STEPS,
)
from experiments.domain_phase_mix.offline_rl.policy_artifact import clip_action, load_policy_artifact, normalize_state
from experiments.domain_phase_mix.three_phase_starcoder_experiment import (
    BATCH_SIZE,
    EVAL_DATASETS_CACHE_PATH,
    EVAL_TASKS,
    SEQ_LEN,
    TARGET_BUDGET,
    TOKENIZER_CACHE_BASE,
    TOKENIZER_NAME,
    get_nemotron_starcoder_domains,
)
from experiments.domain_phase_mix.proxy_sweep import regmix_60m_proxy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobalCosineLrSchedule(LrSchedule):
    """Cosine decay anchored to global rollout steps across chained phase jobs.

    Warmup is handled by Levanter's optimizer wrapper. This schedule should only
    emit the post-warmup decay curve; otherwise warmup is applied twice and the
    LR resets at the warmup boundary.
    """

    total_steps: int
    exponent: float = 1.0

    def build(self, ctx: LrScheduleContext):
        decay_steps = max(self.total_steps - max(ctx.warmup_steps, 0), 1)
        return optax.cosine_decay_schedule(
            init_value=ctx.learning_rate,
            decay_steps=decay_steps,
            alpha=ctx.min_lr_ratio,
            exponent=self.exponent,
        )


@dataclass(frozen=True)
class PhaseTrainPlan:
    """Resolved training plan for one phase."""

    phase_index: int
    cumulative_steps: int
    initialize_from_checkpoint_path: str | None
    reset_data_loader_on_init: bool


@dataclass(frozen=True)
class EvaluateConfig:
    """Config for policy rollout evaluation."""

    policy_artifact_path: str
    output_dir: str
    n_replicates: int = 3
    wandb_entity: str = "marin-community"
    wandb_project: str = "marin"
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC
    run_name_prefix: str = "pinlin_calvin_xu/data_mixture/three_phase_starcoder_policy_eval"
    phase_end_steps: tuple[int, int] = DEFAULT_PHASE_END_STEPS
    total_steps: int = DEFAULT_TOTAL_STEPS
    dry_run: bool = False
    marin_prefix: str | None = None
    allow_local_fallback: bool = True
    max_concurrent: int = 1
    include_cache_steps: bool = False
    tpu_type: str = "v5p-8"


def build_phase_train_plan(phase_index: int, checkpoint_path: str | None = None) -> PhaseTrainPlan:
    """Build phase plan using cumulative step counts and optional resume checkpoint."""
    cumulative_steps = [DEFAULT_PHASE_END_STEPS[0], DEFAULT_PHASE_END_STEPS[1], DEFAULT_TOTAL_STEPS][phase_index]
    return PhaseTrainPlan(
        phase_index=phase_index,
        cumulative_steps=cumulative_steps,
        initialize_from_checkpoint_path=checkpoint_path,
        reset_data_loader_on_init=False if checkpoint_path else True,
    )


def _resolve_prefix(override: str | None = None) -> str:
    if override:
        return override
    prefix = os.environ.get("MARIN_PREFIX")
    if prefix is None:
        prefix = "gs://marin-us-central1"
    return prefix


def _rebase_to_prefix(path: str, prefix: str) -> str:
    """Rebase a gs:// path to the given MARIN prefix while preserving the suffix."""
    if not path.startswith("gs://"):
        return path
    if not prefix.startswith("gs://"):
        return path
    try:
        _, rest = path.split("://", maxsplit=1)
        _, suffix = rest.split("/", maxsplit=1)
    except ValueError:
        return path
    return f"{prefix.rstrip('/')}/{suffix}"


def _discover_latest_checkpoint(training_output_path: str) -> str:
    checkpoint_pattern = os.path.join(training_output_path, "checkpoints", "step-*")
    fs, base = fsspec.core.url_to_fs(checkpoint_pattern)
    matches = fs.glob(base)
    if not matches:
        raise FileNotFoundError(f"No checkpoints found under {checkpoint_pattern}")
    step_re = re.compile(r"step-(\d+)")
    scored = []
    for item in matches:
        match = step_re.search(item)
        if match:
            scored.append((int(match.group(1)), item))
    if not scored:
        raise FileNotFoundError(f"No step checkpoints found under {checkpoint_pattern}")
    scored.sort(key=lambda pair: pair[0])
    protocol = fs.protocol[0] if isinstance(fs.protocol, (tuple, list)) else fs.protocol
    latest = scored[-1][1]
    return f"{protocol}://{latest}" if protocol else latest


def _build_single_phase_experiment(
    name: str,
    num_train_steps: int,
    tpu_type: str,
    eval_datasets_cache_path: str,
) -> MixtureExperiment:
    phase_schedule = PhaseSchedule.from_boundaries([], names=["phase_0"])
    return MixtureExperiment(
        name=name,
        domains=get_nemotron_starcoder_domains(),
        phase_schedule=phase_schedule,
        model_config=regmix_60m_proxy,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=num_train_steps,
        target_budget=TARGET_BUDGET,
        eval_harness_tasks=EVAL_TASKS,
        eval_datasets_cache_path=eval_datasets_cache_path,
        resources=ResourceConfig.with_tpu(tpu_type),
    )


def _build_training_step(
    *,
    run_namespace: str,
    phase_plan: PhaseTrainPlan,
    action_starcoder: float,
    override_output_path: str,
    seed: int,
    global_total_steps: int,
    tpu_type: str,
    eval_datasets_cache_path: str,
):
    experiment = _build_single_phase_experiment(
        name=run_namespace,
        num_train_steps=phase_plan.cumulative_steps,
        tpu_type=tpu_type,
        eval_datasets_cache_path=eval_datasets_cache_path,
    )
    optimizer_config = replace(
        experiment.optimizer_config,
        lr_schedule=GlobalCosineLrSchedule(total_steps=global_total_steps),
    )
    phase_weights = {
        "phase_0": {
            "nemotron_full": 1.0 - action_starcoder,
            "starcoder": action_starcoder,
        }
    }
    weight_config = WeightConfig(run_id=seed, phase_weights=phase_weights)
    return experiment.create_training_step(
        weight_config=weight_config,
        name_prefix=run_namespace,
        run_name=f"phase_{phase_plan.phase_index}",
        num_train_steps=phase_plan.cumulative_steps,
        steps_per_export=phase_plan.cumulative_steps,
        steps_per_eval=200,
        data_seed=seed,
        optimizer_config=optimizer_config,
        initialize_from_checkpoint_path=phase_plan.initialize_from_checkpoint_path,
        reset_data_loader_on_init=phase_plan.reset_data_loader_on_init,
    ).with_output_path(override_output_path)


def _fetch_wandb_run_by_display_name(entity: str, project: str, display_name: str):
    import wandb

    api = wandb.Api(timeout=60)
    runs = list(api.runs(f"{entity}/{project}", filters={"display_name": display_name}))
    if not runs:
        raise ValueError(f"No W&B run found with display_name={display_name}")
    return runs[-1]


def _state_from_completed_run(
    wb_run,
    *,
    phase_index: int,
    decision_step: int,
    prev_action: float,
    objective_metric: str,
    total_steps: int,
) -> dict[str, float]:
    long_rows = collect_history_long_rows(
        wb_run,
        metric_keys=("train/loss", "eval/loss", objective_metric, "throughput/total_tokens"),
        history_samples=DEFAULT_HISTORY_SAMPLES,
    )
    long_df = dedupe_history_rows(pd.DataFrame(long_rows))
    wide_df = build_wide_history(long_df)
    defaults = _feature_defaults(wide_df, ("last_train_loss", "last_eval_loss", "last_obj_bpb"))
    state = extract_decision_state(
        history=wide_df,
        decision_step=decision_step,
        phase_index=phase_index,
        prev_action_starcoder=prev_action,
        total_steps=total_steps,
        objective_metric=objective_metric,
        defaults=defaults,
    )
    return state


def _policy_predict_action(artifact_path: str, state: dict[str, float], device: str = "cpu") -> float:
    artifact = load_policy_artifact(artifact_path)
    normalized = normalize_state(state, artifact).reshape(1, -1)
    artifact_dir = Path(artifact_path).resolve().parent
    model_path = Path(artifact.model_path)
    if model_path.is_absolute():
        if not model_path.exists():
            # Support policy artifacts produced on other machines by falling back
            # to artifact-local paths.
            candidates = [
                (artifact_dir / model_path.name),
                (artifact_dir / "terminal_reward" / model_path.name),
                (artifact_dir / "delta_reward" / model_path.name),
            ]
            for candidate in candidates:
                if candidate.exists():
                    model_path = candidate
                    break
            else:
                raise FileNotFoundError(
                    f"Policy model path {model_path} does not exist and no local fallback was found near {artifact_dir}."
                )
    else:
        model_path = (artifact_dir / model_path).resolve()
    try:
        import d3rlpy
    except ImportError as exc:
        raise RuntimeError("d3rlpy is required for policy evaluation.") from exc

    if hasattr(d3rlpy, "load_learnable"):
        policy = d3rlpy.load_learnable(str(model_path), device=device)
    else:
        from d3rlpy.base import load_learnable

        policy = load_learnable(str(model_path), device=device)
    action = policy.predict(normalized)
    raw = float(action[0][0] if hasattr(action[0], "__len__") else action[0])
    return clip_action(raw, artifact)


def _is_local_region_error(exc: Exception) -> bool:
    patterns = (
        "Could not determine the region of the VM",
        "Failed to determine VM region from GCP metadata",
    )

    queue: list[BaseException] = [exc]
    visited: set[int] = set()

    while queue:
        current = queue.pop()
        current_id = id(current)
        if current_id in visited:
            continue
        visited.add(current_id)

        text = str(current)
        if any(pattern in text for pattern in patterns):
            return True

        for arg in current.args:
            if isinstance(arg, BaseException):
                queue.append(arg)
            elif isinstance(arg, str) and any(pattern in arg for pattern in patterns):
                return True

        cause = current.__cause__
        context = current.__context__
        if cause is not None:
            queue.append(cause)
        if context is not None:
            queue.append(context)

    return False


def _should_force_local_dry_run(config: EvaluateConfig) -> bool:
    if config.dry_run:
        return True
    if not config.allow_local_fallback:
        return False

    try:
        get_vm_region()
        return False
    except Exception:
        logger.warning("Could not resolve VM region metadata; forcing dry-run policy evaluation.")
        return True


def evaluate_policy(config: EvaluateConfig) -> pd.DataFrame:
    """Run 3 chained phase runs for each replicate and return rollout results."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = _resolve_prefix(config.marin_prefix)
    os.environ["MARIN_PREFIX"] = prefix
    eval_datasets_cache_path = _rebase_to_prefix(EVAL_DATASETS_CACHE_PATH, prefix)
    tokenizer_cache_base = _rebase_to_prefix(TOKENIZER_CACHE_BASE, prefix)
    job_token = uuid.uuid4().hex[:8]
    objective_metric = config.objective_metric
    force_dry_run = _should_force_local_dry_run(config)

    rows: list[dict] = []
    for replicate_idx in range(config.n_replicates):
        base_slug = re.sub(r"[^a-zA-Z0-9]+", "", config.run_name_prefix.rsplit("/", maxsplit=1)[-1].lower())
        base_slug = (base_slug[:12] or "tsceval").strip("_")
        run_namespace = f"pinlin_calvin_xu/data_mixture/{base_slug}_{job_token}_r{replicate_idx:02d}"
        actions = []
        run_ids = []

        # Initial state is neutral and normalized by artifact stats.
        state = {
            "phase_index": 0.0,
            "last_train_loss": 0.0,
            "last_eval_loss": 0.0,
            "last_obj_bpb": 0.0,
            "tokens_frac": 0.0,
            "steps_since_last_eval_frac": 1.0,
            "prev_action_starcoder": 0.5,
        }

        checkpoint_path: str | None = None
        final_metric = None
        replicate_force_dry = force_dry_run

        for phase_idx in range(3):
            action = _policy_predict_action(config.policy_artifact_path, state)
            actions.append(action)

            phase_plan = build_phase_train_plan(phase_idx, checkpoint_path=checkpoint_path)
            # Marin infers trainer.id (and therefore WandB run id) from the output path basename.
            # Using a static basename like "phase_0" causes run-id collisions and hidden logging.
            phase_output_basename = f"phase_{phase_idx}_{job_token}_r{replicate_idx:02d}"
            relative_output_path = (
                "domain_phase_mix/offline_rl/policy_eval/"
                f"{job_token}/rep_{replicate_idx:02d}/{phase_output_basename}"
            )
            training_step = _build_training_step(
                run_namespace=run_namespace,
                phase_plan=phase_plan,
                action_starcoder=action,
                override_output_path=relative_output_path,
                seed=replicate_idx * 10 + phase_idx,
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
                        description=f"Policy eval replicate {replicate_idx} phase {phase_idx}",
                    )
                except Exception as exc:
                    if config.allow_local_fallback and _is_local_region_error(exc):
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
                    checkpoint_path = _discover_latest_checkpoint(output_path)

                    display_name = f"{run_namespace}/phase_{phase_idx}"
                    wb_run = _fetch_wandb_run_by_display_name(
                        config.wandb_entity,
                        config.wandb_project,
                        display_name=display_name,
                    )
                    run_ids.append(wb_run.id)

                    if phase_idx < 2:
                        decision_step = config.phase_end_steps[phase_idx]
                        state = _state_from_completed_run(
                            wb_run,
                            phase_index=phase_idx + 1,
                            decision_step=decision_step,
                            prev_action=action,
                            objective_metric=objective_metric,
                            total_steps=config.total_steps,
                        )
                    else:
                        summary_value = wb_run.summary.get(objective_metric)
                        final_metric = float(summary_value) if isinstance(summary_value, int | float) else None

            if replicate_force_dry:
                run_ids.append(f"dry_run_rep{replicate_idx}_phase{phase_idx}")
                checkpoint_path = f"dry://rep{replicate_idx}/phase{phase_idx}/checkpoint"
                state = {
                    "phase_index": float(min(phase_idx + 1, 2)),
                    "last_train_loss": 0.0,
                    "last_eval_loss": 0.0,
                    "last_obj_bpb": 0.0,
                    "tokens_frac": (phase_idx + 1) / 3.0,
                    "steps_since_last_eval_frac": 0.0,
                    "prev_action_starcoder": action,
                }
                final_metric = 0.0

        rows.append(
            {
                "replicate": replicate_idx,
                "phase_0_starcoder": actions[0],
                "phase_1_starcoder": actions[1],
                "phase_2_starcoder": actions[2],
                "final_objective": final_metric,
                "phase_0_run_id": run_ids[0],
                "phase_1_run_id": run_ids[1],
                "phase_2_run_id": run_ids[2],
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
    parser = argparse.ArgumentParser(description="Evaluate a trained offline policy on chained 3-phase runs.")
    parser.add_argument("--policy-artifact-path", type=str, required=True, help="Path to policy_artifact.json.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/three_phase_starcoder/policy_eval",
        help="Directory to write policy rollout results.",
    )
    parser.add_argument("--n-replicates", type=int, default=3, help="Number of full chained replicates.")
    parser.add_argument("--wandb-entity", type=str, default="marin-community", help="W&B entity.")
    parser.add_argument("--wandb-project", type=str, default="marin", help="W&B project.")
    parser.add_argument("--objective-metric", type=str, default=DEFAULT_OBJECTIVE_METRIC, help="Objective metric.")
    parser.add_argument(
        "--run-name-prefix",
        type=str,
        default="pinlin_calvin_xu/data_mixture/three_phase_starcoder_policy_eval",
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
    # executor_main is draccus-wrapped; strip script args so it only sees executor args.
    sys.argv = [sys.argv[0], *remaining]
    evaluate_policy(
        EvaluateConfig(
            policy_artifact_path=args.policy_artifact_path,
            output_dir=args.output_dir,
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
    )


if __name__ == "__main__":
    main()
