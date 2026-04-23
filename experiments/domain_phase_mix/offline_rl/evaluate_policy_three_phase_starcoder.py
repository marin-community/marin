# Copyright The Marin Authors
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
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from typing import Any

import fsspec
import joblib
import numpy as np
import optax
import pandas as pd
import torch
from fray.cluster import ResourceConfig
from rigging.filesystem import marin_region
from levanter.optim.config import LrSchedule, LrScheduleContext

from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.utils import create_cache_tokenizer_step

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.offline_rl.build_transitions import _feature_defaults, extract_decision_state
from experiments.domain_phase_mix.offline_rl.collect_pooled_starcoder_dataset import collect_history_long_rows_batched
from experiments.domain_phase_mix.offline_rl.collect_three_phase_starcoder_dense_dataset import (
    collect_dense_history_from_run,
)
from experiments.domain_phase_mix.offline_rl.collect_three_phase_starcoder_dataset import (
    build_wide_history,
    dedupe_history_rows,
)
from experiments.domain_phase_mix.offline_rl.contracts import (
    DEFAULT_OBJECTIVE_METRIC,
    DEFAULT_PHASE_END_STEPS,
    DEFAULT_TOTAL_STEPS,
)
from experiments.domain_phase_mix.offline_rl.build_three_phase_dense_policy_dataset import (
    SEQUENCE_CHANNELS,
    build_decision_state_features,
)
from experiments.domain_phase_mix.offline_rl.ope import DecisionBatch
from experiments.domain_phase_mix.offline_rl.policy_artifact import (
    AnyPolicyArtifact,
    PolicyArtifactV1,
    PolicyArtifactV2,
    clip_action,
    load_policy_artifact,
    normalize_state,
)
from experiments.domain_phase_mix.offline_rl.train_three_phase_policy_bench_v3 import (
    GRUQNetwork,
    TransformerQNetwork,
)
from experiments.domain_phase_mix.three_phase_starcoder_experiment import (
    EVAL_DATASETS_CACHE_PATH,
    EVAL_TASKS,
    TOKENIZER_CACHE_BASE,
    TOKENIZER_NAME,
    create_three_phase_experiment,
)

logger = logging.getLogger(__name__)

ONLINE_HISTORY_BATCH_SIZE = 4
ONLINE_HISTORY_RETRY_ATTEMPTS = 3
ONLINE_HISTORY_BACKOFF_SECONDS = 2.0


class _EvalIQLMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


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
    phase_end_steps: tuple[int, int] = field(default_factory=lambda: _default_three_phase_phase_end_steps())
    total_steps: int = DEFAULT_TOTAL_STEPS
    dry_run: bool = False
    marin_prefix: str | None = None
    allow_local_fallback: bool = True
    max_concurrent: int = 1
    include_cache_steps: bool = False
    tpu_type: str = "v5p-8"


def build_phase_train_plan(
    phase_index: int,
    checkpoint_path: str | None = None,
    *,
    phase_end_steps: tuple[int, int] = DEFAULT_PHASE_END_STEPS,
    total_steps: int = DEFAULT_TOTAL_STEPS,
) -> PhaseTrainPlan:
    """Build phase plan using cumulative step counts and optional resume checkpoint."""
    cumulative_steps = [phase_end_steps[0], phase_end_steps[1], total_steps][phase_index]
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


def _discover_exact_checkpoint(training_output_path: str, expected_step: int) -> str:
    checkpoint_path = os.path.join(training_output_path, "checkpoints", f"step-{expected_step}")
    fs, base = fsspec.core.url_to_fs(checkpoint_path)
    if not fs.exists(base):
        raise FileNotFoundError(f"Expected exact checkpoint step-{expected_step} under {training_output_path}")
    protocol = fs.protocol[0] if isinstance(fs.protocol, (tuple, list)) else fs.protocol
    return f"{protocol}://{base}" if protocol else base


def _default_three_phase_phase_end_steps() -> tuple[int, int]:
    experiment = create_three_phase_experiment(name="offline_rl_phase_bounds")
    phase_end_steps = _aligned_phase_end_steps(experiment)
    if len(phase_end_steps) != 2:
        raise ValueError(f"Expected 2 three-phase boundaries, found {phase_end_steps}.")
    return (phase_end_steps[0], phase_end_steps[1])


def _aligned_phase_end_steps(experiment) -> tuple[int, ...]:
    """Return native aligned phase boundaries for a mixture experiment."""
    return tuple(
        phase.get_start_step_aligned(experiment.num_train_steps, experiment.batch_size, experiment.mixture_block_size)
        for phase in experiment.phase_schedule.phases[1:]
    )


def _build_rollout_weight_config(
    *,
    run_id: int,
    actions_so_far: list[float],
    total_phases: int,
) -> WeightConfig:
    """Build a full native phase schedule using known actions and safe placeholders.

    Future phases are filled with the most recent chosen action. The rollout
    stops before those future phases are reached, so they only serve to keep
    the native multi-phase schedule structurally valid.
    """
    if not actions_so_far:
        raise ValueError("Need at least one chosen action to build rollout weights.")
    padded_actions = list(actions_so_far) + [actions_so_far[-1]] * (total_phases - len(actions_so_far))
    phase_weights = {
        f"phase_{phase_idx}": {
            "nemotron_full": 1.0 - padded_actions[phase_idx],
            "starcoder": padded_actions[phase_idx],
        }
        for phase_idx in range(total_phases)
    }
    return WeightConfig(run_id=run_id, phase_weights=phase_weights)


def _build_native_rollout_experiment(
    *,
    name: str,
    tpu_type: str,
    eval_datasets_cache_path: str,
):
    experiment = create_three_phase_experiment(
        name=name,
        eval_datasets_cache_path=eval_datasets_cache_path,
    )
    experiment.resources = ResourceConfig.with_tpu(tpu_type)
    experiment.eval_datasets_cache_path = eval_datasets_cache_path
    return experiment


def _build_training_step(
    *,
    run_namespace: str,
    phase_plan: PhaseTrainPlan,
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
        lr_schedule=GlobalCosineLrSchedule(total_steps=global_total_steps),
    )
    weight_config = _build_rollout_weight_config(
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


def _replicate_data_seed(replicate_idx: int) -> int:
    """Return the rollout data seed shared across all phases of one replicate."""
    return replicate_idx


def _phase_run_id(replicate_idx: int, phase_idx: int) -> int:
    """Return a phase-local metadata id without perturbing data ordering."""
    return replicate_idx * 10 + phase_idx


def _fetch_wandb_run_by_display_name(entity: str, project: str, display_name: str):
    import wandb

    api = wandb.Api(timeout=60)
    runs = list(api.runs(f"{entity}/{project}", filters={"display_name": display_name}))
    if not runs:
        raise ValueError(f"No W&B run found with display_name={display_name}")
    return runs[-1]


@cache
def _load_policy_artifact_cached(artifact_path: str) -> AnyPolicyArtifact:
    return load_policy_artifact(artifact_path)


def _artifact_state_defaults(artifact: PolicyArtifactV1 | PolicyArtifactV2) -> dict[str, float]:
    return {key: float(value) for key, value in zip(artifact.state_keys, artifact.state_mean, strict=True)}


def _state_from_defaults(
    artifact_path: str,
    *,
    decision_index: int,
    num_phases_total: int,
    total_steps: int,
    phase_end_steps: tuple[int, ...],
    prior_actions: list[float],
    run_family: str | None = None,
) -> dict[str, float]:
    artifact = _load_policy_artifact_cached(artifact_path)
    state = _artifact_state_defaults(artifact)
    if isinstance(artifact, PolicyArtifactV2):
        decision_defaults = _decision_state_default(
            artifact_path,
            decision_index=decision_index,
            run_family=run_family,
        )
        state.update({key: float(value) for key, value in decision_defaults.items() if key in artifact.state_keys})
        phase_lengths = _phase_lengths(total_steps, phase_end_steps)
        prev_action_value, cumulative_exposure, delta_prev_action = _compute_exposure(
            prior_actions,
            phase_lengths,
            decision_index,
        )
        decision_steps = _decision_steps(phase_end_steps)
        decision_step = decision_steps[decision_index]
        budget_frac_consumed = float(decision_step) / float(total_steps)
        state.update(
            {
                "decision_index": float(decision_index),
                "num_phases_total": float(num_phases_total),
                "remaining_decisions": float(max(0, num_phases_total - decision_index - 1)),
                "budget_frac_consumed": budget_frac_consumed,
                "budget_frac_remaining": 1.0 - budget_frac_consumed,
                "global_step": float(decision_step),
                "steps_since_last_eval_frac": 1.0,
                "prev_action_starcoder": float(prev_action_value),
                "cumulative_starcoder_exposure": float(cumulative_exposure),
                "delta_prev_action": float(delta_prev_action),
            }
        )
        return {key: float(state[key]) for key in artifact.state_keys}

    state.update(
        {
            "phase_index": 0.0,
            "last_train_loss": 0.0,
            "last_eval_loss": 0.0,
            "last_obj_bpb": 0.0,
            "tokens_frac": 0.0,
            "steps_since_last_eval_frac": 1.0,
            "prev_action_starcoder": 0.5,
        }
    )
    return {key: float(state[key]) for key in artifact.state_keys}


def _initial_policy_state(
    artifact_path: str,
    *,
    num_phases_total: int = 3,
    total_steps: int = DEFAULT_TOTAL_STEPS,
    phase_end_steps: tuple[int, ...] = DEFAULT_PHASE_END_STEPS,
    run_family: str | None = None,
) -> dict[str, float]:
    return _state_from_defaults(
        artifact_path,
        decision_index=0,
        num_phases_total=num_phases_total,
        total_steps=total_steps,
        phase_end_steps=phase_end_steps,
        prior_actions=[],
        run_family=run_family,
    )


def _resolve_policy_model_path(artifact_path: str, model_path_str: str) -> Path:
    artifact_dir = Path(artifact_path).resolve().parent
    model_path = Path(model_path_str)
    if not model_path.is_absolute():
        return (artifact_dir / model_path).resolve()
    if model_path.exists():
        return model_path

    candidates = [
        artifact_dir / model_path.name,
        artifact_dir / "terminal_reward" / model_path.name,
        artifact_dir / "delta_reward" / model_path.name,
        artifact_dir / "full_models" / "outcome_planner" / model_path.name,
        artifact_dir / "full_models" / "discrete_iql" / model_path.name,
        artifact_dir / "full_models" / "discrete_cql" / model_path.name,
        artifact_dir / "full_models" / "discrete_bc" / model_path.name,
        artifact_dir / "full_models" / "continuous_cql" / model_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Policy model path {model_path} does not exist and no local fallback was found near {artifact_dir}."
    )


def _resolve_artifact_aux_path(artifact_path: str, target_path_str: str) -> Path:
    artifact_dir = Path(artifact_path).resolve().parent
    target_path = Path(target_path_str)
    if not target_path.is_absolute():
        return (artifact_dir / target_path).resolve()
    if target_path.exists():
        return target_path
    fallback = artifact_dir / target_path.name
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(f"Artifact aux path {target_path} does not exist near {artifact_dir}.")


@cache
def _load_outcome_planner_bundle(artifact_path: str) -> dict[str, Any]:
    artifact = _load_policy_artifact_cached(artifact_path)
    if not isinstance(artifact, PolicyArtifactV2) or artifact.kind != "sklearn_outcome_planner_v2":
        raise ValueError(f"Artifact at {artifact_path} is not a v2 outcome planner.")
    model_path = _resolve_policy_model_path(artifact_path, artifact.model_path)
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected planner bundle dict at {model_path}, found {type(bundle)!r}.")
    required = {"feature_keys", "action_grid", "behavior_policy", "support_threshold"}
    missing = required.difference(bundle)
    if missing:
        raise KeyError(f"Planner bundle at {model_path} is missing keys: {sorted(missing)}")
    if "q_models" not in bundle and "reward_models" not in bundle:
        raise KeyError(f"Planner bundle at {model_path} must contain either q_models or reward_models.")
    return bundle


@cache
def _load_dynamic_q_v3_bundle(artifact_path: str) -> dict[str, Any]:
    artifact = _load_policy_artifact_cached(artifact_path)
    if not isinstance(artifact, PolicyArtifactV2) or artifact.kind != "sklearn_dynamic_q_planner_v3":
        raise ValueError(f"Artifact at {artifact_path} is not a v3 dynamic-Q planner.")
    model_path = _resolve_policy_model_path(artifact_path, artifact.model_path)
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected dict bundle at {model_path}, found {type(bundle)!r}.")
    required = {"feature_keys", "action_grid", "q_models", "behavior_policy", "support_lambda", "support_floor"}
    missing = required.difference(bundle)
    if missing:
        raise KeyError(f"Dynamic-Q bundle at {model_path} is missing keys: {sorted(missing)}")
    return bundle


@cache
def _load_decision_state_defaults(artifact_path: str) -> dict[str, dict[str, float]]:
    artifact = _load_policy_artifact_cached(artifact_path)
    if not isinstance(artifact, PolicyArtifactV2):
        return {}
    path = artifact.aux_paths.get("decision_state_defaults")
    if not path:
        return {}
    resolved = _resolve_artifact_aux_path(artifact_path, path)
    with resolved.open() as f:
        payload = json.load(f)
    return {
        str(decision_index): {str(key): float(value) for key, value in values.items()}
        for decision_index, values in payload.items()
    }


@cache
def _load_family_decision_state_defaults(artifact_path: str) -> dict[str, dict[str, dict[str, float]]]:
    artifact = _load_policy_artifact_cached(artifact_path)
    if not isinstance(artifact, PolicyArtifactV2):
        return {}
    path = artifact.aux_paths.get("decision_state_defaults_by_family")
    if not path:
        return {}
    resolved = _resolve_artifact_aux_path(artifact_path, path)
    with resolved.open() as f:
        payload = json.load(f)
    return {
        str(run_family): {
            str(decision_index): {str(key): float(value) for key, value in values.items()}
            for decision_index, values in family_payload.items()
        }
        for run_family, family_payload in payload.items()
    }


def _decision_state_default(
    artifact_path: str,
    *,
    decision_index: int,
    run_family: str | None = None,
) -> dict[str, float]:
    family_defaults = _load_family_decision_state_defaults(artifact_path)
    if run_family is not None:
        run_family_defaults = family_defaults.get(run_family, {})
        if str(decision_index) in run_family_defaults:
            return dict(run_family_defaults[str(decision_index)])
    return dict(_load_decision_state_defaults(artifact_path).get(str(decision_index), {}))


@cache
def _load_torch_discrete_iql_policy(artifact_path: str, device: str) -> _EvalIQLMLP:
    artifact = _load_policy_artifact_cached(artifact_path)
    if not isinstance(artifact, PolicyArtifactV2) or artifact.kind != "torch_discrete_iql_v2":
        raise ValueError(f"Artifact at {artifact_path} is not a torch discrete IQL policy.")
    model_path = _resolve_policy_model_path(artifact_path, artifact.model_path)
    checkpoint = torch.load(model_path, map_location=device)
    hidden_size = int(checkpoint.get("hidden_size", 64))
    policy_net = _EvalIQLMLP(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["action_dim"]),
        hidden_size=hidden_size,
    ).to(device)
    policy_net.load_state_dict(checkpoint["policy_state_dict"])
    policy_net.eval()
    return policy_net


@cache
def _load_sequence_support_bundle(artifact_path: str) -> dict[str, Any]:
    artifact = _load_policy_artifact_cached(artifact_path)
    if not isinstance(artifact, PolicyArtifactV2):
        raise ValueError(f"Artifact at {artifact_path} is not a v2/v3 policy artifact.")
    path = artifact.aux_paths.get("support_bundle")
    if not path:
        raise KeyError(f"Artifact at {artifact_path} does not define support_bundle.")
    resolved = _resolve_artifact_aux_path(artifact_path, path)
    bundle = joblib.load(resolved)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected support bundle dict at {resolved}, found {type(bundle)!r}.")
    return bundle


@cache
def _load_gru_q_v3_policy(artifact_path: str, device: str) -> GRUQNetwork:
    artifact = _load_policy_artifact_cached(artifact_path)
    if not isinstance(artifact, PolicyArtifactV2) or artifact.kind != "torch_gru_q_v3":
        raise ValueError(f"Artifact at {artifact_path} is not a GRU Q policy.")
    model_path = _resolve_policy_model_path(artifact_path, artifact.model_path)
    checkpoint = torch.load(model_path, map_location=device)
    model = GRUQNetwork(
        summary_dim=len(artifact.state_keys),
        sequence_dim=len(SEQUENCE_CHANNELS),
        action_dim=len(artifact.action_values),
        hidden_size=int(checkpoint["hidden_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@cache
def _load_transformer_q_v3_policy(artifact_path: str, device: str) -> TransformerQNetwork:
    artifact = _load_policy_artifact_cached(artifact_path)
    if not isinstance(artifact, PolicyArtifactV2) or artifact.kind != "torch_transformer_q_v3":
        raise ValueError(f"Artifact at {artifact_path} is not a transformer Q policy.")
    model_path = _resolve_policy_model_path(artifact_path, artifact.model_path)
    checkpoint = torch.load(model_path, map_location=device)
    model = TransformerQNetwork(
        summary_dim=len(artifact.state_keys),
        sequence_dim=len(SEQUENCE_CHANNELS),
        action_dim=len(artifact.action_values),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["transformer_layers"]),
        num_heads=int(checkpoint["transformer_heads"]),
        ffn_dim=int(checkpoint["transformer_ffn_dim"]),
        dropout=float(checkpoint["dropout"]),
        max_length=int(checkpoint["max_length"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _phase_lengths(total_steps: int, phase_end_steps: tuple[int, ...]) -> list[int]:
    starts = [0, *phase_end_steps]
    ends = [*phase_end_steps, total_steps]
    return [int(end - start) for start, end in zip(starts, ends, strict=True)]


def _decision_steps(phase_end_steps: tuple[int, ...]) -> list[int]:
    return [0, *phase_end_steps]


def _last_metric_at_or_before(
    history: pd.DataFrame,
    metric_col: str,
    decision_step: int,
) -> tuple[float | None, int | None]:
    if metric_col not in history.columns:
        return None, None
    metric = history.loc[(history["step"] <= decision_step) & history[metric_col].notna(), ["step", metric_col]]
    if metric.empty:
        return None, None
    row = metric.sort_values("step").iloc[-1]
    return float(row[metric_col]), int(row["step"])


def _mean_metric_between(
    history: pd.DataFrame,
    metric_col: str,
    start_step: int,
    end_step: int,
) -> float | None:
    if metric_col not in history.columns:
        return None
    metric = history.loc[
        (history["step"] > start_step) & (history["step"] <= end_step) & history[metric_col].notna(),
        metric_col,
    ]
    if metric.empty:
        return None
    return float(metric.mean())


def _compute_exposure(actions: list[float], phase_lengths: list[int], decision_index: int) -> tuple[float, float, float]:
    if decision_index <= 0:
        return 0.5, 0.0, 0.0
    if len(actions) < decision_index:
        raise ValueError(
            f"Need {decision_index} prior actions to build state for decision {decision_index}, found {len(actions)}."
        )
    consumed = sum(phase_lengths[:decision_index])
    weighted = sum(actions[idx] * phase_lengths[idx] for idx in range(decision_index))
    prev_action = actions[decision_index - 1]
    prev_prev = actions[decision_index - 2] if decision_index > 1 else prev_action
    cumulative = float(weighted / consumed) if consumed > 0 else 0.0
    delta_prev = float(prev_action - prev_prev) if decision_index > 1 else 0.0
    return float(prev_action), cumulative, delta_prev


def _is_v3_dynamic_q_artifact(artifact: AnyPolicyArtifact) -> bool:
    return isinstance(artifact, PolicyArtifactV2) and artifact.kind == "sklearn_dynamic_q_planner_v3"


def _is_v3_sequence_artifact(artifact: AnyPolicyArtifact) -> bool:
    return isinstance(artifact, PolicyArtifactV2) and artifact.kind in {"torch_gru_q_v3", "torch_transformer_q_v3"}


def _is_v3_artifact(artifact: AnyPolicyArtifact) -> bool:
    return _is_v3_dynamic_q_artifact(artifact) or _is_v3_sequence_artifact(artifact)


def _history_from_completed_run(wb_run, metric_keys: tuple[str, ...]) -> pd.DataFrame:
    long_rows = collect_history_long_rows_batched(
        wb_run,
        metric_keys=metric_keys,
        history_batch_size=ONLINE_HISTORY_BATCH_SIZE,
        retry_attempts=ONLINE_HISTORY_RETRY_ATTEMPTS,
        backoff_seconds=ONLINE_HISTORY_BACKOFF_SECONDS,
    )
    long_df = dedupe_history_rows(pd.DataFrame(long_rows))
    return build_wide_history(long_df).sort_values("step").reset_index(drop=True)


def _dense_policy_input_from_history(
    *,
    artifact_path: str,
    history: pd.DataFrame,
    phase_index: int,
    objective_metric: str,
    total_steps: int,
    prior_actions: list[float],
    phase_end_steps: tuple[int, ...],
    num_phases_total: int = 3,
) -> DecisionBatch:
    artifact = _load_policy_artifact_cached(artifact_path)
    if not isinstance(artifact, PolicyArtifactV2):
        raise TypeError(f"Dense policy input requires a v2/v3 artifact, found {type(artifact)!r}.")
    padded_actions = list(prior_actions) + [prior_actions[-1]] * max(0, num_phases_total - len(prior_actions))
    run_row = pd.Series(
        {
            "wandb_run_id": "online_rollout",
            "source_experiment": "online_rollout",
            "local_run_id": None,
            "run_name": "online_rollout",
            "total_steps": total_steps,
            "phase_boundaries_json": json.dumps(list(phase_end_steps)),
            **{f"phase_{idx}_starcoder": float(padded_actions[idx]) for idx in range(num_phases_total)},
        }
    )
    built = build_decision_state_features(
        run_row=run_row,
        history=history,
        objective_metric=objective_metric,
        decision_index=phase_index,
    )
    if built is None:
        raise ValueError(f"Could not build dense policy input for decision {phase_index}")
    state_features, window, mask = built
    defaults = _artifact_state_defaults(artifact)
    defaults.update(
        {
            key: float(value)
            for key, value in _decision_state_default(artifact_path, decision_index=phase_index).items()
            if key in artifact.state_keys
        }
    )
    defaults.update({key: float(value) for key, value in state_features.items() if key in artifact.state_keys})
    frame = pd.DataFrame([{key: float(defaults[key]) for key in artifact.state_keys}], columns=list(artifact.state_keys))
    return DecisionBatch(
        frame=frame, sequences=window[None, ...].astype(np.float32), masks=mask[None, ...].astype(np.float32)
    )


def _initial_policy_input(
    artifact_path: str,
    *,
    num_phases_total: int = 3,
    total_steps: int = DEFAULT_TOTAL_STEPS,
    phase_end_steps: tuple[int, ...] = DEFAULT_PHASE_END_STEPS,
    run_family: str | None = None,
) -> dict[str, float] | DecisionBatch:
    artifact = _load_policy_artifact_cached(artifact_path)
    state = _initial_policy_state(
        artifact_path,
        num_phases_total=num_phases_total,
        total_steps=total_steps,
        phase_end_steps=phase_end_steps,
        run_family=run_family,
    )
    if _is_v3_artifact(artifact):
        frame = pd.DataFrame(
            [{key: float(state[key]) for key in artifact.state_keys}], columns=list(artifact.state_keys)
        )
        sequences = np.zeros((1, 32, len(SEQUENCE_CHANNELS)), dtype=np.float32)
        masks = np.zeros((1, 32), dtype=np.float32)
        return DecisionBatch(frame=frame, sequences=sequences, masks=masks)
    return state


def _state_from_completed_run(
    wb_run,
    *,
    phase_index: int,
    decision_step: int,
    prev_action: float,
    objective_metric: str,
    total_steps: int,
    artifact_path: str,
    prior_actions: list[float],
    phase_end_steps: tuple[int, ...],
    num_phases_total: int = 3,
    run_family: str | None = None,
) -> dict[str, float] | DecisionBatch:
    artifact = _load_policy_artifact_cached(artifact_path)
    if _is_v3_artifact(artifact):
        history = collect_dense_history_from_run(
            wb_run,
            objective_metric=objective_metric,
            retry_attempts=ONLINE_HISTORY_RETRY_ATTEMPTS,
            backoff_seconds=ONLINE_HISTORY_BACKOFF_SECONDS,
        )
        return _dense_policy_input_from_history(
            artifact_path=artifact_path,
            history=history,
            phase_index=phase_index,
            objective_metric=objective_metric,
            total_steps=total_steps,
            prior_actions=prior_actions,
            phase_end_steps=phase_end_steps,
            num_phases_total=num_phases_total,
        )

    if isinstance(artifact, PolicyArtifactV2):
        history = _history_from_completed_run(
            wb_run,
            metric_keys=(
                "train/loss",
                "eval/loss",
                objective_metric,
                "optim/learning_rate",
                "optim/adam_lr",
                "grad/norm/total",
            ),
        )
        defaults = _artifact_state_defaults(artifact)
        defaults.update(
            {
                key: float(value)
                for key, value in _decision_state_default(
                    artifact_path,
                    decision_index=phase_index,
                    run_family=run_family,
                ).items()
                if key in artifact.state_keys
            }
        )
        phase_lengths = _phase_lengths(total_steps, phase_end_steps)
        decision_steps = _decision_steps(phase_end_steps)
        next_step = decision_steps[phase_index + 1] if phase_index < num_phases_total - 1 else total_steps
        prev_action_value, cumulative_exposure, delta_prev_action = _compute_exposure(
            prior_actions,
            phase_lengths,
            phase_index,
        )

        last_train_loss, _ = _last_metric_at_or_before(history, "train/loss", decision_step)
        last_eval_loss, eval_step = _last_metric_at_or_before(history, "eval/loss", decision_step)
        last_obj_bpb, obj_step = _last_metric_at_or_before(history, objective_metric, decision_step)
        last_lr, _ = _last_metric_at_or_before(history, "optim/learning_rate", decision_step)
        last_adam_lr, _ = _last_metric_at_or_before(history, "optim/adam_lr", decision_step)
        last_grad_norm, _ = _last_metric_at_or_before(history, "grad/norm/total", decision_step)
        avg_lr_to_next_boundary = _mean_metric_between(history, "optim/learning_rate", decision_step, next_step)
        avg_lr_remaining = _mean_metric_between(history, "optim/learning_rate", decision_step, total_steps)

        if any(step is not None for step in (eval_step, obj_step)):
            last_eval_like_step = max(step for step in (eval_step, obj_step) if step is not None)
            steps_since_last_eval_frac = (decision_step - last_eval_like_step) / float(total_steps)
        else:
            steps_since_last_eval_frac = 1.0

        state = defaults.copy()
        state.update(
            {
                "decision_index": float(phase_index),
                "num_phases_total": float(num_phases_total),
                "remaining_decisions": float(max(0, num_phases_total - phase_index - 1)),
                "budget_frac_consumed": float(decision_step) / float(total_steps),
                "budget_frac_remaining": 1.0 - (float(decision_step) / float(total_steps)),
                "last_train_loss": (
                    defaults.get("last_train_loss", 0.0) if last_train_loss is None else float(last_train_loss)
                ),
                "last_eval_loss": (
                    defaults.get("last_eval_loss", 0.0) if last_eval_loss is None else float(last_eval_loss)
                ),
                "last_obj_bpb": defaults.get("last_obj_bpb", 0.0) if last_obj_bpb is None else float(last_obj_bpb),
                "train_eval_gap": (
                    defaults.get("train_eval_gap", 0.0)
                    if last_train_loss is None or last_eval_loss is None
                    else float(last_eval_loss - last_train_loss)
                ),
                "global_step": float(decision_step),
                "steps_since_last_eval_frac": float(steps_since_last_eval_frac),
                "optim/learning_rate": defaults.get("optim/learning_rate", 0.0) if last_lr is None else float(last_lr),
                "optim/adam_lr": defaults.get("optim/adam_lr", 0.0) if last_adam_lr is None else float(last_adam_lr),
                "avg_lr_to_next_boundary": (
                    defaults.get("avg_lr_to_next_boundary", 0.0)
                    if avg_lr_to_next_boundary is None
                    else float(avg_lr_to_next_boundary)
                ),
                "avg_lr_remaining": (
                    defaults.get("avg_lr_remaining", 0.0) if avg_lr_remaining is None else float(avg_lr_remaining)
                ),
                "grad/norm/total": (
                    defaults.get("grad/norm/total", 0.0) if last_grad_norm is None else float(last_grad_norm)
                ),
                "prev_action_starcoder": float(prev_action_value),
                "cumulative_starcoder_exposure": float(cumulative_exposure),
                "delta_prev_action": float(delta_prev_action),
            }
        )
        return {key: float(state[key]) for key in artifact.state_keys}

    long_rows = collect_history_long_rows_batched(
        wb_run,
        metric_keys=("train/loss", "eval/loss", objective_metric, "throughput/total_tokens"),
        history_batch_size=ONLINE_HISTORY_BATCH_SIZE,
        retry_attempts=ONLINE_HISTORY_RETRY_ATTEMPTS,
        backoff_seconds=ONLINE_HISTORY_BACKOFF_SECONDS,
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


def _policy_predict_action(
    artifact_path: str,
    state: dict[str, float] | DecisionBatch,
    device: str = "cpu",
) -> float:
    artifact = _load_policy_artifact_cached(artifact_path)
    if isinstance(state, DecisionBatch):
        batch = state
        if len(batch.frame) != 1:
            raise ValueError("Online policy inference expects a single-row DecisionBatch.")
        state_with_defaults = _artifact_state_defaults(artifact)
        state_with_defaults.update({key: float(value) for key, value in batch.frame.iloc[0].to_dict().items()})
    else:
        batch = None
        state_with_defaults = _artifact_state_defaults(artifact)
        state_with_defaults.update({key: float(value) for key, value in state.items()})

    if isinstance(artifact, PolicyArtifactV2) and artifact.kind == "sklearn_outcome_planner_v2":
        bundle = _load_outcome_planner_bundle(artifact_path)
        planner_state = pd.DataFrame(
            [{key: float(state_with_defaults[key]) for key in bundle["feature_keys"]}],
            columns=list(bundle["feature_keys"]),
        )
        action_grid = np.asarray(bundle["action_grid"], dtype=np.float32)
        tiled = pd.concat([planner_state] * len(action_grid), ignore_index=True)
        support_matrix = bundle["behavior_policy"].predict_proba(tiled)
        support = support_matrix[np.arange(len(action_grid)), np.arange(len(action_grid))]
        support_threshold = float(bundle.get("support_threshold", artifact.metadata.get("support_threshold", 0.0)))
        supported = np.where(support >= support_threshold)[0]
        candidate_indices = supported if len(supported) > 0 else np.where(support == np.max(support))[0]

        if "q_models" in bundle:
            decision_index = round(float(planner_state.iloc[0]["decision_index"]))
            q_model = bundle["q_models"][decision_index]
            q_inputs = np.concatenate(
                [
                    tiled.loc[:, list(bundle["feature_keys"])].to_numpy(dtype=np.float32),
                    action_grid.reshape(-1, 1).astype(np.float32),
                ],
                axis=1,
            )
            scores = q_model.predict(q_inputs).astype(np.float32)
            best_local = int(candidate_indices[np.argmax(scores[candidate_indices])])
            return float(action_grid[best_local])

        reward_models = bundle["reward_models"]
        scores = reward_models.blended_stage_scores(tiled, action_grid)
        best_local = int(candidate_indices[np.argmin(scores[candidate_indices])])
        return float(action_grid[best_local])

    if _is_v3_dynamic_q_artifact(artifact):
        bundle = _load_dynamic_q_v3_bundle(artifact_path)
        if batch is None:
            planner_state = pd.DataFrame(
                [{key: float(state_with_defaults[key]) for key in bundle["feature_keys"]}],
                columns=list(bundle["feature_keys"]),
            )
        else:
            planner_state = batch.frame.loc[:, list(bundle["feature_keys"])].reset_index(drop=True)
        if len(planner_state) != 1:
            raise ValueError("Dynamic-Q rollout inference expects exactly one state row.")
        action_grid = np.asarray(bundle["action_grid"], dtype=np.float32)
        tiled = pd.concat([planner_state] * len(action_grid), ignore_index=True)
        support_matrix = bundle["behavior_policy"].predict_proba(tiled)
        support = support_matrix[np.arange(len(action_grid)), np.arange(len(action_grid))]
        scores = (
            bundle["q_models"][round(float(planner_state.iloc[0]["decision_index"]))]
            .predict(
                np.concatenate(
                    [
                        tiled.loc[:, list(bundle["feature_keys"])].to_numpy(dtype=np.float32),
                        action_grid.reshape(-1, 1).astype(np.float32),
                    ],
                    axis=1,
                )
            )
            .astype(np.float32)
        )
        score = scores + float(bundle["support_lambda"]) * np.log(np.maximum(support, float(bundle["support_floor"])))
        valid = support >= float(bundle["support_floor"])
        if not valid.any():
            best_local = int(np.argmax(support))
        else:
            score = np.where(valid, score, -np.inf)
            best_local = int(np.argmax(score))
        return float(action_grid[best_local])

    if isinstance(artifact, PolicyArtifactV2) and artifact.kind == "torch_discrete_iql_v2":
        policy_net = _load_torch_discrete_iql_policy(artifact_path, device)
        normalized = normalize_state(state_with_defaults, artifact).reshape(1, -1)
        inputs = torch.tensor(normalized, dtype=torch.float32, device=device)
        with torch.no_grad():
            action_idx = int(torch.argmax(policy_net.forward(inputs), dim=1).item())
        return float(artifact.action_values[action_idx])

    if isinstance(artifact, PolicyArtifactV2) and artifact.kind in {"torch_gru_q_v3", "torch_transformer_q_v3"}:
        if batch is None or batch.sequences is None or batch.masks is None:
            raise ValueError(f"{artifact.kind} rollout inference requires a sequence-aware DecisionBatch.")
        state_mean = np.asarray(artifact.state_mean, dtype=np.float32)
        state_std = np.asarray(artifact.state_std, dtype=np.float32)
        safe_std = np.where(state_std <= 0.0, 1.0, state_std)
        summary = (batch.frame.loc[:, list(artifact.state_keys)].to_numpy(dtype=np.float32) - state_mean) / safe_std
        summary_tensor = torch.tensor(summary, dtype=torch.float32, device=device)
        sequence_tensor = torch.tensor(batch.sequences, dtype=torch.float32, device=device)
        mask_tensor = torch.tensor(batch.masks, dtype=torch.float32, device=device)
        model = (
            _load_gru_q_v3_policy(artifact_path, device)
            if artifact.kind == "torch_gru_q_v3"
            else _load_transformer_q_v3_policy(artifact_path, device)
        )
        support_bundle = _load_sequence_support_bundle(artifact_path)
        with torch.no_grad():
            q_values, _, _, _ = model(summary_tensor, sequence_tensor, mask_tensor)
        support = support_bundle["behavior_policy"].predict_proba(batch.frame.loc[:, list(artifact.state_keys)])
        support_tensor = torch.tensor(support, dtype=torch.float32, device=device)
        score = q_values + float(support_bundle["support_lambda"]) * torch.log(
            torch.clamp(support_tensor, min=float(support_bundle["support_floor"]))
        )
        score = torch.where(
            support_tensor >= float(support_bundle["support_floor"]),
            score,
            torch.full_like(score, float("-inf")),
        )
        if (~torch.isfinite(score)).all(dim=1).any():
            action_idx = int(torch.argmax(support_tensor, dim=1)[0].item())
        else:
            action_idx = int(torch.argmax(score, dim=1)[0].item())
        return float(artifact.action_values[action_idx])

    normalized = normalize_state(state_with_defaults, artifact).reshape(1, -1)
    model_path = _resolve_policy_model_path(artifact_path, artifact.model_path)
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
    if isinstance(artifact, PolicyArtifactV2) and artifact.action_values:
        index = int(np.clip(round(raw), 0, len(artifact.action_values) - 1))
        return float(artifact.action_values[index])
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

    if marin_region() is None:
        logger.warning("Could not resolve VM region metadata; forcing dry-run policy evaluation.")
        return True
    return False


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
    artifact = _load_policy_artifact_cached(config.policy_artifact_path)

    rows: list[dict] = []
    for replicate_idx in range(config.n_replicates):
        base_slug = re.sub(r"[^a-zA-Z0-9]+", "", config.run_name_prefix.rsplit("/", maxsplit=1)[-1].lower())
        base_slug = (base_slug[:12] or "tsceval").strip("_")
        run_namespace = f"pinlin_calvin_xu/data_mixture/{base_slug}_{job_token}_r{replicate_idx:02d}"
        replicate_data_seed = _replicate_data_seed(replicate_idx)
        actions = []
        run_ids = []

        policy_input = _initial_policy_input(
            config.policy_artifact_path,
            num_phases_total=3,
            total_steps=config.total_steps,
            phase_end_steps=config.phase_end_steps,
            run_family="three_phase_starcoder",
        )

        checkpoint_path: str | None = None
        final_metric = None
        replicate_force_dry = force_dry_run

        for phase_idx in range(3):
            action = _policy_predict_action(config.policy_artifact_path, policy_input)
            actions.append(action)

            phase_plan = build_phase_train_plan(
                phase_idx,
                checkpoint_path=checkpoint_path,
                phase_end_steps=config.phase_end_steps,
                total_steps=config.total_steps,
            )
            # Marin infers trainer.id (and therefore WandB run id) from the output path basename.
            # Using a static basename like "phase_0" causes run-id collisions and hidden logging.
            phase_output_basename = f"phase_{phase_idx}_{job_token}_r{replicate_idx:02d}"
            relative_output_path = (
                f"domain_phase_mix/offline_rl/policy_eval/{job_token}/rep_{replicate_idx:02d}/{phase_output_basename}"
            )
            training_step = _build_training_step(
                run_namespace=run_namespace,
                phase_plan=phase_plan,
                actions_so_far=actions,
                override_output_path=relative_output_path,
                run_id=_phase_run_id(replicate_idx, phase_idx),
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
                    checkpoint_path = _discover_exact_checkpoint(output_path, phase_plan.cumulative_steps - 1)

                    display_name = f"{run_namespace}/phase_{phase_idx}"
                    wb_run = _fetch_wandb_run_by_display_name(
                        config.wandb_entity,
                        config.wandb_project,
                        display_name=display_name,
                    )
                    run_ids.append(wb_run.id)

                    if phase_idx < 2:
                        decision_step = config.phase_end_steps[phase_idx]
                        policy_input = _state_from_completed_run(
                            wb_run,
                            phase_index=phase_idx + 1,
                            decision_step=decision_step,
                            prev_action=action,
                            objective_metric=objective_metric,
                            total_steps=config.total_steps,
                            artifact_path=config.policy_artifact_path,
                            prior_actions=actions,
                            phase_end_steps=config.phase_end_steps,
                            num_phases_total=3,
                            run_family="three_phase_starcoder",
                        )
                    else:
                        summary_value = wb_run.summary.get(objective_metric)
                        final_metric = float(summary_value) if isinstance(summary_value, int | float) else None

            if replicate_force_dry:
                run_ids.append(f"dry_run_rep{replicate_idx}_phase{phase_idx}")
                checkpoint_path = f"dry://rep{replicate_idx}/phase{phase_idx}/checkpoint"
                synthetic_state = _artifact_state_defaults(artifact)
                if isinstance(artifact, PolicyArtifactV2):
                    decision_index = float(min(phase_idx + 1, 2))
                    budget_frac_consumed = (phase_idx + 1) / 3.0
                    global_step = (
                        float(config.phase_end_steps[min(phase_idx, 1)]) if phase_idx < 2 else float(config.total_steps)
                    )
                    synthetic_state.update(
                        {
                            "decision_index": decision_index,
                            "num_phases_total": 3.0,
                            "remaining_decisions": float(max(0, 2 - (phase_idx + 1))),
                            "budget_frac_consumed": budget_frac_consumed,
                            "budget_frac_remaining": 1.0 - budget_frac_consumed,
                            "global_step": global_step,
                            "steps_since_last_eval_frac": 0.0,
                            "prev_action_starcoder": action,
                            "cumulative_starcoder_exposure": float(np.mean(actions)),
                            "delta_prev_action": float(actions[-1] - actions[-2]) if len(actions) > 1 else 0.0,
                        }
                    )
                    if _is_v3_artifact(artifact):
                        frame = pd.DataFrame(
                            [{key: float(synthetic_state[key]) for key in artifact.state_keys}],
                            columns=list(artifact.state_keys),
                        )
                        policy_input = DecisionBatch(
                            frame=frame,
                            sequences=np.zeros((1, 32, len(SEQUENCE_CHANNELS)), dtype=np.float32),
                            masks=np.zeros((1, 32), dtype=np.float32),
                        )
                    else:
                        policy_input = synthetic_state
                else:
                    synthetic_state.update(
                        {
                            "phase_index": float(min(phase_idx + 1, 2)),
                            "last_train_loss": 0.0,
                            "last_eval_loss": 0.0,
                            "last_obj_bpb": 0.0,
                            "tokens_frac": (phase_idx + 1) / 3.0,
                            "steps_since_last_eval_frac": 0.0,
                            "prev_action_starcoder": action,
                        }
                    )
                    policy_input = synthetic_state
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
