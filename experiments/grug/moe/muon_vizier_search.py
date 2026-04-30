# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Vizier hyperparameter search for Grug MoE Muon at exact gate-1 budgets."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import sqlite3
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import fsspec
from fray.cluster import ResourceConfig
from levanter.optim.util import CoefficientType
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig, NEMOTRON_MIX_WITH_DEFAULT_VALIDATION, run_grug_moe_trial
from experiments.grug.moe.optimizer import GrugMoeMuonConfig, GrugMoeMuonHConfig, build_grug_moe_muon_config
from experiments.grug.moe.optimizer import build_grug_moe_muonh_config
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

logger = logging.getLogger(__name__)

FloatRange = tuple[float, float]
SearchOptimizerConfig = GrugMoeMuonConfig | GrugMoeMuonHConfig
SUGGESTIONS_FILENAME = "vizier_suggestions.json"
UPDATE_FILENAME = "vizier_update.json"
RESOURCE_FILENAME = "vizier_resource.json"
OPTIMAL_FILENAME = "vizier_optimal.json"
VIZIER_DB_FILENAME = "vizier.db"


@dataclass(frozen=True)
class MuonSearchScale:
    name: str
    budget: float
    hidden_dim: int


@dataclass(frozen=True)
class MuonSearchSettings:
    experiment_name: str
    study_owner: str
    num_loops: int
    suggestions_per_loop: int
    search_space: Mapping[str, FloatRange]
    scales: tuple[MuonSearchScale, ...]
    target_steps: int
    seq_len: int
    metric_file: str
    metric_key: str
    metric_mode: str
    vizier_algorithm: str
    coefficient_type: CoefficientType
    base_train_tags: tuple[str, ...]
    batch_multiplier: int = 1
    optimizer_family: str = "muon"
    matrix_lr_multiplier_name: str = "muon_lr_multiplier"
    split_moe_gate_up_for_ortho: bool = False

    def study_id(self, scale: MuonSearchScale) -> str:
        return f"{self.experiment_name}-{scale.name}"

    def study_resource_name(self, scale: MuonSearchScale) -> str:
        return f"owners/{self.study_owner}/studies/{self.study_id(scale)}"


SWEEP = MuonSearchSettings(
    experiment_name="moe-muon-vizier-lr-beta-r2",
    study_owner="marin",
    num_loops=3,
    suggestions_per_loop=4,
    search_space={
        "muon_lr_multiplier": (0.5, 3.0),
        "adam_lr_multiplier": (0.5, 3.0),
        "momentum": (0.92, 0.99),
        "beta1": (0.70, 0.95),
        "beta2": (0.95, 0.999),
    },
    scales=(
        MuonSearchScale("d512-gate1", 2.19e17, 512),
        MuonSearchScale("d768-gate1", 1.70e18, 768),
    ),
    target_steps=2**14,
    seq_len=4096,
    metric_file="tracker_metrics.jsonl",
    metric_key="eval/paloma/macro_loss",
    metric_mode="min",
    vizier_algorithm="DEFAULT",
    coefficient_type="aol",
    base_train_tags=("moe", "muon", "aol", "vizier", "gate1-followup"),
)


@dataclass(frozen=True)
class VizierSuggestConfig:
    study_owner: str
    study_id: str
    input_db_path: str | None
    output_path: str
    num_suggestions: int
    client_id: str
    metric_key: str
    mode: str
    algorithm: str
    search_space: Mapping[str, FloatRange]


@dataclass(frozen=True)
class VizierTrainConfig:
    suggestions_path: str
    suggestion_index: int
    base_launch_config: GrugMoeLaunchConfig
    base_optimizer: SearchOptimizerConfig
    scale_name: str
    loop_index: int
    experiment_name: str
    matrix_lr_multiplier_name: str


@dataclass(frozen=True)
class VizierUpdateConfig:
    study_id: str
    study_resource_name: str
    input_db_path: str | None
    suggestions_path: str
    run_paths: list[str]
    metric_file: str
    metric_key: str
    mode: str
    output_path: str


@dataclass(frozen=True)
class VizierOptimalConfig:
    study_id: str
    study_resource_name: str
    input_db_path: str
    output_path: str


def _local_vizier_db_path(study_id: str) -> str:
    safe_study = re.sub(r"[^A-Za-z0-9_.-]+", "_", study_id)
    return os.path.join(tempfile.gettempdir(), f"vizier-{safe_study}.db")


def _configure_vizier_local_db(local_path: str) -> None:
    from vizier.service import clients

    clients.environment_variables.servicer_kwargs["database_url"] = f"sqlite:///{local_path}"


def _sqlite_sidecar_paths(path: str) -> tuple[str, ...]:
    return (f"{path}-wal", f"{path}-shm", f"{path}-journal")


def _remove_sqlite_sidecars(path: str) -> None:
    for sidecar_path in _sqlite_sidecar_paths(path):
        if os.path.exists(sidecar_path):
            os.remove(sidecar_path)


def _checkpoint_sqlite_db(path: str) -> None:
    if not os.path.exists(path):
        return
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA wal_checkpoint(FULL);")


def _sync_vizier_db_from_gcs(path: str | None, local_path: str) -> bool:
    if not path:
        return False
    fs, _, _ = fsspec.get_fs_token_paths(path)
    if not fs.exists(path):
        return False
    _remove_sqlite_sidecars(local_path)
    with fs.open(path, "rb") as src, open(local_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return True


def _sync_vizier_db_to_gcs(local_path: str, path: str) -> None:
    _checkpoint_sqlite_db(local_path)
    _remove_sqlite_sidecars(local_path)
    fs, _, _ = fsspec.get_fs_token_paths(path)
    fs.makedirs(os.path.dirname(path), exist_ok=True)
    with open(local_path, "rb") as src, fs.open(path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    for sidecar_suffix in ("-wal", "-shm", "-journal"):
        sidecar_path = f"{path}{sidecar_suffix}"
        if fs.exists(sidecar_path):
            fs.rm(sidecar_path)


def _load_suggestions(path: str) -> dict:
    fs, _, _ = fsspec.get_fs_token_paths(path)
    with fs.open(path, "r") as f:
        data = json.load(f)
    if "suggestions" not in data:
        raise ValueError(f"Missing 'suggestions' in {path}")
    return data


def _serialize_parameters(parameters: Mapping[str, object]) -> dict[str, float | int]:
    serialized: dict[str, float | int] = {}
    for key, value in parameters.items():
        raw_value = value.value if hasattr(value, "value") else value
        if isinstance(raw_value, bool):
            serialized[key] = int(raw_value)
        elif isinstance(raw_value, int):
            serialized[key] = raw_value
        elif isinstance(raw_value, float):
            serialized[key] = raw_value
        else:
            try:
                serialized[key] = float(raw_value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Unsupported parameter value for {key!r}: {raw_value!r}") from e
    return serialized


def _metric_goal(mode: str) -> Any:
    from vizier.service import pyvizier as vz

    if mode == "min":
        return vz.ObjectiveMetricGoal.MINIMIZE
    if mode == "max":
        return vz.ObjectiveMetricGoal.MAXIMIZE
    raise ValueError(f"Unsupported metric mode: {mode}")


def _extract_muon_hparams(suggestion: dict[str, object], matrix_lr_multiplier_name: str) -> dict[str, float]:
    parameters = suggestion["parameters"]
    if not isinstance(parameters, Mapping):
        raise ValueError(f"Expected suggestion parameters mapping, got {type(parameters)!r}")
    required = (matrix_lr_multiplier_name, "adam_lr_multiplier", "momentum", "beta1", "beta2")
    hparams = {name: float(parameters[name]) for name in required}
    if "warmup" in parameters:
        hparams["warmup"] = float(parameters["warmup"])
    return hparams


def _trial_optimizer(
    base_optimizer: SearchOptimizerConfig,
    hparams: dict[str, float],
    matrix_lr_multiplier_name: str,
) -> SearchOptimizerConfig:
    optimizer_updates = {
        "learning_rate": base_optimizer.learning_rate * hparams[matrix_lr_multiplier_name],
        "adam_lr": base_optimizer.adam_lr * hparams["adam_lr_multiplier"],
        "momentum": hparams["momentum"],
        "beta1": hparams["beta1"],
        "beta2": hparams["beta2"],
    }
    if "warmup" in hparams:
        optimizer_updates["warmup"] = hparams["warmup"]
    return replace(base_optimizer, **optimizer_updates)


def _build_base_launch_config(scale: MuonSearchScale) -> tuple[GrugMoeLaunchConfig, SearchOptimizerConfig]:
    model_cfg, adamh_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=scale.budget,
        hidden_dim=scale.hidden_dim,
        target_steps=SWEEP.target_steps,
    )
    if SWEEP.split_moe_gate_up_for_ortho:
        model_cfg = replace(model_cfg, split_moe_gate_up_for_ortho=True)
    if SWEEP.batch_multiplier < 1:
        raise ValueError(f"batch_multiplier must be >= 1, got {SWEEP.batch_multiplier}")
    baseline_tokens = num_steps * batch_size * SWEEP.seq_len
    batch_size *= SWEEP.batch_multiplier
    num_steps = max(1, round(baseline_tokens / (batch_size * SWEEP.seq_len)))

    if SWEEP.optimizer_family == "muon":
        optimizer_cfg = build_grug_moe_muon_config(hidden_dim=scale.hidden_dim, coefficient_type=SWEEP.coefficient_type)
    elif SWEEP.optimizer_family == "muonh":
        optimizer_cfg = build_grug_moe_muonh_config(adamh_optimizer, coefficient_type=SWEEP.coefficient_type)
    else:
        raise ValueError(f"Unsupported optimizer_family: {SWEEP.optimizer_family}")

    launch_config = GrugMoeLaunchConfig(
        model=model_cfg,
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=f"{SWEEP.experiment_name}-{scale.name}-base",
        resources=ResourceConfig.with_tpu("v5p-8"),
        steps=num_steps,
        batch_size=batch_size,
        seed=0,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=WandbConfig(
            project="marin_moe",
            tags=list(SWEEP.base_train_tags),
            group=f"{SWEEP.experiment_name}-{scale.name}",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=optimizer_cfg,
        grug_trainer=GrugTrainerConfig(
            z_loss_weight=1e-4,
            ema_beta=None,
            log_every=1,
        ),
        eval=GrugEvalConfig(
            eval_batch_size=512,
            steps_per_eval=1000,
            max_eval_batches=8,
            eval_current=True,
            eval_ema=False,
        ),
    )
    return launch_config, optimizer_cfg


def _best_run(runs: list[dict], mode: str) -> dict | None:
    feasible = [r for r in runs if r.get("feasible", True)]
    if not feasible:
        return None
    return min(feasible, key=lambda r: r["metric"]) if mode == "min" else max(feasible, key=lambda r: r["metric"])


def run_vizier_suggest(config: VizierSuggestConfig) -> None:
    """Create or load a Vizier study, suggest trials, and persist the study DB."""
    from vizier.service import clients
    from vizier.service import pyvizier as vz

    local_db_path = _local_vizier_db_path(config.study_id)
    output_db_path = os.path.join(config.output_path, VIZIER_DB_FILENAME)
    if not _sync_vizier_db_from_gcs(output_db_path, local_db_path):
        _sync_vizier_db_from_gcs(config.input_db_path, local_db_path)
    _configure_vizier_local_db(local_db_path)

    study_config = vz.StudyConfig(algorithm=config.algorithm)
    root = study_config.search_space.root
    for parameter_name, parameter_range in config.search_space.items():
        root.add_float_param(parameter_name, *parameter_range)
    study_config.metric_information.append(vz.MetricInformation(config.metric_key, goal=_metric_goal(config.mode)))

    study = clients.Study.from_study_config(study_config, owner=config.study_owner, study_id=config.study_id)
    suggestions = study.suggest(count=config.num_suggestions, client_id=config.client_id)
    output = {
        "study_resource_name": study.resource_name,
        "client_id": config.client_id,
        "suggestions": [
            {"trial_id": trial.id, "parameters": _serialize_parameters(trial.parameters)} for trial in suggestions
        ],
    }

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(os.path.join(config.output_path, SUGGESTIONS_FILENAME), "w") as f:
        json.dump(output, f, indent=2)

    _sync_vizier_db_to_gcs(local_db_path, output_db_path)


def run_vizier_train(config: VizierTrainConfig) -> None:
    """Train a Grug MoE Muon trial for a single Vizier suggestion."""
    suggestions = _load_suggestions(config.suggestions_path)["suggestions"]
    if config.suggestion_index >= len(suggestions):
        raise IndexError(f"Suggestion index {config.suggestion_index} out of range")

    suggestion = suggestions[config.suggestion_index]
    hparams = _extract_muon_hparams(suggestion, config.matrix_lr_multiplier_name)
    trial_id = int(suggestion["trial_id"])
    base = config.base_launch_config

    tags = list(getattr(base.tracker, "tags", []) or [])
    tags.extend(
        [
            f"{config.matrix_lr_multiplier_name}={hparams[config.matrix_lr_multiplier_name]}",
            f"adam_lr_mult={hparams['adam_lr_multiplier']}",
            f"momentum={hparams['momentum']}",
            f"beta1={hparams['beta1']}",
            f"beta2={hparams['beta2']}",
            f"trial={trial_id}",
            f"loop={config.loop_index}",
            f"scale={config.scale_name}",
        ]
    )
    if "warmup" in hparams:
        tags.append(f"warmup={hparams['warmup']}")

    launch_config = replace(
        base,
        run_id=f"{config.experiment_name}-{config.scale_name}-loop{config.loop_index}-trial{trial_id}",
        tracker=replace(base.tracker, tags=tags, name=None),
        optimizer=_trial_optimizer(config.base_optimizer, hparams, config.matrix_lr_multiplier_name),
    )
    run_grug_moe_trial(launch_config)


def run_vizier_update(config: VizierUpdateConfig) -> None:
    """Load trial results, update Vizier, and write summary output."""
    from vizier.service import clients
    from vizier.service import pyvizier as vz

    local_db_path = _local_vizier_db_path(config.study_id)
    if not config.input_db_path:
        raise ValueError("input_db_path is required for run_vizier_update")
    if not _sync_vizier_db_from_gcs(config.input_db_path, local_db_path):
        raise FileNotFoundError(f"Could not load Vizier DB from input path: {config.input_db_path}")

    output_db_path = os.path.join(config.output_path, VIZIER_DB_FILENAME)
    _configure_vizier_local_db(local_db_path)

    study = clients.Study.from_resource_name(config.study_resource_name)
    suggestions = _load_suggestions(config.suggestions_path)["suggestions"]
    if len(suggestions) != len(config.run_paths):
        raise ValueError(f"Expected {len(suggestions)} run paths but got {len(config.run_paths)}")

    results = []
    for run_path, suggestion in zip(config.run_paths, suggestions, strict=True):
        metric_path = os.path.join(run_path, config.metric_file)
        fs, _, _ = fsspec.get_fs_token_paths(metric_path)
        with fs.open(metric_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            if not lines:
                raise RuntimeError(f"No metrics found at {metric_path}")
            data = json.loads(lines[-1])

        value = float(data["summary"][config.metric_key])
        trial_id = int(suggestion["trial_id"])
        trial = study.get_trial(trial_id)

        if trial.materialize().status == vz.TrialStatus.COMPLETED:
            logger.info("Trial %s already completed, skipping", trial_id)
        elif math.isnan(value) or math.isinf(value):
            trial.complete(infeasible_reason=f"metric is {value}")
        else:
            trial.complete(vz.Measurement({config.metric_key: value}))

        feasible = math.isfinite(value)
        results.append(
            {
                "trial_id": trial_id,
                "metric": value if feasible else None,
                "feasible": feasible,
                "hparams": suggestion["parameters"],
                "run_path": run_path,
            }
        )

    best = _best_run(results, config.mode)
    if best is None:
        raise RuntimeError(f"All {len(results)} trials were infeasible")

    def _sort_key(result: dict) -> tuple[bool, float]:
        metric = result["metric"] or 0.0
        return (not result["feasible"], metric if config.mode == "min" else -metric)

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    output = {
        "study_resource_name": config.study_resource_name,
        "best_hparams": best["hparams"],
        "best_metric": best["metric"],
        "best_run_path": best["run_path"],
        "all_results": sorted(results, key=_sort_key),
    }
    with fs.open(os.path.join(config.output_path, UPDATE_FILENAME), "w") as f:
        json.dump(output, f, indent=2)
    with fs.open(os.path.join(config.output_path, RESOURCE_FILENAME), "w") as f:
        json.dump({"study_resource_name": config.study_resource_name}, f, indent=2)

    _sync_vizier_db_to_gcs(local_db_path, output_db_path)


def run_vizier_optimal(config: VizierOptimalConfig) -> None:
    """Load the final Vizier study and report optimal trials."""
    from vizier.service import clients

    local_db_path = _local_vizier_db_path(config.study_id)
    if not _sync_vizier_db_from_gcs(config.input_db_path, local_db_path):
        raise FileNotFoundError(f"Could not load Vizier DB from: {config.input_db_path}")
    _configure_vizier_local_db(local_db_path)

    study = clients.Study.from_resource_name(config.study_resource_name)
    optimal_trials = []
    for optimal_trial in study.optimal_trials():
        materialized = optimal_trial.materialize()
        optimal_trials.append(
            {
                "trial_id": materialized.id,
                "parameters": _serialize_parameters(materialized.parameters),
                "final_measurement": str(materialized.final_measurement),
            }
        )

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(os.path.join(config.output_path, OPTIMAL_FILENAME), "w") as f:
        json.dump({"optimal_trials": optimal_trials}, f, indent=2)


def _build_suggest_step(scale: MuonSearchScale, *, loop_index: int, input_db_path: str | None) -> ExecutorStep:
    study_id = SWEEP.study_id(scale)
    return ExecutorStep(
        name=f"{study_id}-suggest-loop{loop_index}",
        fn=remote(run_vizier_suggest, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["vizier"]),
        config=VizierSuggestConfig(
            study_owner=SWEEP.study_owner,
            study_id=study_id,
            input_db_path=input_db_path,
            output_path=this_output_path(),
            num_suggestions=SWEEP.suggestions_per_loop,
            client_id=f"{study_id}-loop-{loop_index}",
            metric_key=SWEEP.metric_key,
            mode=SWEEP.metric_mode,
            algorithm=SWEEP.vizier_algorithm,
            search_space=SWEEP.search_space,
        ),
    )


def _build_train_step(
    scale: MuonSearchScale,
    *,
    loop_index: int,
    suggestion_index: int,
    suggestions_path: str,
    base_launch_config: GrugMoeLaunchConfig,
    base_optimizer: SearchOptimizerConfig,
) -> ExecutorStep:
    return ExecutorStep(
        name=os.path.join("checkpoints", f"{SWEEP.study_id(scale)}-loop{loop_index}-trial{suggestion_index}"),
        fn=remote(run_vizier_train, resources=ResourceConfig.with_cpu()),
        config=VizierTrainConfig(
            suggestions_path=suggestions_path,
            suggestion_index=suggestion_index,
            base_launch_config=base_launch_config,
            base_optimizer=base_optimizer,
            scale_name=scale.name,
            loop_index=loop_index,
            experiment_name=SWEEP.experiment_name,
            matrix_lr_multiplier_name=SWEEP.matrix_lr_multiplier_name,
        ),
    )


def _build_update_step(
    scale: MuonSearchScale,
    *,
    input_db_path: str | None,
    suggestions_path: str,
    training_steps: list[ExecutorStep],
) -> ExecutorStep:
    return ExecutorStep(
        name=f"{SWEEP.study_id(scale)}-update",
        fn=remote(run_vizier_update, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["vizier"]),
        config=VizierUpdateConfig(
            study_id=SWEEP.study_id(scale),
            study_resource_name=SWEEP.study_resource_name(scale),
            input_db_path=input_db_path,
            suggestions_path=suggestions_path,
            run_paths=[step.as_input_name() for step in training_steps],
            metric_file=SWEEP.metric_file,
            metric_key=SWEEP.metric_key,
            mode=SWEEP.metric_mode,
            output_path=this_output_path(),
        ),
    )


def _build_optimal_step(scale: MuonSearchScale, *, input_db_path: str) -> ExecutorStep:
    return ExecutorStep(
        name=f"{SWEEP.study_id(scale)}-optimal",
        fn=remote(run_vizier_optimal, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["vizier"]),
        config=VizierOptimalConfig(
            study_id=SWEEP.study_id(scale),
            study_resource_name=SWEEP.study_resource_name(scale),
            input_db_path=input_db_path,
            output_path=this_output_path(),
        ),
    )


def _build_scale_chain(scale: MuonSearchScale) -> ExecutorStep:
    num_loops = 1 if os.getenv("CI") is not None else SWEEP.num_loops
    base_launch_config, base_optimizer = _build_base_launch_config(scale)
    previous_update_step: ExecutorStep | None = None

    for loop_index in range(num_loops):
        input_db_path = previous_update_step / VIZIER_DB_FILENAME if previous_update_step else None
        suggest_step = _build_suggest_step(scale, loop_index=loop_index, input_db_path=input_db_path)
        suggestions_path = suggest_step / SUGGESTIONS_FILENAME
        training_steps = [
            _build_train_step(
                scale,
                loop_index=loop_index,
                suggestion_index=suggestion_index,
                suggestions_path=suggestions_path,
                base_launch_config=base_launch_config,
                base_optimizer=base_optimizer,
            )
            for suggestion_index in range(SWEEP.suggestions_per_loop)
        ]
        previous_update_step = _build_update_step(
            scale,
            input_db_path=suggest_step / VIZIER_DB_FILENAME,
            suggestions_path=suggestions_path,
            training_steps=training_steps,
        )

    return _build_optimal_step(scale, input_db_path=previous_update_step / VIZIER_DB_FILENAME)


if __name__ == "__main__":
    executor_main(
        steps=[_build_scale_chain(scale) for scale in SWEEP.scales],
        description="Grug MoE Muon Vizier LR/beta search at exact gate-1 FLOP scales.",
    )
