# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adam hyperparameter sweep for Grug MoE iter_02 (baseline) on Nemotron mix.

Sweeps over learning rate and momentum parameters (beta1, beta2) using Vizier.
This is the baseline comparison for the iter_03 AdamH sweep.
"""

import json
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
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig

from experiments.grug.moe_scaling_iteration_02.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe_scaling_iteration_02.model import GrugModelConfig
from experiments.grug.moe_scaling_iteration_02.train import GrugEvalConfig, GrugTrainerConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote

FloatRange = tuple[float, float]


@dataclass(frozen=True)
class SweepSettings:
    experiment_name: str
    study_owner: str
    num_loops: int
    suggestions_per_loop: int
    search_space: Mapping[str, FloatRange]
    fixed_batch_size: int
    target_tokens: int
    seq_len: int
    metric_file: str
    metric_key: str
    metric_mode: str
    vizier_algorithm: str
    lr_schedule: str
    warmup_fraction: float
    decay_fraction: float
    base_train_tags: tuple[str, ...]

    @property
    def study_id(self) -> str:
        return self.experiment_name

    @property
    def study_resource_name(self) -> str:
        return f"owners/{self.study_owner}/studies/{self.study_id}"

    @property
    def client_id_prefix(self) -> str:
        return self.experiment_name


SWEEP = SweepSettings(
    experiment_name="ref-sweep-moe-iter02-adam-v3",
    num_loops=20,
    suggestions_per_loop=8,
    search_space={
        "learning_rate": (1e-4, 0.05),
        "beta1": (0.8, 0.99),
        "beta2": (0.9, 0.999),
        "weight_decay": (0.01, 0.3),
    },
    fixed_batch_size=32,
    target_tokens=1_400_000_000,  # 3e18 FLOPs at d=1024
    metric_key="eval/uncheatable_eval/macro_loss",
    metric_mode="min",
    study_owner="marin",
    seq_len=4096,
    metric_file="tracker_metrics.jsonl",
    vizier_algorithm="DEFAULT",
    lr_schedule="linear",
    warmup_fraction=0.1,
    decay_fraction=0.2,
    base_train_tags=("sweep", "grug", "moe", "iter02", "adam"),
)

SUGGESTIONS_FILENAME = "vizier_suggestions.json"
UPDATE_FILENAME = "vizier_update.json"
RESOURCE_FILENAME = "vizier_resource.json"
OPTIMAL_FILENAME = "vizier_optimal.json"
VIZIER_DB_FILENAME = "vizier.db"

FIXED_EPSILON = 1e-8

HIDDEN_DIM = 1024
SWEEP_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=HIDDEN_DIM,
    intermediate_dim=HIDDEN_DIM // 2,
    shared_expert_intermediate_dim=HIDDEN_DIM,
    num_experts=64,
    num_experts_per_token=4,
    num_layers=11,
    num_heads=HIDDEN_DIM // 128,
    num_kv_heads=HIDDEN_DIM // 128,
    max_seq_len=4096,
    num_dense_layers=2,
    dense_intermediate_dim=3 * HIDDEN_DIM,
    sliding_window=4096,
    initializer_std=0.5 / math.sqrt(HIDDEN_DIM),
    qk_mult=1.3,
    load_balancing_loss_coef=0.001,
    bias_update_rate=0.01,
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
    loop_index: int


@dataclass(frozen=True)
class VizierTrainConfig:
    suggestions_path: str
    suggestion_index: int
    base_launch_config: GrugMoeLaunchConfig
    target_tokens: int
    seq_len: int
    fixed_batch_size: int
    loop_index: int


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
    loop_index: int


@dataclass(frozen=True)
class VizierOptimalConfig:
    study_id: str
    study_resource_name: str
    input_db_path: str
    output_path: str


def best_run(runs, mode="min"):
    def metric_key(record: dict) -> float:
        return record["metric"]

    return min(runs, key=metric_key) if mode == "min" else max(runs, key=metric_key)


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
        elif isinstance(raw_value, (int, float)):
            serialized[key] = raw_value
        else:
            try:
                serialized[key] = float(raw_value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Unsupported parameter value for '{key}': {raw_value!r}") from e
    return serialized


def _metric_goal(mode: str) -> Any:
    from vizier.service import pyvizier as vz

    if mode == "min":
        return vz.ObjectiveMetricGoal.MINIMIZE
    if mode == "max":
        return vz.ObjectiveMetricGoal.MAXIMIZE
    raise ValueError(f"Unsupported metric mode: {mode}")


def _extract_hparams(suggestion: dict[str, object]) -> dict[str, float]:
    parameters = suggestion["parameters"]
    required = ("learning_rate", "beta1", "beta2", "weight_decay")
    return {name: float(parameters[name]) for name in required}


def _build_base_launch_config() -> GrugMoeLaunchConfig:
    placeholder_batch_size = SWEEP.fixed_batch_size
    placeholder_steps = SWEEP.target_tokens // (placeholder_batch_size * SWEEP.seq_len)

    return GrugMoeLaunchConfig(
        model=SWEEP_MODEL,
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=f"{SWEEP.experiment_name}-base",
        resources=ResourceConfig.with_tpu("v5p-8"),
        steps=placeholder_steps,
        batch_size=placeholder_batch_size,
        seed=0,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=WandbConfig(
            project="marin",
            tags=list(SWEEP.base_train_tags),
            group=SWEEP.experiment_name,
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=AdamConfig(
            learning_rate=1e-3,
            beta1=0.96,
            beta2=0.98,
            epsilon=1e-15,
            weight_decay=0.1,
            lr_schedule=SWEEP.lr_schedule,
            decay=SWEEP.decay_fraction,
            min_lr_ratio=0.0,
            warmup=SWEEP.warmup_fraction,
            max_grad_norm=1,
        ),
        grug_trainer=GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1),
        eval=GrugEvalConfig(
            eval_batch_size=512,
            steps_per_eval=1000,
            max_eval_batches=8,
            eval_current=True,
            eval_ema=False,
        ),
    )


def run_vizier_suggest(config: VizierSuggestConfig) -> None:
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

    study = clients.Study.from_study_config(
        study_config,
        owner=config.study_owner,
        study_id=config.study_id,
    )
    expected_resource_name = f"owners/{config.study_owner}/studies/{config.study_id}"
    if study.resource_name != expected_resource_name:
        raise ValueError(f"Study resource name mismatch: expected {expected_resource_name}, got {study.resource_name}")

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
    suggestions = _load_suggestions(config.suggestions_path)["suggestions"]
    if config.suggestion_index >= len(suggestions):
        raise IndexError(f"Suggestion index {config.suggestion_index} out of range")

    suggestion = suggestions[config.suggestion_index]
    hparams = _extract_hparams(suggestion)
    batch_size = config.fixed_batch_size
    num_steps = config.target_tokens // (batch_size * config.seq_len)

    base = config.base_launch_config
    trial_id = int(suggestion["trial_id"])

    new_tags = list(getattr(base.tracker, "tags", []) or [])
    new_tags.extend(
        [
            f"lr={hparams['learning_rate']:.4g}",
            f"beta1={hparams['beta1']:.3f}",
            f"beta2={hparams['beta2']:.3f}",
            f"wd={hparams['weight_decay']:.3g}",
            f"trial={trial_id}",
            f"loop={config.loop_index}",
        ]
    )

    tracker = replace(base.tracker, tags=new_tags, name=f"trial-{trial_id}-loop-{config.loop_index}")
    optimizer = replace(
        base.optimizer,
        learning_rate=hparams["learning_rate"],
        beta1=hparams["beta1"],
        beta2=hparams["beta2"],
        weight_decay=hparams["weight_decay"],
    )

    launch_config = replace(
        base,
        run_id=f"{SWEEP.experiment_name}-loop{config.loop_index}-trial{trial_id}",
        steps=num_steps,
        batch_size=batch_size,
        tracker=tracker,
        optimizer=optimizer,
    )

    run_grug_moe_trial(launch_config)


def run_vizier_update(config: VizierUpdateConfig) -> None:
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
        raise ValueError(
            f"Expected {len(suggestions)} run paths but got {len(config.run_paths)} for loop {config.loop_index}"
        )

    results = []
    for run_path, suggestion in zip(config.run_paths, suggestions, strict=True):
        trial_id = int(suggestion["trial_id"])
        trial = study.get_trial(trial_id)
        metric_path = os.path.join(run_path, config.metric_file)
        try:
            fs, _, _ = fsspec.get_fs_token_paths(metric_path)
            with fs.open(metric_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
                if not lines:
                    raise RuntimeError(f"No metrics found at {metric_path}")
                data = json.loads(lines[-1])
            value = float(data["summary"][config.metric_key])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite metric value: {value}")
        except Exception as e:
            print(f"Trial {trial_id}: FAILED ({e}), assigning penalty loss")
            value = 100.0

        measurement = vz.Measurement({config.metric_key: value})
        trial.complete(measurement)

        results.append(
            {
                "trial_id": trial_id,
                "metric": value,
                "hparams": suggestion["parameters"],
                "run_path": run_path,
            }
        )
        print(f"Trial {trial_id}: {config.metric_key} = {value}")

    if not results:
        raise RuntimeError("No valid results found")

    best = best_run(results, config.mode)

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    output = {
        "study_resource_name": config.study_resource_name,
        "best_hparams": best["hparams"],
        "best_metric": best["metric"],
        "best_run_path": best["run_path"],
        "all_results": sorted(results, key=lambda r: r["metric"], reverse=(config.mode == "max")),
    }

    with fs.open(os.path.join(config.output_path, UPDATE_FILENAME), "w") as f:
        json.dump(output, f, indent=2)
    with fs.open(os.path.join(config.output_path, RESOURCE_FILENAME), "w") as f:
        json.dump({"study_resource_name": config.study_resource_name}, f, indent=2)

    _sync_vizier_db_to_gcs(local_db_path, output_db_path)


def run_vizier_optimal(config: VizierOptimalConfig) -> None:
    from vizier.service import clients

    local_db_path = _local_vizier_db_path(config.study_id)
    if not _sync_vizier_db_from_gcs(config.input_db_path, local_db_path):
        raise FileNotFoundError(f"Could not load Vizier DB from: {config.input_db_path}")
    _configure_vizier_local_db(local_db_path)

    study = clients.Study.from_resource_name(config.study_resource_name)
    optimal_trials = []
    for optimal_trial in study.optimal_trials():
        optimal_trial = optimal_trial.materialize()
        print("Optimal Trial Suggestion and Objective:", optimal_trial.parameters, optimal_trial.final_measurement)
        optimal_trials.append(
            {
                "trial_id": optimal_trial.id,
                "parameters": _serialize_parameters(optimal_trial.parameters),
                "final_measurement": str(optimal_trial.final_measurement),
            }
        )

    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(os.path.join(config.output_path, OPTIMAL_FILENAME), "w") as f:
        json.dump({"optimal_trials": optimal_trials}, f, indent=2)


def _build_suggest_step(*, loop_index: int, input_db_path: str | None) -> ExecutorStep:
    client_id = f"{SWEEP.client_id_prefix}-loop-{loop_index}"
    return ExecutorStep(
        name=f"{SWEEP.experiment_name}-suggest-loop{loop_index}",
        fn=remote(run_vizier_suggest, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["vizier"]),
        config=VizierSuggestConfig(
            study_owner=SWEEP.study_owner,
            study_id=SWEEP.study_id,
            input_db_path=input_db_path,
            output_path=this_output_path(),
            num_suggestions=SWEEP.suggestions_per_loop,
            client_id=client_id,
            metric_key=SWEEP.metric_key,
            mode=SWEEP.metric_mode,
            algorithm=SWEEP.vizier_algorithm,
            search_space=SWEEP.search_space,
            loop_index=loop_index,
        ),
    )


def _build_train_step(
    *, loop_index: int, suggestion_index: int, suggestions_path: str, base_launch_config: GrugMoeLaunchConfig
) -> ExecutorStep:
    return ExecutorStep(
        name=os.path.join(
            "checkpoints",
            f"{SWEEP.client_id_prefix}-loop{loop_index}-trial{suggestion_index}",
        ),
        fn=remote(run_vizier_train, resources=ResourceConfig.with_cpu()),
        config=VizierTrainConfig(
            suggestions_path=suggestions_path,
            suggestion_index=suggestion_index,
            base_launch_config=base_launch_config,
            target_tokens=SWEEP.target_tokens,
            seq_len=SWEEP.seq_len,
            fixed_batch_size=SWEEP.fixed_batch_size,
            loop_index=loop_index,
        ),
    )


def _build_update_step(
    *,
    loop_index: int,
    study_resource_name: str,
    input_db_path: str | None,
    suggestions_path: str,
    training_steps: list[ExecutorStep],
) -> ExecutorStep:
    return ExecutorStep(
        name=f"{SWEEP.experiment_name}-update-loop{loop_index}",
        fn=remote(run_vizier_update, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["vizier"]),
        config=VizierUpdateConfig(
            study_id=SWEEP.study_id,
            study_resource_name=SWEEP.study_resource_name,
            input_db_path=input_db_path,
            suggestions_path=suggestions_path,
            run_paths=[step.as_input_name() for step in training_steps],
            metric_file=SWEEP.metric_file,
            metric_key=SWEEP.metric_key,
            mode=SWEEP.metric_mode,
            output_path=this_output_path(),
            loop_index=loop_index,
        ),
    )


def _build_optimal_step(*, input_db_path: str, study_resource_name: str) -> ExecutorStep:
    return ExecutorStep(
        name=f"{SWEEP.experiment_name}-optimal",
        fn=remote(run_vizier_optimal, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["vizier"]),
        config=VizierOptimalConfig(
            study_id=SWEEP.study_id,
            study_resource_name=SWEEP.study_resource_name,
            input_db_path=input_db_path,
            output_path=this_output_path(),
        ),
    )


if __name__ == "__main__":
    num_loops = SWEEP.num_loops
    if os.getenv("CI", None) is not None:
        num_loops = 1
    suggestions_per_loop = SWEEP.suggestions_per_loop

    previous_update_step: ExecutorStep | None = None
    base_launch_config = _build_base_launch_config()

    for loop_index in range(num_loops):
        input_db_path = previous_update_step / VIZIER_DB_FILENAME if previous_update_step else None
        suggest_step = _build_suggest_step(loop_index=loop_index, input_db_path=input_db_path)

        suggestions_path = suggest_step / SUGGESTIONS_FILENAME
        training_steps = [
            _build_train_step(
                loop_index=loop_index,
                suggestion_index=suggestion_index,
                suggestions_path=suggestions_path,
                base_launch_config=base_launch_config,
            )
            for suggestion_index in range(suggestions_per_loop)
        ]

        update_step = _build_update_step(
            loop_index=loop_index,
            study_resource_name=SWEEP.study_resource_name,
            input_db_path=suggest_step / VIZIER_DB_FILENAME,
            suggestions_path=suggestions_path,
            training_steps=training_steps,
        )
        previous_update_step = update_step

    optimal_step = _build_optimal_step(
        input_db_path=previous_update_step / VIZIER_DB_FILENAME,
        study_resource_name=SWEEP.study_resource_name,
    )
    executor_main(steps=[optimal_step])
