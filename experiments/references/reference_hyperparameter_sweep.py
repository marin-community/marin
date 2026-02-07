# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AdamH hyperparameter sweep for a ~130M Qwen3 model on Nemotron mix."""

import json
import os
import re
import shutil
import sqlite3
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace

import fsspec
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim import AdamHConfig
from vizier.service import clients
from vizier.service import pyvizier as vz

from experiments.defaults import default_train
from experiments.simple_train_config import SimpleTrainConfig
from experiments.tootsie.exp1295_32b import nemotron_mix
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

FloatRange = tuple[float, float]


@dataclass(frozen=True)
class SweepSettings:
    """User-editable sweep settings.

    Edit only this block for routine sweep changes.
    """

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


# Edit this single object to tune the sweep.
SWEEP = SweepSettings(
    # Common edits.
    experiment_name="ref-sweep-qwen3-130m-vizier-v3",
    num_loops=10,
    suggestions_per_loop=10,
    search_space={
        "lr": (0.005, 0.03),
        "beta1": (0.85, 0.999),
        "adam_lr": (0.005, 0.03),
        "beta2": (0.85, 0.999),
        "epsilon": (1e-15, 1e-6),
        "max_grad_norm": (0.1, 5.0),
    },
    fixed_batch_size=64,
    target_tokens=1_000_000_000,
    metric_key="eval/uncheatable_eval/macro_loss",
    metric_mode="min",
    # Rare edits.
    study_owner="marin",
    seq_len=4096,
    metric_file="tracker_metrics.jsonl",
    vizier_algorithm="DEFAULT",
    lr_schedule="linear",
    warmup_fraction=0.1,
    decay_fraction=0.2,
    base_train_tags=("sweep", "qwen3", "130m", "adamh"),
)

SUGGESTIONS_FILENAME = "vizier_suggestions.json"
UPDATE_FILENAME = "vizier_update.json"
RESOURCE_FILENAME = "vizier_resource.json"
VIZIER_DB_FILENAME = "vizier.db"

qwen3_130m = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=512,
    intermediate_dim=512 * 4,
    num_heads=4,
    num_kv_heads=4,
    num_layers=6,
    rope=Llama3RotaryEmbeddingsConfig(),
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
    base_pod_config: TrainLmOnPodConfig
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


def best_run(runs, mode="min"):
    """Return the run with the best metric."""

    def metric_key(record: dict) -> float:
        return record["metric"]

    return min(runs, key=metric_key) if mode == "min" else max(runs, key=metric_key)


def _local_vizier_db_path(study_id: str) -> str:
    safe_study = re.sub(r"[^A-Za-z0-9_.-]+", "_", study_id)
    return os.path.join(tempfile.gettempdir(), f"vizier-{safe_study}.db")


def _configure_vizier_local_db(local_path: str) -> None:
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
        # Vizier returns ParameterValue objects from trial.parameters.
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
                raise ValueError(f"Unsupported parameter value for '{key}': {raw_value!r}") from e
    return serialized


def _metric_goal(mode: str) -> vz.ObjectiveMetricGoal:
    if mode == "min":
        return vz.ObjectiveMetricGoal.MINIMIZE
    if mode == "max":
        return vz.ObjectiveMetricGoal.MAXIMIZE
    raise ValueError(f"Unsupported metric mode: {mode}")


def _extract_adamh_hparams(suggestion: dict[str, object]) -> dict[str, float]:
    parameters = suggestion["parameters"]
    if not isinstance(parameters, Mapping):
        raise ValueError(f"Expected suggestion parameters mapping, got {type(parameters)!r}")
    required = ("lr", "beta1", "adam_lr", "beta2", "epsilon", "max_grad_norm")
    return {name: float(parameters[name]) for name in required}


def _build_adamh_config(
    *,
    learning_rate: float,
    beta1: float,
    adam_learning_rate: float,
    beta2: float,
    epsilon: float,
    max_grad_norm: float,
) -> AdamHConfig:
    return AdamHConfig(
        learning_rate=learning_rate,
        adam_lr=adam_learning_rate,
        min_lr_ratio=0.0,
        warmup=SWEEP.warmup_fraction,
        decay=SWEEP.decay_fraction,
        lr_schedule=SWEEP.lr_schedule,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        max_grad_norm=max_grad_norm,
        nesterov=False,
    )


def _build_base_pod_config() -> TrainLmOnPodConfig:
    placeholder_lr = SWEEP.search_space["lr"][0]
    placeholder_beta1 = SWEEP.search_space["beta1"][0]
    placeholder_adam_lr = SWEEP.search_space["adam_lr"][0]
    placeholder_beta2 = SWEEP.search_space["beta2"][0]
    placeholder_epsilon = SWEEP.search_space["epsilon"][0]
    placeholder_max_grad_norm = SWEEP.search_space["max_grad_norm"][0]
    placeholder_batch_size = SWEEP.fixed_batch_size
    placeholder_steps = SWEEP.target_tokens // (placeholder_batch_size * SWEEP.seq_len)

    train_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu("v5p-8"),
        train_batch_size=placeholder_batch_size,
        num_train_steps=placeholder_steps,
        learning_rate=placeholder_lr,
        train_seq_len=SWEEP.seq_len,
        z_loss_weight=5e-6,
        optimizer_config=_build_adamh_config(
            learning_rate=placeholder_lr,
            beta1=placeholder_beta1,
            adam_learning_rate=placeholder_adam_lr,
            beta2=placeholder_beta2,
            epsilon=placeholder_epsilon,
            max_grad_norm=placeholder_max_grad_norm,
        ),
        steps_per_eval=500,
    )

    base_step = default_train(
        name=f"{SWEEP.experiment_name}-base",
        tokenized=nemotron_mix,
        model_config=qwen3_130m,
        train_config=train_config,
        tags=list(SWEEP.base_train_tags),
        eval_harness_tasks=[],
    )
    return base_step.config


def run_vizier_suggest(config: VizierSuggestConfig) -> None:
    """Create or load a Vizier study, suggest trials, and persist the study DB."""
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
    """Train a model for a single Vizier suggestion."""
    suggestions = _load_suggestions(config.suggestions_path)["suggestions"]
    if config.suggestion_index >= len(suggestions):
        raise IndexError(f"Suggestion index {config.suggestion_index} out of range")

    suggestion = suggestions[config.suggestion_index]
    hparams = _extract_adamh_hparams(suggestion)
    batch_size = config.fixed_batch_size
    num_steps = config.target_tokens // (batch_size * config.seq_len)

    base_pod_config = config.base_pod_config
    base_train_config = base_pod_config.train_config
    base_trainer = base_train_config.trainer

    new_tags = list(getattr(base_trainer.tracker, "tags", []) or [])
    trial_id = int(suggestion["trial_id"])
    new_tags.extend(
        [
            f"lr={hparams['lr']}",
            f"beta1={hparams['beta1']}",
            f"adam_lr={hparams['adam_lr']}",
            f"beta2={hparams['beta2']}",
            f"eps={hparams['epsilon']}",
            f"mgn={hparams['max_grad_norm']}",
            f"bs={batch_size}",
            f"trial={trial_id}",
            f"loop={config.loop_index}",
        ]
    )

    tracker = replace(base_trainer.tracker, tags=new_tags)
    trainer = replace(
        base_trainer,
        train_batch_size=batch_size,
        num_train_steps=num_steps,
        tracker=tracker,
    )

    train_config = replace(
        base_train_config,
        trainer=trainer,
        optimizer=_build_adamh_config(
            learning_rate=hparams["lr"],
            beta1=hparams["beta1"],
            adam_learning_rate=hparams["adam_lr"],
            beta2=hparams["beta2"],
            epsilon=hparams["epsilon"],
            max_grad_norm=hparams["max_grad_norm"],
        ),
    )
    pod_config = replace(base_pod_config, train_config=train_config)

    run_levanter_train_lm(pod_config)


def run_vizier_update(config: VizierUpdateConfig) -> None:
    """Load trial results, update Vizier, and write summary output."""
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
        metric_path = os.path.join(run_path, config.metric_file)
        fs, _, _ = fsspec.get_fs_token_paths(metric_path)
        with fs.open(metric_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            if not lines:
                raise RuntimeError(f"No metrics found at {metric_path}")
            data = json.loads(lines[-1])

        value = data["summary"][config.metric_key]
        trial_id = int(suggestion["trial_id"])
        trial = study.get_trial(trial_id)
        measurement = vz.Measurement({config.metric_key: float(value)})
        trial.complete(measurement)

        results.append(
            {
                "trial_id": trial_id,
                "metric": float(value),
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


def _build_suggest_step(
    *,
    loop_index: int,
    input_db_path: str | None,
) -> ExecutorStep:
    client_id = f"{SWEEP.client_id_prefix}-loop-{loop_index}"
    return ExecutorStep(
        name=f"{SWEEP.experiment_name}-suggest-loop{loop_index}",
        fn=run_vizier_suggest,
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
    *,
    loop_index: int,
    suggestion_index: int,
    suggestions_path: str,
    base_pod_config: TrainLmOnPodConfig,
) -> ExecutorStep:
    return ExecutorStep(
        name=os.path.join(
            "checkpoints",
            f"{SWEEP.client_id_prefix}-loop{loop_index}-trial{suggestion_index}",
        ),
        fn=run_vizier_train,
        config=VizierTrainConfig(
            suggestions_path=suggestions_path,
            suggestion_index=suggestion_index,
            base_pod_config=base_pod_config,
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
        fn=run_vizier_update,
        config=VizierUpdateConfig(
            study_id=SWEEP.study_id,
            study_resource_name=study_resource_name,
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


if __name__ == "__main__":
    num_loops = SWEEP.num_loops
    if os.getenv("CI", None) is not None:
        num_loops = 1
    suggestions_per_loop = SWEEP.suggestions_per_loop

    previous_update_step: ExecutorStep | None = None
    base_pod_config = _build_base_pod_config()

    for loop_index in range(num_loops):
        input_db_path = previous_update_step / VIZIER_DB_FILENAME if previous_update_step else None
        suggest_step = _build_suggest_step(loop_index=loop_index, input_db_path=input_db_path)

        suggestions_path = suggest_step / SUGGESTIONS_FILENAME
        training_steps = [
            _build_train_step(
                loop_index=loop_index,
                suggestion_index=suggestion_index,
                suggestions_path=suggestions_path,
                base_pod_config=base_pod_config,
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
        executor_main(steps=[update_step])
        previous_update_step = update_step
