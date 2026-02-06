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

# Sweep knobs (easy to edit in one place)
NUM_LOOPS = 10
SUGGESTIONS_PER_LOOP = 10
VIZIER_STUDY_OWNER = "marin"
VIZIER_STUDY_ID = "ref-sweep-vizier-v3"
VIZIER_STUDY_RESOURCE_NAME = f"owners/{VIZIER_STUDY_OWNER}/studies/{VIZIER_STUDY_ID}"
VIZIER_CLIENT_ID_PREFIX = "ref-sweep-vizier-v3"
VIZIER_ALGORITHM = "DEFAULT"

LEARNING_RATE_RANGE = (0.005, 0.03)
BETA1_RANGE = (0.85, 0.999)
ADAM_LEARNING_RATE_RANGE = (0.005, 0.03)
BETA2_RANGE = (0.85, 0.999)
EPSILON_RANGE = (1e-15, 1e-6)
MAX_GRAD_NORM_RANGE = (0.1, 5.0)
FIXED_BATCH_SIZE = 64
TARGET_TOKENS = 1_000_000_000
SEQ_LEN = 4096
METRIC_FILE = "tracker_metrics.jsonl"
METRIC_KEY = "eval/uncheatable_eval/macro_loss"
METRIC_MODE = "min"
LR_SCHEDULE = "linear"
WARMUP_FRACTION = 0.1
DECAY_FRACTION = 0.2

SUGGESTIONS_FILENAME = "vizier_suggestions.json"
UPDATE_FILENAME = "vizier_update.json"
RESOURCE_FILENAME = "vizier_resource.json"
VIZIER_DB_FILENAME = "vizier.db"

BASE_TRAIN_TAGS = ["sweep", "qwen3", "130m", "adamh"]

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
    study_resource_name: str
    input_db_path: str | None
    output_path: str
    num_suggestions: int
    client_id: str
    metric_key: str
    mode: str
    learning_rate_range: tuple[float, float]
    beta1_range: tuple[float, float]
    adam_learning_rate_range: tuple[float, float]
    beta2_range: tuple[float, float]
    epsilon_range: tuple[float, float]
    max_grad_norm_range: tuple[float, float]
    loop_index: int


@dataclass(frozen=True)
class VizierTrainConfig:
    suggestions_path: str
    suggestion_index: int
    base_pod_config: TrainLmOnPodConfig
    target_tokens: int
    seq_len: int
    loop_index: int


@dataclass(frozen=True)
class VizierUpdateConfig:
    study_id: str
    study_resource_name: str
    input_db_path: str | None
    suggestions_path: str
    run_paths: list
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


def _local_vizier_db_path(study_id: str, loop_index: int) -> str:
    safe_study = re.sub(r"[^A-Za-z0-9_.-]+", "_", study_id)
    return os.path.join(tempfile.gettempdir(), f"vizier-{safe_study}-loop-{loop_index}.db")


def _configure_vizier_local_db(local_path: str) -> None:
    clients.environment_variables.servicer_kwargs["database_url"] = f"sqlite:///{local_path}"


def _sync_vizier_db_from_gcs(path: str | None, local_path: str) -> bool:
    if not path:
        return False
    fs, _, _ = fsspec.get_fs_token_paths(path)
    if not fs.exists(path):
        return False
    with fs.open(path, "rb") as src, open(local_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return True


def _sync_vizier_db_to_gcs(local_path: str, path: str) -> None:
    fs, _, _ = fsspec.get_fs_token_paths(path)
    fs.makedirs(os.path.dirname(path), exist_ok=True)
    with open(local_path, "rb") as src, fs.open(path, "wb") as dst:
        shutil.copyfileobj(src, dst)


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
        warmup=WARMUP_FRACTION,
        decay=DECAY_FRACTION,
        lr_schedule=LR_SCHEDULE,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        max_grad_norm=max_grad_norm,
        nesterov=False,
    )


def _build_base_pod_config() -> TrainLmOnPodConfig:
    placeholder_lr = LEARNING_RATE_RANGE[0]
    placeholder_beta1 = BETA1_RANGE[0]
    placeholder_adam_lr = ADAM_LEARNING_RATE_RANGE[0]
    placeholder_beta2 = BETA2_RANGE[0]
    placeholder_epsilon = EPSILON_RANGE[0]
    placeholder_max_grad_norm = MAX_GRAD_NORM_RANGE[0]
    placeholder_batch_size = FIXED_BATCH_SIZE
    placeholder_steps = TARGET_TOKENS // (placeholder_batch_size * SEQ_LEN)

    train_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu("v5p-8"),
        train_batch_size=placeholder_batch_size,
        num_train_steps=placeholder_steps,
        learning_rate=placeholder_lr,
        train_seq_len=SEQ_LEN,
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
        name="ref-sweep-qwen3-130m-vizier-v3-base",
        tokenized=nemotron_mix,
        model_config=qwen3_130m,
        train_config=train_config,
        tags=BASE_TRAIN_TAGS,
        eval_harness_tasks=[],
    )
    return base_step.config


def run_vizier_suggest(config: VizierSuggestConfig) -> None:
    """Create or load a Vizier study, suggest trials, and persist the study DB."""
    local_db_path = _local_vizier_db_path(config.study_id, config.loop_index)
    output_db_path = os.path.join(config.output_path, VIZIER_DB_FILENAME)
    if not _sync_vizier_db_from_gcs(output_db_path, local_db_path):
        _sync_vizier_db_from_gcs(config.input_db_path, local_db_path)
    _configure_vizier_local_db(local_db_path)

    study_config = vz.StudyConfig(algorithm=VIZIER_ALGORITHM)
    root = study_config.search_space.root
    root.add_float_param("lr", *config.learning_rate_range)
    root.add_float_param("beta1", *config.beta1_range)
    root.add_float_param("adam_lr", *config.adam_learning_rate_range)
    root.add_float_param("beta2", *config.beta2_range)
    root.add_float_param("epsilon", *config.epsilon_range)
    root.add_float_param("max_grad_norm", *config.max_grad_norm_range)
    study_config.metric_information.append(vz.MetricInformation(config.metric_key, goal=_metric_goal(config.mode)))

    study = clients.Study.from_study_config(
        study_config,
        owner=config.study_owner,
        study_id=config.study_id,
    )
    if study.resource_name != config.study_resource_name:
        raise ValueError(
            f"Study resource name mismatch: expected {config.study_resource_name}, got {study.resource_name}"
        )

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
    lr = float(suggestion["parameters"]["lr"])
    beta1 = float(suggestion["parameters"]["beta1"])
    adam_lr = float(suggestion["parameters"]["adam_lr"])
    beta2 = float(suggestion["parameters"]["beta2"])
    epsilon = float(suggestion["parameters"]["epsilon"])
    max_grad_norm = float(suggestion["parameters"]["max_grad_norm"])
    batch_size = FIXED_BATCH_SIZE
    num_steps = config.target_tokens // (batch_size * config.seq_len)

    base_pod_config = config.base_pod_config
    base_train_config = base_pod_config.train_config
    base_trainer = base_train_config.trainer

    new_tags = list(getattr(base_trainer.tracker, "tags", []) or [])
    trial_id = int(suggestion["trial_id"])
    new_tags.extend(
        [
            f"lr={lr}",
            f"beta1={beta1}",
            f"adam_lr={adam_lr}",
            f"beta2={beta2}",
            f"eps={epsilon}",
            f"mgn={max_grad_norm}",
            f"bs={FIXED_BATCH_SIZE}",
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
            learning_rate=lr,
            beta1=beta1,
            adam_learning_rate=adam_lr,
            beta2=beta2,
            epsilon=epsilon,
            max_grad_norm=max_grad_norm,
        ),
    )
    pod_config = replace(base_pod_config, train_config=train_config)

    run_levanter_train_lm(pod_config)


def run_vizier_update(config: VizierUpdateConfig) -> None:
    """Load trial results, update Vizier, and write summary output."""
    local_db_path = _local_vizier_db_path(config.study_id, config.loop_index)
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
    client_id = f"{VIZIER_CLIENT_ID_PREFIX}-loop-{loop_index}"
    return ExecutorStep(
        name=f"ref-sweep-qwen3-130m-vizier-v3-suggest-loop{loop_index}",
        fn=run_vizier_suggest,
        config=VizierSuggestConfig(
            study_owner=VIZIER_STUDY_OWNER,
            study_id=VIZIER_STUDY_ID,
            study_resource_name=VIZIER_STUDY_RESOURCE_NAME,
            input_db_path=input_db_path,
            output_path=this_output_path(),
            num_suggestions=SUGGESTIONS_PER_LOOP,
            client_id=client_id,
            metric_key=METRIC_KEY,
            mode=METRIC_MODE,
            learning_rate_range=LEARNING_RATE_RANGE,
            beta1_range=BETA1_RANGE,
            adam_learning_rate_range=ADAM_LEARNING_RATE_RANGE,
            beta2_range=BETA2_RANGE,
            epsilon_range=EPSILON_RANGE,
            max_grad_norm_range=MAX_GRAD_NORM_RANGE,
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
            f"{VIZIER_CLIENT_ID_PREFIX}-loop{loop_index}-trial{suggestion_index}",
        ),
        fn=run_vizier_train,
        config=VizierTrainConfig(
            suggestions_path=suggestions_path,
            suggestion_index=suggestion_index,
            base_pod_config=base_pod_config,
            target_tokens=TARGET_TOKENS,
            seq_len=SEQ_LEN,
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
        name=f"ref-sweep-qwen3-130m-vizier-v3-update-loop{loop_index}",
        fn=run_vizier_update,
        config=VizierUpdateConfig(
            study_id=VIZIER_STUDY_ID,
            study_resource_name=study_resource_name,
            input_db_path=input_db_path,
            suggestions_path=suggestions_path,
            run_paths=[step.as_input_name() for step in training_steps],
            metric_file=METRIC_FILE,
            metric_key=METRIC_KEY,
            mode=METRIC_MODE,
            output_path=this_output_path(),
            loop_index=loop_index,
        ),
    )


if __name__ == "__main__":
    all_steps: list[ExecutorStep] = []
    previous_update_step: ExecutorStep | None = None
    base_pod_config = _build_base_pod_config()

    for loop_index in range(NUM_LOOPS):
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
            for suggestion_index in range(SUGGESTIONS_PER_LOOP)
        ]

        update_step = _build_update_step(
            loop_index=loop_index,
            study_resource_name=VIZIER_STUDY_RESOURCE_NAME,
            input_db_path=suggest_step / VIZIER_DB_FILENAME,
            suggestions_path=suggestions_path,
            training_steps=training_steps,
        )

        all_steps.extend([suggest_step, *training_steps, update_step])
        previous_update_step = update_step

    executor_main(steps=all_steps)
