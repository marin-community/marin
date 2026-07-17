# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AdamH hyperparameter sweep for a ~130M Grug model on Nemotron mix.

Demonstrates how to integrate a third-party BO framework (Google Vizier) with
the Marin ``ArtifactStep`` execution model. The sweep runs ``SWEEP.num_loops``
Bayesian optimization rounds; each round:

  1. **Suggest** — queries Vizier for ``SWEEP.suggestions_per_loop`` trial HP sets
     (or creates the study on the first loop).
  2. **Train** — trains one Grug 130M run per suggestion on the Nemotron mix.
  3. **Update** — reports the eval metric for each trial back to Vizier, carrying
     the study DB forward.

After all loops an **Optimal** step reads the best trial from Vizier and writes
a summary JSON. Each step is an :class:`~marin.execution.lazy.ArtifactStep` and
the chain is wired via ``deps``; the :class:`~marin.execution.step_runner.StepRunner`
materializes them in dependency order.

Run:
    uv run iris --cluster=marin job run --no-wait --cpu=1 --memory=2G --extra=cpu \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.references.reference_hyperparameter_sweep
"""

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
from levanter.optim.adamh import AdamHConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.tokenize import TokenizedCache
from rigging.filesystem import prefix_join

from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.proofpile import proofpile_dataset
from experiments.datasets.starcoder import starcoder_dataset
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.base.launch import GRUG_130M_MODEL, GrugBaseLaunchConfig, run_grug_base_trial
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

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
    experiment_name="ref-sweep-grug-130m-vizier",
    num_loops=10,
    suggestions_per_loop=4,
    search_space={
        "lr": (0.00005, 0.03),
        "beta1": (0.5, 1.0),
        "adam_lr": (0.00005, 0.03),
        "beta2": (0.5, 1.0),
        "epsilon": (1e-15, 1e-3),
        "max_grad_norm": (0.1, 1.0),
        "z_loss_weight": (1e-7, 0.1),
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
    base_train_tags=("sweep", "grug", "130m", "adamh"),
)

SUGGESTIONS_FILENAME = "vizier_suggestions.json"
UPDATE_FILENAME = "vizier_update.json"
OPTIMAL_FILENAME = "vizier_optimal.json"
VIZIER_DB_FILENAME = "vizier.db"


class VizierSuggestArtifact(Artifact):
    """Output of a suggest step: a Vizier DB snapshot and a suggestions JSON."""

    @property
    def db_path(self) -> str:
        return prefix_join(self.path, VIZIER_DB_FILENAME)

    @property
    def suggestions_path(self) -> str:
        return prefix_join(self.path, SUGGESTIONS_FILENAME)


class VizierUpdateArtifact(Artifact):
    """Output of an update step: a Vizier DB snapshot with completed trial measurements."""

    @property
    def db_path(self) -> str:
        return prefix_join(self.path, VIZIER_DB_FILENAME)


# --- Data handles (shared across all training steps) ---
_NEMOTRON_WEIGHTS = {
    "hq_actual": 0.91351,
    "hq_synth": 2.72,
    "medium_high": 0.82471,
    "medium": 3.38,
    "medium_low": 1.54,
    "low_actual": 0.70123,
    "low_synth": 0.62771,
}
_STARCODER_WEIGHT = 0.25
_PROOFPILE_WEIGHT = 0.055

_nem = nemotron_datasets(tokenizer=llama3_tokenizer)
_NEMOTRON_TRAIN: dict[ArtifactStep[TokenizedCache], float] = {
    _nem[split]: weight for split, weight in _NEMOTRON_WEIGHTS.items()
}
_NEMOTRON_TRAIN[starcoder_dataset(tokenizer=llama3_tokenizer)] = _STARCODER_WEIGHT
_NEMOTRON_TRAIN[proofpile_dataset(tokenizer=llama3_tokenizer)] = _PROOFPILE_WEIGHT
_VALIDATION: list[ArtifactStep[TokenizedCache]] = [
    *paloma_datasets().values(),
    *uncheatable_datasets().values(),
]
_ALL_DATA_DEPS: tuple[ArtifactStep[TokenizedCache], ...] = (*_NEMOTRON_TRAIN, *_VALIDATION)


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
    base_launch_config: GrugBaseLaunchConfig
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


def best_run(runs: list[dict], mode: str = "min") -> dict | None:
    """Return the run with the best finite metric, or None if all are infeasible."""
    feasible = [r for r in runs if r.get("feasible", True)]
    if not feasible:
        return None
    return min(feasible, key=lambda r: r["metric"]) if mode == "min" else max(feasible, key=lambda r: r["metric"])


def _local_vizier_db_path(study_id: str) -> str:
    safe_study = re.sub(r"[^A-Za-z0-9_.-]+", "_", study_id)
    return os.path.join(tempfile.gettempdir(), f"vizier-{safe_study}.db")


def _configure_vizier_local_db(local_path: str) -> None:
    from vizier.service import clients  # noqa: PLC0415  # optional dep: vizier

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
                raise ValueError(f"Unsupported parameter value for '{key}': {raw_value!r}") from e
    return serialized


def _metric_goal(mode: str) -> Any:
    from vizier.service import pyvizier as vz  # noqa: PLC0415  # optional dep: vizier

    if mode == "min":
        return vz.ObjectiveMetricGoal.MINIMIZE
    if mode == "max":
        return vz.ObjectiveMetricGoal.MAXIMIZE
    raise ValueError(f"Unsupported metric mode: {mode}")


def _extract_adamh_hparams(suggestion: dict[str, object]) -> dict[str, float]:
    parameters = suggestion["parameters"]
    if not isinstance(parameters, Mapping):
        raise ValueError(f"Expected suggestion parameters mapping, got {type(parameters)!r}")
    required = ("lr", "beta1", "adam_lr", "beta2", "epsilon", "max_grad_norm", "z_loss_weight")
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


def run_vizier_suggest(config: VizierSuggestConfig) -> None:
    """Create or load a Vizier study, suggest trials, and persist the study DB."""
    from vizier.service import clients  # noqa: PLC0415  # optional dep: vizier
    from vizier.service import pyvizier as vz  # noqa: PLC0415  # optional dep: vizier

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

    base = config.base_launch_config
    trial_id = int(suggestion["trial_id"])

    new_tags = list(getattr(base.tracker, "tags", []) or [])
    new_tags.extend(
        [
            f"lr={hparams['lr']}",
            f"beta1={hparams['beta1']}",
            f"adam_lr={hparams['adam_lr']}",
            f"beta2={hparams['beta2']}",
            f"eps={hparams['epsilon']}",
            f"mgn={hparams['max_grad_norm']}",
            f"zloss={hparams['z_loss_weight']}",
            f"bs={batch_size}",
            f"trial={trial_id}",
            f"loop={config.loop_index}",
        ]
    )

    tracker = replace(base.tracker, tags=new_tags, name=f"trial-{trial_id}-loop-{config.loop_index}")

    launch_config = replace(
        base,
        run_id=f"{SWEEP.experiment_name}-loop{config.loop_index}-trial{trial_id}",
        steps=num_steps,
        batch_size=batch_size,
        tracker=tracker,
        optimizer=_build_adamh_config(
            learning_rate=hparams["lr"],
            beta1=hparams["beta1"],
            adam_learning_rate=hparams["adam_lr"],
            beta2=hparams["beta2"],
            epsilon=hparams["epsilon"],
            max_grad_norm=hparams["max_grad_norm"],
        ),
        z_loss_weight=hparams["z_loss_weight"],
    )
    run_grug_base_trial(launch_config)


def run_vizier_update(config: VizierUpdateConfig) -> None:
    """Load trial results, update Vizier, and write summary output."""
    from vizier.service import clients  # noqa: PLC0415  # optional dep: vizier
    from vizier.service import pyvizier as vz  # noqa: PLC0415  # optional dep: vizier

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
    if not suggestions:
        raise RuntimeError("No suggestions found")

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

        if trial.materialize().status == vz.TrialStatus.COMPLETED:
            logger.info(f"Trial {trial_id}: already completed, skipping")
        elif math.isnan(float(value)) or math.isinf(float(value)):
            trial.complete(infeasible_reason=f"metric is {value}")
            logger.info(f"Trial {trial_id}: infeasible ({config.metric_key} = {value})")
        else:
            measurement = vz.Measurement({config.metric_key: float(value)})
            trial.complete(measurement)
            logger.info(f"Trial {trial_id}: {config.metric_key} = {value}")

        feasible = math.isfinite(float(value))
        results.append(
            {
                "trial_id": trial_id,
                "metric": float(value) if feasible else None,
                "feasible": feasible,
                "hparams": suggestion["parameters"],
                "run_path": run_path,
            }
        )

    best = best_run(results, config.mode)
    if best is None:
        raise RuntimeError(f"All {len(results)} trials were infeasible (NaN/Inf loss)")

    def _sort_key(r: dict) -> tuple[bool, float]:
        m = r["metric"] or 0.0
        return (not r["feasible"], m if config.mode == "min" else -m)

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

    _sync_vizier_db_to_gcs(local_db_path, output_db_path)


def run_vizier_optimal(config: VizierOptimalConfig) -> None:
    """Load the final Vizier study and report optimal trials."""
    from vizier.service import clients  # noqa: PLC0415  # optional dep: vizier

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


def _suggest_step(
    *,
    loop_index: int,
    prev_update: ArtifactStep[VizierUpdateArtifact] | None,
    version: str,
) -> ArtifactStep[VizierSuggestArtifact]:
    client_id = f"{SWEEP.client_id_prefix}-loop-{loop_index}"

    def build_config(ctx: StepContext) -> VizierSuggestConfig:
        return VizierSuggestConfig(
            study_owner=SWEEP.study_owner,
            study_id=SWEEP.study_id,
            input_db_path=(
                VizierUpdateArtifact(path=ctx.artifact_path(prev_update)).db_path if prev_update is not None else None
            ),
            output_path=ctx.output_path,
            num_suggestions=SWEEP.suggestions_per_loop,
            client_id=client_id,
            metric_key=SWEEP.metric_key,
            mode=SWEEP.metric_mode,
            algorithm=SWEEP.vizier_algorithm,
            search_space=SWEEP.search_space,
            loop_index=loop_index,
        )

    deps = (prev_update,) if prev_update is not None else ()
    return ArtifactStep(
        name=user_namespaced_name(f"{SWEEP.experiment_name}/suggest/loop-{loop_index}", version),
        version=version,
        artifact_type=VizierSuggestArtifact,
        run=remote(run_vizier_suggest, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["vizier"]),
        build_config=build_config,
        deps=deps,
    )


def _train_step(
    *,
    loop_index: int,
    trial_index: int,
    suggest: ArtifactStep[VizierSuggestArtifact],
    version: str,
) -> ArtifactStep[Artifact]:
    def build_config(ctx: StepContext) -> VizierTrainConfig:
        steps = SWEEP.target_tokens // (SWEEP.fixed_batch_size * SWEEP.seq_len)
        data = mixture(ctx, _NEMOTRON_TRAIN, validation=_VALIDATION)
        base = GrugBaseLaunchConfig(
            model=GRUG_130M_MODEL,
            data=data,
            output_path=ctx.output_path,
            run_id=f"{SWEEP.experiment_name}-loop{loop_index}-trial{trial_index}",
            resources=ResourceConfig.with_tpu("v4-8"),
            steps=steps,
            batch_size=SWEEP.fixed_batch_size,
            seed=0,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=WandbConfig(
                project="marin",
                tags=list(SWEEP.base_train_tags),
                group=SWEEP.experiment_name,
                name=None,
                replicate_path=ctx.output_path,
            ),
            optimizer=_build_adamh_config(
                learning_rate=SWEEP.search_space["lr"][0],
                beta1=SWEEP.search_space["beta1"][0],
                adam_learning_rate=SWEEP.search_space["adam_lr"][0],
                beta2=SWEEP.search_space["beta2"][0],
                epsilon=SWEEP.search_space["epsilon"][0],
                max_grad_norm=SWEEP.search_space["max_grad_norm"][0],
            ),
            z_loss_weight=5e-6,
            steps_per_eval=500,
        )
        return VizierTrainConfig(
            suggestions_path=VizierSuggestArtifact(path=ctx.artifact_path(suggest)).suggestions_path,
            suggestion_index=trial_index,
            base_launch_config=base,
            target_tokens=SWEEP.target_tokens,
            seq_len=SWEEP.seq_len,
            fixed_batch_size=SWEEP.fixed_batch_size,
            loop_index=loop_index,
        )

    return ArtifactStep(
        name=user_namespaced_name(f"{SWEEP.experiment_name}/trial/loop-{loop_index}-trial-{trial_index}", version),
        version=version,
        artifact_type=Artifact,
        run=remote(run_vizier_train, resources=ResourceConfig.with_cpu()),
        build_config=build_config,
        deps=(suggest, *_ALL_DATA_DEPS),
    )


def _update_step(
    *,
    loop_index: int,
    suggest: ArtifactStep[VizierSuggestArtifact],
    training_steps: list[ArtifactStep[Artifact]],
    version: str,
) -> ArtifactStep[VizierUpdateArtifact]:
    def build_config(ctx: StepContext) -> VizierUpdateConfig:
        suggest_artifact = VizierSuggestArtifact(path=ctx.artifact_path(suggest))
        return VizierUpdateConfig(
            study_id=SWEEP.study_id,
            study_resource_name=SWEEP.study_resource_name,
            input_db_path=suggest_artifact.db_path,
            suggestions_path=suggest_artifact.suggestions_path,
            run_paths=[ctx.artifact_path(t) for t in training_steps],
            metric_file=SWEEP.metric_file,
            metric_key=SWEEP.metric_key,
            mode=SWEEP.metric_mode,
            output_path=ctx.output_path,
            loop_index=loop_index,
        )

    return ArtifactStep(
        name=user_namespaced_name(f"{SWEEP.experiment_name}/update/loop-{loop_index}", version),
        version=version,
        artifact_type=VizierUpdateArtifact,
        run=remote(run_vizier_update, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["vizier"]),
        build_config=build_config,
        deps=(suggest, *training_steps),
    )


def _optimal_step(
    *,
    final_update: ArtifactStep[VizierUpdateArtifact],
    version: str,
) -> ArtifactStep[Artifact]:
    def build_config(ctx: StepContext) -> VizierOptimalConfig:
        return VizierOptimalConfig(
            study_id=SWEEP.study_id,
            study_resource_name=SWEEP.study_resource_name,
            input_db_path=VizierUpdateArtifact(path=ctx.artifact_path(final_update)).db_path,
            output_path=ctx.output_path,
        )

    return ArtifactStep(
        name=user_namespaced_name(f"{SWEEP.experiment_name}/optimal", version),
        version=version,
        artifact_type=Artifact,
        run=remote(run_vizier_optimal, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["vizier"]),
        build_config=build_config,
        deps=(final_update,),
    )


def build(*, num_loops: int | None = None, version: str = "dev") -> ArtifactStep[Artifact]:
    """Build the full Vizier sweep DAG and return the terminal optimal step.

    ``StepRunner().run([build().lower()])`` materializes the entire chain.
    ``num_loops`` overrides ``SWEEP.num_loops`` (useful for CI smoke tests).
    """
    loops = num_loops if num_loops is not None else SWEEP.num_loops
    prev_update: ArtifactStep[VizierUpdateArtifact] | None = None

    for loop_index in range(loops):
        suggest = _suggest_step(loop_index=loop_index, prev_update=prev_update, version=version)
        trials = [
            _train_step(loop_index=loop_index, trial_index=i, suggest=suggest, version=version)
            for i in range(SWEEP.suggestions_per_loop)
        ]
        prev_update = _update_step(loop_index=loop_index, suggest=suggest, training_steps=trials, version=version)

    assert prev_update is not None, "num_loops must be > 0"
    return _optimal_step(final_update=prev_update, version=version)


if __name__ == "__main__":
    import os as _os

    _num_loops = 1 if _os.getenv("CI") else None
    StepRunner().run([build(num_loops=_num_loops).lower()])
