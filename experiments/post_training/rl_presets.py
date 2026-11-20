# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for per-model RL experiment tables."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Annotated, ClassVar, Literal

import yaml
from finestore.rl.rl_progress import Invariant
from marin.execution.lazy import ArtifactStep
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    MegatronGeometry,
    SkyRLGenerationPlan,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
)
from pydantic import BaseModel, Field
from skyrlrecipe import RecipePatch, sections
from skyrlrecipe.declared import (
    BestCheckpointCallback,
    Callback,
    CheckpointCallback,
    EvaluationCallback,
    EvaluationProfile,
    HarborEvaluationCallback,
    HFModelSaveCallback,
)

from experiments.post_training.task_curation.selection import TRAIN_FILENAME

UPDATE_SECONDS_METRIC = "speed/update_seconds"
GENERATE_WALL_SECONDS_METRIC = "generate/wall_seconds"
MINIMUM_TASK_COVERAGE = 0.99


def harbor_evaluation(path: Path) -> HarborEvaluationCallback:
    """Load the pinned cohorts and job settings for in-training Harbor evaluation."""
    config = yaml.safe_load(path.read_text())
    cohorts = []
    for cohort in config["cohorts"]:
        job = deepcopy({**config["job"], **cohort.get("job", {})})
        job["tasks"] = [
            {
                "git_url": cohort["git_url"],
                "git_commit_id": cohort["git_commit_id"],
                "path": task_path,
                "source": cohort["name"],
            }
            for task_path in cohort["paths"]
        ]
        cohorts.append({"name": cohort["name"], "job": job, "reward_key": "reward"})
    return HarborEvaluationCallback(
        type="harbor_evaluation",
        cohorts=tuple(cohorts),
        timeout=config["timeout"],
        minimum_coverage=config["minimum_coverage"],
        minimum_native_coverage=MINIMUM_TASK_COVERAGE,
        daytona_api_key_env="DAYTONA_EVAL_API_KEY",
    )


class WindowStatistic(StrEnum):
    MEAN = "mean"
    PHASE_SUM_MEDIAN = "phase-sum-median"
    GAIN_PER_GPU_HOUR = "gain-per-gpu-hour"
    FIRST = "first"
    SELECTED_TOKEN_WEIGHTED_MEAN = "selected-token-weighted-mean"


@dataclass(frozen=True)
class Evaluation:
    name: str
    file: str
    profile: EvaluationProfile
    source: Literal["default", "tasks", "held_out"] = "default"
    sample: int | None = None
    every: int | None = None
    at_start: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError(f"evaluation name must be an identifier: {self.name!r}")
        if self.profile.n_samples_per_prompt is None or self.profile.sampling_params is None:
            raise ValueError("evaluation profiles require explicit sampling and answer counts")

    @property
    def answers(self) -> int:
        assert self.profile.n_samples_per_prompt is not None
        return self.profile.n_samples_per_prompt

    @property
    def held_out(self) -> bool:
        return self.source == "held_out"


@dataclass(frozen=True)
class Training:
    plan: SkyRLRolePlan
    completion_metric: ClassVar[str] = "timing/step_wall/unaccounted"
    outcome_prefix: ClassVar[str] = "generate/outcome/"
    additional_schedule: ClassVar[Literal["final"]] = "final"
    setting_sections: ClassVar[frozenset[str] | None] = None

    def update_count(self, max_steps: int) -> int:
        return max_steps

    def planned_steps(self, updates: int) -> range:
        return range(1, updates + 1)

    def scored_step(self, updates: int) -> int:
        return updates

    def metric_step(self, updates: int) -> int:
        return updates

    def completed_row(self, row: Mapping) -> bool:
        return row.get("trainer/global_step") == row["step"]

    def training_data(self, selection: ArtifactStep) -> tuple[ArtifactDataSource, ...]:
        return (ArtifactDataSource(selection, relative_path=TRAIN_FILENAME),)

    def batch_parameters(self, config: RecipePatch) -> tuple[int, int]:
        return self.plan.micro_train_batch_size_per_gpu, self.plan.train_batch_size

    def with_batches(self, changes: dict[str, int]) -> Training:
        return replace(self, plan=replace(self.plan, **changes))

    def invariants(self, updates: int, metrics: Mapping[int, Mapping[str, float]]) -> tuple[Invariant, ...]:
        planned = set(self.planned_steps(updates))
        completed = planned.intersection(metrics)
        training = [metrics[step] for step in sorted(completed)]
        finite = all(
            all(
                math.isfinite(row.get(key, math.nan))
                for key in ("policy/raw_grad_norm", "policy/policy_loss", "policy/final_loss")
            )
            and any(key.startswith("loss/") for key in row)
            and all(math.isfinite(row[key]) for key in row if key.startswith("loss/") or key == "policy/raw_grad_norm")
            for row in training
        )
        return (
            Invariant(
                name="planned_updates",
                value=len(completed),
                limit=len(planned),
                comparator="==",
                holds=completed == planned,
            ),
            Invariant(name="finite_losses_and_gradients", value=finite, limit=True, comparator="==", holds=finite),
        )

    def fixed_reply_holds(self, fixed: int, training: list[dict], response_token_counts: set[int | None]) -> bool:
        return all(row.get("generate/min_num_tokens") == row.get("generate/max_num_tokens") == fixed for row in training)

    def evaluation_patch(self, answer_count: int) -> RecipePatch:
        return RecipePatch()

    def callbacks(
        self,
        preset: Preset,
        evaluation: Evaluation | None,
        profiles: dict[str, EvaluationProfile],
        expected_tasks: int,
        initial_model_uri: str | None,
    ) -> tuple[Callback, ...]:
        callbacks: list[Callback] = []
        if preset.checkpoint_every:
            callbacks.append(CheckpointCallback(type="checkpoint", save_steps=preset.checkpoint_every))
        if evaluation is not None:
            callbacks.append(
                EvaluationCallback(
                    type="evaluation",
                    eval_steps=evaluation.every or preset.updates + 1,
                    eval_before_train=evaluation.at_start,
                    eval_on_train_end=not evaluation.every or preset.updates % evaluation.every != 0,
                    **({"additional_evaluations": profiles} if profiles else {}),
                )
            )
        minimum_scored_tasks = math.ceil(MINIMUM_TASK_COVERAGE * expected_tasks)
        if preset.harbor is not None:
            callbacks.append(preset.harbor)
            for cohort in preset.harbor.cohorts:
                count = len(cohort["job"]["tasks"])
                expected_tasks += count
                minimum_scored_tasks += math.ceil(preset.harbor.minimum_coverage * count)
        if initial_model_uri is not None:
            callbacks.append(
                BestCheckpointCallback(
                    type="best_checkpoint",
                    initial_model_uri=initial_model_uri,
                    expected_tasks=expected_tasks,
                    minimum_scored_tasks=minimum_scored_tasks,
                )
            )
        if preset.checkpoint_every:
            callbacks.append(
                HFModelSaveCallback(
                    type="hf_model_save",
                    save_steps=((preset.updates // preset.checkpoint_every) + 1) * preset.checkpoint_every,
                    save_on_train_end=True,
                )
            )
        if preset.export_hf and not preset.checkpoint_every:
            callbacks.extend(
                (
                    CheckpointCallback(type="checkpoint", save_steps=preset.updates + 1, save_on_train_end=True),
                    HFModelSaveCallback(type="hf_model_save", save_steps=-1, save_on_train_end=True),
                )
            )
        return tuple(callbacks)


@dataclass(frozen=True)
class Generation:
    plan: SkyRLGenerationPlan
    completion_metric: ClassVar[str] = GENERATE_WALL_SECONDS_METRIC
    outcome_prefix: ClassVar[str] = "eval/all/outcome/"
    additional_schedule: ClassVar[Literal["every_evaluation"]] = "every_evaluation"
    setting_sections: ClassVar[frozenset[str]] = frozenset({"generator"})

    def update_count(self, max_steps: int) -> int:
        return 0

    def planned_steps(self, updates: int) -> range:
        return range(1)

    def scored_step(self, updates: int) -> int:
        return -1

    def metric_step(self, updates: int) -> int:
        return 0

    def completed_row(self, row: Mapping) -> bool:
        return True

    def training_data(self, selection: ArtifactStep) -> tuple[ArtifactDataSource, ...]:
        return ()

    def batch_parameters(self, config: RecipePatch) -> tuple[int, int]:
        return config.trainer.micro_train_batch_size_per_gpu, config.trainer.train_batch_size

    def with_batches(self, changes: dict[str, int]) -> Generation:
        return replace(
            self,
            plan=replace(
                self.plan, n_samples_per_prompt=changes.get("n_samples_per_prompt", self.plan.n_samples_per_prompt)
            ),
        )

    def invariants(self, updates: int, metrics: Mapping[int, Mapping[str, float]]) -> tuple[Invariant, ...]:
        return (
            Invariant(name="planned_updates", value=int(0 in metrics), limit=1, comparator="==", holds=0 in metrics),
        )

    def fixed_reply_holds(self, fixed: int, training: list[dict], response_token_counts: set[int | None]) -> bool:
        return response_token_counts == {fixed}

    def evaluation_patch(self, answer_count: int) -> RecipePatch:
        return RecipePatch(trajectory_runner=sections.TrajectoryRunner(max_concurrent_tasks=answer_count))

    def callbacks(
        self,
        preset: Preset,
        evaluation: Evaluation | None,
        profiles: dict[str, EvaluationProfile],
        expected_tasks: int,
        initial_model_uri: str | None,
    ) -> tuple[Callback, ...]:
        return (EvaluationCallback(type="evaluation", additional_evaluations=profiles),) if profiles else ()


@dataclass(frozen=True)
class Layout:
    name: str
    workload: Training | Generation
    megatron: MegatronGeometry
    cluster: str
    host_memory: str
    gpu_variant: str
    host_cpu: int
    host_disk: str
    priority: str
    coordinator_timeout_hours: int
    job_timeout_seconds: int


@dataclass(frozen=True)
class Recipe:
    layout: Layout
    fixed_reply_tokens: int | None
    config: RecipePatch


@dataclass(frozen=True)
class Metric:
    key: str
    higher_is_better: bool
    window: WindowStatistic = WindowStatistic.MEAN


@dataclass(frozen=True, kw_only=True)
class Preset:
    name: str
    revision: str
    recipe: Recipe
    updates: int
    measured: range
    tasks_input: Literal["forbidden", "required", "optional"] = "forbidden"
    accepts_settings: bool = True
    accepts_init: bool = False
    publishes: bool = False
    control_preset: str | None = None
    checkpoint_every: int | None = None
    owned_keys: frozenset[str] = frozenset()
    data_settings: frozenset[str] = frozenset()
    tasks: str
    model: ArtifactHfModel
    evaluations: tuple[Evaluation, ...]
    seed: int
    sample_namespace: str
    wandb_entity: str
    retention: SkyRLRetentionPolicy
    export_hf: bool = False
    held_out_tasks: str | None = None
    harbor: HarborEvaluationCallback | None = None
    guards: tuple[Metric, ...] = ()
    objectives: tuple[Metric, ...] = ()
    gaps: tuple[str, ...] = ()
    budget: str = "hours to days"

    def __post_init__(self) -> None:
        if not set(self.measured).issubset(self.recipe.layout.workload.planned_steps(self.updates)):
            raise ValueError("measured window must lie within the planned steps")
        if self.gated and not self.guards:
            raise ValueError("gated presets require guards")
        if self.held_out is not None and not self.held_out_tasks:
            raise ValueError("held-out evaluations require a held-out task selection")
        if self.checkpoint_every is not None and (self.checkpoint_every <= 0 or not self.evaluations):
            raise ValueError("checkpoint export requires a positive checkpoint interval and an evaluation")
        for evaluation in self.evaluations[1:]:
            if evaluation.every is not None or evaluation.at_start:
                raise ValueError(f"non-primary evaluation {evaluation.name} accepts neither every nor at_start")

    @property
    def gated(self) -> bool:
        return self.control_preset is not None

    def with_updates(self, updates: int) -> Preset:
        return replace(
            self,
            updates=updates,
            measured=self.measured if self.gated else self.recipe.layout.workload.planned_steps(updates),
        )

    @property
    def held_out(self) -> Evaluation | None:
        return next((evaluation for evaluation in self.evaluations if evaluation.held_out), None)

    @property
    def guarded(self) -> Evaluation | None:
        return (self.held_out or self.evaluations[0]) if self.gated and self.evaluations else None


@dataclass(frozen=True)
class ControlPin:
    preset: str
    commit: str
    revision: str
    replicates: int


@dataclass(frozen=True)
class ModelPresets:
    model: str
    presets: tuple[Preset, ...]
    namespace: str
    control_pins: Mapping[str, ControlPin]

    @property
    def by_name(self) -> Mapping[str, Preset]:
        return {preset.name: preset for preset in self.presets}


class HfInitializer(BaseModel, frozen=True):
    kind: Literal["hf"] = "hf"
    artifact_name: str
    artifact_version: str
    artifact_root: str
    fingerprint_payload: str
    relative_path: str | None
    tokenizer_uri: str | None = None
    tokenizer_revision: str | None = None


class PolicyInitializer(BaseModel, frozen=True):
    kind: Literal["policy"] = "policy"
    artifact_name: str
    artifact_version: str
    artifact_root: str
    fingerprint_payload: str


type StoredInitializer = Annotated[HfInitializer | PolicyInitializer, Field(discriminator="kind")]
