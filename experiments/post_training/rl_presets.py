# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for per-model RL experiment tables."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from fractions import Fraction
from typing import Annotated, ClassVar, Literal

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
from pydantic import BaseModel, Field, PositiveInt
from skyrlrecipe import RecipePatch, sections
from skyrlrecipe.declared import (
    BestCheckpointCallback,
    Callback,
    CheckpointCallback,
    EvaluationCallback,
    EvaluationSampling,
    HarborEvaluationCallback,
    HFModelSaveCallback,
)
from skyrlrecipe.locality import RoleTopologyMode

from experiments.post_training.task_curation.selection import (
    TRAIN_FILENAME,
)

UPDATE_SECONDS_METRIC = "speed/update_seconds"


GENERATE_WALL_SECONDS_METRIC = "generate/wall_seconds"


MINIMUM_SCORED_FRACTION = 0.99


class WindowStatistic(StrEnum):
    MEAN = "mean"
    PHASE_SUM_MEDIAN = "phase-sum-median"
    GAIN_PER_GPU_HOUR = "gain-per-h100-hour"
    FIRST = "first"
    SELECTED_TOKEN_WEIGHTED_MEAN = "selected-token-weighted-mean"


@dataclass(frozen=True)
class Evaluation:
    name: str
    file: str
    profile: EvaluationSampling
    source: Literal["default", "tasks", "eval_lite"] = "default"
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
        return self.source == "eval_lite"


@dataclass(frozen=True)
class Training:
    plan: SkyRLRolePlan
    epochs_follow_updates: bool = False
    completion_metric: ClassVar[str] = "timing/step_wall/unaccounted"
    outcome_prefix: ClassVar[str] = "generate/outcome/"
    named_schedule: ClassVar[Literal["final"]] = "final"
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

    def fixed_reply_holds(self, fixed: int, training: list[dict], answers: list) -> bool:
        return all(row.get("generate/min_num_tokens") == row.get("generate/max_num_tokens") == fixed for row in training)

    def evaluation_patch(self, answer_count: int) -> RecipePatch:
        return RecipePatch()

    def callbacks(
        self,
        preset: Preset,
        evaluation: Evaluation | None,
        profiles: dict[str, EvaluationSampling],
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
        minimum_scored_tasks = math.ceil(MINIMUM_SCORED_FRACTION * expected_tasks)
        if preset.harbor is not None:
            callbacks.append(preset.harbor)
            for cohort in preset.harbor.cohorts:
                count = len(cohort["job"]["tasks"])
                expected_tasks += count
                minimum_scored_tasks += math.ceil(preset.harbor.minimum_coverage * count)
        if preset.checkpoint_every and evaluation is not None:
            assert initial_model_uri is not None
            callbacks.append(
                BestCheckpointCallback(
                    type="best_checkpoint",
                    initial_model_uri=initial_model_uri,
                    expected_tasks=expected_tasks,
                    minimum_scored_tasks=minimum_scored_tasks,
                )
            )
        if preset.export != "none":
            every = preset.checkpoint_every
            if every is None:
                callbacks.append(
                    CheckpointCallback(type="checkpoint", save_steps=preset.updates + 1, save_on_train_end=True)
                )
            callbacks.append(
                HFModelSaveCallback(
                    type="hf_model_save",
                    save_steps=(preset.updates // every + 1) * every if every else -1,
                    save_on_train_end=True,
                )
            )
        return tuple(callbacks)


@dataclass(frozen=True)
class Generation:
    plan: SkyRLGenerationPlan
    training_reference_plan: SkyRLRolePlan | None = None
    completion_metric: ClassVar[str] = GENERATE_WALL_SECONDS_METRIC
    outcome_prefix: ClassVar[str] = "eval/all/outcome/"
    named_schedule: ClassVar[Literal["every_evaluation"]] = "every_evaluation"
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

    def fixed_reply_holds(self, fixed: int, training: list[dict], answers: list) -> bool:
        return {len(row["response_token_ids"]) for _, _, row in answers} == {fixed}

    def evaluation_patch(self, answer_count: int) -> RecipePatch:
        return RecipePatch(trajectory_runner=sections.TrajectoryRunner(max_concurrent_tasks=answer_count))

    def callbacks(
        self,
        preset: Preset,
        evaluation: Evaluation | None,
        profiles: dict[str, EvaluationSampling],
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
    role_topology: RoleTopologyMode | None = None


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


@dataclass(frozen=True)
class BatchMix:
    """Exact shares of prompt groups per training update."""

    shares: Mapping[str, Fraction]

    def __post_init__(self) -> None:
        if any(not family or share <= 0 for family, share in self.shares.items()) or sum(self.shares.values()) != 1:
            raise ValueError("batch mix needs named families with positive shares that sum to 1")

    def groups(self, batch_size: int, selection_families: set[str]) -> dict[str, int]:
        if self.shares.keys() != selection_families:
            raise ValueError(
                f"batch mix names {sorted(self.shares)}, but the training selection has {sorted(selection_families)}"
            )
        counts = {family: share * batch_size for family, share in sorted(self.shares.items())}
        if any(count.denominator != 1 for count in counts.values()):
            raise ValueError(f"batch mix needs a whole number of groups per family at batch size {batch_size}")
        return {family: int(count) for family, count in counts.items()}


def validate_agentic_recipe(config: RecipePatch) -> None:
    document = config.to_skyrl()
    terminal = document.get("terminal_bench")
    if terminal is None:
        return
    harbor = terminal.get("harbor", {})
    daytona = document.get("environment", {}).get("task_sessions", {}).get("daytona", {})
    attempt_timeout = harbor.get("trial_attempt_timeout_sec")
    if attempt_timeout is not None and "ttl_minutes" in daytona and daytona["ttl_minutes"] * 60 < attempt_timeout:
        raise ValueError("Daytona ttl_minutes must cover trial_attempt_timeout_sec")
    stall = config.trainer.algorithm.group_admission.stall_timeout
    if (
        stall is not None
        and attempt_timeout is not None
        and harbor.get("max_retries", 0)
        and stall < 2 * attempt_timeout
    ):
        raise ValueError("group_admission.stall_timeout must cover one Harbor attempt and retry")
    agent = terminal.get("installed_agent", {}).get("agent", {})
    if agent.get("name") == "pi":
        allowed = {
            "version",
            "mcp_adapter_version",
            "thinking",
            "thinking_format",
            "compaction_reserve_tokens",
            "http_idle_timeout",
        }
        if unknown := sorted(agent.get("kwargs", {}).keys() - allowed):
            raise ValueError(f"Unknown Pi recipe kwargs: {', '.join(unknown)}")


@dataclass(frozen=True)
class Preset:
    name: str
    revision: str
    recipe: Recipe
    updates: int
    measured: range
    tasks_input: Literal["forbidden", "required", "optional"]
    accepts_settings: bool
    accepts_init: bool
    publishes: bool
    controls: str | None
    checkpoint_every: int | None
    owned_keys: frozenset[str]
    data_settings: frozenset[str]
    tasks: str
    model: ArtifactHfModel
    eval_lite: str
    evaluations: tuple[Evaluation, ...]
    seed: int
    sample_namespace: str
    wandb_entity: str
    retention: SkyRLRetentionPolicy
    export: Literal["none", "latest", "best"]
    batch_mix: BatchMix | None = None
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
        if self.batch_mix is not None and not isinstance(self.recipe.layout.workload, Training):
            raise ValueError("batch mix requires a training workload")
        if self.checkpoint_every is not None and self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if isinstance(self.recipe.layout.workload, Generation) and (self.checkpoint_every or self.export != "none"):
            raise ValueError("checkpointing and export require a training workload")
        if self.export == "best" and (self.checkpoint_every is None or not self.evaluations):
            raise ValueError("best export requires checkpointing and an evaluation")
        if self.export == "best" and self.evaluations[0].every:
            assert self.checkpoint_every is not None
            if self.evaluations[0].every % self.checkpoint_every:
                raise ValueError("scheduled evaluations for best export must fall on checkpoint steps")
        validate_agentic_recipe(self.recipe.config)
        for evaluation in self.evaluations[1:]:
            if evaluation.every is not None or evaluation.at_start:
                raise ValueError(f"non-primary evaluation {evaluation.name} accepts neither every nor at_start")

    @property
    def gated(self) -> bool:
        return self.controls is not None

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
    runs: int


@dataclass(frozen=True)
class ModelPresets:
    model: str
    presets: tuple[Preset, ...]
    namespace: str
    controls: Mapping[str, ControlPin]

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


class StoredResume(BaseModel, frozen=True):
    artifact_name: str
    artifact_version: str
    step: PositiveInt
    checkpoint: str
