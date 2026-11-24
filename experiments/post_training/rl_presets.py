# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for per-model RL experiment tables."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Annotated, ClassVar, Literal

import yaml
from finestore.rl.rl_progress import Invariant
from marin.evaluation.eval_env import DAYTONA_EVAL_API_KEY_ENV
from marin.execution.lazy import ArtifactStep
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    MegatronGeometry,
    ResolvedModelLocator,
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
    EvaluationProfile,
    HarborEvaluationCallback,
    HarborEvaluationCohort,
    HFModelSaveCallback,
)
from skyrlrecipe.locality import RoleTopologyMode

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
        cohorts.append(HarborEvaluationCohort(name=cohort["name"], job=job, reward_key="reward"))
    return HarborEvaluationCallback(
        type="harbor_evaluation",
        cohorts=tuple(cohorts),
        timeout=config["timeout"],
        minimum_coverage=config["minimum_coverage"],
        minimum_native_coverage=MINIMUM_TASK_COVERAGE,
        daytona_api_key_env=DAYTONA_EVAL_API_KEY_ENV,
    )


HARBOR_EVALUATION = harbor_evaluation(Path(__file__).with_name("rl_harbor.yaml"))
STOP_TOKEN_IDS = (128001, 128009)
SAMPLED_TRAJECTORIES = sections.TrajectoryRetention(
    phases=("train", "eval"),
    sample_fraction=0.1,
    sample_count_per_step=0,
    required=True,
    always_retain_failures=False,
    always_retain_non_terminating=False,
    always_retain_loops=False,
)


RL_TRAINING_BASE = RecipePatch(
    environment=sections.Environment(task_sessions={"local": {}}),
    data=sections.Data(kind="parquet", shuffle=True),
    trainer=sections.Trainer(
        strategy="megatron",
        gradient_checkpointing=True,
        offload_optimizer_during_rollouts=False,
        update_epochs_per_batch=1,
        logger=("console", "records", "wandb"),
        tracker_commit_each_step=True,
        training_metrics=True,
        rollout_spans=True,
        policy_train_spans=True,
        algorithm=sections.Algorithm(
            advantage_estimator="rloo_n",
            policy_loss_type="regular",
            loss_reduction="token_mean",
            use_kl_loss=False,
            group_advantage_min_size=2,
            advantage_batch_normalize=False,
            rloo_n_filter_zero_reward_groups=True,
        ),
        rollout_buffer=sections.RolloutBuffer(object_store_root=None),
    ),
    generator=sections.Generator(
        backend="vllm",
        vllm_attention_backend="FLASH_ATTN",
        run_engines_locally=True,
        weight_sync_backend="nccl",
        weight_sync_transport="expert_block",
        weight_sync_pause=sections.WeightSyncPause(mode="keep", clear_cache=True),
        expert_block_sync=sections.ExpertBlockSync(timeout_seconds=600, verify=False),
        gpu_memory_utilization=0.75,
        max_num_batched_tokens=8192,
        chat_template=None,
        trajectory_reward_shaping=sections.TrajectoryRewardShaping(
            enabled=True,
            overlong=sections.Overlong(penalty_scale=0.0),
            reasoning_span=sections.ReasoningSpan(penalty=0.1, target="marker"),
        ),
        engine_init_kwargs={"moe_backend": "triton", "enable_auto_tools": True, "tool_parser": "hermes"},
    ),
    extra_env={"NCCL_IB_HCA": "ibp", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
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
        if preset.checkpoints != "none":
            callbacks.append(
                CheckpointCallback(type="checkpoint", save_steps=preset.checkpoint_every or preset.updates + 1)
            )
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
                count = len(cohort.job["tasks"])
                expected_tasks += count
                minimum_scored_tasks += math.ceil(preset.harbor.minimum_coverage * count)
        if preset.checkpoints == "best":
            assert initial_model_uri is not None
            callbacks.append(
                BestCheckpointCallback(
                    type="best_checkpoint",
                    initial_model_uri=initial_model_uri,
                    expected_tasks=expected_tasks,
                    minimum_scored_tasks=minimum_scored_tasks,
                )
            )
        every = preset.checkpoint_every
        if preset.export != "none":
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
    role_topology: RoleTopologyMode | None = None


@dataclass(frozen=True)
class Recipe:
    layout: Layout
    fixed_reply_tokens: int | None
    config: RecipePatch
    training_plan: SkyRLRolePlan | None = None


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
    harbor = (document.get("terminal_bench") or {}).get("harbor", {})
    attempt_timeout = harbor.get("trial_attempt_timeout_sec")
    if attempt_timeout is None:
        return
    daytona = document.get("environment", {}).get("task_sessions", {}).get("daytona", {})
    if "ttl_minutes" in daytona and daytona["ttl_minutes"] * 60 < attempt_timeout:
        raise ValueError("Daytona ttl_minutes must cover trial_attempt_timeout_sec")
    stall = config.trainer.algorithm.group_admission.stall_timeout
    if stall is not None and stall < (1 + harbor.get("max_retries", 0)) * attempt_timeout:
        raise ValueError("group_admission.stall_timeout must cover the Harbor attempt and configured retries")


CheckpointPolicy = Literal["none", "latest", "best"]


@dataclass(frozen=True, kw_only=True)
class Preset:
    """Describe an RL preset with an export policy of ``none``, ``latest``, or ``best``.

    The runner enables ``skyrl_step``'s ``export_hf`` flag for ``latest`` and ``best``.
    """

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
    checkpoints: CheckpointPolicy = "best"
    keep_every: int | None = None
    owned_keys: frozenset[str] = frozenset()
    data_settings: frozenset[str] = frozenset()
    tasks: str
    model: ArtifactHfModel
    evaluations: tuple[Evaluation, ...]
    seed: int
    sample_namespace: str
    wandb_entity: str
    retention: SkyRLRetentionPolicy
    temporary_output: bool = False
    export: CheckpointPolicy = "none"
    batch_mix: BatchMix | None = None
    held_out_tasks: str | None = None
    harbor: HarborEvaluationCallback | None = None
    then: Callable[[ArtifactStep], ArtifactStep] | None = None
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
        cadence = self.checkpoint_every
        if cadence is not None and cadence <= 0:
            raise ValueError("checkpoint_every must be positive")
        if self.keep_every is not None and self.keep_every <= 0:
            raise ValueError("keep_every must be positive")
        if self.export not in ("none", self.checkpoints):
            raise ValueError("export must be none or match checkpoints")
        if isinstance(self.recipe.layout.workload, Generation) and self.checkpoints != "none":
            raise ValueError("generation presets do not save checkpoints")
        if self.checkpoints == "none":
            if cadence is not None or self.keep_every is not None:
                raise ValueError('checkpoints="none" requires checkpoint_every and keep_every to be unset')
        elif self.checkpoints == "best":
            if cadence is None:
                raise ValueError('checkpoints="best" requires checkpoint_every')
            if not self.evaluations:
                raise ValueError('checkpoints="best" requires an evaluation')
            every = self.evaluations[0].every
            if every and every % cadence:
                raise ValueError('scheduled evaluations for checkpoints="best" must fall on checkpoint steps')
        validate_agentic_recipe(self.recipe.config)
        if self.harbor is not None:
            if not isinstance(self.recipe.layout.workload, Training) or not self.evaluations:
                raise ValueError("Harbor evaluation requires a training preset with an evaluation")
            generator = self.recipe.config.generator
            if generator is not None and generator.to_skyrl().get("enable_http_endpoint") is False:
                raise ValueError("Harbor evaluation requires generator.enable_http_endpoint")
        for evaluation in self.evaluations[1:]:
            if evaluation.every is not None or evaluation.at_start:
                raise ValueError(f"non-primary evaluation {evaluation.name} accepts neither every nor at_start")

    @property
    def gated(self) -> bool:
        return self.control_preset is not None

    @property
    def output_ttl_days(self) -> int | None:
        return self.retention.temporary_storage_ttl_days if self.temporary_output else None

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
    """Keep every completed, measured run in the cohort and retry only infrastructure failures."""

    preset: str
    commit: str
    revision: str
    members: tuple[tuple[int, int], ...]  # Ordered (native replicate, retry) pairs.
    cluster: str | None = None

    def __post_init__(self) -> None:
        if len(set(self.members)) != len(self.members):
            raise ValueError("each control pin member must name a distinct run")

    @property
    def initial_replicates(self) -> int:
        return max(replicate for replicate, _ in self.members) + 1


@dataclass(frozen=True)
class ModelPresets:
    model: str
    presets: tuple[Preset, ...]
    namespace: str
    control_pins: Mapping[str, ControlPin]
    needs_completed_default_export: bool = False

    def __post_init__(self) -> None:
        names: set[str] = set()
        for preset in self.presets:
            if preset.name in names:
                raise ValueError(f"Duplicate preset name {preset.name!r} for {self.model}")
            names.add(preset.name)

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
    model: ResolvedModelLocator | None = None
