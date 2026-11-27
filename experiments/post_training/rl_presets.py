# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for per-model RL experiment tables."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from fractions import Fraction
from typing import Annotated, ClassVar, Literal

from finestore.rl.rl_progress import Invariant
from marin.execution.lazy import ArtifactStep
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    MegatronGeometry,
    ResolvedModelLocator,
    SkyRLGenerationPlan,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
)
from pydantic import BaseModel, Field, PositiveInt
from rolloutengine.agents.pi.agent import PI_HARNESS
from skyrlrecipe import RecipePatch, sections
from skyrlrecipe.declared import (
    BestCheckpointCallback,
    Callback,
    CheckpointCallback,
    EvaluationCallback,
    EvaluationCoverage,
    EvaluationProfile,
    HFModelSaveCallback,
    PooledCheckpointEvaluation,
)
from skyrlrecipe.locality import RoleTopologyMode
from skyrlrecipe.sampling import EvalSamplingParams

from experiments.post_training.rl_checkpoint_eval import (
    CHECKPOINT_EVALUATION_TIMEOUT,
    MINIMUM_COHORT_COVERAGE,
    CheckpointCohort,
    checkpoint_cohorts,
)
from experiments.post_training.task_curation.selection import TRAIN_FILENAME, TrainingHoldout

UPDATE_SECONDS_METRIC = "speed/update_seconds"
GENERATE_WALL_SECONDS_METRIC = "generate/wall_seconds"
MINIMUM_TASK_COVERAGE = 0.99

EVAL_LITE_PROFILE = EvaluationProfile(
    n_samples_per_prompt=1,
    sampling_params=EvalSamplingParams(
        temperature=0.0, top_p=1.0, logprobs=None, ignore_eos=False, max_generate_length=8192
    ),
)
CHECKPOINT_EVALUATION = checkpoint_cohorts()
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
    training_holdout: TrainingHoldout | None = None

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
        minimum_scored_tasks = math.ceil(preset.minimum_task_coverage * expected_tasks)
        checkpoint_scores = None
        if preset.checkpoint_evaluation:
            cohorts = {
                cohort.name: EvaluationCoverage(
                    expected_tasks=len(cohort.packages),
                    minimum_scored_tasks=math.ceil(MINIMUM_COHORT_COVERAGE * len(cohort.packages)),
                )
                for cohort in preset.checkpoint_evaluation
            }
            checkpoint_scores = PooledCheckpointEvaluation(
                primary=EvaluationCoverage(expected_tasks=expected_tasks, minimum_scored_tasks=minimum_scored_tasks),
                cohorts=cohorts,
                timeout=CHECKPOINT_EVALUATION_TIMEOUT,
            )
            expected_tasks += sum(coverage.expected_tasks for coverage in cohorts.values())
            minimum_scored_tasks += sum(coverage.minimum_scored_tasks for coverage in cohorts.values())
        if evaluation is not None:
            callbacks.append(
                EvaluationCallback(
                    type="evaluation",
                    eval_steps=evaluation.every or preset.updates + 1,
                    eval_before_train=evaluation.at_start,
                    eval_on_train_end=not evaluation.every or preset.updates % evaluation.every != 0,
                    **({"additional_evaluations": profiles} if profiles else {}),
                    **({"checkpoint_scores": checkpoint_scores} if checkpoint_scores is not None else {}),
                )
            )
        if preset.checkpoints in ("best", "best_and_final"):
            assert initial_model_uri is not None
            callbacks.append(
                BestCheckpointCallback(
                    type="best_checkpoint",
                    initial_model_uri=initial_model_uri,
                    expected_tasks=expected_tasks,
                    minimum_scored_tasks=minimum_scored_tasks,
                    **({"selection_window": preset.best_window} if preset.best_window > 1 else {}),
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
    excluded_nvlink_domains: tuple[str, ...] = ()


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

    source: str | None = None

    def __post_init__(self) -> None:
        if (self.window is WindowStatistic.GAIN_PER_GPU_HOUR) != bool(self.source):
            raise ValueError("Gain-per-GPU-hour metrics require an explicit source key")


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
    sessions = document.get("environment", {}).get("task_sessions", {})
    attempt_timeout = sessions.get("task_spec", {}).get("session", {}).get("attempt_timeout")
    if attempt_timeout is None:
        return
    daytona = sessions.get("daytona") or {}
    if "ttl_minutes" in daytona and daytona["ttl_minutes"] * 60 < attempt_timeout:
        raise ValueError("Daytona ttl_minutes must cover the task attempt timeout")
    stall = config.trainer.algorithm.group_admission.stall_timeout
    if PI_HARNESS in sessions.get("agent_harnesses", {}) and stall is not None and stall < 2 * attempt_timeout:
        raise ValueError("group_admission.stall_timeout must cover the Pi attempt and one retry")


CheckpointPolicy = Literal["none", "latest", "best", "best_and_final"]


@dataclass(frozen=True, kw_only=True)
class Preset:
    """Describe an RL preset's workload, checkpoint retention, and export."""

    name: str
    recipe: Recipe
    updates: int
    measured: range
    tasks_input: Literal["forbidden", "required", "optional"] = "forbidden"
    accepts_settings: bool = True
    accepts_init: bool = False
    checkpoint_every: int | None = None
    checkpoints: CheckpointPolicy = "best"
    keep_every: int | None = None
    owned_keys: frozenset[str] = frozenset()
    data_settings: frozenset[str] = frozenset()
    tasks: str
    model: ArtifactHfModel
    speculative_decoding_model: sections.SpeculatorModel | None = None
    evaluations: tuple[Evaluation, ...]
    seed: int
    sample_namespace: str
    wandb_entity: str
    retention: SkyRLRetentionPolicy
    temporary_output: bool = False
    export: CheckpointPolicy = "none"
    batch_mix: BatchMix | None = None
    held_out_tasks: str | None = None
    checkpoint_evaluation: tuple[CheckpointCohort, ...] = ()
    then: Callable[[ArtifactStep[SkyRLRun]], ArtifactStep] | None = None
    after_run: tuple[str, ...] = ()
    guards: tuple[Metric, ...] = ()
    objectives: tuple[Metric, ...] = ()
    gaps: tuple[str, ...] = ()
    budget: str = "hours to days"
    best_window: int = 1
    minimum_task_coverage: float = MINIMUM_TASK_COVERAGE

    def __post_init__(self) -> None:
        if not set(self.measured).issubset(self.recipe.layout.workload.planned_steps(self.updates)):
            raise ValueError("measured window must lie within the planned steps")
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
        elif self.checkpoints in ("best", "best_and_final"):
            if cadence is None:
                raise ValueError(f'checkpoints="{self.checkpoints}" requires checkpoint_every')
            if not self.evaluations:
                raise ValueError(f'checkpoints="{self.checkpoints}" requires an evaluation')
            every = self.evaluations[0].every
            if every and every % cadence:
                raise ValueError(
                    f'scheduled evaluations for checkpoints="{self.checkpoints}" must fall on checkpoint steps'
                )
        validate_agentic_recipe(self.recipe.config)
        if self.checkpoint_evaluation:
            if not isinstance(self.recipe.layout.workload, Training) or not self.evaluations:
                raise ValueError("Checkpoint cohorts require a training preset with an evaluation")
            if self.evaluations[0].answers != 1:
                raise ValueError("Pooled checkpoint evaluation requires one answer per primary task")
        for evaluation in self.evaluations[1:]:
            if evaluation.every is not None or evaluation.at_start:
                raise ValueError(f"non-primary evaluation {evaluation.name} accepts neither every nor at_start")

    @property
    def gated(self) -> bool:
        return bool(self.guards)

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
class ModelPresets:
    model: str
    presets: tuple[Preset, ...]
    namespace: str
    default_export_marker: str | None = None
    initializers: Mapping[str, ArtifactHfModel] = field(default_factory=dict)

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
