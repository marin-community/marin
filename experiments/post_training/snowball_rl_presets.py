# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball experiment presets, evaluators and input identities."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import ClassVar, Literal

import click
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from marin.execution.artifact import is_mutable_version, read_record, result_type_name
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.rl.rl_progress import Invariant
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    MegatronGeometry,
    SkyRLGenerationPlan,
    SkyRLRolePlan,
    SkyRLRun,
)
from marin.rl.skyrl_records import RESOLVED_LAUNCH_FILENAME
from marin.training.training import LevanterCheckpoint
from rigging.config_discovery import find_project_root
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join
from skyrlrecipe import ContextBudget, RecipePatch, sections
from skyrlrecipe.declared import (
    BestCheckpointCallback,
    Callback,
    CheckpointCallback,
    EvaluationCallback,
    EvaluationSampling,
    HFModelSaveCallback,
)
from skyrlrecipe.documents import MISSING, get_path, set_path
from skyrlrecipe.sampling import EvalSamplingParams
from taskcompendium.pipeline.sampling import seeded_order, seeded_sample

from experiments.post_training.task_curation.driver import CANONICAL_ARTIFACT_ROOT
from experiments.post_training.task_curation.selection import (
    FULL_VALIDATION_FILENAME,
    SELECTION_PREFIX,
    SUMMARY_FILENAME,
    TASK_MANIFEST_FILENAME,
    TASK_SPEC_COLUMN,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
)

REPO_ROOT = find_project_root(Path(__file__))
RUN_PREFIX = "checkpoints/snowball-rl"
UPDATE_SECONDS_METRIC = "speed/update_seconds"
GENERATE_WALL_SECONDS_METRIC = "generate/wall_seconds"
# Fraction of a native task set that must be scored before its scores count.
MINIMUM_SCORED_FRACTION = 0.99
SEED = 20261020
MODEL_ID = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
MODEL_REVISION = "fe666478454e79fe6027a0bab0388d9abd68f51a"
MODEL_URI = "s3://marin-us-east-02a/models/open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
SELECTION_ROOT = str(StoragePath(CANONICAL_ARTIFACT_ROOT) / SELECTION_PREFIX)
DEFAULT_SELECTION_VERSION = "2026.11.04.1"
DEFAULT_EVAL_LITE_NAME = "eval-lite"
DEFAULT_EVAL_LITE_VERSION = "2026.11.11.1"
EVAL_REPLY_TOKENS = 8192
GRAFANA_URL = "https://grafana.oa.dev/d/marin-rl-runs/rl-post-training"

_DEFAULT_TASKS = f"snowball-100k@{DEFAULT_SELECTION_VERSION}"
_DEFAULT_EVAL_LITE = f"{DEFAULT_EVAL_LITE_NAME}@{DEFAULT_EVAL_LITE_VERSION}"
_DEFAULT_MODEL = ArtifactHfModel(
    ArtifactStep.adopt(
        f"models/snowball-datakit-sft-{MODEL_REVISION}", "2026.09.21", MODEL_URI, kind=LevanterCheckpoint
    ),
    MODEL_ID,
    MODEL_REVISION,
    relative_path="",
)
_DEFAULT_PATCH = RecipePatch(
    context_budget=ContextBudget(request_window_tokens=24576, max_new_tokens_per_turn=8192, max_turns=1),
    environment=sections.Environment(task_sessions={"local": {}}),
    data=sections.Data(kind="parquet", shuffle=True),
    trainer=sections.Trainer(
        strategy="megatron",
        flash_attn=False,
        use_sample_packing=False,
        gradient_checkpointing=True,
        offload_optimizer_during_rollouts=False,
        epochs=50,
        update_epochs_per_batch=1,
        eval_batch_size=256,
        eval_before_train=False,
        logger=("console", "records", "wandb"),
        project_name="marin-snowball-rl",
        tracker_commit_each_step=True,
        training_metrics=True,
        rollout_spans=True,
        policy_train_spans=True,
        algorithm=sections.Algorithm(
            advantage_estimator="rloo_n",
            off_policy_correction="none",
            policy_loss_type="regular",
            loss_reduction="token_mean",
            use_kl_loss=False,
            kl_estimator_type="k3_unbiased_gradient",
            eps_clip_low=0.2,
            eps_clip_high=0.2,
            group_advantage_min_size=2,
            advantage_batch_normalize=False,
            rloo_n_filter_zero_reward_groups=True,
            use_entropy_loss=False,
        ),
        policy=sections.Policy(
            optimizer_config=sections.PolicyOptimizerConfig(
                optimizer="AdamW",
                lr=1e-6,
                adam_betas=(0.9, 0.999),
                weight_decay=0.0,
                max_grad_norm=0.5,
                scheduler="constant_with_warmup",
                num_warmup_steps=0,
            ),
        ),
        rollout_buffer=sections.RolloutBuffer(object_store_root=None),
        ckpt_interval=0,
        hf_save_interval=-1,
    ),
    generator=sections.Generator(
        backend="vllm",
        vllm_attention_backend="FLASH_ATTN",
        run_engines_locally=True,
        weight_sync_backend="nccl",
        weight_sync_pause=sections.WeightSyncPause(mode="keep", clear_cache=True),
        weight_sync_transport="expert_block",
        expert_block_sync=sections.ExpertBlockSync(timeout_seconds=600, verify=False),
        gpu_memory_utilization=0.75,
        max_num_seqs=1024,
        max_num_batched_tokens=8192,
        enable_prefix_caching=True,
        trajectory_reward_shaping=sections.TrajectoryRewardShaping(
            enabled=True,
            overlong=sections.Overlong(penalty_scale=0.0),
            reasoning_span=sections.ReasoningSpan(penalty=0.1, target="marker"),
        ),
        enable_chunked_prefill=True,
        enforce_eager=False,
        enable_http_endpoint=False,
        chat_template=None,
        apply_overlong_filtering=False,
        engine_init_kwargs={"moe_backend": "triton", "enable_auto_tools": True, "tool_parser": "hermes"},
        sampling_params=sections.SamplingParams(temperature=1.0, top_p=1.0, logprobs=0),
    ),
    trajectory_runner=sections.TrajectoryRunner(max_concurrent_tasks=1024),
    extra_env={
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "NCCL_NET": "Socket",
        "NCCL_IB_DISABLE": "1",
        "NCCL_SOCKET_FAMILY": "AF_INET",
    },
)


class WindowStatistic(StrEnum):
    MEAN = "mean"
    PHASE_SUM_MEDIAN = "phase-sum-median"
    GAIN_PER_H100_HOUR = "gain-per-h100-hour"
    FIRST = "first"
    SELECTED_TOKEN_WEIGHTED_MEAN = "selected-token-weighted-mean"


class Gap(StrEnum):
    LAYOUT_P40_ONLY = "layout-p40-only"
    AAII_COVERAGE = "aaii-coverage"
    NO_AGENTIC_TASKS = "no-agentic-tasks"
    NO_IMAGE_SANDBOX = "no-image-sandbox"
    MISMATCH_IS_AGREEMENT = "mismatch-is-agreement"


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

    def tasks(self, selection: ArtifactStep, default: ArtifactStep, eval_lite: str) -> ArtifactStep:
        if self.source == "default":
            return default
        if self.source == "tasks":
            return selection
        return selection_artifact(eval_lite)

    def bound_tasks(self, tasks: ArtifactStep, seed: int, *, revision: str) -> ArtifactStep:
        if self.sample is None:
            return tasks
        return sampled_tasks_step(tasks, self.file, self.sample, seed, revision=revision)


@dataclass(frozen=True)
class Training:
    plan: SkyRLRolePlan
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
        if preset.checkpoint_every:
            assert initial_model_uri is not None
            callbacks.extend(
                (
                    BestCheckpointCallback(
                        type="best_checkpoint",
                        initial_model_uri=initial_model_uri,
                        expected_tasks=expected_tasks,
                        minimum_scored_tasks=math.ceil(MINIMUM_SCORED_FRACTION * expected_tasks),
                    ),
                    HFModelSaveCallback(
                        type="hf_model_save",
                        save_steps=((preset.updates // preset.checkpoint_every) + 1) * preset.checkpoint_every,
                        save_on_train_end=True,
                    ),
                )
            )
        if preset.publishes and not preset.checkpoint_every:
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


_MODE_KEYS = (
    "trainer.rollout_buffer.max_staleness_steps",
    "trainer.rollout_buffer.batch_policy",
    "trainer.algorithm.off_policy_correction",
)


@dataclass(frozen=True)
class Recipe:
    layout: Layout
    fixed_reply_tokens: int | None
    config: RecipePatch = _DEFAULT_PATCH


@dataclass(frozen=True)
class Metric:
    key: str
    higher_is_better: bool
    window: WindowStatistic = WindowStatistic.MEAN


def override_document(settings: tuple[str, ...]) -> dict:
    """Canonical explicit user values for preset ownership and stored input identity."""
    return RecipePatch().with_settings(settings).to_skyrl()


def configure_evaluation(
    evaluation: Evaluation, config: RecipePatch, tasks: ArtifactStep, *, revision: str
) -> tuple[RecipePatch, tuple[ArtifactDataSource, ...]]:
    tasks = evaluation.bound_tasks(tasks, SEED, revision=revision)
    sampling = evaluation.profile.sampling_params
    configured = config.merge(
        RecipePatch(
            trainer=sections.Trainer(eval_num_prompts=None, dump_eval_results=True),
            generator=sections.Generator(eval_n_samples_per_prompt=evaluation.answers, eval_sampling_params=sampling),
        )
    )
    return configured, (ArtifactDataSource(tasks, relative_path=f"{evaluation.file}.parquet"),)


_HELD_OUT_SAMPLING = EvaluationSampling(
    n_samples_per_prompt=1,
    sampling_params=EvalSamplingParams(
        temperature=0.0, top_p=1.0, logprobs=None, ignore_eos=False, max_generate_length=EVAL_REPLY_TOKENS
    ),
)
_THROUGHPUT_SAMPLING = EvaluationSampling(
    n_samples_per_prompt=4,
    sampling_params=EvalSamplingParams(temperature=1.0, top_p=1.0, logprobs=None, ignore_eos=True),
)
_SCREENING_SAMPLING = EvaluationSampling(
    n_samples_per_prompt=4,
    sampling_params=EvalSamplingParams(temperature=1.0, top_p=1.0, logprobs=None, ignore_eos=False),
)
EVALUATIONS = MappingProxyType(
    {
        evaluation.name: evaluation
        for evaluation in (
            Evaluation(
                name="eval_lite",
                file=Path(FULL_VALIDATION_FILENAME).stem,
                source="eval_lite",
                profile=_HELD_OUT_SAMPLING,
            ),
            Evaluation(
                name="eval_lite_100",
                file=Path(VALIDATION_FILENAME).stem,
                source="eval_lite",
                profile=_HELD_OUT_SAMPLING,
            ),
            Evaluation(
                name="speed_512x4",
                file=Path(TRAIN_FILENAME).stem,
                sample=512,
                profile=_THROUGHPUT_SAMPLING,
            ),
            Evaluation(
                name="screen_256x4",
                file=Path(TRAIN_FILENAME).stem,
                source="tasks",
                sample=256,
                profile=_SCREENING_SAMPLING,
            ),
        )
    }
)
_P40_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=4,
    policy_num_gpus_per_node=8,
    num_inference_engines=1,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=8,
    inference_engine_expert_parallel_size=8,
    train_batch_size=128,
    policy_mini_batch_size=128,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=4,
)
P40 = Layout(
    name="p40",
    workload=Training(_P40_PLAN),
    megatron=MegatronGeometry(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
    ),
    cluster="cw-us-east-02a",
    host_memory="1800GB",
)
E8 = replace(
    P40,
    name="e8",
    workload=Generation(
        SkyRLGenerationPlan(
            gpus_per_node=8,
            num_inference_engines=1,
            inference_engine_tensor_parallel_size=1,
            inference_engine_pipeline_parallel_size=1,
            inference_engine_data_parallel_size=8,
            inference_engine_expert_parallel_size=8,
            n_samples_per_prompt=4,
        )
    ),
)
_GENERATION_DOCUMENT = {}
for _key, _value in _P40_PLAN.config_values().items():
    if _key.startswith("trainer."):
        set_path(_GENERATION_DOCUMENT, _key, _value)
_GENERATION_NATIVE_DEFAULTS = RecipePatch.from_document(_GENERATION_DOCUMENT)
_ASYNC_P40 = Recipe(
    layout=P40,
    fixed_reply_tokens=None,
    config=_DEFAULT_PATCH.merge(
        RecipePatch(
            trainer=sections.Trainer(
                rollout_buffer=sections.RolloutBuffer(max_staleness_steps=2, batch_policy="full_batch"),
                algorithm=sections.Algorithm(off_policy_correction="tis"),
            )
        )
    ),
)
_GENERATING = Recipe(layout=E8, fixed_reply_tokens=None, config=_DEFAULT_PATCH.merge(_GENERATION_NATIVE_DEFAULTS))

_GAPS = (
    Gap.AAII_COVERAGE,
    Gap.NO_AGENTIC_TASKS,
    Gap.NO_IMAGE_SANDBOX,
)
_TASK_SET = EVALUATIONS["eval_lite_100"]
BATCH_SETTINGS = {
    "trainer.train_batch_size": "train_batch_size",
    "trainer.policy_mini_batch_size": "policy_mini_batch_size",
    "generator.n_samples_per_prompt": "n_samples_per_prompt",
    "trainer.micro_train_batch_size_per_gpu": "micro_train_batch_size_per_gpu",
}

_BATCH_KEYS = frozenset(BATCH_SETTINGS)
_OWNED_KEYS = (
    frozenset(
        {
            "entrypoint",
            "trainer.resume_mode",
            "trainer.dump_eval_results",
            "trainer.placement",
            "generator.trajectory_retention",
            "generator.run_engines_locally",
            "trainer.callbacks",
            "trainer.ckpt_interval",
            "trainer.hf_save_interval",
        }
    )
    | (frozenset(_P40_PLAN.config_values()) - _BATCH_KEYS)
    | frozenset(_MODE_KEYS)
)
_GATED_KEYS = frozenset(
    {
        "trainer.max_steps",
        "trainer.logger",
        "trainer.project_name",
        "context_budget",
        "generator.sampling_params.ignore_eos",
        "generator.sampling_params.min_tokens",
    }
)
_SPEED_KEYS = frozenset(
    {"generator.sampling_params.logprobs", "trainer.training_metrics", "generator.apply_overlong_filtering"}
)


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
    tasks: str = _DEFAULT_TASKS
    model: ArtifactHfModel = _DEFAULT_MODEL
    eval_lite: str = _DEFAULT_EVAL_LITE
    evaluations: tuple[Evaluation, ...] = ()
    guards: tuple[Metric, ...] = ()
    objectives: tuple[Metric, ...] = ()
    gaps: tuple[Gap, ...] = ()
    budget: str = "hours to days"

    def __post_init__(self) -> None:
        if not set(self.measured).issubset(self.recipe.layout.workload.planned_steps(self.updates)):
            raise ValueError("measured window must lie within the planned steps")
        if self.gated and not self.guards:
            raise ValueError("gated presets require guards")
        if self.checkpoint_every is not None and (self.checkpoint_every <= 0 or not self.evaluations):
            raise ValueError("checkpoint export requires a positive checkpoint interval and an evaluation")
        for evaluation in self.evaluations[1:]:
            if evaluation.every is not None or evaluation.at_start:
                raise ValueError(f"non-primary evaluation {evaluation.name} accepts neither every nor at_start")

    @property
    def gated(self) -> bool:
        return self.controls is not None

    @property
    def held_out(self) -> Evaluation | None:
        return next((evaluation for evaluation in self.evaluations if evaluation.held_out), None)

    @property
    def guarded(self) -> Evaluation | None:
        return (self.held_out or self.evaluations[0]) if self.gated and self.evaluations else None


_LEARNING_MEDIUM = Preset(
    name="learning-medium",
    revision="2026.11.12.1",
    recipe=_ASYNC_P40,
    updates=25,
    measured=range(21, 26),
    tasks_input="forbidden",
    accepts_settings=True,
    accepts_init=True,
    publishes=True,
    controls="learning-medium",
    checkpoint_every=None,
    owned_keys=_OWNED_KEYS | _GATED_KEYS,
    data_settings=frozenset({"shuffle", "sampling"}),
    evaluations=(replace(EVALUATIONS["eval_lite"], at_start=True),),
    guards=(Metric("eval_lite/family_mean", True),),
    objectives=(
        Metric("eval_lite/family_mean", True),
        Metric("eval_lite/family_mean/gain_per_h100_hour", True, WindowStatistic.GAIN_PER_H100_HOUR),
    ),
    gaps=_GAPS,
    budget="asynchronous runtime unmeasured",
)
PRESETS = MappingProxyType(
    {
        preset.name: preset
        for preset in (
            Preset(
                name="speed-short",
                revision="2026.11.13.1",
                recipe=replace(_GENERATING, fixed_reply_tokens=4096),
                updates=0,
                measured=range(1),
                tasks_input="forbidden",
                accepts_settings=True,
                accepts_init=False,
                publishes=False,
                controls="speed-short",
                checkpoint_every=None,
                owned_keys=_OWNED_KEYS | _GATED_KEYS | _SPEED_KEYS | _BATCH_KEYS,
                data_settings=frozenset(),
                evaluations=(EVALUATIONS["speed_512x4"], EVALUATIONS["eval_lite"]),
                guards=(Metric("generate/tokens_per_second_per_gpu", True), Metric("eval_lite/family_mean", True)),
                objectives=(Metric("generate/tokens_per_second_per_gpu", True),),
                gaps=_GAPS,
                budget="14-17 min; 2 H100-h",
            ),
            Preset(
                name="data-short",
                revision="2026.11.13.1",
                recipe=_GENERATING,
                updates=0,
                measured=range(1),
                tasks_input="required",
                accepts_settings=False,
                accepts_init=False,
                publishes=False,
                controls="data-short",
                checkpoint_every=None,
                owned_keys=_OWNED_KEYS | _GATED_KEYS | _BATCH_KEYS,
                data_settings=frozenset(),
                evaluations=(EVALUATIONS["screen_256x4"],),
                guards=(Metric("screen_256x4/mixed_fraction", True),),
                objectives=(Metric("screen_256x4/mixed_fraction", True),),
                gaps=_GAPS,
                budget="10-15 min; 2 H100-h",
            ),
            Preset(
                name="speed-medium",
                revision="2026.11.12.1",
                recipe=replace(
                    _ASYNC_P40,
                    fixed_reply_tokens=4096,
                    config=_ASYNC_P40.config.merge(
                        RecipePatch(
                            trainer=sections.Trainer(
                                algorithm=sections.Algorithm(rloo_n_filter_zero_reward_groups=False)
                            )
                        )
                    ),
                ),
                updates=10,
                measured=range(2, 11),
                tasks_input="forbidden",
                accepts_settings=True,
                accepts_init=False,
                publishes=False,
                controls="speed-medium",
                checkpoint_every=None,
                owned_keys=_OWNED_KEYS | _GATED_KEYS | _SPEED_KEYS | _BATCH_KEYS,
                data_settings=frozenset(),
                guards=(
                    Metric(UPDATE_SECONDS_METRIC, False, WindowStatistic.PHASE_SUM_MEDIAN),
                    Metric("policy/mismatch/staleness0/kl_k3", False, WindowStatistic.SELECTED_TOKEN_WEIGHTED_MEAN),
                ),
                objectives=(Metric(UPDATE_SECONDS_METRIC, False, WindowStatistic.PHASE_SUM_MEDIAN),),
                gaps=(*_GAPS, Gap.MISMATCH_IS_AGREEMENT, Gap.LAYOUT_P40_ONLY),
                budget="asynchronous runtime unmeasured",
            ),
            Preset(
                name="speed-long",
                revision="2026.11.11.1",
                recipe=_ASYNC_P40,
                updates=20,
                measured=range(3, 21),
                tasks_input="forbidden",
                accepts_settings=True,
                accepts_init=False,
                publishes=False,
                controls="speed-long",
                checkpoint_every=None,
                owned_keys=_OWNED_KEYS | _GATED_KEYS | _SPEED_KEYS | _BATCH_KEYS,
                data_settings=frozenset(),
                guards=(
                    Metric(UPDATE_SECONDS_METRIC, False, WindowStatistic.PHASE_SUM_MEDIAN),
                    Metric("policy/mismatch/staleness0/kl_k3", False, WindowStatistic.SELECTED_TOKEN_WEIGHTED_MEAN),
                ),
                objectives=(Metric(UPDATE_SECONDS_METRIC, False, WindowStatistic.PHASE_SUM_MEDIAN),),
                gaps=(*_GAPS, Gap.LAYOUT_P40_ONLY),
                budget="45-75 min; 30-50 H100-h",
            ),
            _LEARNING_MEDIUM,
            replace(
                _LEARNING_MEDIUM,
                name="data-medium",
                tasks_input="required",
                accepts_settings=False,
                accepts_init=False,
                controls=_LEARNING_MEDIUM.name,
                data_settings=frozenset(),
            ),
            Preset(
                name="train-long",
                revision="2026.11.11.1",
                recipe=_ASYNC_P40,
                updates=100,
                measured=range(1, 101),
                tasks_input="optional",
                accepts_settings=True,
                accepts_init=True,
                publishes=True,
                controls=None,
                checkpoint_every=10,
                owned_keys=_OWNED_KEYS,
                data_settings=frozenset({"shuffle", "sampling"}),
                evaluations=(replace(_TASK_SET, every=10, at_start=True),),
                gaps=_GAPS,
            ),
        )
    }
)


def selection_summary(selection: ArtifactStep) -> dict:
    return json.loads((StoragePath(selection.path()) / SUMMARY_FILENAME).read_text())


def evaluation_answer_count(evaluation: Evaluation, tasks: ArtifactStep) -> int:
    key = "training_rows" if not evaluation.held_out else f"{evaluation.file}_rows"
    return (evaluation.sample or selection_summary(tasks)[key]) * evaluation.answers


def _write_sample(selection: str, filename: str, count: int, seed: int, output: str) -> None:
    fs, key = filesystem_for(prefix_join(selection, f"{filename}.parquet"))
    with fs.open(key, "rb") as stream:
        source = pq.ParquetFile(stream)
        schema = source.schema_arrow
        _, rows = seeded_sample(
            (row for batch in source.iter_batches(batch_size=256) for row in batch.to_pylist()),
            size=count,
            key=lambda row: seeded_order(json.loads(row[TASK_SPEC_COLUMN])["id"], seed),
        )
    if len(rows) != count:
        raise ValueError(f"selection has only {len(rows)} eligible tasks; {count} are required")
    fs, key = filesystem_for(prefix_join(output, f"{filename}.parquet"))
    fs.makedirs(str(PurePosixPath(key).parent), exist_ok=True)
    with fs.open(key, "wb") as stream:
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), stream)
    source_fs, source_key = filesystem_for(prefix_join(selection, TASK_MANIFEST_FILENAME))
    if source_fs.exists(source_key):
        fs, key = filesystem_for(prefix_join(output, TASK_MANIFEST_FILENAME))
        with source_fs.open(source_key, "rb") as source, fs.open(key, "wb") as target:
            target.write(source.read())


def sampled_tasks_step(selection: ArtifactStep, filename: str, count: int, seed: int, *, revision: str) -> ArtifactStep:
    identity = fingerprint_hash(json.dumps((selection.name, selection.version, filename, count, seed)))
    return apply(
        f"datasets/snowball-rl/sample-{identity}",
        _write_sample,
        version=revision,
        selection=selection,
        filename=filename,
        count=count,
        seed=seed,
        output=OUT,
    )


def apply_settings(config: RecipePatch, settings: tuple[str, ...], preset: Preset) -> RecipePatch:
    """Validate typed settings, then enforce the preset's input and measurement ownership."""
    try:
        updated = config.with_settings(settings)
        values = override_document(settings)
    except (ValueError, TypeError) as error:
        raise click.BadParameter(str(error)) from error

    def has(path: str) -> bool:
        return get_path(values, path) is not MISSING

    trainer = values.get("trainer", {})
    generator = values.get("generator", {})
    setting_sections = preset.recipe.layout.workload.setting_sections
    owners = {
        "use the preset's declared owned keys": any(has(path) for path in preset.owned_keys),
        "choose network transport in Layout": any(key.startswith("NCCL_") for key in values.get("extra_env", {})),
        "evaluation requests are fixed by the preset": any(key.startswith("eval_") for key in (*trainer, *generator)),
        "select curated data with --tasks": has("data") and bool(values["data"].keys() - preset.data_settings),
        "use the preset's permitted setting sections": (
            setting_sections is not None and (bool(values.keys() - setting_sections) or has("generator.sampling_params"))
        ),
    }
    fix = next((message for message, owned in owners.items() if owned), None)
    if fix:
        raise click.BadParameter(f"setting is owned by the preset: {fix}")
    return updated


def task_selections(preset: Preset, tasks: ArtifactStep | None = None) -> tuple[ArtifactStep, ArtifactStep]:
    default = selection_artifact(preset.tasks)
    return default, tasks or default


def cluster_config_path(cluster: str) -> str:
    return f"lib/iris/config/{cluster}.yaml"


def artifact_prefix(cluster: str) -> str:
    config = yaml.safe_load((REPO_ROOT / cluster_config_path(cluster)).read_text())
    return config["defaults"]["task_env"]["MARIN_PREFIX"]


def selection_artifact(value: str) -> ArtifactStep:
    name, separator, version = value.rpartition("@")
    if not separator or is_mutable_version(version):
        raise click.BadParameter("use NAME@VERSION with an immutable task selection version")
    return ArtifactStep.adopt(f"{SELECTION_PREFIX}/{name}", version, str(StoragePath(SELECTION_ROOT) / name / version))


def published_artifact(value: str) -> ArtifactStep:
    name, separator, version = value.rpartition("@")
    if not separator:
        raise click.BadParameter("use NAME@VERSION for published artifacts")
    if is_mutable_version(version):
        raise click.BadParameter("use an immutable artifact version instead of dev")
    root = str(StoragePath(artifact_prefix(P40.cluster)) / name / version)
    if (StoragePath(root) / RESOLVED_LAUNCH_FILENAME).exists():
        return ArtifactStep.adopt(name, version, root, kind=SkyRLRun)
    record = read_record(root)
    if record is not None:
        kind = SkyRLRun if record.result_type == result_type_name(SkyRLRun) else LevanterCheckpoint
        return ArtifactStep.adopt(name, version, record.source or root, kind=kind, config=record.config)
    raise click.BadParameter(f"no published artifact record for {name}@{version}")


def run_path(cluster: str, name: str, version: str) -> str:
    return str(StoragePath(artifact_prefix(cluster)) / name / version)


def utc_millis(value: str) -> int:
    """Parse a native receipt's UTC timestamp for duration and dashboard windows."""
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
