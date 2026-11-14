# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball experiment presets, evaluators and input identities."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import click
import yaml
from marin.execution.artifact import is_mutable_version, read_record
from marin.execution.lazy import ArtifactStep
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    MegatronGeometry,
    SkyRLRolePlan,
)
from marin.training.training import LevanterCheckpoint
from rigging.config_discovery import find_project_root
from rigging.filesystem.storage_path import StoragePath
from skyrlrecipe import ContextBudget, RecipePatch, sections
from skyrlrecipe.declared import (
    Callback,
    CheckpointCallback,
    EvaluationCallback,
    EvaluationSampling,
    HFModelSaveCallback,
)
from skyrlrecipe.sampling import EvalSamplingParams

from experiments.post_training.task_curation.driver import CANONICAL_ARTIFACT_ROOT
from experiments.post_training.task_curation.inventory import (
    FULL_VALIDATION_FILENAME,
    SELECTION_PREFIX,
    TRAIN_FILENAME,
)

REPO_ROOT = find_project_root(Path(__file__))
RUN_PREFIX = "checkpoints/snowball-rl"
SEED = 20261020
MODEL_ID = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
MODEL_REVISION = "fe666478454e79fe6027a0bab0388d9abd68f51a"
MODEL_URI = "s3://marin-us-east-02a/models/open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
SELECTION_ROOT = str(StoragePath(CANONICAL_ARTIFACT_ROOT) / SELECTION_PREFIX)
DEFAULT_SELECTION_VERSION = "2026.11.04.1"
DEFAULT_EVAL_LITE_NAME = "eval-lite"
DEFAULT_EVAL_LITE_VERSION = "2026.11.11.1"
EVAL_REPLY_TOKENS = 8192

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
        logger=("console", "wandb"),
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


@dataclass(frozen=True)
class Evaluation:
    name: str
    file: str
    profile: EvaluationSampling
    source: Literal["default", "eval_lite"] = "default"
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

    def tasks(self, default: ArtifactStep, eval_lite: str) -> ArtifactStep:
        if self.source == "default":
            return default
        return selection_artifact(eval_lite)


@dataclass(frozen=True)
class Training:
    plan: SkyRLRolePlan

    def training_data(self, selection: ArtifactStep) -> tuple[ArtifactDataSource, ...]:
        return (ArtifactDataSource(selection, relative_path=TRAIN_FILENAME),)

    def batch_parameters(self) -> tuple[int, int]:
        """The per-GPU micro-batch size and the number of prompt groups in flight."""
        return self.plan.micro_train_batch_size_per_gpu, self.plan.train_batch_size

    def callbacks(self, preset: Preset, evaluation: Evaluation | None) -> tuple[Callback, ...]:
        callbacks: list[Callback] = []
        if evaluation is not None:
            callbacks.append(
                EvaluationCallback(
                    type="evaluation",
                    eval_steps=evaluation.every or preset.updates + 1,
                    eval_before_train=evaluation.at_start,
                    eval_on_train_end=not evaluation.every or preset.updates % evaluation.every != 0,
                )
            )
        if preset.publishes:
            callbacks.extend(
                (
                    CheckpointCallback(type="checkpoint", save_steps=preset.updates + 1, save_on_train_end=True),
                    HFModelSaveCallback(type="hf_model_save", save_steps=-1, save_on_train_end=True),
                )
            )
        return tuple(callbacks)


@dataclass(frozen=True)
class Layout:
    name: str
    workload: Training
    megatron: MegatronGeometry
    cluster: str
    host_memory: str


@dataclass(frozen=True)
class Recipe:
    layout: Layout
    config: RecipePatch = _DEFAULT_PATCH


def configure_evaluation(
    evaluation: Evaluation, config: RecipePatch, tasks: ArtifactStep
) -> tuple[RecipePatch, tuple[ArtifactDataSource, ...]]:
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
    sampling_params=EvalSamplingParams(temperature=0.0, top_p=1.0, logprobs=None, max_generate_length=EVAL_REPLY_TOKENS),
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
_ASYNC_P40 = Recipe(
    layout=P40,
    config=_DEFAULT_PATCH.merge(
        RecipePatch(
            trainer=sections.Trainer(
                rollout_buffer=sections.RolloutBuffer(max_staleness_steps=2, batch_policy="full_batch"),
                algorithm=sections.Algorithm(off_policy_correction="tis"),
            )
        )
    ),
)


@dataclass(frozen=True)
class Preset:
    name: str
    revision: str
    recipe: Recipe
    updates: int
    publishes: bool
    tasks: str = _DEFAULT_TASKS
    model: ArtifactHfModel = _DEFAULT_MODEL
    eval_lite: str = _DEFAULT_EVAL_LITE
    evaluations: tuple[Evaluation, ...] = ()


_LEARNING_MEDIUM = Preset(
    name="learning-medium",
    revision="2026.11.12.1",
    recipe=_ASYNC_P40,
    updates=25,
    publishes=True,
    evaluations=(replace(EVALUATIONS["eval_lite"], at_start=True),),
)
PRESETS = MappingProxyType({preset.name: preset for preset in (_LEARNING_MEDIUM,)})


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
    record = read_record(root)
    if record is None:
        raise click.BadParameter(f"no published artifact record for {name}@{version}")
    return ArtifactStep.adopt(name, version, record.source or root, kind=LevanterCheckpoint, config=record.config)
