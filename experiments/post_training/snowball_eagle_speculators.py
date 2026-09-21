# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train and benchmark a reusable Snowball EAGLE-3 draft.

The corpus consists of fresh responses from the target checkpoint across math,
code, instruction-following, science, finance, tool-use, terminal, and software
engineering tasks. Run on the US East 02A controller so the target checkpoint
and generated hidden states stay in the same object-store region::

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
      --enable-extra-resources --target-cluster cw-us-east-02a \
      -- python experiments/post_training/snowball_eagle_speculators.py \
      --version 2026.09.21 --stage draft --run

The agentic benchmark stage requires ``DAYTONA_API_KEY`` in the coordinator
environment. The Harbor corpus jobs resolve that secret through their task
configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import click
import yaml
from fray.types import ResourceConfig
from marin.evaluation.evalchemy.config import EvalchemyConfig, load_evalchemy_config
from marin.evaluation.evalchemy.result import FineStoreEvalchemyResult
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import GenerationConfig, ModelConfig, ResourceHint, ServeConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.evaluation import evaluate_evalchemy
from marin.experiment.namespacing import user_owned_name
from marin.external_dependencies import SPECULATORS
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    EagleDraftArtifact,
    IrisSkyRLExecution,
    SkyRLDataSource,
    SkyRLModel,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    TaskTroveDataSource,
    TaskTroveSelection,
    skyrl_step,
)
from marin.training.speculators import (
    SPECULATORS_DATA_FILENAME,
    DraftTrainingConfig,
    HfSnapshotConfig,
    HiddenStateCaptureConfig,
    RolloutConversationConfig,
    VerifierViewConfig,
    build_verifier_view,
    capture_hidden_states,
    mirror_hf_snapshot,
    train_draft,
    write_rollout_conversations,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import prefix_join

from experiments.evaluation.evals import evalchemy_run_config
from experiments.evaluation.models import SNOWBALL_VLLM_ARGS
from experiments.evaluation.pipeline import EvaluationResult
from experiments.evaluation.pipeline import eval_step as evaluation_step
from experiments.post_training.curriculum_rl.launch import (
    GPUS_PER_NODE,
    POOL_ARTIFACT_NAME,
)
from experiments.post_training.curriculum_rl.pool import TRAIN_FILENAME, VALIDATION_FILENAME, pool_step

VERSION = "2026.09.21"
CORPUS_EVALS = (
    "olympiadbench",
    "gsm8k",
    "mbppplus",
    "cruxeval",
    "ifeval",
    "gpqa-diamond",
    "financebench",
)
CORPUS_SEEDS = (17, 29, 43)
CORPUS_LIMIT_PER_TASK = 2048
AGENTIC_CORPUS_EVALS = "bfcl,tb2,swebench-full"
AGENTIC_CORPUS_REPETITIONS = (1, 2, 3)
AGENTIC_CORPUS_LIMIT_PER_TASK = 512
CORPUS_MAX_SAMPLES = 16_384
CORPUS_MINIMUM_VALID_TOKENS = 32
_EVALCHEMY_CONFIG_DIR = Path(__file__).parents[1] / "evaluation" / "configs" / "evalchemy"
TARGET_MODEL_NAME = "snowball-67b-a2b-sft-s3-agentic-step1903"
TARGET_MODEL_URI = "s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b-sft-s3-agentic/step-1903/hf-bf16-vllm/"
TARGET_TOKENIZER = "penfever/grug-67b-a2b-sft-s2-thinking-step630-tok"
TARGET_TOKENIZER_REVISION = "f0eac008b7fcd67025266a260d8722dbfd36e819"
TARGET_MODEL = ArtifactStep.adopt(
    f"models/{TARGET_MODEL_NAME}",
    "2026.09.21",
    TARGET_MODEL_URI,
    kind=LevanterCheckpoint,
)
INITIAL_DRAFT_REPO = "laion/snowball-64k-eagle3-draft-r2egym"
INITIAL_DRAFT_REVISION = "4bdb47c08e5b5190bea3c7a93c3e14470230e469"
TARGET_LAYER_IDS = (2, 13, 23)
VERIFIER_NUM_HIDDEN_LAYERS = 26
SEQUENCE_LENGTH = 32768
RL_DATA_VERSION = "2026.09.18"
RL_ARTIFACT_NAME = "checkpoints/snowball-e3-bench"
TASKTROVE_RELEASE = ArtifactStep.adopt(
    "tasktrove/clean",
    "2026.09.18.3",
    "s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.18.3",
)
AGENTIC_BENCHMARK_SOURCE = "DCAgent2__nl2bash-tasks-cleaned-oracle-v2"
CLUSTER = "cw-us-east-02a"
GPU_VARIANT = "H100"
_DRAFT_GPU_COUNT = 8
_BENCHMARK_PROMPTS = 128
_BENCHMARK_SAMPLES_PER_PROMPT = 4
_AGENTIC_BENCHMARK_PROMPTS = 128
# Speculators installs torchaudio through its multimodal dependencies. Pin the
# CUDA 12.8 wheel used by the Iris H100 PyTorch runtime so Transformers imports.
_TORCHAUDIO_CU128_REQUIREMENT = (
    "torchaudio @ https://download.pytorch.org/whl/cu128/"
    "torchaudio-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl"
    "#sha256=78b86a17f164bdaabdcee93fdfde2587fc43b9ebf15cd61dcf730b4f8615176b"
)


def _rl_benchmark_role_plan() -> SkyRLRolePlan:
    return SkyRLRolePlan(
        colocate_all=False,
        policy_num_nodes=4,
        policy_num_gpus_per_node=GPUS_PER_NODE,
        num_inference_engines=GPUS_PER_NODE,
        inference_engine_tensor_parallel_size=1,
        inference_engine_data_parallel_size=GPUS_PER_NODE,
        inference_engine_expert_parallel_size=GPUS_PER_NODE,
        train_batch_size=_BENCHMARK_PROMPTS,
        policy_mini_batch_size=64,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=_BENCHMARK_SAMPLES_PER_PROMPT,
    )


def _rl_benchmark_config(role_plan: SkyRLRolePlan, *, speculative: bool) -> str:
    config = {
        "entrypoint": "standard",
        "context_budget": {
            "request_window_tokens": 9856,
            "max_new_tokens_per_turn": 8192,
            "max_turns": 1,
        },
        "environment": {"env_class": "aime"},
        "trainer": {
            "strategy": "megatron",
            "flash_attn": False,
            "use_sample_packing": False,
            "offload_optimizer_during_rollouts": True,
            "gradient_checkpointing": True,
            "algorithm": {
                "advantage_estimator": "rloo_n",
                "group_advantage_min_size": 4,
                "use_kl_loss": False,
            },
            "epochs": 1,
            "max_steps": 1,
            "update_epochs_per_batch": 1,
            "train_batch_size": role_plan.train_batch_size,
            "policy_mini_batch_size": role_plan.policy_mini_batch_size,
            "eval_batch_size": _BENCHMARK_PROMPTS,
            "micro_forward_batch_size_per_gpu": 1,
            "micro_train_batch_size_per_gpu": role_plan.micro_train_batch_size_per_gpu,
            "eval_before_train": False,
            "eval_interval": -1,
            "ckpt_interval": 100,
            "resume_mode": "none",
            "logger": "console",
            "policy": {
                "optimizer_config": {"lr": 1.0e-6, "max_grad_norm": 1.0},
                "megatron_config": {
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 2,
                    "context_parallel_size": 1,
                    "expert_model_parallel_size": 8,
                    "expert_tensor_parallel_size": 1,
                    "optimizer_checkpoint_sharding_type": "dp_reshardable",
                    "ddp_config": {
                        "overlap_grad_reduce": True,
                        "overlap_param_gather": True,
                        "grad_reduce_in_fp32": False,
                    },
                },
            },
            "placement": {
                "colocate_all": role_plan.colocate_all,
                "policy_strict_spread_pg": True,
                "policy_num_nodes": role_plan.policy_num_nodes,
                "policy_num_gpus_per_node": role_plan.policy_num_gpus_per_node,
            },
        },
        "generator": {
            "backend": "vllm",
            "model_dtype": "bfloat16",
            "vllm_attention_backend": "FLASH_ATTN",
            "inference_engine_tensor_parallel_size": role_plan.inference_engine_tensor_parallel_size,
            "inference_engine_pipeline_parallel_size": 1,
            "inference_engine_data_parallel_size": role_plan.inference_engine_data_parallel_size,
            "inference_engine_expert_parallel_size": role_plan.inference_engine_expert_parallel_size,
            "num_inference_engines": role_plan.num_inference_engines,
            "n_samples_per_prompt": role_plan.n_samples_per_prompt,
            "gpu_memory_utilization": 0.75,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 16384,
            "enforce_eager": False,
            "vllm_v1_disable_multiproc": False,
            "run_engines_locally": True,
            "weight_sync_backend": "nccl",
            "async_engine": True,
            "batched": False,
            "engine_init_kwargs": {"async_scheduling": False, "enable_mfu_metrics": True},
            "sampling_params": {"temperature": 1.0, "top_p": 1.0},
        },
        "data": {"kind": "parquet", "train_data": [], "val_data": [], "shuffle": False},
        "extra_env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    }
    if speculative:
        config["generator"]["speculative_decoding"] = {
            "method": "eagle3",
            "model": {},
            "num_speculative_tokens": 3,
            "training": None,
        }
    return yaml.safe_dump(config, sort_keys=False)


def _agentic_benchmark_config(role_plan: SkyRLRolePlan, *, speculative: bool) -> str:
    config = yaml.safe_load(_rl_benchmark_config(role_plan, speculative=speculative))
    config["entrypoint"] = "terminal_bench"
    config["config_groups"] = {"terminal_bench_config": "terminal_bench"}
    config["context_budget"] = {
        "request_window_tokens": 32_768,
        "max_new_tokens_per_turn": 4096,
        "max_turns": 8,
    }
    config["terminal_bench"] = {
        "harbor": {
            "name": "terminus-2",
            "enable_summarize": False,
            "store_all_messages": True,
            "strict_json_parser": True,
            "interleaved_thinking": False,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            "override_timeout_sec": 600,
            "override_cpus": 1,
            "override_memory_mb": 2048,
            "override_storage_mb": 2048,
            "auto_snapshot": True,
            "verifier_override_timeout_sec": 300,
            "max_retries": 2,
            "min_wait_sec": 30.0,
            "max_wait_sec": 300.0,
            "wait_multiplier": 2.0,
            "exclude_exceptions": [
                "VerifierTimeoutError",
                "VerifierRuntimeError",
                "RewardFileNotFoundError",
                "RewardFileEmptyError",
                "VerifierOutputParseError",
            ],
            "n_concurrent_trials": 64,
            "collect_rollout_details": False,
            "enable_error_classification": True,
            "mask_exceptions": [
                "DaytonaError",
                "EnvironmentStartTimeoutError",
                "NetworkError",
                "ConnectionError",
                "RewardFileNotFoundError",
                "RewardFileEmptyError",
                "AgentEnvironmentTimeoutError",
                "ContextLengthExceededError",
            ],
            "default_error_treatment": "zero",
            "passthrough_exceptions": ["AgentTimeoutError"],
        },
        "model_info": None,
        "archiving": {"enabled": False},
        "trace_upload": {"enabled": False},
    }
    config["generator"]["enable_http_endpoint"] = True
    config["data"]["kind"] = "tasks"
    config["trajectory_runner"] = {"process_pool": {"num_coordinators": 8, "cpus_per_coordinator": 4}}
    return yaml.safe_dump(config, sort_keys=False)


def _corpus_evalchemy_config(seed: int) -> EvalchemyConfig:
    sources = [load_evalchemy_config(_EVALCHEMY_CONFIG_DIR / f"{name}.yaml") for name in CORPUS_EVALS]
    return EvalchemyConfig.model_validate(
        {
            "tasks": [task for source in sources for task in source.tasks],
            "task_options": {
                name: options.model_dump(exclude_none=True)
                for source in sources
                for name, options in source.task_options.items()
            },
            "apply_chat_template": True,
            "limit": CORPUS_LIMIT_PER_TASK,
            "batch_size": 1,
            "seed": seed,
            "gen_kwargs": "temperature=1.0,top_p=1.0",
            "runtime_extras": list(dict.fromkeys(extra for source in sources for extra in source.runtime_extras)),
        }
    )


@dataclass(frozen=True)
class ArtifactEvaluationModel:
    """An HF export artifact adapted to the shared evaluation launcher."""

    step: ArtifactStep[LevanterCheckpoint]
    model: ModelConfig

    def deps(self) -> tuple[ArtifactStep, ...]:
        return (self.step,)

    def resolve(self, ctx: StepContext) -> ModelConfig:
        return replace(self.model, location=ctx.artifact_path(self.step))


def _target_evaluation_model() -> ArtifactEvaluationModel:
    return ArtifactEvaluationModel(
        step=TARGET_MODEL,
        model=ModelConfig(
            name=TARGET_MODEL_NAME,
            location="<artifact>",
            tokenizer=TARGET_TOKENIZER,
            apply_chat_template=True,
            resource_hint=ResourceHint(gpu={GPU_VARIANT: GPUS_PER_NODE}, cpu=64, memory="512GB", disk="2TB"),
            serve=ServeConfig(
                tensor_parallel_size=1,
                data_parallel_size=GPUS_PER_NODE,
                max_model_len=65_536,
                max_num_batched_tokens=7168,
                max_num_seqs=32,
                tool_call_parser="hermes",
                auto_overrides=False,
                vllm_extra_args=SNOWBALL_VLLM_ARGS,
            ),
            generation=GenerationConfig(max_gen_toks=8192),
        ),
    )


def _qa_rollout_steps() -> tuple[ArtifactStep[FineStoreEvalchemyResult], ...]:
    steps = []
    for seed in CORPUS_SEEDS:
        config = evalchemy_run_config(f"snowball-eagle-mixed-s{seed}", _corpus_evalchemy_config(seed))
        steps.append(
            evaluate_evalchemy(
                model_name=TARGET_MODEL_NAME,
                model=TARGET_MODEL,
                config=config,
                serve=ServeConfig(
                    tensor_parallel_size=1,
                    data_parallel_size=GPUS_PER_NODE,
                    max_model_len=SEQUENCE_LENGTH,
                    max_num_batched_tokens=16384,
                    max_num_seqs=16,
                    vllm_extra_args=SNOWBALL_VLLM_ARGS,
                ),
                resource_hint=ResourceHint(
                    gpu={GPU_VARIANT: GPUS_PER_NODE},
                    cpu=64,
                    memory="512GB",
                    disk="2TB",
                ),
                accelerator=AcceleratorChoice(
                    platform=Platform.GPU,
                    gpu_type=GPU_VARIANT,
                    gpu_count=GPUS_PER_NODE,
                    target_cluster=CLUSTER,
                ),
                tokenizer=TARGET_TOKENIZER,
                discover_latest_checkpoint=False,
                version=VERSION,
            )
        )
    return tuple(steps)


def _agentic_rollout_steps() -> tuple[ArtifactStep[EvaluationResult], ...]:
    model = _target_evaluation_model()
    return tuple(
        evaluation_step(
            model,
            AGENTIC_CORPUS_EVALS,
            version=f"{VERSION}.{repetition}",
            limit=AGENTIC_CORPUS_LIMIT_PER_TASK,
            accelerator=f"{GPU_VARIANT}x{GPUS_PER_NODE}",
            submission_cluster=CLUSTER,
            federated_cluster=CLUSTER,
        )
        for repetition in AGENTIC_CORPUS_REPETITIONS
    )


def _rollout_steps() -> tuple[ArtifactStep, ...]:
    return (*_qa_rollout_steps(), *_agentic_rollout_steps())


@dataclass(frozen=True)
class DraftSftArtifactNames:
    """Artifact names for each reusable draft-SFT stage."""

    conversations: str
    initial_draft: str
    verifier: str
    captured_data: str
    draft: str


@dataclass(frozen=True)
class DraftSftExecutionConfig:
    """Model-specific hidden-state capture and training settings."""

    processor_model: str
    transformers_model_type: str
    target_layer_ids: tuple[int, ...]
    verifier_num_hidden_layers: int
    sequence_length: int
    gpu_count: int
    max_samples: int | None
    minimum_valid_tokens: int | None


@dataclass(frozen=True)
class DraftSftTrainingConfig:
    """Optimizer and validation settings for one draft fit."""

    epochs: int
    learning_rate: float
    muon_learning_rate: float
    train_data_ratio: float
    save_best: bool


def _conversation_step(*, name: str, rollouts: tuple[ArtifactStep, ...]) -> ArtifactStep[Artifact]:
    def build_config(ctx: StepContext) -> RolloutConversationConfig:
        source_archives: list[str] = []
        for rollout in rollouts:
            if rollout.artifact_type is not EvaluationResult:
                source_archives.append(ctx.artifact_path(rollout))
                continue
            if ctx.is_fingerprint:
                source_archives.append(prefix_join(ctx.artifact_path(rollout), "<evaluation-output>"))
                continue
            evaluation = ctx.resolved(rollout)
            source_archives.extend(
                prefix_join(evaluation.records_prefix, f"{run_id}/results") for run_id in evaluation.run_ids
            )
        return RolloutConversationConfig(
            source_archives=tuple(source_archives),
            output_path=ctx.output_path,
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=Artifact,
        run=remote(
            write_rollout_conversations,
            resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="16g"),
        ),
        build_config=build_config,
        deps=rollouts,
    )


def _initial_draft_step(*, name: str, repo_id: str, revision: str) -> ArtifactStep[EagleDraftArtifact]:
    def build_config(ctx: StepContext) -> HfSnapshotConfig:
        return HfSnapshotConfig(
            repo_id=repo_id,
            revision=revision,
            output_path=ctx.output_path,
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=EagleDraftArtifact,
        run=remote(
            mirror_hf_snapshot,
            resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="64g"),
        ),
        build_config=build_config,
    )


def _verifier_step(
    *,
    name: str,
    target_model: ArtifactStep,
    transformers_model_type: str,
) -> ArtifactStep[Artifact]:
    def build_config(ctx: StepContext) -> VerifierViewConfig:
        return VerifierViewConfig(
            source_model=ctx.artifact_path(target_model),
            transformers_model_type=transformers_model_type,
            output_path=ctx.output_path,
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=Artifact,
        run=remote(
            build_verifier_view,
            resources=ResourceConfig.with_cpu(cpu=8, ram="32g", disk="32g"),
        ),
        build_config=build_config,
        deps=(target_model,),
    )


def _capture_step(
    *,
    name: str,
    dataset: ArtifactStep,
    target_model: ArtifactStep,
    execution: DraftSftExecutionConfig,
) -> ArtifactStep[Artifact]:
    def build_config(ctx: StepContext) -> HiddenStateCaptureConfig:
        return HiddenStateCaptureConfig(
            dataset_path=prefix_join(ctx.artifact_path(dataset), SPECULATORS_DATA_FILENAME),
            target_model=ctx.artifact_path(target_model),
            processor_model=execution.processor_model,
            output_path=ctx.output_path,
            target_layer_ids=execution.target_layer_ids,
            verifier_num_hidden_layers=execution.verifier_num_hidden_layers,
            sequence_length=execution.sequence_length,
            data_parallel_size=execution.gpu_count,
            concurrency=64,
            max_samples=execution.max_samples,
            minimum_valid_tokens=execution.minimum_valid_tokens,
            gpu_memory_utilization=0.9,
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=Artifact,
        run=remote(
            capture_hidden_states,
            resources=ResourceConfig.with_gpu(GPU_VARIANT, count=execution.gpu_count, cpu=96, ram="512g", disk="1t"),
            pip_packages=[SPECULATORS.requirement(), _TORCHAUDIO_CU128_REQUIREMENT],
            max_retries_failure=2,
        ),
        build_config=build_config,
        deps=(dataset, target_model),
    )


def _draft_step(
    *,
    name: str,
    captured_data: ArtifactStep,
    verifier: ArtifactStep,
    initial_draft: ArtifactStep,
    execution: DraftSftExecutionConfig,
    training: DraftSftTrainingConfig,
) -> ArtifactStep[EagleDraftArtifact]:
    def build_config(ctx: StepContext) -> DraftTrainingConfig:
        return DraftTrainingConfig(
            captured_data_path=ctx.artifact_path(captured_data),
            verifier_path=ctx.artifact_path(verifier),
            initial_draft_path=ctx.artifact_path(initial_draft),
            output_path=ctx.output_path,
            target_layer_ids=execution.target_layer_ids,
            sequence_length=execution.sequence_length,
            epochs=training.epochs,
            learning_rate=training.learning_rate,
            muon_learning_rate=training.muon_learning_rate,
            num_processes=execution.gpu_count,
            train_data_ratio=training.train_data_ratio,
            save_best=training.save_best,
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=EagleDraftArtifact,
        run=remote(
            train_draft,
            resources=ResourceConfig.with_gpu(GPU_VARIANT, count=execution.gpu_count, cpu=96, ram="512g", disk="1t"),
            pip_packages=[SPECULATORS.requirement(), _TORCHAUDIO_CU128_REQUIREMENT],
        ),
        build_config=build_config,
        deps=(captured_data, verifier, initial_draft),
    )


@dataclass(frozen=True)
class DraftSftPipeline:
    """Artifact handles produced by a draft-SFT pipeline."""

    conversations: ArtifactStep[Artifact]
    initial_draft: ArtifactStep[EagleDraftArtifact]
    verifier: ArtifactStep[Artifact]
    captured_data: ArtifactStep[Artifact]
    draft: ArtifactStep[EagleDraftArtifact]


def sft_draft_model(
    *,
    names: DraftSftArtifactNames,
    rollouts: tuple[ArtifactStep, ...],
    target_model: ArtifactStep,
    initial_draft_repo: str,
    initial_draft_revision: str,
    execution: DraftSftExecutionConfig,
    training: DraftSftTrainingConfig,
) -> DraftSftPipeline:
    """Build reusable Speculators SFT artifacts for a target model."""
    conversations = _conversation_step(name=names.conversations, rollouts=rollouts)
    initial_draft = _initial_draft_step(
        name=names.initial_draft,
        repo_id=initial_draft_repo,
        revision=initial_draft_revision,
    )
    verifier = _verifier_step(
        name=names.verifier,
        target_model=target_model,
        transformers_model_type=execution.transformers_model_type,
    )
    captured_data = _capture_step(
        name=names.captured_data,
        dataset=conversations,
        target_model=target_model,
        execution=execution,
    )
    draft = _draft_step(
        name=names.draft,
        captured_data=captured_data,
        verifier=verifier,
        initial_draft=initial_draft,
        execution=execution,
        training=training,
    )
    return DraftSftPipeline(
        conversations=conversations,
        initial_draft=initial_draft,
        verifier=verifier,
        captured_data=captured_data,
        draft=draft,
    )


def snowball_eagle_sft() -> DraftSftPipeline:
    """Build the Snowball-specific EAGLE-3 SFT pipeline."""
    rollouts = _rollout_steps()
    return sft_draft_model(
        names=DraftSftArtifactNames(
            conversations="data/snowball-eagle-mixed-conversations",
            initial_draft="models/snowball-eagle3-initial-draft",
            verifier="models/snowball-eagle3-verifier-view",
            captured_data="data/snowball-eagle3-hidden-states",
            draft="models/snowball-eagle3-speculators",
        ),
        rollouts=rollouts,
        target_model=TARGET_MODEL,
        initial_draft_repo=INITIAL_DRAFT_REPO,
        initial_draft_revision=INITIAL_DRAFT_REVISION,
        execution=DraftSftExecutionConfig(
            processor_model=TARGET_TOKENIZER,
            transformers_model_type="llama",
            target_layer_ids=TARGET_LAYER_IDS,
            verifier_num_hidden_layers=VERIFIER_NUM_HIDDEN_LAYERS,
            sequence_length=SEQUENCE_LENGTH,
            gpu_count=_DRAFT_GPU_COUNT,
            max_samples=CORPUS_MAX_SAMPLES,
            minimum_valid_tokens=CORPUS_MINIMUM_VALID_TOKENS,
        ),
        training=DraftSftTrainingConfig(
            epochs=2,
            learning_rate=1e-5,
            muon_learning_rate=0.02,
            train_data_ratio=0.9,
            save_best=True,
        ),
    )


@dataclass(frozen=True)
class SnowballDraftPipeline:
    """Snowball draft SFT stages and matched rollout benchmarks."""

    conversations: ArtifactStep[Artifact]
    initial_draft: ArtifactStep[EagleDraftArtifact]
    verifier: ArtifactStep[Artifact]
    captured_data: ArtifactStep[Artifact]
    draft: ArtifactStep[EagleDraftArtifact]
    benchmarks: dict[str, ArtifactStep[SkyRLModel]]


def build_rl_benchmark(
    *,
    pool: ArtifactStep,
    label: str,
    data_file: str,
    draft: ArtifactStep[EagleDraftArtifact] | None,
) -> ArtifactStep[SkyRLModel]:
    """Run one matched production-shaped rollout benchmark."""
    role_plan = _rl_benchmark_role_plan()
    name = f"{RL_ARTIFACT_NAME}-{label}"
    return _benchmark_step(
        name=name,
        config_yaml=_rl_benchmark_config(role_plan, speculative=draft is not None),
        train_data=(ArtifactDataSource(pool, relative_path=data_file),),
        draft=draft,
        role_plan=role_plan,
        seed=17,
    )


def _benchmark_step(
    *,
    name: str,
    config_yaml: str,
    train_data: tuple[SkyRLDataSource, ...],
    draft: ArtifactStep[EagleDraftArtifact] | None,
    role_plan: SkyRLRolePlan,
    seed: int,
) -> ArtifactStep[SkyRLModel]:
    """Build a benchmark with the shared target, topology, and execution policy."""
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name(name),
            version=resolve_version(name, None),
            config_yaml=config_yaml,
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON),
            model=ArtifactHfModel(
                step=TARGET_MODEL,
                tokenizer_uri=TARGET_TOKENIZER,
                tokenizer_revision=TARGET_TOKENIZER_REVISION,
            ),
            train_data=train_data,
            validation_data=(),
            topology=SkyRLTopology(
                num_nodes=12,
                gpus_per_node=GPUS_PER_NODE,
                gpu_variant=GPU_VARIANT,
                role_plan=role_plan,
            ),
            retention=SkyRLRetentionPolicy(),
            seed=seed,
            draft_model=draft,
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
            cpu=32,
            memory="512GB",
            disk="2TB",
            priority="interactive",
            max_retries=1,
            wandb_entity="marin-community",
        ),
    )


def build_agentic_rl_benchmark(
    *,
    label: str,
    draft: ArtifactStep[EagleDraftArtifact] | None,
) -> ArtifactStep[SkyRLModel]:
    """Run a matched acceptance benchmark on disjoint multi-turn terminal tasks."""
    role_plan = _rl_benchmark_role_plan()
    name = f"{RL_ARTIFACT_NAME}-agentic-{label}"
    return _benchmark_step(
        name=name,
        config_yaml=_agentic_benchmark_config(role_plan, speculative=draft is not None),
        train_data=(
            TaskTroveDataSource(
                TASKTROVE_RELEASE,
                TaskTroveSelection(
                    sources=(AGENTIC_BENCHMARK_SOURCE,),
                    limit=_AGENTIC_BENCHMARK_PROMPTS,
                    seed=71,
                ),
            ),
        ),
        draft=draft,
        role_plan=role_plan,
        seed=71,
    )


def build_pipeline() -> SnowballDraftPipeline:
    """Build the offline capture and draft-training artifact graph."""
    sft = snowball_eagle_sft()
    pool = pool_step(POOL_ARTIFACT_NAME, RL_DATA_VERSION)
    benchmarks = {
        f"{split}-{arm}": build_rl_benchmark(
            pool=pool,
            label=f"{split}-{arm}",
            data_file=data_file,
            draft=draft,
        )
        for split, data_file in (("production", TRAIN_FILENAME), ("heldout", VALIDATION_FILENAME))
        for arm, draft in (("control", None), ("starting", sft.initial_draft), ("trained", sft.draft))
    }
    benchmarks.update(
        {
            f"agentic-{arm}": build_agentic_rl_benchmark(label=arm, draft=draft)
            for arm, draft in (("control", None), ("starting", sft.initial_draft), ("trained", sft.draft))
        }
    )
    return SnowballDraftPipeline(
        conversations=sft.conversations,
        initial_draft=sft.initial_draft,
        verifier=sft.verifier,
        captured_data=sft.captured_data,
        draft=sft.draft,
        benchmarks=benchmarks,
    )


@click.command(help=__doc__)
@click.option(
    "--stage",
    type=click.Choice(("conversations", "initial_draft", "verifier", "captured_data", "draft", "benchmarks")),
    default="draft",
    show_default=True,
)
@build_options
def main(stage: str) -> ArtifactStep | dict[str, ArtifactStep]:
    return getattr(build_pipeline(), stage)


if __name__ == "__main__":
    main()
