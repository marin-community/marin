# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train and benchmark a reusable Snowball EAGLE-3 draft.

The corpus is a bounded sample of 512 fresh math-reasoning responses from the
target checkpoint. Run on the US East 02A controller so the target checkpoint
and generated hidden states stay in the same object-store region::

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
      --enable-extra-resources --target-cluster cw-us-east-02a \
      -- python experiments/post_training/snowball_eagle_speculators.py \
      --version 2026.09.22.1 --stage draft --run
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
from marin.evaluation.model_config import ResourceHint, ServeConfig
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
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_smoke,
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
from experiments.post_training.curriculum_rl.launch import (
    GPUS_PER_NODE,
    POOL_ARTIFACT_NAME,
)
from experiments.post_training.curriculum_rl.pool import VALIDATION_FILENAME, pool_step

CORPUS_EVALS = (
    ("olympiadbench", 30),
    ("gsm8k", 482),
)
CORPUS_SEEDS = (17,)
CORPUS_MAX_GENERATION_TOKENS = 4096
CORPUS_MAX_SAMPLES = 512
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
RL_ARTIFACT_NAME = "benchmarks/snowball-e3"
CLUSTER = "cw-us-east-02a"
GPU_VARIANT = "H100"
_DRAFT_GPU_COUNT = 8
_BENCHMARK_PROMPTS = 64
_BENCHMARK_SAMPLES_PER_PROMPT = 1
_BENCHMARK_MEMORY = "512GB"
_BENCHMARK_DISK = "2TB"
_BENCHMARK_DISTRIBUTED_TIMEOUT = 60
_MAX_NUM_SEQS = 16
_MAX_NUM_BATCHED_TOKENS = 16_384
# The 512-example packed corpus yields one optimizer update per epoch.
_SFT_EPOCHS = 32
_DRAFT_TASK_CPU = 96
_DRAFT_TASK_MEMORY = "512g"
_DRAFT_TASK_DISK = "1t"
# Speculators installs torchaudio through its multimodal dependencies. Pin the
# CUDA 12.8 wheel used by the Iris H100 PyTorch runtime so Transformers imports.
_TORCHAUDIO_CU128_REQUIREMENT = (
    "torchaudio @ https://download.pytorch.org/whl/cu128/"
    "torchaudio-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl"
    "#sha256=78b86a17f164bdaabdcee93fdfde2587fc43b9ebf15cd61dcf730b4f8615176b"
)


def _rl_benchmark_role_plan() -> SkyRLRolePlan:
    return SkyRLRolePlan(
        colocate_all=True,
        policy_num_nodes=1,
        policy_num_gpus_per_node=GPUS_PER_NODE,
        num_inference_engines=1,
        inference_engine_tensor_parallel_size=1,
        inference_engine_data_parallel_size=GPUS_PER_NODE,
        inference_engine_expert_parallel_size=GPUS_PER_NODE,
        train_batch_size=_BENCHMARK_PROMPTS,
        policy_mini_batch_size=_BENCHMARK_PROMPTS,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=_BENCHMARK_SAMPLES_PER_PROMPT,
    )


def _rl_benchmark_config_yaml(role_plan: SkyRLRolePlan) -> str:
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
            "max_num_seqs": _MAX_NUM_SEQS,
            "max_num_batched_tokens": _MAX_NUM_BATCHED_TOKENS,
            "enforce_eager": False,
            "vllm_v1_disable_multiproc": False,
            "run_engines_locally": True,
            "weight_sync_backend": "nccl",
            "async_engine": True,
            "batched": False,
            "engine_init_kwargs": {
                "async_scheduling": False,
                "cpu_distributed_timeout_seconds": _BENCHMARK_DISTRIBUTED_TIMEOUT,
                "enable_mfu_metrics": True,
            },
            "sampling_params": {"temperature": 1.0, "top_p": 1.0},
        },
        "data": {"kind": "parquet", "train_data": [], "val_data": [], "shuffle": False},
        "extra_env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    }
    return yaml.safe_dump(config, sort_keys=False)


def _eagle_benchmark_config_yaml(role_plan: SkyRLRolePlan) -> str:
    config = yaml.safe_load(_rl_benchmark_config_yaml(role_plan))
    config["generator"]["speculative_decoding"] = {
        "method": "eagle3",
        "model": {},
        "num_speculative_tokens": 3,
        "training": None,
    }
    return yaml.safe_dump(config, sort_keys=False)


def _corpus_evalchemy_config(name: str, limit: int, seed: int) -> EvalchemyConfig:
    source = load_evalchemy_config(_EVALCHEMY_CONFIG_DIR / f"{name}.yaml")
    return EvalchemyConfig.model_validate(
        {
            "tasks": source.tasks,
            "task_options": {
                name: options.model_dump(exclude_none=True) for name, options in source.task_options.items()
            },
            "apply_chat_template": True,
            "limit": limit,
            "batch_size": 1,
            "seed": seed,
            "gen_kwargs": "temperature=1.0,top_p=1.0",
            "max_tokens": CORPUS_MAX_GENERATION_TOKENS,
            "runtime_extras": source.runtime_extras,
        }
    )


def _qa_rollout_steps() -> tuple[ArtifactStep[FineStoreEvalchemyResult], ...]:
    steps = []
    for seed in CORPUS_SEEDS:
        for name, limit in CORPUS_EVALS:
            config = evalchemy_run_config(
                f"snowball-eagle-{name}-s{seed}",
                _corpus_evalchemy_config(name, limit, seed),
            )
            step = evaluate_evalchemy(
                model_name=TARGET_MODEL_NAME,
                model=TARGET_MODEL,
                config=config,
                serve=ServeConfig(
                    tensor_parallel_size=1,
                    data_parallel_size=GPUS_PER_NODE,
                    max_model_len=SEQUENCE_LENGTH,
                    max_num_batched_tokens=_MAX_NUM_BATCHED_TOKENS,
                    max_num_seqs=_MAX_NUM_SEQS,
                    vllm_extra_args=SNOWBALL_VLLM_ARGS,
                ),
                resource_hint=ResourceHint(
                    gpu={GPU_VARIANT: GPUS_PER_NODE},
                    cpu=64,
                    memory=_BENCHMARK_MEMORY,
                    disk=_BENCHMARK_DISK,
                ),
                accelerator=AcceleratorChoice(
                    platform=Platform.GPU,
                    gpu_type=GPU_VARIANT,
                    gpu_count=GPUS_PER_NODE,
                ),
                tokenizer=TARGET_TOKENIZER,
                discover_latest_checkpoint=False,
                version=None,
            )
            steps.append(replace(step, run=replace(step.run, max_retries_failure=0)))
    return tuple(steps)


def _conversation_step(*, name: str, rollouts: tuple[ArtifactStep, ...]) -> ArtifactStep[Artifact]:
    def build_config(ctx: StepContext) -> RolloutConversationConfig:
        return RolloutConversationConfig(
            source_archives=tuple(ctx.artifact_path(rollout) for rollout in rollouts),
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
) -> ArtifactStep[Artifact]:
    def build_config(ctx: StepContext) -> HiddenStateCaptureConfig:
        return HiddenStateCaptureConfig(
            dataset_path=prefix_join(ctx.artifact_path(dataset), SPECULATORS_DATA_FILENAME),
            target_model=ctx.artifact_path(target_model),
            processor_model=TARGET_TOKENIZER,
            output_path=ctx.output_path,
            target_layer_ids=TARGET_LAYER_IDS,
            verifier_num_hidden_layers=VERIFIER_NUM_HIDDEN_LAYERS,
            sequence_length=SEQUENCE_LENGTH,
            data_parallel_size=_DRAFT_GPU_COUNT,
            concurrency=64,
            max_samples=CORPUS_MAX_SAMPLES,
            minimum_valid_tokens=CORPUS_MINIMUM_VALID_TOKENS,
            gpu_memory_utilization=0.9,
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=Artifact,
        run=remote(
            capture_hidden_states,
            resources=ResourceConfig.with_gpu(
                GPU_VARIANT,
                count=_DRAFT_GPU_COUNT,
                cpu=_DRAFT_TASK_CPU,
                ram=_DRAFT_TASK_MEMORY,
                disk=_DRAFT_TASK_DISK,
            ),
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
) -> ArtifactStep[EagleDraftArtifact]:
    def build_config(ctx: StepContext) -> DraftTrainingConfig:
        return DraftTrainingConfig(
            captured_data_path=ctx.artifact_path(captured_data),
            verifier_path=ctx.artifact_path(verifier),
            initial_draft_path=ctx.artifact_path(initial_draft),
            output_path=ctx.output_path,
            target_layer_ids=TARGET_LAYER_IDS,
            sequence_length=SEQUENCE_LENGTH,
            epochs=_SFT_EPOCHS,
            learning_rate=1e-5,
            muon_learning_rate=0.02,
            num_processes=_DRAFT_GPU_COUNT,
            train_data_ratio=0.9,
            save_best=False,
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=EagleDraftArtifact,
        run=remote(
            train_draft,
            resources=ResourceConfig.with_gpu(
                GPU_VARIANT,
                count=_DRAFT_GPU_COUNT,
                cpu=_DRAFT_TASK_CPU,
                ram=_DRAFT_TASK_MEMORY,
                disk=_DRAFT_TASK_DISK,
            ),
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


def snowball_eagle_sft() -> DraftSftPipeline:
    conversations = _conversation_step(
        name="data/snowball-eagle-mixed-conversations",
        rollouts=_qa_rollout_steps(),
    )
    initial_draft = _initial_draft_step(
        name="models/snowball-eagle3-initial-draft",
        repo_id=INITIAL_DRAFT_REPO,
        revision=INITIAL_DRAFT_REVISION,
    )
    verifier = _verifier_step(
        name="models/snowball-eagle3-verifier-view",
        target_model=TARGET_MODEL,
        transformers_model_type="llama",
    )
    captured_data = _capture_step(
        name="data/snowball-eagle3-hidden-states",
        dataset=conversations,
        target_model=TARGET_MODEL,
    )
    draft = _draft_step(
        name="models/snowball-eagle3-speculators",
        captured_data=captured_data,
        verifier=verifier,
        initial_draft=initial_draft,
    )
    return DraftSftPipeline(
        conversations=conversations,
        initial_draft=initial_draft,
        verifier=verifier,
        captured_data=captured_data,
        draft=draft,
    )


@dataclass(frozen=True)
class SnowballDraftPipeline:
    """Snowball draft SFT stages and matched rollout benchmarks."""

    sft: DraftSftPipeline
    benchmarks: dict[str, ArtifactStep[SkyRLRun]]


def build_rl_benchmark(
    *,
    pool: ArtifactStep,
    label: str,
    data_file: str,
    draft: ArtifactStep[EagleDraftArtifact] | None,
) -> ArtifactStep[SkyRLRun]:
    """Build one matched production-shaped rollout benchmark."""
    role_plan = _rl_benchmark_role_plan()
    name = f"{RL_ARTIFACT_NAME}-{label}"
    return _benchmark_step(
        name=name,
        config_yaml=(_rl_benchmark_config_yaml(role_plan) if draft is None else _eagle_benchmark_config_yaml(role_plan)),
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
) -> ArtifactStep[SkyRLRun]:
    """Build a benchmark with the shared target, topology, and execution policy."""
    return skyrl_smoke(
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
                num_nodes=role_plan.policy_num_nodes,
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
            memory=_BENCHMARK_MEMORY,
            disk=_BENCHMARK_DISK,
            priority="interactive",
            max_retries=1,
            wandb_entity="marin-community",
        ),
    )


def build_pipeline() -> SnowballDraftPipeline:
    """Build the bounded draft SFT pilot and matched held-out benchmarks."""
    sft = snowball_eagle_sft()
    pool = pool_step(POOL_ARTIFACT_NAME, RL_DATA_VERSION)
    arms = (("control", None), ("starting", sft.initial_draft), ("trained", sft.draft))
    benchmarks = {
        f"heldout-{arm}": build_rl_benchmark(
            pool=pool,
            label=f"heldout-{arm}",
            data_file=VALIDATION_FILENAME,
            draft=draft,
        )
        for arm, draft in arms
    }
    return SnowballDraftPipeline(
        sft=sft,
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
    pipeline = build_pipeline()
    if stage == "benchmarks":
        return pipeline.benchmarks
    return getattr(pipeline.sft, stage)


if __name__ == "__main__":
    main()
