# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch curriculum-RL arms: SkyRL GRPO on Qwen3-0.6B over a graded math pool.

Each arm trains the same policy on the same pool with the same step budget and
differs only in how prompts are sampled per step. The naive arm is the pinned
trainer's uniform shuffle; the other arms select a ``data.sampling`` policy from
the pinned MarinSkyRL curriculum sampler.

Plan or run arms::

    python -m experiments.post_training.curriculum_rl.launch --version 2026.08.29 --scale smoke
    python -m experiments.post_training.curriculum_rl.launch --version 2026.08.29 --scale smoke --run
    python -m experiments.post_training.curriculum_rl.launch --version 2026.08.29 \
        --scale full --arm naive --arm thompson --run

Tracking issue: https://github.com/marin-community/marin/issues/8765
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import click
from fray.types import ResourceConfig
from huggingface_hub import snapshot_download
from marin.evaluation.model_config import GenerationConfig, ModelConfig, ResourceHint, ServeConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLHardware,
    SkyRLRetentionPolicy,
    SkyRLRun,
    SkyRLSpec,
    skyrl_step,
)
from marin.skyrl_recipe import (
    Algorithm,
    ContextBudget,
    Data,
    DynamicSampling,
    Environment,
    Generator,
    Placement,
    Policy,
    PolicyMegatronConfig,
    PolicyOptimizerConfig,
    RecipePatch,
    Ref,
    RefMegatronConfig,
    RLEntrypoint,
    Sampling,
    SamplingParams,
    SkyRLRecipe,
    Trainer,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.provenance import username_segment

from experiments.evaluation.models import SNOWBALL_SFT_EXPORT_URI, SNOWBALL_VLLM_ARGS
from experiments.evaluation.pipeline import EvaluationResult
from experiments.post_training.curriculum_rl.pool import (
    MAX_PROMPT_TOKENS,
    QWEN3_MODEL,
    QWEN3_REVISION,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    pool_step,
)
from experiments.post_training.skyrl_evaluation import SKYRL_POLICY_LOCATION, skyrl_eval_step

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "curriculum-rl"
HF_EXPORT_SUBDIR = "hf"
POOL_ARTIFACT_NAME = f"documents/{EXPERIMENT_NAME}-pool"
MODEL_ARTIFACT_NAME = f"models/{EXPERIMENT_NAME}-qwen3-0.6b"
GPU_VARIANT = "H100"
GPUS_PER_NODE = 8
WANDB_PROJECT = f"marin-{EXPERIMENT_NAME}"
SEED = 17
MARIN_TOKENIZER = "marin-community/marin-tokenizer"
MARIN_TOKENIZER_REVISION = "a5ca45f"


@dataclass(frozen=True)
class PolicySpec:
    """Which policy model arms train, where it runs, and its model-specific profile."""

    label: str
    cluster: str
    tokenizer_uri: str
    tokenizer_revision: str
    model_relative_path: str
    # Host memory for every training and engine task. The Snowball export
    # streams ~134GB of bf16 shards through host buffers on load (per node,
    # policy and engine alike); 128GB of host RAM OOM-killed its first smoke.
    task_memory: str
    # Evaluation serving profile: GPUs and host memory per serving instance,
    # engine data parallelism, and model-specific vLLM flags and sampling
    # kwargs.
    serve_gpus: int
    recipe: RecipePatch = field(
        default_factory=lambda: RecipePatch(
            trainer=Trainer(
                policy=Policy(
                    megatron_config=PolicyMegatronConfig(
                        tensor_model_parallel_size=1,
                        pipeline_model_parallel_size=1,
                        context_parallel_size=1,
                        expert_model_parallel_size=1,
                        expert_tensor_parallel_size=1,
                    )
                ),
                ref=Ref(
                    megatron_config=RefMegatronConfig(
                        tensor_model_parallel_size=1,
                        pipeline_model_parallel_size=1,
                        context_parallel_size=1,
                        expert_model_parallel_size=1,
                        expert_tensor_parallel_size=1,
                    )
                ),
            ),
            generator=Generator(chat_template_kwargs={"enable_thinking": False}),
        )
    )
    serve_memory: str | None = None
    serve_data_parallel_size: int | None = None
    serve_vllm_extra_args: tuple[str, ...] = ()
    serve_gen_kwargs: tuple[tuple[str, str], ...] = ()
    # Pre-existing model artifact; None mirrors the HF model via model_step.
    adopted_model: ArtifactStep[LevanterCheckpoint] | None = None


QWEN_POLICY = PolicySpec(
    label="qwen",
    cluster="cw-rno2a",
    tokenizer_uri=QWEN3_MODEL,
    tokenizer_revision=QWEN3_REVISION,
    model_relative_path=HF_EXPORT_SUBDIR,
    # Thinking mode ate the whole generation budget at 0.6B (85% truncation in
    # the round-1 smoke); Qwen arms train and roll out in non-thinking mode.
    task_memory="128GB",
    serve_gpus=1,
)

# Adopted in place from the exports prefix (~134GB referenced, not copied).
SNOWBALL_MODEL = ArtifactStep.adopt(
    "models/snowball-67b-a2b-sft-s2-thinking",
    "2026.08.30",
    SNOWBALL_SFT_EXPORT_URI,
    kind=LevanterCheckpoint,
)

# The Snowball SFT trains where its export lives (us-east-02a) so 134GB of
# shards never stream cross-region. Rollout and serving engines mirror the
# export's profile: one node-sized vLLM instance, tensor_parallel_size=1,
# experts sharded across the node's eight ranks (ep = dp * tp). The frozen
# query-bias router is the pinned MarinSkyRL default.
SNOWBALL_POLICY = PolicySpec(
    label="snowball",
    cluster="cw-us-east-02a",
    tokenizer_uri=MARIN_TOKENIZER,
    tokenizer_revision=MARIN_TOKENIZER_REVISION,
    model_relative_path="",
    task_memory="512GB",
    serve_gpus=GPUS_PER_NODE,
    serve_memory="512g",
    serve_data_parallel_size=GPUS_PER_NODE,
    serve_vllm_extra_args=SNOWBALL_VLLM_ARGS,
    # Emit the canonical think tokens verbatim (the export otherwise scores 0)
    # and curb thinking loops with a light repetition penalty.
    serve_gen_kwargs=(("skip_special_tokens", "false"), ("repetition_penalty", "1.1")),
    adopted_model=SNOWBALL_MODEL,
    recipe=RecipePatch(
        trainer=Trainer(
            flash_attn=False,
            gradient_checkpointing=True,
            offload_optimizer_during_rollouts=True,
            policy=Policy(
                megatron_config=PolicyMegatronConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=2,
                    context_parallel_size=1,
                    expert_model_parallel_size=GPUS_PER_NODE,
                    expert_tensor_parallel_size=1,
                    optimizer_checkpoint_sharding_type="dp_reshardable",
                )
            ),
            ref=Ref(
                megatron_config=RefMegatronConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=2,
                    context_parallel_size=1,
                    expert_model_parallel_size=GPUS_PER_NODE,
                    expert_tensor_parallel_size=1,
                )
            ),
        )
    ),
)
POLICIES = {policy.label: policy for policy in (QWEN_POLICY, SNOWBALL_POLICY)}


class SamplerKind(StrEnum):
    """How each step's rollout prompts are drawn from the graded pool.

    A ``grade-uniform`` kind (equal budget per grade) is intentionally absent:
    it trailed every other curriculum arm when measured.
    """

    NAIVE = "naive"
    THOMPSON = "thompson"
    LEARNABILITY = "learnability"
    GRADE_ADAPTIVE = "grade-adaptive"
    GRADE_PRIOR = "grade-prior"


@dataclass(frozen=True)
class ArmSpec:
    """One experiment arm: a sampling policy, optionally with DAPO filtering."""

    name: str
    sampler: SamplerKind
    dapo: bool = False


ARMS = {
    spec.name: spec
    for spec in (
        ArmSpec("naive", SamplerKind.NAIVE),
        ArmSpec("thompson", SamplerKind.THOMPSON),
        ArmSpec("learnability", SamplerKind.LEARNABILITY),
        ArmSpec("grade-adaptive", SamplerKind.GRADE_ADAPTIVE),
        ArmSpec("grade-prior", SamplerKind.GRADE_PRIOR),
        ArmSpec("naive-dapo", SamplerKind.NAIVE, dapo=True),
        ArmSpec("thompson-dapo", SamplerKind.THOMPSON, dapo=True),
        ArmSpec("learnability-dapo", SamplerKind.LEARNABILITY, dapo=True),
        ArmSpec("grade-prior-dapo", SamplerKind.GRADE_PRIOR, dapo=True),
    )
}

# Round-3 directional arms weight bins by the probability a GRPO group
# survives dynamic-sampling filtering (1 - p^n - (1-p)^n) rather than by
# per-sample reward variance p(1-p): the filter's actual rollout-cost model,
# near-flat across mid difficulties.
GROUP_INFORMATIVE_SAMPLERS = frozenset({SamplerKind.LEARNABILITY, SamplerKind.GRADE_PRIOR})


@dataclass(frozen=True)
class ScalePreset:
    """An authored recipe and its terminal evaluation suite."""

    label: str
    recipe: SkyRLRecipe
    evals: str
    tuning: RecipePatch | None = None


SMOKE = ScalePreset(
    label="smoke",
    evals="gsm8k-smoke",
    recipe=SkyRLRecipe.combine(
        geometry=RecipePatch(
            trainer=Trainer(
                placement=Placement(
                    colocate_all=False,
                    colocate_policy_ref=True,
                    policy_num_nodes=1,
                    policy_num_gpus_per_node=GPUS_PER_NODE,
                    ref_num_nodes=1,
                    ref_num_gpus_per_node=GPUS_PER_NODE,
                ),
                train_batch_size=64,
                policy_mini_batch_size=32,
                micro_train_batch_size_per_gpu=4,
            ),
            generator=Generator(
                num_inference_engines=GPUS_PER_NODE,
                inference_engine_tensor_parallel_size=1,
                inference_engine_pipeline_parallel_size=1,
                inference_engine_data_parallel_size=1,
                inference_engine_expert_parallel_size=1,
                n_samples_per_prompt=4,
            ),
        ),
        loop=RecipePatch(
            context_budget=ContextBudget(request_window_tokens=2048, max_new_tokens_per_turn=1024, max_turns=1),
            trainer=Trainer(max_steps=4, eval_interval=-1, ckpt_interval=2),
        ),
    ),
)

# 64 GPUs: generation dominates for a 0.6B policy, so 2 training nodes and
# 6 single-GPU vLLM engine nodes. ~120 steps * 512 prompts * 8 samples at
# <=1024 non-thinking tokens bounds the run at roughly 0.2-0.5B output tokens.
FULL = ScalePreset(
    label="full",
    evals="math500,gsm8k-0shot",
    recipe=SkyRLRecipe.combine(
        geometry=RecipePatch(
            trainer=Trainer(
                placement=Placement(
                    colocate_all=False,
                    colocate_policy_ref=True,
                    policy_num_nodes=2,
                    policy_num_gpus_per_node=GPUS_PER_NODE,
                    ref_num_nodes=2,
                    ref_num_gpus_per_node=GPUS_PER_NODE,
                ),
                train_batch_size=512,
                policy_mini_batch_size=64,
                micro_train_batch_size_per_gpu=8,
            ),
            generator=Generator(
                num_inference_engines=6 * GPUS_PER_NODE,
                inference_engine_tensor_parallel_size=1,
                inference_engine_pipeline_parallel_size=1,
                inference_engine_data_parallel_size=1,
                inference_engine_expert_parallel_size=1,
                n_samples_per_prompt=8,
            ),
        ),
        loop=RecipePatch(
            context_budget=ContextBudget(request_window_tokens=2048, max_new_tokens_per_turn=1024, max_turns=1),
            trainer=Trainer(max_steps=120, eval_interval=10, ckpt_interval=10),
        ),
    ),
)

SNOWBALL_RESOURCES = RecipePatch(
    trainer=Trainer(
        placement=Placement(
            colocate_all=False,
            colocate_policy_ref=False,
            policy_num_nodes=4,
            policy_num_gpus_per_node=GPUS_PER_NODE,
            ref_num_nodes=2,
            ref_num_gpus_per_node=GPUS_PER_NODE,
        )
    ),
    generator=Generator(
        inference_engine_tensor_parallel_size=1,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=GPUS_PER_NODE,
        inference_engine_expert_parallel_size=GPUS_PER_NODE,
    ),
)


# The 67B-A2B smoke: four policy nodes, two reference nodes, and one
# node-sized expert-parallel engine.
# NCCL rank-drop failures on this stack appeared only at 32k contexts; this
# preset stays at the 2k window.
SNOWBALL_SMOKE = ScalePreset(
    label="snowball-smoke",
    evals="gsm8k-smoke",
    recipe=SkyRLRecipe.combine(
        resources=SNOWBALL_RESOURCES,
        geometry=RecipePatch(
            trainer=Trainer(
                train_batch_size=32,
                policy_mini_batch_size=32,
                micro_train_batch_size_per_gpu=4,
            ),
            generator=Generator(
                num_inference_engines=1,
                n_samples_per_prompt=4,
            ),
        ),
        loop=RecipePatch(
            context_budget=ContextBudget(request_window_tokens=2048, max_new_tokens_per_turn=1024, max_turns=1),
            trainer=Trainer(max_steps=4, eval_interval=-1, ckpt_interval=4),
        ),
    ),
)

# The 67B-A2B measurement point: four policy nodes, two reference nodes,
# and four expert-parallel engine nodes. The smoke averaged 884 generated tokens
# against a 1024 cap,
# so the full runs widen the window to 3072 with a 2048 response budget
# (the 1024-token prompt budget still admits every pool row). 60 steps at
# 128x8 responses bounds an arm near the round-2 per-arm token budget.
SNOWBALL_FULL = ScalePreset(
    label="snowball-full",
    evals="math500,gsm8k-0shot",
    recipe=SkyRLRecipe.combine(
        resources=SNOWBALL_RESOURCES,
        geometry=RecipePatch(
            trainer=Trainer(
                train_batch_size=128,
                policy_mini_batch_size=64,
                micro_train_batch_size_per_gpu=4,
            ),
            generator=Generator(
                num_inference_engines=4,
                n_samples_per_prompt=8,
            ),
        ),
        loop=RecipePatch(
            context_budget=ContextBudget(request_window_tokens=3072, max_new_tokens_per_turn=2048, max_turns=1),
            trainer=Trainer(max_steps=60, eval_interval=10, ckpt_interval=10),
        ),
    ),
)

# Shared Snowball RL recipe. Optimizer: MuonH (grug_moe_muonh_v1, the recipe
# the base model pretrained and SFT'd under: FP32 master weights,
# orthogonalized matrix updates, zero weight decay) at 1e-5, 1/5 of the SFT
# peak; AdamW at GRPO-typical rates barely moves validation on this model and
# destabilized one run despite max_grad_norm=1.0. reversion_mass keeps starved
# bins re-probeable (inert for the naive arm).
SNOWBALL_MUONH_TUNING = RecipePatch(
    trainer=Trainer(
        policy=Policy(
            optimizer_config=PolicyOptimizerConfig(optimizer="MuonH", lr=1.0e-5, weight_decay=0.0),
            megatron_config=PolicyMegatronConfig(
                pipeline_model_parallel_size=4,
                transformer_config_kwargs={
                    "num_layers_in_first_pipeline_stage": 6,
                    "num_layers_in_last_pipeline_stage": 6,
                },
            ),
        )
    ),
    data=Data(sampling=Sampling(reversion_mass=2.0)),
)

# The round-4 recipe uses four policy nodes and two reference nodes. MuonH
# keeps FP32 master weights and momentum, so each policy rank trains one
# sequence per microbatch.
SNOWBALL_SMOKE_R4 = ScalePreset(
    label="snowball-smoke-r4",
    evals="gsm8k-smoke",
    recipe=SkyRLRecipe.combine(
        resources=SNOWBALL_RESOURCES,
        geometry=RecipePatch(
            trainer=Trainer(
                train_batch_size=64,
                policy_mini_batch_size=64,
                micro_train_batch_size_per_gpu=1,
            ),
            generator=Generator(
                num_inference_engines=1,
                n_samples_per_prompt=8,
            ),
        ),
        loop=RecipePatch(
            context_budget=ContextBudget(request_window_tokens=3072, max_new_tokens_per_turn=2048, max_turns=1),
            trainer=Trainer(max_steps=4, eval_interval=-1, ckpt_interval=4),
        ),
    ),
    tuning=SNOWBALL_MUONH_TUNING,
)

SNOWBALL_FULL_R4 = ScalePreset(
    label="snowball-full-r4",
    evals="math500,gsm8k-0shot",
    recipe=SkyRLRecipe.combine(
        resources=SNOWBALL_RESOURCES,
        geometry=RecipePatch(
            trainer=Trainer(
                train_batch_size=64,
                policy_mini_batch_size=64,
                micro_train_batch_size_per_gpu=1,
            ),
            generator=Generator(
                num_inference_engines=4,
                n_samples_per_prompt=8,
            ),
        ),
        loop=RecipePatch(
            context_budget=ContextBudget(request_window_tokens=3072, max_new_tokens_per_turn=2048, max_turns=1),
            trainer=Trainer(max_steps=120, eval_interval=10, ckpt_interval=10),
        ),
    ),
    tuning=SNOWBALL_MUONH_TUNING,
)

# The long-budget Snowball presets: an 8192-token response budget over the
# same 1024-token prompt budget, sized so frontier-grade reasoning can finish
# instead of truncating (at 2048, 40-48% of rollouts hit the cap and the
# hardest bins truncated near-totally). Activation memory tracks tokens per
# micro batch; these presets use one sequence per policy rank per microbatch.
SNOWBALL_SMOKE_R5 = ScalePreset(
    label="snowball-smoke-r5",
    evals="gsm8k-smoke",
    recipe=SkyRLRecipe.combine(
        resources=SNOWBALL_RESOURCES,
        geometry=RecipePatch(
            trainer=Trainer(
                train_batch_size=64,
                policy_mini_batch_size=64,
                micro_train_batch_size_per_gpu=1,
            ),
            generator=Generator(
                num_inference_engines=1,
                n_samples_per_prompt=8,
            ),
        ),
        loop=RecipePatch(
            context_budget=ContextBudget(request_window_tokens=9216, max_new_tokens_per_turn=8192, max_turns=1),
            trainer=Trainer(max_steps=4, eval_interval=-1, ckpt_interval=4),
        ),
    ),
    tuning=SNOWBALL_MUONH_TUNING,
)

SNOWBALL_FULL_R5 = ScalePreset(
    label="snowball-full-r5",
    evals="math500,gsm8k-0shot",
    recipe=SkyRLRecipe.combine(
        resources=SNOWBALL_RESOURCES,
        geometry=RecipePatch(
            trainer=Trainer(
                train_batch_size=64,
                policy_mini_batch_size=64,
                micro_train_batch_size_per_gpu=1,
            ),
            generator=Generator(
                num_inference_engines=4,
                n_samples_per_prompt=8,
            ),
        ),
        loop=RecipePatch(
            context_budget=ContextBudget(request_window_tokens=9216, max_new_tokens_per_turn=8192, max_turns=1),
            trainer=Trainer(max_steps=120, eval_interval=10, ckpt_interval=10),
        ),
    ),
    tuning=SNOWBALL_MUONH_TUNING,
)

SCALES = {
    preset.label: preset
    for preset in (
        SMOKE,
        FULL,
        SNOWBALL_SMOKE,
        SNOWBALL_FULL,
        SNOWBALL_SMOKE_R4,
        SNOWBALL_FULL_R4,
        SNOWBALL_SMOKE_R5,
        SNOWBALL_FULL_R5,
    )
}

# The pool filter must keep every retained prompt under each preset's
# max_input_length (request window minus generation budget), or retained rows
# skip generation and fully skipped GRPO groups fail admission.
for _preset in SCALES.values():
    assert (
        MAX_PROMPT_TOKENS
        <= _preset.recipe.context_budget.request_window_tokens - _preset.recipe.context_budget.max_new_tokens_per_turn
    ), _preset.label


@dataclass(frozen=True)
class HfSnapshotConfig:
    output_path: str
    repo_id: str = QWEN3_MODEL
    revision: str = QWEN3_REVISION


def mirror_hf_model(config: HfSnapshotConfig) -> None:
    """Stage one pinned HF model snapshot under the artifact's ``hf/`` subdir."""
    with tempfile.TemporaryDirectory() as workdir:
        local = Path(
            snapshot_download(
                repo_id=config.repo_id,
                revision=config.revision,
                local_dir=str(Path(workdir) / "snapshot"),
                cache_dir=str(Path(workdir) / "cache"),
            )
        )
        for surplus in (local / ".cache", local / ".gitattributes"):
            if surplus.is_dir():
                shutil.rmtree(surplus)
            elif surplus.exists():
                surplus.unlink()
        destination = prefix_join(config.output_path, HF_EXPORT_SUBDIR)
        StoragePath(destination).upload_from(f"{local}/", recursive=True)
        logger.info("Mirrored %s@%s to %s", config.repo_id, config.revision, destination)


def model_step(version: str) -> ArtifactStep[LevanterCheckpoint]:
    # Typed as LevanterCheckpoint because ArtifactHfModel consumes HF checkpoint
    # artifacts through that handle; the mirrored snapshot has the same layout.
    return ArtifactStep(
        name=MODEL_ARTIFACT_NAME,
        version=version,
        artifact_type=LevanterCheckpoint,
        run=remote(mirror_hf_model, resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="32g")),
        build_config=lambda ctx: HfSnapshotConfig(output_path=ctx.output_path),
    )


def rl_recipe(preset: ScalePreset, arm: ArmSpec, policy: PolicySpec) -> SkyRLRecipe:
    sampling = Sampling(
        **({"kind": arm.sampler.value} if arm.sampler is not SamplerKind.NAIVE else {}),
        **({"weighting": "group-informative"} if arm.sampler in GROUP_INFORMATIVE_SAMPLERS else {}),
    )
    arm_part = RecipePatch(
        data=Data(sampling=sampling),
        trainer=Trainer(
            algorithm=Algorithm(**({"dynamic_sampling": DynamicSampling(type="filter")} if arm.dapo else {}))
        ),
    )
    recipe = SkyRLRecipe.combine(
        base=RecipePatch(
            entrypoint=RLEntrypoint.STANDARD,
            environment=Environment(env_class="gsm8k"),
            trainer=Trainer(
                strategy="megatron",
                flash_attn=True,
                use_sample_packing=False,
                algorithm=Algorithm(advantage_estimator="grpo", use_kl_loss=True),
                epochs=50,
                update_epochs_per_batch=1,
                eval_batch_size=256,
                resume_mode="latest",
                logger="wandb",
                project_name=WANDB_PROJECT,
                policy=Policy(optimizer_config=PolicyOptimizerConfig(lr=2e-06, max_grad_norm=1.0)),
                hf_hub_repo_id=None,
            ),
            generator=Generator(
                backend="vllm",
                model_dtype="bfloat16",
                vllm_attention_backend="FLASH_ATTN",
                gpu_memory_utilization=0.75,
                enforce_eager=False,
                run_engines_locally=True,
                weight_sync_backend="nccl",
                sampling_params=SamplingParams(temperature=1.0, top_p=1.0),
            ),
            data=Data(kind="parquet", train_data=(), val_data=()),
        ),
        preset=preset.recipe,
        policy=policy.recipe,
        arm=arm_part,
        evaluation=RecipePatch(trainer=Trainer(eval_before_train=preset.recipe.trainer.eval_interval > 0)),
    )
    return recipe.merge(preset.tuning) if preset.tuning is not None else recipe


@dataclass(frozen=True)
class CurriculumArm:
    spec: ArmSpec
    rl: ArtifactStep[SkyRLRun]
    evaluation: ArtifactStep[EvaluationResult]


def evaluation_model_config(policy: PolicySpec, preset: ScalePreset, name: str) -> ModelConfig:
    """Model configuration for the trained policy checkpoint, from the policy's
    ``serve_*`` fields and the preset's context window."""
    return ModelConfig(
        name=name,
        location=SKYRL_POLICY_LOCATION,
        tokenizer=policy.tokenizer_uri,
        apply_chat_template=True,
        resource_hint=ResourceHint(gpu={GPU_VARIANT: policy.serve_gpus}, memory=policy.serve_memory),
        serve=ServeConfig(
            tensor_parallel_size=1,
            data_parallel_size=policy.serve_data_parallel_size,
            max_model_len=preset.recipe.context_budget.request_window_tokens
            + preset.recipe.context_budget.max_new_tokens_per_turn,
            max_num_seqs=64,
            vllm_extra_args=policy.serve_vllm_extra_args,
        ),
        generation=GenerationConfig(
            max_gen_toks=preset.recipe.context_budget.max_new_tokens_per_turn,
            extra_gen_kwargs=dict(policy.serve_gen_kwargs),
        ),
    )


def build_arm(
    *,
    spec: ArmSpec,
    preset: ScalePreset,
    policy: PolicySpec,
    version: str | None,
    model: ArtifactStep[LevanterCheckpoint],
    pool: ArtifactStep,
) -> CurriculumArm:
    suffix = "" if preset is FULL else f"-{preset.label}"
    # Qwen arm names predate the policy axis and stay unprefixed so existing
    # artifact addresses remain valid.
    policy_prefix = "" if policy is QWEN_POLICY else f"{policy.label}-"
    cluster_config = f"lib/iris/config/{policy.cluster}.yaml"
    rl_base_name = f"checkpoints/{EXPERIMENT_NAME}/{policy_prefix}{spec.name}{suffix}"
    rl = skyrl_step(
        SkyRLSpec(
            name=user_owned_name(rl_base_name),
            version=version or resolve_version(rl_base_name, None),
            recipe=rl_recipe(preset, spec, policy),
            model=ArtifactHfModel(
                step=model,
                tokenizer_uri=policy.tokenizer_uri,
                tokenizer_revision=policy.tokenizer_revision,
                relative_path=policy.model_relative_path,
            ),
            train_data=(ArtifactDataSource(pool, relative_path=TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(pool, relative_path=VALIDATION_FILENAME),),
            hardware=SkyRLHardware(gpus_per_node=GPUS_PER_NODE, gpu_variant=GPU_VARIANT),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
            seed=SEED,
        ),
        IrisSkyRLExecution(
            cluster=policy.cluster,
            cluster_config=cluster_config,
            cpu=16,
            memory=policy.task_memory,
            disk="2TB",
            priority="interactive",
            # Fail fast: a broken config surfaces on the first attempt, and a
            # healthy run resumes from its latest checkpoint on resubmission.
            max_retries=1,
            target_cluster=policy.cluster,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=72,
            wandb_entity="marin-community",
        ),
        export_hf=True,
    )
    # The eval artifact is keyed on the model name; include the owner so two
    # users at the same fixed version evaluate their own checkpoints rather
    # than sharing one cached result (the RL step is already user-owned).
    evaluation_model_name = f"{username_segment()}-{EXPERIMENT_NAME}-{policy_prefix}{spec.name}{suffix}"
    evaluation_base_name = f"evals/{evaluation_model_name}/{preset.evals}"
    evaluation_version = version or resolve_version(evaluation_base_name, None)
    evaluation_model = evaluation_model_config(policy, preset, evaluation_model_name)
    evaluation = skyrl_eval_step(
        rl,
        evaluation_model,
        preset.evals,
        version=evaluation_version,
        accelerator=f"{GPU_VARIANT}x{policy.serve_gpus}",
        submission_cluster=policy.cluster,
        federated_cluster=policy.cluster,
    )
    return CurriculumArm(spec=spec, rl=rl, evaluation=evaluation)


def build_arms(
    *,
    specs: tuple[ArmSpec, ...],
    scale: str,
    policy: PolicySpec = QWEN_POLICY,
    version: str | None = None,
) -> dict[str, CurriculumArm]:
    if (policy is SNOWBALL_POLICY) != scale.startswith("snowball-"):
        raise ValueError(f"Scale {scale!r} is incompatible with policy {policy.label!r}")
    preset = SCALES[scale]
    pool = pool_step(POOL_ARTIFACT_NAME, version or resolve_version(POOL_ARTIFACT_NAME, None))
    if policy.adopted_model is not None:
        model = policy.adopted_model
    else:
        model = model_step(version or resolve_version(MODEL_ARTIFACT_NAME, None))
    return {
        spec.name: build_arm(spec=spec, preset=preset, policy=policy, version=version, model=model, pool=pool)
        for spec in specs
    }


@click.command(help=__doc__)
@click.option(
    "--arm",
    "arms",
    multiple=True,
    type=click.Choice(sorted(ARMS)),
    default=("naive",),
    show_default=True,
)
@click.option("--scale", type=click.Choice(sorted(SCALES)), default="smoke", show_default=True)
@click.option("--model", "model_label", type=click.Choice(sorted(POLICIES)), default="qwen", show_default=True)
@click.option(
    "--stage",
    type=click.Choice(("rl", "evaluation")),
    default="evaluation",
    show_default=True,
    help="Terminal stage per arm; dependencies are included automatically.",
)
@rl_build_options
def main(arms: tuple[str, ...], scale: str, model_label: str, stage: str) -> dict[str, ArtifactStep]:
    built = build_arms(specs=tuple(ARMS[arm] for arm in arms), scale=scale, policy=POLICIES[model_label])
    return {name: getattr(arm, stage) for name, arm in built.items()}


if __name__ == "__main__":
    main()
