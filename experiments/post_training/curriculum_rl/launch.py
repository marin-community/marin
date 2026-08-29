# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch curriculum-RL arms: SkyRL GRPO on Qwen3-0.6B over a graded math pool.

Each arm trains the same policy on the same pool with the same step budget and
differs only in how prompts are sampled per step. The naive arm is the pinned
trainer's uniform shuffle; the other arms require the curriculum sampler branch
of MarinSkyRL and fail fast until that lands.

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
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import click
from fray.types import ResourceConfig
from huggingface_hub import snapshot_download
from marin.evaluation.model_config import GenerationConfig, ModelConfig, ResourceHint, ServeConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import (
    SKYRL_POLICY_LOCATION,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLEvaluationModel,
    SkyRLModel,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.evaluation.pipeline import EvaluationResult, eval_step
from experiments.post_training.curriculum_rl.pool import TRAIN_FILENAME, VALIDATION_FILENAME, pool_step

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "curriculum-rl"
QWEN3_MODEL = "Qwen/Qwen3-0.6B"
QWEN3_REVISION = "c1899de"
HF_EXPORT_SUBDIR = "hf"
POOL_ARTIFACT_NAME = f"documents/{EXPERIMENT_NAME}-pool"
MODEL_ARTIFACT_NAME = f"models/{EXPERIMENT_NAME}-qwen3-0.6b"
CLUSTER = "cw-rno2a"
CLUSTER_CONFIG = f"lib/iris/config/{CLUSTER}.yaml"
GPU_VARIANT = "H100"
GPUS_PER_NODE = 8
WANDB_PROJECT = f"marin-{EXPERIMENT_NAME}"
SEED = 17


class SamplerKind(StrEnum):
    """How each step's rollout prompts are drawn from the graded pool."""

    NAIVE = "naive"
    THOMPSON = "thompson"
    GRADE_UNIFORM = "grade-uniform"
    GRADE_ADAPTIVE = "grade-adaptive"
    GRADE_PRIOR = "grade-prior"


# Widened once the MarinSkyRL curriculum sampler branch lands and the pin moves.
SUPPORTED_SAMPLERS = frozenset({SamplerKind.NAIVE})


def sampler_overrides(sampler: SamplerKind) -> tuple[str, ...]:
    if sampler is SamplerKind.NAIVE:
        return ()
    raise NotImplementedError(
        f"Sampler {sampler} needs the MarinSkyRL curriculum branch; the current pin only supports uniform shuffle."
    )


@dataclass(frozen=True)
class ScalePreset:
    """One resource/budget point: smoke validates wiring, full measures arms."""

    label: str
    num_nodes: int
    role_plan: SkyRLRolePlan
    max_steps: int
    eval_interval: int
    ckpt_interval: int
    request_window_tokens: int
    max_new_tokens: int
    micro_forward_batch_size_per_gpu: int
    evals: str


SMOKE = ScalePreset(
    label="smoke",
    num_nodes=2,
    role_plan=SkyRLRolePlan(
        colocate_all=False,
        policy_num_nodes=1,
        policy_num_gpus_per_node=GPUS_PER_NODE,
        num_inference_engines=GPUS_PER_NODE,
        inference_engine_tensor_parallel_size=1,
        train_batch_size=64,
        policy_mini_batch_size=32,
        micro_train_batch_size_per_gpu=4,
        n_samples_per_prompt=4,
    ),
    max_steps=4,
    eval_interval=-1,
    ckpt_interval=2,
    request_window_tokens=2048,
    max_new_tokens=1024,
    micro_forward_batch_size_per_gpu=8,
    evals="gsm8k-smoke",
)

# 64 GPUs: generation dominates for a 0.6B policy, so 2 training nodes and
# 6 single-GPU vLLM engine nodes. ~120 steps * 512 prompts * 8 samples at
# <=2048 new tokens bounds the run at roughly 0.3-1.0B output tokens.
FULL = ScalePreset(
    label="full",
    num_nodes=8,
    role_plan=SkyRLRolePlan(
        colocate_all=False,
        policy_num_nodes=2,
        policy_num_gpus_per_node=GPUS_PER_NODE,
        num_inference_engines=6 * GPUS_PER_NODE,
        inference_engine_tensor_parallel_size=1,
        train_batch_size=512,
        policy_mini_batch_size=64,
        micro_train_batch_size_per_gpu=8,
        n_samples_per_prompt=8,
    ),
    max_steps=120,
    eval_interval=10,
    ckpt_interval=10,
    request_window_tokens=3072,
    max_new_tokens=2048,
    micro_forward_batch_size_per_gpu=16,
    evals="math500,gsm8k-0shot",
)

SCALES = {preset.label: preset for preset in (SMOKE, FULL)}


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


def rl_config_yaml(preset: ScalePreset) -> str:
    plan = preset.role_plan
    return f"""\
entrypoint: standard

context_budget:
  request_window_tokens: {preset.request_window_tokens}
  max_new_tokens_per_turn: {preset.max_new_tokens}
  max_turns: 1

environment:
  env_class: gsm8k

trainer:
  strategy: fsdp2
  flash_attn: true
  use_sample_packing: false
  algorithm:
    advantage_estimator: grpo
    use_kl_loss: true
  epochs: 50
  max_steps: {preset.max_steps}
  update_epochs_per_batch: 1
  train_batch_size: {plan.train_batch_size}
  policy_mini_batch_size: {plan.policy_mini_batch_size}
  eval_batch_size: 256
  micro_forward_batch_size_per_gpu: {preset.micro_forward_batch_size_per_gpu}
  micro_train_batch_size_per_gpu: {plan.micro_train_batch_size_per_gpu}
  eval_before_train: {str(preset.eval_interval > 0).lower()}
  eval_interval: {preset.eval_interval}
  ckpt_interval: {preset.ckpt_interval}
  resume_mode: latest
  logger: wandb
  project_name: {WANDB_PROJECT}
  policy:
    optimizer_config:
      lr: 2.0e-6
      max_grad_norm: 1.0
    fsdp_config:
      cpu_offload: false
      reshard_after_forward: true
  placement:
    colocate_all: {str(plan.colocate_all).lower()}

generator:
  backend: vllm
  model_dtype: bfloat16
  vllm_attention_backend: FLASH_ATTN
  inference_engine_tensor_parallel_size: {plan.inference_engine_tensor_parallel_size}
  num_inference_engines: {plan.num_inference_engines}
  n_samples_per_prompt: {plan.n_samples_per_prompt}
  gpu_memory_utilization: 0.75
  enforce_eager: false
  run_engines_locally: true
  weight_sync_backend: nccl
  async_engine: true
  batched: true
  sampling_params:
    temperature: 1.0
    top_p: 1.0

data:
  kind: parquet
  train_data: []
  val_data: []
"""


@dataclass(frozen=True)
class CurriculumArm:
    sampler: SamplerKind
    rl: ArtifactStep[SkyRLModel]
    evaluation: ArtifactStep[EvaluationResult]


def build_arm(
    *,
    sampler: SamplerKind,
    preset: ScalePreset,
    version: str | None,
    model: ArtifactStep[LevanterCheckpoint],
    pool: ArtifactStep,
) -> CurriculumArm:
    suffix = "" if preset is FULL else f"-{preset.label}"
    rl_base_name = f"checkpoints/{EXPERIMENT_NAME}/{sampler.value}{suffix}"
    rl = skyrl_step(
        SkyRLSpec(
            name=user_owned_name(rl_base_name),
            version=version or resolve_version(rl_base_name, None),
            config_yaml=rl_config_yaml(preset),
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.FSDP),
            model=ArtifactHfModel(
                step=model,
                tokenizer_uri=QWEN3_MODEL,
                tokenizer_revision=QWEN3_REVISION,
                relative_path=HF_EXPORT_SUBDIR,
            ),
            train_data=(ArtifactDataSource(pool, relative_path=TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(pool, relative_path=VALIDATION_FILENAME),),
            topology=SkyRLTopology(
                num_nodes=preset.num_nodes,
                gpus_per_node=GPUS_PER_NODE,
                gpu_variant=GPU_VARIANT,
                role_plan=preset.role_plan,
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
            seed=SEED,
            overrides=sampler_overrides(sampler),
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=CLUSTER_CONFIG,
            cpu=16,
            memory="128GB",
            disk="2TB",
            priority="interactive",
            # vLLM engine startup can race on torch TCPStore ports when eight
            # single-GPU engines start on one node; those failures are transient
            # and resume from the latest checkpoint, so keep a wide budget.
            max_retries=6,
            wandb_entity="marin-community",
        ),
    )
    evaluation_base_name = f"evals/{EXPERIMENT_NAME}-{sampler.value}{suffix}/{preset.evals}"
    evaluation = eval_step(
        SkyRLEvaluationModel(
            step=rl,
            model=ModelConfig(
                name=f"{EXPERIMENT_NAME}-{sampler.value}{suffix}",
                location=SKYRL_POLICY_LOCATION,
                tokenizer=QWEN3_MODEL,
                apply_chat_template=True,
                resource_hint=ResourceHint(gpu={GPU_VARIANT: 1}),
                serve=ServeConfig(
                    tensor_parallel_size=1,
                    max_model_len=preset.request_window_tokens + preset.max_new_tokens,
                    max_num_seqs=64,
                ),
                generation=GenerationConfig(max_gen_toks=preset.max_new_tokens),
            ),
        ),
        preset.evals,
        version=version or resolve_version(evaluation_base_name, None),
        accelerator=f"{GPU_VARIANT}x1",
        submission_cluster=CLUSTER,
        federated_cluster=CLUSTER,
    )
    return CurriculumArm(sampler=sampler, rl=rl, evaluation=evaluation)


def build_arms(*, samplers: tuple[SamplerKind, ...], scale: str, version: str | None = None) -> dict[str, CurriculumArm]:
    preset = SCALES[scale]
    pool = pool_step(POOL_ARTIFACT_NAME, version or resolve_version(POOL_ARTIFACT_NAME, None))
    model = model_step(version or resolve_version(MODEL_ARTIFACT_NAME, None))
    return {
        sampler.value: build_arm(sampler=sampler, preset=preset, version=version, model=model, pool=pool)
        for sampler in samplers
    }


@click.command(help=__doc__)
@click.option(
    "--arm",
    "arms",
    multiple=True,
    type=click.Choice([kind.value for kind in SamplerKind]),
    default=(SamplerKind.NAIVE.value,),
    show_default=True,
)
@click.option("--scale", type=click.Choice(sorted(SCALES)), default="smoke", show_default=True)
@click.option(
    "--stage",
    type=click.Choice(("rl", "evaluation")),
    default="evaluation",
    show_default=True,
    help="Terminal stage per arm; dependencies are included automatically.",
)
@build_options
def main(arms: tuple[str, ...], scale: str, stage: str) -> dict[str, ArtifactStep]:
    built = build_arms(samplers=tuple(SamplerKind(arm) for arm in arms), scale=scale)
    return {name: getattr(arm, stage) for name, arm in built.items()}


if __name__ == "__main__":
    main()
