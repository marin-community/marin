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
from rigging.provenance import username_segment

from experiments.evaluation.models import SNOWBALL_VLLM_ARGS
from experiments.evaluation.pipeline import EvaluationResult, eval_step
from experiments.models import snowball_67b_a2b_sft
from experiments.post_training.curriculum_rl.pool import (
    MAX_PROMPT_TOKENS,
    QWEN3_MODEL,
    QWEN3_REVISION,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    pool_step,
)

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
    """Which policy model arms train, where it runs, and its model-specific overrides."""

    label: str
    cluster: str
    tokenizer_uri: str
    tokenizer_revision: str
    model_relative_path: str
    overrides: tuple[str, ...]


QWEN_POLICY = PolicySpec(
    label="qwen",
    cluster="cw-rno2a",
    tokenizer_uri=QWEN3_MODEL,
    tokenizer_revision=QWEN3_REVISION,
    model_relative_path=HF_EXPORT_SUBDIR,
    # Thinking mode ate the whole generation budget at 0.6B (85% truncation in
    # the round-1 smoke); Qwen arms train and roll out in non-thinking mode.
    overrides=("++generator.chat_template_kwargs.enable_thinking=false",),
)

# The Snowball SFT trains where its export lives (us-east-02a) so 134GB of
# shards never stream cross-region. Rollout engines mirror the export's
# serving profile: one node-sized vLLM instance, tensor_parallel_size=1,
# experts sharded across the node's eight ranks (ep = dp * tp). The frozen
# query-bias router is the pinned MarinSkyRL default.
SNOWBALL_POLICY = PolicySpec(
    label="snowball",
    cluster="cw-us-east-02a",
    tokenizer_uri=MARIN_TOKENIZER,
    tokenizer_revision=MARIN_TOKENIZER_REVISION,
    model_relative_path="",
    overrides=(
        "generator.inference_engine_data_parallel_size=8",
        "generator.inference_engine_expert_parallel_size=8",
    ),
)
POLICIES = {policy.label: policy for policy in (QWEN_POLICY, SNOWBALL_POLICY)}


class SamplerKind(StrEnum):
    """How each step's rollout prompts are drawn from the graded pool.

    Round 1 also ran a ``grade-uniform`` arm (equal budget per grade); it
    trailed every other curriculum arm and was dropped from the catalog.
    """

    NAIVE = "naive"
    THOMPSON = "thompson"
    LEARNABILITY = "learnability"
    GRADE_ADAPTIVE = "grade-adaptive"
    GRADE_PRIOR = "grade-prior"


# The launcher auto-defaults trainer.hf_hub_repo_id to laion/<job_name>, and the
# export job then needs create access to that org. Exports stay in object storage.
# (enable_thinking rides on the Qwen policy's overrides through ++ because the
# config flattener emits bare keys and hydra rejects new children under the
# empty chat_template_kwargs.)
BASE_OVERRIDES = ("++trainer.hf_hub_repo_id=null",)


# DAPO-style dynamic sampling: drop zero-advantage GRPO groups and keep
# drawing batches until a full batch of informative groups accumulates. The
# curriculum sampler is updated on raw pre-filter batches, so its statistics
# stay unbiased under filtering.
DAPO_OVERRIDE = "trainer.algorithm.dynamic_sampling.type=filter"


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
GROUP_INFORMATIVE_OVERRIDE = "data.sampling.weighting=group-informative"


def arm_overrides(spec: ArmSpec, policy: PolicySpec) -> tuple[str, ...]:
    """Per-arm hydra overrides; curriculum arms select a data.sampling policy.

    The naive sampler keeps ``data.sampling.kind`` at its null default, i.e. the
    stock uniform shuffle without replacement. Curriculum arms use the branch
    defaults for decay, priors, and adaptive thresholds so arms differ only in
    kind.
    """
    overrides = (*BASE_OVERRIDES, *policy.overrides)
    if spec.sampler is not SamplerKind.NAIVE:
        overrides = (*overrides, f"data.sampling.kind={spec.sampler.value}")
    if spec.sampler in GROUP_INFORMATIVE_SAMPLERS:
        overrides = (*overrides, GROUP_INFORMATIVE_OVERRIDE)
    if spec.dapo:
        overrides = (*overrides, DAPO_OVERRIDE)
    return overrides


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
# <=1024 non-thinking tokens bounds the run at roughly 0.2-0.5B output tokens.
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
    request_window_tokens=2048,
    max_new_tokens=1024,
    micro_forward_batch_size_per_gpu=16,
    evals="math500,gsm8k-0shot",
)

# The 67B-A2B smoke: four FSDP2 policy nodes hold the sharded parameters and
# AdamW state (~34GB/GPU), one node-sized expert-parallel engine generates.
# The #7786 campaign's NCCL rank-drop failure mode appeared only at 32k
# contexts; this preset stays at the 2k window.
SNOWBALL_SMOKE = ScalePreset(
    label="snowball-smoke",
    num_nodes=5,
    role_plan=SkyRLRolePlan(
        colocate_all=False,
        policy_num_nodes=4,
        policy_num_gpus_per_node=GPUS_PER_NODE,
        num_inference_engines=1,
        inference_engine_tensor_parallel_size=1,
        train_batch_size=32,
        policy_mini_batch_size=32,
        micro_train_batch_size_per_gpu=4,
        n_samples_per_prompt=4,
    ),
    max_steps=4,
    eval_interval=-1,
    ckpt_interval=4,
    request_window_tokens=2048,
    max_new_tokens=1024,
    micro_forward_batch_size_per_gpu=2,
    evals="gsm8k-smoke",
)

# The 67B-A2B measurement point: 4 FSDP2 policy nodes + 4 expert-parallel
# engine nodes. The smoke averaged 884 generated tokens against a 1024 cap,
# so the full runs widen the window to 3072 with a 2048 response budget
# (the 1024-token prompt budget still admits every pool row). 60 steps at
# 128x8 responses bounds an arm near the round-2 per-arm token budget.
SNOWBALL_FULL = ScalePreset(
    label="snowball-full",
    num_nodes=8,
    role_plan=SkyRLRolePlan(
        colocate_all=False,
        policy_num_nodes=4,
        policy_num_gpus_per_node=GPUS_PER_NODE,
        num_inference_engines=4,
        inference_engine_tensor_parallel_size=1,
        train_batch_size=128,
        policy_mini_batch_size=64,
        # At micro=1 the FSDP update ran 32 sequential micro-steps, each
        # re-gathering the full 134GB of shards; policy_train was ~1660s of a
        # ~1770s step. micro=4 quarters the all-gather traffic per step.
        micro_train_batch_size_per_gpu=4,
        n_samples_per_prompt=8,
    ),
    max_steps=60,
    eval_interval=10,
    ckpt_interval=10,
    request_window_tokens=3072,
    max_new_tokens=2048,
    micro_forward_batch_size_per_gpu=2,
    evals="math500,gsm8k-0shot",
)

SCALES = {preset.label: preset for preset in (SMOKE, FULL, SNOWBALL_SMOKE, SNOWBALL_FULL)}

# The pool filter must keep every retained prompt under each preset's
# max_input_length (request window minus generation budget), or retained rows
# skip generation and fully skipped GRPO groups fail admission.
for _preset in SCALES.values():
    assert MAX_PROMPT_TOKENS <= _preset.request_window_tokens - _preset.max_new_tokens, _preset.label


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
  batched: false
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
    spec: ArmSpec
    rl: ArtifactStep[SkyRLModel]
    evaluation: ArtifactStep[EvaluationResult]


def _evaluation_serving(policy: PolicySpec, preset: ScalePreset, name: str) -> ModelConfig:
    """Serving profile for the trained policy: single-GPU for Qwen, a full
    expert-parallel node for the Snowball export (see evaluation/models.py)."""
    max_model_len = preset.request_window_tokens + preset.max_new_tokens
    if policy is SNOWBALL_POLICY:
        return ModelConfig(
            name=name,
            location=SKYRL_POLICY_LOCATION,
            tokenizer=policy.tokenizer_uri,
            apply_chat_template=True,
            resource_hint=ResourceHint(gpu={GPU_VARIANT: GPUS_PER_NODE}, memory="512g"),
            serve=ServeConfig(
                tensor_parallel_size=1,
                data_parallel_size=GPUS_PER_NODE,
                max_model_len=max_model_len,
                max_num_seqs=64,
                vllm_extra_args=SNOWBALL_VLLM_ARGS,
            ),
            generation=GenerationConfig(
                max_gen_toks=preset.max_new_tokens,
                extra_gen_kwargs={"skip_special_tokens": "false", "repetition_penalty": "1.1"},
            ),
        )
    return ModelConfig(
        name=name,
        location=SKYRL_POLICY_LOCATION,
        tokenizer=policy.tokenizer_uri,
        apply_chat_template=True,
        resource_hint=ResourceHint(gpu={GPU_VARIANT: 1}),
        serve=ServeConfig(
            tensor_parallel_size=1,
            max_model_len=max_model_len,
            max_num_seqs=64,
        ),
        generation=GenerationConfig(max_gen_toks=preset.max_new_tokens),
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
    # Qwen arm names predate the policy axis and stay unprefixed so round-2
    # artifacts keep their addresses.
    policy_prefix = "" if policy is QWEN_POLICY else f"{policy.label}-"
    cluster_config = f"lib/iris/config/{policy.cluster}.yaml"
    rl_base_name = f"checkpoints/{EXPERIMENT_NAME}/{policy_prefix}{spec.name}{suffix}"
    rl = skyrl_step(
        SkyRLSpec(
            name=user_owned_name(rl_base_name),
            version=version or resolve_version(rl_base_name, None),
            config_yaml=rl_config_yaml(preset),
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.FSDP),
            model=ArtifactHfModel(
                step=model,
                tokenizer_uri=policy.tokenizer_uri,
                tokenizer_revision=policy.tokenizer_revision,
                relative_path=policy.model_relative_path,
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
            overrides=arm_overrides(spec, policy),
        ),
        IrisSkyRLExecution(
            cluster=policy.cluster,
            cluster_config=cluster_config,
            cpu=16,
            # The Snowball export streams ~134GB of bf16 shards through host
            # buffers on load (per node, policy and engine alike); 128GB of
            # host RAM OOM-killed the first smoke.
            memory="512GB" if policy is SNOWBALL_POLICY else "128GB",
            disk="2TB",
            priority="interactive",
            # Fail fast: a broken config surfaces on the first attempt, and a
            # healthy run resumes from its latest checkpoint on resubmission.
            max_retries=1,
            wandb_entity="marin-community",
        ),
    )
    # The eval artifact is keyed on the model name; include the owner so two
    # users at the same fixed version evaluate their own checkpoints rather
    # than sharing one cached result (the RL step is already user-owned).
    evaluation_model_name = f"{username_segment()}-{EXPERIMENT_NAME}-{policy_prefix}{spec.name}{suffix}"
    evaluation_base_name = f"evals/{evaluation_model_name}/{preset.evals}"
    evaluation = eval_step(
        SkyRLEvaluationModel(
            step=rl,
            model=_evaluation_serving(policy, preset, evaluation_model_name),
        ),
        preset.evals,
        version=version or resolve_version(evaluation_base_name, None),
        accelerator=f"{GPU_VARIANT}x{GPUS_PER_NODE if policy is SNOWBALL_POLICY else 1}",
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
    preset = SCALES[scale]
    pool = pool_step(POOL_ARTIFACT_NAME, version or resolve_version(POOL_ARTIFACT_NAME, None))
    if policy is SNOWBALL_POLICY:
        model = snowball_67b_a2b_sft
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
@build_options
def main(arms: tuple[str, ...], scale: str, model_label: str, stage: str) -> dict[str, ArtifactStep]:
    built = build_arms(specs=tuple(ARMS[arm] for arm in arms), scale=scale, policy=POLICIES[model_label])
    return {name: getattr(arm, stage) for name, arm in built.items()}


if __name__ == "__main__":
    main()
