# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Optimize the step-12 student with verifier-selected complement preferences."""

from dataclasses import dataclass, replace

import click
import jmp
from fray.types import ResourceConfig
from levanter.data.text.datasets import DatasetComponent
from levanter.data.text.preference import PreferenceChatLmDatasetFormat, PreferenceLmDataConfig
from levanter.main.train_dpo import SeparateReferenceConfig, TrainDpoConfig
from levanter.models.snowball import SnowballConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import ArtifactHfModel, SkyRLRun
from marin.training.training import (
    LevanterCheckpoint,
    TrainDpoOnPodConfig,
    resolve_training_env,
    run_levanter_train_dpo,
)
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.bfcl_rl.collect import COLLECTION_EXECUTION, MODELS, SMOKE_TASKS, collection_step
from experiments.post_training.bfcl_rl.recovery import recovery_cache_step
from experiments.post_training.bfcl_rl.recovery_data import RecoveryPreferenceCache

RECOVERY_CONTEXT = 40960


@dataclass(frozen=True)
class RecoveryOptimization:
    num_train_steps: int
    batch_size: int
    beta: float
    learning_rate: float
    hf_save_steps: int
    num_nodes: int
    expert_axis: int
    context_axis: int
    jax_memory_fraction: float


def dispatch_recovery_training(config: TrainDpoOnPodConfig) -> None:
    """Resolve the GPU environment before the worker imports JAX."""
    env = resolve_training_env(config.env_vars, config.resources)
    remote(run_levanter_train_dpo, resources=config.resources, env_vars=env)(config)


def initial_student_model() -> ArtifactHfModel:
    source = MODELS["student"]
    initial_student = ArtifactStep.adopt(
        user_owned_name("models/bfcl-rl-student"),
        source.version,
        source.uri,
        kind=LevanterCheckpoint,
        config={"model": source.model, "revision": source.revision},
    )
    return ArtifactHfModel(initial_student, source.model, source.revision, relative_path="")


def recovery_optimizer_step(
    cache: ArtifactStep[RecoveryPreferenceCache],
    *,
    initial_policy: ArtifactHfModel,
    selection_name: str,
    optimization: RecoveryOptimization,
    resume_checkpoint: ArtifactStep[LevanterCheckpoint] | None = None,
) -> ArtifactStep[LevanterCheckpoint]:
    """Bind preference tokens and an explicit round-start policy to the existing DPO trainer."""
    source = MODELS["student"]
    if (initial_policy.tokenizer_uri, initial_policy.tokenizer_revision) != (source.model, source.revision):
        raise ValueError("DPO initialization must retain the mandated Snowball tokenizer")
    resources = ResourceConfig.with_gpu(
        "H100",
        count=8,
        replicas=optimization.num_nodes,
        cpu=48,
        ram="1611Gi",
        disk="21745Gi",
    )
    if resources.chip_count() > 128:
        raise ValueError("Recovery optimizer exceeds the campaign's 128-H100 limit")
    mesh = MeshConfig(
        axes={
            "data": -1,
            "replica": 1,
            "model": 1,
            "expert": optimization.expert_axis,
        },
        dcn_axes={"context": optimization.context_axis, "replica_dcn": -1},
        compute_mapping={
            "batch": ["replica_dcn", "data", "expert"],
            "vocab": "model",
            "position": "context",
        },
    )
    ici, dcn = mesh.axis_shapes(resources.chip_count(), optimization.num_nodes)
    data_parallel_size = dcn["replica_dcn"] * ici["data"] * ici["expert"]
    if optimization.batch_size % data_parallel_size:
        raise ValueError("Recovery batch must be divisible by the data/expert mesh width")
    if RECOVERY_CONTEXT % optimization.context_axis:
        raise ValueError("Recovery context must be divisible by the context mesh width")
    name = user_owned_name(f"models/bfcl-rl-recovery-dpo-{selection_name}")

    def build_config(ctx: StepContext) -> TrainDpoOnPodConfig:
        if ctx.is_fingerprint:
            cache_path = ctx.artifact_path(cache)
        else:
            preferences = ctx.resolved(cache)
            if preferences.num_preferences == 0:
                raise ValueError("No verifier-selected preferences; no optimizer update")
            if preferences.max_length != RECOVERY_CONTEXT:
                raise ValueError("Preference cache and optimizer context differ")
            if (preferences.tokenizer_uri, preferences.tokenizer_revision) != (source.model, source.revision):
                raise ValueError("Preference cache tokenizer differs from the initial student")
            cache_path = preferences.path
        model_path = initial_policy.resolve(ctx).uri
        data = PreferenceLmDataConfig(
            tokenizer=f"{source.model}@{source.revision}",
            auto_build_caches=False,
            shuffle=True,
            components={
                "bfcl_complement": DatasetComponent(
                    cache_dir=prefix_join(cache_path, "train"),
                    split="train",
                    flat_cache=True,
                    format=PreferenceChatLmDatasetFormat(pack=False, mask_user_turns=True, slice_strategy="raise"),
                )
            },
        )
        trainer = TrainerConfig(
            seed=42,
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            mesh=mesh,
            use_explicit_mesh_axes=True,
            train_batch_size=optimization.batch_size,
            per_device_parallelism=1,
            num_train_steps=optimization.num_train_steps,
            load_checkpoint=True if resume_checkpoint is not None else None,
            load_checkpoint_path=ctx.artifact_path(resume_checkpoint) if resume_checkpoint is not None else None,
            tracker=WandbConfig(project="bfcl-rl", group="verifier-selected-recovery", mode="online"),
            log_jaxprs=False,
            log_xla_hlo=False,
        )
        train = TrainDpoConfig(
            data=data,
            trainer=trainer,
            model=SnowballConfig(
                max_seq_len=262144,
                qk_mult=1.75,
                initializer_std=0.009882117688026186,
                attention_implementation="gpu_fa4_cute",
                moe_implementation="ring",
            ),
            train_seq_len=RECOVERY_CONTEXT,
            optimizer=AdamConfig(
                learning_rate=optimization.learning_rate,
                weight_decay=0.0,
                max_grad_norm=0.5,
                beta1=0.9,
                beta2=0.999,
                warmup=0.0,
                lr_schedule="constant",
            ),
            initialize_from_hf=model_path,
            reference=SeparateReferenceConfig(model_path=model_path, is_hf=True),
            beta=optimization.beta,
            validation_split_fraction=None,
            run_initial_eval=False,
            hf_save_steps=optimization.hf_save_steps,
            hf_save_dtype="bfloat16",
        )
        return TrainDpoOnPodConfig(
            train,
            resources,
            output_path=ctx.output_path,
            auto_build_caches=False,
            env_vars={"XLA_PYTHON_CLIENT_MEM_FRACTION": str(optimization.jax_memory_fraction)},
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=LevanterCheckpoint,
        run=dispatch_recovery_training,
        build_config=build_config,
        deps=(cache, initial_policy.step) + ((resume_checkpoint,) if resume_checkpoint is not None else ()),
    )


@click.command(help=__doc__)
@click.option("--task", type=click.Choice(SMOKE_TASKS), default=None)
@click.option("--cache-version", default=None, help="Reuse a completed recovery cache without scheduling collection.")
@click.option("--collection-version", default=None, help="Build preferences from completed paired collection artifacts.")
@click.option("--resume-version", default=None, help="Resume a native checkpoint from an earlier recovery run.")
@click.option("--resume-checkpoint-step", type=click.IntRange(min=0), default=None)
@click.option("--python-image", required=True)
@click.option("--java-image", required=True)
@click.option("--javascript-image", required=True)
@click.option("--num-train-steps", type=click.IntRange(min=1), required=True)
@click.option("--batch-size", type=click.IntRange(min=1), required=True)
@click.option("--beta", type=click.FloatRange(min=0, min_open=True), required=True)
@click.option("--learning-rate", type=click.FloatRange(min=0, min_open=True), required=True)
@click.option("--hf-save-steps", type=click.IntRange(min=1), required=True)
@click.option("--num-nodes", type=click.IntRange(min=1, max=16), required=True)
@click.option("--expert-axis", type=click.IntRange(min=1), required=True)
@click.option("--context-axis", type=click.IntRange(min=1), required=True)
@click.option("--jax-memory-fraction", type=click.FloatRange(min=0, max=1, min_open=True), required=True)
@rl_build_options
def main(
    task: str | None,
    cache_version: str | None,
    collection_version: str | None,
    resume_version: str | None,
    resume_checkpoint_step: int | None,
    python_image: str,
    java_image: str,
    javascript_image: str,
    num_train_steps: int,
    batch_size: int,
    beta: float,
    learning_rate: float,
    hf_save_steps: int,
    num_nodes: int,
    expert_axis: int,
    context_axis: int,
    jax_memory_fraction: float,
) -> ArtifactStep:
    selection = task or "full"
    if (resume_version is None) != (resume_checkpoint_step is None):
        raise ValueError("Specify both resume version and native checkpoint step")
    resume_checkpoint = None
    if resume_version is not None:
        recovery_name = user_owned_name(f"models/bfcl-rl-recovery-dpo-{selection}")
        resume_checkpoint = ArtifactStep.adopt(
            f"{recovery_name}-native-input",
            resume_version,
            f"{recovery_name}/{resume_version}/checkpoints/step-{resume_checkpoint_step}",
            kind=LevanterCheckpoint,
        )
    if cache_version is not None and collection_version is not None:
        raise ValueError("Specify a cache version or a collection version, not both")
    if cache_version is not None:
        cache_name = user_owned_name(f"data/bfcl-rl-recovery-preferences-{selection}")
        cache = ArtifactStep.adopt(
            f"{cache_name}-input",
            cache_version,
            f"{cache_name}/{cache_version}",
            kind=RecoveryPreferenceCache,
        )
    elif collection_version is not None:
        rollouts = tuple(
            ArtifactStep.adopt(
                user_owned_name(f"rollouts/bfcl-rl-recovery-{model}-{selection}-input"),
                collection_version,
                f"{user_owned_name(f'rollouts/bfcl-rl-recovery-{model}-{selection}')}/{collection_version}",
                kind=SkyRLRun,
            )
            for model in ("teacher", "student")
        )
        cache = recovery_cache_step(*rollouts, selection_name=selection, max_length=RECOVERY_CONTEXT)
    else:
        images = (python_image, java_image, javascript_image)
        teacher = collection_step("teacher", task, images)
        student = collection_step("student", task, images)
        cache = recovery_cache_step(teacher, student, selection_name=selection, max_length=RECOVERY_CONTEXT)
    optimization = RecoveryOptimization(
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        beta=beta,
        learning_rate=learning_rate,
        hf_save_steps=hf_save_steps,
        num_nodes=num_nodes,
        expert_axis=expert_axis,
        context_axis=context_axis,
        jax_memory_fraction=jax_memory_fraction,
    )
    optimizer = recovery_optimizer_step(
        cache,
        initial_policy=initial_student_model(),
        selection_name=selection,
        optimization=optimization,
        resume_checkpoint=resume_checkpoint,
    )
    return replace(optimizer, runtime_args={"execution": COLLECTION_EXECUTION})


if __name__ == "__main__":
    main()
