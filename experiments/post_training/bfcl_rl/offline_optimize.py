# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue the recovered student on verifier-correct, student-tokenized Qwen traces."""

from dataclasses import replace

import click
import jmp
from fray.types import ResourceConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.snowball import SnowballConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.datakit.sft import SftTokenStore, sft_data_config
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import ArtifactHfModel
from marin.training.training import (
    LevanterCheckpoint,
    TrainLmOnPodConfig,
    resolve_training_env,
    run_levanter_train_lm,
)
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.bfcl_rl.collect import COLLECTION_EXECUTION, MODELS
from experiments.post_training.bfcl_rl.launch import recovered_model
from experiments.post_training.bfcl_rl.optimize import RECOVERY_CONTEXT


def dispatch_offline_training(config: TrainLmOnPodConfig) -> None:
    env = resolve_training_env(config.env_vars, config.resources)
    remote(run_levanter_train_lm, resources=config.resources, env_vars=env)(config)


def offline_optimizer_step(
    corpus: ArtifactStep[SftTokenStore], policy: ArtifactHfModel, *, num_train_steps: int
) -> ArtifactStep[LevanterCheckpoint]:
    """Bind the shared masked SFT store to the recovered student's next update."""
    source = MODELS["student"]
    tokenizer = f"{source.model}@{source.revision}"
    resources = ResourceConfig.with_gpu("H100", count=8, replicas=16, cpu=48, ram="1611Gi", disk="21745Gi")
    mesh = MeshConfig(
        axes={"data": -1, "replica": 1, "model": 1, "expert": 8},
        dcn_axes={"context": 16, "replica_dcn": -1},
        compute_mapping={"batch": ["replica_dcn", "data", "expert"], "vocab": "model", "position": "context"},
    )
    mesh.axis_shapes(resources.chip_count(), resources.replicas)
    name = user_owned_name("models/bfcl-rl-offline-sft")

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        if ctx.is_fingerprint:
            store = SftTokenStore(
                path=ctx.artifact_path(corpus),
                cache_path=prefix_join(ctx.artifact_path(corpus), "student-store/train"),
                tokenizer=tokenizer,
                max_length=RECOVERY_CONTEXT,
                seed=42,
                sources={},
                packed_sequences=1,
            )
        else:
            store = ctx.resolved(corpus)
            if store.tokenizer != tokenizer or store.max_length != RECOVERY_CONTEXT:
                raise ValueError("Offline corpus must use the pinned student tokenizer and recovery context")
        train = TrainLmConfig(
            data=sft_data_config({"bfcl_complement": store}, minimum_weight=1.0),
            trainer=TrainerConfig(
                seed=42,
                mp=jmp.get_policy("p=f32,c=bfloat16"),
                mesh=mesh,
                use_explicit_mesh_axes=True,
                train_batch_size=16,
                per_device_parallelism=1,
                num_train_steps=num_train_steps,
                load_checkpoint=False,
                tracker=WandbConfig(project="bfcl-rl", group="verifier-selected-offline-sft", mode="online"),
                log_jaxprs=False,
                log_xla_hlo=False,
            ),
            model=SnowballConfig(
                max_seq_len=262144,
                qk_mult=1.75,
                initializer_std=0.009882117688026186,
                attention_implementation="gpu_fa4_cute",
                moe_implementation="ring",
            ),
            train_seq_len=RECOVERY_CONTEXT,
            optimizer=AdamConfig(
                learning_rate=4e-6,
                weight_decay=0.0,
                max_grad_norm=0.5,
                beta1=0.9,
                beta2=0.999,
                warmup=0.0,
                lr_schedule="constant",
            ),
            initialize_from_hf=policy.resolve(ctx).uri,
            hf_save_steps=num_train_steps,
            hf_save_dtype="bfloat16",
        )
        return TrainLmOnPodConfig(
            train,
            resources,
            output_path=ctx.output_path,
            auto_build_caches=False,
            env_vars={"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.75"},
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=LevanterCheckpoint,
        run=dispatch_offline_training,
        build_config=build_config,
        deps=(corpus, *policy.deps()),
    )


@click.command(help=__doc__)
@click.option("--corpus-version", required=True)
@click.option("--recovery-version", required=True)
@click.option("--policy-export-version", required=True)
@click.option("--policy-checkpoint-step", type=click.IntRange(min=0), required=True)
@click.option("--num-train-steps", type=click.IntRange(min=1), required=True)
@rl_build_options
def main(
    corpus_version: str,
    recovery_version: str,
    policy_export_version: str,
    policy_checkpoint_step: int,
    num_train_steps: int,
) -> ArtifactStep:
    corpus_name = user_owned_name("data/bfcl-rl-qwen36-harmony")
    corpus = ArtifactStep.adopt(corpus_name, corpus_version, f"{corpus_name}/{corpus_version}", kind=SftTokenStore)
    policy = replace(
        recovered_model(recovery_version, policy_export_version), relative_path=f"hf/step-{policy_checkpoint_step}"
    )
    step = offline_optimizer_step(corpus, policy, num_train_steps=num_train_steps)
    return replace(step, runtime_args={"execution": COLLECTION_EXECUTION})


if __name__ == "__main__":
    main()
