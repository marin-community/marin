# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 32B to 0.6B fixed-budget distillation study.

Run a one-arm systems smoke before the screen:

    python -m experiments.qwen_distillation --version dev --stage smoke --run

Launch the two-seed screen from ``cw-us-east-08a`` at batch priority:

    python -m experiments.qwen_distillation --version dev --stage screen --run --max-concurrent 18
"""

import math
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum

import click
import jmp
from fray.types import GpuConfig, ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.distillation import DistillationObjective
from levanter.distillation_initialization import TeacherInitialization
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from levanter.main.distill_lm import TrainLmDistillationConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from levanter.tokenizers import TokenizerBackend
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.tokenize import TokenizeConfig, TokenizedCache, tokenize
from marin.training.training import (
    LevanterCheckpoint,
    TrainLmOnPodConfig,
    resolve_training_env,
    run_levanter_distill_lm,
    run_levanter_train_lm,
)

from experiments.models import qwen3_0_6b_base, qwen3_4b, qwen3_32b

SERIES = "QD"
WANDB_GROUP = "QD-qwen32b-to-0p6b"
SEQ_LEN = 2048
BATCH_SIZE = 8
SCREEN_TOKENS = 100_000_000
SCREEN_STEPS = math.ceil(SCREEN_TOKENS / (SEQ_LEN * BATCH_SIZE))
SMOKE_STEPS = 12
SCREEN_SEEDS = (0, 1)

SAMPLE_0P1B = ArtifactStep.adopt(
    "qwen-distillation/raw/datakit-0p1b",
    "2026.07.26",
    "s3://marin-us-east-02a/marin/datakit/sample_0.1b_7d7d8fd7",
    kind=Artifact,
)

_VALIDATION_SOURCE = "cp/wikiteam"
_TRAIN_SOURCES = (
    "cp/arxiv_abstracts",
    "starcoder2/ir_python",
    "numinamath-1.5",
    "finepdfs/spa_Latn",
    "nemotron_cc_v2/medium_quality",
    "nemotron_sft/sft_general",
    "hplt_v3",
    "swe-rebench-openhands",
)

TOKENIZE_RESOURCES = ResourceConfig.with_cpu(cpu=8, ram="32g", disk="64g")
TRAIN_RESOURCES = ResourceConfig.with_gpu(
    "GB200",
    count=4,
    cpu=64,
    ram="512g",
    disk="1t",
)

_TOKEN_AXES = (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)


class Arm(StrEnum):
    CE_SCRATCH = "QD-C0"
    KL_SCRATCH = "QD-C1"
    CE_BASE = "QD-C2"
    KL_BASE = "QD-C3"
    HIDDEN = "QD-001"
    FACTORIZED = "QD-002"
    TAID = "QD-003"
    STRUCTURED = "QD-004"
    FOUR_B_TEACHER = "QD-005"


@dataclass(frozen=True)
class ArmConfig:
    objective: DistillationObjective | None
    student_base: bool = False
    initialization: TeacherInitialization | None = None
    teacher_4b: bool = False


ARMS = {
    Arm.CE_SCRATCH: ArmConfig(objective=None),
    Arm.KL_SCRATCH: ArmConfig(objective=DistillationObjective.FORWARD_KL),
    Arm.CE_BASE: ArmConfig(objective=None, student_base=True),
    Arm.KL_BASE: ArmConfig(objective=DistillationObjective.FORWARD_KL, student_base=True),
    Arm.HIDDEN: ArmConfig(objective=DistillationObjective.PROJECTED_HIDDEN),
    Arm.FACTORIZED: ArmConfig(
        objective=DistillationObjective.FORWARD_KL,
        initialization=TeacherInitialization.FACTORIZED,
    ),
    Arm.TAID: ArmConfig(objective=DistillationObjective.TAID),
    Arm.STRUCTURED: ArmConfig(
        objective=DistillationObjective.FORWARD_KL,
        initialization=TeacherInitialization.STRUCTURED,
    ),
    Arm.FOUR_B_TEACHER: ArmConfig(
        objective=DistillationObjective.FORWARD_KL,
        teacher_4b=True,
    ),
}


def qwen3_0p6b_config(*, reference_checkpoint: str) -> Qwen3Config:
    return Qwen3Config(
        max_seq_len=32_768,
        hidden_dim=1024,
        intermediate_dim=3072,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
        layer_norm_epsilon=1e-6,
        tie_word_embeddings=True,
        rope=DefaultRotaryEmbeddingsConfig(theta=1_000_000),
        use_sliding_window=False,
        reference_checkpoint=reference_checkpoint,
    )


def _mesh() -> MeshConfig:
    return MeshConfig(
        axes={"replica": 1, "data": 1, "model": 4},
        shared_mapping={"vocab": "model"},
        compute_mapping={"token": _TOKEN_AXES, "token_repeat": _TOKEN_AXES},
    )


def qwen_datakit_cache() -> ArtifactStep[TokenizedCache]:
    name = "qwen-distillation/data/datakit-0p1b-qwen3"
    version = resolve_version(name, None)

    def build_config(ctx: StepContext) -> TokenizeConfig:
        root = ctx.artifact_path(SAMPLE_0P1B)
        return TokenizeConfig(
            train_paths=[f"{root}/{source}/**/*.parquet" for source in _TRAIN_SOURCES],
            validation_paths=[f"{root}/{_VALIDATION_SOURCE}/**/*.parquet"],
            cache_path=ctx.output_path,
            tokenizer=ctx.artifact_path(qwen3_0_6b_base),
            tokenizer_backend=TokenizerBackend.HF,
            format=TextLmDatasetFormat(text_key="text"),
            max_workers=256,
            worker_resources=ResourceConfig.with_cpu(cpu=2, ram="12g", disk="16g"),
            tags=["datakit", "qwen-distillation"],
        )

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=TokenizedCache,
        run=remote(tokenize, resources=TOKENIZE_RESOURCES),
        build_config=build_config,
        deps=(SAMPLE_0P1B, qwen3_0_6b_base),
    )


def _optimizer() -> AdamConfig:
    return AdamConfig(
        learning_rate=3e-4,
        weight_decay=0.1,
        warmup=0.05,
        decay=0.1,
    )


def _trainer(
    *,
    run_id: str,
    arm: Arm,
    seed: int,
    num_train_steps: int,
    output_path: str,
) -> TrainerConfig:
    eval_interval = max(1, math.ceil(num_train_steps / 8))
    return TrainerConfig(
        id=run_id,
        seed=seed,
        tracker=WandbConfig(
            project="marin",
            name=run_id,
            group=WANDB_GROUP,
            tags=[SERIES, "issue-7656", arm.value, f"seed-{seed}"],
            replicate_path=output_path,
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=BATCH_SIZE,
        per_device_parallelism=-1,
        num_train_steps=num_train_steps,
        steps_per_eval=eval_interval,
        max_eval_batches=16,
        checkpointer=CheckpointerConfig(
            save_interval=timedelta(minutes=10),
            keep=[],
        ),
        mesh=_mesh(),
        per_device_eval_parallelism=-1,
        allow_nondivisible_batch_size=True,
    )


def _training_job(config: TrainLmOnPodConfig) -> None:
    env_vars = (
        resolve_training_env(config.env_vars, config.resources) if isinstance(config.resources.device, GpuConfig) else {}
    )
    entrypoint = run_levanter_train_lm if isinstance(config.train_config, TrainLmConfig) else run_levanter_distill_lm
    remote(entrypoint, resources=config.resources, env_vars=env_vars)(config)


def training_step(
    arm: Arm,
    *,
    seed: int,
    num_train_steps: int,
    label: str,
    data: ArtifactStep[TokenizedCache],
) -> ArtifactStep[LevanterCheckpoint]:
    name = f"qwen-distillation/{label}/{arm.value.lower()}-seed-{seed}"
    version = resolve_version(name, None)
    arm_config = ARMS[arm]
    teacher_checkpoint = qwen3_4b if arm_config.teacher_4b else qwen3_32b

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        data_config = mixture(ctx, {data: 1.0}, shuffle=True)
        student_checkpoint = ctx.artifact_path(qwen3_0_6b_base)
        teacher_path = ctx.artifact_path(teacher_checkpoint)
        run_id = f"{arm.value}-{label}-seed-{seed}"
        trainer = _trainer(
            run_id=run_id,
            arm=arm,
            seed=seed,
            num_train_steps=num_train_steps,
            output_path=ctx.output_path,
        )

        if arm_config.objective is None:
            inner = TrainLmConfig(
                data=data_config,
                trainer=trainer,
                model=qwen3_0p6b_config(reference_checkpoint=student_checkpoint),
                optimizer=_optimizer(),
                train_seq_len=SEQ_LEN,
                initialize_from_hf=student_checkpoint if arm_config.student_base else False,
                use_hf_model_config=arm_config.student_base,
                pad_tokenizer_to_match_model=True,
            )
        else:
            inner = TrainLmDistillationConfig(
                data=data_config,
                trainer=trainer,
                student_model=qwen3_0p6b_config(reference_checkpoint=student_checkpoint),
                teacher_model=Qwen3Config(reference_checkpoint=teacher_path),
                optimizer=_optimizer(),
                train_seq_len=SEQ_LEN,
                objective=arm_config.objective,
                student_initialize_from_hf=student_checkpoint if arm_config.student_base else False,
                student_use_hf_model_config=arm_config.student_base,
                teacher_initialize_from_hf=teacher_path,
                teacher_use_hf_model_config=True,
                teacher_initialization=arm_config.initialization,
            )

        return TrainLmOnPodConfig(
            train_config=inner,
            resources=ctx.runtime_arg("train_resources"),
            output_path=ctx.output_path,
        )

    model_deps = (qwen3_0_6b_base,)
    if arm_config.objective is not None:
        model_deps = (*model_deps, teacher_checkpoint)
    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=_training_job,
        build_config=build_config,
        deps=(data, *model_deps),
        runtime_args={"train_resources": TRAIN_RESOURCES},
    )


def build(stage: str) -> list[ArtifactStep]:
    data = qwen_datakit_cache()
    if stage == "data":
        return [data]
    if stage == "smoke":
        return [
            training_step(
                Arm.KL_SCRATCH,
                seed=0,
                num_train_steps=SMOKE_STEPS,
                label="smoke",
                data=data,
            )
        ]
    if stage == "screen":
        return [
            training_step(
                arm,
                seed=seed,
                num_train_steps=SCREEN_STEPS,
                label="screen",
                data=data,
            )
            for arm in Arm
            for seed in SCREEN_SEEDS
        ]
    raise ValueError(f"Unsupported stage: {stage}")


@click.command()
@click.option("--stage", type=click.Choice(["data", "smoke", "screen"]), required=True)
@build_options
def main(stage: str):
    return build(stage)


if __name__ == "__main__":
    main()
