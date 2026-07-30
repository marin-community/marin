# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Matched chat SFT from the corrected augmented d768 burn checkpoints."""

import dataclasses
import os
from enum import StrEnum

from fray.cluster import ResourceConfig
from marin.execution.lazy import ArtifactStep
from marin.execution.step_runner import StepRunner
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.grug.moe.launch_cw_scale import build_scale_model
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig
from experiments.grug.moe.sft_launch import GrugModel
from experiments.marin_tokenizer import MARIN_CHAT_TEMPLATE, marin_tokenizer
from experiments.sft.launcher import DatasetSpec, SFTSpec, sft_step

_EXPERIMENT_ID = "NEST-AUGDK-SFT"
_SEQUENCE_LENGTH = 8192
_BATCH_SIZE = 32
_WILDCHAT_STEPS = 1_000
_THINKING_STEPS = 1_000

_WILDCHAT = DatasetSpec(
    slug="wildchat_386k",
    hf_dataset_id="nyu-dice-lab/wildchat50m-rewild-sft-385700",
    revision="46a5bb5",
    adapter_kwargs={"conversation_column": "conversation"},
    weight=1.0,
)
_THINKING = DatasetSpec(
    slug="nemotron_science_think",
    hf_dataset_id="laion/llama-nemotron-science-reasoning-on-canonical-think-full",
    revision="bae881d",
    adapter_kwargs={},
    weight=1.0,
)
_OPTIMIZER = GrugMoeAdamHConfig(
    learning_rate=5e-5,
    adam_lr=5e-5,
    beta1=0.9,
    beta2=0.95,
    epsilon=1e-8,
    max_grad_norm=1.0,
    weight_decay=0.0,
    min_lr_ratio=0.1,
    warmup=0.03,
    lr_schedule="cosine",
)


class SFTArm(StrEnum):
    E256 = "e256"
    E128_STANDALONE = "e128-standalone"
    FIXED25 = "fixed25"
    E128_NAIVE25 = "e128-naive25"
    E16_NAIVE25 = "e16-naive25"
    E128_LAYER25 = "e128-layer25"
    E16_LAYER25 = "e16-layer25"

    @property
    def nested_expert_counts(self) -> tuple[int, ...]:
        if self in (SFTArm.E256, SFTArm.E128_STANDALONE):
            return ()
        if self is SFTArm.FIXED25:
            return (128, 16)
        if self in (SFTArm.E128_NAIVE25, SFTArm.E128_LAYER25):
            return (128,)
        return (16,)

    @property
    def num_experts(self) -> int:
        if self is SFTArm.E128_STANDALONE:
            return 128
        return 256

    @property
    def nested_layer_fraction(self) -> float:
        if self in (SFTArm.E128_LAYER25, SFTArm.E16_LAYER25):
            return 0.25
        return 1.0


class SFTStage(StrEnum):
    WILDCHAT = "wildchat"
    THINKING = "thinking"

    @property
    def dataset(self) -> DatasetSpec:
        if self is SFTStage.WILDCHAT:
            return _WILDCHAT
        return _THINKING

    @property
    def steps(self) -> int:
        if self is SFTStage.WILDCHAT:
            return _WILDCHAT_STEPS
        return _THINKING_STEPS


def _required_env(key: str) -> str:
    value = os.environ.get(key)
    if value is None:
        raise ValueError(f"{key} must be set")
    return value


def sft_routing_model(model: GrugModelConfig, arm: SFTArm, routing: str) -> GrugModelConfig:
    """Validate an SFT model against its pretraining arm and select routing."""
    if model.num_experts != arm.num_experts:
        raise ValueError(f"{arm.value} SFT requires SCALE_NUM_EXPERTS={arm.num_experts}")
    if model.nested_expert_counts != arm.nested_expert_counts:
        raise ValueError(
            f"{arm.value} SFT requires SCALE_NESTED_COUNTS="
            f"{','.join(str(count) for count in arm.nested_expert_counts)}"
        )
    if arm in (SFTArm.E256, SFTArm.E128_STANDALONE):
        return model
    if model.nested_layer_fraction != arm.nested_layer_fraction:
        raise ValueError(f"{arm.value} SFT requires SCALE_NESTED_LAYER_FRACTION={arm.nested_layer_fraction}")
    if routing == "full":
        return dataclasses.replace(model, nested_batch_fraction=0.0)
    if routing != "nested":
        raise ValueError("AUGDK_SFT_ROUTING must be 'nested' or 'full'")
    if model.nested_batch_fraction != 0.25:
        raise ValueError("nested SFT requires SCALE_NESTED_FRACTION=0.25")
    return model


def build(*, version: str = "dev") -> ArtifactStep[LevanterCheckpoint]:
    """Build one matched SFT stage."""
    arm = SFTArm(_required_env("AUGDK_SFT_ARM"))
    stage = SFTStage(_required_env("AUGDK_SFT_STAGE"))
    init_from = _required_env("AUGDK_SFT_INIT_FROM")
    explicit_steps = int(os.environ.get("AUGDK_SFT_STEPS", stage.steps))
    if explicit_steps <= 0:
        raise ValueError("AUGDK_SFT_STEPS must be positive")

    model = build_scale_model()
    model = sft_routing_model(model, arm, os.environ.get("AUGDK_SFT_ROUTING", "nested"))

    run_id = f"nest-augdk-{arm.value}-{stage.value}-sft-r1"
    suffix = os.environ.get("AUGDK_SFT_RUN_SUFFIX")
    if suffix:
        run_id = f"{run_id}-{suffix}"
    step_name = f"experiments/nested-moe-augdk-sft/{run_id}"

    model_source = GrugModel(
        model=model,
        tokenizer_path=marin_tokenizer,
        init_from_path=init_from,
        expert_parallel=1,
        replica_axis=1,
        per_device_parallelism=2,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        z_loss_weight=1e-4,
        checkpoint_keep=[{"every": explicit_steps}],
        wandb_tags=["moe", "nested-moe", "aug-dk", "sft", arm.value, stage.value, "h100"],
        wandb_group=_EXPERIMENT_ID,
    )
    spec = SFTSpec(
        name=user_namespaced_name(step_name, version),
        version=version,
        model=model_source,
        chat_template=MARIN_CHAT_TEMPLATE,
        datasets=[stage.dataset],
        optimizer=_OPTIMIZER,
        seq_len=_SEQUENCE_LENGTH,
        batch_size=_BATCH_SIZE,
        num_train_steps=explicit_steps,
        wandb_project="marin_moe_sft",
    )
    resources = ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="256g", disk="256g", replicas=1)
    return sft_step(spec, resources)


if __name__ == "__main__":
    StepRunner().run([build(version=os.environ.get("AUGDK_SFT_VERSION", "dev")).lower()])
