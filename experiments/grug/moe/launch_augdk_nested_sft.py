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
    FIXED25 = "fixed25"


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


def build(*, version: str = "dev") -> ArtifactStep[LevanterCheckpoint]:
    """Build one matched SFT stage."""
    arm = SFTArm(_required_env("AUGDK_SFT_ARM"))
    stage = SFTStage(_required_env("AUGDK_SFT_STAGE"))
    init_from = _required_env("AUGDK_SFT_INIT_FROM")
    explicit_steps = int(os.environ.get("AUGDK_SFT_STEPS", stage.steps))
    if explicit_steps <= 0:
        raise ValueError("AUGDK_SFT_STEPS must be positive")

    model = build_scale_model()
    if arm is SFTArm.FIXED25:
        if model.nested_expert_counts != (128, 16):
            raise ValueError("fixed25 SFT requires SCALE_NESTED_COUNTS=128,16")
        routing = os.environ.get("AUGDK_SFT_ROUTING", "nested")
        if routing == "full":
            model = dataclasses.replace(model, nested_batch_fraction=0.0)
        elif routing == "nested" and model.nested_batch_fraction != 0.25:
            raise ValueError("nested fixed25 SFT requires SCALE_NESTED_FRACTION=0.25")
        elif routing != "nested":
            raise ValueError("AUGDK_SFT_ROUTING must be 'nested' or 'full'")
    elif model.nested_expert_counts:
        raise ValueError("E256 SFT must not configure nested expert counts")

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
