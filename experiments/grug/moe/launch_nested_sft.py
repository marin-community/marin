# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Short matched WildChat SFT transfer check for the nested-MoE proxy."""

import dataclasses
import os
from enum import StrEnum

import click
from fray.cluster import ResourceConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.grug.moe.heuristic import MoeHeuristic, build_from_heuristic
from experiments.grug.moe.launch_nested_experts import NestedArm, _arm_model
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig
from experiments.grug.moe.sft_launch import GrugModel
from experiments.llama import llama3_instruct_trainable_chat_template, llama3_tokenizer
from experiments.sft.launcher import DatasetSpec, SFTSpec, sft_step

_BUDGET = 3.46e19
_TARGET_STEPS = 8192
_HIDDEN_DIM = 768
_SEQUENCE_LENGTH = 2048
_CAPACITY_FACTOR = 1.25
_EXPERT_PARALLEL = 64
_NODES = 16
_GPUS_PER_NODE = 4
_BATCH = 256
_SFT_STEPS = 8
_WILDCHAT = DatasetSpec(
    slug="wildchat_386k",
    hf_dataset_id="nyu-dice-lab/wildchat50m-rewild-sft-385700",
    revision="46a5bb5",
    adapter_kwargs={"conversation_column": "conversation"},
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
    LARGE = "large"
    SMALL = "small"
    NESTED_FULL = "nested_full"
    BREAKOUT = "breakout"


def _required_env(key: str) -> str:
    value = os.environ.get(key)
    if value is None:
        raise ValueError(f"{key} must be set")
    return value


def _model(arm: SFTArm):
    base_model, _, _, _ = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        heuristic=MoeHeuristic(),
        target_steps=_TARGET_STEPS,
        seq_len=_SEQUENCE_LENGTH,
    )
    base_model = dataclasses.replace(base_model, capacity_factor=_CAPACITY_FACTOR)
    if arm is SFTArm.LARGE:
        model = _arm_model(base_model, NestedArm.LARGE)
    elif arm in (SFTArm.SMALL, SFTArm.BREAKOUT):
        model = _arm_model(base_model, NestedArm.SMALL)
    else:
        model = dataclasses.replace(
            _arm_model(base_model, NestedArm.NESTED_25),
            nested_batch_fraction=0.0,
        )
    return dataclasses.replace(model, attention_implementation="reference")


def build(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Build one matched eight-step completion-masked SFT arm."""
    arm = SFTArm(_required_env("NESTED_SFT_ARM"))
    init_from = _required_env("NESTED_SFT_INIT_FROM")
    run_id = f"nest-moe-sft-{arm.value}-d{_HIDDEN_DIM}-s{_SEQUENCE_LENGTH}"
    run_suffix = os.environ.get("NESTED_SFT_RUN_SUFFIX")
    if run_suffix:
        run_id = f"{run_id}-{run_suffix}"
    step_name = f"experiments/nested-moe-sft/{run_id}"
    version = resolve_version(step_name, version)
    model_source = GrugModel(
        model=_model(arm),
        tokenizer_path=llama3_tokenizer,
        init_from=init_from,
        expert_parallel=_EXPERT_PARALLEL,
        replica_axis=1,
        per_device_parallelism=-1,
        mp="params=float32,compute=float32,output=float32",
        z_loss_weight=1e-4,
        checkpoint_keep=[{"every": _SFT_STEPS}],
        wandb_tags=["moe", "nested-moe", "sft", arm.value, "gb200"],
        wandb_group="NEST-MOE-20260727-SFT",
    )
    spec = SFTSpec(
        name=user_namespaced_name(step_name, version),
        version=version,
        model=model_source,
        chat_template=llama3_instruct_trainable_chat_template,
        datasets=[_WILDCHAT],
        optimizer=_OPTIMIZER,
        seq_len=_SEQUENCE_LENGTH,
        batch_size=_BATCH,
        num_train_steps=_SFT_STEPS,
        wandb_project="marin_moe_sft",
    )
    resources = ResourceConfig.with_gpu(
        "GB200",
        count=_GPUS_PER_NODE,
        cpu=64,
        ram="512g",
        disk="512g",
        replicas=_NODES,
    )
    return sft_step(spec, resources)


@click.command()
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    return build()


if __name__ == "__main__":
    main()
