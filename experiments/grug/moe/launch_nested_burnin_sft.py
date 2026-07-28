# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Matched two-stage SFT for the NEST-BURN-001 endpoint checkpoints.

Launch WildChat first, then launch the thinking stage with the WildChat output
as ``BURNIN_SFT_INIT_FROM``. Each invocation trains one packed epoch.

Required environment:

    BURNIN_SFT_ARM       e256 | fixed25
    BURNIN_SFT_STAGE     wildchat | thinking
    BURNIN_SFT_INIT_FROM native checkpoint directory

Optional environment:

    BURNIN_SFT_RUN_SUFFIX
    BURNIN_SFT_STEPS      explicit smoke length instead of one epoch
"""

import dataclasses
import os
from enum import StrEnum

from fray.cluster import ResourceConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import experiment_main
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch_nested_experts import NestedArm, _arm_model
from experiments.grug.moe.model import NestedSubsetSchedule, RouterBalanceMode
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig
from experiments.grug.moe.sft_launch import GrugModel
from experiments.marin_tokenizer import MARIN_CHAT_TEMPLATE, marin_tokenizer
from experiments.sft.launcher import DatasetSpec, SFTSpec, sft_step

_EXPERIMENT_ID = "NEST-BURN-001-SFT"
_COMPUTE_BUDGET = 4.14e18
_HIDDEN_DIM = 768
_PRETRAIN_SEQUENCE_LENGTH = 8192
_HEURISTIC_TARGET_STEPS = 2**15
_CAPACITY_FACTOR = 1.25
_SFT_SEQUENCE_LENGTH = 8192
_SFT_BATCH_SIZE = 32
_NODES = 4
_GPUS_PER_NODE = 4

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


class BurninSFTArm(StrEnum):
    E256 = "e256"
    FIXED_25 = "fixed25"


class BurninSFTStage(StrEnum):
    WILDCHAT = "wildchat"
    THINKING = "thinking"

    @property
    def dataset(self) -> DatasetSpec:
        if self is BurninSFTStage.WILDCHAT:
            return _WILDCHAT
        return _THINKING


def _required_env(key: str) -> str:
    value = os.environ.get(key)
    if value is None:
        raise ValueError(f"{key} must be set")
    return value


def _model(arm: BurninSFTArm):
    base_model, _, _, _ = build_from_heuristic(
        budget=_COMPUTE_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_HEURISTIC_TARGET_STEPS,
        seq_len=_PRETRAIN_SEQUENCE_LENGTH,
    )
    base_model = dataclasses.replace(
        base_model,
        capacity_factor=_CAPACITY_FACTOR,
        router_balance_mode=RouterBalanceMode.ELIGIBILITY_QB,
    )
    if arm is BurninSFTArm.E256:
        return _arm_model(base_model, NestedArm.LARGE, "reference")

    fixed = _arm_model(base_model, NestedArm.FIXED_25, "reference")
    return dataclasses.replace(
        fixed,
        nested_subset_schedule=NestedSubsetSchedule.PREFIX,
        nested_batch_fraction=0.0,
    )


def build(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Build one full-model SFT arm and stage."""
    arm = BurninSFTArm(_required_env("BURNIN_SFT_ARM"))
    stage = BurninSFTStage(_required_env("BURNIN_SFT_STAGE"))
    init_from = _required_env("BURNIN_SFT_INIT_FROM")
    explicit_steps_value = os.environ.get("BURNIN_SFT_STEPS")
    explicit_steps = int(explicit_steps_value) if explicit_steps_value is not None else None
    if explicit_steps is not None and explicit_steps <= 0:
        raise ValueError("BURNIN_SFT_STEPS must be positive")

    run_id = f"{_EXPERIMENT_ID.lower()}-{arm.value}-{stage.value}-d{_HIDDEN_DIM}-s{_SFT_SEQUENCE_LENGTH}"
    run_suffix = os.environ.get("BURNIN_SFT_RUN_SUFFIX")
    if run_suffix:
        run_id = f"{run_id}-{run_suffix}"
    step_name = f"experiments/nested-moe-burnin-sft/{run_id}"
    version = resolve_version(step_name, version)

    model_source = GrugModel(
        model=_model(arm),
        tokenizer_path=marin_tokenizer,
        init_from_path=init_from,
        expert_parallel=1,
        replica_axis=1,
        per_device_parallelism=2,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        z_loss_weight=1e-4,
        checkpoint_keep=[{"every": 1_000}],
        wandb_tags=["moe", "nested-moe", "burnin", "sft", arm.value, stage.value, "gb200"],
        wandb_group=_EXPERIMENT_ID,
    )
    spec = SFTSpec(
        name=user_namespaced_name(step_name, version),
        version=version,
        model=model_source,
        chat_template=MARIN_CHAT_TEMPLATE,
        datasets=[stage.dataset],
        optimizer=_OPTIMIZER,
        seq_len=_SFT_SEQUENCE_LENGTH,
        batch_size=_SFT_BATCH_SIZE,
        num_train_steps=explicit_steps,
        num_train_epochs=None if explicit_steps is not None else 1,
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


if __name__ == "__main__":
    experiment_main(build)()
