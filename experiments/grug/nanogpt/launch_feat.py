# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch all 3 nanogpt variants with grug features (XSA, attn gate, gated norm, QK gain, SwiGLU)."""

import dataclasses
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import DatasetComponent, LmDataConfig
from levanter.optim import OptimizerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.training.training import temporary_checkpoint_base_path

from experiments.grug.nanogpt.launch import NanoGPTOptimizerConfig, _adamh_optimizer_for_nanogpt
from experiments.grug.nanogpt.launch_adamh_ref import NanoGPTAdamHRefConfig
from experiments.grug.nanogpt.model import BATCH_SIZE, TRAIN_STEPS
from experiments.grug.nanogpt.model_grug import GrugNanoGPTConfig
from experiments.grug.nanogpt.train_grug import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig
from experiments.grug.nanogpt.train_grug import run_grug as run_grug_feat


def _fineweb_gpt2_data() -> LmDataConfig:
    from rigging.filesystem import marin_prefix

    base = os.path.join(marin_prefix(), "data", "fineweb10B-gpt2")
    return LmDataConfig(
        tokenizer="gpt2",
        block_cross_document_attention=False,
        auto_build_caches=False,
        components={
            "fineweb_train": DatasetComponent(cache_dir=base, split="train"),
            "fineweb_val": DatasetComponent(cache_dir=base, split="validation"),
        },
        train_weights={"fineweb_train": 1.0},
    )


@dataclass(frozen=True)
class FeatLaunchConfig:
    model: GrugNanoGPTConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str
    tracker: WandbConfig
    optimizer: OptimizerConfig
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)


def _resolve_tracker(tracker, run_id):
    return dataclasses.replace(tracker, name=run_id)


def run_feat_trial(config: FeatLaunchConfig) -> None:
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[{"every": 500}],
        ),
    )
    grug_trainer = dataclasses.replace(config.grug_trainer, trainer=trainer)
    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_grug_feat(run_config)


EVAL_CFG = GrugEvalConfig(
    eval_batch_size=BATCH_SIZE,
    steps_per_eval=125,
    max_eval_batches=20,
    eval_current=True,
    eval_ema=False,
)
TRAINER_CFG = GrugTrainerConfig(z_loss_weight=0.0, ema_beta=None, log_every=1)

# ---- Models ----
FEAT_MODEL_MUON = GrugNanoGPTConfig(zero_init_proj=True)
FEAT_MODEL_ADAMH = GrugNanoGPTConfig(zero_init_proj=False)

# ---- 1. Muon + features (3600 steps) ----

nanogpt_feat_muon = ExecutorStep(
    name="grug/nanogpt-feat-muon",
    fn=run_feat_trial,
    config=FeatLaunchConfig(
        model=versioned(FEAT_MODEL_MUON),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id="nanogpt-feat-muon",
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(TRAIN_STEPS),
        batch_size=versioned(BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="dial_moe", tags=["nanogpt", "feat", "muon"], group="nanogpt-feat", name="nanogpt-feat-muon"
        ),
        optimizer=versioned(NanoGPTOptimizerConfig()),
        grug_trainer=versioned(TRAINER_CFG),
        eval=versioned(EVAL_CFG),
    ),
)

# ---- 2. AdamH heuristic + features (3600 steps) ----

nanogpt_feat_adamh = ExecutorStep(
    name="grug/nanogpt-feat-adamh-debug",
    fn=run_feat_trial,
    config=FeatLaunchConfig(
        model=versioned(FEAT_MODEL_ADAMH),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id="nanogpt-feat-adamh-debug",
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(50),
        batch_size=versioned(BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="dial_moe",
            tags=["nanogpt", "feat", "adamh", "debug"],
            group="nanogpt-feat",
            name="nanogpt-feat-adamh-debug",
        ),
        optimizer=versioned(_adamh_optimizer_for_nanogpt()),
        grug_trainer=versioned(TRAINER_CFG),
        eval=versioned(EVAL_CFG),
    ),
)

# ---- 3. AdamH ref + features (4875 steps) ----

nanogpt_feat_adamh_ref = ExecutorStep(
    name="grug/nanogpt-feat-adamh-debug-ref",
    fn=run_feat_trial,
    config=FeatLaunchConfig(
        model=versioned(FEAT_MODEL_ADAMH),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id="nanogpt-feat-adamh-debug-ref",
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(50),
        batch_size=versioned(BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="dial_moe",
            tags=["nanogpt", "feat", "adamh-ref", "debug"],
            group="nanogpt-feat",
            name="nanogpt-feat-adamh-debug-ref",
        ),
        optimizer=versioned(NanoGPTAdamHRefConfig()),
        grug_trainer=versioned(TRAINER_CFG),
        eval=versioned(EVAL_CFG),
    ),
)


# ---- 4. AdamH heuristic + features, half batch (256 seqs x 7200 steps) ----

HALF_BATCH = BATCH_SIZE // 2  # 256
DOUBLE_STEPS = TRAIN_STEPS * 2  # 7200

nanogpt_feat_adamh_halfbatch = ExecutorStep(
    name="grug/nanogpt-feat-adamh-debug-halfbatch",
    fn=run_feat_trial,
    config=FeatLaunchConfig(
        model=versioned(FEAT_MODEL_ADAMH),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id="nanogpt-feat-adamh-debug-halfbatch",
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(DOUBLE_STEPS),
        batch_size=versioned(HALF_BATCH),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="dial_moe",
            tags=["nanogpt", "feat", "adamh", "halfbatch"],
            group="nanogpt-feat",
            name="nanogpt-feat-adamh-debug-halfbatch",
        ),
        optimizer=versioned(_adamh_optimizer_for_nanogpt(batch_size=HALF_BATCH, steps=DOUBLE_STEPS)),
        grug_trainer=versioned(TRAINER_CFG),
        eval=versioned(EVAL_CFG),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[nanogpt_feat_adamh, nanogpt_feat_adamh_ref],
        description="NanoGPT feat-adamh debug runs.",
    )
