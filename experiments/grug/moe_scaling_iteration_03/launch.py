# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug MoE iteration 03 — single trial run for fast iteration on model changes.

Reuses model and train loop from iteration_02. Default config: d=512, 1e18 budget
(~5.8k steps, ~1.5B tokens on v5p-8) — runs in roughly 1 hour.
"""

import dataclasses
import math
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.moe_scaling_iteration_03.model import GrugModelConfig
from experiments.grug.moe_scaling_iteration_03.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.pretraining_datasets import nemotron_mix_block_shuffle


@dataclass(frozen=True)
class GrugMoeLaunchConfig:
    """Last-mile run config for the MoE grug template."""

    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str  # jmp policy string, e.g. "params=float32,compute=bfloat16,output=bfloat16".
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)


SEQ_LEN: int = 4096
VOCAB_SIZE: int = 128_256

NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix_block_shuffle,
    default_validation_sets(tokenizer=nemotron_mix_block_shuffle.tokenizer),
)


def _resolve_run_id(default_run_id: str) -> str:
    run_id = os.environ.get("GRUG_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def run_grug_moe_trial(config: GrugMoeLaunchConfig) -> None:
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=100, perfetto_link=False),
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": 4}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[{"every": 1000}],
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
    run_grug(run_config)


# ============================================================
# Default trial: d=768, 1e18 budget (~1 hour on v5p-8)
# ============================================================

HIDDEN_DIM = 512
NUM_HEADS = HIDDEN_DIM // 128  # 4
NUM_LAYERS = 6

TRIAL_MODEL = GrugModelConfig(
    vocab_size=VOCAB_SIZE,
    hidden_dim=HIDDEN_DIM,
    intermediate_dim=HIDDEN_DIM // 2,
    shared_expert_intermediate_dim=HIDDEN_DIM,
    num_experts=64,
    num_experts_per_token=4,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    num_kv_heads=NUM_HEADS,
    max_seq_len=SEQ_LEN,
    num_dense_layers=2,
    dense_intermediate_dim=3 * HIDDEN_DIM,
    load_balancing_loss_coef=None,
    sliding_window=4096,
    initializer_std=0.5 / math.sqrt(HIDDEN_DIM),
    qk_mult=1.3,
)

# doubled from iter_02's bs=32 → 64, halved steps to keep same 1e18 token budget
# effective_bs = 64, lr = min(0.01, 0.33*sqrt(64)/512) = 0.00516
# beta2 = max(0.95, 0.98^(64/128)) = 0.99
BATCH_SIZE = 64
TRAIN_STEPS = 5818
LR = 0.00516
BETA2 = 0.99

KP = 0.05
KI = 0.002
KD = 0.02
CLAMP = 50.0
RUN_NAME = f"t1pid-kp{KP}-ki{KI}-kd{KD}-cl{CLAMP}-nolbl"
RESOLVED_RUN_ID = _resolve_run_id(RUN_NAME)

grug_moe_trial = ExecutorStep(
    name=f"grug/{RUN_NAME}",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(TRIAL_MODEL),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(TRAIN_STEPS),
        batch_size=versioned(BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "moe-core", "iter03", "pid", "d512", "1e18"],
            group="moe-iter03",
            name=None,
        ),
        optimizer=versioned(
            AdamConfig(
                learning_rate=LR,
                beta1=0.96,
                beta2=BETA2,
                epsilon=1e-15,
                weight_decay=0.1,
                lr_schedule="linear",
                decay=0.2,
                min_lr_ratio=0.0,
                warmup=0.1,
                max_grad_norm=1,
            )
        ),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
                pid_kp=KP,
                pid_ki=KI,
                pid_kd=KD,
                pid_integral_clamp=CLAMP,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=512,
                steps_per_eval=1000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[grug_moe_trial],
        description="MoE iteration 03 PID sweep: d=512, 1e18 FLOPs (~1 hour on v5p-8).",
    )
