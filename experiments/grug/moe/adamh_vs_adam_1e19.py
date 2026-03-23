# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare AdamH vs Adam on the grug MoE architecture at ~1e19 FLOPs.

Launches two runs on the same d=1024 model (E=8, K=2, shared expert, 13 layers):
one with standard Adam and one with GrugAdamHConfig. Both use the same data,
batch size, and training budget so the only variable is the optimizer.

Part of #4024 / #4013.
"""

import dataclasses
import math
import os
from dataclasses import field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.optim import AdamConfig, GrugAdamHConfig, OptimizerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.mesh import MeshConfig

from experiments.defaults import default_validation_sets
from experiments.grug.moe.launch import GrugMoeLaunchConfig, run_grug_moe
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.pretraining_datasets import nemotron_mix_block_shuffle
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

# ---------- constants ----------
SEQ_LEN = 4096
VOCAB_SIZE = 128_256
BUDGET = 1e19

NEMOTRON_MIX = add_validation_sets_to_mixture(
    nemotron_mix_block_shuffle,
    default_validation_sets(tokenizer=nemotron_mix_block_shuffle.tokenizer),
)

# ---------- model ----------
HIDDEN_DIM = 1024
NUM_HEADS = HIDDEN_DIM // 128  # 8
NUM_LAYERS = 13
INTERMEDIATE_DIM = HIDDEN_DIM * 3  # 3072 (SiGLU expert MLP)

MODEL = GrugModelConfig(
    vocab_size=VOCAB_SIZE,
    hidden_dim=HIDDEN_DIM,
    intermediate_dim=INTERMEDIATE_DIM,
    shared_expert_intermediate_dim=INTERMEDIATE_DIM,
    num_experts=8,
    num_experts_per_token=2,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    num_kv_heads=NUM_HEADS,
    max_seq_len=SEQ_LEN,
)

# ---------- training budget ----------
_FPT = lm_flops_per_token(
    hidden_dim=HIDDEN_DIM,
    intermediate_dim=INTERMEDIATE_DIM,
    num_layers=NUM_LAYERS,
    num_kv_heads=NUM_HEADS,
    num_heads=NUM_HEADS,
    seq_len=SEQ_LEN,
    vocab_size=VOCAB_SIZE,
    glu=True,
    num_experts=8,
    num_shared_experts=1,
    num_experts_per_tok=2,
    shared_intermediate_dim=INTERMEDIATE_DIM,
)
_TOKENS = BUDGET / (3 * _FPT)
BATCH_SIZE = 64
TRAIN_STEPS = round(_TOKENS / (BATCH_SIZE * SEQ_LEN))

# ---------- shared hyperparameters ----------
_EFFECTIVE_BS = BATCH_SIZE * SEQ_LEN / 4096
_LR = min(0.01, (0.33 * math.sqrt(_EFFECTIVE_BS)) / HIDDEN_DIM)
_BETA2 = max(0.95, 0.98 ** (_EFFECTIVE_BS / 128))

# ---------- optimizers ----------
ADAM_OPTIMIZER = AdamConfig(
    learning_rate=_LR,
    weight_decay=0.1,
    beta1=0.9,
    beta2=_BETA2,
    epsilon=1e-8,
    lr_schedule="linear",
    decay=0.2,
    min_lr_ratio=0.0,
    warmup=0.1,
    max_grad_norm=1.0,
)

# AdamH LR heuristic: sqrt(lr * weight_decay) for the scale-invariant component,
# standard lr for the Adam component (embeddings, norms, routers).
_ADAMH_LR = math.sqrt(_LR * 0.1)
ADAMH_OPTIMIZER = GrugAdamHConfig(
    learning_rate=_ADAMH_LR,
    adam_lr=_LR,
    beta1=0.9,
    beta2=_BETA2,
    epsilon=1e-8,
    lr_schedule="linear",
    decay=0.2,
    min_lr_ratio=0.0,
    warmup=0.1,
    max_grad_norm=0.1,
    weight_decay=0.0,
)

GRUG_TRAINER = GrugTrainerConfig(
    z_loss_weight=1e-4,
    ema_beta=None,
    log_every=1,
)

EVAL = GrugEvalConfig(
    eval_batch_size=512,
    steps_per_eval=1000,
    max_eval_batches=8,
    eval_current=True,
    eval_ema=False,
)


def _resolve_run_id(base: str) -> str:
    run_id = os.environ.get("GRUG_RUN_ID", base)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _make_step(name: str, optimizer: OptimizerConfig, tags: list[str]) -> ExecutorStep:
    run_id = _resolve_run_id(name)
    return ExecutorStep(
        name=f"grug/{name}",
        fn=run_grug_moe,
        config=GrugMoeLaunchConfig(
            model=versioned(MODEL),
            data=NEMOTRON_MIX,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v5p-8")),
            steps=versioned(TRAIN_STEPS),
            batch_size=versioned(BATCH_SIZE),
            seed=versioned(42),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="marin",
                tags=["grug", "moe", "adamh-vs-adam", "1e19", *tags],
                group="moe-adamh-vs-adam-1e19",
                name=None,
            ),
            optimizer=versioned(optimizer),
            grug_trainer=versioned(GRUG_TRAINER),
            eval=versioned(EVAL),
        ),
    )


adam_step = _make_step(
    "moe-adam-1e19-d1024",
    ADAM_OPTIMIZER,
    ["adam", "baseline"],
)

adamh_step = _make_step(
    "moe-adamh-1e19-d1024",
    ADAMH_OPTIMIZER,
    ["adamh"],
)


if __name__ == "__main__":
    executor_main(
        steps=[adam_step, adamh_step],
        description="AdamH vs Adam on grug MoE at ~1e19 FLOPs (d=1024, E=8, K=2). Part of #4024.",
    )
