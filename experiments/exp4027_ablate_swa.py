# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ablate sliding-window attention in the MoE good-10T gate.

Runs two ~1e19-FLOP MoE models at identical config except one uses
sliding_window=4096 and the other uses full causal attention (the default).

Tracking issue: https://github.com/marin-community/marin/issues/4027
Parent gate: https://github.com/marin-community/marin/issues/4013
"""

from dataclasses import replace

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.moe.launch import run_grug_moe, GrugMoeLaunchConfig
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.pretraining_datasets import nemotron_mix_block_shuffle

# ---------------------------------------------------------------------------
# Model: ~750M-activated MoE targeting ~1e19 training FLOPs.
#
# With 8 experts, top-2 routing, and a shared expert the activated parameter
# count is roughly:
#   embed + 24 * (attn + 2*FFN_active + shared_FFN) + lm_head
# We pick hidden_dim=1024, intermediate_dim=2816, 24 layers which lands near
# 750M activated params. At batch_size=512, seq_len=4096 the token throughput
# is ~2M tokens/step. ~1e19 FLOPs => ~4800 steps.
# ---------------------------------------------------------------------------

SLIDING_WINDOW_SIZE = 4096

BASE_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=1024,
    intermediate_dim=2816,
    shared_expert_intermediate_dim=2816,
    num_experts=8,
    num_experts_per_token=2,
    num_layers=24,
    num_heads=16,
    num_kv_heads=16,
    max_seq_len=4096,
)

SWA_MODEL = replace(BASE_MODEL, sliding_window=SLIDING_WINDOW_SIZE)

NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix_block_shuffle,
    default_validation_sets(tokenizer=nemotron_mix_block_shuffle.tokenizer),
)

TRAIN_STEPS = 4800
BATCH_SIZE = 512

OPTIMIZER = AdamConfig(
    learning_rate=3e-3,
    weight_decay=0.1,
    lr_schedule="cosine",
    decay=0.2,
    min_lr_ratio=0.1,
    warmup=1000,
)

GRUG_TRAINER = GrugTrainerConfig(
    z_loss_weight=1e-4,
    ema_beta=None,
    log_every=1,
)

EVAL_CONFIG = GrugEvalConfig(
    eval_batch_size=512,
    steps_per_eval=1000,
    max_eval_batches=8,
    eval_current=True,
    eval_ema=False,
)


def _make_launch_config(
    model: GrugModelConfig,
    run_id: str,
) -> GrugMoeLaunchConfig:
    return GrugMoeLaunchConfig(
        model=versioned(model),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=run_id,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(TRAIN_STEPS),
        batch_size=versioned(BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["moe", "good-10t-gate", "swa-ablation"],
            group="exp4027-ablate-swa",
            name=None,
        ),
        optimizer=versioned(OPTIMIZER),
        grug_trainer=versioned(GRUG_TRAINER),
        eval=versioned(EVAL_CONFIG),
    )


baseline_no_swa = ExecutorStep(
    name="exp4027/moe-no-swa",
    fn=run_grug_moe,
    config=_make_launch_config(BASE_MODEL, "exp4027-moe-no-swa"),
)

baseline_swa = ExecutorStep(
    name="exp4027/moe-swa-4096",
    fn=run_grug_moe,
    config=_make_launch_config(SWA_MODEL, "exp4027-moe-swa-4096"),
)

if __name__ == "__main__":
    executor_main(
        steps=[baseline_no_swa, baseline_swa],
        description="Ablate sliding-window attention (window=4096 vs full) on ~750M MoE at ~1e19 FLOPs. Fixes #4027.",
    )
