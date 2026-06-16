# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Isoflop d=1280 cell at 3e18 FLOPs on v5p-32.

Companion to ``grug_moe_isoflop_v3e18.py``: same group, same budget, but the
d=1280 cell needs more memory than v5p-8 has comfortable headroom for at this
scale. To fit on v5p-32 with reasonable throughput we double the batch size
from the heuristic-derived 32 -> 64 and halve the step count to keep total
tokens (and hence compute budget) unchanged.

Submit on us-east5-a, interactive priority, v5p-32:

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.grug_moe_isoflop_v3e18_d1280
"""

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic_adamh import MoeAdamHHeuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_BUDGET: float = 3e18
_HIDDEN_DIM: int = 1280
_SEQ_LEN: int = 4096
_TPU: str = "v5p-32"
_BATCH_SIZE: int = 64  # doubled from heuristic-derived 32
_NUM_STEPS: int = 4965  # halved from heuristic-derived 9929 to preserve tokens
_RUN_SUFFIX: str = "v1"
_GROUP_NAME: str = "grug-moe-isoflop-v3e18"
_WARMUP_FRACTION: float = 0.01


def _build_step() -> ExecutorStep:
    h = MoeAdamHHeuristic()
    model = h.build_model_config(_HIDDEN_DIM, seq_len=_SEQ_LEN)
    tokens = float(_NUM_STEPS * _BATCH_SIZE * _SEQ_LEN)
    base_optimizer = h.build_optimizer_config(_BATCH_SIZE, tokens, _HIDDEN_DIM, seq_len=_SEQ_LEN)

    optimizer = GrugMoeMuonHConfig(
        learning_rate=base_optimizer.learning_rate,
        adam_lr=base_optimizer.adam_lr,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=_WARMUP_FRACTION,
        beta1=base_optimizer.beta1,
        beta2=base_optimizer.beta2,
        epsilon=base_optimizer.epsilon,
        max_grad_norm=None,
        lr_schedule=base_optimizer.lr_schedule,
        decay=base_optimizer.decay,
    )

    run_id = f"grug-moe-isoflop-v3e18-d{_HIDDEN_DIM}-{_RUN_SUFFIX}"
    step_name = f"grug/grug_moe_isoflop_v3e18/{run_id}"

    return ExecutorStep(
        name=step_name,
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu(_TPU)),
            steps=versioned(_NUM_STEPS),
            batch_size=versioned(_BATCH_SIZE),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="marin_moe",
                tags=[
                    "moe",
                    "grug_moe_isoflop_v3e18",
                    "newlr",
                    f"d{_HIDDEN_DIM}",
                ],
                group=_GROUP_NAME,
                name=None,
            ),
            optimizer=versioned(optimizer),
            grug_trainer=versioned(
                GrugTrainerConfig(
                    z_loss_weight=0.0,
                    ema_beta=None,
                    log_every=1,
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
            checkpoint_keep_every=None,
        ),
    )


if __name__ == "__main__":
    executor_main(
        steps=[_build_step()],
        description=(
            f"Isoflop d=1280 at {_BUDGET:.1e} FLOPs on {_TPU}: bs={_BATCH_SIZE} "
            f"(doubled from heuristic 32), steps={_NUM_STEPS} (halved from 9929 to "
            "preserve total tokens). Refit LR heuristic (#5951), no permanent "
            "step-interval checkpoints, 1pct-noclip schedule."
        ),
    )
