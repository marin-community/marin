# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single d=1024 run at 1e19 FLOPs with PKO disabled (partial rope unchanged).

Companion to the 1e19 isoflop sweep: same budget, same model size, same LR
heuristic — only the per-layer ``use_pko`` (K-half shift + doc-start zero) is
turned off via ``GrugModelConfig.disable_pko=True``. Sliding-window
short/long pattern and partial rope are unchanged.

Submit on us-central1, interactive priority, v5p-32:

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --zone us-central1-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.grug_moe_nopko_d1024
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic_adamh import build_from_heuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_BUDGET: float = 1e19
_HIDDEN_DIM: int = 1024
_TPU: str = "v5p-32"
_RUN_SUFFIX: str = "v1"
_GROUP_NAME: str = "grug-moe-nopko"
_WARMUP_FRACTION: float = 0.01


def _build_step() -> ExecutorStep:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(budget=_BUDGET, hidden_dim=_HIDDEN_DIM)
    model = dataclasses.replace(model, disable_pko=True)

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

    run_id = f"grug-moe-nopko-d{_HIDDEN_DIM}-1e19-{_RUN_SUFFIX}"
    step_name = f"grug/grug_moe_nopko/{run_id}"

    return ExecutorStep(
        name=step_name,
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu(_TPU)),
            steps=versioned(num_steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="marin_moe",
                tags=[
                    "moe",
                    "grug_moe_nopko",
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
            f"d={_HIDDEN_DIM} at {_BUDGET:.1e} FLOPs on {_TPU} with disable_pko=True "
            "(K-half shift disabled on all layers; sliding window + partial rope "
            "unchanged). Refit LR heuristic (#5951), no permanent step-interval "
            "checkpoints, 1pct-noclip schedule."
        ),
    )
