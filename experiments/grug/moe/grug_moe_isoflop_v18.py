# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Isoflop sweep at 1e18 FLOPs across d=512/768/1024 with the refit LR heuristic.

3 runs, all at the same compute budget. Model size, batch size, and step count
all come from ``build_from_heuristic`` (issue #5951's corrected coefficients).

Submit on us-east5-a, interactive priority, v5p-8:

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.grug_moe_isoflop_v18
"""


from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic_v1 import build_from_heuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_BUDGET: float = 1e18
_TPU: str = "v5p-8"
_RUN_SUFFIX: str = "v1"
_GROUP_NAME: str = "grug-moe-isoflop-v18"
_WARMUP_FRACTION: float = 0.01
_HIDDEN_DIMS: tuple[int, ...] = (512, 768, 1024)


def _build_step(hidden_dim: int) -> ExecutorStep:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(budget=_BUDGET, hidden_dim=hidden_dim)
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

    run_id = f"grug-moe-isoflop-v18-d{hidden_dim}-{_RUN_SUFFIX}"
    step_name = f"grug/grug_moe_isoflop_v18/{run_id}"

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
                    "grug_moe_isoflop_v18",
                    "newlr",
                    f"d{hidden_dim}",
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
    steps = [_build_step(d) for d in _HIDDEN_DIMS]
    executor_main(
        steps=steps,
        description=(
            f"Isoflop sweep at {_BUDGET:.1e} FLOPs across d={_HIDDEN_DIMS}, "
            f"TPU={_TPU}, refit LR heuristic (#5951), no permanent step-interval "
            "checkpoints, 1pct-noclip schedule (warmup=1%, max_grad_norm=None)."
        ),
    )
