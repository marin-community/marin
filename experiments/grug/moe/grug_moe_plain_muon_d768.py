# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single d=768 run at 3e18 FLOPs with plain Muon (no hyperball) on the muonh group.

Companion to ``grug_moe_lmhead_adam_d768.py``: same scale, same LR heuristic
— only the ``muonh`` group is replaced with plain Muon via
``GrugMoeMuonHConfig.use_muon_h=False``. The Newton-Schulz-orthogonalised
direction is scaled by ``-learning_rate`` directly instead of going through
the Frobenius hyperball step.

Submit on us-east5-a, interactive priority, v5p-32:

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.grug_moe_plain_muon_d768
"""

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

_BUDGET: float = 3e18
_HIDDEN_DIM: int = 768
_TPU: str = "v5p-32"
_RUN_SUFFIX: str = "v1"
_GROUP_NAME: str = "grug-moe-plain-muon"
_WARMUP_FRACTION: float = 0.01


def _build_step() -> ExecutorStep:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(budget=_BUDGET, hidden_dim=_HIDDEN_DIM)

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
        use_muon_h=False,
    )

    run_id = f"grug-moe-plain-muon-d{_HIDDEN_DIM}-3e18-{_RUN_SUFFIX}"
    step_name = f"grug/grug_moe_plain_muon/{run_id}"

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
                    "grug_moe_plain_muon",
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
            f"d={_HIDDEN_DIM} at {_BUDGET:.1e} FLOPs on {_TPU} with use_muon_h=False "
            "(plain Muon on the muonh group; hyperball step skipped). Compares against "
            "grug-moe-isoflop-v3e18-d768-v1 baseline (MuonH). Refit LR heuristic "
            "(#5951), no permanent step-interval checkpoints, 1pct-noclip schedule."
        ),
    )
