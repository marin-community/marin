# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""May MoE Recipe baseline at d768 / 1.70e18 (gate-1 reference).

Mirrors ``direct_launch.py`` (which targets d512 / 2.19e17) so we have an
apples-to-apples serial baseline at the larger gate-1 scale. Used as the
comparison point for variant ablations like
``parallel_attn_mlp_sweep.py``.

Usage::

    .venv/bin/iris --cluster=marin job run \\
      --no-wait --zone us-east5-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.direct_launch_d768
"""

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import this_output_path, versioned

from experiments.grug.moe.direct_launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeDirectLaunchConfig,
    _resolve_run_id,
    train_grug_moe,
)
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_HIDDEN_DIM: int = 768
_BUDGET: float = 1.70e18
_TARGET_STEPS: int = 2**14
_TPU: str = "v5p-8"
_SUFFIX: str = "v1"


def _build_launch() -> GrugMoeDirectLaunchConfig:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_TARGET_STEPS,
    )
    optimizer = GrugMoeMuonHConfig(
        learning_rate=base_optimizer.learning_rate,
        adam_lr=base_optimizer.adam_lr,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=0.01,
        beta1=base_optimizer.beta1,
        beta2=base_optimizer.beta2,
        epsilon=base_optimizer.epsilon,
        max_grad_norm=None,
        lr_schedule=base_optimizer.lr_schedule,
        decay=base_optimizer.decay,
    )
    budget_tag = f"{_BUDGET:.2e}".replace("+", "")
    run_id = _resolve_run_id(f"grug-moe-direct-d{_HIDDEN_DIM}-{budget_tag}-{_SUFFIX}")
    return GrugMoeDirectLaunchConfig(
        model=versioned(model),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=run_id,
        resources=ResourceConfig.with_tpu(_TPU),
        steps=versioned(num_steps),
        batch_size=versioned(batch_size),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            entity="marin-community",
            project="marin_moe",
            tags=["moe", "moe_direct", "may_recipe", f"d{_HIDDEN_DIM}", "baseline"],
            group="grug-moe-direct",
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
        checkpoint_keep_every=1000,
    )


if __name__ == "__main__":
    launch = _build_launch()
    job_id = train_grug_moe(
        name=f"grug/moe-direct-d{_HIDDEN_DIM}-{_SUFFIX}",
        launch=launch,
        wait=True,
    )
    print(f"Training job finished: {job_id}")
