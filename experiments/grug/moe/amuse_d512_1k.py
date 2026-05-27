# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""1k-step screening run for AMUSE at d512.

Drives the sweep without paying the full 6302-step training cost. Keeps the
heuristic-derived ``batch_size`` (same data per step as the full run) but
truncates to 1000 steps so we can compare against a 1000-step MuonH baseline
(`muonh_d512_1k.py`) before committing to the full sweep point.

Submit:

    .venv/bin/iris --config lib/iris/config/marin.yaml job run --no-wait \\
      --preemptible \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.amuse_d512_1k
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
from experiments.grug.moe.optimizer import GrugMoeAmuseConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_HIDDEN_DIM: int = 512
_BUDGET: float = 2.19e17
_TARGET_STEPS: int = 2**14
_TPU: str = "v5p-8"
_NUM_STEPS: int = 1000  # truncated screening budget
_RUN_SUFFIX: str = "amuse-1k-lr1.5x-b1-0.6-rho-0.8-T0-80"
# Best sweep point so far (v8 = c4 X 3.74). T_0 scaled to ~8% of 1000 steps
# (v8 used T_0=500 ≈ 8% of 6302).
_LR_MULTIPLIER: float = 1.5
_AMUSE_BETA1: float = 0.6
_AMUSE_RHO: float = 0.8
_AMUSE_T0: int = 80


def _build_launch() -> tuple[str, GrugMoeDirectLaunchConfig]:
    model, base_optimizer, batch_size, _heuristic_num_steps = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_TARGET_STEPS,
    )

    optimizer = GrugMoeAmuseConfig(
        learning_rate=base_optimizer.learning_rate * _LR_MULTIPLIER,
        adam_lr=base_optimizer.adam_lr * _LR_MULTIPLIER,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=0.01,  # ~10 steps at total=1000
        weight_decay=base_optimizer.weight_decay,
        amuse_beta1=_AMUSE_BETA1,
        amuse_rho=_AMUSE_RHO,
        amuse_warmup_steps=_AMUSE_T0,
        muon_momentum=0.95,
        backend_steps=5,
        coefficient_type="quintic",
        beta2=0.95,
        epsilon=base_optimizer.epsilon,
        max_grad_norm=None,
        lr_schedule="constant",
        decay=None,
    )

    run_id = _resolve_run_id(f"amuse-d{_HIDDEN_DIM}-{_BUDGET:.2e}-{_RUN_SUFFIX}".replace("+", ""))
    name = f"grug/amuse-d{_HIDDEN_DIM}-{_RUN_SUFFIX}"

    launch = GrugMoeDirectLaunchConfig(
        model=versioned(model),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=run_id,
        resources=ResourceConfig.with_tpu(_TPU),
        steps=versioned(_NUM_STEPS),
        batch_size=versioned(batch_size),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            entity="marin-community",
            project="marin_moe",
            tags=["moe", "amuse", "amuse_d512_1k", f"d{_HIDDEN_DIM}", "screen-1k"],
            group="amuse-d512-1k-screen",
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
                # More frequent evals in the 1k screening run so we have a
                # loss trajectory, not just the final point.
                steps_per_eval=200,
                max_eval_batches=8,
                eval_current=True,  # Y eval
                eval_ema=True,  # X eval
            )
        ),
        checkpoint_keep_every=1000,
    )
    return name, launch


if __name__ == "__main__":
    name, launch = _build_launch()
    print(f"Submitting AMUSE d512 1k-screen run (truncated to {_NUM_STEPS} steps)")
    job_id = train_grug_moe(name=name, launch=launch, wait=True)
    print(f"Training job finished: {job_id}")
