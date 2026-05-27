# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""1k-step MuonH baseline at d512.

Companion to ``amuse_d512_1k.py`` — same model / data / batch_size /
total_steps so the two are directly comparable. MuonH gets the standard
heuristic LR with cosine decay across the 1000 steps (the May Recipe).

Submit:

    .venv/bin/iris --config lib/iris/config/marin.yaml job run --no-wait \\
      --preemptible \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_d512_1k
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

_HIDDEN_DIM: int = 512
_BUDGET: float = 2.19e17
_TARGET_STEPS: int = 2**14
_TPU: str = "v5p-8"
_NUM_STEPS: int = 1000
_RUN_SUFFIX: str = "muonh-1k-baseline"


def _build_launch() -> tuple[str, GrugMoeDirectLaunchConfig]:
    model, base_optimizer, batch_size, _heuristic_num_steps = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_TARGET_STEPS,
    )

    # May Recipe baseline: MuonH on matrices + AdamH on lm_head + plain Adam on
    # router/embed/1-D leaves. Heuristic LR, warmup=0.01, cosine decay across
    # the 1000 step budget. Same setup as direct_launch._baseline_optimizer
    # except for the explicit step count override below.
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

    run_id = _resolve_run_id(f"muonh-d{_HIDDEN_DIM}-{_BUDGET:.2e}-{_RUN_SUFFIX}".replace("+", ""))
    name = f"grug/muonh-d{_HIDDEN_DIM}-{_RUN_SUFFIX}"

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
            tags=["moe", "muonh", "muonh_d512_1k", f"d{_HIDDEN_DIM}", "screen-1k"],
            group="muonh-d512-1k-screen",
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
                steps_per_eval=200,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,  # MuonH has no X sequence
            )
        ),
        checkpoint_keep_every=1000,
    )
    return name, launch


if __name__ == "__main__":
    name, launch = _build_launch()
    print(f"Submitting MuonH d512 1k-baseline run ({_NUM_STEPS} steps)")
    job_id = train_grug_moe(name=name, launch=launch, wait=True)
    print(f"Training job finished: {job_id}")
