# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delayed-gradient staleness sweep launcher for the grug MoE template.

Reuses the baseline grug-MoE training machinery (`run_grug_moe_trial`) but swaps
the optimizer for a delayed-gradient wrapper (see `delay_optim.py`). The arm is
selected entirely by environment variables so a single module can be submitted
many times to Iris with different staleness / corrector settings:

    GRUG_OPT        muon | adamh                 (default muon)
    GRUG_TAU        gradient delay in steps      (default 0)
    GRUG_CORRECTOR  none | dc_asgd | dc_asgd_ema (default none)
    GRUG_DC_LAMBDA  DC-ASGD strength             (default 1.0)
    GRUG_STEPS      train steps (short for fast iteration)  (default 3000)
    GRUG_SEED       seed                         (default 0)
    GRUG_HIDDEN     model hidden dim             (default 512)
    GRUG_BUDGET     compute budget for heuristic sizing/LR  (default 2.19e17)
    GRUG_TPU        TPU type to reserve          (default v6e-8)
    GRUG_GROUP      wandb group                  (default delay-pp-batch1)

Submit, e.g.:

    GRUG_OPT=muon GRUG_TAU=8 GRUG_CORRECTOR=dc_asgd_ema \
      .venv/bin/iris --cluster=marin job run --no-wait --reserve v6e-8 \
      -e WANDB_API_KEY "$WANDB_API_KEY" -- python -m experiments.grug.moe_delay.launch
"""

import dataclasses
import os

from fray.cluster import ResourceConfig
from levanter.optim import OptimizerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.grug.moe_delay.delay_optim import DelayedGrugMoeAdamHConfig, DelayedGrugMuonConfig

# Default Muon matrix-path LR. MuonConfig.lr is unused by the build path (the
# scheduler reads `learning_rate`), so we set `learning_rate` explicitly.
DEFAULT_MUON_LR: float = 0.01


def _env(key: str, default: str) -> str:
    raw = os.environ.get(key, "")
    return raw if raw else default


def _build_optimizer(opt: str, base_opt, tau: int, corrector: str, dc_lambda: float) -> OptimizerConfig:
    """Build a delayed optimizer config for the selected arm.

    ``base_opt`` is the heuristic-tuned ``GrugMoeAdamHConfig`` for this model
    size; the Adam arm reuses its hyperparameters verbatim, the Muon arm borrows
    its schedule/AdamW-path LR.
    """
    if opt == "adamh":
        fields = {f.name: getattr(base_opt, f.name) for f in dataclasses.fields(base_opt)}
        return DelayedGrugMoeAdamHConfig(**fields, tau=tau, corrector=corrector, dc_lambda=dc_lambda)
    if opt == "muon":
        return DelayedGrugMuonConfig(
            learning_rate=DEFAULT_MUON_LR,
            adam_lr=base_opt.adam_lr,
            beta1=base_opt.beta1,
            beta2=base_opt.beta2,
            epsilon=base_opt.epsilon,
            max_grad_norm=base_opt.max_grad_norm or 1.0,
            min_lr_ratio=base_opt.min_lr_ratio,
            warmup=base_opt.warmup,
            lr_schedule=base_opt.lr_schedule,
            tau=tau,
            corrector=corrector,
            dc_lambda=dc_lambda,
        )
    raise ValueError(f"unknown GRUG_OPT={opt!r}; expected 'muon' or 'adamh'")


def _make_step() -> ExecutorStep:
    opt = _env("GRUG_OPT", "muon")
    tau = int(_env("GRUG_TAU", "0"))
    corrector = _env("GRUG_CORRECTOR", "none")
    dc_lambda = float(_env("GRUG_DC_LAMBDA", "1.0"))
    steps = int(_env("GRUG_STEPS", "3000"))
    seed = int(_env("GRUG_SEED", "0"))
    hidden = int(_env("GRUG_HIDDEN", "512"))
    budget = float(_env("GRUG_BUDGET", "2.19e17"))
    tpu = _env("GRUG_TPU", "v6e-8")
    group = _env("GRUG_GROUP", "delay-pp-batch1")

    model, base_opt, batch, _full_steps = build_from_heuristic(budget=budget, hidden_dim=hidden)
    optimizer = _build_optimizer(opt, base_opt, tau, corrector, dc_lambda)

    corr_tag = corrector if corrector == "none" else f"{corrector}-l{dc_lambda:g}"
    run_id = f"delay-{opt}-d{hidden}-tau{tau}-{corr_tag}-s{seed}-st{steps}"

    launch = GrugMoeLaunchConfig(
        model=versioned(model),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=run_id,
        resources=versioned(ResourceConfig.with_tpu(tpu)),
        steps=versioned(steps),
        batch_size=versioned(batch),
        seed=versioned(seed),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        # checkpointer=None -> run_grug_moe_trial builds the default (one save per
        # 10 min); short experiment runs save at most a final checkpoint.
        checkpointer=None,
        tracker=WandbConfig(project="marin_moe", tags=["moe", "delay-pp"], group=group, name=None),
        optimizer=versioned(optimizer),
        grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=512,
                steps_per_eval=1000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            )
        ),
    )

    return ExecutorStep(name=f"grug/delay/{run_id}", fn=run_grug_moe_trial, config=launch)


if __name__ == "__main__":
    executor_main(
        steps=[_make_step()],
        description="Grug MoE delayed-gradient staleness sweep.",
    )
