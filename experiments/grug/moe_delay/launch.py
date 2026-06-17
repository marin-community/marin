# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delayed-gradient staleness sweep launcher for the grug MoE template.

Runs grug-MoE training directly on the TPU job (``run_inline`` -> `_run_grug_local`,
no Fray driver/dispatch hop) with the optimizer swapped for a delayed-gradient
wrapper (see `delay_optim.py`). The arm is selected entirely by environment
variables so a single module can be submitted many times with different
staleness / corrector settings:

    GRUG_OPT        muon | adamh                 (default muon)
    GRUG_TAU        uniform gradient delay in steps (default 0; ignored if STAGES>0)
    GRUG_STAGES     pipeline stages for the per-stage delay profile (default 0=uniform)
    GRUG_CORRECTOR  none | dc_asgd | dc_asgd_ema | weight_pred | lr_damp |
                    wp_preorth | wp_cautious | wp_trust | wp_confidence  (default none)
    GRUG_DC_LAMBDA  DC-ASGD strength             (default 1.0)
    GRUG_PRED_SCALE weight_pred / wp_* horizon as a multiple of tau    (default 1.0)
    GRUG_PRED_BETA  wp_* raw-momentum EMA decay  (default 0.95)
    GRUG_TRUST      wp_trust trust-ratio clamp   (default 0.01)
    GRUG_LR_DAMP    lr_damp step multiplier      (default 1.0)
    GRUG_STEPS      train steps (short for fast iteration)  (default 3000)
    GRUG_SEED       seed                         (default 0)
    GRUG_HIDDEN     model hidden dim             (default 512)
    GRUG_BUDGET     compute budget for heuristic sizing/LR  (default 2.19e17)
    GRUG_GROUP      wandb group                  (default delay-pp-batch1)

Submit directly on a TPU (no reservation, no driver job), e.g.:

    GRUG_OPT=muon GRUG_TAU=8 GRUG_CORRECTOR=dc_asgd_ema \
      .venv/bin/iris --cluster=marin job run --no-wait --tpu v6e-8 \
      --enable-extra-resources --extra marin-core:tpu \
      -e WANDB_API_KEY "$WANDB_API_KEY" -- python -m experiments.grug.moe_delay.launch
"""

import dataclasses
import os
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.optim import OptimizerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned
from marin.training.training import temporary_checkpoint_base_path

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    _resolve_tracker,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, _run_grug_local
from experiments.grug.moe_delay.delay_optim import DelayedGrugMoeAdamHConfig, DelayedGrugMuonConfig


def run_inline(config: GrugMoeLaunchConfig) -> None:
    """Run grug-MoE training in-process (no Fray driver/dispatch hop).

    Mirrors ``experiments.grug.moe.launch.run_grug_moe_trial`` but calls
    ``_run_grug_local`` directly so the training runs on whatever job holds the
    TPU (submitted with ``--tpu``), instead of dispatching a separate worker.
    The executor still resolves the data-config cache paths before this runs.
    """
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=config.profiler,
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=config.checkpointer
        or CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(hours=24),
            keep=None,
        ),
    )
    grug_trainer = dataclasses.replace(config.grug_trainer, trainer=trainer)
    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    _run_grug_local(run_config)


# Default Muon matrix-path LR. MuonConfig.lr is unused by the build path (the
# scheduler reads `learning_rate`), so we set `learning_rate` explicitly.
DEFAULT_MUON_LR: float = 0.01


def _env(key: str, default: str) -> str:
    raw = os.environ.get(key, "")
    return raw if raw else default


def _build_optimizer(
    opt: str,
    base_opt,
    *,
    tau: int,
    corrector: str,
    dc_lambda: float,
    pred_scale: float,
    pred_beta: float,
    trust: float,
    lr_damp: float,
    num_stages: int,
    num_layers: int,
) -> OptimizerConfig:
    """Build a delayed optimizer config for the selected arm.

    ``base_opt`` is the heuristic-tuned ``GrugMoeAdamHConfig`` for this model
    size; the Adam arm reuses its hyperparameters verbatim, the Muon arm borrows
    its schedule/AdamW-path LR. ``num_stages > 0`` selects the realistic per-stage
    pipeline-parallel delay profile (over ``num_layers`` blocks) instead of the
    uniform global ``tau``.
    """
    delay = dict(
        tau=tau,
        corrector=corrector,
        dc_lambda=dc_lambda,
        pred_scale=pred_scale,
        pred_beta=pred_beta,
        trust=trust,
        lr_damp=lr_damp,
        num_stages=num_stages,
        num_layers=num_layers,
    )
    if opt == "adamh":
        fields = {f.name: getattr(base_opt, f.name) for f in dataclasses.fields(base_opt)}
        return DelayedGrugMoeAdamHConfig(**fields, **delay)
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
            **delay,
        )
    raise ValueError(f"unknown GRUG_OPT={opt!r}; expected 'muon' or 'adamh'")


def _make_step() -> ExecutorStep:
    opt = _env("GRUG_OPT", "muon")
    tau = int(_env("GRUG_TAU", "0"))
    corrector = _env("GRUG_CORRECTOR", "none")
    dc_lambda = float(_env("GRUG_DC_LAMBDA", "1.0"))
    pred_scale = float(_env("GRUG_PRED_SCALE", "1.0"))
    pred_beta = float(_env("GRUG_PRED_BETA", "0.95"))
    trust = float(_env("GRUG_TRUST", "0.01"))
    lr_damp = float(_env("GRUG_LR_DAMP", "1.0"))
    num_stages = int(_env("GRUG_STAGES", "0"))
    steps = int(_env("GRUG_STEPS", "3000"))
    seed = int(_env("GRUG_SEED", "0"))
    hidden = int(_env("GRUG_HIDDEN", "512"))
    budget = float(_env("GRUG_BUDGET", "2.19e17"))
    tpu = _env("GRUG_TPU", "v6e-8")
    group = _env("GRUG_GROUP", "delay-pp-batch1")

    model, base_opt, batch, _full_steps = build_from_heuristic(budget=budget, hidden_dim=hidden)
    num_layers = model.num_layers
    optimizer = _build_optimizer(
        opt,
        base_opt,
        tau=tau,
        corrector=corrector,
        dc_lambda=dc_lambda,
        pred_scale=pred_scale,
        pred_beta=pred_beta,
        trust=trust,
        lr_damp=lr_damp,
        num_stages=num_stages,
        num_layers=num_layers,
    )

    if corrector == "none":
        corr_tag = "none"
    elif corrector == "weight_pred":
        corr_tag = f"weight_pred-p{pred_scale:g}"
    elif corrector == "wp_trust":
        corr_tag = f"wp_trust-p{pred_scale:g}t{trust:g}"
    elif corrector in ("wp_preorth", "wp_cautious", "wp_confidence"):
        corr_tag = f"{corrector}-p{pred_scale:g}"
    elif corrector == "lr_damp":
        corr_tag = f"lr_damp-d{lr_damp:g}"
    else:
        corr_tag = f"{corrector}-l{dc_lambda:g}"
    # Per-stage PP runs encode the stage count (uniform tau is ignored); uniform
    # runs encode tau.
    delay_tag = f"pp{num_stages}" if num_stages > 0 else f"tau{tau}"
    run_id = f"delay-{opt}-d{hidden}-{delay_tag}-{corr_tag}-s{seed}-st{steps}"

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

    return ExecutorStep(name=f"grug/delay/{run_id}", fn=run_inline, config=launch)


if __name__ == "__main__":
    executor_main(
        steps=[_make_step()],
        description="Grug MoE delayed-gradient staleness sweep.",
    )
