# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-point AMUSE run at d512 (arxiv 2605.22432).

AMUSE = Anytime Muon with Stable Gradient Evaluation: time-varying
Schedule-Free wrapper over Muon (matrix params) + AdamW-without-β1 (the rest).
Defaults β_1=0.6, ρ=0.8, T_0=2000 follow the paper's d512 LLM recipe.

The LR schedule is set to ``constant`` (warmup + flat thereafter); AMUSE
removes the need for a decay phase. Same model / data / batch as the existing
klsoaph d512 launches.

Submit:

    .venv/bin/iris --config lib/iris/config/marin.yaml job run --no-wait \\
      --preemptible \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.amuse_d512
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
_RUN_SUFFIX: str = "amuse-v7-lr2x-b1-0.4-rho-0.6"
# AMUSE sweep knobs (override per run):
#   _LR_MULTIPLIER scales the matrix LR (and adam_lr proportionally) over the
#     heuristic MuonH baseline. AMUSE has no decay phase so the *average*
#     effective LR is higher than a cosine-decay baseline at the same peak.
#   _AMUSE_BETA1 is the SF interpolation coefficient initial value
#     (paper d512: {0.4, 0.6}). Smaller β_1 → more "fast Muon" early.
#   _AMUSE_RHO is the β_t growth exponent (paper d512: {0.6, 0.8}).
#     Smaller ρ → slower β_t ramp toward 1.
_LR_MULTIPLIER: float = 2.0
_AMUSE_BETA1: float = 0.4
_AMUSE_RHO: float = 0.6
_AMUSE_T0: int = 500


def _build_launch() -> tuple[str, GrugMoeDirectLaunchConfig]:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_TARGET_STEPS,
    )

    # AMUSE recommended d512 LLM hyperparameters from arxiv 2605.22432:
    #   β_1 ∈ {0.4, 0.6}, ρ ∈ {0.6, 0.8}, T_0 = 2000.
    # LR schedule is constant (warmup + flat) per AMUSE's "no decay needed"
    # principle; we sweep LR multiplier (vs heuristic MuonH baseline) and
    # (β_1, ρ) here.
    optimizer = GrugMoeAmuseConfig(
        learning_rate=base_optimizer.learning_rate * _LR_MULTIPLIER,
        adam_lr=base_optimizer.adam_lr * _LR_MULTIPLIER,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        # Shorter LR warmup matching the MuonH baseline (warmup=0.01 → ~63
        # steps at 6302 total). Then constant LR per AMUSE.
        warmup=0.01,
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
        steps=versioned(num_steps),
        batch_size=versioned(batch_size),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            entity="marin-community",
            project="marin_moe",
            tags=["moe", "amuse", "amuse_d512", f"d{_HIDDEN_DIM}"],
            group="amuse-d512",
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
                eval_current=True,  # also report eval at Y (current params) for comparison
                eval_ema=True,  # eval at AMUSE's X (averaged sequence) via ema_params slot
            )
        ),
        checkpoint_keep_every=1000,
    )
    return name, launch


if __name__ == "__main__":
    name, launch = _build_launch()
    print("Submitting AMUSE d512 run (arxiv 2605.22432)")
    job_id = train_grug_moe(name=name, launch=launch, wait=True)
    print(f"Training job finished: {job_id}")
