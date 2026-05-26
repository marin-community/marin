# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-point KL-Soap-Muon-H run at d512 with ``precond_freq=1``.

Variant of ``klsoaph_d512_freq1.py``: applies the same Newton-Schulz quintic
orthogonalization that ``GrugMoeMuonHConfig`` uses (steps=5, coefficient_type=
"quintic") to the KLSOAPH direction *before* the hyperball post-step. The
intent is to test whether NS5 orthogonalization stacks productively on top of
SOAP preconditioning at d512.

Submit:

    .venv/bin/iris --config lib/iris/config/marin.yaml job run --no-wait \\
      --preemptible \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.klsoaph_muon_h_d512_freq1
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
from experiments.grug.moe.optimizer import GrugMoeKLSoapMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_HIDDEN_DIM: int = 512
_BUDGET: float = 2.19e17
_TARGET_STEPS: int = 2**14
_TPU: str = "v5p-8"
_RUN_SUFFIX: str = "muon-h-v3"


def _build_launch() -> tuple[str, GrugMoeDirectLaunchConfig]:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_TARGET_STEPS,
    )

    # Upstream KLSOAPH "passing" tuple (PR #290) + precond_freq=1 + Muon NS5.
    # NS coefficients / step count match GrugMoeMuonHConfig defaults.
    optimizer = GrugMoeKLSoapMuonHConfig(
        learning_rate=base_optimizer.learning_rate,
        adam_lr=base_optimizer.adam_lr,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=0.01,
        beta1=0.95,
        beta2=0.9,
        shampoo_beta=0.9,
        epsilon=base_optimizer.epsilon,
        precond_freq=1,
        init_factor=0.1,
        max_grad_norm=None,
        ns5_steps=5,
        ns5_coefficient_type="quintic",
        lr_schedule=base_optimizer.lr_schedule,
        decay=base_optimizer.decay,
    )

    run_id = _resolve_run_id(f"klsoaph-muon-h-d{_HIDDEN_DIM}-{_BUDGET:.2e}-freq1-{_RUN_SUFFIX}".replace("+", ""))
    name = f"grug/klsoaph-muon-h-d{_HIDDEN_DIM}-freq1-{_RUN_SUFFIX}"

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
            tags=["moe", "klsoaph_muon_h", "klsoaph_muon_h_d512_freq1", f"d{_HIDDEN_DIM}"],
            group="klsoaph-muon-h-d512-freq1",
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
    return name, launch


if __name__ == "__main__":
    name, launch = _build_launch()
    print("Submitting KL-Soap-Muon-H d512 freq=1 run")
    job_id = train_grug_moe(name=name, launch=launch, wait=True)
    print(f"Training job finished: {job_id}")
