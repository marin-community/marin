# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Light KL SOAP H hyperparameter sweep at d512 on the May MoE Recipe baseline.

Sweep design: 1-D-at-a-time over (beta1, shampoo_beta, learning-rate multiplier),
anchored at the upstream "passing" tuple from modded-nanogpt PR #290
(beta1=0.95, beta2=0.9, shampoo_beta=0.9, lr=heuristic). 7 unique points; the
center appears once.

Each sweep point is its own coordinator job — submit via:

    for i in 0 1 2 3 4 5 6; do
      .venv/bin/iris --config lib/iris/examples/marin.yaml job run --no-wait \\
        --preemptible \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -e KLSOAPH_SWEEP_POINT $i \\
        -- python -m experiments.grug.moe.klsoaph_d512_sweep
    done

Each coordinator blocks on its training job (wait=True) so the child survives.
KLSOAPH_SWEEP_POINT selects which row of ``SWEEP_POINTS`` to run.
"""

import dataclasses
import os
from dataclasses import dataclass

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
from experiments.grug.moe.optimizer import GrugMoeKLSoapHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_HIDDEN_DIM: int = 512
_BUDGET: float = 2.19e17
_TARGET_STEPS: int = 2**14
_TPU: str = "v5p-8"
_GROUP: str = "klsoaph-d512-sweep"
_RUN_SUFFIX: str = "v1"


@dataclass(frozen=True)
class SweepPoint:
    label: str
    beta1: float
    beta2: float
    shampoo_beta: float
    lr_mult: float  # multiplier on the heuristic learning_rate (and adam_lr)


# Anchor = upstream "passing" tuple (modded-nanogpt PR #290): beta1=0.95, beta2=0.9,
# shampoo_beta=0.9, lr = heuristic baseline.
_CENTER = SweepPoint(label="center", beta1=0.95, beta2=0.9, shampoo_beta=0.9, lr_mult=1.0)

SWEEP_POINTS: list[SweepPoint] = [
    _CENTER,
    # beta1 (momentum) sweep
    dataclasses.replace(_CENTER, label="beta1-0.90", beta1=0.90),
    dataclasses.replace(_CENTER, label="beta1-0.99", beta1=0.99),
    # shampoo_beta sweep
    dataclasses.replace(_CENTER, label="shampoo-0.80", shampoo_beta=0.80),
    dataclasses.replace(_CENTER, label="shampoo-0.95", shampoo_beta=0.95),
    # learning-rate multiplier sweep
    dataclasses.replace(_CENTER, label="lr-0.5x", lr_mult=0.5),
    dataclasses.replace(_CENTER, label="lr-2.0x", lr_mult=2.0),
]


def _build_launch(point: SweepPoint) -> tuple[str, GrugMoeDirectLaunchConfig]:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_TARGET_STEPS,
    )

    # KL SOAP H optimizer with May Recipe baseline schedule: warmup=1%, no clipping.
    optimizer = GrugMoeKLSoapHConfig(
        learning_rate=base_optimizer.learning_rate * point.lr_mult,
        adam_lr=base_optimizer.adam_lr * point.lr_mult,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=0.01,
        beta1=point.beta1,
        beta2=point.beta2,
        shampoo_beta=point.shampoo_beta,
        epsilon=base_optimizer.epsilon,
        precond_freq=5,
        init_factor=0.1,
        max_grad_norm=None,
        lr_schedule=base_optimizer.lr_schedule,
        decay=base_optimizer.decay,
    )

    run_id = _resolve_run_id(f"klsoaph-d{_HIDDEN_DIM}-{_BUDGET:.2e}-{point.label}-{_RUN_SUFFIX}".replace("+", ""))
    name = f"grug/klsoaph-d{_HIDDEN_DIM}-{point.label}-{_RUN_SUFFIX}"

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
            tags=[
                "moe",
                "klsoaph",
                "klsoaph_d512_sweep",
                f"d{_HIDDEN_DIM}",
                point.label,
            ],
            group=_GROUP,
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


def _select_point() -> SweepPoint:
    raw = os.environ.get("KLSOAPH_SWEEP_POINT")
    if raw is None:
        raise SystemExit(
            "KLSOAPH_SWEEP_POINT env var required — pick a row from SWEEP_POINTS by "
            f"index (0..{len(SWEEP_POINTS) - 1}) or by label."
        )
    raw = raw.strip()
    if raw.isdigit():
        idx = int(raw)
        if idx < 0 or idx >= len(SWEEP_POINTS):
            raise SystemExit(f"KLSOAPH_SWEEP_POINT={raw} out of range 0..{len(SWEEP_POINTS) - 1}")
        return SWEEP_POINTS[idx]
    for p in SWEEP_POINTS:
        if p.label == raw:
            return p
    raise SystemExit(f"KLSOAPH_SWEEP_POINT={raw!r} did not match any label in SWEEP_POINTS")


if __name__ == "__main__":
    point = _select_point()
    name, launch = _build_launch(point)
    print(f"Submitting KL SOAP H sweep point: {point}")
    # wait=True so the iris coordinator survives across the training child.
    job_id = train_grug_moe(name=name, launch=launch, wait=True)
    print(f"Training job finished: {job_id}")
