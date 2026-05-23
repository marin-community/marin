# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""muon_epsilon sweep at d512 / 2.19e17.

GrugMoeMuonHConfig's ``muon_epsilon`` (the epsilon used inside the MuonH
update on matrix-shaped params) defaults to ``1e-8`` and is not touched by
any of the current launch scripts or the AdamH heuristic. This sweep tests
``muon_epsilon ∈ {1e-6, 1e-10, 1e-16}`` at the d512 gate-1 scale.

All other knobs (model shape, AdamH ``epsilon``, learning rates, betas,
warmup, schedule, parallel_mode=SERIAL, etc.) match the May Recipe
baseline at d512 (i.e. the run produced by ``direct_launch.py``).

The script submits the three child training jobs in parallel from a single
Iris coordinator (``ThreadPoolExecutor`` + ``wait=True``), so the outer
``iris job run --no-wait`` wrapper acts as the coordinator.

Usage::

    .venv/bin/iris --cluster=marin job run \\
      --no-wait --zone us-east5-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muon_epsilon_sweep
"""

import dataclasses
import os
from concurrent.futures import ThreadPoolExecutor

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
_SUFFIX: str = "v1"

# (tag, muon_epsilon)
_MUON_EPSILONS: tuple[tuple[str, float], ...] = (
    ("muon-eps-1e-6", 1e-6),
    ("muon-eps-1e-10", 1e-10),
    ("muon-eps-1e-16", 1e-16),
)


def _build_launch(*, variant_tag: str, muon_epsilon: float) -> tuple[str, GrugMoeDirectLaunchConfig]:
    base_model, base_optimizer, batch_size, num_steps = build_from_heuristic(
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
        muon_epsilon=muon_epsilon,
        max_grad_norm=None,
        lr_schedule=base_optimizer.lr_schedule,
        decay=base_optimizer.decay,
    )
    budget_tag = f"{_BUDGET:.2e}".replace("+", "")
    run_id = _resolve_run_id(f"grug-moe-{variant_tag}-d{_HIDDEN_DIM}-{budget_tag}-{_SUFFIX}")
    name = f"grug/{variant_tag}-d{_HIDDEN_DIM}-{_SUFFIX}"

    launch = GrugMoeDirectLaunchConfig(
        model=versioned(base_model),
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
            tags=["moe", "moe_direct", "may_recipe", "muon_epsilon_sweep", variant_tag, f"d{_HIDDEN_DIM}"],
            group="muon-epsilon-sweep",
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
    # Force-verify the sweep axis lands in the materialized optimizer.
    assert dataclasses.replace(optimizer).muon_epsilon == muon_epsilon
    return name, launch


def _submit_one(name: str, launch: GrugMoeDirectLaunchConfig) -> str:
    """Blocking submit so the Iris coordinator (this process) stays the parent."""
    return train_grug_moe(name=name, launch=launch, wait=True)


def main() -> None:
    if "WANDB_API_KEY" not in os.environ:
        raise RuntimeError("WANDB_API_KEY must be set in the Iris coordinator env")

    jobs = [_build_launch(variant_tag=tag, muon_epsilon=eps) for tag, eps in _MUON_EPSILONS]
    print(f"Submitting {len(jobs)} training jobs to Iris in parallel:")
    for name, launch in jobs:
        print(f"  - {name}  run_id={launch.run_id}")

    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futures = {ex.submit(_submit_one, name, launch): name for name, launch in jobs}
        for fut in futures:
            name = futures[fut]
            try:
                job_id = fut.result()
                print(f"  [done] {name} -> {job_id}")
            except Exception as exc:
                print(f"  [FAIL] {name}: {exc!r}")
                raise


if __name__ == "__main__":
    main()
