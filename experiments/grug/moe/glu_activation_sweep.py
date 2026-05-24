# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GLU vs SwiGLU activation: gate-1 sweep.

Replaces the silu gate inside every MLP (shared expert + routed experts)
with plain sigmoid, turning the SwiGLU block ``silu(gate) * up`` into the
classic GLU formulation ``sigmoid(gate) * up`` (Dauphin et al. 2017).

Toggled via the new ``GrugModelConfig.mlp_gate_activation`` field, which
threads into both ``MoEMLP.__call__`` (routed experts) and the
``Block`` call to its shared ``DenseMLP``. The baseline keeps
``mlp_gate_activation=ActivationFunctionEnum.silu``; this sweep flips it
to ``sigmoid`` at the two gate-1 scales (d512 / 2.19e17, d768 / 1.70e18).

Usage::

    .venv/bin/iris --cluster=marin job run \\
      --no-wait --zone us-east5-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.glu_activation_sweep
"""

import dataclasses
import os
from concurrent.futures import ThreadPoolExecutor

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from levanter.utils.activation import ActivationFunctionEnum
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

_TARGET_STEPS: int = 2**14
_TPU: str = "v5p-8"
_SUFFIX: str = "v1"
_VARIANT_TAG: str = "glu-sigmoid"

# Gate 1 scales from agent.md.
_SCALES: tuple[tuple[int, float], ...] = (
    (512, 2.19e17),
    (768, 1.70e18),
)


def _build_launch(*, hidden_dim: int, budget: float) -> tuple[str, GrugMoeDirectLaunchConfig]:
    base_model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden_dim,
        target_steps=_TARGET_STEPS,
    )
    model = dataclasses.replace(base_model, mlp_gate_activation=ActivationFunctionEnum.sigmoid)
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
    budget_tag = f"{budget:.2e}".replace("+", "")
    run_id = _resolve_run_id(f"grug-moe-{_VARIANT_TAG}-d{hidden_dim}-{budget_tag}-{_SUFFIX}")
    name = f"grug/{_VARIANT_TAG}-d{hidden_dim}-{_SUFFIX}"

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
            tags=["moe", "moe_direct", "may_recipe", "glu_activation", _VARIANT_TAG, f"d{hidden_dim}"],
            group="glu-activation",
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
    assert model.mlp_gate_activation is ActivationFunctionEnum.sigmoid
    return name, launch


def _submit_one(name: str, launch: GrugMoeDirectLaunchConfig) -> str:
    """Blocking submit so the Iris coordinator (this process) stays the parent."""
    return train_grug_moe(name=name, launch=launch, wait=True)


def main() -> None:
    if "WANDB_API_KEY" not in os.environ:
        raise RuntimeError("WANDB_API_KEY must be set in the Iris coordinator env")

    jobs = [_build_launch(hidden_dim=d, budget=b) for d, b in _SCALES]
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
