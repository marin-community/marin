# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate 1 sweep: parallel attn/MLP variants of the May MoE recipe.

Runs three variants of the block layout against the May Recipe baseline:

- ``parallel_all``: every layer runs attention and MLP in parallel against the
  residual stream (instead of the baseline serial attn -> mlp wiring).
- ``parallel_half``: only the second half of layers go parallel; the first
  half stays serial.
- ``parallel_merged_norm``: every layer is parallel AND the per-block
  ``RMSNorm`` / ``GatedNorm`` pair is shared between attn and mlp (one norm
  pair per layer instead of two).

Each variant is submitted at the two Gate-1 scales (d512 / 2.19e17,
d768 / 1.70e18). The script submits all six child training jobs in parallel
to Iris from a single coordinator process (so the outer ``iris job run
--no-wait`` wrapper acts as the coordinator) and waits for them to finish.

Usage::

    .venv/bin/iris --cluster=marin job run \\
      --no-wait --zone us-east5-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.parallel_attn_mlp_sweep
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
from experiments.grug.moe.model import ParallelMode
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_TPU: str = "v5p-8"
_TARGET_STEPS: int = 2**14
_SUFFIX: str = "v1"

# Gate 1 scales from agent.md.
_SCALES: tuple[tuple[int, float], ...] = (
    (512, 2.19e17),
    (768, 1.70e18),
)

# (tag, parallel_mode, parallel_second_half_only)
_VARIANTS: tuple[tuple[str, ParallelMode, bool], ...] = (
    ("parallel-all", ParallelMode.PARALLEL, False),
    ("parallel-half", ParallelMode.PARALLEL, True),
    ("parallel-merged", ParallelMode.PARALLEL_MERGED_NORM, False),
)


def _build_launch(
    *,
    variant_tag: str,
    parallel_mode: ParallelMode,
    parallel_second_half_only: bool,
    hidden_dim: int,
    budget: float,
) -> tuple[str, GrugMoeDirectLaunchConfig]:
    base_model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden_dim,
        target_steps=_TARGET_STEPS,
    )
    model = dataclasses.replace(
        base_model,
        parallel_mode=parallel_mode,
        parallel_second_half_only=parallel_second_half_only,
    )
    # Match the May Recipe optimizer (MuonH + AdamH + Adam) from direct_launch.
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
    run_id = _resolve_run_id(f"grug-moe-{variant_tag}-d{hidden_dim}-{budget_tag}-{_SUFFIX}")
    name = f"grug/{variant_tag}-d{hidden_dim}-{_SUFFIX}"

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
            tags=["moe", "moe_direct", "may_recipe", "parallel_attn_mlp", variant_tag, f"d{hidden_dim}"],
            group="parallel-attn-mlp",
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


def _submit_one(name: str, launch: GrugMoeDirectLaunchConfig) -> str:
    """Blocking submit so the iris coordinator (this process) stays the parent."""
    return train_grug_moe(name=name, launch=launch, wait=True)


def main() -> None:
    if "WANDB_API_KEY" not in os.environ:
        raise RuntimeError("WANDB_API_KEY must be set in the Iris coordinator env")

    jobs: list[tuple[str, GrugMoeDirectLaunchConfig]] = []
    for variant_tag, parallel_mode, second_half in _VARIANTS:
        for hidden_dim, budget in _SCALES:
            jobs.append(
                _build_launch(
                    variant_tag=variant_tag,
                    parallel_mode=parallel_mode,
                    parallel_second_half_only=second_half,
                    hidden_dim=hidden_dim,
                    budget=budget,
                )
            )

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
