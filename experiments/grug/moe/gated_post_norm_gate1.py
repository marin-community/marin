# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gated post-norm — gate 1 (d512, d768) on the canonical grug-moe baseline.

Variant: r_{L+1} = GatedNorm(RMSNorm(r_L + f(GatedNorm(RMSNorm(r_L))))).

Adds an independent RMSNorm+GatedNorm on the residual stream after each
sub-layer in addition to the existing pre-sublayer RMSNorm+GatedNorm. No other
architectural changes — strict A/B vs. the README compute-opt baseline.

Series ID: MOE-GPN — sub-issue of #4281.
Run ids:   MOE-GPN-001 (d512), MOE-GPN-002 (d768).
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

# (run_id, hidden_dim, compute_budget) — matches README compute-opt baseline rows.
GATE1_SCALES: list[tuple[str, int, float]] = [
    ("MOE-GPN-001", 512, 2.19e17),
    ("MOE-GPN-002", 768, 1.70e18),
]


def _make_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for series_id, dim, budget in GATE1_SCALES:
        model, optimizer, batch, num_steps = build_from_heuristic(budget=budget, hidden_dim=dim)
        model = dataclasses.replace(model, gated_post_norm=True)

        run_id = f"{series_id}-gpn-d{dim}-{budget:.2e}"

        steps.append(
            ExecutorStep(
                name=f"grug/{run_id}",
                fn=run_grug_moe_trial,
                config=GrugMoeLaunchConfig(
                    model=versioned(model),
                    data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                    output_path=this_output_path(),
                    run_id=run_id,
                    resources=versioned(ResourceConfig.with_tpu("v5p-8")),
                    steps=versioned(num_steps),
                    batch_size=versioned(batch),
                    seed=versioned(0),
                    mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                    tracker=WandbConfig(
                        entity="marin-community",
                        project="marin_moe",
                        tags=["gated-post-norm", "gate1", f"d={dim}", f"budget={budget:.2e}", series_id],
                        group="gated-post-norm",
                        name=run_id,
                    ),
                    optimizer=versioned(optimizer),
                    grug_trainer=versioned(
                        GrugTrainerConfig(
                            z_loss_weight=1e-4,
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
                ),
            )
        )
    return steps


all_steps = _make_steps()

if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description="gated_post_norm gate 1: d512, d768.",
    )
