# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""may_arch + gated post-norm scaling sweep.

Sandwich gated norm: in addition to the pre-sublayer RMSNorm+GatedNorm, also
apply RMSNorm+GatedNorm to the residual stream after each sub-layer addition.

    r_{L+1} = GatedNorm(RMSNorm(r_L + f(GatedNorm(RMSNorm(r_L)))))

Runs the four compute-optimal scales (d512/d768/d1024/d1280) on top of the
may_arch baseline (PKO every_4th + partial RoPE + last_layer_pko + k4e256).
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

SCALES: list[tuple[int, float]] = [
    (512, 2.19e17),
    (768, 1.70e18),
    (1024, 9.00e18),
    (1280, 2.83e19),
]


def _make_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for dim, budget in SCALES:
        model, optimizer, batch, num_steps = build_from_heuristic(budget=budget, hidden_dim=dim)
        model = dataclasses.replace(
            model,
            partial_key_offset="every_4th",
            use_partial_rope=True,
            last_layer_pko=True,
            num_experts=256,
            num_experts_per_token=4,
            gated_post_norm=True,
        )

        run_id = f"may-arch-gpn-d{dim}-{budget:.2e}"

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
                    enable_cross_region_ckpt_read=True,
                    steps=versioned(num_steps),
                    batch_size=versioned(batch),
                    seed=versioned(0),
                    mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                    tracker=WandbConfig(
                        project="dial_moe",
                        tags=["may-arch", "gated-post-norm", f"d={dim}", f"budget={budget:.2e}"],
                        group="may-arch-gated-post-norm",
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
        description="may_arch + gated post-norm scaling sweep: d512/d768/d1024/d1280.",
    )
