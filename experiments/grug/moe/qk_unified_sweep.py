# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unified QK sweep: norm_scope x gate_target x gate_type x model_size.

12 setups x 4 sizes = 48 runs total.

Already done on other branches (gate 1 only):
- per_head + q + scalar: d512, d768
- per_head + k + simple: d512, d768
- per_head + k + gated_norm: d512, d768
- per_head + q + simple: d512, d768
- per_head + q + gated_norm: d512, d768
- full + q + scalar: d512, d768

This sweep runs the remaining 36 runs.

GitHub issue: https://github.com/marin-community/marin/issues/5373
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

ALL_CONFIGS: list[tuple[int, float]] = [
    (512, 2.19e17),
    (768, 1.70e18),
    (1024, 9.00e18),
    (1280, 2.83e19),
]

# (norm_scope, gate_target, gate_type)
ALL_SETUPS: list[tuple[str, str, str]] = [
    ("per_head", "q", "scalar"),
    ("per_head", "k", "scalar"),
    ("per_head", "q", "simple"),
    ("per_head", "k", "simple"),
    ("per_head", "q", "gated_norm"),
    ("per_head", "k", "gated_norm"),
    ("full", "q", "scalar"),
    ("full", "k", "scalar"),
    ("full", "q", "simple"),
    ("full", "k", "simple"),
    ("full", "q", "gated_norm"),
    ("full", "k", "gated_norm"),
]

# Already completed on other branches (gate 1 = d512+d768)
DONE: set[tuple[str, str, str, int]] = {
    ("per_head", "q", "scalar", 512),
    ("per_head", "q", "scalar", 768),
    ("per_head", "k", "simple", 512),
    ("per_head", "k", "simple", 768),
    ("per_head", "k", "gated_norm", 512),
    ("per_head", "k", "gated_norm", 768),
    ("per_head", "q", "simple", 512),
    ("per_head", "q", "simple", 768),
    ("per_head", "q", "gated_norm", 512),
    ("per_head", "q", "gated_norm", 768),
    ("full", "q", "scalar", 512),
    ("full", "q", "scalar", 768),
}


def _make_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for norm_scope, gate_target, gate_type in ALL_SETUPS:
        for dim, budget in ALL_CONFIGS:
            if (norm_scope, gate_target, gate_type, dim) in DONE:
                continue

            model, optimizer, batch, num_steps = build_from_heuristic(budget=budget, hidden_dim=dim)
            model = dataclasses.replace(
                model,
                qk_norm_scope=norm_scope,
                qk_gate_target=gate_target,
                qk_gate_type=gate_type,
            )

            norm_label = "fn" if norm_scope == "full" else "ph"
            run_id = f"qk-{norm_label}-{gate_target}-{gate_type}-d{dim}-{budget:.2e}"

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
                            tags=["qk-unified", norm_label, gate_target, gate_type, f"d={dim}"],
                            group="qk-unified",
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
        description="Unified QK sweep: remaining 36 runs.",
    )
