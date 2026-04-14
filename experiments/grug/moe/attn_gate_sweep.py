# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Attention gate sizing sweep: no gate, truncated, and LoRA variants.

GitHub issue: https://github.com/marin-community/marin/issues/4716
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

# Gate 1 compute-optimal points.
GATE1_CONFIGS: list[tuple[int, float]] = [
    (512, 2.19e17),
    (768, 1.70e18),
]

FRACTIONS = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64]

GATE_CONFIGS: list[tuple[str, float]] = [
    ("none", 0.0),
    *[("truncated", f) for f in FRACTIONS],
    *[("lora", f) for f in FRACTIONS],
]


def _fraction_label(frac: float) -> str:
    if frac == 0.0:
        return "0"
    inv = round(1 / frac)
    return f"1_{inv}"


def _make_sweep_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for dim, budget in GATE1_CONFIGS:
        base_model, optimizer, batch, num_steps = build_from_heuristic(budget=budget, hidden_dim=dim)
        for gate_mode, frac in GATE_CONFIGS:
            model = dataclasses.replace(base_model, attn_gate_mode=gate_mode, attn_gate_fraction=frac)
            frac_label = _fraction_label(frac)
            run_id = f"attn-gate-d{dim}-{budget:.2e}-{gate_mode}-{frac_label}"

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
                            project="dial_moe",
                            tags=["attn-gate-sweep", f"d={dim}", f"budget={budget:.2e}", gate_mode],
                            group="attn-gate-sweep",
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


all_steps = _make_sweep_steps()

if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description="Attention gate sizing sweep: none/truncated/lora at gate 1 scales.",
    )
