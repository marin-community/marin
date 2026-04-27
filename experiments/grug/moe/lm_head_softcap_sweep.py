# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LM-head softcap sweep across the four compute-optimal MoE scales.

Apply Gemma2-style tanh soft-capping to the final lm_head logits:
``y = c * tanh(logits / c)``. For each (hidden_dim, budget) point in the v16
isoflop compute-optimal table, launch one run per soft-cap value. Everything
else (heuristic, optimizer, batch, data, mp policy) matches the baseline
launch in launch.py.

Set the env var LM_HEAD_SOFTCAP_GATE to control which scales run:

    LM_HEAD_SOFTCAP_GATE=1     # default: d512, d768 (gate 1 only)
    LM_HEAD_SOFTCAP_GATE=2     # d1024, d1280 (gate 2 only)
    LM_HEAD_SOFTCAP_GATE=both  # all four scales

The full set of variants is configured by `_LM_HEAD_SOFT_CAPS` (default 30.0,
matching Gemma 2's final-logit cap).
"""

import dataclasses
import os

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

_GATE_1_POINTS: tuple[tuple[int, float], ...] = (
    (512, 2.19e17),
    (768, 1.70e18),
)
_GATE_2_POINTS: tuple[tuple[int, float], ...] = (
    (1024, 9.00e18),
    (1280, 2.83e19),
)

_LM_HEAD_SOFT_CAPS: tuple[float, ...] = (30.0,)

_BASELINE_TARGET_STEPS: int = 2**14
_GROUP: str = "lm-head-softcap-sweep"


def _format_cap(cap: float) -> str:
    if float(cap).is_integer():
        return f"c{int(cap)}"
    return f"c{cap:.2f}".replace(".", "p")


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _build_step(hidden_dim: int, budget: float, lm_head_soft_cap: float) -> ExecutorStep:
    base_model, optimizer, batch_size, num_steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden_dim,
        target_steps=_BASELINE_TARGET_STEPS,
    )
    model = dataclasses.replace(base_model, lm_head_soft_cap=lm_head_soft_cap)

    cap_label = _format_cap(lm_head_soft_cap)
    run_id = f"lm-head-softcap-{cap_label}-d{hidden_dim}-{_format_budget(budget)}"
    step_name = f"grug/lm_head_softcap_sweep/{run_id}"

    return ExecutorStep(
        name=step_name,
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v5p-8")),
            steps=versioned(num_steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="marin_moe",
                tags=["moe", "lm_head_softcap_sweep", cap_label, f"d{hidden_dim}"],
                group=_GROUP,
                name=None,
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


def _build_steps(gate: str) -> list[ExecutorStep]:
    if gate == "1":
        points = _GATE_1_POINTS
    elif gate == "2":
        points = _GATE_2_POINTS
    elif gate == "both":
        points = _GATE_1_POINTS + _GATE_2_POINTS
    else:
        raise ValueError(f"unknown gate: {gate!r} (expected '1', '2', or 'both')")

    return [
        _build_step(hidden_dim=hd, budget=bd, lm_head_soft_cap=cap) for hd, bd in points for cap in _LM_HEAD_SOFT_CAPS
    ]


if __name__ == "__main__":
    gate = os.environ.get("LM_HEAD_SOFTCAP_GATE", "1")
    steps = _build_steps(gate)
    executor_main(
        steps=steps,
        description=f"MoE lm_head softcap sweep (gate={gate}): caps={_LM_HEAD_SOFT_CAPS}.",
    )
