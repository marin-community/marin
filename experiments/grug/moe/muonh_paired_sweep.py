# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MuonH paired-experts sweep across the compute-optimal MoE gate points.

Variant of ``muonh_matrix_sweep.py`` where Muon's Newton-Schulz iteration
runs on **pairs of experts** instead of one expert at a time. Each 3D
expert tensor ``(E, A, B)`` is reshaped before NS to pair adjacent
experts along its **smaller** non-leading axis, so the resulting NS
matrix is as close to square as possible — and so the per-3D-leaf NS
count drops from 64 to 32.

This sweep also splits the standard ``w_gate_up`` MoE parameter into
separate ``w_gate`` and ``w_up`` tensors (concatenated only on the
forward pass before entering the MoE kernel). With Grug MoE defaults
(E=64, ``intermediate_dim = d / 2``), the paired NS matrices for
``w_gate``, ``w_up``, and ``w_down`` are all square ``(d, d)`` per pair.

Model shape, data, batch size, step count, schedule, z-loss, and eval
cadence match the AdamH baseline. Optimizer routing:

* MuonH-paired for matrix-shaped Grug MoE leaves that the AdamH baseline
  routes to AdamH or AdamH-expert.
* AdamH for the lm head / ``output_proj`` matrix.
* Adam for leaves that the AdamH baseline routes to Adam.

Gate and run-suffix are pinned in module-level constants at the bottom
of this file (``_GATE`` and ``_RUN_SUFFIX``); edit them directly to
change scope. Submit with no env vars:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait --reserve v5p-8 \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_paired_sweep
"""

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHPairedConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_GATE_1_POINTS: tuple[tuple[int, float], ...] = (
    (512, 2.19e17),
    (768, 1.70e18),
)
_GATE_2_POINTS: tuple[tuple[int, float], ...] = (
    (1024, 9.00e18),
    (1280, 2.83e19),
)

_BASELINE_TARGET_STEPS: int = 2**14
_GROUP: str = "muonh-paired-sweep"


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _format_run_id(hidden_dim: int, budget: float, run_suffix: str = "") -> str:
    budget_label = _format_budget(budget)
    normalized_suffix = run_suffix.strip()
    if normalized_suffix and not normalized_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_suffix may only contain letters, numbers, hyphens, and underscores")
    suffix = f"{normalized_suffix}-" if normalized_suffix else ""
    return f"muonh-paired-{suffix}d{hidden_dim}-{budget_label}"


def _muonh_paired_optimizer_from_baseline(base_optimizer: GrugMoeAdamHConfig) -> GrugMoeMuonHPairedConfig:
    return GrugMoeMuonHPairedConfig(
        learning_rate=base_optimizer.learning_rate,
        adam_lr=base_optimizer.adam_lr,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=base_optimizer.warmup,
        beta1=base_optimizer.beta1,
        beta2=base_optimizer.beta2,
        epsilon=base_optimizer.epsilon,
        max_grad_norm=base_optimizer.max_grad_norm,
        lr_schedule=base_optimizer.lr_schedule,
        decay=base_optimizer.decay,
    )


def _build_step(hidden_dim: int, budget: float, run_suffix: str = "") -> ExecutorStep:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden_dim,
        target_steps=_BASELINE_TARGET_STEPS,
    )
    optimizer = _muonh_paired_optimizer_from_baseline(base_optimizer)

    run_id = _format_run_id(hidden_dim=hidden_dim, budget=budget, run_suffix=run_suffix)
    step_name = f"grug/muonh_paired_sweep/{run_id}"

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
                entity="marin-community",
                project="marin_moe",
                tags=["moe", "muonh_paired_sweep", f"d{hidden_dim}"],
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


def _build_steps(gate: str, run_suffix: str = "") -> list[ExecutorStep]:
    if gate == "1":
        points = _GATE_1_POINTS
    elif gate == "2":
        points = _GATE_2_POINTS
    elif gate == "both":
        points = _GATE_1_POINTS + _GATE_2_POINTS
    else:
        raise ValueError(f"unknown gate: {gate!r} (expected '1', '2', or 'both')")

    return [_build_step(hidden_dim=hidden_dim, budget=budget, run_suffix=run_suffix) for hidden_dim, budget in points]


# Sweep scope is pinned here (no env-var indirection). Edit and resubmit
# rather than relying on the launch command.
_GATE: str = "1"  # "1" | "2" | "both"
_RUN_SUFFIX: str = "split-v4"


if __name__ == "__main__":
    steps = _build_steps(_GATE, run_suffix=_RUN_SUFFIX)
    executor_main(
        steps=steps,
        description=f"MoE MuonH paired-experts sweep (gate={_GATE}, run_suffix={_RUN_SUFFIX!r}).",
    )
