# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MuonH learning-rate sweep at fixed model + token budgets.

Stacks on the corrected MuonH ablation (`muonh_matrix_sweep.py`): same
optimizer mask (MuonH on AdamH/AdamH-expert matrix groups, AdamH on the
lm head, Adam preserved on the baseline Adam group), same `warmup=0.1`,
same compute budgets and model sizing. The two knobs that change here
are the MuonH and Adam peak learning rates, each as a multiplier on the
v16 AdamH heuristic LR.

Each `MUONH_LR_SWEEP_POINT` env var is a `(hidden_dim, budget,
muonh_mult, adam_mult)` tuple selected by its index into
`_DEFAULT_GRID`. Set `MUONH_LR_SWEEP_INDEX=i` (0-based) to run a single
grid cell. Set `MUONH_LR_SWEEP_INDEX=all` to enqueue every cell (the
marin executor will only run the ones whose outputs aren't cached).

Set ``MUONH_LR_SWEEP_RUN_SUFFIX`` to append a unique suffix for relaunches.
"""

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
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

# (hidden_dim, budget, muonh_lr_multiplier, adam_lr_multiplier)
# Multipliers are applied to the v16 AdamH heuristic baseline LRs.
# Default grid: 3 adam multipliers x 4 muonh multipliers at d512 (2.19e17).
# Skip the (1.0, 1.0) cell — it's the existing MuonH baseline-adam-mask
# run from #5596. Append a single d768 sanity cell at (1.0, 1.0) for
# reference once the grid optimum is known.
_DEFAULT_GRID: tuple[tuple[int, float, float, float], ...] = (
    (512, 2.19e17, 0.5, 0.5),
    (512, 2.19e17, 0.5, 1.0),
    (512, 2.19e17, 0.5, 2.0),
    (512, 2.19e17, 1.0, 0.5),
    # (512, 2.19e17, 1.0, 1.0) is the existing baseline-adam-mask run.
    (512, 2.19e17, 1.0, 2.0),
    (512, 2.19e17, 2.0, 0.5),
    (512, 2.19e17, 2.0, 1.0),
    (512, 2.19e17, 2.0, 2.0),
    (512, 2.19e17, 4.0, 0.5),
    (512, 2.19e17, 4.0, 1.0),
    (512, 2.19e17, 4.0, 2.0),
)

# Phase 1 warmup ablation: same 11 cells but with warmup=0.0 (vs the v16
# heuristic default 0.1). Distinct run id prefix `muonh-lr-nowarm-...` so
# the cached outputs from the baseline-warmup grid are not reused.
_NOWARMUP_GRID: tuple[tuple[int, float, float, float], ...] = _DEFAULT_GRID

# Phase 3 d1024 LR sweep at the current MuonH:Adam ratio (13/3, i.e. the
# v16 heuristic), with warmup=0.0. Both multipliers move together.
# The existing baseline-adam-mask d1024 (from #5596, warmup=0.1) is NOT
# a substitute for the warmup=0.0 1.0x cell, so the 1.0x is included.
_PHASE3_GRID: tuple[tuple[int, float, float, float], ...] = (
    (1024, 9.00e18, 0.5, 0.5),
    (1024, 9.00e18, 1.0, 1.0),
    (1024, 9.00e18, 2.0, 2.0),
    (1024, 9.00e18, 4.0, 4.0),
)

_BASELINE_TARGET_STEPS: int = 2**14
_GROUP: str = "muonh-lr-sweep"
_GROUP_NOWARMUP: str = "muonh-lr-sweep-nowarmup"
_GROUP_PHASE3: str = "muonh-lr-sweep-d1024"


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _format_mult(value: float) -> str:
    if value == int(value):
        return f"{int(value)}"
    return f"{value:.3g}".replace(".", "p")


def _format_run_id(
    hidden_dim: int,
    budget: float,
    muonh_mult: float,
    adam_mult: float,
    run_suffix: str = "",
    *,
    nowarmup: bool = False,
) -> str:
    budget_label = _format_budget(budget)
    normalized_suffix = run_suffix.strip()
    if normalized_suffix and not normalized_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_suffix may only contain letters, numbers, hyphens, and underscores")
    suffix = f"{normalized_suffix}-" if normalized_suffix else ""
    muonh_label = _format_mult(muonh_mult)
    adam_label = _format_mult(adam_mult)
    nowarm_label = "nowarm-" if nowarmup else ""
    return f"muonh-lr-{nowarm_label}{suffix}d{hidden_dim}-{budget_label}-muonh{muonh_label}x-adam{adam_label}x"


def _muonh_optimizer_from_baseline(
    base_optimizer: GrugMoeAdamHConfig,
    muonh_mult: float,
    adam_mult: float,
    *,
    warmup_override: float | None = None,
) -> GrugMoeMuonHConfig:
    warmup = warmup_override if warmup_override is not None else base_optimizer.warmup
    return GrugMoeMuonHConfig(
        learning_rate=base_optimizer.learning_rate * muonh_mult,
        adam_lr=base_optimizer.adam_lr * adam_mult,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=warmup,
        beta1=base_optimizer.beta1,
        beta2=base_optimizer.beta2,
        epsilon=base_optimizer.epsilon,
        max_grad_norm=base_optimizer.max_grad_norm,
        lr_schedule=base_optimizer.lr_schedule,
        decay=base_optimizer.decay,
    )


def _build_step(
    hidden_dim: int,
    budget: float,
    muonh_mult: float,
    adam_mult: float,
    run_suffix: str = "",
    *,
    nowarmup: bool = False,
) -> ExecutorStep:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden_dim,
        target_steps=_BASELINE_TARGET_STEPS,
    )
    optimizer = _muonh_optimizer_from_baseline(
        base_optimizer,
        muonh_mult=muonh_mult,
        adam_mult=adam_mult,
        warmup_override=0.0 if nowarmup else None,
    )

    run_id = _format_run_id(
        hidden_dim=hidden_dim,
        budget=budget,
        muonh_mult=muonh_mult,
        adam_mult=adam_mult,
        run_suffix=run_suffix,
        nowarmup=nowarmup,
    )
    step_name = f"grug/muonh_lr_sweep/{run_id}"

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
                tags=[
                    "moe",
                    "muonh_lr_sweep",
                    f"d{hidden_dim}",
                    f"muonh{_format_mult(muonh_mult)}x",
                    f"adam{_format_mult(adam_mult)}x",
                    *(["nowarmup"] if nowarmup else []),
                ],
                group=_GROUP_NOWARMUP if nowarmup else _GROUP,
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


_VARIANT_GRIDS: dict[str, tuple[tuple[int, float, float, float], ...]] = {
    "baseline": _DEFAULT_GRID,
    "nowarmup": _NOWARMUP_GRID,
    "phase3": _PHASE3_GRID,
}


def _select_points(spec: str, variant: str) -> list[tuple[int, float, float, float]]:
    if variant not in _VARIANT_GRIDS:
        raise ValueError(f"MUONH_LR_SWEEP_VARIANT must be one of {sorted(_VARIANT_GRIDS)}, got {variant!r}")
    grid = _VARIANT_GRIDS[variant]
    if spec == "all":
        return list(grid)
    try:
        idx = int(spec)
    except ValueError as exc:
        raise ValueError(f"MUONH_LR_SWEEP_INDEX must be 'all' or an int, got {spec!r}") from exc
    if not 0 <= idx < len(grid):
        raise IndexError(f"MUONH_LR_SWEEP_INDEX={idx} out of range [0, {len(grid)})")
    return [grid[idx]]


def _build_steps(spec: str, variant: str, run_suffix: str = "") -> list[ExecutorStep]:
    # Variants that drop the warmup phase entirely (LR jumps to peak at step 0).
    nowarmup = variant in {"nowarmup", "phase3"}
    return [
        _build_step(
            hidden_dim=hd,
            budget=b,
            muonh_mult=mm,
            adam_mult=am,
            run_suffix=run_suffix,
            nowarmup=nowarmup,
        )
        for hd, b, mm, am in _select_points(spec, variant)
    ]


if __name__ == "__main__":
    spec = os.environ.get("MUONH_LR_SWEEP_INDEX", "all")
    variant = os.environ.get("MUONH_LR_SWEEP_VARIANT", "baseline")
    run_suffix = os.environ.get("MUONH_LR_SWEEP_RUN_SUFFIX", "")
    steps = _build_steps(spec, variant=variant, run_suffix=run_suffix)
    executor_main(
        steps=steps,
        description=(
            f"MoE MuonH LR sweep (variant={variant!r}, spec={spec!r}, "
            f"{len(steps)} step(s), run_suffix={run_suffix!r})."
        ),
    )
