# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""All-GatedNorms-to-Adam routing sweep across the compute-optimal MoE gate points.

Identical to the v16 AdamH baseline in ``README.md`` except for one mask
change in :class:`GrugMoeAdamHConfig.create_mask`: every ``GatedNorm``
instance (per-block ``attn_gated_norm`` / ``mlp_gated_norm``, model-level
``embed_gated_norm`` / ``final_gated_norm``) now routes to the plain
``adam`` group at the smaller ``adam_lr``.

This is the symmetric "down" routing for GatedNorm matrices. The
corresponding "up" routing (all four -> ``adamh`` at ``learning_rate``)
lives on the ``moe_attn_gated_norm_to_adamh`` branch.

Gate and run-suffix are pinned in module constants at the bottom of this
file (``_GATE`` and ``_RUN_SUFFIX``); edit them directly to change scope.
Submit with no env vars:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.all_gated_norms_adam_sweep
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
_GROUP: str = "all-gated-norms-adam-sweep"


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _format_run_id(hidden_dim: int, budget: float, run_suffix: str = "") -> str:
    budget_label = _format_budget(budget)
    normalized_suffix = run_suffix.strip()
    if normalized_suffix and not normalized_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_suffix may only contain letters, numbers, hyphens, and underscores")
    suffix = f"{normalized_suffix}-" if normalized_suffix else ""
    return f"all-gated-norms-adam-{suffix}d{hidden_dim}-{budget_label}"


def _build_step(hidden_dim: int, budget: float, run_suffix: str = "") -> ExecutorStep:
    model, optimizer, batch_size, num_steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden_dim,
        target_steps=_BASELINE_TARGET_STEPS,
    )

    run_id = _format_run_id(hidden_dim=hidden_dim, budget=budget, run_suffix=run_suffix)
    step_name = f"grug/all_gated_norms_adam_sweep/{run_id}"

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
                tags=["moe", "all_gated_norms_adam_sweep", f"d{hidden_dim}"],
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
            # Not versioned: if the worker's region differs from the existing
            # checkpoint's bucket, scan all regions and resume from the
            # highest-step checkpoint found. One-time ~6 GB cross-region read
            # under the default 10 GB transfer budget.
            enable_cross_region_ckpt_read=True,
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


_GATE: str = "1"  # "1" | "2" | "both"
_RUN_SUFFIX: str = "v1"


if __name__ == "__main__":
    steps = _build_steps(_GATE, run_suffix=_RUN_SUFFIX)
    executor_main(
        steps=steps,
        description=f"MoE all GatedNorms -> Adam sweep (gate={_GATE}, run_suffix={_RUN_SUFFIX!r}).",
    )
