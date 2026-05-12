# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MuonH + split gate/up storage + GatedNorms-to-AdamH + no warmup sweep.

Stacks three changes on top of the MuonH baseline (where the AdamH
baseline routes matrix params through Frobenius-hyperball + Muon-NS):

1. Split-storage MoEMLP: ``w_gate`` and ``w_up`` are stored as separate
   ``(E, d, i)`` tensors (concatenated only on the forward pass into the
   moe_mlp kernel). This lets Muon orthogonalize them as two independent
   ``(d, i)`` matrices instead of a single ``(d, 2i)`` block. Comes from
   ``model.py`` change cherry-picked from ``moe_muonh_paired_v2``.
2. GatedNorm matrices route to ``adamh`` at ``learning_rate``. Fixes the
   ``attn_gate`` substring-bug routing (substring check now uses
   ``endswith(".attn_gate")``) and adds an explicit
   ``"gated_norm" in path_lower -> "adamh"`` branch in
   ``GrugMoeMuonHConfig.create_mask``.
3. ``warmup=0.0`` on the MuonH optimizer. The grug/moe heuristic sets
   ``warmup=0.1`` (10% of train steps); zeroing skips a ~640-step ramp
   at d512 and ~1,034 at d768. MuonH ramps directly to peak LR.

The inner Fray TPU resource is pinned to ``regions=("us-east5",
"us-central1")`` so the parent's region doesn't bleed into the child
constraint and so checkpoints stay in-region with the output bucket.

Gate and run-suffix are pinned in module constants at the bottom of
this file. Submit with no env vars and default (interactive) priority:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_split_gate_up_gn_adamh_no_warmup_sweep

Wandb runs go to the ``dial_moe`` project (matching the v16 isoflop
baselines + muonh-matrix-baseline runs).
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
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_GATE_1_POINTS: tuple[tuple[int, float], ...] = (
    # Restricted to d768 only for the v2 re-kickoff; d512 already ran under v1.
    (768, 1.70e18),
)
_GATE_2_POINTS: tuple[tuple[int, float], ...] = (
    (1024, 9.00e18),
    (1280, 2.83e19),
)

_BASELINE_TARGET_STEPS: int = 2**14
_GROUP: str = "muonh-split-gn-adamh-no-warmup-sweep"


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _format_run_id(hidden_dim: int, budget: float, run_suffix: str = "") -> str:
    budget_label = _format_budget(budget)
    normalized_suffix = run_suffix.strip()
    if normalized_suffix and not normalized_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_suffix may only contain letters, numbers, hyphens, and underscores")
    suffix = f"{normalized_suffix}-" if normalized_suffix else ""
    return f"muonh-split-gn-adamh-no-warmup-{suffix}d{hidden_dim}-{budget_label}"


def _muonh_optimizer_from_baseline(base_optimizer: GrugMoeAdamHConfig) -> GrugMoeMuonHConfig:
    """Mirror the baseline AdamH config onto MuonH, then zero out warmup."""
    return GrugMoeMuonHConfig(
        learning_rate=base_optimizer.learning_rate,
        adam_lr=base_optimizer.adam_lr,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=0.0,
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
    optimizer = _muonh_optimizer_from_baseline(base_optimizer)

    run_id = _format_run_id(hidden_dim=hidden_dim, budget=budget, run_suffix=run_suffix)
    step_name = f"grug/muonh_split_gn_adamh_no_warmup_sweep/{run_id}"

    return ExecutorStep(
        name=step_name,
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            # us-east5 only: matches the marin-us-east5 output bucket so all
            # checkpoint reads / writes stay in-region (no cross-region GCS I/O).
            resources=versioned(ResourceConfig.with_tpu("v5p-8", regions=("us-east5",))),
            steps=versioned(num_steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="dial_moe",
                tags=["moe", "muonh_split_gn_adamh_no_warmup_sweep", f"d{hidden_dim}"],
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


_GATE: str = "1"  # "1" | "2" | "both"
_RUN_SUFFIX: str = "v2"


if __name__ == "__main__":
    steps = _build_steps(_GATE, run_suffix=_RUN_SUFFIX)
    executor_main(
        steps=steps,
        description=(
            f"MoE MuonH + split gate/up + GN->AdamH + no-warmup sweep " f"(gate={_GATE}, run_suffix={_RUN_SUFFIX!r})."
        ),
    )
