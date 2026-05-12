# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NorMuonH (wider-axis) + split gate/up + GN->AdamH + no warmup sweep.

Stacks on top of the MuonH baseline with four changes:

1. **Split-storage MoEMLP**: ``w_gate`` and ``w_up`` stored as separate
   ``(E, d, i)`` tensors, concat-on-forward. Lets the optimizer treat
   them as independent matrices.
2. **NorMuon along the wider non-leading axis**: the second-moment
   statistic buffer is along whichever of the last two axes is wider —
   "the axis Muon misses". For matrices that are square in their
   trailing two dims (e.g. attention ``w_q``, ``w_o``, ``shared.w_*``),
   NorMuon is skipped entirely (Muon already controls both axes).
3. **GatedNorms -> AdamH** + ``attn_gate`` substring fix.
4. **``warmup=0.0``** on the NorMuonH optimizer.

Matrices affected by the wider-axis NorMuon, given the split-storage
MoEMLP and grug/moe defaults:

* ``attn.w_k`` / ``attn.w_v`` with GQA 4:1 — tall ``(d, d/4)``, stat
  along ``d`` (rows).
* ``mlp.w_gate`` / ``mlp.w_up`` per expert — tall ``(d, d/2)``, stat
  along ``d``.
* ``mlp.w_down`` per expert — wide ``(d/2, d)``, stat along ``d``.

Square matrices (``attn.w_q``, ``attn.w_o``, ``shared.w_gate``,
``shared.w_up``, ``shared.w_down`` at default ``shared_intermediate_dim
= d``) pass through NorMuon unchanged.

Submit with default (interactive) priority:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.normuon_wider_axis_split_gn_adamh_no_warmup_sweep

Wandb runs go to the ``dial_moe`` project.
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
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeNorMuonHWiderAxisConfig
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
_GROUP: str = "normuon-wider-axis-split-gn-adamh-no-warmup-sweep"


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _format_run_id(hidden_dim: int, budget: float, run_suffix: str = "") -> str:
    budget_label = _format_budget(budget)
    normalized_suffix = run_suffix.strip()
    if normalized_suffix and not normalized_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_suffix may only contain letters, numbers, hyphens, and underscores")
    suffix = f"{normalized_suffix}-" if normalized_suffix else ""
    return f"normuon-wider-axis-split-gn-adamh-no-warmup-{suffix}d{hidden_dim}-{budget_label}"


def _normuonh_wider_axis_optimizer_from_baseline(
    base_optimizer: GrugMoeAdamHConfig,
) -> GrugMoeNorMuonHWiderAxisConfig:
    """Mirror the baseline AdamH config onto NorMuonH-wider-axis, zero warmup."""
    return GrugMoeNorMuonHWiderAxisConfig(
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
    optimizer = _normuonh_wider_axis_optimizer_from_baseline(base_optimizer)

    run_id = _format_run_id(hidden_dim=hidden_dim, budget=budget, run_suffix=run_suffix)
    step_name = f"grug/normuon_wider_axis_split_gn_adamh_no_warmup_sweep/{run_id}"

    return ExecutorStep(
        name=step_name,
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v5p-8", regions=("us-east5", "us-central1"))),
            steps=versioned(num_steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="dial_moe",
                tags=["moe", "normuon_wider_axis_split_gn_adamh_no_warmup_sweep", f"d{hidden_dim}"],
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
_RUN_SUFFIX: str = "v1"


if __name__ == "__main__":
    steps = _build_steps(_GATE, run_suffix=_RUN_SUFFIX)
    executor_main(
        steps=steps,
        description=(
            f"MoE NorMuonH (wider axis) + split gate/up + GN->AdamH + no-warmup sweep "
            f"(gate={_GATE}, run_suffix={_RUN_SUFFIX!r})."
        ),
    )
