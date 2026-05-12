# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MuonH + GatedNorms-to-AdamH + NO WARMUP routing sweep.

Combines two changes vs. the v16 AdamH baseline in ``README.md``:

1. Optimizer swap to MuonH (per-expert NS direction inside the Frobenius
   hyperball update) for the matrix-shaped leaves the AdamH baseline
   routes to AdamH / AdamH-expert.
2. Mask change: route every ``GatedNorm`` instance (per-block
   ``attn_gated_norm`` and ``mlp_gated_norm``, model-level
   ``embed_gated_norm`` and ``final_gated_norm``) to the AdamH group at
   ``learning_rate``.
3. Set ``warmup=0`` on the MuonH optimizer. The grug/moe heuristic
   overrides levanter's default 1% warmup to ``warmup=0.1`` (10% of
   train steps), so the AdamH baseline burns ~640 / 6,386 steps at
   d512 ramping LR; this variant skips that entirely and starts at
   peak LR from step 0.

Gate and run-suffix are pinned at the bottom of this file (``_GATE`` and
``_RUN_SUFFIX``). Submit with no env vars:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait --priority batch \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_gn_adamh_no_warmup_sweep
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
    (512, 2.19e17),
    (768, 1.70e18),
)
_GATE_2_POINTS: tuple[tuple[int, float], ...] = (
    (1024, 9.00e18),
    (1280, 2.83e19),
)

_BASELINE_TARGET_STEPS: int = 2**14
_GROUP: str = "muonh-gn-adamh-no-warmup-sweep"


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _format_run_id(hidden_dim: int, budget: float, run_suffix: str = "") -> str:
    budget_label = _format_budget(budget)
    normalized_suffix = run_suffix.strip()
    if normalized_suffix and not normalized_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_suffix may only contain letters, numbers, hyphens, and underscores")
    suffix = f"{normalized_suffix}-" if normalized_suffix else ""
    return f"muonh-gn-adamh-no-warmup-{suffix}d{hidden_dim}-{budget_label}"


def _muonh_optimizer_from_baseline(base_optimizer: GrugMoeAdamHConfig) -> GrugMoeMuonHConfig:
    """Build a MuonH config that mirrors the baseline AdamH config, but with
    ``warmup`` zeroed out so the LR schedule starts at peak.

    The grug/moe heuristic sets ``warmup=0.1`` (10% of train steps); zeroing
    skips a ~640-step ramp at d512 and ~1,034 at d768.
    """
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
    step_name = f"grug/muonh_gn_adamh_no_warmup_sweep/{run_id}"

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
                tags=["moe", "muonh_gn_adamh_no_warmup_sweep", f"d{hidden_dim}"],
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
        description=f"MoE MuonH + GN-AdamH + no-warmup sweep (gate={_GATE}, run_suffix={_RUN_SUFFIX!r}).",
    )
