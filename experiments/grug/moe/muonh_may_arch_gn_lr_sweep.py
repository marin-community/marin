# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""may_arch + MuonH + GN-on-Adam + GN-LR sensitivity sweep.

Recipe:

- MuonH baseline (matrices through Newton-Schulz + Frobenius hyperball).
- ``warmup=0.02`` (2% LR warmup).
- ``token_embed`` routed to ``adamh_embed`` (Frobenius hyperball at full
  ``learning_rate``).
- All 4 GatedNorm instances routed to two adam sub-groups
  (``adam_gn_wdown``, ``adam_gn_wup``) at ``adam_lr``.
- ``num_experts=256`` (4x the heuristic's 64), top-k stays at 4.
- ``partial_key_offset='every_4th'`` + ``use_partial_rope=True`` +
  ``last_layer_pko=True`` (cherry-picked from the ``may_arch`` recipe).
- "Long" attention layers use ``sliding_window=None`` (full causal); only
  the short layers retain the half-window mask.
- ``router_z_loss_coef = 0.0`` and ``z_loss_weight = 0.0`` -- both
  z-loss stabilizers disabled.

For each of d512 / d768 / d1024 we perturb the GatedNorm LR in two ways
(4 trials per scale = 12 total):

| trial    | what changes                                    |
|----------|-------------------------------------------------|
| gn-1p3   | All GN params (w_up + w_down): adam_lr * 1.3    |
| gn-0p7   | All GN params: adam_lr * 0.7                    |
| gnwup-1p3| Just GN.w_up (output projection): adam_lr * 1.3 |
| gnwup-0p7| Just GN.w_up: adam_lr * 0.7                     |

Submit with:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --zone us-east5-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_may_arch_gn_lr_sweep
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
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHMayArchGNLrConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_POINTS: tuple[tuple[int, float], ...] = (
    (512, 2.19e17),
    (768, 1.70e18),
    (1024, 9.00e18),
)

# (label, override-dict on GrugMoeMuonHMayArchGNLrConfig)
_TRIALS: tuple[tuple[str, dict], ...] = (
    # Unperturbed anchor (gn_lr_scale=1.0, gn_wup_lr_scale=1.0).
    ("default", {}),
    ("gn-1p3", {"gn_lr_scale": 1.3}),
    ("gn-0p7", {"gn_lr_scale": 0.7}),
    ("gnwup-1p3", {"gn_wup_lr_scale": 1.3}),
    ("gnwup-0p7", {"gn_wup_lr_scale": 0.7}),
)

_WARMUP_FRACTION: float = 0.02
_NUM_EXPERTS: int = 256
_BASELINE_TARGET_STEPS: int = 2**14
_GROUP_NAME: str = "muonh-may-arch-gn-lr-sweep"


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _format_run_id(trial_label: str, hidden_dim: int, budget: float, run_suffix: str = "") -> str:
    budget_label = _format_budget(budget)
    normalized_suffix = run_suffix.strip()
    if normalized_suffix and not normalized_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_suffix may only contain letters, numbers, hyphens, and underscores")
    suffix = f"{normalized_suffix}-" if normalized_suffix else ""
    return f"muonh-may-arch-{trial_label}-{suffix}d{hidden_dim}-{budget_label}"


def _muonh_optimizer(base_optimizer: GrugMoeAdamHConfig, overrides: dict) -> GrugMoeMuonHMayArchGNLrConfig:
    cfg = GrugMoeMuonHMayArchGNLrConfig(
        learning_rate=base_optimizer.learning_rate,
        adam_lr=base_optimizer.adam_lr,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=_WARMUP_FRACTION,
        beta1=base_optimizer.beta1,
        beta2=base_optimizer.beta2,
        epsilon=base_optimizer.epsilon,
        max_grad_norm=base_optimizer.max_grad_norm,
        lr_schedule=base_optimizer.lr_schedule,
        decay=base_optimizer.decay,
    )
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    return cfg


def _build_step(hidden_dim: int, budget: float, trial_label: str, overrides: dict, run_suffix: str = "") -> ExecutorStep:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden_dim,
        target_steps=_BASELINE_TARGET_STEPS,
    )
    # may_arch architecture additions on top of the heuristic config:
    # 256 experts, PKO + partial rope + last_layer_pko, no router z-loss.
    model = dataclasses.replace(
        model,
        num_experts=_NUM_EXPERTS,
        partial_key_offset="every_4th",
        use_partial_rope=True,
        last_layer_pko=True,
        router_z_loss_coef=0.0,
    )
    optimizer = _muonh_optimizer(base_optimizer, overrides)

    run_id = _format_run_id(trial_label, hidden_dim, budget, run_suffix=run_suffix)
    step_name = f"grug/muonh_may_arch_gn_lr_sweep/{run_id}"

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
                tags=["moe", "muonh_may_arch_gn_lr_sweep", f"d{hidden_dim}", trial_label],
                group=_GROUP_NAME,
                name=None,
            ),
            optimizer=versioned(optimizer),
            grug_trainer=versioned(
                GrugTrainerConfig(
                    z_loss_weight=0.0,
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
            enable_cross_region_ckpt_read=True,
        ),
    )


_RUN_SUFFIX: str = "v1"


if __name__ == "__main__":
    steps = [
        _build_step(hidden_dim=dim, budget=budget, trial_label=label, overrides=overrides, run_suffix=_RUN_SUFFIX)
        for dim, budget in _POINTS
        for label, overrides in _TRIALS
    ]
    executor_main(
        steps=steps,
        description=(
            f"MoE may_arch + MuonH + GN->adam + 2% warmup + GN-LR sensitivity "
            f"(3 sizes x 4 trials = 12 runs, run_suffix={_RUN_SUFFIX!r})."
        ),
    )
