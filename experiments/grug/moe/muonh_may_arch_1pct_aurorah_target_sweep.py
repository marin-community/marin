# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""may_arch + 1pct-noclip baseline, AuroraH ablation on K/V vs GN vs expert-MLP-out.

Built on the current best 1pct-noclip recipe (``muonh-may-arch-gn-muonh-1pct-noclip``,
d512=3.6427, d768=3.3040). Replaces plain MuonH NS-polar with Aurora's
leverage-uniform polar (``aurora.py`` from ``tilde-research/aurora-release``)
on one of three rectangular-matrix subgroups:

| trial label        | AuroraH target                                                    |
|--------------------|-------------------------------------------------------------------|
| ``aurorah-kv``     | ``attn.w_k`` and ``attn.w_v``                                     |
| ``aurorah-gn``     | all 4 GatedNorm matrices (``w_up`` / ``w_down``)                  |
| ``aurorah-expout`` | expert MLP output projections (``.mlp.w_down`` / ``.shared.w_down``) |

(All three have the 1pct-noclip recipe: 1% warmup, ``max_grad_norm=None``,
embed -> AdamH, GN -> muonh by default, may_arch architecture.)

3 variants x 3 scales = 9 runs.

Submit on us-east5-a with batch priority:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority batch \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_may_arch_1pct_aurorah_target_sweep
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
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHMayArchAuroraTargetConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_POINTS: tuple[tuple[int, float], ...] = (
    (512, 2.19e17),
    (768, 1.70e18),
    (1024, 9.00e18),
)

_TRIALS: tuple[tuple[str, dict], ...] = (
    ("aurorah-kv", {"kv_aurora": True}),
    ("aurorah-gn", {"gn_aurora": True}),
    ("aurorah-expout", {"expout_aurora": True}),
)

_WARMUP_FRACTION: float = 0.01
_NUM_EXPERTS: int = 256
_BASELINE_TARGET_STEPS: int = 2**14
_GROUP_NAME: str = "muonh-may-arch-1pct-aurorah-target-sweep"


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _format_run_id(trial_label: str, hidden_dim: int, budget: float, run_suffix: str = "") -> str:
    budget_label = _format_budget(budget)
    normalized_suffix = run_suffix.strip()
    if normalized_suffix and not normalized_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_suffix may only contain letters, numbers, hyphens, and underscores")
    suffix = f"{normalized_suffix}-" if normalized_suffix else ""
    return f"muonh-may-arch-1pct-{trial_label}-{suffix}d{hidden_dim}-{budget_label}"


def _muonh_optimizer(base_optimizer: GrugMoeAdamHConfig, overrides: dict) -> GrugMoeMuonHMayArchAuroraTargetConfig:
    cfg = GrugMoeMuonHMayArchAuroraTargetConfig(
        learning_rate=base_optimizer.learning_rate,
        adam_lr=base_optimizer.adam_lr,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=_WARMUP_FRACTION,
        beta1=base_optimizer.beta1,
        beta2=base_optimizer.beta2,
        epsilon=base_optimizer.epsilon,
        max_grad_norm=None,
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
    step_name = f"grug/muonh_may_arch_1pct_aurorah_target_sweep/{run_id}"

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
                tags=["moe", "muonh_may_arch_1pct_aurorah_target_sweep", f"d{hidden_dim}", trial_label],
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
            f"MoE may_arch + 1pct-noclip + AuroraH (Aurora's leverage-uniform polar) "
            f"on K/V vs GN vs expert-MLP-out, 3 scales x 3 variants = 9 runs, "
            f"run_suffix={_RUN_SUFFIX!r}."
        ),
    )
