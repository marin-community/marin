# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""may_arch + 1pct-noclip with finer per-subgroup MuonH LR scales.

Motivation: empirically the gradient norms differ between attention QK
vs VO matrices, and between MLP up/gate vs down matrices. This sweep
probes whether re-balancing the MuonH LR across those sub-groups helps,
without changing total optimizer footprint.

Built on the 1pct-noclip baseline (d512=3.6427, d768=3.3040). Tests four
±30% LR splits using the new ``GrugMoeMuonHMayArch1pctFinerLrConfig``:

| trial label              | qk_lr | vo_lr | mlp_out_lr | mlp_in_lr |
|--------------------------|-------|-------|------------|-----------|
| ``qk-1p3-vo-0p7``        | 1.3x  | 0.7x  | 1.0x       | 1.0x      |
| ``qk-0p7-vo-1p3``        | 0.7x  | 1.3x  | 1.0x       | 1.0x      |
| ``mlpout-1p3-mlpin-0p7`` | 1.0x  | 1.0x  | 1.3x       | 0.7x      |
| ``mlpout-0p7-mlpin-1p3`` | 1.0x  | 1.0x  | 0.7x       | 1.3x      |

4 trials x 2 sizes (d512, d768) = 8 runs.

Submit on us-east5-a:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --zone us-east5-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_may_arch_1pct_finer_lr_sweep
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
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHMayArch1pctFinerLrConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

# Original 8-run sweep at d512/d768 is content-hash cached; the entries
# are commented out so a fresh ``iris job run`` only enqueues the new
# d1024 follow-up trial below.
#
# _POINTS = ((512, 2.19e17), (768, 1.70e18))
# _TRIALS = (
#     ("qk-1p3-vo-0p7", {"qk_lr_scale": 1.3, "vo_lr_scale": 0.7}),
#     ("qk-0p7-vo-1p3", {"qk_lr_scale": 0.7, "vo_lr_scale": 1.3}),
#     ("mlpout-1p3-mlpin-0p7", {"mlp_out_lr_scale": 1.3, "mlp_in_lr_scale": 0.7}),
#     ("mlpout-0p7-mlpin-1p3", {"mlp_out_lr_scale": 0.7, "mlp_in_lr_scale": 1.3}),
# )
#
# Second-round extension: ``qk-1p2-vo-0p8`` at d1024 only. Motivated by
# ``qk-1p3-vo-0p7`` tying for d768 leader (3.3002) — softens the
# perturbation and tests whether the win generalizes to d1024.
_POINTS: tuple[tuple[int, float], ...] = ((1024, 9.00e18),)

_TRIALS: tuple[tuple[str, dict], ...] = (("qk-1p2-vo-0p8", {"qk_lr_scale": 1.2, "vo_lr_scale": 0.8}),)

_WARMUP_FRACTION: float = 0.01
_NUM_EXPERTS: int = 256
_BASELINE_TARGET_STEPS: int = 2**14
_GROUP_NAME: str = "muonh-may-arch-1pct-finer-lr-sweep"


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _format_run_id(trial_label: str, hidden_dim: int, budget: float, run_suffix: str = "") -> str:
    budget_label = _format_budget(budget)
    normalized_suffix = run_suffix.strip()
    if normalized_suffix and not normalized_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_suffix may only contain letters, numbers, hyphens, and underscores")
    suffix = f"{normalized_suffix}-" if normalized_suffix else ""
    return f"muonh-may-arch-1pct-finer-lr-{trial_label}-{suffix}d{hidden_dim}-{budget_label}"


def _muonh_optimizer(base_optimizer: GrugMoeAdamHConfig, overrides: dict) -> GrugMoeMuonHMayArch1pctFinerLrConfig:
    cfg = GrugMoeMuonHMayArch1pctFinerLrConfig(
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
    step_name = f"grug/muonh_may_arch_1pct_finer_lr_sweep/{run_id}"

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
                tags=["moe", "muonh_may_arch_1pct_finer_lr_sweep", f"d{hidden_dim}", trial_label],
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
            f"MoE may_arch + 1pct-noclip + finer per-subgroup MuonH LR "
            f"(4 trials x 2 sizes = 8 runs, run_suffix={_RUN_SUFFIX!r})."
        ),
    )
