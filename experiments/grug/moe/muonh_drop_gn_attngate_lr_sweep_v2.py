# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LR sensitivity sweep v2 on the muonh-drop-gn-attngate recipe.

Refinement of the v1 LR sweep (#5737) based on its results: moves
``token_embed`` out of the small-LR ``adam`` group into a new
``adamh_embed`` group (Frobenius hyperball), splits the existing
``adamh`` (which had only ``output_proj``) into ``adamh_lmhead``, and
sets a new "tuned" default point:

| group         | params                                 | new default scale (vs heuristic) |
|---------------|----------------------------------------|----------------------------------|
| ``muonh``        | matrices (attn, MoE MLPs, shared)       | 1.0x ``learning_rate``           |
| ``adamh_lmhead`` | ``output_proj`` / ``lm_head``               | 0.7x ``learning_rate``           |
| ``adamh_embed``  | ``token_embed``                            | 0.7x ``learning_rate``           |
| ``adam``         | ``router``, ``router_bias``, norms           | 1.3x ``adam_lr``                 |

From this new default, perturbs each group's scale and re-runs at d512.

| perturbation   | runs                          |
|----------------|-------------------------------|
| (default)      | 1                             |
| muonh +-10%   | 2 (0.9x, 1.1x)                |
| embed +-30%   | 2 (0.49x, 0.91x effective)    |
| lm_head +-20% | 2 (0.56x, 0.84x effective)    |
| adam +-20%    | 2 (1.04x, 1.56x effective)    |

Total: 9 runs, all at d512 (2.19e17 FLOPs).

Submit:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --zone us-east5-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_drop_gn_attngate_lr_sweep_v2
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
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHFourGroupLrConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_HIDDEN_DIM: int = 512
_BUDGET: float = 2.19e17
_BASELINE_TARGET_STEPS: int = 2**14
_GROUP_NAME: str = "muonh-drop-gn-attngate-lr-sweep-v2"

# New default scales (tuned point from v1).
_DEFAULT_MUONH_SCALE: float = 1.0
_DEFAULT_ADAMH_LMHEAD_SCALE: float = 0.7
_DEFAULT_ADAMH_EMBED_SCALE: float = 0.7
_DEFAULT_ADAM_SCALE: float = 1.3


# (label, kwargs to override on the default config)
_TRIALS: tuple[tuple[str, dict], ...] = (
    ("default", {}),
    # muonh +-10%
    ("muonh-1p1", {"muonh_lr_scale": _DEFAULT_MUONH_SCALE * 1.1}),
    ("muonh-0p9", {"muonh_lr_scale": _DEFAULT_MUONH_SCALE * 0.9}),
    # embed +-30%
    ("embed-1p3", {"adamh_embed_lr_scale": _DEFAULT_ADAMH_EMBED_SCALE * 1.3}),
    ("embed-0p7", {"adamh_embed_lr_scale": _DEFAULT_ADAMH_EMBED_SCALE * 0.7}),
    # lm_head +-20%
    ("lmhead-1p2", {"adamh_lmhead_lr_scale": _DEFAULT_ADAMH_LMHEAD_SCALE * 1.2}),
    ("lmhead-0p8", {"adamh_lmhead_lr_scale": _DEFAULT_ADAMH_LMHEAD_SCALE * 0.8}),
    # adam +-20%
    ("adam-1p2", {"adam_lr_scale": _DEFAULT_ADAM_SCALE * 1.2}),
    ("adam-0p8", {"adam_lr_scale": _DEFAULT_ADAM_SCALE * 0.8}),
)


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _format_run_id(trial_label: str, hidden_dim: int, budget: float, run_suffix: str = "") -> str:
    budget_label = _format_budget(budget)
    normalized_suffix = run_suffix.strip()
    if normalized_suffix and not normalized_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_suffix may only contain letters, numbers, hyphens, and underscores")
    suffix = f"{normalized_suffix}-" if normalized_suffix else ""
    return f"muonh-drop-gn-attngate-lr-v2-{trial_label}-{suffix}d{hidden_dim}-{budget_label}"


def _muonh_optimizer(base_optimizer: GrugMoeAdamHConfig, overrides: dict) -> GrugMoeMuonHFourGroupLrConfig:
    cfg = GrugMoeMuonHFourGroupLrConfig(
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
        muonh_lr_scale=_DEFAULT_MUONH_SCALE,
        adamh_lmhead_lr_scale=_DEFAULT_ADAMH_LMHEAD_SCALE,
        adamh_embed_lr_scale=_DEFAULT_ADAMH_EMBED_SCALE,
        adam_lr_scale=_DEFAULT_ADAM_SCALE,
    )
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    return cfg


def _build_step(trial_label: str, overrides: dict, run_suffix: str = "") -> ExecutorStep:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_BASELINE_TARGET_STEPS,
    )
    optimizer = _muonh_optimizer(base_optimizer, overrides)

    run_id = _format_run_id(trial_label, _HIDDEN_DIM, _BUDGET, run_suffix=run_suffix)
    step_name = f"grug/muonh_drop_gn_attngate_lr_sweep_v2/{run_id}"

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
                tags=["moe", "muonh_drop_gn_attngate_lr_sweep_v2", f"d{_HIDDEN_DIM}", trial_label],
                group=_GROUP_NAME,
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
            enable_cross_region_ckpt_read=True,
        ),
    )


_RUN_SUFFIX: str = "v1"


if __name__ == "__main__":
    steps = [_build_step(label, overrides, run_suffix=_RUN_SUFFIX) for label, overrides in _TRIALS]
    executor_main(
        steps=steps,
        description=(
            f"MoE MuonH + drop GN + drop attn_gate, LR sweep v2 "
            f"(d512, 9 trials around tuned default, run_suffix={_RUN_SUFFIX!r})."
        ),
    )
