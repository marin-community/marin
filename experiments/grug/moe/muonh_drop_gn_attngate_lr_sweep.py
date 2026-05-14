# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LR sensitivity sweep on the muonh-drop-gn-attngate recipe (#5726).

Base recipe: MuonH + drop GatedNorm + drop attn_gate, baseline 10%
warmup. Three optimizer groups:

- ``muonh``: matrix-shaped leaves (attn matrices, MoE MLP w_*).
- ``adamh``: ``lm_head`` and ``output_proj``.
- ``adam``: ``token_embed``, ``router``, ``router_bias``.

For each of the three groups we shift its peak LR by ±30% (x1.3 and
x0.7) while keeping the other two at 1.0x. Schedule shape (linear
warmup to linear decay) is preserved. Repeated at d512, d768, d1024
for 3 x 2 x 3 = 18 runs.

Submission:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --zone us-east5-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_drop_gn_attngate_lr_sweep
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
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHPerGroupLrConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

# 3 scales x 3 groups x 2 signs = 18 runs.
_POINTS: tuple[tuple[int, float], ...] = (
    (512, 2.19e17),
    (768, 1.70e18),
    (1024, 9.00e18),
)

# (group, scale_factor, suffix_label)
_LR_PERTURBATIONS: tuple[tuple[str, float, str], ...] = (
    ("muonh", 1.3, "muonh-1p3"),
    ("muonh", 0.7, "muonh-0p7"),
    ("adamh", 1.3, "adamh-1p3"),
    ("adamh", 0.7, "adamh-0p7"),
    ("adam", 1.3, "adam-1p3"),
    ("adam", 0.7, "adam-0p7"),
)

_BASELINE_TARGET_STEPS: int = 2**14
_GROUP_NAME: str = "muonh-drop-gn-attngate-lr-sweep"


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _format_run_id(hidden_dim: int, budget: float, lr_label: str, run_suffix: str = "") -> str:
    budget_label = _format_budget(budget)
    normalized_suffix = run_suffix.strip()
    if normalized_suffix and not normalized_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_suffix may only contain letters, numbers, hyphens, and underscores")
    suffix = f"{normalized_suffix}-" if normalized_suffix else ""
    return f"muonh-drop-gn-attngate-lr-{lr_label}-{suffix}d{hidden_dim}-{budget_label}"


def _muonh_optimizer(
    base_optimizer: GrugMoeAdamHConfig,
    *,
    group: str,
    scale: float,
) -> GrugMoeMuonHPerGroupLrConfig:
    """MuonH per-group-LR config mirroring the AdamH baseline knobs but
    applying ``scale`` to ``group``'s LR."""
    muonh_scale = scale if group == "muonh" else 1.0
    adamh_scale = scale if group == "adamh" else 1.0
    adam_scale = scale if group == "adam" else 1.0
    return GrugMoeMuonHPerGroupLrConfig(
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
        muonh_lr_scale=muonh_scale,
        adamh_lr_scale=adamh_scale,
        adam_lr_scale=adam_scale,
    )


def _build_step(
    hidden_dim: int,
    budget: float,
    *,
    group: str,
    scale: float,
    lr_label: str,
    run_suffix: str = "",
) -> ExecutorStep:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden_dim,
        target_steps=_BASELINE_TARGET_STEPS,
    )
    optimizer = _muonh_optimizer(base_optimizer, group=group, scale=scale)

    run_id = _format_run_id(hidden_dim=hidden_dim, budget=budget, lr_label=lr_label, run_suffix=run_suffix)
    step_name = f"grug/muonh_drop_gn_attngate_lr_sweep/{run_id}"

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
                tags=["moe", "muonh_drop_gn_attngate_lr_sweep", f"d{hidden_dim}", lr_label],
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
    steps = [
        _build_step(
            hidden_dim=hidden_dim,
            budget=budget,
            group=group,
            scale=scale,
            lr_label=lr_label,
            run_suffix=_RUN_SUFFIX,
        )
        for hidden_dim, budget in _POINTS
        for group, scale, lr_label in _LR_PERTURBATIONS
    ]
    executor_main(
        steps=steps,
        description=(
            f"MoE MuonH + drop GN + drop attn_gate, LR sensitivity sweep "
            f"(3 sizes x 3 groups x +-30%, run_suffix={_RUN_SUFFIX!r})."
        ),
    )
