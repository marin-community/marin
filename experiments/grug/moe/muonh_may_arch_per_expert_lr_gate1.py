# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate-1 sweep for per-expert MuonH LR scaling on may_arch.

Builds on #5763:
- may_arch with 256 experts
- GatedNorms in MuonH
- 1% warmup
- no gradient clipping

Hypothesis: routed expert weights see only a sparse fraction of tokens, so
their MuonH LR should be scaled down by sqrt(sparsity), where
sparsity = num_experts_per_token / num_experts.

Gate 1 runs d512 and d768 for three candidates:
1. shrink-expert: keep non-expert MuonH LR at 1.0x, set expert LR to sqrt(sparsity).
2. boost-nonexpert: keep expert LR at 1.0x, set non-expert MuonH LR to 1/sqrt(sparsity).
3. mid-ratio: keep the same non-expert/expert ratio as above, but set
   expert LR to the geometric middle between sqrt(sparsity) and 1.0.

Submit:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --reserve v5p-8 \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_may_arch_per_expert_lr_gate1
"""

import dataclasses
import math

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHMayArchPerExpertLrConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_GATE1_POINTS: tuple[tuple[int, float], ...] = (
    (512, 2.19e17),
    (768, 1.70e18),
)

_CANDIDATES: tuple[str, ...] = (
    "shrink-expert",
    "boost-nonexpert",
    "mid-ratio",
)

_WARMUP_FRACTION: float = 0.01
_NUM_EXPERTS: int = 256
_BASELINE_TARGET_STEPS: int = 2**14
_GROUP_NAME: str = "muonh-may-arch-per-expert-lr-gate1"
_RUN_SUFFIX: str = "v1"


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _format_run_id(candidate: str, hidden_dim: int, budget: float, run_suffix: str = "") -> str:
    budget_label = _format_budget(budget)
    normalized_suffix = run_suffix.strip()
    if normalized_suffix and not normalized_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_suffix may only contain letters, numbers, hyphens, and underscores")
    suffix = f"{normalized_suffix}-" if normalized_suffix else ""
    return f"muonh-may-arch-per-expert-lr-gate1-{candidate}-{suffix}d{hidden_dim}-{budget_label}"


def _sqrt_sparsity(model: GrugModelConfig) -> float:
    sparsity = model.num_experts_per_token / model.num_experts
    if not 0.0 < sparsity <= 1.0:
        raise ValueError(f"sparsity must be in (0, 1], got {sparsity}")
    return math.sqrt(sparsity)


def _candidate_lr_scales(candidate: str, model: GrugModelConfig) -> tuple[float, float]:
    sqrt_sparsity = _sqrt_sparsity(model)
    if candidate == "shrink-expert":
        return 1.0, sqrt_sparsity
    if candidate == "boost-nonexpert":
        return 1.0 / sqrt_sparsity, 1.0
    if candidate == "mid-ratio":
        expert_lr_scale = math.sqrt(sqrt_sparsity)
        return expert_lr_scale / sqrt_sparsity, expert_lr_scale
    raise ValueError(f"unknown per-expert LR candidate: {candidate}")


def _muonh_optimizer(
    base_optimizer: GrugMoeAdamHConfig,
    *,
    model: GrugModelConfig,
    candidate: str,
) -> GrugMoeMuonHMayArchPerExpertLrConfig:
    muonh_lr_scale, expert_lr_scale = _candidate_lr_scales(candidate, model)
    return GrugMoeMuonHMayArchPerExpertLrConfig(
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
        muonh_lr_scale=muonh_lr_scale,
        expert_lr_scale=expert_lr_scale,
    )


def _build_step(candidate: str, hidden_dim: int, budget: float, run_suffix: str = "") -> ExecutorStep:
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
    optimizer = _muonh_optimizer(base_optimizer, model=model, candidate=candidate)

    run_id = _format_run_id(candidate, hidden_dim, budget, run_suffix=run_suffix)
    step_name = f"grug/muonh_may_arch_per_expert_lr_gate1/{run_id}"

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
                tags=["moe", "per_expert_lr", "gate1", candidate, f"d{hidden_dim}"],
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


if __name__ == "__main__":
    steps = [
        _build_step(candidate, hidden_dim=d, budget=c, run_suffix=_RUN_SUFFIX)
        for candidate in _CANDIDATES
        for d, c in _GATE1_POINTS
    ]
    executor_main(
        steps=steps,
        description=(
            "Gate-1 MoE may_arch per-expert MuonH LR scaling "
            f"(candidates={_CANDIDATES!r}, run_suffix={_RUN_SUFFIX!r})."
        ),
    )
