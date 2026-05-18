# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combined "kitchen sink" sweep: stack the best findings + K=5 / half shared.

Two variants at d512 / d768 / d1024 = 6 runs:

- ``combined``: K=5, E=256, shared MLP = hidden_dim // 2, routing_renorm
  X=2.5, split w_gate/w_up storage, pko_first_bos_zero, embed -> plain
  Adam at 1.0x adam_lr.
- ``baseline``: K=4, E=256, shared MLP = hidden_dim (default), no other
  changes (= standard 1pct-noclip recipe).

The ``baseline`` cell is the standard 1pct-noclip recipe; ``combined``
stacks the best findings from #5773 / #5797 / #5800 / #5801 / #5802 /
#5804 plus pushes K from 4 to 5 and halves the shared MLP.

Submit on us-east5-a:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --zone us-east5-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_may_arch_1pct_combined_kitchen_sink_sweep
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
from experiments.grug.moe.optimizer import GrugMoeMuonHMayArchGNMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_POINTS: tuple[tuple[int, float], ...] = (
    (512, 2.19e17),
    (768, 1.70e18),
    (1024, 9.00e18),
)

_WARMUP_FRACTION: float = 0.01
_BASELINE_TARGET_STEPS: int = 2**14
_GROUP_NAME: str = "muonh-may-arch-1pct-combined-kitchen-sink-sweep"
_RUN_SUFFIX: str = "v1"

# Combined trial knobs.
_COMBINED_NUM_EXPERTS: int = 256
_COMBINED_K: int = 5
_COMBINED_ROUTING_RENORM_SUM: float = 2.5
_COMBINED_EMBED_ADAM_LR_SCALE: float = 1.0

# Baseline comparator (= standard 1pct-noclip).
_BASELINE_NUM_EXPERTS: int = 256
_BASELINE_K: int = 4


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _format_run_id(trial: str, hidden_dim: int, budget: float, run_suffix: str = "") -> str:
    budget_label = _format_budget(budget)
    normalized_suffix = run_suffix.strip()
    if normalized_suffix and not normalized_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_suffix may only contain letters, numbers, hyphens, and underscores")
    suffix = f"{normalized_suffix}-" if normalized_suffix else ""
    return f"muonh-may-arch-1pct-combined-{trial}-{suffix}d{hidden_dim}-{budget_label}"


def _build_step(hidden_dim: int, budget: float, trial: str, run_suffix: str = "") -> ExecutorStep:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden_dim,
        target_steps=_BASELINE_TARGET_STEPS,
    )

    if trial == "combined":
        model = dataclasses.replace(
            model,
            num_experts=_COMBINED_NUM_EXPERTS,
            num_experts_per_token=_COMBINED_K,
            shared_expert_intermediate_dim=hidden_dim // 2,
            partial_key_offset="every_4th",
            use_partial_rope=True,
            last_layer_pko=True,
            router_z_loss_coef=0.0,
            routing_renorm_sum=_COMBINED_ROUTING_RENORM_SUM,
            split_w_gate_up=True,
            pko_norm_order="pko_first_bos_zero",
        )
        optimizer = GrugMoeMuonHMayArchGNMuonHConfig(
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
            embed_adam_lr_scale=_COMBINED_EMBED_ADAM_LR_SCALE,
        )
    elif trial == "baseline":
        model = dataclasses.replace(
            model,
            num_experts=_BASELINE_NUM_EXPERTS,
            num_experts_per_token=_BASELINE_K,
            # leave shared_expert_intermediate_dim at the heuristic default
            partial_key_offset="every_4th",
            use_partial_rope=True,
            last_layer_pko=True,
            router_z_loss_coef=0.0,
        )
        optimizer = GrugMoeMuonHMayArchGNMuonHConfig(
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
    else:
        raise ValueError(f"unknown trial label: {trial!r}")

    run_id = _format_run_id(trial, hidden_dim, budget, run_suffix=run_suffix)
    step_name = f"grug/muonh_may_arch_1pct_combined_kitchen_sink_sweep/{run_id}"

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
                tags=[
                    "moe",
                    "muonh_may_arch_1pct_combined_kitchen_sink_sweep",
                    f"d{hidden_dim}",
                    trial,
                ],
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
    trials = ("baseline", "combined")
    steps = [_build_step(hidden_dim=d, budget=c, trial=t, run_suffix=_RUN_SUFFIX) for d, c in _POINTS for t in trials]
    executor_main(
        steps=steps,
        description=(
            "MoE may_arch + 1pct-noclip combined kitchen-sink: K=5 / E=256 / half shared / X=2.5 / "
            "split / pko_first_bos_zero / embed->Adam vs K=4 / E=256 / full shared baseline, at "
            f"d512/d768/d1024 (run_suffix={_RUN_SUFFIX!r})."
        ),
    )
