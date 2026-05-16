# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""may_arch + 1pct-noclip + NorMuon (intermediate-axis) on routed expert I/O.

Counterpart to #5785 (AuroraH on expert I/O). Same target — routed
expert ``w_gate`` / ``w_up`` / ``w_down`` — but with NorMuon instead of
Aurora's leverage-uniform polar. NorMuon adds a per-channel adaptive
second-moment normalization on top of the Muon NS direction.

Crucially this variant normalizes on the **longer** of the last two
trailing axes (the *intermediate* dim) for each tensor:

| tensor         | shape (e, m, n) | normalized axis |
|----------------|-----------------|-----------------|
| ``mlp.w_gate`` | (e, d, 4d)      | -1 (4d intermediate)  |
| ``mlp.w_up``   | (e, d, 4d)      | -1 (4d intermediate)  |
| ``mlp.w_down`` | (e, 4d, d)      | -2 (4d intermediate)  |

The intermediate axis is the one Muon's NS-polar does NOT orthonormalize
for these tensors; NorMuon here captures the per-intermediate-channel
contribution to/from the residual stream that plain MuonH leaves
uncontrolled.

Requires the ``MoEMLP`` storage layout where ``w_gate`` and ``w_up``
are kept as separate ``(e, d, i)`` tensors and concatenated on the
forward pass (introduced on the AuroraH expert-I/O branch).

3 runs at d512, d768, d1024.

Submit on us-east5-a:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --zone us-east5-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_may_arch_1pct_normuon_expert_io_sweep
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
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHMayArch1pctNorMuonExpertIoConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_POINTS: tuple[tuple[int, float], ...] = (
    (512, 2.19e17),
    (768, 1.70e18),
    (1024, 9.00e18),
)

_TRIAL_LABEL: str = "normuon-expert-io"

_WARMUP_FRACTION: float = 0.01
_NUM_EXPERTS: int = 256
_BASELINE_TARGET_STEPS: int = 2**14
_GROUP_NAME: str = "muonh-may-arch-1pct-normuon-expert-io-sweep"


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _format_run_id(hidden_dim: int, budget: float, run_suffix: str = "") -> str:
    budget_label = _format_budget(budget)
    normalized_suffix = run_suffix.strip()
    if normalized_suffix and not normalized_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_suffix may only contain letters, numbers, hyphens, and underscores")
    suffix = f"{normalized_suffix}-" if normalized_suffix else ""
    return f"muonh-may-arch-1pct-{_TRIAL_LABEL}-{suffix}d{hidden_dim}-{budget_label}"


def _muonh_optimizer(base_optimizer: GrugMoeAdamHConfig) -> GrugMoeMuonHMayArch1pctNorMuonExpertIoConfig:
    return GrugMoeMuonHMayArch1pctNorMuonExpertIoConfig(
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


def _build_step(hidden_dim: int, budget: float, run_suffix: str = "") -> ExecutorStep:
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
    optimizer = _muonh_optimizer(base_optimizer)

    run_id = _format_run_id(hidden_dim, budget, run_suffix=run_suffix)
    step_name = f"grug/muonh_may_arch_1pct_normuon_expert_io_sweep/{run_id}"

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
                tags=["moe", "muonh_may_arch_1pct_normuon_expert_io_sweep", f"d{hidden_dim}", _TRIAL_LABEL],
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
    steps = [_build_step(hidden_dim=d, budget=c, run_suffix=_RUN_SUFFIX) for d, c in _POINTS]
    executor_main(
        steps=steps,
        description=(
            f"MoE may_arch + 1pct-noclip + NorMuon (intermediate-axis) on routed expert I/O "
            f"(d512/d768/d1024, run_suffix={_RUN_SUFFIX!r})."
        ),
    )
