# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""may_arch + 1pct-noclip + PKO SiLU-gated query (rotating + stationary).

Redefines the query in PKO layers:

    q = (q_w @ x) * silu(g_w @ x)

where ``g_w`` has shape ``(hidden_dim, num_heads, 2)``; the two trailing
scalars are independent multipliers for the rotating and stationary
halves of each q-head:

    gate = silu(einsum("bsd,dn2->bsn2", x, g_w))     # (B, S, num_heads, 2)
    q_rot  *= gate[..., 0:1]
    q_stat *= gate[..., 1:2]

Applied BEFORE rms_norm(q) and partial-RoPE, only on PKO layers.

``g_w`` initialised with ``cfg.initializer_std`` (matching other weight
matrices) so the gate has non-degenerate small random values at init;
silu(0) = 0 so zero-init would kill q entirely.

Single run at d512. Submit on us-east5-a:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --zone us-east5-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_may_arch_1pct_pko_q_silu_gate_sweep
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
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHMayArchGNMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_BUDGET: float = 2.19e17
_HIDDEN_DIM: int = 512

_WARMUP_FRACTION: float = 0.01
_NUM_EXPERTS: int = 256
_BASELINE_TARGET_STEPS: int = 2**14
_GROUP_NAME: str = "muonh-may-arch-1pct-pko-q-silu-gate-sweep"
_RUN_SUFFIX: str = "v1"


def _muonh_optimizer(base_optimizer: GrugMoeAdamHConfig) -> GrugMoeMuonHMayArchGNMuonHConfig:
    return GrugMoeMuonHMayArchGNMuonHConfig(
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


def _build_step() -> ExecutorStep:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_BASELINE_TARGET_STEPS,
    )
    model = dataclasses.replace(
        model,
        num_experts=_NUM_EXPERTS,
        partial_key_offset="every_4th",
        use_partial_rope=True,
        last_layer_pko=True,
        router_z_loss_coef=0.0,
        pko_q_silu_gate=True,
    )
    optimizer = _muonh_optimizer(base_optimizer)

    run_id = f"muonh-may-arch-1pct-pko-q-silu-gate-{_RUN_SUFFIX}-d{_HIDDEN_DIM}"
    step_name = f"grug/muonh_may_arch_1pct_pko_q_silu_gate_sweep/{run_id}"

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
                    "muonh_may_arch_1pct_pko_q_silu_gate_sweep",
                    f"d{_HIDDEN_DIM}",
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
    executor_main(
        steps=[_build_step()],
        description=(
            "MoE may_arch + 1pct-noclip + PKO SiLU-gated query " f"(d{_HIDDEN_DIM}, run_suffix={_RUN_SUFFIX!r})."
        ),
    )
