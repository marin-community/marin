# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""may_arch + 1pct-noclip + per-layer learned scalar gate on the shared expert.

Builds on the 1pct-noclip recipe. Each block learns a ``shared_gate``
weight of shape ``(hidden_dim, 1)``, zero-init, that produces a scalar
per token applied to the shared expert output:

| mode | gate range | gate at init | init behaviour |
|---|---|---|---|
| ``sigmoid``    | [0, 1] | 0.5 | halved shared expert |
| ``sigmoid_2x`` | [0, 2] | 1.0 | init-neutral vs baseline |

The ``sigmoid_2x`` mirrors the attention head gate's
``2 * sigmoid(...)`` pattern. The ``sigmoid`` variant lets the model
learn to fully suppress (gate=0) the shared expert per token.

Per-token gate value: ``sigmoid(mlp_in @ shared_gate)`` of shape
``(B, S, 1)``, broadcast against the shared expert's full output.

Gate weights go to the small-LR ``adam`` group via
``GrugMoeMuonHMayArchGNMuonHConfig.create_mask`` (matching ``attn_gate``
and ``router_bias``).

2 variants x 2 scales = **4 runs**.

Submit on us-east5-a:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --zone us-east5-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_may_arch_1pct_shared_gate_sweep
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

_POINTS: tuple[tuple[int, float], ...] = (
    (512, 2.19e17),
    (768, 1.70e18),
)

_GATE_MODES: tuple[str, ...] = ("sigmoid", "sigmoid_2x")

_WARMUP_FRACTION: float = 0.01
_NUM_EXPERTS: int = 256
_BASELINE_TARGET_STEPS: int = 2**14
_GROUP_NAME: str = "muonh-may-arch-1pct-shared-gate-sweep"


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _format_gate(mode: str) -> str:
    # "sigmoid" -> "sig", "sigmoid_2x" -> "sig2x"
    return {"sigmoid": "sig", "sigmoid_2x": "sig2x"}[mode]


def _format_run_id(hidden_dim: int, budget: float, gate_mode: str, run_suffix: str = "") -> str:
    budget_label = _format_budget(budget)
    normalized_suffix = run_suffix.strip()
    if normalized_suffix and not normalized_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_suffix may only contain letters, numbers, hyphens, and underscores")
    suffix = f"{normalized_suffix}-" if normalized_suffix else ""
    return f"muonh-may-arch-1pct-shared-gate-{_format_gate(gate_mode)}-{suffix}d{hidden_dim}-{budget_label}"


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


def _build_step(hidden_dim: int, budget: float, gate_mode: str, run_suffix: str = "") -> ExecutorStep:
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
        shared_gate_mode=gate_mode,
    )
    optimizer = _muonh_optimizer(base_optimizer)

    run_id = _format_run_id(hidden_dim, budget, gate_mode, run_suffix=run_suffix)
    step_name = f"grug/muonh_may_arch_1pct_shared_gate_sweep/{run_id}"

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
                    "muonh_may_arch_1pct_shared_gate_sweep",
                    f"d{hidden_dim}",
                    f"gate-{_format_gate(gate_mode)}",
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


_RUN_SUFFIX: str = "v1"


if __name__ == "__main__":
    steps = [
        _build_step(hidden_dim=d, budget=c, gate_mode=m, run_suffix=_RUN_SUFFIX) for d, c in _POINTS for m in _GATE_MODES
    ]
    executor_main(
        steps=steps,
        description=(
            "MoE may_arch + 1pct-noclip + per-layer learned scalar gate on the shared expert "
            f"(modes={list(_GATE_MODES)}, d512/d768, run_suffix={_RUN_SUFFIX!r})."
        ),
    )
