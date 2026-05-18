# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""50-step profiled TPS test: baseline vs aurorah-gn vs split-only.

To diagnose the +7% MFU gap between the "slow cluster" (~332K TPS d512,
all baseline-style 1pct-noclip runs) and the "fast cluster" (~352-357K
TPS d512, runs using AuroraTargetConfig and/or split MoEMLP storage):

- CPU HLO dumps showed the optimizer step compute is bit-identical
  between GNMuonHConfig and AuroraTargetConfig(no flags).
- model.py / launch.py / heuristic.py are byte-identical between the
  baseline branch (c86b36008) and the AuroraTargetConfig branch
  (7666f8a19).
- Adding an unused multi_transform branch to GNMuonHConfig didn't
  reproduce the speedup.

This sweep enables the JAX profiler so we can pull perfetto traces from
wandb and compare kernel-level timings on TPU.

Three trials at d512, 50 steps each, profiler starts at step 5 for 25
steps:

| trial label | optimizer config                                          | MoEMLP storage |
|---|---|---|
| ``baseline``     | ``GNMuonHConfig``                                  | concatenated (no split) |
| ``aurorah-gn``   | ``AuroraTargetConfig(gn_aurora=True)``             | concatenated (no split) |
| ``split-only``   | ``GNMuonHConfig`` (same as baseline)               | split ``w_gate`` / ``w_up`` |

If trace shows the same kernels with different per-kernel latency =
host/pool effect. If different kernels or different fusion = compile
effect.

Submit on us-east5-a:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --zone us-east5-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_may_arch_1pct_mfu_profile_test
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.optimizer import (
    GrugMoeAdamHConfig,
    GrugMoeMuonHMayArchAuroraTargetConfig,
    GrugMoeMuonHMayArchGNMuonHConfig,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_BUDGET: float = 2.19e17
_HIDDEN_DIM: int = 512
_NUM_STEPS_OVERRIDE: int = 50

_WARMUP_FRACTION: float = 0.01
_NUM_EXPERTS: int = 256
_BASELINE_TARGET_STEPS: int = 2**14
_GROUP_NAME: str = "muonh-may-arch-1pct-mfu-profile-test"
_RUN_SUFFIX: str = "v1"


def _gn_muonh_optimizer(base_optimizer: GrugMoeAdamHConfig) -> GrugMoeMuonHMayArchGNMuonHConfig:
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


def _aurora_gn_optimizer(base_optimizer: GrugMoeAdamHConfig) -> GrugMoeMuonHMayArchAuroraTargetConfig:
    return GrugMoeMuonHMayArchAuroraTargetConfig(
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
        gn_aurora=True,
    )


def _build_step(trial_label: str, split: bool, use_aurora: bool) -> ExecutorStep:
    model, base_optimizer, batch_size, _num_steps = build_from_heuristic(
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
        split_w_gate_up=split,
    )
    optimizer = _aurora_gn_optimizer(base_optimizer) if use_aurora else _gn_muonh_optimizer(base_optimizer)

    run_id = f"muonh-may-arch-1pct-mfu-profile-{trial_label}-{_RUN_SUFFIX}-d{_HIDDEN_DIM}"

    return ExecutorStep(
        name=f"grug/muonh_may_arch_1pct_mfu_profile_test/{run_id}",
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v5p-8")),
            steps=versioned(_NUM_STEPS_OVERRIDE),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="marin_moe",
                tags=["moe", "muonh_may_arch_1pct_mfu_profile_test", f"d{_HIDDEN_DIM}", trial_label, "50step"],
                group=_GROUP_NAME,
                name=None,
            ),
            optimizer=versioned(optimizer),
            profiler=ProfilerConfig(enabled=True, start_step=5, num_steps=25, perfetto_link=False),
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
                    steps_per_eval=10000,  # don't eval during this 50-step probe
                    max_eval_batches=8,
                    eval_current=False,
                    eval_ema=False,
                )
            ),
            enable_cross_region_ckpt_read=False,
        ),
    )


_TRIALS: tuple[tuple[str, bool, bool], ...] = (
    # (label, split_w_gate_up, use_aurora)
    ("baseline", False, False),
    ("aurorah-gn", False, True),
    ("split-only", True, False),
)


if __name__ == "__main__":
    steps = [_build_step(label, split, use_aurora) for label, split, use_aurora in _TRIALS]
    executor_main(
        steps=steps,
        description=(
            "50-step profiled MFU test: baseline (GNMuonH, no split) vs aurorah-gn (AuroraTarget(gn=True), "
            "no split) vs split-only (GNMuonH, split). Profiler captures steps 5-30."
        ),
    )
