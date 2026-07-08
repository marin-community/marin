# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimalist smoke test on v4-2048 to isolate the SIGSEGV that keeps hitting
``experiments.grug.moe.moe_67b_a2b_d2560_resume15k_bs8192_rep8_muon_10T``.

Everything possible is stripped from the failing launcher while keeping the
suspected culprit (``rep=8`` mesh + v4-2048 slice) intact::

- ``num_layers`` = 6 (down from 26) → fewer stacked-block ops per step.
- ``BS`` = 1024 constant (no BS ramp, no piecewise ``IntSchedule``, no
  ``skip_batch_size_schedule_head_validation``, no custom LR subclass).
- Data → ``NEMOTRON_MIX_WITH_DEFAULT_VALIDATION`` (not the datakit mix, so no
  phase-1 mixture transition to trip over).
- Stock ``GrugMoeMuonHConfig`` optimizer via the standard heuristic — no
  ``GrugMoeMuonHResumeConfig`` subclass, no piecewise LR schedule, no
  ``lr_at_ramp_end``.
- No ``initialize_from_path`` → fresh model init.
- Stock ``run_grug_moe_trial`` from ``launch.py`` (not ``run_grug_moe_trial_2x_bs``).

Everything below is what the failing launcher has in common with this one::

- ``d=2560``, GQA 4:1, ``sliding_window=2048``, ``disable_pko``,
  ``disable_long_rope``, ``use_array_stacked_blocks``.
- Mesh ``(replica_dcn=8, data=128, expert=1, model=1)`` on v4-2048.
- ``EP=1``, ``mp=params=float32,compute=bfloat16,output=bfloat16``.

Interpretation:
- If this run reaches step 200 without a SIGSEGV: the failure is triggered
  by the BS ramp, the piecewise schedule, the initialize_from load, or the
  custom optimizer subclass — not the mesh or the model itself.
- If this run still SIGSEGVs at ~10 min: the culprit is deeper — the rep=8
  mesh, the v4-2048 slice, or the base MuonH stacked-block execution.

Submit (us-central2, production, --no-preemptible)::

    WANDB_KEY=$(python3 -c "import os; print(os.environ['WANDB_API_KEY'])") && \\
    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production --no-preemptible -e WANDB_API_KEY "$WANDB_KEY" \\
        -- python -m experiments.grug.moe.moe_smoke_d2560_L6_rep8_bs1024_v4_2048
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_DIM: int = 2560
_NUM_LAYERS: int = 6
_BS: int = 1024  # 1024 * 8192 = 8,388,608 tokens/step. Constant, no ramp.
_SEQ: int = 8192
_STEPS: int = 200
_EP: int = 1
_REPLICA_AXIS: int = 8
_SLICE: str = "v4-2048"
_LOGIT_Z_LOSS_WEIGHT: float = 1e-4

_heuristic = MoeMuonHHeuristic(min_lr_ratio=0.05)
_model_base = _heuristic.build_model_config(_DIM, seq_len=_SEQ)
_model = dataclasses.replace(
    _model_base,
    num_layers=_NUM_LAYERS,
    disable_pko=True,
    disable_long_rope=True,
    sliding_window=2048,
    use_array_stacked_blocks=True,
)
_tokens = float(_STEPS * _BS * _SEQ)
_optimizer = _heuristic.build_muonh_config(_BS, _tokens, _DIM, seq_len=_SEQ)

_run_id = (
    f"moe_smoke_d{_DIM}_L{_NUM_LAYERS}_ep{_EP}_rep{_REPLICA_AXIS}_"
    f"bs{_BS}_seq{_SEQ}_v4_2048"
)
step = ExecutorStep(
    name=f"grug/{_run_id}",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(_model),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=_run_id,
        resources=versioned(ResourceConfig.with_tpu(_SLICE, preemptible=False)),
        steps=versioned(_STEPS),
        batch_size=versioned(_BS),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin_moe",
            tags=[
                "moe",
                "smoke_test",
                f"d{_DIM}",
                f"L{_NUM_LAYERS}",
                f"ep{_EP}",
                f"rep{_REPLICA_AXIS}",
                f"bs{_BS}",
                "nemotron_mix",
                "disable_pko",
                "no_long_rope",
                "stacked",
                "v4_2048",
            ],
            group="moe-smoke-v4-2048",
            name=None,
        ),
        optimizer=versioned(_optimizer),
        expert_parallel=_EP,
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=_LOGIT_Z_LOSS_WEIGHT,
                ema_beta=None,
                log_every=1,
                replica_axis_size=_REPLICA_AXIS,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                # rep=8, data=128, expert=1 → batch_shards = 1024.
                # eval_batch_size must be divisible by 1024.
                eval_batch_size=1024,
                steps_per_eval=100,
                max_eval_batches=1,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[step],
        description=(
            f"Minimalist smoke test: d={_DIM} L={_NUM_LAYERS} BS={_BS} rep={_REPLICA_AXIS} "
            f"seq={_SEQ} on {_SLICE}, {_STEPS} steps, Nemotron mix, fresh init, "
            f"stock MuonH heuristic. Isolates whether the resume launcher's SIGSEGV is "
            f"config-driven or mesh/model-driven."
        ),
    )
