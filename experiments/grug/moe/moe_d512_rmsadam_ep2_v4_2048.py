# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""d=512 rmsadam 200-step run on v4-2048 with EP=2 (no replica).

Companion to ``moe_d512_rmsadam_v4_2048.py`` (replica=2). Same model /
data / hardware, but the divisibility issue at d=512 / 1024-chip mesh is
solved with expert parallelism instead of data-parallel replication:
mesh (1, 512, 2, 1) → data=512 ✓ and the 256 routed experts split 2-way
across the expert axis with ring-collective dispatch.

Useful for isolating "is the v4-1024 vs v4-2048 trajectory delta driven
by the cross-replica psum specifically, or by anything that introduces a
non-pure-FSDP collective"; if rep=2 and EP=2 produce the same bias vs
rep=1 (which we can't run at d=512), the answer is "any extra collective
on top of FSDP changes the bf16 reduction order".

Submit (us-central2, production, reserved)::

    WANDB_KEY=$(python3 -c "import os; print(os.environ['WANDB_API_KEY'])") && \\
    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production --no-preemptible -e WANDB_API_KEY "$WANDB_KEY" \\
        -- python -m experiments.grug.moe.moe_d512_rmsadam_ep2_v4_2048
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.grug.moe.launch import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION, GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_DIM: int = 512
_BS: int = 1024  # 1024 * 8192 = 8,388,608 tokens/step
_SEQ: int = 8192
_STEP_COUNT: int = 200
_EP: int = 2  # 256 experts split 2-way across the expert axis
_REPLICA_AXIS: int = 1  # no replica axis — divisibility handled by EP
_SLICE: str = "v4-2048"
_LOGIT_Z_LOSS_WEIGHT: float = 1e-4

_heuristic = MoeMuonHHeuristic(min_lr_ratio=0.05)
_model_base = _heuristic.build_model_config(_DIM, seq_len=_SEQ)
_model = dataclasses.replace(
    _model_base,
    disable_pko=True,
    disable_long_rope=True,
    sliding_window=2048,
    use_array_stacked_blocks=True,
)
_tokens = float(_STEP_COUNT * _BS * _SEQ)
_optimizer = _heuristic.build_muonh_config(_BS, _tokens, _DIM, seq_len=_SEQ)
_optimizer = dataclasses.replace(_optimizer, rmsnorm_to_adam=True)

_run_id = f"moe_d{_DIM}_rmsadam_ep{_EP}_rep{_REPLICA_AXIS}_bs{_BS}_seq{_SEQ}_v4_2048_{_STEP_COUNT}steps"
step = ExecutorStep(
    name=f"grug/{_run_id}",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(_model),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=_run_id,
        resources=versioned(ResourceConfig.with_tpu(_SLICE, preemptible=False)),
        steps=versioned(_STEP_COUNT),
        batch_size=versioned(_BS),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin_moe",
            tags=[
                "moe",
                "d512_baseline",
                f"d{_DIM}",
                f"ep{_EP}",
                f"rep{_REPLICA_AXIS}",
                f"bs{_BS}",
                "nemotron_mix",
                "disable_pko",
                "no_long_rope",
                "stacked",
                "rmsadam",
                "v4_2048",
                f"{_STEP_COUNT}steps",
            ],
            group="moe-d512-rmsadam-v4-2048",
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
                # v4-2048 = 1024 devices, replica=1, EP=2 → data=512.
                # Batch shards = 1*512*2 = 1024 → eval_batch_size must be
                # divisible by 1024.
                eval_batch_size=1024,
                steps_per_eval=1000,
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
            f"d={_DIM} rmsadam BS={_BS} seq={_SEQ} EP={_EP} replica={_REPLICA_AXIS} on {_SLICE} "
            f"({_STEP_COUNT} steps). Nemotron mix, stacked blocks."
        ),
    )
