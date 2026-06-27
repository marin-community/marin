# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""d=512 BS=1024 ``rmsnorm_to_adam=True`` runs on v4-2048 (replica=16).

Companion to the v5p-8 us-east5-a baselines in
``moe_d512_baseline_v5p_8.py`` — same model / data / optimizer settings,
just on v4-2048 reserved with ``replica_axis_size=16`` (the smallest
practical replica value at d=512 on 1024 chips that still leaves room for
reasonable MXU utilization). Both the 200-step and 1000-step variants run
with ``rmsnorm_to_adam=True`` (stacked rms scales route to plain Adam
instead of muonh).

Submit (us-central2, production, reserved)::

    WANDB_KEY=$(python3 -c "import os; print(os.environ['WANDB_API_KEY'])") && \\
    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production --no-preemptible -e WANDB_API_KEY "$WANDB_KEY" \\
        -- python -m experiments.grug.moe.moe_d512_rmsadam_v4_2048
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
_EP: int = 1
# v4-2048 = 1024 chips. data axis must divide hidden_dim=512, so the smallest
# replica is 2 (data=512, per-shard hidden=1, 128x MXU pad bloat). The submitted
# tuples below are evaluated as (replica_axis_size, step_count).
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

_VARIANTS: tuple[tuple[int, int], ...] = (
    # Submitted earlier (kept here for documentation; commented so a re-run
    # won't duplicate iris tasks):
    # (2, 200),   ← bf16 baseline + fp32 + fp32+fp32NS variants already submitted
    # (16, 200), (16, 1000)  ← from job 20260627-082159
    (1024, 200),  # extreme DP regime: data=1, no FSDP within a replica
)

steps: list[ExecutorStep] = []
for _replica_axis, _step_count in _VARIANTS:
    _tokens = float(_step_count * _BS * _SEQ)
    _optimizer = _heuristic.build_muonh_config(_BS, _tokens, _DIM, seq_len=_SEQ)
    _optimizer = dataclasses.replace(_optimizer, rmsnorm_to_adam=True)
    _run_id = f"moe_d{_DIM}_rmsadam_rep{_replica_axis}_bs{_BS}_seq{_SEQ}_v4_2048_{_step_count}steps"
    steps.append(
        ExecutorStep(
            name=f"grug/{_run_id}",
            fn=run_grug_moe_trial,
            config=GrugMoeLaunchConfig(
                model=versioned(_model),
                data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                output_path=this_output_path(),
                run_id=_run_id,
                resources=versioned(ResourceConfig.with_tpu(_SLICE, preemptible=False)),
                steps=versioned(_step_count),
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
                        f"rep{_replica_axis}",
                        f"bs{_BS}",
                        "nemotron_mix",
                        "disable_pko",
                        "no_long_rope",
                        "stacked",
                        "rmsadam",
                        "v4_2048",
                        f"{_step_count}steps",
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
                        replica_axis_size=_replica_axis,
                    )
                ),
                eval=versioned(
                    GrugEvalConfig(
                        # v4-2048 = 1024 devices, replica=16, EP=1 → data=64.
                        # batch shards = 16*64*1 = 1024 → eval_batch_size must
                        # be divisible by 1024.
                        eval_batch_size=1024,
                        steps_per_eval=1000,
                        max_eval_batches=1,
                        eval_current=True,
                        eval_ema=False,
                    )
                ),
            ),
        )
    )


if __name__ == "__main__":
    executor_main(
        steps=steps,
        description=(
            f"d={_DIM} rmsadam BS={_BS} seq={_SEQ} on {_SLICE} "
            f"(variants: {_VARIANTS}), Nemotron mix, stacked blocks, "
            f"rmsnorm_to_adam=True."
        ),
    )
