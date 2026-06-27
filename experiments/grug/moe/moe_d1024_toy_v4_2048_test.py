# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Toy d=1024 / 4-layer model on v4-2048 with pure FSDP (no replica axis).

Built to isolate whether the v4-2048 mesh shape itself causes any of the
trajectory issues we saw at d=2560. At ``d=1024`` and ``num_layers=4`` the
model fits cleanly with ``replica_axis_size=1`` because 1024 / 1024 = 1,
so pure FSDP across all 1024 chips just works without any helper axis.

Submit (us-central2, reserved)::

    WANDB_KEY=$(python3 -c "import os; print(os.environ['WANDB_API_KEY'])") && \\
    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production --no-preemptible -e WANDB_API_KEY "$WANDB_KEY" \\
        -- python -m experiments.grug.moe.moe_d1024_toy_v4_2048_test
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.launch_datakit_moe_mix import _datakit_data_config
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_DIM: int = 1024
_NUM_LAYERS: int = 4
_BS: int = 1024  # min BS for v4-2048 (batch_shards=1024); 1024*2048 = 2.1M tokens/step
_SEQ: int = 2048
_STEPS: int = 50
_EP: int = 1
# Replica-axis ablation: rep=1 already ran; rep=2 and rep=4 isolate the
# numerical effect of introducing the cross-replica psum on top of FSDP.
_REPLICA_AXES: tuple[int, ...] = (2, 4)
_SLICE: str = "v4-2048"
_LOGIT_Z_LOSS_WEIGHT: float = 1e-4

_heuristic = MoeMuonHHeuristic(min_lr_ratio=0.05)
_model_base = _heuristic.build_model_config(_DIM, seq_len=_SEQ)
# Cut the layer count down from the heuristic default to keep the model toy-sized.
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

steps: list[ExecutorStep] = []
for _replica_axis in _REPLICA_AXES:
    _run_id = f"moe_d{_DIM}_L{_NUM_LAYERS}_rep{_replica_axis}_bs{_BS}_seq{_SEQ}_v4_2048_toy"
    steps.append(
        ExecutorStep(
            name=f"grug/{_run_id}",
            fn=run_grug_moe_trial,
            config=GrugMoeLaunchConfig(
                model=versioned(_model),
                data=_datakit_data_config(
                    total_steps=_STEPS,
                    batch_size=_BS,
                    max_seq_len=_SEQ,
                    enable_simulated_epoching=False,
                ),
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
                        "june_tpu",
                        "toy",
                        "replica_ablation",
                        f"d{_DIM}",
                        f"L{_NUM_LAYERS}",
                        f"ep{_EP}",
                        f"rep{_replica_axis}",
                        "datakit_mix",
                        "stacked",
                        "v4_2048",
                    ],
                    group="june-tpu-toy-d1024-replica-ablation",
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
                        # v4-2048 = 1024 devices, EP=1 -> batch shards = replica * data = 1024.
                        # eval_batch_size must be divisible by 1024.
                        eval_batch_size=1024,
                        steps_per_eval=_STEPS,
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
            f"Toy d={_DIM} L={_NUM_LAYERS} on {_SLICE} replica ablation "
            f"(replica in {_REPLICA_AXES}, EP={_EP}, BS={_BS}, seq={_SEQ}, "
            f"{_STEPS} steps). Same model + data as the rep=1 run; only the "
            f"replica_axis_size changes, so any trajectory delta isolates the "
            f"replica-induced bf16 reduction-order bias."
        ),
    )
